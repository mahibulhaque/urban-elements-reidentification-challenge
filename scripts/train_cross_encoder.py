"""Cross-encoder re-ranker on top of frozen Hplus features.

Trains a small MLP that takes a (query_feat, gallery_feat) pair and outputs a
match-probability logit. Trained on pairs sampled from the train split:
  - For each train image used as query, take its top-K nearest-by-cosine
    candidates from the rest of train; label same-pid=1, else 0.
  - Always include the same-pid positives (even if outside top-K) to
    guarantee supervision.

Hold out 10% of pids for in-training validation — used only for picking the
best checkpoint, not for grid search on val.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


class CrossEncoder(nn.Module):
    """MLP over [q, g, |q-g|, q*g] -> match logit."""
    def __init__(self, feat_dim=1280, hidden=1024, dropout=0.2):
        super().__init__()
        in_dim = 4 * feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, q, g):
        x = torch.cat([q, g, (q - g).abs(), q * g], dim=-1)
        return self.mlp(x).squeeze(-1)


def l2(x, eps=1e-12):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=eps)


def build_pairs(feats, pids, top_k=50, exclude_self=True):
    """For each image i (used as anchor), take its top-K cosine NN from the
    rest of the dataset, plus all same-pid samples. Returns lists:
        anchor_idx : (N_pairs,)
        cand_idx   : (N_pairs,)
        label      : (N_pairs,) {0,1}
    """
    feats_n = l2(feats)
    sims = feats_n @ feats_n.T  # (N, N)
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)
    N = len(feats)
    anchors, cands, labels = [], [], []

    for i in range(N):
        # Top-K candidates by similarity
        topk = np.argpartition(-sims[i], top_k)[:top_k]
        topk = topk[np.argsort(-sims[i][topk])]
        # Same-pid positives (always include)
        pos_idx = np.where((pids == pids[i]) & (np.arange(N) != i))[0]
        # Union
        cand = np.unique(np.concatenate([topk, pos_idx]))
        anchors.extend([i] * len(cand))
        cands.extend(cand.tolist())
        labels.extend((pids[cand] == pids[i]).astype(np.int8).tolist())

    return (np.asarray(anchors, dtype=np.int64),
            np.asarray(cands,   dtype=np.int64),
            np.asarray(labels,  dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', default='feat_cache/HpMS_train')
    ap.add_argument('--out_dir',   required=True)
    ap.add_argument('--top_k',    type=int, default=50)
    ap.add_argument('--epochs',   type=int, default=15)
    ap.add_argument('--batch',    type=int, default=2048)
    ap.add_argument('--lr',       type=float, default=3e-4)
    ap.add_argument('--wd',       type=float, default=1e-4)
    ap.add_argument('--hidden',   type=int, default=1024)
    ap.add_argument('--dropout',  type=float, default=0.2)
    ap.add_argument('--seed',     type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load train cache
    feats = np.load(os.path.join(args.cache_dir, 'train_feats.npy')).astype(np.float32)
    pids  = np.load(os.path.join(args.cache_dir, 'train_pids.npy')).astype(np.int64)
    print(f'[load] feats={feats.shape}  pids={len(np.unique(pids))}')

    feats_n = l2(feats)

    # Hold out 10% of pids for in-training val (picks the epoch checkpoint).
    unique_pids = np.unique(pids)
    rng.shuffle(unique_pids)
    n_val_pids = max(1, int(0.1 * len(unique_pids)))
    val_pids = set(unique_pids[:n_val_pids].tolist())
    train_mask = np.array([p not in val_pids for p in pids])
    val_mask   = ~train_mask
    train_idx_global = np.where(train_mask)[0]
    val_idx_global   = np.where(val_mask)[0]
    print(f'[split] train_imgs={len(train_idx_global)}  '
          f'val_imgs={len(val_idx_global)}  '
          f'val_pids={len(val_pids)}')

    # Build pairs only on train side
    print('[pairs] building training pairs (top-K NN + all positives)...')
    t0 = time.time()
    sub_feats = feats_n[train_idx_global]
    sub_pids  = pids[train_idx_global]
    a_idx, c_idx, lab = build_pairs(sub_feats, sub_pids, top_k=args.top_k)
    # Convert sub-indices back to global indices for clarity
    a_global = train_idx_global[a_idx]
    c_global = train_idx_global[c_idx]
    print(f'[pairs] built {len(lab):,} pairs  pos_rate={lab.mean():.3f}  '
          f'time={time.time()-t0:.1f}s')

    # ---- model ----
    feat_dim = feats.shape[1]
    model = CrossEncoder(feat_dim=feat_dim, hidden=args.hidden,
                         dropout=args.dropout).cuda()
    print(f'[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    pos_weight = torch.tensor([(1 - lab.mean()) / max(lab.mean(), 1e-6)],
                              dtype=torch.float32).cuda()
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    feats_gpu = torch.from_numpy(feats_n).cuda()  # (N, D)

    # Mini-val: build val-side pairs once for monitoring AP@all on each
    # held-out pid (using its own pid's gallery as the answer key).
    def _miniscore():
        model.eval()
        if len(val_idx_global) < 2: return 0.0
        with torch.no_grad():
            v_feats = feats_gpu[val_idx_global]
            v_pids  = pids[val_idx_global]
            # Score every val image vs every other val image
            B = len(val_idx_global)
            n_chunk = 512
            aps = []
            for i in range(B):
                q = v_feats[i:i+1]  # (1, D)
                gs = torch.cat([v_feats[:i], v_feats[i+1:]], dim=0)
                gp = np.concatenate([v_pids[:i], v_pids[i+1:]])
                scores = []
                for j in range(0, len(gs), n_chunk):
                    chunk = gs[j:j+n_chunk]
                    qrep = q.expand(len(chunk), -1)
                    s = model(qrep, chunk).cpu().numpy()
                    scores.append(s)
                scores = np.concatenate(scores)
                order = np.argsort(-scores)
                matches = (gp[order] == v_pids[i]).astype(np.float32)
                if matches.sum() == 0: continue
                cum = np.cumsum(matches)
                prec = cum / (np.arange(len(matches)) + 1)
                aps.append(float((prec * matches).sum() / matches.sum()))
            return float(np.mean(aps)) if aps else 0.0

    best_score = -1.0
    a_idx_t = torch.from_numpy(a_global)
    c_idx_t = torch.from_numpy(c_global)
    lab_t   = torch.from_numpy(lab)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(lab_t))
        ep_loss = 0.0; n_steps = 0
        t0 = time.time()
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s+args.batch]
            q = feats_gpu[a_idx_t[idx]]
            g = feats_gpu[c_idx_t[idx]]
            y = lab_t[idx].cuda()
            logits = model(q, g)
            loss = bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); n_steps += 1
        sched.step()
        score = _miniscore() if (ep + 1) % 3 == 0 or ep == args.epochs - 1 else None
        msg = f'[ep {ep+1:2d}/{args.epochs}] loss={ep_loss/n_steps:.4f}  '\
              f'lr={opt.param_groups[0]["lr"]:.2e}  time={time.time()-t0:.1f}s'
        if score is not None:
            msg += f'  miniVal mAP={score:.4f}'
            if score > best_score:
                best_score = score
                torch.save({'model': model.state_dict(),
                            'feat_dim': feat_dim, 'hidden': args.hidden,
                            'dropout': args.dropout, 'val_score': score},
                           os.path.join(args.out_dir, 'cross_best.pth'))
                msg += '  *new best*'
        print(msg, flush=True)

    # Always save final too
    torch.save({'model': model.state_dict(),
                'feat_dim': feat_dim, 'hidden': args.hidden,
                'dropout': args.dropout, 'val_score': best_score},
               os.path.join(args.out_dir, 'cross_final.pth'))
    with open(os.path.join(args.out_dir, 'train_meta.json'), 'w') as f:
        json.dump({'top_k': args.top_k, 'epochs': args.epochs,
                   'batch': args.batch, 'lr': args.lr, 'hidden': args.hidden,
                   'dropout': args.dropout, 'pos_rate': float(lab.mean()),
                   'n_pairs': int(len(lab)),
                   'best_miniVal_mAP': best_score}, f, indent=2)
    print(f'[done] best_miniVal_mAP={best_score:.4f}  -> {args.out_dir}')


if __name__ == '__main__':
    main()
