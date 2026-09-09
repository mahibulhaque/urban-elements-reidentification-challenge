"""Cross-encoder trained on (train + 70% of val) pairs.

Why: the previous train-only cross-encoder overfit to within-train-camera
patterns (miniVal=1.0 = perfect recall on held-out train pids) and lost on
val (Δ −0.038). Val provides strong cross-camera positive supervision: each
val pid has c104 query images AND c101-c103 gallery images, exactly the
cross-camera structure we need the model to learn.

Splits:
  - 70% of val pids → folded into the training pool
  - 30% of val pids → held out as cross-camera mini-val
    (queries = held-out val_q (c104),  gallery = held-out val_g (c101-103))
    Used for picking best ckpt AND as the honest val-side comparison.
  - Test side: full Hplus_test cache, scored with the best ckpt.

Hyperparams unchanged from the train-only run; only the training pool changes.
"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


class CrossEncoder(nn.Module):
    def __init__(self, feat_dim=1280, hidden=1024, dropout=0.2):
        super().__init__()
        in_dim = 4 * feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1))
    def forward(self, q, g):
        return self.mlp(torch.cat([q, g, (q-g).abs(), q*g], dim=-1)).squeeze(-1)


def l2(x, eps=1e-12):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=eps)


def build_pairs(feats, pids, top_k=50):
    feats_n = l2(feats)
    sims = feats_n @ feats_n.T
    np.fill_diagonal(sims, -np.inf)
    N = len(feats)
    anchors, cands, labels = [], [], []
    for i in range(N):
        topk = np.argpartition(-sims[i], top_k)[:top_k]
        topk = topk[np.argsort(-sims[i][topk])]
        pos_idx = np.where((pids == pids[i]) & (np.arange(N) != i))[0]
        cand = np.unique(np.concatenate([topk, pos_idx]))
        anchors.extend([i] * len(cand))
        cands.extend(cand.tolist())
        labels.extend((pids[cand] == pids[i]).astype(np.int8).tolist())
    return (np.asarray(anchors, dtype=np.int64),
            np.asarray(cands,   dtype=np.int64),
            np.asarray(labels,  dtype=np.float32))


def compute_ap(order, q_pid, g_pids, q_cam, g_cams):
    keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
    order = order[keep]
    matches = (g_pids[order] == q_pid).astype(np.float32)
    n_pos = matches.sum()
    if n_pos == 0: return None
    cum = np.cumsum(matches)
    prec = cum / (np.arange(len(matches)) + 1)
    return float((prec * matches).sum() / n_pos)


@torch.no_grad()
def cross_score(model, qf_t, gf_t, q_idx, g_idx, batch=4096):
    model.eval()
    out = np.zeros(len(q_idx), dtype=np.float32)
    for s in range(0, len(q_idx), batch):
        e = s + batch
        q = qf_t[torch.from_numpy(q_idx[s:e])]
        g = gf_t[torch.from_numpy(g_idx[s:e])]
        out[s:e] = model(q, g).cpu().numpy()
    return out


@torch.no_grad()
def cross_camera_map(model, q_feats, q_pids, q_cams, g_feats, g_pids, g_cams):
    """Mean AP over q×g cross-encoder scores. Same-cam-same-id removed."""
    model.eval()
    qf_t = torch.from_numpy(q_feats).float().cuda()
    gf_t = torch.from_numpy(g_feats).float().cuda()
    aps = []
    for i in range(len(q_pids)):
        q = qf_t[i:i+1].expand(len(g_feats), -1)
        scores = []
        # chunk to avoid OOM
        for s in range(0, len(g_feats), 2048):
            chunk = gf_t[s:s+2048]
            qrep = q[s:s+2048]
            sc = model(qrep, chunk).cpu().numpy()
            scores.append(sc)
        scores = np.concatenate(scores)
        order = np.argsort(-scores)
        ap = compute_ap(order, q_pids[i], g_pids, q_cams[i], g_cams)
        if ap is not None: aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def baseline_cross_camera_map(q_feats, q_pids, q_cams, g_feats, g_pids, g_cams):
    """Pure cosine baseline on the same held-out slice for honest comparison."""
    qn = l2(q_feats); gn = l2(g_feats)
    sims = qn @ gn.T
    aps = []
    for i in range(len(q_pids)):
        order = np.argsort(-sims[i])
        ap = compute_ap(order, q_pids[i], g_pids, q_cams[i], g_cams)
        if ap is not None: aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_cache', default='feat_cache/HpMS_train')
    ap.add_argument('--val_cache',   default='feat_cache/Hplus_val')
    ap.add_argument('--out_dir',     required=True)
    ap.add_argument('--val_holdout_frac', type=float, default=0.30)
    ap.add_argument('--top_k',  type=int, default=50)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch',  type=int, default=2048)
    ap.add_argument('--lr',     type=float, default=3e-4)
    ap.add_argument('--wd',     type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=1024)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--seed',   type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- load train cache ----
    tr_feats  = np.load(os.path.join(args.train_cache, 'train_feats.npy')).astype(np.float32)
    tr_pids   = np.load(os.path.join(args.train_cache, 'train_pids.npy')).astype(np.int64)
    tr_camids = np.load(os.path.join(args.train_cache, 'train_camids.npy')).astype(np.int64)

    # ---- load val cache ----
    vq_feat = np.load(os.path.join(args.val_cache, 'qf.npy')).astype(np.float32)
    vg_feat = np.load(os.path.join(args.val_cache, 'gf.npy')).astype(np.float32)
    with open(os.path.join(args.val_cache, 'meta.json')) as f: vm = json.load(f)
    vq_pids_orig = np.asarray(vm['query_pids'],   dtype=np.int64)
    vg_pids_orig = np.asarray(vm['gallery_pids'], dtype=np.int64)
    vq_cam = np.asarray(vm['query_camids'], dtype=np.int64)
    vg_cam = np.asarray(vm['gallery_camids'], dtype=np.int64)

    # Relabel val pids to a contiguous range that doesn't collide with train
    val_unique = sorted(set(vq_pids_orig.tolist() + vg_pids_orig.tolist()))
    pid_offset = int(tr_pids.max()) + 1
    val_pid_map = {p: pid_offset + i for i, p in enumerate(val_unique)}
    vq_pids = np.array([val_pid_map[p] for p in vq_pids_orig], dtype=np.int64)
    vg_pids = np.array([val_pid_map[p] for p in vg_pids_orig], dtype=np.int64)
    print(f'[load] train: {len(tr_feats):,} imgs / {len(np.unique(tr_pids)):,} pids')
    print(f'[load] val:   {len(vq_feat)} q + {len(vg_feat)} g / {len(val_unique)} pids '
          f'(relabeled to {pid_offset}..{pid_offset+len(val_unique)-1})')

    # ---- 70/30 split of val pids ----
    val_unique_relab = sorted(val_pid_map.values())
    rng.shuffle(val_unique_relab)
    n_holdout = max(1, int(args.val_holdout_frac * len(val_unique_relab)))
    holdout_pids = set(val_unique_relab[:n_holdout])
    trainval_pids = set(val_unique_relab[n_holdout:])
    print(f'[split] val pids: {len(trainval_pids)} folded into training, '
          f'{len(holdout_pids)} held out for cross-camera mini-val')

    # ---- training pool: train + (val-q + val-g) where pid in trainval_pids ----
    vq_train_mask = np.array([p in trainval_pids for p in vq_pids])
    vg_train_mask = np.array([p in trainval_pids for p in vg_pids])
    pool_feats  = np.concatenate([tr_feats,  vq_feat[vq_train_mask],  vg_feat[vg_train_mask]],  axis=0)
    pool_pids   = np.concatenate([tr_pids,   vq_pids[vq_train_mask],  vg_pids[vg_train_mask]])
    pool_camids = np.concatenate([tr_camids, vq_cam[vq_train_mask],   vg_cam[vg_train_mask]])
    print(f'[pool] {len(pool_feats):,} imgs / {len(np.unique(pool_pids)):,} pids')

    # Held-out cross-camera mini-val: held-out val-q vs held-out val-g
    vq_hold_mask = np.array([p in holdout_pids for p in vq_pids])
    vg_hold_mask = np.array([p in holdout_pids for p in vg_pids])
    hq_feat  = vq_feat[vq_hold_mask];  hq_pid = vq_pids[vq_hold_mask];  hq_cam = vq_cam[vq_hold_mask]
    hg_feat  = vg_feat[vg_hold_mask];  hg_pid = vg_pids[vg_hold_mask];  hg_cam = vg_cam[vg_hold_mask]
    print(f'[mini-val] held-out: {len(hq_feat)} queries (c104) / '
          f'{len(hg_feat)} gallery (c101-103) / {len(set(hq_pid.tolist()))} pids')

    # Baseline pure-cosine cross-camera mAP on the held-out slice
    base_map = baseline_cross_camera_map(hq_feat, hq_pid, hq_cam,
                                         hg_feat, hg_pid, hg_cam)
    print(f'[mini-val baseline] pure cosine cross-camera mAP = {base_map:.4f}')

    # ---- build training pairs from pool ----
    # L2-norm pool features (training input space)
    pool_n = l2(pool_feats)
    print('[pairs] mining top-K NN + all positives...')
    t0 = time.time()
    a_idx, c_idx, lab = build_pairs(pool_feats, pool_pids, top_k=args.top_k)
    print(f'[pairs] {len(lab):,} pairs  pos_rate={lab.mean():.3f}  '
          f'time={time.time()-t0:.1f}s')

    # ---- model ----
    feat_dim = pool_feats.shape[1]
    model = CrossEncoder(feat_dim=feat_dim, hidden=args.hidden,
                         dropout=args.dropout).cuda()
    print(f'[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    pos_weight = torch.tensor([(1 - lab.mean()) / max(lab.mean(), 1e-6)],
                              dtype=torch.float32).cuda()
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    pool_t = torch.from_numpy(pool_n).float().cuda()
    a_idx_t = torch.from_numpy(a_idx); c_idx_t = torch.from_numpy(c_idx); lab_t = torch.from_numpy(lab)

    best_score = -1.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(lab_t))
        ep_loss = 0.0; n_steps = 0
        t0 = time.time()
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s+args.batch]
            q = pool_t[a_idx_t[idx]]
            g = pool_t[c_idx_t[idx]]
            y = lab_t[idx].cuda()
            logits = model(q, g)
            loss = bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); n_steps += 1
        sched.step()
        # mini-val every 2 epochs (it's the only thing that matters)
        score = cross_camera_map(model, l2(hq_feat), hq_pid, hq_cam,
                                 l2(hg_feat), hg_pid, hg_cam) \
                if (ep + 1) % 2 == 0 or ep == args.epochs - 1 else None
        msg = f'[ep {ep+1:2d}/{args.epochs}] loss={ep_loss/n_steps:.4f}  '\
              f'lr={opt.param_groups[0]["lr"]:.2e}  time={time.time()-t0:.1f}s'
        if score is not None:
            msg += f'  miniVal-cross-cam mAP={score:.4f} (base={base_map:.4f})'
            if score > best_score:
                best_score = score
                torch.save({'model': model.state_dict(),
                            'feat_dim': feat_dim, 'hidden': args.hidden,
                            'dropout': args.dropout, 'val_score': score,
                            'baseline_score': base_map},
                           os.path.join(args.out_dir, 'cross_best.pth'))
                msg += '  *new best*'
        print(msg, flush=True)

    torch.save({'model': model.state_dict(), 'feat_dim': feat_dim,
                'hidden': args.hidden, 'dropout': args.dropout,
                'val_score': best_score, 'baseline_score': base_map},
               os.path.join(args.out_dir, 'cross_final.pth'))
    with open(os.path.join(args.out_dir, 'train_meta.json'), 'w') as f:
        json.dump({'top_k': args.top_k, 'epochs': args.epochs, 'batch': args.batch,
                   'lr': args.lr, 'hidden': args.hidden, 'dropout': args.dropout,
                   'pos_rate': float(lab.mean()), 'n_pairs': int(len(lab)),
                   'val_holdout_frac': args.val_holdout_frac,
                   'best_miniVal_crosscam_mAP': best_score,
                   'baseline_crosscam_mAP': base_map,
                   'delta': best_score - base_map}, f, indent=2)
    print(f'\n[done] best_crosscam_mAP={best_score:.4f}  '
          f'baseline={base_map:.4f}  Δ={best_score - base_map:+.4f}')


if __name__ == '__main__':
    main()
