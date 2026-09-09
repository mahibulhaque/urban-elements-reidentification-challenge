"""Offline contrastive head on frozen Hplus features.

Pipeline:
  1. Load cached train features (D=1280) + pid + camid.
  2. Define a 2-layer MLP projection head:  D -> hidden -> D
     with BN + GELU + Dropout, output L2-normalized.
  3. Sample batches via PK sampler (P identities, K instances each)
     -> batch_size = P*K.  P=512, K=8 -> 4096.
  4. Train with MultiSimilarityLoss + MultiSimilarityMiner from
     pytorch-metric-learning.
  5. Save head weights to <out>/proj_head.pth.

Eval is in eval_offline_msloss.py — applies the head to val/test caches,
grids alpha-blend with raw, picks the best, writes a per-class submission.
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


class ProjHead(nn.Module):
    """1280 -> hidden -> 1280, residual + L2-norm."""
    def __init__(self, in_dim=1280, hidden=2048, out_dim=None, dropout=0.1):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn  = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.in_dim, self.out_dim = in_dim, out_dim

    def forward(self, x):
        z = self.fc2(self.drop(self.act(self.bn(self.fc1(x)))))
        return F.normalize(z, dim=-1)


class PKSampler(torch.utils.data.Sampler):
    """Random P identities per epoch step, K instances each."""
    def __init__(self, pids, P=512, K=8, num_steps=None):
        self.pids = np.asarray(pids, dtype=np.int64)
        self.P = P; self.K = K
        self.unique = np.unique(self.pids)
        # Index list per pid
        self.idx_by_pid = {p: np.where(self.pids == p)[0] for p in self.unique}
        if num_steps is None:
            num_steps = max(1, len(self.pids) // (P * K))
        self.num_steps = int(num_steps)

    def __iter__(self):
        rng = np.random.default_rng()
        out = []
        for _ in range(self.num_steps):
            chosen = rng.choice(self.unique, size=min(self.P, len(self.unique)),
                                replace=False)
            for p in chosen:
                idxs = self.idx_by_pid[p]
                if len(idxs) >= self.K:
                    pick = rng.choice(idxs, size=self.K, replace=False)
                else:
                    pick = rng.choice(idxs, size=self.K, replace=True)
                out.extend(pick.tolist())
        return iter(out)

    def __len__(self):
        return self.num_steps * self.P * self.K


class CachedFeats(torch.utils.data.Dataset):
    def __init__(self, feats, pids):
        self.feats = torch.from_numpy(feats).float()
        self.pids  = torch.from_numpy(pids).long()
    def __len__(self): return len(self.feats)
    def __getitem__(self, i): return self.feats[i], self.pids[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', required=True,
                    help='Dir from extract_train_features.py')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--P', type=int, default=512)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=2048)
    ap.add_argument('--out_dim', type=int, default=0,
                    help='Output dim; 0 = same as input (enables residual blend)')
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--steps_per_epoch', type=int, default=0,
                    help='0 = auto (~ N / batch)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load cached train features ----
    feats = np.load(os.path.join(args.cache_dir, 'train_feats.npy')).astype(np.float32)
    pids  = np.load(os.path.join(args.cache_dir, 'train_pids.npy')).astype(np.int64)
    print(f'[load] feats={feats.shape}  unique_pids={len(np.unique(pids))}')

    # PK sampler needs pids that have at least 2 instances (for positives).
    # Filter out singleton pids — they can't supply a positive pair.
    cnt = np.bincount(pids - pids.min())
    valid_pids = set((np.where(cnt >= 2)[0] + pids.min()).tolist())
    keep = np.array([i for i, p in enumerate(pids) if p in valid_pids], dtype=np.int64)
    print(f'[filter] kept {len(keep):,}/{len(pids):,} imgs '
          f'across {len(valid_pids)}/{len(np.unique(pids))} pids '
          f'(singletons removed)')
    feats, pids = feats[keep], pids[keep]

    # L2-normalize the input features (matches eval-time pipeline)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-12)

    ds = CachedFeats(feats, pids)
    sampler = PKSampler(pids, P=args.P, K=args.K,
                        num_steps=args.steps_per_epoch or None)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.P * args.K, sampler=sampler,
        num_workers=2, pin_memory=True, drop_last=True)
    print(f'[sampler] P={args.P}  K={args.K}  '
          f'batch={args.P*args.K}  steps/epoch={sampler.num_steps}')

    # ---- model + loss ----
    in_dim = feats.shape[1]
    out_dim = args.out_dim if args.out_dim > 0 else in_dim
    head = ProjHead(in_dim=in_dim, hidden=args.hidden, out_dim=out_dim,
                    dropout=args.dropout).cuda()
    print(f'[head] {in_dim} -> {args.hidden} -> {out_dim}  '
          f'params={sum(p.numel() for p in head.parameters())/1e6:.2f}M')

    from pytorch_metric_learning import losses, miners
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    loss_fn = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---- train loop ----
    head.train()
    for ep in range(args.epochs):
        t0 = time.time()
        ep_loss = 0.0; ep_pairs = 0; n_steps = 0
        for x, y in loader:
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            z = head(x)
            pairs = miner(z, y)
            loss = loss_fn(z, y, pairs)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); n_steps += 1
            ep_pairs += int(sum(p.numel() for p in pairs) / 4)
        sched.step()
        print(f'[ep {ep+1:3d}/{args.epochs}] loss={ep_loss/max(n_steps,1):.4f}  '
              f'pairs/step={ep_pairs/max(n_steps,1):.0f}  '
              f'lr={opt.param_groups[0]["lr"]:.2e}  '
              f'time={time.time()-t0:.1f}s', flush=True)

    torch.save({'head': head.state_dict(),
                'in_dim': in_dim, 'hidden': args.hidden, 'out_dim': out_dim,
                'dropout': args.dropout},
               os.path.join(args.out_dir, 'proj_head.pth'))
    with open(os.path.join(args.out_dir, 'train_meta.json'), 'w') as f:
        json.dump({'P': args.P, 'K': args.K, 'epochs': args.epochs,
                   'lr': args.lr, 'wd': args.wd, 'hidden': args.hidden,
                   'out_dim': out_dim, 'dropout': args.dropout,
                   'n_train': int(len(feats)),
                   'n_pids': int(len(np.unique(pids)))}, f, indent=2)
    print(f'[done] -> {args.out_dir}/proj_head.pth')


if __name__ == '__main__':
    main()
