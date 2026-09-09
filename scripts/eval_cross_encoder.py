"""Apply trained cross-encoder to re-rank top-K candidates from a per-class
k-reciprocal pipeline. Pipeline:

  1. Compute per-class k-reciprocal distance using De90's tuned params.
  2. For each query, take top-K candidates by k-reciprocal distance.
  3. Score each (q, candidate) pair with the cross-encoder; re-order top-K
     by descending score.
  4. Append positions K+1..top_out from the original k-reciprocal order.

Compares against the no-cross-encoder baseline on val. Writes a test
submission (per-class with cross-encoder re-rank) and the baseline (per-class
without cross-encoder, for diff-checking).
"""
import argparse
import csv
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from urban_elements_reid_challenge.utils.re_ranking import re_ranking  # noqa: E402

DATA = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026/Dataset/UrbanUAM_Merged'
CLASSES = ['container', 'crosswalk', 'rubbishbins', 'trafficsign']

# De90's per-class hyperparams (val-tuned on a model that hadn't seen val)
DE90_PARAMS = {
    'container':   (50, 6, 0.2),
    'crosswalk':   (20, 5, 0.1),
    'rubbishbins': (20, 2, 0.0),
    'trafficsign': (10, 3, 0.1),
}


class CrossEncoder(nn.Module):
    def __init__(self, feat_dim, hidden, dropout):
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


def norm_class(c):
    c = c.strip().lower()
    return 'trafficsign' if c == 'trafficsignal' else c


def read_classes(path):
    d = {}
    with open(path, newline='') as f:
        rd = csv.reader(f); hdr = next(rd)
        ni = hdr.index('imageName'); ci = hdr.index('Class')
        for r in rd: d[r[ni]] = norm_class(r[ci])
    return d


def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def compute_ap(order, q_pid, g_pids, q_cam, g_cams):
    keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
    order = order[keep]
    matches = (g_pids[order] == q_pid).astype(np.float32)
    if matches.sum() == 0: return None
    cum = np.cumsum(matches)
    prec = cum / (np.arange(len(matches)) + 1)
    return float((prec * matches).sum() / matches.sum())


@torch.no_grad()
def cross_score(model, qf, gf, indices, batch=4096):
    """For each query i, score (qf[i], gf[indices[i, :]]) with the model.
    Returns (Q, K) score matrix."""
    model.eval()
    Q, K = indices.shape
    qf_t = torch.from_numpy(qf).float().cuda()
    gf_t = torch.from_numpy(gf).float().cuda()
    out = np.zeros((Q, K), dtype=np.float32)
    # Flatten pairs and chunk
    flat_q, flat_g = [], []
    for i in range(Q):
        flat_q.append(np.full(K, i, dtype=np.int64))
        flat_g.append(indices[i])
    flat_q = np.concatenate(flat_q); flat_g = np.concatenate(flat_g)
    for s in range(0, len(flat_q), batch):
        e = s + batch
        q = qf_t[torch.from_numpy(flat_q[s:e])]
        g = gf_t[torch.from_numpy(flat_g[s:e])]
        out.flat[s:e] = model(q, g).cpu().numpy()
    return out


def perclass_topk(qf, gf, qcls, gcls, params, top_k=50):
    """Per-class k-reciprocal: for each query, return its top_k gallery
    indices (within the same class first, then fall back to cross-class).

    Returns:
        top_ind: (Q, top_k) int gallery indices
        top_dist: (Q, top_k) float (re-ranked or cosine, whichever is used)
    """
    n_q = qf.shape[0]
    out_ind = np.zeros((n_q, top_k), dtype=np.int64)
    out_dist = np.full((n_q, top_k), 2.0, dtype=np.float32)
    for c in CLASSES:
        qi = np.where(qcls == c)[0]
        gi_s = np.where(gcls == c)[0]
        gi_o = np.where(gcls != c)[0]
        if len(qi) == 0 or len(gi_s) == 0: continue
        k1, k2, lam = params.get(c, (50, 15, 0.3))
        qfq = qf[qi]; gfc = gf[gi_s]
        d_s = re_ranking(qfq @ gfc.T, qfq @ qfq.T, gfc @ gfc.T,
                         k1=k1, k2=k2, lambda_value=lam)
        d_o = 1.0 - qfq @ gf[gi_o].T if len(gi_o) > 0 else None
        for li, gq in enumerate(qi):
            order_s = np.argsort(d_s[li])
            ranked = gi_s[order_s]
            d_ranked = d_s[li, order_s]
            if len(ranked) >= top_k:
                out_ind[gq] = ranked[:top_k]
                out_dist[gq] = d_ranked[:top_k]
            else:
                need = top_k - len(ranked)
                order_o = np.argsort(d_o[li])[:need]
                out_ind[gq] = np.concatenate([ranked, gi_o[order_o]])
                out_dist[gq] = np.concatenate(
                    [d_ranked, d_o[li, order_o]])
    return out_ind, out_dist


def map_from_ranked(ranked_global_idx, qp, gp, qc, gc):
    """Given for each query a list of gallery indices already ordered by
    final ranking, compute mAP."""
    aps = []
    for i in range(len(qp)):
        order = ranked_global_idx[i]
        ap = compute_ap(np.arange(len(order)), qp[i], gp[order], qc[i], gc[order])
        # the helper above expects an "order over gp"; we already permuted gp
        # so we pass identity-order. Adjusted helper below for clarity:
        aps.append(ap if ap is not None else None)
    aps = [a for a in aps if a is not None]
    return float(np.mean(aps)) if aps else 0.0


def map_with_ranked(ranked_global_idx, qp, gp, qc, gc):
    aps = []
    for i in range(len(qp)):
        order = ranked_global_idx[i]
        keep = ~((gp[order] == qp[i]) & (gc[order] == qc[i]))
        order = order[keep]
        matches = (gp[order] == qp[i]).astype(np.float32)
        if matches.sum() == 0: continue
        cum = np.cumsum(matches)
        prec = cum / (np.arange(len(matches)) + 1)
        aps.append(float((prec * matches).sum() / matches.sum()))
    return float(np.mean(aps)) if aps else 0.0


def write_submission(ranked, out_csv, top_k=100):
    """ranked: (Q, K) int gallery indices already in final order."""
    n_q = ranked.shape[0]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(n_q):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (ranked[i, :top_k] + 1).tolist()))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',     required=True)
    ap.add_argument('--val_dir',  default='feat_cache/Hplus_val')
    ap.add_argument('--test_dir', default='feat_cache/Hplus_test')
    ap.add_argument('--tag',      required=True)
    ap.add_argument('--out_dir',  default='submissions')
    ap.add_argument('--rerank_k', type=int, default=50,
                    help='top-K re-ordered by cross-encoder; 51..100 keep '
                         'their k-reciprocal positions.')
    args = ap.parse_args()

    # ---- val ----
    qf_v = l2(np.load(os.path.join(args.val_dir, 'qf.npy')).astype(np.float32))
    gf_v = l2(np.load(os.path.join(args.val_dir, 'gf.npy')).astype(np.float32))
    with open(os.path.join(args.val_dir, 'meta.json')) as f: mv = json.load(f)
    qp = np.asarray(mv['query_pids'],   dtype=np.int64)
    gp = np.asarray(mv['gallery_pids'], dtype=np.int64)
    qc = np.asarray(mv['query_camids'], dtype=np.int64)
    gc = np.asarray(mv['gallery_camids'], dtype=np.int64)
    nq_v = read_classes(os.path.join(DATA, 'val_query_classes.csv'))
    ng_v = read_classes(os.path.join(DATA, 'val_test_classes.csv'))
    qcls_v = np.array([nq_v[os.path.basename(p)] for p in mv['query_paths']])
    gcls_v = np.array([ng_v[os.path.basename(p)] for p in mv['gallery_paths']])

    # ---- model ----
    ck = torch.load(args.ckpt, map_location='cpu')
    model = CrossEncoder(feat_dim=ck['feat_dim'], hidden=ck['hidden'],
                         dropout=ck['dropout']).cuda()
    model.load_state_dict(ck['model'])

    # ---- baseline per-class k-reciprocal top-100 ----
    print('[val] computing per-class k-reciprocal top-100...')
    base_ind, _ = perclass_topk(qf_v, gf_v, qcls_v, gcls_v, DE90_PARAMS, top_k=100)
    m_base = map_with_ranked(base_ind, qp, gp, qc, gc)
    print(f'[val] baseline (perclass) mAP = {m_base:.4f}')

    # Cross-encoder re-rank of top-K
    K = args.rerank_k
    print(f'[val] cross-encoder re-ranking top-{K}...')
    topK_ind = base_ind[:, :K]
    scores = cross_score(model, qf_v, gf_v, topK_ind)
    # Re-order top-K by descending score; concat 51..100 from baseline
    new_topK = np.take_along_axis(topK_ind, np.argsort(-scores, axis=1), axis=1)
    final_ind = np.concatenate([new_topK, base_ind[:, K:]], axis=1)
    m_x = map_with_ranked(final_ind, qp, gp, qc, gc)
    print(f'[val] cross-encoder mAP    = {m_x:.4f}  (Δ = {m_x - m_base:+.4f})')

    # Per-class breakdown
    print('[val] per-class breakdown:')
    for c in CLASSES:
        qi = np.where(qcls_v == c)[0]
        if len(qi) == 0: continue
        m_c_base = map_with_ranked(base_ind[qi], qp[qi], gp, qc[qi], gc)
        m_c_x    = map_with_ranked(final_ind[qi], qp[qi], gp, qc[qi], gc)
        print(f'  {c:12s} nq={len(qi):4d}  base={m_c_base:.4f}  '
              f'cross={m_c_x:.4f}  Δ={m_c_x - m_c_base:+.4f}')

    # ---- test side: build submission with whichever wins on val ----
    qf_t = l2(np.load(os.path.join(args.test_dir, 'qf.npy')).astype(np.float32))
    gf_t = l2(np.load(os.path.join(args.test_dir, 'gf.npy')).astype(np.float32))
    with open(os.path.join(args.test_dir, 'meta.json')) as f: mt = json.load(f)
    nq_t = read_classes(os.path.join(DATA, 'query_classes.csv'))
    ng_t = read_classes(os.path.join(DATA, 'test_classes.csv'))
    qcls_t = np.array([nq_t[os.path.basename(p)] for p in mt['query_paths']])
    gcls_t = np.array([ng_t[os.path.basename(p)] for p in mt['gallery_paths']])

    print('[test] computing per-class k-reciprocal top-100...')
    base_ind_t, _ = perclass_topk(qf_t, gf_t, qcls_t, gcls_t, DE90_PARAMS, top_k=100)
    print(f'[test] cross-encoder re-ranking top-{K}...')
    topK_ind_t = base_ind_t[:, :K]
    scores_t = cross_score(model, qf_t, gf_t, topK_ind_t)
    new_topK_t = np.take_along_axis(topK_ind_t, np.argsort(-scores_t, axis=1), axis=1)
    final_ind_t = np.concatenate([new_topK_t, base_ind_t[:, K:]], axis=1)

    out_x = os.path.join(args.out_dir, f'submission_{args.tag}_cross.csv')
    write_submission(final_ind_t, out_x)
    out_b = os.path.join(args.out_dir, f'submission_{args.tag}_baseline.csv')
    write_submission(base_ind_t, out_b)
    print(f'[submission] cross    -> {out_x}')
    print(f'[submission] baseline -> {out_b}')

    log_path = os.path.join(args.out_dir, f'eval_{args.tag}.json')
    with open(log_path, 'w') as f:
        json.dump({'tag': args.tag,
                   'val_baseline_mAP': m_base,
                   'val_cross_mAP': m_x,
                   'rerank_k': K,
                   'win': m_x > m_base}, f, indent=2)
    print(f'[log] {log_path}')


if __name__ == '__main__':
    main()
