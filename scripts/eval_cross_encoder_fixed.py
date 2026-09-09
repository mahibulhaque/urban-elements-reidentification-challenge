"""Re-evaluate the trained HpXC cross-encoder using the *correct* full-gallery
mAP metric (matches eval_variants.py exactly). Loads the cached cross-encoder
checkpoint + cached Hplus features, no retraining."""
import argparse, csv, json, os, sys
import numpy as np
import torch, torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from urban_elements_reid_challenge.utils.re_ranking import re_ranking  # noqa: E402

DATA = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026/Dataset/UrbanUAM_Merged'
CLASSES = ['container', 'crosswalk', 'rubbishbins', 'trafficsign']
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
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1))
    def forward(self, q, g):
        return self.mlp(torch.cat([q, g, (q-g).abs(), q*g], dim=-1)).squeeze(-1)


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
    """STANDARD mAP — denominator is total positives in full gallery (after
    same-camera removal). Matches eval_variants.compute_ap exactly."""
    keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
    order = order[keep]
    matches = (g_pids[order] == q_pid).astype(np.float32)
    n_pos = matches.sum()
    if n_pos == 0: return None
    cum = np.cumsum(matches)
    prec = cum / (np.arange(len(matches)) + 1)
    return float((prec * matches).sum() / n_pos)


def map_from_full_order(full_order, qp, gp, qc, gc):
    """full_order: (Q, G) int — full-gallery ranking per query."""
    aps = []
    for i in range(len(qp)):
        ap = compute_ap(full_order[i], qp[i], gp, qc[i], gc)
        if ap is not None: aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def perclass_full_order(qf, gf, qcls, gcls, params):
    """Per-class k-reciprocal full ranking: same-class items first (ranked by
    re-ranked distance), then cross-class items appended in cosine order.
    Returns (Q, G) int."""
    n_q, n_g = qf.shape[0], gf.shape[0]
    out = np.zeros((n_q, n_g), dtype=np.int64)
    for c in CLASSES:
        qi = np.where(qcls == c)[0]
        gi_s = np.where(gcls == c)[0]
        gi_o = np.where(gcls != c)[0]
        if len(qi) == 0: continue
        k1, k2, lam = params.get(c, (50, 15, 0.3))
        qfq = qf[qi]
        if len(gi_s) > 0:
            gfc = gf[gi_s]
            d_s = re_ranking(qfq @ gfc.T, qfq @ qfq.T, gfc @ gfc.T,
                             k1=k1, k2=k2, lambda_value=lam)
        if len(gi_o) > 0:
            d_o = 1.0 - qfq @ gf[gi_o].T
        for li, gq in enumerate(qi):
            parts = []
            if len(gi_s) > 0:
                parts.append(gi_s[np.argsort(d_s[li])])
            if len(gi_o) > 0:
                parts.append(gi_o[np.argsort(d_o[li])])
            full = np.concatenate(parts)
            # Pad if any gallery item missing (should not happen) — defensive
            if len(full) < n_g:
                missing = np.setdiff1d(np.arange(n_g), full)
                full = np.concatenate([full, missing])
            out[gq] = full[:n_g]
    return out


@torch.no_grad()
def cross_score(model, qf, gf, indices, batch=4096):
    model.eval()
    Q, K = indices.shape
    qf_t = torch.from_numpy(qf).float().cuda()
    gf_t = torch.from_numpy(gf).float().cuda()
    flat_q = np.repeat(np.arange(Q, dtype=np.int64), K)
    flat_g = indices.reshape(-1)
    out = np.zeros(Q*K, dtype=np.float32)
    for s in range(0, Q*K, batch):
        e = s + batch
        q = qf_t[torch.from_numpy(flat_q[s:e])]
        g = gf_t[torch.from_numpy(flat_g[s:e])]
        out[s:e] = model(q, g).cpu().numpy()
    return out.reshape(Q, K)


def write_submission(ranked, out_csv, top_k=100):
    n_q = ranked.shape[0]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(n_q):
            w.writerow(['{:06d}.jpg'.format(i+1),
                        ' '.join(map(str, (ranked[i, :top_k] + 1).tolist()))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',     default='outputs/HpXC/cross_best.pth')
    ap.add_argument('--val_dir',  default='feat_cache/Hplus_val')
    ap.add_argument('--test_dir', default='feat_cache/Hplus_test')
    ap.add_argument('--tag',      default='HpXC')
    ap.add_argument('--out_dir',  default='submissions')
    ap.add_argument('--rerank_k', type=int, default=50)
    args = ap.parse_args()

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

    ck = torch.load(args.ckpt, map_location='cpu')
    model = CrossEncoder(feat_dim=ck['feat_dim'], hidden=ck['hidden'],
                         dropout=ck['dropout']).cuda()
    model.load_state_dict(ck['model'])
    print(f'[ckpt] miniVal_score={ck.get("val_score", "?")}')

    # ---- val ----
    base_full = perclass_full_order(qf_v, gf_v, qcls_v, gcls_v, DE90_PARAMS)
    m_base = map_from_full_order(base_full, qp, gp, qc, gc)
    print(f'[val] baseline (perclass, full-gallery)  mAP = {m_base:.4f}')

    K = args.rerank_k
    topK = base_full[:, :K]
    sc = cross_score(model, qf_v, gf_v, topK)
    new_topK = np.take_along_axis(topK, np.argsort(-sc, axis=1), axis=1)
    cross_full = np.concatenate([new_topK, base_full[:, K:]], axis=1)
    m_x = map_from_full_order(cross_full, qp, gp, qc, gc)
    print(f'[val] cross-encoder (full-gallery)        mAP = {m_x:.4f}  '
          f'(Δ = {m_x - m_base:+.4f})')

    # per-class breakdown (full-gallery)
    print('[val] per-class:')
    for c in CLASSES:
        qi = np.where(qcls_v == c)[0]
        if len(qi) == 0: continue
        m_c_b = map_from_full_order(base_full[qi], qp[qi], gp, qc[qi], gc)
        m_c_x = map_from_full_order(cross_full[qi], qp[qi], gp, qc[qi], gc)
        print(f'  {c:12s} nq={len(qi):4d}  base={m_c_b:.4f}  '
              f'cross={m_c_x:.4f}  Δ={m_c_x - m_c_b:+.4f}')

    # ---- test side ----
    qf_t = l2(np.load(os.path.join(args.test_dir, 'qf.npy')).astype(np.float32))
    gf_t = l2(np.load(os.path.join(args.test_dir, 'gf.npy')).astype(np.float32))
    with open(os.path.join(args.test_dir, 'meta.json')) as f: mt = json.load(f)
    nq_t = read_classes(os.path.join(DATA, 'query_classes.csv'))
    ng_t = read_classes(os.path.join(DATA, 'test_classes.csv'))
    qcls_t = np.array([nq_t[os.path.basename(p)] for p in mt['query_paths']])
    gcls_t = np.array([ng_t[os.path.basename(p)] for p in mt['gallery_paths']])

    base_full_t = perclass_full_order(qf_t, gf_t, qcls_t, gcls_t, DE90_PARAMS)
    topK_t = base_full_t[:, :K]
    sc_t = cross_score(model, qf_t, gf_t, topK_t)
    new_topK_t = np.take_along_axis(topK_t, np.argsort(-sc_t, axis=1), axis=1)
    cross_full_t = np.concatenate([new_topK_t, base_full_t[:, K:]], axis=1)

    out_x = os.path.join(args.out_dir, f'submission_{args.tag}_cross_fixed.csv')
    out_b = os.path.join(args.out_dir, f'submission_{args.tag}_baseline_fixed.csv')
    write_submission(cross_full_t, out_x)
    write_submission(base_full_t,  out_b)
    print(f'[submission] {out_x}')
    print(f'[submission] {out_b}')

    with open(os.path.join(args.out_dir, f'eval_{args.tag}_fixed.json'), 'w') as f:
        json.dump({'tag': args.tag,
                   'val_baseline_mAP_full': m_base,
                   'val_cross_mAP_full':    m_x,
                   'rerank_k': K,
                   'win': m_x > m_base}, f, indent=2)


if __name__ == '__main__':
    main()
