"""Apply trained projection head to cached val/test Hplus features, grid an
alpha-blend with the raw features, score on val (pure cosine + per-class
k-rerank), and write the test submission for the winner.
"""
import argparse
import csv
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from urban_elements_reid_challenge.utils.re_ranking import re_ranking  # noqa: E402

DATA = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026/Dataset/UrbanUAM_Merged'
CLASSES = ['container', 'crosswalk', 'rubbishbins', 'trafficsign']
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


class ProjHead(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn  = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        z = self.fc2(self.drop(self.act(self.bn(self.fc1(x)))))
        return F.normalize(z, dim=-1)


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


@torch.no_grad()
def project(head, x_np):
    """Project a (N, D) numpy array through the head; return (N, D) np L2-norm."""
    head.eval()
    out = []
    for i in range(0, len(x_np), 1024):
        chunk = torch.from_numpy(x_np[i:i+1024]).cuda().float()
        out.append(head(chunk).cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_ap(order, q_pid, g_pids, q_cam, g_cams):
    keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
    order = order[keep]
    matches = (g_pids[order] == q_pid).astype(np.float32)
    if matches.sum() == 0: return None
    cum = np.cumsum(matches)
    prec = cum / (np.arange(len(matches)) + 1)
    return float((prec * matches).sum() / matches.sum())


def map_from_dist(dist, qp, gp, qc, gc):
    aps = []
    for i in range(len(qp)):
        order = np.argsort(dist[i])
        ap = compute_ap(order, qp[i], gp, qc[i], gc)
        if ap is not None: aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def perclass_grid(qf, gf, qp, gp, qcls, gcls, qc, gc, K1, K2, LAM):
    best = {}; tnum, tden = 0.0, 0
    for c in CLASSES:
        qi = np.where(qcls == c)[0]; gi = np.where(gcls == c)[0]
        if len(qi) == 0 or len(gi) == 0: continue
        qfc, gfc = qf[qi], gf[gi]
        qpc, gpc = qp[qi], gp[gi]; qcc, gcc = qc[qi], gc[gi]
        bm, bp = -1.0, None
        for k1 in K1:
            for k2 in K2:
                if k2 > k1: continue
                for lam in LAM:
                    d = re_ranking(qfc @ gfc.T, qfc @ qfc.T, gfc @ gfc.T,
                                   k1=k1, k2=k2, lambda_value=lam)
                    m = map_from_dist(d, qpc, gpc, qcc, gcc)
                    if m > bm: bm, bp = m, (int(k1), int(k2), float(lam))
        best[c] = {'mAP': bm, 'params': bp, 'n_query': int(len(qi))}
        tnum += bm * len(qi); tden += len(qi)
    return tnum / max(tden, 1), best


def write_perclass_submission(qf, gf, qcls, gcls, best_params, out_csv, top_k=100):
    n_q = qf.shape[0]
    out = np.zeros((n_q, top_k), dtype=np.int64)
    for c in np.unique(qcls):
        qi = np.where(qcls == c)[0]
        gi_s = np.where(gcls == c)[0]
        gi_o = np.where(gcls != c)[0]
        if c in best_params and best_params[c].get('params') is not None:
            k1, k2, lam = best_params[c]['params']
        else:
            k1, k2, lam = 50, 15, 0.3
        qfq = qf[qi]; gfc = gf[gi_s]
        d_s = re_ranking(qfq @ gfc.T, qfq @ qfq.T, gfc @ gfc.T,
                         k1=k1, k2=k2, lambda_value=lam)
        d_o = 1.0 - qfq @ gf[gi_o].T if len(gi_o) > 0 else None
        for li, gq in enumerate(qi):
            order = np.argsort(d_s[li])
            ranked = gi_s[order]
            if len(ranked) >= top_k:
                out[gq] = ranked[:top_k]
            else:
                need = top_k - len(ranked)
                order_o = np.argsort(d_o[li])[:need]
                out[gq] = np.concatenate([ranked, gi_o[order_o]])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(n_q):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (out[i] + 1).tolist()))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head_pth', required=True)
    ap.add_argument('--val_dir',  default='feat_cache/Hplus_val')
    ap.add_argument('--test_dir', default='feat_cache/Hplus_test')
    ap.add_argument('--tag',      required=True)
    ap.add_argument('--out_dir',  default='submissions')
    args = ap.parse_args()

    ck = torch.load(args.head_pth, map_location='cpu')
    head = ProjHead(in_dim=ck['in_dim'], hidden=ck['hidden'],
                    out_dim=ck['out_dim'], dropout=ck.get('dropout', 0.1)).cuda()
    head.load_state_dict(ck['head'])

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

    qf_p = project(head, qf_v); gf_p = project(head, gf_v)

    # Only meaningful if same dim for residual blend
    can_blend = (qf_p.shape[1] == qf_v.shape[1])
    if not can_blend:
        print('[blend] proj-dim != input-dim; using alpha=1.0 only')
    alphas = ALPHAS if can_blend else [1.0]

    print(f'[val] qf={qf_v.shape}  proj={qf_p.shape}  blendable={can_blend}')
    results = {}
    for a in alphas:
        if can_blend:
            qf_a = l2(a * qf_p + (1 - a) * qf_v)
            gf_a = l2(a * gf_p + (1 - a) * gf_v)
        else:
            qf_a, gf_a = qf_p, gf_p
        # pure intra-class cosine summarized as weighted-mean per-class
        # (we fold this into perclass_grid by running it once with no rerank
        # via small K — but easier: report perclass-rerank only).
        m_pc, pc_best = perclass_grid(
            qf_a, gf_a, qp, gp, qcls_v, gcls_v, qc, gc,
            K1=[10, 15, 20, 30, 50], K2=[2, 3, 5, 6], LAM=[0.0, 0.1, 0.2])
        # also a global pure-cosine eval (no per-class restriction)
        d_pure = 1.0 - qf_a @ gf_a.T
        m_pure = map_from_dist(d_pure, qp, gp, qc, gc)
        d_rer = re_ranking(qf_a @ gf_a.T, qf_a @ qf_a.T, gf_a @ gf_a.T,
                           k1=20, k2=6, lambda_value=0.3)
        m_rer = map_from_dist(d_rer, qp, gp, qc, gc)
        results[a] = {'pure': m_pure, 'rerank': m_rer, 'perclass': m_pc,
                      'perclass_best': pc_best}
        print(f'  alpha={a:.2f}  pure={m_pure:.4f}  rerank={m_rer:.4f}  perclass={m_pc:.4f}')
        for c, info in pc_best.items():
            bp = info['params']
            print(f'      {c:12s} nq={info["n_query"]:4d}  '
                  f'mAP={info["mAP"]:.4f}  (k1={bp[0]}, k2={bp[1]}, lam={bp[2]})')

    # Pick winner
    best_score = -1.0; best_a = None; best_proc = None
    for a, r in results.items():
        for proc in ['pure', 'rerank', 'perclass']:
            if r[proc] > best_score:
                best_score, best_a, best_proc = r[proc], a, proc
    print(f'\n==> winner alpha={best_a}  proc={best_proc}  val_mAP={best_score:.4f}')

    # ---- test side: write submission for winner + perclass-best (separately) ----
    qf_t = l2(np.load(os.path.join(args.test_dir, 'qf.npy')).astype(np.float32))
    gf_t = l2(np.load(os.path.join(args.test_dir, 'gf.npy')).astype(np.float32))
    with open(os.path.join(args.test_dir, 'meta.json')) as f: mt = json.load(f)
    nq_t = read_classes(os.path.join(DATA, 'query_classes.csv'))
    ng_t = read_classes(os.path.join(DATA, 'test_classes.csv'))
    qcls_t = np.array([nq_t[os.path.basename(p)] for p in mt['query_paths']])
    gcls_t = np.array([ng_t[os.path.basename(p)] for p in mt['gallery_paths']])

    qf_tp = project(head, qf_t); gf_tp = project(head, gf_t)

    def _build(a):
        if can_blend:
            return l2(a * qf_tp + (1 - a) * qf_t), l2(a * gf_tp + (1 - a) * gf_t)
        return qf_tp, gf_tp

    # Always write the perclass submission for the alpha that won perclass.
    pc_winner_a = max(alphas, key=lambda a: results[a]['perclass'])
    qf_w, gf_w = _build(pc_winner_a)
    out_csv = os.path.join(args.out_dir,
                           f'submission_{args.tag}_a{pc_winner_a:.2f}_perclass.csv')
    write_perclass_submission(qf_w, gf_w, qcls_t, gcls_t,
                              results[pc_winner_a]['perclass_best'], out_csv)
    print(f'[submission] perclass-winner -> {out_csv}')

    log_path = os.path.join(args.out_dir, f'eval_{args.tag}_msloss.json')
    # results dict has tuples in perclass_best; json-friendly:
    out = {a: {k: (v if k != 'perclass_best' else v) for k, v in r.items()}
           for a, r in results.items()}
    with open(log_path, 'w') as f:
        json.dump({'tag': args.tag,
                   'alphas': alphas, 'results': out,
                   'best': {'alpha': best_a, 'proc': best_proc, 'val_mAP': best_score},
                   'perclass_winner_alpha': pc_winner_a}, f, indent=2)
    print(f'[log] {log_path}')


if __name__ == '__main__':
    main()
