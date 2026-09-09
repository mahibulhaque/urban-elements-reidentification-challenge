"""Class-Conditioned Feature Modulation (training-free) on cached features.

For each super-class c in {container, crosswalk, rubbishbins, trafficsign}:
  μ_c = mean of (L2-normalized) gallery features in class c
  σ_c = std  of (L2-normalized) gallery features in class c

Modulation modes (applied independently to query and gallery, then L2-renorm):
  baseline : f' = f
  cent     : f' = f - μ_c
  std      : f' = f / σ_c
  centstd  : f' = (f - μ_c) / σ_c

For each mode, score (a) pure cosine within class and (b) per-class
k-reciprocal re-rank (grid search). Pick the (mode, proc) with highest
weighted val mAP; write the corresponding test-side submission.

Stats are computed *per split* — val stats from val gallery, test stats
from test gallery — to avoid any cross-split leakage.
"""
import argparse
import csv
import json
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from urban_elements_reid_challenge.utils.re_ranking import re_ranking  # noqa: E402

DATA = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026/Dataset/UrbanUAM_Merged'
CLASSES = ['container', 'crosswalk', 'rubbishbins', 'trafficsign']
MODES = ['baseline', 'cent', 'std', 'centstd']


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


def l2(x, eps=1e-12):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=eps)


def compute_ap(order, q_pid, g_pids, q_cam, g_cams):
    keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
    order = order[keep]
    matches = (g_pids[order] == q_pid).astype(np.float32)
    n_pos = matches.sum()
    if n_pos == 0: return None
    cum = np.cumsum(matches)
    prec = cum / (np.arange(len(matches)) + 1)
    return float((prec * matches).sum() / n_pos)


def map_from_dist(dist, qp, gp, qc, gc):
    aps = []
    for i in range(len(qp)):
        order = np.argsort(dist[i])
        ap = compute_ap(order, qp[i], gp, qc[i], gc)
        if ap is not None: aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def modulate(qf, gf, q_cls, g_cls, mode, eps=1e-6):
    """Apply class-conditioned modulation. Stats computed from gallery."""
    if mode == 'baseline':
        return l2(qf), l2(gf)
    qf = qf.astype(np.float32).copy()
    gf = gf.astype(np.float32).copy()
    qf_n = l2(qf); gf_n = l2(gf)
    qf_out = np.empty_like(qf_n)
    gf_out = np.empty_like(gf_n)

    for c in CLASSES:
        gi = np.where(g_cls == c)[0]
        qi = np.where(q_cls == c)[0]
        if len(gi) == 0:
            if 'cent' in mode or 'std' in mode:
                # No gallery — fall back to raw for these (rare)
                if len(qi): qf_out[qi] = qf_n[qi]
            continue
        gal = gf_n[gi]
        mu = gal.mean(axis=0, keepdims=True) if 'cent' in mode else 0.0
        if 'std' in mode:
            sd = gal.std(axis=0, keepdims=True) + eps
        else:
            sd = 1.0
        gf_out[gi] = (gal - mu) / sd
        if len(qi):
            qf_out[qi] = (qf_n[qi] - mu) / sd

    # Any class with no gallery: copy raw queries through (defensive)
    untouched = np.setdiff1d(np.arange(len(qf_n)),
                             np.concatenate([np.where(q_cls == c)[0] for c in CLASSES] or [np.array([], dtype=int)]))
    if len(untouched):
        qf_out[untouched] = qf_n[untouched]

    return l2(qf_out), l2(gf_out)


def perclass_grid(qf, gf, qp, gp, qcls, gcls, qc, gc, K1, K2, LAM):
    best = {}
    tnum, tden = 0.0, 0
    for c in CLASSES:
        qi = np.where(qcls == c)[0]
        gi = np.where(gcls == c)[0]
        if len(qi) == 0 or len(gi) == 0: continue
        qfc, gfc = qf[qi], gf[gi]
        qpc, gpc = qp[qi], gp[gi]
        qcc, gcc = qc[qi], gc[gi]
        bm, bp = -1.0, None
        for k1 in K1:
            for k2 in K2:
                if k2 > k1: continue
                for lam in LAM:
                    d = re_ranking(qfc @ gfc.T, qfc @ qfc.T, gfc @ gfc.T,
                                   k1=k1, k2=k2, lambda_value=lam)
                    m = map_from_dist(d, qpc, gpc, qcc, gcc)
                    if m > bm:
                        bm, bp = m, (int(k1), int(k2), float(lam))
        best[c] = {'mAP': bm, 'params': bp, 'n_query': int(len(qi))}
        tnum += bm * len(qi); tden += len(qi)
    return tnum / max(tden, 1), best


def perclass_pure(qf, gf, qp, gp, qcls, gcls, qc, gc):
    """Pure intra-class cosine — equivalent to k-rerank with k1=0/lam=1."""
    tnum, tden = 0.0, 0
    detail = {}
    for c in CLASSES:
        qi = np.where(qcls == c)[0]
        gi = np.where(gcls == c)[0]
        if len(qi) == 0 or len(gi) == 0: continue
        d = 1.0 - qf[qi] @ gf[gi].T
        m = map_from_dist(d, qp[qi], gp[gi], qc[qi], gc[gi])
        detail[c] = {'mAP': m, 'n_query': int(len(qi))}
        tnum += m * len(qi); tden += len(qi)
    return tnum / max(tden, 1), detail


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


def write_perclass_pure_submission(qf, gf, qcls, gcls, out_csv, top_k=100):
    n_q = qf.shape[0]
    out = np.zeros((n_q, top_k), dtype=np.int64)
    for c in np.unique(qcls):
        qi = np.where(qcls == c)[0]
        gi_s = np.where(gcls == c)[0]
        gi_o = np.where(gcls != c)[0]
        d_s = 1.0 - qf[qi] @ gf[gi_s].T
        d_o = 1.0 - qf[qi] @ gf[gi_o].T if len(gi_o) > 0 else None
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
    ap.add_argument('--val_dir',  default='feat_cache/Hplus_val')
    ap.add_argument('--test_dir', default='feat_cache/Hplus_test')
    ap.add_argument('--tag',      default='HpCCFM')
    ap.add_argument('--out_dir',  default='submissions')
    args = ap.parse_args()

    # ---- val side ----
    qf_v = np.load(os.path.join(args.val_dir, 'qf.npy')).astype(np.float32)
    gf_v = np.load(os.path.join(args.val_dir, 'gf.npy')).astype(np.float32)
    with open(os.path.join(args.val_dir, 'meta.json')) as f: mv = json.load(f)
    qp = np.asarray(mv['query_pids'],   dtype=np.int64)
    gp = np.asarray(mv['gallery_pids'], dtype=np.int64)
    qc = np.asarray(mv['query_camids'], dtype=np.int64)
    gc = np.asarray(mv['gallery_camids'], dtype=np.int64)
    nq_v = read_classes(os.path.join(DATA, 'val_query_classes.csv'))
    ng_v = read_classes(os.path.join(DATA, 'val_test_classes.csv'))
    qcls_v = np.array([nq_v[os.path.basename(p)] for p in mv['query_paths']])
    gcls_v = np.array([ng_v[os.path.basename(p)] for p in mv['gallery_paths']])

    print(f'[val] qf={qf_v.shape}  gf={gf_v.shape}')
    results = {}
    for mode in MODES:
        qf_m, gf_m = modulate(qf_v, gf_v, qcls_v, gcls_v, mode)
        m_pure, pure_detail = perclass_pure(qf_m, gf_m, qp, gp, qcls_v, gcls_v, qc, gc)
        m_pc, pc_best = perclass_grid(
            qf_m, gf_m, qp, gp, qcls_v, gcls_v, qc, gc,
            K1=[10, 15, 20, 30, 50], K2=[2, 3, 5, 6], LAM=[0.0, 0.1, 0.2])
        results[mode] = {'pure': m_pure, 'perclass': m_pc,
                         'pure_detail': pure_detail, 'perclass_best': pc_best}
        print(f'  [{mode:8s}] pure={m_pure:.4f}  perclass={m_pc:.4f}')
        for c, info in pc_best.items():
            bp = info['params']
            print(f'      {c:12s} nq={info["n_query"]:4d}  '
                  f'mAP={info["mAP"]:.4f}  (k1={bp[0]}, k2={bp[1]}, lam={bp[2]})')

    # Pick winner
    best_score = -1.0; best_mode = None; best_proc = None
    for mode, r in results.items():
        for proc in ['pure', 'perclass']:
            if r[proc] > best_score:
                best_score, best_mode, best_proc = r[proc], mode, proc
    print(f'\n==> winner = {best_mode} / {best_proc}  (val mAP {best_score:.4f})')

    # ---- test side: same modulation, write submission with winner's recipe ----
    qf_t = np.load(os.path.join(args.test_dir, 'qf.npy')).astype(np.float32)
    gf_t = np.load(os.path.join(args.test_dir, 'gf.npy')).astype(np.float32)
    with open(os.path.join(args.test_dir, 'meta.json')) as f: mt = json.load(f)
    nq_t = read_classes(os.path.join(DATA, 'query_classes.csv'))
    ng_t = read_classes(os.path.join(DATA, 'test_classes.csv'))
    qcls_t = np.array([nq_t[os.path.basename(p)] for p in mt['query_paths']])
    gcls_t = np.array([ng_t[os.path.basename(p)] for p in mt['gallery_paths']])

    qf_tm, gf_tm = modulate(qf_t, gf_t, qcls_t, gcls_t, best_mode)
    out_csv = os.path.join(args.out_dir,
                           f'submission_{args.tag}_{best_mode}_{best_proc}.csv')
    if best_proc == 'pure':
        write_perclass_pure_submission(qf_tm, gf_tm, qcls_t, gcls_t, out_csv)
    else:
        write_perclass_submission(qf_tm, gf_tm, qcls_t, gcls_t,
                                  results[best_mode]['perclass_best'], out_csv)
    print(f'[submission] wrote {out_csv}')

    # Also write the best per-class re-rank submission unconditionally
    pc_winner = max(results, key=lambda m: results[m]['perclass'])
    pc_score = results[pc_winner]['perclass']
    qf_tm2, gf_tm2 = modulate(qf_t, gf_t, qcls_t, gcls_t, pc_winner)
    out2 = os.path.join(args.out_dir,
                        f'submission_{args.tag}_{pc_winner}_perclass.csv')
    write_perclass_submission(qf_tm2, gf_tm2, qcls_t, gcls_t,
                              results[pc_winner]['perclass_best'], out2)
    print(f'[submission/perclass-also] {pc_winner} ({pc_score:.4f}) -> {out2}')

    log_path = os.path.join(args.out_dir, f'eval_{args.tag}.json')
    with open(log_path, 'w') as f:
        json.dump({'tag': args.tag, 'results': results,
                   'best': {'mode': best_mode, 'proc': best_proc,
                            'val_mAP': best_score}}, f, indent=2)
    print(f'[log] {log_path}')


if __name__ == '__main__':
    main()
