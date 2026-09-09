"""Evaluate every token-pool variant from extract_dinov3_tokens.py on val:

  Singles : cls, gap, mxp, reg, stripe4
  Concats : cls+gap, cls+reg, cls+stripe4, cls+gap+reg

For each variant, compute:
  pure cosine, k-reciprocal rerank (default k1=20,k2=6,lam=0.3),
  per-class rerank with grid search.

Pick the (variant, post-proc) pair with the highest val mAP, then write a
test-side submission using that exact pipeline. Also print a per-class table.
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

SINGLES = ['cls', 'gap', 'mxp', 'reg', 'stripe4']
CONCATS = {
    'cls+gap':         ['cls', 'gap'],
    'cls+reg':         ['cls', 'reg'],
    'cls+stripe4':     ['cls', 'stripe4'],
    'cls+gap+reg':     ['cls', 'gap', 'reg'],
}


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


def load_variant(d, name):
    if name in SINGLES:
        q = np.load(os.path.join(d, f'{name}_q.npy')).astype(np.float32)
        g = np.load(os.path.join(d, f'{name}_g.npy')).astype(np.float32)
        return l2(q), l2(g)
    parts = CONCATS[name]
    qs = [l2(np.load(os.path.join(d, f'{p}_q.npy')).astype(np.float32)) for p in parts]
    gs = [l2(np.load(os.path.join(d, f'{p}_g.npy')).astype(np.float32)) for p in parts]
    q = np.concatenate(qs, axis=1)
    g = np.concatenate(gs, axis=1)
    return l2(q), l2(g)


def compute_ap(order, q_pid, g_pids, q_cam, g_cams):
    keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
    order = order[keep]
    matches = (g_pids[order] == q_pid).astype(np.float32)
    n_pos = matches.sum()
    if n_pos == 0: return None
    cum = np.cumsum(matches)
    prec = cum / (np.arange(len(matches)) + 1)
    return float((prec * matches).sum() / n_pos)


def map_from_dist(dist, q_pids, g_pids, q_cam, g_cam):
    aps = []
    for i in range(len(q_pids)):
        order = np.argsort(dist[i])
        ap = compute_ap(order, q_pids[i], g_pids, q_cam[i], g_cam)
        if ap is not None: aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def pure_map(qf, gf, qp, gp, qc, gc):
    return map_from_dist(1.0 - qf @ gf.T, qp, gp, qc, gc)


def rerank_map(qf, gf, qp, gp, qc, gc, k1=20, k2=6, lam=0.3):
    dist = re_ranking(qf @ gf.T, qf @ qf.T, gf @ gf.T, k1=k1, k2=k2, lambda_value=lam)
    return map_from_dist(dist, qp, gp, qc, gc)


def perclass_grid(qf, gf, qp, gp, qcls, gcls, qc, gc, K1, K2, LAM):
    best = {}
    tnum, tden = 0.0, 0
    for cls in ['container', 'crosswalk', 'rubbishbins', 'trafficsign']:
        qi = np.where(qcls == cls)[0]
        gi = np.where(gcls == cls)[0]
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
        best[cls] = {'mAP': bm, 'params': bp, 'n_query': int(len(qi))}
        tnum += bm * len(qi); tden += len(qi)
    return tnum / max(tden, 1), best


def write_perclass_submission(qf, gf, qcls, gcls, best, out_csv, top_k=100):
    n_q = qf.shape[0]
    out = np.zeros((n_q, top_k), dtype=np.int64)
    for cls in np.unique(qcls):
        qi = np.where(qcls == cls)[0]
        gi_s = np.where(gcls == cls)[0]
        gi_o = np.where(gcls != cls)[0]
        if cls in best and best[cls].get('params') is not None:
            k1, k2, lam = best[cls]['params']
        else:
            k1, k2, lam = 50, 15, 0.3
        qfq = qf[qi]; gfc = gf[gi_s]
        d_s = re_ranking(qfq @ gfc.T, qfq @ qfq.T, gfc @ gfc.T,
                         k1=k1, k2=k2, lambda_value=lam)
        d_o = 1.0 - qfq @ gf[gi_o].T if len(gi_o) > 0 else None
        for li, gq in enumerate(qi):
            order_s = np.argsort(d_s[li])
            ranked_s = gi_s[order_s]
            if len(ranked_s) >= top_k:
                out[gq] = ranked_s[:top_k]
            else:
                need = top_k - len(ranked_s)
                order_o = np.argsort(d_o[li])[:need]
                out[gq] = np.concatenate([ranked_s, gi_o[order_o]])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(n_q):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (out[i] + 1).tolist()))])


def write_rerank_submission(qf, gf, k1, k2, lam, out_csv, top_k=100):
    dist = re_ranking(qf @ gf.T, qf @ qf.T, gf @ gf.T, k1=k1, k2=k2, lambda_value=lam)
    top = np.argsort(dist, axis=1)[:, :top_k]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(len(top)):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (top[i] + 1).tolist()))])


def write_pure_submission(qf, gf, out_csv, top_k=100):
    dist = 1.0 - qf @ gf.T
    top = np.argsort(dist, axis=1)[:, :top_k]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(len(top)):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (top[i] + 1).tolist()))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_dir', required=True)
    ap.add_argument('--test_dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out_dir', default='submissions')
    ap.add_argument('--k1', type=int, default=20)
    ap.add_argument('--k2', type=int, default=6)
    ap.add_argument('--lam', type=float, default=0.3)
    args = ap.parse_args()

    with open(os.path.join(args.val_dir, 'meta.json')) as f:
        mv = json.load(f)
    qp = np.asarray(mv['query_pids'], dtype=np.int64)
    gp = np.asarray(mv['gallery_pids'], dtype=np.int64)
    qc = np.asarray(mv['query_camids'], dtype=np.int64)
    gc = np.asarray(mv['gallery_camids'], dtype=np.int64)
    nq_v = read_classes(os.path.join(DATA, 'val_query_classes.csv'))
    ng_v = read_classes(os.path.join(DATA, 'val_test_classes.csv'))
    qcls_v = np.array([nq_v[os.path.basename(p)] for p in mv['query_paths']])
    gcls_v = np.array([ng_v[os.path.basename(p)] for p in mv['gallery_paths']])

    variant_names = SINGLES + list(CONCATS.keys())
    results = {}
    for name in variant_names:
        qf, gf = load_variant(args.val_dir, name)
        m_pure = pure_map(qf, gf, qp, gp, qc, gc)
        m_rerk = rerank_map(qf, gf, qp, gp, qc, gc, args.k1, args.k2, args.lam)
        m_pc, pc_best = perclass_grid(
            qf, gf, qp, gp, qcls_v, gcls_v, qc, gc,
            K1=[10, 15, 20, 30, 50], K2=[2, 3, 5, 6], LAM=[0.0, 0.1, 0.2])
        results[name] = {
            'dim': int(qf.shape[1]),
            'pure': m_pure, 'rerank': m_rerk, 'perclass': m_pc,
            'perclass_best': pc_best,
        }
        print(f'[{name:14s}] dim={qf.shape[1]:5d}  pure={m_pure:.4f}  '
              f'rerank={m_rerk:.4f}  perclass={m_pc:.4f}')
        for cls, info in pc_best.items():
            bp = info['params']
            print(f'      {cls:12s} nq={info["n_query"]:4d}  '
                  f'mAP={info["mAP"]:.4f}  (k1={bp[0]}, k2={bp[1]}, lam={bp[2]})')

    # Pick the (variant, post-proc) with the best val mAP overall.
    best_score = -1.0
    best_name, best_proc = None, None
    for name, r in results.items():
        for proc in ['pure', 'rerank', 'perclass']:
            if r[proc] > best_score:
                best_score = r[proc]; best_name = name; best_proc = proc
    print(f'\n==> best = {best_name} / {best_proc}  (val mAP {best_score:.4f})')

    # Write submission for the winner.
    qf_t, gf_t = load_variant(args.test_dir, best_name)
    out_csv = os.path.join(args.out_dir,
                           f'submission_{args.tag}_{best_name}_{best_proc}.csv')

    if best_proc == 'pure':
        write_pure_submission(qf_t, gf_t, out_csv)
    elif best_proc == 'rerank':
        write_rerank_submission(qf_t, gf_t, args.k1, args.k2, args.lam, out_csv)
    else:
        with open(os.path.join(args.test_dir, 'meta.json')) as f:
            mt = json.load(f)
        nq_t = read_classes(os.path.join(DATA, 'query_classes.csv'))
        ng_t = read_classes(os.path.join(DATA, 'test_classes.csv'))
        qcls_t = np.array([nq_t[os.path.basename(p)] for p in mt['query_paths']])
        gcls_t = np.array([ng_t[os.path.basename(p)] for p in mt['gallery_paths']])
        write_perclass_submission(
            qf_t, gf_t, qcls_t, gcls_t,
            results[best_name]['perclass_best'], out_csv)

    print(f'[submission] wrote {out_csv}')

    # Also dump the perclass winner submission unconditionally, in case the
    # absolute winner is "pure" (rare but cheap to keep both).
    if best_proc != 'perclass':
        # Pick best-perclass variant separately and write it too.
        pc_best_name = max(results, key=lambda n: results[n]['perclass'])
        pc_score = results[pc_best_name]['perclass']
        qf_t2, gf_t2 = load_variant(args.test_dir, pc_best_name)
        with open(os.path.join(args.test_dir, 'meta.json')) as f:
            mt = json.load(f)
        nq_t = read_classes(os.path.join(DATA, 'query_classes.csv'))
        ng_t = read_classes(os.path.join(DATA, 'test_classes.csv'))
        qcls_t = np.array([nq_t[os.path.basename(p)] for p in mt['query_paths']])
        gcls_t = np.array([ng_t[os.path.basename(p)] for p in mt['gallery_paths']])
        out_csv2 = os.path.join(
            args.out_dir,
            f'submission_{args.tag}_{pc_best_name}_perclass.csv')
        write_perclass_submission(
            qf_t2, gf_t2, qcls_t, gcls_t,
            results[pc_best_name]['perclass_best'], out_csv2)
        print(f'[submission/perclass-also] {pc_best_name} ({pc_score:.4f}) -> {out_csv2}')

    log_path = os.path.join(args.out_dir, f'eval_{args.tag}_tokens.json')
    with open(log_path, 'w') as f:
        json.dump({'tag': args.tag, 'results': results,
                   'best': {'variant': best_name, 'proc': best_proc,
                            'val_mAP': best_score}}, f, indent=2)
    print(f'[log] {log_path}')


if __name__ == '__main__':
    main()
