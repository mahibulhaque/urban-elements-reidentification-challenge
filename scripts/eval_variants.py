"""Compare three post-processing variants on a single cached feature dir:

  1. Pure cosine (no post-processing)
  2. Standard k-reciprocal re-ranking (k1=20, k2=6, lam=0.3)
  3. Per-class k-reciprocal re-ranking with per-class grid search

Reports val mAP for each, then writes a test submission from the winner.
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


def load_feat(d):
    qf = np.load(os.path.join(d, 'qf.npy'))
    gf = np.load(os.path.join(d, 'gf.npy'))
    with open(os.path.join(d, 'meta.json')) as f: m = json.load(f)
    return qf, gf, m


def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


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


def pure_map(qf, gf, q_pids, g_pids, q_cam, g_cam):
    return map_from_dist(1.0 - qf @ gf.T, q_pids, g_pids, q_cam, g_cam)


def rerank_map(qf, gf, q_pids, g_pids, q_cam, g_cam, k1=20, k2=6, lam=0.3):
    dist = re_ranking(qf @ gf.T, qf @ qf.T, gf @ gf.T, k1=k1, k2=k2, lambda_value=lam)
    return map_from_dist(dist, q_pids, g_pids, q_cam, g_cam)


def perclass_grid(qf, gf, q_pids, g_pids, q_cls, g_cls, q_cam, g_cam,
                  K1, K2, LAM):
    """Per-class grid search. Returns (overall_weighted_mAP, best_params_dict)."""
    best = {}
    total_num, total_den = 0.0, 0
    for cls in ['container', 'crosswalk', 'rubbishbins', 'trafficsign']:
        qi = np.where(q_cls == cls)[0]
        gi = np.where(g_cls == cls)[0]
        if len(qi) == 0 or len(gi) == 0: continue
        qf_c, gf_c = qf[qi], gf[gi]
        qp_c, gp_c = q_pids[qi], g_pids[gi]
        qc_c, gc_c = q_cam[qi], g_cam[gi]
        best_m, best_p = -1.0, None
        for k1 in K1:
            for k2 in K2:
                if k2 > k1: continue
                for lam in LAM:
                    dist = re_ranking(qf_c @ gf_c.T, qf_c @ qf_c.T, gf_c @ gf_c.T,
                                      k1=k1, k2=k2, lambda_value=lam)
                    m = map_from_dist(dist, qp_c, gp_c, qc_c, gc_c)
                    if m > best_m:
                        best_m, best_p = m, (int(k1), int(k2), float(lam))
        best[cls] = {'mAP': best_m, 'params': best_p, 'n_query': int(len(qi))}
        total_num += best_m * len(qi)
        total_den += len(qi)
    overall = total_num / max(total_den, 1)
    return overall, best


def write_submission_fullrank(qf, gf, out_csv, k1, k2, lam, top_k=100):
    dist = re_ranking(qf @ gf.T, qf @ qf.T, gf @ gf.T, k1=k1, k2=k2, lambda_value=lam)
    top = np.argsort(dist, axis=1)[:, :top_k]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(len(top)):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (top[i] + 1).tolist()))])


def write_submission_pure(qf, gf, out_csv, top_k=100):
    dist = 1.0 - qf @ gf.T
    top = np.argsort(dist, axis=1)[:, :top_k]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(len(top)):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (top[i] + 1).tolist()))])


def write_submission_perclass(qf, gf, q_cls, g_cls, best_params, out_csv, top_k=100):
    n_q = qf.shape[0]
    all_ranked = np.zeros((n_q, top_k), dtype=np.int64)
    for cls in np.unique(q_cls):
        qi = np.where(q_cls == cls)[0]
        gi_same = np.where(g_cls == cls)[0]
        gi_other = np.where(g_cls != cls)[0]
        if cls in best_params and best_params[cls].get('params') is not None:
            k1, k2, lam = best_params[cls]['params']
        else:
            k1, k2, lam = 50, 15, 0.3
        qf_q = qf[qi]; gf_c = gf[gi_same]
        dist_same = re_ranking(qf_q @ gf_c.T, qf_q @ qf_q.T, gf_c @ gf_c.T,
                               k1=k1, k2=k2, lambda_value=lam)
        dist_other = 1.0 - qf_q @ gf[gi_other].T if len(gi_other) > 0 else None
        for local_i, global_q in enumerate(qi):
            order_same = np.argsort(dist_same[local_i])
            ranked_same = gi_same[order_same]
            if len(ranked_same) >= top_k:
                all_ranked[global_q] = ranked_same[:top_k]
            else:
                need = top_k - len(ranked_same)
                order_other = np.argsort(dist_other[local_i])[:need]
                all_ranked[global_q] = np.concatenate(
                    [ranked_same, gi_other[order_other]])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(n_q):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (all_ranked[i] + 1).tolist()))])


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

    # --- val side ---
    qf_v, gf_v, mv = load_feat(args.val_dir)
    qf_v, gf_v = l2(qf_v.astype(np.float32)), l2(gf_v.astype(np.float32))
    q_pids = np.asarray(mv['query_pids'], dtype=np.int64)
    g_pids = np.asarray(mv['gallery_pids'], dtype=np.int64)
    q_cam = np.asarray(mv['query_camids'], dtype=np.int64)
    g_cam = np.asarray(mv['gallery_camids'], dtype=np.int64)

    # classes for per-class (val files use uamq_/uamt_ prefixes; no collision)
    nq_v = read_classes(os.path.join(DATA, 'val_query_classes.csv'))
    ng_v = read_classes(os.path.join(DATA, 'val_test_classes.csv'))
    q_cls_v = np.array([nq_v[os.path.basename(p)] for p in mv['query_paths']])
    g_cls_v = np.array([ng_v[os.path.basename(p)] for p in mv['gallery_paths']])

    print(f'[val] qf={qf_v.shape}  gf={gf_v.shape}')
    m_pure = pure_map(qf_v, gf_v, q_pids, g_pids, q_cam, g_cam)
    m_rerk = rerank_map(qf_v, gf_v, q_pids, g_pids, q_cam, g_cam,
                        k1=args.k1, k2=args.k2, lam=args.lam)
    m_pc, pc_best = perclass_grid(qf_v, gf_v, q_pids, g_pids, q_cls_v, g_cls_v,
                                  q_cam, g_cam,
                                  K1=[10, 15, 20, 30, 50],
                                  K2=[2, 3, 5, 6],
                                  LAM=[0.0, 0.1, 0.2])
    print()
    print(f'  [{args.tag}] pure cosine         val mAP = {m_pure:.4f}')
    print(f'  [{args.tag}] k-reciprocal rerank val mAP = {m_rerk:.4f}  '
          f'(k1={args.k1}, k2={args.k2}, lam={args.lam})')
    print(f'  [{args.tag}] per-class rerank    val mAP = {m_pc:.4f}')
    for cls, info in pc_best.items():
        bp = info['params']
        print(f'      {cls:12s} nq={info["n_query"]:4d}  mAP={info["mAP"]:.4f}  '
              f'(k1={bp[0]}, k2={bp[1]}, lam={bp[2]})')

    scores = {'pure': m_pure, 'rerank': m_rerk, 'perclass': m_pc}
    best_name = max(scores, key=scores.get)
    print(f'\n==> best variant: {best_name}  ({scores[best_name]:.4f})')

    # --- test side: build submission from the winner ---
    qf_t, gf_t, mt = load_feat(args.test_dir)
    qf_t, gf_t = l2(qf_t.astype(np.float32)), l2(gf_t.astype(np.float32))

    out_csv = os.path.join(args.out_dir, f'submission_{args.tag}_{best_name}.csv')
    if best_name == 'pure':
        write_submission_pure(qf_t, gf_t, out_csv)
    elif best_name == 'rerank':
        write_submission_fullrank(qf_t, gf_t, out_csv, args.k1, args.k2, args.lam)
    else:
        # per-class: need test-side class labels (separate q/g maps to avoid
        # the basename-collision bug we already hit)
        nq_t = read_classes(os.path.join(DATA, 'query_classes.csv'))
        ng_t = read_classes(os.path.join(DATA, 'test_classes.csv'))
        q_cls_t = np.array([nq_t[os.path.basename(p)] for p in mt['query_paths']])
        g_cls_t = np.array([ng_t[os.path.basename(p)] for p in mt['gallery_paths']])
        write_submission_perclass(qf_t, gf_t, q_cls_t, g_cls_t, pc_best, out_csv)

    print(f'[submission] wrote {out_csv}')

    # dump everything
    log_path = os.path.join(args.out_dir, f'eval_{args.tag}.json')
    with open(log_path, 'w') as f:
        json.dump({'tag': args.tag, 'val_scores': scores,
                   'best_variant': best_name,
                   'perclass_params': pc_best,
                   'submission': out_csv}, f, indent=2)
    print(f'[log] saved scores to {log_path}')


if __name__ == '__main__':
    main()
