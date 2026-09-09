"""Test-time 'last-N CLS tokens' experiment with CONCATENATION for ViT-based ReID.

Given a cached per-block CLS cube (qf_all, gf_all of shape (N, depth, D)
produced by extract_features_allblocks.py), sweep N in {1,2,3,4,6,8,12,16,24},
form CONCATENATION of the last-N CLS tokens, L2-normalize, and report val mAP with
per-class breakdown, (a) pure cosine and (b) k-reciprocal re-ranking.
Pick the best-N and write a test submission.
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


def map_from_dist_perclass(dist, q_pids, g_pids, q_cam, g_cam, q_classes=None):
    """Compute mAP overall and per-class if classes provided."""
    aps = []
    class_aps = {}

    for i in range(len(q_pids)):
        order = np.argsort(dist[i])
        ap = compute_ap(order, q_pids[i], g_pids, q_cam[i], g_cam)
        if ap is not None:
            aps.append(ap)
            if q_classes is not None:
                cls = q_classes[i]
                if cls not in class_aps:
                    class_aps[cls] = []
                class_aps[cls].append(ap)

    # Compute per-class mAP
    class_map = {}
    for cls, cls_aps in class_aps.items():
        class_map[str(cls)] = float(np.mean(cls_aps))

    overall_map = float(np.mean(aps)) if aps else 0.0
    return overall_map, class_map


def pure_map(qf, gf, q_pids, g_pids, q_cam, g_cam, q_classes=None):
    return map_from_dist_perclass(1.0 - qf @ gf.T, q_pids, g_pids, q_cam, g_cam, q_classes)


def rerank_map(qf, gf, q_pids, g_pids, q_cam, g_cam, q_classes=None, k1=20, k2=6, lam=0.3):
    dist = re_ranking(qf @ gf.T, qf @ qf.T, gf @ gf.T, k1=k1, k2=k2, lambda_value=lam)
    return map_from_dist_perclass(dist, q_pids, g_pids, q_cam, g_cam, q_classes)


def perclass_grid(qf, gf, q_pids, g_pids, q_cls, g_cls, q_cam, g_cam,
                  K1=[10, 15, 20, 30, 50], K2=[2, 3, 5, 6], LAM=[0.0, 0.1, 0.2, 0.3]):
    """Per-class grid search for optimal re-ranking parameters."""
    best = {}
    total_num, total_den = 0.0, 0

    for cls in ['container', 'crosswalk', 'rubbishbins', 'trafficsign']:
        qi = np.where(q_cls == cls)[0]
        gi = np.where(g_cls == cls)[0]
        if len(qi) == 0 or len(gi) == 0:
            continue

        qf_c, gf_c = qf[qi], gf[gi]
        qp_c, gp_c = q_pids[qi], g_pids[gi]
        qc_c, gc_c = q_cam[qi], g_cam[gi]

        # Limit k1/k2 to gallery size
        max_k1 = min(len(gi) - 1, 50)
        valid_k1 = [k for k in K1 if k <= max_k1]
        if not valid_k1:
            valid_k1 = [min(len(gi) - 1, 15)]

        best_m, best_p = -1.0, None
        for k1 in valid_k1:
            for k2 in K2:
                if k2 >= k1:
                    continue
                for lam in LAM:
                    dist = re_ranking(qf_c @ gf_c.T, qf_c @ qf_c.T, gf_c @ gf_c.T,
                                      k1=k1, k2=k2, lambda_value=lam)
                    m, _ = map_from_dist_perclass(dist, qp_c, gp_c, qc_c, gc_c)
                    if m > best_m:
                        best_m, best_p = m, (int(k1), int(k2), float(lam))

        best[cls] = {'mAP': best_m, 'params': best_p, 'n_query': int(len(qi))}
        total_num += best_m * len(qi)
        total_den += len(qi)

    overall = total_num / max(total_den, 1)
    return overall, best


def write_submission_perclass(qf, gf, q_cls, g_cls, best_params, out_csv, top_k=100):
    """Build submission using class-specific re-ranking parameters."""
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

        qf_q = qf[qi]
        gf_c = gf[gi_same]
        dist_same = re_ranking(qf_q @ gf_c.T, qf_q @ qf_q.T, gf_c @ gf_c.T,
                               k1=k1, k2=k2, lambda_value=lam)
        dist_other = 1.0 - qf_q @ gf[gi_other].T if len(gi_other) > 0 else None

        for local_i, global_q in enumerate(qi):
            order_same = np.argsort(dist_same[local_i])
            ranked_same = gi_same[order_same]

            if len(ranked_same) >= top_k:
                all_ranked[global_q] = ranked_same[:top_k]
            else:
                all_ranked[global_q, :len(ranked_same)] = ranked_same
                if len(gi_other) > 0:
                    order_other = np.argsort(dist_other[local_i])
                    ranked_other = gi_other[order_other]
                    remain = top_k - len(ranked_same)
                    all_ranked[global_q, len(ranked_same):] = ranked_other[:remain]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(len(all_ranked)):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (all_ranked[i] + 1).tolist()))])


def last_n_concat(cube, n):
    """cube (num_samples, depth, D)  ->  L2-normed (num_samples, D) MEAN of last n."""
    depth = cube.shape[1]
    n = min(n, depth)
    return l2(cube[:, depth - n: depth, :].mean(axis=1).astype(np.float32))


def build_submission(qf, gf, out_csv, variant, k1=20, k2=6, lam=0.3, top_k=100):
    if variant == 'pure':
        dist = 1.0 - qf @ gf.T
    else:
        dist = re_ranking(qf @ gf.T, qf @ qf.T, gf @ gf.T, k1=k1, k2=k2, lambda_value=lam)
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
    ap.add_argument('--val_dir', required=True,
                    help='dir with qf_all.npy / gf_all.npy / meta.json')
    ap.add_argument('--test_dir', required=True)
    ap.add_argument('--tag', required=True, help='e.g. Hplus_cls_concat')
    ap.add_argument('--out_dir', default='submissions')
    ap.add_argument('--ns', type=int, nargs='+',
                    default=[1, 2, 3, 4, 6, 8, 12, 16, 24])
    ap.add_argument('--k1', type=int, default=20)
    ap.add_argument('--k2', type=int, default=6)
    ap.add_argument('--lam', type=float, default=0.3)
    args = ap.parse_args()

    qf_cube = np.load(os.path.join(args.val_dir, 'qf_all.npy'))
    gf_cube = np.load(os.path.join(args.val_dir, 'gf_all.npy'))
    with open(os.path.join(args.val_dir, 'meta.json')) as f: mv = json.load(f)
    q_pids = np.asarray(mv['query_pids'], dtype=np.int64)
    g_pids = np.asarray(mv['gallery_pids'], dtype=np.int64)
    q_cam = np.asarray(mv['query_camids'], dtype=np.int64)
    g_cam = np.asarray(mv['gallery_camids'], dtype=np.int64)

    # Try to load query/gallery classes if available
    q_classes = None
    g_classes = None
    try:
        if 'query_classes' in mv:
            q_classes = np.asarray(mv['query_classes'], dtype=str)
        if 'gallery_classes' in mv:
            g_classes = np.asarray(mv['gallery_classes'], dtype=str)
    except:
        pass

    # If classes still not loaded, try loading from CSV
    if q_classes is None or g_classes is None:
        try:
            import csv as csv_module
            query_csv = os.path.join(os.path.dirname(args.val_dir), '..', 'Dataset', 'UrbanUAM_Merged', 'val_query_classes.csv')
            gallery_csv = os.path.join(os.path.dirname(args.val_dir), '..', 'Dataset', 'UrbanUAM_Merged', 'val_test_classes.csv')

            if os.path.exists(query_csv):
                q_classes = {}
                with open(query_csv, 'r', newline='') as f:
                    reader = csv_module.DictReader(f)
                    for i, row in enumerate(reader):
                        cls = row.get('Class', '').strip().lower()
                        if cls == 'trafficsignal': cls = 'trafficsign'
                        q_classes[i] = cls
                q_classes = np.array([q_classes.get(i, 'unknown') for i in range(len(q_pids))])

            if os.path.exists(gallery_csv):
                g_classes = {}
                with open(gallery_csv, 'r', newline='') as f:
                    reader = csv_module.DictReader(f)
                    for i, row in enumerate(reader):
                        cls = row.get('Class', '').strip().lower()
                        if cls == 'trafficsignal': cls = 'trafficsign'
                        g_classes[i] = cls
                g_classes = np.array([g_classes.get(i, 'unknown') for i in range(len(g_pids))])
        except Exception as e:
            print(f"  Warning: Could not load classes from CSV: {e}")
            pass

    depth = qf_cube.shape[1]
    ns = [n for n in args.ns if n <= depth]
    print(f'val cube: qf={qf_cube.shape} gf={gf_cube.shape}  depth={depth}')
    print(f'sweeping N in {ns}')
    print(f'Using CONCATENATION strategy')
    print()

    results = []
    for n in ns:
        qf_n = last_n_concat(qf_cube, n)
        gf_n = last_n_concat(gf_cube, n)

        # Use per-class grid search
        m_pc, pc_best = perclass_grid(qf_n, gf_n, q_pids, g_pids, q_classes, g_classes,
                                      q_cam, g_cam, K1=[10, 15, 20, 30, 50],
                                      K2=[2, 3, 5, 6], LAM=[0.0, 0.1, 0.2, 0.3])

        results.append({'N': n, 'perclass': m_pc,
                        'perclass_params': pc_best, 'n_dim': qf_n.shape[1]})
        print(f'  N={n:2d}  dim={qf_n.shape[1]:5d}  perclass_mAP={m_pc:.4f}')

    best = max(results, key=lambda r: r['perclass'])
    print()
    print(f'==> best N={best["N"]}  dim={best["n_dim"]}  perclass_mAP={best["perclass"]:.4f}')

    qf_test = np.load(os.path.join(args.test_dir, 'qf_all.npy'))
    gf_test = np.load(os.path.join(args.test_dir, 'gf_all.npy'))
    qf_sub = last_n_concat(qf_test, best['N'])
    gf_sub = last_n_concat(gf_test, best['N'])

    out_csv = os.path.join(args.out_dir,
                           f'submission_{args.tag}_clsN{best["N"]}_perclass.csv')
    write_submission_perclass(qf_sub, gf_sub, q_classes, g_classes, best['perclass_params'], out_csv)
    print(f'[submission] wrote {out_csv}')

    log = os.path.join(args.out_dir, f'eval_cls_n_{args.tag}.json')
    with open(log, 'w') as f:
        json.dump({'tag': args.tag, 'strategy': 'concatenation_perclass', 'sweep': results, 'best': best,
                   'submission': out_csv}, f, indent=2)
    print(f'[log] saved to {log}')


if __name__ == '__main__':
    main()
