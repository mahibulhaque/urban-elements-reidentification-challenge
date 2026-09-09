"""Test-time 'last-N CLS tokens' experiment for ViT-based ReID.

Given a cached per-block CLS cube (qf_all, gf_all of shape (N, depth, D)
produced by extract_features_allblocks.py), sweep N in {1,2,3,4,6,8,12,16,24},
form the mean of the last-N CLS tokens, L2-normalize, and report val mAP with
(a) pure cosine and (b) k-reciprocal re-ranking. Pick the best-N (by its
better variant on val) and write a test submission.
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


def pure_map(qf, gf, q_pids, g_pids, q_cam, g_cam):
    return map_from_dist(1.0 - qf @ gf.T, q_pids, g_pids, q_cam, g_cam)


def rerank_map(qf, gf, q_pids, g_pids, q_cam, g_cam, k1=20, k2=6, lam=0.3):
    dist = re_ranking(qf @ gf.T, qf @ qf.T, gf @ gf.T, k1=k1, k2=k2, lambda_value=lam)
    return map_from_dist(dist, q_pids, g_pids, q_cam, g_cam)


def last_n_mean(cube, n):
    """cube (num_samples, depth, D) -> L2-normed (num_samples, N*D) concat of last n.

    Per-block L2-normalize FIRST, then concatenate, then final L2-norm.
    This makes cos(concat_a, concat_b) = mean over blocks of cos(layer_i_a, layer_i_b),
    so each block contributes equally to similarity. Without per-block norm
    the concat is dominated by whichever block has the largest activations
    (deeper layers usually win, making concat ≈ last-block-only).
    """
    depth = cube.shape[1]
    n = min(n, depth)
    selected = cube[:, depth - n: depth, :].astype(np.float32)  # (B, n, D)
    # Per-block L2-norm along feature dim
    norms = np.linalg.norm(selected, axis=2, keepdims=True).clip(min=1e-12)
    selected = selected / norms
    concat_feat = selected.reshape(selected.shape[0], -1)  # (B, n*D)
    return l2(concat_feat)


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
    ap.add_argument('--tag', required=True, help='e.g. run3_ep30')
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

    depth = qf_cube.shape[1]
    ns = [n for n in args.ns if n <= depth]
    print(f'val cube: qf={qf_cube.shape} gf={gf_cube.shape}  depth={depth}')
    print(f'sweeping N in {ns}')
    print()

    results = []
    for n in ns:
        qf_n = last_n_mean(qf_cube, n)
        gf_n = last_n_mean(gf_cube, n)
        m_pure = pure_map(qf_n, gf_n, q_pids, g_pids, q_cam, g_cam)
        m_rerk = rerank_map(qf_n, gf_n, q_pids, g_pids, q_cam, g_cam,
                            k1=args.k1, k2=args.k2, lam=args.lam)
        best_variant = 'rerank' if m_rerk > m_pure else 'pure'
        best_m = max(m_pure, m_rerk)
        results.append({'N': n, 'pure': m_pure, 'rerank': m_rerk,
                        'best_variant': best_variant, 'best_mAP': best_m})
        print(f'  N={n:2d}  pure={m_pure:.4f}  rerank={m_rerk:.4f}  '
              f'(better={best_variant})')

    best = max(results, key=lambda r: r['best_mAP'])
    print()
    print(f'==> best N={best["N"]}  variant={best["best_variant"]}  '
          f'mAP={best["best_mAP"]:.4f}')

    qf_test = np.load(os.path.join(args.test_dir, 'qf_all.npy'))
    gf_test = np.load(os.path.join(args.test_dir, 'gf_all.npy'))
    qf_sub = last_n_mean(qf_test, best['N'])
    gf_sub = last_n_mean(gf_test, best['N'])

    out_csv = os.path.join(args.out_dir,
                           f'submission_{args.tag}_clsN{best["N"]}_{best["best_variant"]}.csv')
    build_submission(qf_sub, gf_sub, out_csv, best['best_variant'],
                     k1=args.k1, k2=args.k2, lam=args.lam)
    print(f'[submission] wrote {out_csv}')

    log = os.path.join(args.out_dir, f'eval_cls_n_{args.tag}.json')
    with open(log, 'w') as f:
        json.dump({'tag': args.tag, 'sweep': results, 'best': best,
                   'submission': out_csv}, f, indent=2)
    print(f'[log] saved to {log}')


if __name__ == '__main__':
    main()
