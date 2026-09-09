"""Evaluate concat-of-last-N CLS tokens with **per-class re-ranking**.

Each block's CLS token is L2-normalized FIRST, then concatenated, then
final L2-normalized. Per-class grid-searched k-reciprocal rerank is run
within each class on val to pick (k1, k2, lam); same params are applied to
test → submission CSV.
"""
import argparse, csv, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from urban_elements_reid_challenge.utils.re_ranking import re_ranking

DATA = os.path.join(REPO, 'Dataset/UrbanUAM_Merged')


def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def norm_cls(c): c = c.strip().lower(); return 'trafficsign' if c == 'trafficsignal' else c


def read_cls(p):
    d = {}
    with open(p, newline='') as f:
        rd = csv.reader(f); hdr = next(rd); ni = hdr.index('imageName'); ci = hdr.index('Class')
        for r in rd: d[r[ni]] = norm_cls(r[ci])
    return d


def last_n_concat(cube, n):
    depth = cube.shape[1]; n = min(n, depth)
    sel = cube[:, depth - n: depth, :].astype(np.float32)
    norms = np.linalg.norm(sel, axis=2, keepdims=True).clip(min=1e-12)
    sel = sel / norms
    return l2(sel.reshape(sel.shape[0], -1))


def compute_ap(order, q_pid, g_pids, q_cam, g_cams):
    keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
    order = order[keep]
    matches = (g_pids[order] == q_pid).astype(np.float32)
    if matches.sum() == 0: return None
    cum = np.cumsum(matches); prec = cum / (np.arange(len(matches)) + 1)
    return float((prec * matches).sum() / matches.sum())


def perclass_grid(qf, gf, q_pids, g_pids, q_cls, g_cls, q_cam, g_cam, K1, K2, LAMS):
    classes = ['container', 'crosswalk', 'rubbishbins', 'trafficsign']
    best = {}; tot_n, tot_d = 0.0, 0
    for cls in classes:
        qi = np.where(q_cls == cls)[0]; gi = np.where(g_cls == cls)[0]
        qf_c, gf_c = qf[qi], gf[gi]
        qp_c, gp_c = q_pids[qi], g_pids[gi]
        qc_c, gc_c = q_cam[qi], g_cam[gi]
        best_m, best_p = -1.0, None
        for k1 in K1:
            for k2 in K2:
                if k2 > k1: continue
                for lam in LAMS:
                    d = re_ranking(qf_c @ gf_c.T, qf_c @ qf_c.T, gf_c @ gf_c.T,
                                   k1=k1, k2=k2, lambda_value=lam)
                    aps = []
                    for i in range(len(qi)):
                        ap = compute_ap(np.argsort(d[i]), qp_c[i], gp_c, qc_c[i], gc_c)
                        if ap is not None: aps.append(ap)
                    m = float(np.mean(aps)) if aps else 0.0
                    if m > best_m: best_m, best_p = m, (int(k1), int(k2), float(lam))
        best[cls] = {'mAP': best_m, 'params': best_p, 'n_query': int(len(qi))}
        tot_n += best_m * len(qi); tot_d += len(qi)
    overall = tot_n / max(tot_d, 1)
    return overall, best


def write_perclass_submission(qf_t, gf_t, q_cls_t, g_cls_t, best_params, out_csv, top_k=100):
    n_q = qf_t.shape[0]
    all_ranked = np.zeros((n_q, top_k), dtype=np.int64)
    for cls in np.unique(q_cls_t):
        qi = np.where(q_cls_t == cls)[0]
        gi_same = np.where(g_cls_t == cls)[0]
        gi_other = np.where(g_cls_t != cls)[0]
        if cls in best_params and best_params[cls]['params'] is not None:
            k1, k2, lam = best_params[cls]['params']
        else:
            k1, k2, lam = 50, 15, 0.3
        qf_q, gf_c = qf_t[qi], gf_t[gi_same]
        d_same = re_ranking(qf_q @ gf_c.T, qf_q @ qf_q.T, gf_c @ gf_c.T,
                            k1=k1, k2=k2, lambda_value=lam)
        d_other = 1.0 - qf_q @ gf_t[gi_other].T if len(gi_other) > 0 else None
        for li, gq in enumerate(qi):
            order = np.argsort(d_same[li])
            ranked = gi_same[order]
            if len(ranked) >= top_k:
                all_ranked[gq] = ranked[:top_k]
            else:
                need = top_k - len(ranked)
                order_o = np.argsort(d_other[li])[:need]
                all_ranked[gq] = np.concatenate([ranked, gi_other[order_o]])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(n_q):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (all_ranked[i] + 1).tolist()))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_dir',  required=True)
    ap.add_argument('--test_dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out_dir', default='submissions')
    ap.add_argument('--ns', type=int, nargs='+', default=[1, 2, 3, 4, 6, 8])
    args = ap.parse_args()

    qf_v_cube = np.load(os.path.join(args.val_dir, 'qf_all.npy'))
    gf_v_cube = np.load(os.path.join(args.val_dir, 'gf_all.npy'))
    with open(os.path.join(args.val_dir, 'meta.json')) as f: mv = json.load(f)
    q_pids = np.asarray(mv['query_pids']); g_pids = np.asarray(mv['gallery_pids'])
    q_cam = np.asarray(mv['query_camids']); g_cam = np.asarray(mv['gallery_camids'])
    nq_v = read_cls(os.path.join(DATA, 'val_query_classes.csv'))
    ng_v = read_cls(os.path.join(DATA, 'val_test_classes.csv'))
    q_cls_v = np.array([nq_v[os.path.basename(p)] for p in mv['query_paths']])
    g_cls_v = np.array([ng_v[os.path.basename(p)] for p in mv['gallery_paths']])

    depth = qf_v_cube.shape[1]
    ns = [n for n in args.ns if n <= depth]
    print(f'val cube: qf={qf_v_cube.shape} gf={gf_v_cube.shape}  depth={depth}')
    print(f'sweeping N in {ns}\n')

    K1=[10,15,20,30,50]; K2=[2,3,5,6]; LAMS=[0.0,0.1,0.2]
    best_overall = (-1.0, None, None)
    sweep = []
    for n in ns:
        qf_n = last_n_concat(qf_v_cube, n)
        gf_n = last_n_concat(gf_v_cube, n)
        m, per_cls = perclass_grid(qf_n, gf_n, q_pids, g_pids, q_cls_v, g_cls_v,
                                    q_cam, g_cam, K1, K2, LAMS)
        sweep.append({'N': n, 'perclass_mAP': m, 'per_class': per_cls})
        print(f'  N={n}  perclass val mAP = {m:.4f}')
        for cls, info in per_cls.items():
            p = info['params']
            print(f'      {cls:12s}  mAP={info["mAP"]:.4f}  (k1={p[0]}, k2={p[1]}, lam={p[2]})')
        if m > best_overall[0]: best_overall = (m, n, per_cls)

    best_m, best_n, best_pc = best_overall
    print(f'\n==> best N={best_n}  perclass val mAP = {best_m:.4f}')

    # Test submission with best N + best per-class params
    qf_t_cube = np.load(os.path.join(args.test_dir, 'qf_all.npy'))
    gf_t_cube = np.load(os.path.join(args.test_dir, 'gf_all.npy'))
    with open(os.path.join(args.test_dir, 'meta.json')) as f: mt = json.load(f)
    nq_t = read_cls(os.path.join(DATA, 'query_classes.csv'))
    ng_t = read_cls(os.path.join(DATA, 'test_classes.csv'))
    q_cls_t = np.array([nq_t[os.path.basename(p)] for p in mt['query_paths']])
    g_cls_t = np.array([ng_t[os.path.basename(p)] for p in mt['gallery_paths']])
    qf_t = last_n_concat(qf_t_cube, best_n)
    gf_t = last_n_concat(gf_t_cube, best_n)
    out = os.path.join(args.out_dir, f'submission_{args.tag}_clsConcatN{best_n}_perclass.csv')
    write_perclass_submission(qf_t, gf_t, q_cls_t, g_cls_t, best_pc, out)
    log = os.path.join(args.out_dir, f'eval_clsConcat_perclass_{args.tag}.json')
    with open(log, 'w') as f:
        json.dump({'tag': args.tag, 'best_N': best_n,
                   'best_perclass_val_mAP': best_m,
                   'best_per_class': best_pc, 'sweep': sweep,
                   'submission': out}, f, indent=2)
    print(f'\n[submission] {out}')
    print(f'[log] {log}')


if __name__ == '__main__':
    main()
