"""Two new post-processing tricks evaluated on Hplus val:

1) Gallery feature calibration — z-score query/gallery features using per-class
   gallery statistics. Aligns c004 (novel test camera) toward the c001-c003
   gallery distribution. Fallback to global stats if a class has <10 gallery
   items. Final L2-normalize preserved.

2) Adaptive k-reciprocal depth — scale k1 (and k2 ≈ k1/3) per query based on
   top-1 cosine confidence:
     ≥0.85 → k1 = base_k1 - 10  (trust initial rank)
     ≥0.70 → k1 = base_k1
     <0.70 → k1 = base_k1 + 15  (aggressive recovery)
   Trafficsign uses lower thresholds (0.75 / 0.60) reflecting its overall
   lower confidence.

We try each variant alone, in combination, and on top of Hplus's existing
per-class re-ranking params (read from submissions/eval_Hplus.json).
Reports val per-class mAP and writes a test submission for the best variant.
"""
import argparse, csv, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from urban_elements_reid_challenge.utils.re_ranking import re_ranking

DATA = os.path.join(REPO, 'Dataset/UrbanUAM_Merged')


def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
def norm_cls(c): c = c.strip().lower(); return 'trafficsign' if c == 'trafficsignal' else c
def read_classes(p):
    d = {}
    with open(p, newline='') as f:
        rd = csv.reader(f); hdr = next(rd); ni = hdr.index('imageName'); ci = hdr.index('Class')
        for r in rd: d[r[ni]] = norm_cls(r[ci])
    return d
def compute_ap(order, q_pid, g_pids, q_cam, g_cams):
    keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
    order = order[keep]
    matches = (g_pids[order] == q_pid).astype(np.float32)
    if matches.sum() == 0: return None
    cum = np.cumsum(matches); prec = cum / (np.arange(len(matches)) + 1)
    return float((prec * matches).sum() / matches.sum())


# ---- Method 1: Gallery Feature Calibration ----
def calibrate(qf, gf, q_cls, g_cls):
    """Per-class z-score using GALLERY statistics (with global fallback when n<10).
    Returns (qf_cal, gf_cal) — both transformed and re-L2-normed.
    """
    classes = np.unique(np.concatenate([q_cls, g_cls]))
    qf_c = qf.copy().astype(np.float32)
    gf_c = gf.copy().astype(np.float32)

    # Global fallback stats from full gallery
    g_mu = gf.mean(axis=0, keepdims=True)
    g_sd = gf.std(axis=0, keepdims=True)

    for cls in classes:
        gi = np.where(g_cls == cls)[0]
        if len(gi) >= 10:
            mu = gf[gi].mean(axis=0, keepdims=True)
            sd = gf[gi].std(axis=0, keepdims=True)
        else:
            mu, sd = g_mu, g_sd
        denom = (sd + 1e-6)
        # apply to queries of this class
        qi = np.where(q_cls == cls)[0]
        if len(qi) > 0:
            qf_c[qi] = (qf[qi] - mu) / denom
        # apply to gallery of this class
        gf_c[gi] = (gf[gi] - mu) / denom
    return l2(qf_c), l2(gf_c)


# ---- Method 2: Adaptive k1 ----
def adaptive_k1_for_query(top1_sim, base_k1, hi_thr, lo_thr):
    if top1_sim >= hi_thr: return max(base_k1 - 10, 4)
    if top1_sim >= lo_thr: return base_k1
    return base_k1 + 15


def class_thresholds(cls):
    if cls == 'trafficsign':
        return 0.75, 0.60
    return 0.85, 0.70


def perclass_rerank(qf, gf, q_pids, g_pids, q_cls, g_cls, q_cam, g_cam,
                    pc_params, adaptive=False):
    """Run per-class rerank using the given pc_params dict {cls: (k1, k2, lam)}.
    If adaptive=True, scale k1/k2 per query inside each class by top-1 sim
    using the class's threshold pair.
    Returns weighted overall val mAP and per-class mAPs.
    """
    classes = ['container', 'crosswalk', 'rubbishbins', 'trafficsign']
    per_cls = {}
    tot_n, tot_d = 0.0, 0
    for cls in classes:
        qi = np.where(q_cls == cls)[0]; gi = np.where(g_cls == cls)[0]
        if len(qi) == 0 or len(gi) == 0: continue
        qf_c, gf_c = qf[qi], gf[gi]
        qp_c, gp_c = q_pids[qi], g_pids[gi]
        qc_c, gc_c = q_cam[qi], g_cam[gi]
        base_k1, base_k2, lam = pc_params[cls]

        if not adaptive:
            d = re_ranking(qf_c @ gf_c.T, qf_c @ qf_c.T, gf_c @ gf_c.T,
                           k1=base_k1, k2=base_k2, lambda_value=lam)
        else:
            # group queries by adaptive k1 bucket
            sims = qf_c @ gf_c.T
            top1 = sims.max(axis=1)
            hi_thr, lo_thr = class_thresholds(cls)
            k1s = np.array([adaptive_k1_for_query(s, base_k1, hi_thr, lo_thr) for s in top1])
            d = np.full((len(qi), len(gi)), np.inf, dtype=np.float32)
            unique_k1s = np.unique(k1s)
            for kk1 in unique_k1s:
                kk1 = int(kk1)
                kk2 = max(2, kk1 // 3)
                qmask = (k1s == kk1)
                qf_bucket = qf_c[qmask]
                if len(qf_bucket) == 0: continue
                # Need q-q sims for the bucket too (re_ranking expects symmetric)
                # but the bucket is a subset of queries. Use sims to gallery + sims among bucket.
                d_bucket = re_ranking(
                    qf_bucket @ gf_c.T,
                    qf_bucket @ qf_bucket.T,
                    gf_c @ gf_c.T,
                    k1=kk1, k2=kk2, lambda_value=lam)
                d[qmask] = d_bucket

        aps = []
        for i in range(len(qi)):
            ap = compute_ap(np.argsort(d[i]), qp_c[i], gp_c, qc_c[i], gc_c)
            if ap is not None: aps.append(ap)
        m = float(np.mean(aps)) if aps else 0.0
        per_cls[cls] = {'mAP': m, 'n_query': int(len(qi))}
        tot_n += m * len(qi); tot_d += len(qi)
    return tot_n / max(tot_d, 1), per_cls


def write_submission_perclass(qf, gf, q_cls, g_cls, pc_params, out_csv,
                              adaptive=False, top_k=100):
    n_q = qf.shape[0]
    all_ranked = np.zeros((n_q, top_k), dtype=np.int64)
    for cls in np.unique(q_cls):
        qi = np.where(q_cls == cls)[0]; gi_same = np.where(g_cls == cls)[0]
        gi_other = np.where(g_cls != cls)[0]
        base_k1, base_k2, lam = pc_params.get(cls, (50, 15, 0.3))
        qf_c, gf_c = qf[qi], gf[gi_same]
        if not adaptive:
            d = re_ranking(qf_c @ gf_c.T, qf_c @ qf_c.T, gf_c @ gf_c.T,
                           k1=base_k1, k2=base_k2, lambda_value=lam)
        else:
            sims = qf_c @ gf_c.T
            top1 = sims.max(axis=1)
            hi_thr, lo_thr = class_thresholds(cls)
            k1s = np.array([adaptive_k1_for_query(s, base_k1, hi_thr, lo_thr) for s in top1])
            d = np.full((len(qi), len(gi_same)), np.inf, dtype=np.float32)
            for kk1 in np.unique(k1s):
                kk1 = int(kk1); kk2 = max(2, kk1 // 3)
                qmask = (k1s == kk1)
                qfb = qf_c[qmask]
                if len(qfb) == 0: continue
                db = re_ranking(qfb @ gf_c.T, qfb @ qfb.T, gf_c @ gf_c.T,
                                k1=kk1, k2=kk2, lambda_value=lam)
                d[qmask] = db
        d_other = 1.0 - qf_c @ gf[gi_other].T if len(gi_other) > 0 else None
        for li, gq in enumerate(qi):
            order = np.argsort(d[li]); ranked = gi_same[order]
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
    ap.add_argument('--val_dir',  default='feat_cache/Hplus_val')
    ap.add_argument('--test_dir', default='feat_cache/Hplus_test')
    ap.add_argument('--eval_json', default='submissions/eval_Hplus.json',
                    help='JSON with per-class params; we use those as base_k1/k2/lam')
    ap.add_argument('--tag', default='Hplus_calibAdaptive')
    args = ap.parse_args()

    # Load val features + meta
    qf_v = l2(np.load(os.path.join(args.val_dir, 'qf.npy'))).astype(np.float32)
    gf_v = l2(np.load(os.path.join(args.val_dir, 'gf.npy'))).astype(np.float32)
    with open(os.path.join(args.val_dir, 'meta.json')) as f: mv = json.load(f)
    q_pids_v = np.asarray(mv['query_pids']); g_pids_v = np.asarray(mv['gallery_pids'])
    q_cam_v  = np.asarray(mv['query_camids']); g_cam_v  = np.asarray(mv['gallery_camids'])
    nq_v = read_classes(os.path.join(DATA, 'val_query_classes.csv'))
    ng_v = read_classes(os.path.join(DATA, 'val_test_classes.csv'))
    q_cls_v = np.array([nq_v[os.path.basename(p)] for p in mv['query_paths']])
    g_cls_v = np.array([ng_v[os.path.basename(p)] for p in mv['gallery_paths']])

    # Per-class params from Hplus eval
    with open(args.eval_json) as f: ev = json.load(f)
    pc = {c: tuple(ev['perclass_params'][c]['params']) for c in ev['perclass_params']}
    print(f'baseline per-class params: {pc}\n')

    results = {}

    # A: baseline (no calib, no adaptive) — sanity
    m_A, pc_A = perclass_rerank(qf_v, gf_v, q_pids_v, g_pids_v,
                                 q_cls_v, g_cls_v, q_cam_v, g_cam_v, pc, adaptive=False)
    results['A_baseline'] = {'overall': m_A, 'per_class': pc_A}
    print(f'A) baseline (Hplus per-class)                              val mAP = {m_A:.4f}')

    # B: calibration only
    qf_cal, gf_cal = calibrate(qf_v, gf_v, q_cls_v, g_cls_v)
    m_B, pc_B = perclass_rerank(qf_cal, gf_cal, q_pids_v, g_pids_v,
                                 q_cls_v, g_cls_v, q_cam_v, g_cam_v, pc, adaptive=False)
    results['B_calib'] = {'overall': m_B, 'per_class': pc_B}
    print(f'B) calibration + Hplus per-class                           val mAP = {m_B:.4f}')

    # C: adaptive only
    m_C, pc_C = perclass_rerank(qf_v, gf_v, q_pids_v, g_pids_v,
                                 q_cls_v, g_cls_v, q_cam_v, g_cam_v, pc, adaptive=True)
    results['C_adaptive'] = {'overall': m_C, 'per_class': pc_C}
    print(f'C) adaptive k1 + Hplus per-class                           val mAP = {m_C:.4f}')

    # D: calibration + adaptive
    m_D, pc_D = perclass_rerank(qf_cal, gf_cal, q_pids_v, g_pids_v,
                                 q_cls_v, g_cls_v, q_cam_v, g_cam_v, pc, adaptive=True)
    results['D_calib_adaptive'] = {'overall': m_D, 'per_class': pc_D}
    print(f'D) calibration + adaptive k1 + Hplus per-class             val mAP = {m_D:.4f}')

    print('\nper-class breakdown:')
    classes = ['container','crosswalk','rubbishbins','trafficsign']
    print('  ' + 'class      '.ljust(13) + ' | '.join([k.ljust(15) for k in ['A_base','B_calib','C_adapt','D_both']]))
    for cls in classes:
        cells = []
        for k in ['A_baseline','B_calib','C_adaptive','D_calib_adaptive']:
            v = results[k]['per_class'].get(cls, {}).get('mAP', 0.0)
            cells.append(f'{v:.4f}'.ljust(15))
        print(f'  {cls:12s} ' + ' | '.join(cells))

    # Pick best variant
    best_name = max(results, key=lambda k: results[k]['overall'])
    print(f'\n==> best: {best_name}  val mAP = {results[best_name]["overall"]:.4f}')

    # Build test submission for best variant
    qf_t = l2(np.load(os.path.join(args.test_dir, 'qf.npy'))).astype(np.float32)
    gf_t = l2(np.load(os.path.join(args.test_dir, 'gf.npy'))).astype(np.float32)
    with open(os.path.join(args.test_dir, 'meta.json')) as f: mt = json.load(f)
    nq_t = read_classes(os.path.join(DATA, 'query_classes.csv'))
    ng_t = read_classes(os.path.join(DATA, 'test_classes.csv'))
    q_cls_t = np.array([nq_t[os.path.basename(p)] for p in mt['query_paths']])
    g_cls_t = np.array([ng_t[os.path.basename(p)] for p in mt['gallery_paths']])

    # Write submissions for ALL 4 variants — calibration may help on test
    # (c004 novel camera) even though it hurts on val (c104 seen).
    qf_t_cal, gf_t_cal = calibrate(qf_t, gf_t, q_cls_t, g_cls_t)
    variants = {
        'A_baseline':        (qf_t,     gf_t,     False),
        'B_calib':           (qf_t_cal, gf_t_cal, False),
        'C_adaptive':        (qf_t,     gf_t,     True),
        'D_calib_adaptive':  (qf_t_cal, gf_t_cal, True),
    }
    out_csv = None
    for name, (qf_u, gf_u, use_adaptive) in variants.items():
        path = f'submissions/submission_{args.tag}_{name}.csv'
        write_submission_perclass(qf_u, gf_u, q_cls_t, g_cls_t, pc, path,
                                  adaptive=use_adaptive)
        if name == best_name: out_csv = path
        print(f'  [{name}] -> {path}')
    print(f'\n==> best on val: {out_csv}')

    log = f'submissions/eval_{args.tag}.json'
    with open(log, 'w') as f:
        json.dump({'tag': args.tag, 'baseline_per_class_params': pc,
                   'results': {k: {'overall': v['overall']} for k, v in results.items()},
                   'detail': results, 'best_variant': best_name,
                   'submission': out_csv}, f, indent=2)
    print(f'[log] {log}')


if __name__ == '__main__':
    main()
