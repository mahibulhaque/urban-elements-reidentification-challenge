"""Build the TSpec hybrid submission.

For trafficsign test queries, use the TSpec specialist's per-class rerank
output (computed from cached TSpec features). For the other 3 classes, copy
rows from the Hplus per-class submission.

Also reports the expected val mAP using each model's per-class number.
"""
import argparse, csv, json, os, numpy as np

REPO = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026'
DATA = os.path.join(REPO, 'Dataset/UrbanUAM_Merged')


def norm(c): c=c.strip().lower(); return 'trafficsign' if c=='trafficsignal' else c
def read_classes(p):
    d={}
    with open(p, newline='') as f:
        rd=csv.reader(f); hdr=next(rd); ni=hdr.index('imageName'); ci=hdr.index('Class')
        for r in rd: d[r[ni]] = norm(r[ci])
    return d
def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
def read_sub(p):
    rows={}
    with open(p, newline='') as f:
        rd=csv.reader(f); next(rd)
        for r in rd: rows[r[0]] = r[1]
    return rows


def per_class_rerank_top100(qf_v, gf_v, q_pids_v, g_pids_v, q_cam_v, g_cam_v,
                            qf_t, gf_t, K1, K2, LAMS, n_class_query, top_k=100):
    """Run per-class grid search on val (single class), apply best params to test.

    Returns: (val_mAP, best_params_tuple, test_top100_array (n_q, 100, 0-based))
    """
    import sys; sys.path.insert(0, REPO)
    from utils.re_ranking import re_ranking

    def compute_ap(order, q_pid, g_pids, q_cam, g_cams):
        keep = ~((g_pids[order]==q_pid) & (g_cams[order]==q_cam))
        order = order[keep]
        m = (g_pids[order]==q_pid).astype(np.float32)
        if m.sum()==0: return None
        return float((np.cumsum(m)/(np.arange(len(m))+1) * m).sum() / m.sum())

    best_m, best_p = -1.0, None
    for k1 in K1:
        for k2 in K2:
            if k2>k1: continue
            for lam in LAMS:
                d = re_ranking(qf_v@gf_v.T, qf_v@qf_v.T, gf_v@gf_v.T, k1=k1, k2=k2, lambda_value=lam)
                aps=[]
                for i in range(len(qf_v)):
                    ap=compute_ap(np.argsort(d[i]), q_pids_v[i], g_pids_v, q_cam_v[i], g_cam_v)
                    if ap is not None: aps.append(ap)
                m = float(np.mean(aps)) if aps else 0.0
                if m>best_m: best_m, best_p = m, (k1, k2, lam)

    k1,k2,lam = best_p
    d_t = re_ranking(qf_t@gf_t.T, qf_t@qf_t.T, gf_t@gf_t.T, k1=k1, k2=k2, lambda_value=lam)
    test_top = np.argsort(d_t, axis=1)[:, :top_k]
    return best_m, best_p, test_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tspec_val',  required=True, help='feat_cache/TSpec_val (trafficsign-only)')
    ap.add_argument('--tspec_test', required=True, help='feat_cache/TSpec_test (trafficsign-only)')
    ap.add_argument('--hplus_sub',  default='submissions/submission_Hplus_perclass.csv',
                    help='Hplus per-class TEST submission for non-trafficsign classes')
    ap.add_argument('--out',        required=True)
    args = ap.parse_args()

    # 1. Compute TSpec trafficsign val mAP + test top-100 via per-class rerank
    qf_v = l2(np.load(os.path.join(args.tspec_val,  'qf.npy'))).astype(np.float32)
    gf_v = l2(np.load(os.path.join(args.tspec_val,  'gf.npy'))).astype(np.float32)
    qf_t = l2(np.load(os.path.join(args.tspec_test, 'qf.npy'))).astype(np.float32)
    gf_t = l2(np.load(os.path.join(args.tspec_test, 'gf.npy'))).astype(np.float32)
    with open(os.path.join(args.tspec_val,  'meta.json')) as f: mv = json.load(f)
    with open(os.path.join(args.tspec_test, 'meta.json')) as f: mt = json.load(f)

    K1=[10,15,20,30,50]; K2=[2,3,5,6]; LAMS=[0.0,0.1,0.2]
    val_m, params, test_top = per_class_rerank_top100(
        qf_v, gf_v,
        np.asarray(mv['query_pids']), np.asarray(mv['gallery_pids']),
        np.asarray(mv['query_camids']), np.asarray(mv['gallery_camids']),
        qf_t, gf_t, K1, K2, LAMS, n_class_query=int(qf_v.shape[0]))

    print(f'TSpec trafficsign val mAP = {val_m:.4f}  (k1={params[0]}, k2={params[1]}, lam={params[2]})')

    # 2. Build trafficsign rows (1-based indices into TSpec gallery)
    g_names_t = [os.path.basename(p) for p in mt['gallery_paths']]
    q_names_t = [os.path.basename(p) for p in mt['query_paths']]
    # IMPORTANT: TSpec gallery contains only trafficsign items, but the
    # submission requires global gallery indices into the full test gallery.
    # Map TSpec gallery names back to the FULL test gallery 1..2844 ordering.
    full_g_csv = os.path.join(DATA, 'test_classes.csv')
    full_g_names = []
    with open(full_g_csv, newline='') as f:
        rd = csv.reader(f); next(rd)
        for r in rd: full_g_names.append(r[1])
    name_to_global_idx = {n: i + 1 for i, n in enumerate(full_g_names)}  # 1-based

    tspec_rows = {}
    for qi, qn in enumerate(q_names_t):
        tspec_local = test_top[qi]  # 0-based indices into TSpec gallery
        global_indices = [name_to_global_idx[g_names_t[li]] for li in tspec_local]
        tspec_rows[qn] = ' '.join(map(str, global_indices))
    print(f'TSpec rows: {len(tspec_rows)} (should match test trafficsign count)')

    # 3. Read Hplus full-test submission for non-trafficsign rows
    hplus = read_sub(args.hplus_sub)
    q_cls_test = read_classes(os.path.join(DATA, 'query_classes.csv'))

    # 4. Build hybrid: trafficsign -> TSpec, others -> Hplus
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    common = sorted(set(hplus.keys()), key=lambda s: int(s.split('.')[0]))
    counts = {'tspec': 0, 'hplus': 0}
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for name in common:
            cls = q_cls_test.get(name)
            if cls == 'trafficsign' and name in tspec_rows:
                w.writerow([name, tspec_rows[name]]); counts['tspec'] += 1
            else:
                w.writerow([name, hplus[name]]); counts['hplus'] += 1
    print(f'wrote {args.out}')
    print(f'  routing counts: TSpec(trafficsign)={counts["tspec"]}  Hplus(other)={counts["hplus"]}')

    log = args.out.replace('.csv', '.json')
    with open(log, 'w') as f:
        json.dump({'tspec_val_trafficsign_mAP': val_m, 'tspec_params': params,
                   'tspec_count': counts['tspec'], 'hplus_count': counts['hplus']}, f, indent=2)


if __name__ == '__main__':
    main()
