"""SAM 3 top-K re-ranker on top of a Hplus per-class submission.

For each test query:
  1. Take the top-K candidates from the Hplus per-class CSV (already class-filtered).
  2. Compute SAM 3 cosine distance between query and each candidate (using
     cached SAM 3 features from extract_sam3.py).
  3. Final distance = α · d_Hplus_rank + (1-α) · d_SAM3_cos.
  4. Re-rank top-K, keep order outside top-K, write submission.

Same logic on val (with labels) for selecting α.
"""
import argparse, csv, json, os
import numpy as np

REPO = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026'
DATA = os.path.join(REPO, 'Dataset/UrbanUAM_Merged')


def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def load_sub(p):
    rows = {}
    with open(p, newline='') as f:
        rd = csv.reader(f); next(rd)
        for r in rd: rows[r[0]] = list(map(int, r[1].split()))
    return rows


def norm_cls(c): c = c.strip().lower(); return 'trafficsign' if c == 'trafficsignal' else c


def read_classes(p):
    d = {}
    with open(p, newline='') as f:
        rd = csv.reader(f); hdr = next(rd)
        ni = hdr.index('imageName'); ci = hdr.index('Class')
        for r in rd: d[r[ni]] = norm_cls(r[ci])
    return d


def compute_ap(order, q_pid, g_pids, q_cam, g_cams):
    keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
    order = order[keep]
    matches = (g_pids[order] == q_pid).astype(np.float32)
    if matches.sum() == 0: return None
    cum = np.cumsum(matches); prec = cum / (np.arange(len(matches)) + 1)
    return float((prec * matches).sum() / matches.sum())


def rerank_with_sam3(sub_csv, sam3_qf, sam3_gf, top_k, alpha,
                     q_basenames, g_basenames):
    """
    sub_csv: dict {query_basename: [g_idx_1based, ...]}  (top-100)
    sam3_*f: L2-normalized SAM 3 features
    top_k: rerank only within top_k (keep rest fixed)
    alpha: weight on Hplus rank ordering (0..1). 0 = SAM3 only.
    """
    new_rows = {}
    # idx map
    g_idx = {n: i for i, n in enumerate(g_basenames)}
    q_idx = {n: i for i, n in enumerate(q_basenames)}
    for name, top in sub_csv.items():
        # Hplus distance proxy: rank in [0, 100). Lower = better.
        h_dist = np.arange(len(top), dtype=np.float32) / max(len(top) - 1, 1)
        # SAM 3 cos dist for query vs each candidate
        qi = q_idx[name]
        cand_idx = np.array([t - 1 for t in top])  # 0-based gallery indices
        s_dist = (1.0 - sam3_qf[qi:qi+1] @ sam3_gf[cand_idx].T)[0].astype(np.float32)
        # Scale s_dist to [0,1] for fair combination
        s_min, s_max = float(s_dist.min()), float(s_dist.max())
        s_range = max(s_max - s_min, 1e-6)
        s_norm = (s_dist - s_min) / s_range
        # Combined: only rerank top_k
        combined = alpha * h_dist + (1.0 - alpha) * s_norm
        head_order = np.argsort(combined[:top_k])
        head = [top[i] for i in head_order]
        tail = top[top_k:]
        new_rows[name] = head + tail
    return new_rows


def write_sub(rows, out_csv, name_order):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['imageName', 'Corresponding Indexes'])
        for n in name_order:
            w.writerow([n, ' '.join(map(str, rows[n]))])


def val_rerank_mAP(sub_val, sam3_qf, sam3_gf, top_k, alpha, meta_v):
    """Reapply rerank to a 'val submission' and compute mAP."""
    q_pids = np.asarray(meta_v['query_pids']); g_pids = np.asarray(meta_v['gallery_pids'])
    q_cam = np.asarray(meta_v['query_camids']); g_cam = np.asarray(meta_v['gallery_camids'])
    q_names = [os.path.basename(p) for p in meta_v['query_paths']]
    g_names = [os.path.basename(p) for p in meta_v['gallery_paths']]
    new_rows = rerank_with_sam3(sub_val, sam3_qf, sam3_gf, top_k, alpha, q_names, g_names)
    aps = []
    for i, qn in enumerate(q_names):
        if qn not in new_rows: continue
        order = np.array([t - 1 for t in new_rows[qn]])
        ap = compute_ap(order, q_pids[i], g_pids, q_cam[i], g_cam)
        if ap is not None: aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hplus_val_sub',  required=True, help='Hplus per-class val submission CSV')
    ap.add_argument('--hplus_test_sub', required=True, help='Hplus per-class test submission CSV')
    ap.add_argument('--sam3_val_dir',   required=True, help='cached SAM 3 val feat dir')
    ap.add_argument('--sam3_test_dir',  required=True, help='cached SAM 3 test feat dir')
    ap.add_argument('--out',            required=True, help='output test submission CSV')
    ap.add_argument('--top_k',  type=int, default=50, help='re-rank only top-K (rest unchanged)')
    ap.add_argument('--alphas', nargs='+', type=float, default=[0.3, 0.5, 0.7])
    args = ap.parse_args()

    # ---- Val side: pick best alpha ----
    sam3_qf_v = l2(np.load(os.path.join(args.sam3_val_dir, 'qf.npy'))).astype(np.float32)
    sam3_gf_v = l2(np.load(os.path.join(args.sam3_val_dir, 'gf.npy'))).astype(np.float32)
    with open(os.path.join(args.sam3_val_dir, 'meta.json')) as f:
        meta_v = json.load(f)
    sub_v = load_sub(args.hplus_val_sub)

    print('val α-sweep (rerank top-K of Hplus with α·Hplus + (1-α)·SAM3):')
    best_a, best_m = None, -1.0
    for a in args.alphas:
        m = val_rerank_mAP(sub_v, sam3_qf_v, sam3_gf_v, args.top_k, a, meta_v)
        print(f'  α={a:.2f}  val mAP = {m:.4f}')
        if m > best_m: best_a, best_m = a, m
    print(f'\nbest α = {best_a:.2f}  val mAP = {best_m:.4f}')

    # ---- Test side: build the final submission with best α ----
    sam3_qf_t = l2(np.load(os.path.join(args.sam3_test_dir, 'qf.npy'))).astype(np.float32)
    sam3_gf_t = l2(np.load(os.path.join(args.sam3_test_dir, 'gf.npy'))).astype(np.float32)
    with open(os.path.join(args.sam3_test_dir, 'meta.json')) as f:
        meta_t = json.load(f)
    sub_t = load_sub(args.hplus_test_sub)
    q_names_t = [os.path.basename(p) for p in meta_t['query_paths']]
    g_names_t = [os.path.basename(p) for p in meta_t['gallery_paths']]
    new_t = rerank_with_sam3(sub_t, sam3_qf_t, sam3_gf_t, args.top_k, best_a,
                             q_names_t, g_names_t)
    write_sub(new_t, args.out, [n for n in q_names_t if n in new_t])
    log = args.out.replace('.csv', '.json')
    with open(log, 'w') as f:
        json.dump({'best_alpha': best_a, 'best_val_mAP': best_m, 'top_k': args.top_k,
                   'alphas_tested': args.alphas}, f, indent=2)
    print(f'[submission] {args.out}')
    print(f'[log]        {log}')


if __name__ == '__main__':
    main()
