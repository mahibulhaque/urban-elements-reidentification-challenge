"""Mutual Nearest Neighbor pseudo-labeling for c004 → c001-c003 domain adaptation.

For each c004 test query, find its top-1 closest c001-c003 test gallery item;
for each gallery item, find its top-1 closest query. Keep only the pairs where
the relationship is mutual (q_i's top match is g_j AND g_j's top match is q_i).

Each MNN pair becomes a *synthetic identity*: both images get the same pseudo-id
(starting from a configurable offset to avoid collision with original train pids).
A consensus-similarity threshold filters out the lowest-confidence matches.

Output: outputs/MNN/pseudo_labels.csv with columns:
  pseudo_id, image_path, camid, similarity
(two rows per MNN pair: one for the c004 query, one for the c001-c003 gallery).
"""
import argparse, csv, json, os
import numpy as np

REPO = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026'
DATA = os.path.join(REPO, 'Dataset/UrbanUAM_Merged')


def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def cam_int_from_str(s: str) -> int:
    return int(''.join(ch for ch in s if ch.isdigit()) or -1)


def read_test_gallery_camids():
    """Return dict {imageName: camid_int} for test gallery (c001-c003)."""
    d = {}
    with open(os.path.join(DATA, 'test_classes.csv'), newline='') as f:
        rd = csv.reader(f); next(rd)
        for row in rd:
            d[row[1]] = cam_int_from_str(row[0])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat_dir', default='feat_cache/Hplus_test',
                    help='Cached Hplus test features (qf=c004 queries, gf=c001-c003 gallery)')
    ap.add_argument('--out',     default='outputs/MNN/pseudo_labels.csv')
    ap.add_argument('--pid_offset', type=int, default=2000,
                    help='Pseudo-id offset (must exceed the largest training objectID).')
    ap.add_argument('--min_sim', type=float, default=None,
                    help='Optional minimum cosine sim to keep an MNN pair (default: keep all).')
    ap.add_argument('--top_n',   type=int,   default=None,
                    help='Optional max number of MNN pairs (highest-sim first).')
    args = ap.parse_args()

    qf = l2(np.load(os.path.join(REPO, args.feat_dir, 'qf.npy')).astype(np.float32))
    gf = l2(np.load(os.path.join(REPO, args.feat_dir, 'gf.npy')).astype(np.float32))
    with open(os.path.join(REPO, args.feat_dir, 'meta.json')) as f:
        meta = json.load(f)
    q_paths = meta['query_paths']; g_paths = meta['gallery_paths']
    nq, ng = len(q_paths), len(g_paths)
    print(f'Hplus test features: {nq} c004 queries x {ng} c001-c003 gallery')

    # Cosine sim matrix (already L2-normed) — chunked to limit peak memory
    sim = qf @ gf.T  # (nq, ng) ≈ 928 x 2844 = ~10 MB float32 — fine
    print(f'sim matrix: {sim.shape}, dtype={sim.dtype}')

    q_to_g_top1 = sim.argmax(axis=1)        # (nq,)
    g_to_q_top1 = sim.argmax(axis=0)        # (ng,)

    pairs = []  # (qi, gj, similarity)
    for qi in range(nq):
        gj = int(q_to_g_top1[qi])
        if int(g_to_q_top1[gj]) == qi:
            pairs.append((qi, gj, float(sim[qi, gj])))
    pairs.sort(key=lambda x: -x[2])
    print(f'MNN pairs found: {len(pairs)} / {nq} ({100*len(pairs)/nq:.1f}%)')
    if pairs:
        print(f'  similarity range: [{pairs[-1][2]:.4f} .. {pairs[0][2]:.4f}]')
        print(f'  median: {pairs[len(pairs)//2][2]:.4f}')

    if args.min_sim is not None:
        kept = [p for p in pairs if p[2] >= args.min_sim]
        print(f'  after min_sim={args.min_sim}: {len(kept)} pairs')
        pairs = kept
    if args.top_n is not None:
        pairs = pairs[:args.top_n]
        print(f'  after top_n={args.top_n}: {len(pairs)} pairs')

    g_camids = read_test_gallery_camids()
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pseudo_id', 'image_path', 'camid', 'similarity'])
        for k, (qi, gj, s) in enumerate(pairs):
            pid = args.pid_offset + k
            qname = os.path.basename(q_paths[qi])
            gname = os.path.basename(g_paths[gj])
            qpath = os.path.join(DATA, 'image_query', qname)
            gpath = os.path.join(DATA, 'image_test',  gname)
            qcam  = 4   # c004 test queries are all camera 4
            gcam  = g_camids.get(gname, -1)
            w.writerow([pid, qpath, qcam, f'{s:.6f}'])
            w.writerow([pid, gpath, gcam, f'{s:.6f}'])
    print(f'wrote {out_path}  ({2 * len(pairs)} rows = {len(pairs)} synthetic identities)')


if __name__ == '__main__':
    main()
