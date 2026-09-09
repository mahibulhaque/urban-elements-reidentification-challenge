"""Per-class hybrid of two already-generated per-class submission CSVs.

For each class, pick the submission whose per-class val mAP is higher (as
recorded in its eval_*.json). For each test query, route to the winning
submission's top-100 based on the query's test-side class label.

Also verifies the expected val mAP using the per-class val mAPs from the
two eval JSONs.
"""
import argparse, csv, json, os, numpy as np

DATA = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026/Dataset/UrbanUAM_Merged'


def norm_class(c):
    c = c.strip().lower()
    return 'trafficsign' if c == 'trafficsignal' else c


def read_classes(p):
    d = {}
    with open(p, newline='') as f:
        rd = csv.reader(f); hdr = next(rd)
        ni = hdr.index('imageName'); ci = hdr.index('Class')
        for r in rd: d[r[ni]] = norm_class(r[ci])
    return d


def read_sub(path):
    rows = {}
    with open(path, newline='') as f:
        rd = csv.reader(f); next(rd)
        for r in rd: rows[r[0]] = r[1]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sub_a', required=True, help='submission CSV of model A (e.g. DV3 perclass)')
    ap.add_argument('--eval_a', required=True, help='eval_*.json of model A')
    ap.add_argument('--sub_b', required=True, help='submission CSV of model B (e.g. PAT perclass)')
    ap.add_argument('--eval_b', required=True, help='eval_*.json of model B')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # Test-side query classes (note: query and gallery CSVs share basenames
    # — always use the query-side CSV here).
    q_cls_test = read_classes(os.path.join(DATA, 'query_classes.csv'))

    A = read_sub(args.sub_a)
    B = read_sub(args.sub_b)
    with open(args.eval_a) as f: E_A = json.load(f)
    with open(args.eval_b) as f: E_B = json.load(f)
    pc_A = E_A['perclass_params']
    pc_B = E_B['perclass_params']

    classes = sorted(set(pc_A) | set(pc_B))
    winners = {}
    expected_val_num, expected_val_den = 0.0, 0
    for cls in classes:
        mA = pc_A.get(cls, {}).get('mAP', -1.0)
        mB = pc_B.get(cls, {}).get('mAP', -1.0)
        if mA >= mB:
            winner, m_win = 'A', mA
        else:
            winner, m_win = 'B', mB
        nq = pc_A.get(cls, {}).get('n_query', 0) or pc_B.get(cls, {}).get('n_query', 0)
        winners[cls] = winner
        expected_val_num += m_win * nq
        expected_val_den += nq
        tag_A = os.path.basename(args.eval_a)
        tag_B = os.path.basename(args.eval_b)
        print(f'  {cls:12s} nq={nq:4d}  {tag_A}={mA:.4f}  {tag_B}={mB:.4f}  -> {winner} ({m_win:.4f})')
    expected_val = expected_val_num / max(expected_val_den, 1)
    print(f'\n  expected hybrid val mAP = {expected_val:.4f}')

    # Build hybrid test CSV: iterate 000001..{N}.jpg, lookup class, pick side.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    common_names = sorted(set(A) & set(B), key=lambda s: int(s.split('.')[0]))
    n_A_used = n_B_used = 0
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for name in common_names:
            cls = q_cls_test[name]
            winner = winners.get(cls, 'A')
            if winner == 'A':
                w.writerow([name, A[name]]); n_A_used += 1
            else:
                w.writerow([name, B[name]]); n_B_used += 1
    print(f'\n[submission] wrote {args.out}  (from A={n_A_used}, from B={n_B_used})')


if __name__ == '__main__':
    main()
