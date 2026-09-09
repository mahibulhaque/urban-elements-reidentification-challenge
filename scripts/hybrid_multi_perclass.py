"""N-way per-class hybrid of already-generated per-class submission CSVs.

Given a routing dict {class_name: submission_csv_path}, walks every test
query, looks up its class in query_classes.csv, and copies the top-100 row
from the chosen submission. Verifies expected val mAP from each model's
eval_*.json.
"""
import argparse, csv, json, os
import numpy as np

DATA = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026/Dataset/UrbanUAM_Merged'


def norm(c):
    c = c.strip().lower()
    return 'trafficsign' if c == 'trafficsignal' else c


def read_classes(p):
    d = {}
    with open(p, newline='') as f:
        rd = csv.reader(f); hdr = next(rd)
        ni = hdr.index('imageName'); ci = hdr.index('Class')
        for r in rd: d[r[ni]] = norm(r[ci])
    return d


def read_sub(path):
    rows = {}
    with open(path, newline='') as f:
        rd = csv.reader(f); next(rd)
        for r in rd: rows[r[0]] = r[1]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', nargs='+', required=True,
                    help='Pairs cls=path, e.g. container=submissions/submission_Hplus_perclass.csv ...')
    ap.add_argument('--eval_jsons', nargs='+', default=[],
                    help='Optional: list of eval_*.json files to compute expected hybrid val mAP')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    routing = {}
    for r in args.route:
        cls, path = r.split('=', 1)
        routing[cls.strip()] = path.strip()
    print('routing:', routing)

    # Optionally compute expected val mAP from eval JSONs (per-class mAPs)
    if args.eval_jsons:
        per_class_map = {}
        for j in args.eval_jsons:
            with open(j) as f: d = json.load(f)
            tag = os.path.basename(j).replace('eval_', '').replace('.json', '')
            for cls, info in d['perclass_params'].items():
                per_class_map.setdefault(cls, {})[tag] = (info['mAP'], info['n_query'])

        # Map each routed submission file back to a tag
        def path_to_tag(p): return os.path.basename(p).replace('submission_', '').replace('_perclass.csv', '').replace('.csv', '')
        total_n, total_d = 0.0, 0
        print('\nexpected val mAP (sum of per-class winners):')
        for cls, sub_path in routing.items():
            chosen_tag = path_to_tag(sub_path)
            best = per_class_map.get(cls, {})
            chosen_m = None; chosen_n = None
            for tag, (m, n) in best.items():
                if tag in chosen_tag or chosen_tag in tag:
                    chosen_m, chosen_n = m, n
                    break
            if chosen_m is None:
                print(f'  {cls:12s} (cannot resolve mAP for {chosen_tag})')
                continue
            print(f'  {cls:12s} nq={chosen_n:4d}  mAP={chosen_m:.4f}  via {chosen_tag}')
            total_n += chosen_m * chosen_n
            total_d += chosen_n
        if total_d > 0:
            print(f'  -- expected weighted val mAP: {total_n/total_d:.4f}')

    # Load test-query classes
    q_cls_test = read_classes(os.path.join(DATA, 'query_classes.csv'))

    # Load all routed submissions once
    cached = {p: read_sub(p) for p in set(routing.values())}

    # Determine the canonical query order (use any submission's keys, sorted)
    any_sub = next(iter(cached.values()))
    names = sorted(any_sub.keys(), key=lambda s: int(s.split('.')[0]))

    # Build hybrid CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    src_count = {p: 0 for p in routing.values()}
    missing = 0
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for name in names:
            cls = q_cls_test.get(name)
            if cls is None or cls not in routing:
                missing += 1
                # fallback to first routed submission
                fallback = next(iter(routing.values()))
                w.writerow([name, cached[fallback][name]])
                src_count[fallback] += 1
                continue
            sub_path = routing[cls]
            w.writerow([name, cached[sub_path][name]])
            src_count[sub_path] += 1

    print(f'\nwrote {args.out}')
    print(f'queries routed:')
    for p, n in src_count.items():
        print(f'  {n:4d} from {p}')
    if missing:
        print(f'  ({missing} queries had unknown class — used fallback)')


if __name__ == '__main__':
    main()
