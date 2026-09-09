"""Write a per-class re-rank submission using *fixed* hyperparameters from a
prior eval JSON (e.g. De90). Skips the val-side grid search entirely, which
is necessary when val has been folded into training (DeVal ablation).
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


def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def write_perclass(qf, gf, qcls, gcls, best_params, out_csv, top_k=100):
    n_q = qf.shape[0]
    out = np.zeros((n_q, top_k), dtype=np.int64)
    for cls in np.unique(qcls):
        qi = np.where(qcls == cls)[0]
        gi_s = np.where(gcls == cls)[0]
        gi_o = np.where(gcls != cls)[0]
        if cls in best_params:
            k1, k2, lam = best_params[cls]
        else:
            k1, k2, lam = 50, 15, 0.3
        qfq = qf[qi]; gfc = gf[gi_s]
        d_s = re_ranking(qfq @ gfc.T, qfq @ qfq.T, gfc @ gfc.T,
                         k1=k1, k2=k2, lambda_value=lam)
        d_o = 1.0 - qfq @ gf[gi_o].T if len(gi_o) > 0 else None
        for li, gq in enumerate(qi):
            order = np.argsort(d_s[li])
            ranked = gi_s[order]
            if len(ranked) >= top_k:
                out[gq] = ranked[:top_k]
            else:
                need = top_k - len(ranked)
                order_o = np.argsort(d_o[li])[:need]
                out[gq] = np.concatenate([ranked, gi_o[order_o]])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(n_q):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (out[i] + 1).tolist()))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', required=True)
    ap.add_argument('--params_json', required=True,
                    help='Prior eval_*.json file; reads perclass_params from it.')
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()

    qf = l2(np.load(os.path.join(args.test_dir, 'qf.npy')).astype(np.float32))
    gf = l2(np.load(os.path.join(args.test_dir, 'gf.npy')).astype(np.float32))
    with open(os.path.join(args.test_dir, 'meta.json')) as f:
        mt = json.load(f)
    nq = read_classes(os.path.join(DATA, 'query_classes.csv'))
    ng = read_classes(os.path.join(DATA, 'test_classes.csv'))
    qcls = np.array([nq[os.path.basename(p)] for p in mt['query_paths']])
    gcls = np.array([ng[os.path.basename(p)] for p in mt['gallery_paths']])

    with open(args.params_json) as f:
        prior = json.load(f)
    pc = prior['perclass_params']
    best_params = {cls: tuple(info['params']) for cls, info in pc.items()}
    print('[params]', best_params)

    write_perclass(qf, gf, qcls, gcls, best_params, args.out_csv)
    print(f'[submission] wrote {args.out_csv}')


if __name__ == '__main__':
    main()
