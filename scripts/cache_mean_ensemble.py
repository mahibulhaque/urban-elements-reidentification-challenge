"""L2-normalize each input feat dir, then arithmetic mean, then L2-norm.

Outputs one virtual checkpoint compatible with eval_variants.py.
Use this for multi-resolution / TTA averaging. Feature dim of the output is
the dim of the inputs (must all match), unlike cache_ensemble.py which concatenates.
"""
import argparse, json, os, numpy as np


def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    qf_acc = gf_acc = None
    meta = None
    for d in args.dirs:
        qf = l2(np.load(os.path.join(d, 'qf.npy')).astype(np.float32))
        gf = l2(np.load(os.path.join(d, 'gf.npy')).astype(np.float32))
        with open(os.path.join(d, 'meta.json')) as f:
            m = json.load(f)
        if qf_acc is None:
            qf_acc, gf_acc, meta = qf.copy(), gf.copy(), m
        else:
            assert qf.shape == qf_acc.shape and gf.shape == gf_acc.shape, \
                f'dim mismatch: {qf.shape} vs {qf_acc.shape}'
            qf_acc += qf
            gf_acc += gf
    n = len(args.dirs)
    qf = l2(qf_acc / n).astype(np.float32)
    gf = l2(gf_acc / n).astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'qf.npy'), qf)
    np.save(os.path.join(args.out, 'gf.npy'), gf)
    meta['feat_dim'] = int(qf.shape[1])
    meta['source'] = {'mean_of': args.dirs, 'n': n}
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f'[mean-ensemble n={n}] qf={qf.shape} gf={gf.shape} -> {args.out}')


if __name__ == '__main__':
    main()
