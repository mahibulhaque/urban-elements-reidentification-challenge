"""Weighted concat of two L2-normalized feature sets.

Given two cached dirs A, B and a weight α ∈ [0,1]:
  f = [sqrt(α)·f_A_norm, sqrt(1-α)·f_B_norm]    (unit norm)
So cosine similarity of two such vectors equals α·sim_A + (1-α)·sim_B.

Writes a new feat cache (qf.npy, gf.npy, meta.json) compatible with
eval_variants.py and the per_class_rerank pipeline.
"""
import argparse, json, os, numpy as np


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir_a', required=True)
    ap.add_argument('--dir_b', required=True)
    ap.add_argument('--alpha', type=float, required=True, help='weight on A (0..1)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    qf_a = l2(np.load(os.path.join(args.dir_a, 'qf.npy')))
    gf_a = l2(np.load(os.path.join(args.dir_a, 'gf.npy')))
    qf_b = l2(np.load(os.path.join(args.dir_b, 'qf.npy')))
    gf_b = l2(np.load(os.path.join(args.dir_b, 'gf.npy')))

    wA = np.sqrt(args.alpha)
    wB = np.sqrt(1.0 - args.alpha)
    qf = np.concatenate([wA * qf_a, wB * qf_b], axis=1).astype(np.float32)
    gf = np.concatenate([wA * gf_a, wB * gf_b], axis=1).astype(np.float32)

    with open(os.path.join(args.dir_a, 'meta.json')) as f:
        meta = json.load(f)
    meta['feat_dim'] = int(qf.shape[1])
    meta['source'] = {'dir_a': args.dir_a, 'dir_b': args.dir_b, 'alpha': args.alpha}

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'qf.npy'), qf)
    np.save(os.path.join(args.out, 'gf.npy'), gf)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f'[done α={args.alpha:.2f}] qf={qf.shape} gf={gf.shape} -> {args.out}')


if __name__ == '__main__':
    main()
