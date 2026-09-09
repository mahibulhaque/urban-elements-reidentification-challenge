"""Per-class CAJ rerank where each class has its own heuristic CAJ params.

Heuristic mapping from the LB-validated per-class k-reciprocal params:

    class       std k1 / k2 / lam   ->  CAJ k1_intra / k1_inter / k2 / lam
    container       30 / 3 / 0.2          5 / 25 / 5 / 0.2
    crosswalk       30 / 3 / 0.0          5 / 25 / 5 / 0.1
    rubbishbins     20 / 3 / 0.2          3 / 17 / 5 / 0.2
    trafficsign     10 / 2 / 0.2          3 / 7  / 3 / 0.2

Rationale:
  - k1_intra small (3 for small/noisy classes, 5 otherwise) since most
    same-camera neighbors are different identities.
  - k1_inter = std_k1 - k1_intra so the total reciprocal radius matches the
    per-class k-rerank tuning that won on LB.
  - k2 mostly kept at 5 (the LB-winner global value). Tighter (3) for
    trafficsign where ranking is noisiest.
  - lambda bumped slightly above zero for crosswalk (Jaccard alone is unstable
    when raw cosine is bimodal); kept at 0.2 elsewhere.

All params are FIXED heuristics — no val grid search.
"""
import argparse, csv, json, os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from urban_elements_reid_challenge.utils.caj_re_ranking import re_ranking_caj  # noqa: E402

DATA = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026/Dataset/UrbanUAM_Merged'

# (k1_intra, k1_inter, k2, lam) per class
PER_CLASS_CAJ = {
    'container':   (5, 25, 5, 0.2),
    'crosswalk':   (5, 25, 5, 0.1),
    'rubbishbins': (3, 17, 5, 0.2),
    'trafficsign': (3,  7, 3, 0.2),
}


def norm_cls(c):
    c = c.strip().lower()
    return 'trafficsign' if c == 'trafficsignal' else c


def read_classes(p):
    d = {}
    with open(p, newline='') as f:
        rd = csv.reader(f); hdr = next(rd)
        ni = hdr.index('imageName'); ci = hdr.index('Class')
        for r in rd: d[r[ni]] = norm_cls(r[ci])
    return d


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def caj_dist(qf_c, gf_c, q_cam, g_cam, k1_intra, k1_inter, k2, lam):
    P = torch.from_numpy(qf_c.astype(np.float32))
    G = torch.from_numpy(gf_c.astype(np.float32))
    cids = torch.from_numpy(np.concatenate([q_cam, g_cam]).astype(np.int64))
    return re_ranking_caj(
        P, G, cids,
        k1=max(k1_intra, k1_inter),
        k2=k2, lambda_value=lam,
        ckrnns=True, k1_intra=int(k1_intra), k1_inter=int(k1_inter),
        clqe=False, k2_intra=int(k2), k2_inter=int(k2),
    )


def map_from_dist(dist, q_pid, g_pid, q_cam, g_cam):
    aps = []
    for i in range(len(q_pid)):
        order = np.argsort(dist[i])
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        order = order[keep]
        m = (g_pid[order] == q_pid[i]).astype(np.float32)
        if m.sum() == 0: continue
        cum = np.cumsum(m); prec = cum / (np.arange(len(m)) + 1)
        aps.append(float((prec * m).sum() / m.sum()))
    return float(np.mean(aps)) if aps else 0.0


def per_class_caj(qf, gf, qcls, gcls, qcam, gcam, top_k=100,
                  q_pid=None, g_pid=None):
    n_q = qf.shape[0]
    out = np.zeros((n_q, top_k), dtype=np.int64)
    cls_mAP = {}
    for cls in np.unique(qcls):
        qi = np.where(qcls == cls)[0]
        gi_s = np.where(gcls == cls)[0]
        gi_o = np.where(gcls != cls)[0]
        if len(qi) == 0 or len(gi_s) == 0: continue
        params = PER_CLASS_CAJ.get(cls, (5, 25, 5, 0.2))
        k1_intra, k1_inter, k2, lam = params
        print(f'  [{cls:12s}] k1_intra={k1_intra} k1_inter={k1_inter} '
              f'k2={k2} lam={lam}   nq={len(qi)}')
        d_qg = caj_dist(qf[qi], gf[gi_s], qcam[qi], gcam[gi_s],
                        k1_intra, k1_inter, k2, lam)
        if q_pid is not None:
            cls_mAP[cls] = map_from_dist(d_qg, q_pid[qi], g_pid[gi_s],
                                         qcam[qi], gcam[gi_s])
        d_o = 1.0 - qf[qi] @ gf[gi_o].T if len(gi_o) > 0 else None
        for li, gq in enumerate(qi):
            order = np.argsort(d_qg[li])
            ranked = gi_s[order]
            if len(ranked) >= top_k:
                out[gq] = ranked[:top_k]
            else:
                need = top_k - len(ranked)
                order_o = np.argsort(d_o[li])[:need]
                out[gq] = np.concatenate([ranked, gi_o[order_o]])
    return out, cls_mAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', required=True)
    ap.add_argument('--val_dir',  default='')
    ap.add_argument('--out_csv',  required=True)
    args = ap.parse_args()

    print('[per-class CAJ heuristic params]')
    for cls, p in PER_CLASS_CAJ.items():
        print(f'  {cls:12s}  k1_intra={p[0]}  k1_inter={p[1]}  k2={p[2]}  lam={p[3]}')

    if args.val_dir:
        qf = l2(np.load(os.path.join(args.val_dir, 'qf.npy')).astype(np.float32))
        gf = l2(np.load(os.path.join(args.val_dir, 'gf.npy')).astype(np.float32))
        with open(os.path.join(args.val_dir, 'meta.json')) as f: m = json.load(f)
        nq = read_classes(os.path.join(DATA, 'val_query_classes.csv'))
        ng = read_classes(os.path.join(DATA, 'val_test_classes.csv'))
        qcls = np.array([nq[os.path.basename(p)] for p in m['query_paths']])
        gcls = np.array([ng[os.path.basename(p)] for p in m['gallery_paths']])
        q_pid = np.asarray(m['query_pids'], dtype=np.int64)
        g_pid = np.asarray(m['gallery_pids'], dtype=np.int64)
        q_cam = np.asarray(m['query_camids'], dtype=np.int64)
        g_cam = np.asarray(m['gallery_camids'], dtype=np.int64)
        _, cls_mAP = per_class_caj(qf, gf, qcls, gcls, q_cam, g_cam,
                                   q_pid=q_pid, g_pid=g_pid)
        num, den = 0.0, 0
        for cls, m_v in cls_mAP.items():
            n = int((qcls == cls).sum())
            print(f'  [val/{cls:12s}] mAP={m_v:.4f}  nq={n}')
            num += m_v * n; den += n
        print(f'  [val/overall ] mAP={num/max(den,1):.4f}  '
              f'(per-class CAJ heuristic)')

    qf = l2(np.load(os.path.join(args.test_dir, 'qf.npy')).astype(np.float32))
    gf = l2(np.load(os.path.join(args.test_dir, 'gf.npy')).astype(np.float32))
    with open(os.path.join(args.test_dir, 'meta.json')) as f: mt = json.load(f)
    nq = read_classes(os.path.join(DATA, 'query_classes.csv'))
    ng = read_classes(os.path.join(DATA, 'test_classes.csv'))
    qcls = np.array([nq[os.path.basename(p)] for p in mt['query_paths']])
    gcls = np.array([ng[os.path.basename(p)] for p in mt['gallery_paths']])
    q_cam = np.asarray(mt['query_camids'], dtype=np.int64)
    g_cam = np.asarray(mt['gallery_camids'], dtype=np.int64)
    out, _ = per_class_caj(qf, gf, qcls, gcls, q_cam, g_cam)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['imageName', 'Corresponding Indexes'])
        for i in range(len(out)):
            w.writerow(['{:06d}.jpg'.format(i + 1),
                        ' '.join(map(str, (out[i] + 1).tolist()))])
    print(f'[submission] wrote {args.out_csv}')


if __name__ == '__main__':
    main()
