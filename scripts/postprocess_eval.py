"""
Postprocessing evaluation on val split v2.
Steps:
  0. Baseline (L2 + cosine distance, no tricks)
  1. Class filtering
  2. Class filtering + global k-reciprocal re-ranking  (k1=20, k2=6, λ=0.3)
  3. Class filtering + per-class re-ranking  (grid-searched per class)

Run from repo root:
  python postprocess_eval.py
"""

import sys, os
sys.path.insert(0, '.')

import torch
import numpy as np
import pandas as pd
from itertools import product
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from configs import cfg
from urban_elements_reid_challenge.model import make_model
from urban_elements_reid_challenge.utils.re_ranking import re_ranking_from_features as re_ranking

# ── config ────────────────────────────────────────────────────────────────────
CKPT          = '/kaggle/working/outputs/split_v2/part_attention_vit_60.pth'
CFG_FILE      = './config/split_v2_train.yml'
SPLIT_DIR     = '/kaggle/working/splits_v2'
UAM_ROOT      = '/kaggle/input/datasets/mahibulhaque/uam-dataset/UAM_Unified'
UAM_QUERY_DIR = f'{UAM_ROOT}/image_query'
UAM_TEST_DIR  = f'{UAM_ROOT}/image_test'
BATCH_SIZE    = 128
DEVICE        = 'cuda'

cfg.merge_from_file(CFG_FILE)
cfg.freeze()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID


# ── helpers ───────────────────────────────────────────────────────────────────
class FlatDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.paths = [os.path.join(img_dir, f) for f in df['filename']]
        self.pids  = df['identity_id'].tolist()
        self.cams  = df['camera_id'].tolist()
        self.clss  = df['class'].tolist()
        self.transform = transform

    def __len__(self):  return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.transform(img), self.pids[i], self.cams[i], self.clss[i]


def build_transform():
    return transforms.Compose([
        transforms.Resize(cfg.INPUT.SIZE_TEST, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


@torch.no_grad()
def extract_features(model, df, img_dir, transform):
    ds     = FlatDataset(df, img_dir, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    feats, pids, cams, clss = [], [], [], []
    model.eval()
    for imgs, pid, cam, cls in loader:
        imgs = imgs.to(DEVICE)
        f    = model(imgs)
        feats.append(f.cpu())
        pids.extend(pid.numpy().tolist())
        cams.extend(cam.numpy().tolist())
        clss.extend(cls)
    feats = torch.cat(feats, 0)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats, np.array(pids), np.array(cams), np.array(clss)


def eval_func(distmat, q_pids, g_pids, q_cams, g_cams):
    """Standard ReID eval: same-pid-same-cam gallery removed."""
    num_q, num_g = distmat.shape
    max_rank = min(50, num_g)
    indices  = np.argsort(distmat, axis=1)
    matches  = (g_pids[indices] == q_pids[:, None]).astype(np.int32)

    all_cmc, all_ap, n_valid = [], [], 0
    for q in range(num_q):
        order  = indices[q]
        remove = (g_pids[order] == q_pids[q]) & (g_cams[order] == q_cams[q])
        keep   = ~remove
        orig   = matches[q][keep]
        if not orig.any():
            continue
        n_valid += 1
        cmc = orig.cumsum(); cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        n_rel = orig.sum()
        tmp   = orig.cumsum() / np.arange(1, len(orig) + 1)
        all_ap.append((tmp * orig).sum() / n_rel)

    cmc_arr = np.stack(all_cmc).mean(0)
    return cmc_arr, float(np.mean(all_ap))


def compute_baseline_distmat(qf, gf):
    """Euclidean on L2-normed = cosine distance."""
    qf_np = qf.numpy()
    gf_np = gf.numpy()
    dist  = 2 - 2 * (qf_np @ gf_np.T)
    return dist.astype(np.float32)


def compute_rerank_distmat(qf, gf, k1, k2, lam):
    return re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lam)


def report(label, distmat, q_pids, g_pids, q_cams, g_cams, q_cls, g_cls, classes):
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_cams, g_cams)
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  Overall   mAP={mAP:.4f}  R1={cmc[0]:.4f}  R5={cmc[4]:.4f}  R10={cmc[9]:.4f}")
    for cls in classes:
        qi = np.where(q_cls == cls)[0]
        gi = np.where(g_cls == cls)[0]
        if len(qi) == 0: continue
        sub_dist = distmat[np.ix_(qi, gi)]
        sub_cmc, sub_mAP = eval_func(sub_dist, q_pids[qi], g_pids[gi], q_cams[qi], g_cams[gi])
        print(f"  {cls:<14} mAP={sub_mAP:.4f}  R1={sub_cmc[0]:.4f}  (q={len(qi)} g={len(gi)})")
    return mAP, cmc


# ── load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0, camera_num=None, view_num=None)
model.load_param(CKPT)
model.to(DEVICE)
model.eval()
print("Model loaded.")

# ── load splits ───────────────────────────────────────────────────────────────
qdf = pd.read_csv(os.path.join(SPLIT_DIR, 'val_query.csv'))
gdf = pd.read_csv(os.path.join(SPLIT_DIR, 'val_gallery.csv'))
transform = build_transform()

print("Extracting query features...")
qf, q_pids, q_cams, q_cls = extract_features(model, qdf, UAM_QUERY_DIR, transform)
print("Extracting gallery features...")
gf, g_pids, g_cams, g_cls = extract_features(model, gdf, UAM_TEST_DIR, transform)

print(f"Query:   {len(qf)} items   Gallery: {len(gf)} items")
classes = sorted(qdf['class'].unique())
print(f"Classes: {classes}")

# ── Step 0: baseline ─────────────────────────────────────────────────────────
base_dist = compute_baseline_distmat(qf, gf)
report("STEP 0 — Baseline (cosine, no filtering)", base_dist,
       q_pids, g_pids, q_cams, g_cams, q_cls, g_cls, classes)

# ── Step 1: class filtering ───────────────────────────────────────────────────
# For each query, set distance to non-same-class gallery items = +inf
cf_dist = base_dist.copy()
for i, qc in enumerate(q_cls):
    mask = (g_cls != qc)
    cf_dist[i, mask] = 1e9

report("STEP 1 — Class filtering", cf_dist,
       q_pids, g_pids, q_cams, g_cams, q_cls, g_cls, classes)

# ── Step 2: class filtering + global re-ranking ───────────────────────────────
print("\nRunning global re-ranking (k1=20, k2=6, λ=0.3)...")
rr_dist_global = np.full_like(base_dist, 1e9)
for cls in classes:
    qi = np.where(q_cls == cls)[0]
    gi = np.where(g_cls == cls)[0]
    if len(qi) == 0 or len(gi) == 0: continue
    rr = compute_rerank_distmat(qf[qi], gf[gi], k1=20, k2=6, lam=0.3)
    rr_dist_global[np.ix_(qi, gi)] = rr

report("STEP 2 — Class filtering + global re-ranking (k1=20,k2=6,λ=0.3)",
       rr_dist_global, q_pids, g_pids, q_cams, g_cams, q_cls, g_cls, classes)

# ── Step 3: per-class re-ranking grid search ──────────────────────────────────
K1S = [10, 15, 20, 25, 30]
K2S = [3, 5, 6, 8, 10]
LAMS = [0.1, 0.2, 0.3, 0.5]

print("\nGrid searching per-class re-ranking parameters...")
best_params = {}
for cls in classes:
    qi = np.where(q_cls == cls)[0]
    gi = np.where(g_cls == cls)[0]
    if len(qi) == 0 or len(gi) == 0:
        best_params[cls] = (20, 6, 0.3)
        continue

    best_map, best_k1, best_k2, best_lam = -1, 20, 6, 0.3
    total = len(K1S) * len(K2S) * len(LAMS)
    done  = 0
    for k1, k2, lam in product(K1S, K2S, LAMS):
        # skip k2 >= k1
        if k2 >= k1: continue
        rr = compute_rerank_distmat(qf[qi], gf[gi], k1=k1, k2=k2, lam=lam)
        _, m = eval_func(rr, q_pids[qi], g_pids[gi], q_cams[qi], g_cams[gi])
        if m > best_map:
            best_map, best_k1, best_k2, best_lam = m, k1, k2, lam
        done += 1
    best_params[cls] = (best_k1, best_k2, best_lam)
    print(f"  {cls:<14}: best k1={best_k1}, k2={best_k2}, λ={best_lam}  →  mAP={best_map:.4f}  ({done} combos)")

# apply best params per class
rr_dist_perclass = np.full_like(base_dist, 1e9)
for cls in classes:
    qi = np.where(q_cls == cls)[0]
    gi = np.where(g_cls == cls)[0]
    if len(qi) == 0 or len(gi) == 0: continue
    k1, k2, lam = best_params[cls]
    rr = compute_rerank_distmat(qf[qi], gf[gi], k1=k1, k2=k2, lam=lam)
    rr_dist_perclass[np.ix_(qi, gi)] = rr

report("STEP 3 — Class filtering + per-class re-ranking (grid-searched)",
       rr_dist_perclass, q_pids, g_pids, q_cams, g_cams, q_cls, g_cls, classes)

print("\nPer-class best parameters:")
for cls, (k1, k2, lam) in best_params.items():
    print(f"  {cls:<14}: k1={k1}, k2={k2}, λ={lam}")
