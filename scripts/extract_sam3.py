"""Extract SAM 3 vision-encoder global features for ReID.

For each image, runs SAM 3's vision backbone and takes a global descriptor by
average-pooling the deepest backbone FPN level. Output format matches
extract_dinov3.py so eval_variants.py / per-class rerank work unchanged.

Requires: sam3 package + HF auth (gated repo facebook/sam3 or facebook/sam3.1).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DATA = '/home/mahibulhaque/ReID/Urban-Elements-ReID-Challenge-2026/Dataset/UrbanUAM_Merged'


def list_split(split: str):
    """Return ([(name, full_path)], pids[, camids]) for one split."""
    if split == 'val':
        q_dir = os.path.join(DATA, 'val_image_query')
        g_dir = os.path.join(DATA, 'val_image_test')
        q_csv = os.path.join(DATA, 'val_query_classes.csv')
        g_csv = os.path.join(DATA, 'val_test_classes.csv')
    elif split == 'test':
        q_dir = os.path.join(DATA, 'image_query')
        g_dir = os.path.join(DATA, 'image_test')
        q_csv = os.path.join(DATA, 'query_classes.csv')
        g_csv = os.path.join(DATA, 'test_classes.csv')
    else:
        raise ValueError(split)

    def read_csv(path):
        rows = []
        with open(path, newline='') as f:
            rd = csv.DictReader(f)
            for r in rd: rows.append(r)
        return rows

    q_rows = read_csv(q_csv)
    g_rows = read_csv(g_csv)
    return q_dir, g_dir, q_rows, g_rows


def get_attr(row, key, default=-1):
    if key in row and row[key] != '': return int(row[key])
    return int(default)


@torch.inference_mode()
def extract_global(processor: Sam3Processor, model, pil_img: Image.Image) -> np.ndarray:
    """Forward one image through SAM 3 backbone, return a 1-D global descriptor.

    Strategy: average-pool the deepest sam3_features (most semantic) over its
    spatial dims. dim = embed_dim of the ViT (1024 for the SAM 3 ViT-1024).
    """
    state = processor.set_image(pil_img.convert('RGB'))
    backbone_out = state['backbone_out']
    feats = backbone_out['vision_features']  # last (smallest) FPN level, (1, C, H, W)
    feat = feats.float().mean(dim=(2, 3)).squeeze(0).cpu()
    return feat.numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], required=True)
    ap.add_argument('--out', required=True, help='output dir for qf.npy / gf.npy / meta.json')
    ap.add_argument('--version', default='sam3', choices=['sam3', 'sam3.1'])
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    print(f'building SAM 3 image model (version={args.version}) ...')
    model = build_sam3_image_model(device=args.device, eval_mode=True)
    processor = Sam3Processor(model, device=args.device)
    print('model ready.')

    q_dir, g_dir, q_rows, g_rows = list_split(args.split)
    queries = [(r['imageName'], os.path.join(q_dir, r['imageName']), r) for r in q_rows]
    gallery = [(r['imageName'], os.path.join(g_dir, r['imageName']), r) for r in g_rows]
    print(f'queries: {len(queries)}, gallery: {len(gallery)}')

    qf, q_paths, q_pids, q_cams = [], [], [], []
    gf, g_paths, g_pids, g_cams = [], [], [], []

    def _do(items, feats_list, paths_list, pids_list, cams_list):
        for i, (name, p, row) in enumerate(items):
            img = Image.open(p)
            f = extract_global(processor, model, img)
            feats_list.append(f)
            paths_list.append(p)
            pids_list.append(get_attr(row, 'objectID', -1))
            cam_str = row.get('cameraID', 'c-1')
            cams_list.append(int(''.join(ch for ch in cam_str if ch.isdigit()) or -1))
            if (i + 1) % 100 == 0:
                print(f'  {i+1}/{len(items)}')

    print('extracting query features ...')
    _do(queries, qf, q_paths, q_pids, q_cams)
    print('extracting gallery features ...')
    _do(gallery, gf, g_paths, g_pids, g_cams)

    qf = np.stack(qf, 0).astype(np.float32)
    gf = np.stack(gf, 0).astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'qf.npy'), qf)
    np.save(os.path.join(args.out, 'gf.npy'), gf)
    meta = {
        'query_paths':    q_paths,
        'gallery_paths':  g_paths,
        'query_pids':     q_pids,
        'gallery_pids':   g_pids,
        'query_camids':   q_cams,
        'gallery_camids': g_cams,
        'split': args.split,
        'feat_dim': int(qf.shape[1]),
        'source': f'SAM3-{args.version} vision_features (avg-pooled)',
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f'[done] qf={qf.shape} gf={gf.shape} -> {args.out}')


if __name__ == '__main__':
    main()
