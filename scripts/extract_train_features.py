"""Extract Hplus (or any DinoV3ReID checkpoint) features over the train
split, saving the pre-BN feature vectors with their pid + camid labels.

Used by the offline contrastive head — we need (feat, pid, camid) tuples
across the full train set to mine negatives at large batch.

Output layout (numpy):
  <out>/train_feats.npy   (N, D) float32
  <out>/train_pids.npy    (N,)   int64
  <out>/train_camids.npy  (N,)   int64
  <out>/train_paths.json  list of N image paths
  <out>/meta.json
"""
import argparse
import json
import os
import sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from configs import cfg  # noqa: E402
from urban_elements_reid_challenge.data.build_DG_dataloader import build_reid_test_loader  # noqa: E402
from urban_elements_reid_challenge.model.backbones.DINO_v3_timm import DinoV3ReID  # noqa: E402


@torch.no_grad()
def extract(backbone, loader):
    backbone.eval()
    feats, pids, camids, paths = [], [], [], []
    for batch in loader:
        img = batch['images'].cuda(non_blocking=True)
        f = backbone(img).float().cpu()
        feats.append(f)
        tgt = batch['targets']; cam = batch['camid']
        pids.extend(tgt.tolist() if torch.is_tensor(tgt) else list(tgt))
        camids.extend(cam.tolist() if torch.is_tensor(cam) else list(cam))
        paths.extend(batch['img_path'])
    return torch.cat(feats, 0).numpy().astype(np.float32), pids, camids, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--weight', required=True)
    ap.add_argument('--dataset', default='UrbanElementsReID',
                    help='Dataset name whose train+query+gallery all point at '
                         'image_train/ — train split is what we extract.')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg.merge_from_file(args.config)
    cfg.freeze()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', cfg.MODEL.DEVICE_ID)

    state = torch.load(args.weight, map_location='cpu')
    if 'state_dict' in state: state = state['state_dict']
    if 'model' in state: state = state['model']
    num_classes = int(state['classifier.weight'].shape[0]) if 'classifier.weight' in state else 1
    model = DinoV3ReID(num_classes=num_classes, cfg=cfg).cuda()
    model.load_state_dict(state, strict=False)

    # UrbanElementsReID has train==query==gallery, all pointing at
    # image_train/. We use the gallery loader path which iterates the gallery
    # split (= all train images, with their pids and camids preserved
    # because relabel=False on the gallery side).
    loader, _num_query = build_reid_test_loader(cfg, args.dataset)
    feats, pids, camids, paths = extract(model.base, loader)

    os.makedirs(args.out, exist_ok=True)
    # The gallery and query are duplicates here — keep unique image paths.
    # We dedup by path while preserving order (numpy via dict trick).
    seen, keep = set(), []
    for i, p in enumerate(paths):
        if p in seen: continue
        seen.add(p); keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    feats = feats[keep]
    pids   = [pids[i]   for i in keep]
    camids = [camids[i] for i in keep]
    paths  = [paths[i]  for i in keep]
    print(f'[dedup] kept {len(keep):,} unique imgs')

    np.save(os.path.join(args.out, 'train_feats.npy'),  feats)
    np.save(os.path.join(args.out, 'train_pids.npy'),   np.asarray(pids,   dtype=np.int64))
    np.save(os.path.join(args.out, 'train_camids.npy'), np.asarray(camids, dtype=np.int64))
    with open(os.path.join(args.out, 'train_paths.json'), 'w') as f:
        json.dump(paths, f)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump({'weight': args.weight, 'feat_dim': int(feats.shape[1]),
                   'n': int(feats.shape[0])}, f)
    print(f'[done] feats={feats.shape}  unique_pids={len(set(pids))}  -> {args.out}')


if __name__ == '__main__':
    main()
