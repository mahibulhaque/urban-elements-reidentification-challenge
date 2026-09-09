"""Extract per-block CLS tokens for the part_attention_vit backbone.

Saves a feature cube of shape (N_samples, depth, embed_dim) so a follow-up
CPU script can sweep "average last-N CLS tokens" as a feature.
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
from urban_elements_reid_challenge.model import make_model  # noqa: E402
from urban_elements_reid_challenge.data.build_DG_dataloader import build_reid_test_loader  # noqa: E402


@torch.no_grad()
def extract_allblocks(model, loader, tta=False):
    """Return (N, depth, D) array of per-block CLS tokens + metadata."""
    model.eval()
    feats, paths, pids, camids = [], [], [], []
    for batch in loader:
        img = batch['images'].cuda(non_blocking=True)
        # model.base is the part_Attention_ViT backbone — returns a list of
        # layerwise tokens (already LayerNorm'd), each (B, num_tokens, D).
        layerwise = model.base(img)
        cls_stack = torch.stack([t[:, 0] for t in layerwise], dim=1).float()  # (B, depth, D)
        if tta:
            layerwise_f = model.base(torch.flip(img, dims=[-1]))
            cls_stack_f = torch.stack([t[:, 0] for t in layerwise_f], dim=1).float()
            cls_stack = (cls_stack + cls_stack_f) / 2.0
        feats.append(cls_stack.cpu())
        paths.extend(batch['img_path'])
        tgt = batch['targets']; cam = batch['camid']
        pids.extend(tgt.tolist() if torch.is_tensor(tgt) else list(tgt))
        camids.extend(cam.tolist() if torch.is_tensor(cam) else list(cam))
    return torch.cat(feats, 0), paths, pids, camids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--weight', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tta', action='store_true')
    args = ap.parse_args()

    cfg.merge_from_file(args.config)
    cfg.freeze()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', cfg.MODEL.DEVICE_ID)

    model = make_model(cfg, cfg.MODEL.NAME, num_class=0, camera_num=None, view_num=None)
    model.load_param(args.weight)
    model.cuda()

    loader, num_query = build_reid_test_loader(cfg, args.dataset)
    feats, paths, pids, camids = extract_allblocks(model, loader, tta=args.tta)

    feats = feats.numpy().astype(np.float32)  # (N, depth, D) - NOT normalized
    qf, gf = feats[:num_query], feats[num_query:]

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'qf_all.npy'), qf)
    np.save(os.path.join(args.out, 'gf_all.npy'), gf)
    meta = {
        'query_paths':   paths[:num_query],
        'gallery_paths': paths[num_query:],
        'query_pids':    pids[:num_query],
        'gallery_pids':  pids[num_query:],
        'query_camids':  camids[:num_query],
        'gallery_camids': camids[num_query:],
        'dataset': args.dataset,
        'weight':  args.weight,
        'depth':   int(qf.shape[1]),
        'feat_dim': int(qf.shape[2]),
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f'[done] qf={qf.shape} gf={gf.shape} -> {args.out}')


if __name__ == '__main__':
    main()
