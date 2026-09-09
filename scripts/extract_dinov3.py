"""Extract DINOv3 features for query+gallery on the Urban Elements dataset.

Two modes:
  - zero-shot (default): load pretrained DINOv3 backbone only, no head.
  - trained: pass --weight <path> to a DinoV3ReID checkpoint; extracts
    pre-BN CLS (matches NECK_FEAT='before').

Optionally dumps per-block CLS cube with --allblocks (adds qf_all.npy /
gf_all.npy for the last-N sweep).
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
from urban_elements_reid_challenge.model.backbones.DINO_v3_timm import DinoV3Backbone, DinoV3ReID  # noqa: E402


@torch.no_grad()
def extract(model_backbone, loader, allblocks=False, tta=False):
    model_backbone.eval()
    feats, cubes, paths, pids, camids = [], [], [], [], []
    for batch in loader:
        img = batch['images'].cuda(non_blocking=True)
        feat = model_backbone(img).float()
        if tta:
            feat_f = model_backbone(torch.flip(img, dims=[-1])).float()
            feat = (feat + feat_f) / 2.0
        feats.append(feat.cpu())
        if allblocks:
            cube = model_backbone.forward_allblocks(img).float()
            if tta:
                cube_f = model_backbone.forward_allblocks(torch.flip(img, dims=[-1])).float()
                cube = (cube + cube_f) / 2.0
            cubes.append(cube.cpu())
        paths.extend(batch['img_path'])
        tgt = batch['targets']; cam = batch['camid']
        pids.extend(tgt.tolist() if torch.is_tensor(tgt) else list(tgt))
        camids.extend(cam.tolist() if torch.is_tensor(cam) else list(cam))
    feat = torch.cat(feats, 0).numpy().astype(np.float32)
    cube = torch.cat(cubes, 0).numpy().astype(np.float32) if allblocks else None
    return feat, cube, paths, pids, camids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='DINOv3 config yml (for img size/norm)')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weight', default=None,
                    help='Optional trained DinoV3ReID checkpoint; '
                         'if omitted, uses the raw pretrained backbone (zero-shot).')
    ap.add_argument('--allblocks', action='store_true',
                    help='Also dump per-block CLS cube (qf_all.npy, gf_all.npy).')
    ap.add_argument('--tta', action='store_true')
    args = ap.parse_args()

    cfg.merge_from_file(args.config)
    cfg.freeze()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', cfg.MODEL.DEVICE_ID)

    if args.weight:
        # Load full trained ReID model and extract its CLS (pre-BN) features.
        # We need num_classes > 0 to match the classifier shape, but we'll
        # discard it; use the shape from the state dict.
        state = torch.load(args.weight, map_location='cpu')
        if 'state_dict' in state: state = state['state_dict']
        if 'model' in state: state = state['model']
        if 'classifier.weight' in state:
            num_classes = int(state['classifier.weight'].shape[0])
        else:
            num_classes = 1  # dummy
        model = DinoV3ReID(num_classes=num_classes, cfg=cfg).cuda()
        model.load_state_dict(state, strict=False)
        backbone = model.base
        mode = 'trained'
    else:
        backbone = DinoV3Backbone(cfg.MODEL.DINOV3_VARIANT,
                                  pretrained_path=cfg.MODEL.PRETRAIN_PATH).cuda()
        mode = 'zero-shot'

    loader, num_query = build_reid_test_loader(cfg, args.dataset)
    feat, cube, paths, pids, camids = extract(backbone, loader,
                                              allblocks=args.allblocks, tta=args.tta)

    qf, gf = feat[:num_query], feat[num_query:]
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'qf.npy'), qf)
    np.save(os.path.join(args.out, 'gf.npy'), gf)
    if cube is not None:
        qf_all, gf_all = cube[:num_query], cube[num_query:]
        np.save(os.path.join(args.out, 'qf_all.npy'), qf_all)
        np.save(os.path.join(args.out, 'gf_all.npy'), gf_all)

    meta = {
        'query_paths':   paths[:num_query],
        'gallery_paths': paths[num_query:],
        'query_pids':    pids[:num_query],
        'gallery_pids':  pids[num_query:],
        'query_camids':  camids[:num_query],
        'gallery_camids': camids[num_query:],
        'dataset': args.dataset,
        'mode': mode,
        'weight': args.weight or '(pretrained backbone)',
        'feat_dim': int(qf.shape[1]),
    }
    if cube is not None:
        meta['depth'] = int(cube.shape[1])
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f'[done/{mode}] qf={qf.shape} gf={gf.shape} -> {args.out}')


if __name__ == '__main__':
    main()
