"""Multi-resolution feature extraction for a trained DinoV3ReID checkpoint.

For each (resolution, flip) pair:
  - build a test transform pipeline at that resolution
  - run inference; save qf.npy, gf.npy, meta.json

Output dirs are named feat_cache/<tag>_<HxW>[_flip]_(val|test).
"""
import argparse, json, os, sys, copy
import numpy as np
import torch
import torchvision.transforms as T

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from configs import cfg  # noqa: E402
from urban_elements_reid_challenge.data.build_DG_dataloader import build_reid_test_loader  # noqa: E402
from urban_elements_reid_challenge.model.backbones.DINO_v3_timm import DinoV3ReID  # noqa: E402


@torch.no_grad()
def extract(backbone, loader, hflip=False):
    backbone.eval()
    feats, paths, pids, camids = [], [], [], []
    for batch in loader:
        img = batch['images'].cuda(non_blocking=True)
        if hflip:
            img = torch.flip(img, dims=[-1])
        feats.append(backbone(img).float().cpu())
        paths.extend(batch['img_path'])
        tgt = batch['targets']; cam = batch['camid']
        pids.extend(tgt.tolist() if torch.is_tensor(tgt) else list(tgt))
        camids.extend(cam.tolist() if torch.is_tensor(cam) else list(cam))
    return torch.cat(feats, 0).numpy().astype(np.float32), paths, pids, camids


def save_split(out, qf, gf, meta_extra):
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, 'qf.npy'), qf)
    np.save(os.path.join(out, 'gf.npy'), gf)
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta_extra, f)
    print(f'  wrote {out}  qf={qf.shape} gf={gf.shape}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--weight', required=True)
    ap.add_argument('--datasets', nargs='+', required=True,
                    help='Dataset names to extract (e.g. UrbanElementsReID_val UrbanElementsReID_test)')
    ap.add_argument('--sizes', nargs='+', type=int, required=True,
                    help='Even-length list of HxW pairs, e.g. 192 96 224 112 256 128 288 144')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--feat_root', default='feat_cache')
    ap.add_argument('--with_flip', action='store_true', help='Also extract horizontally flipped variants')
    args = ap.parse_args()

    cfg.merge_from_file(args.config)
    cfg.freeze()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', cfg.MODEL.DEVICE_ID)

    state = torch.load(args.weight, map_location='cpu')
    if 'state_dict' in state: state = state['state_dict']
    if 'model' in state: state = state['model']
    n_classes = int(state['classifier.weight'].shape[0]) if 'classifier.weight' in state else 1
    model = DinoV3ReID(num_classes=n_classes, cfg=cfg).cuda()
    model.load_state_dict(state, strict=False)
    backbone = model.base

    assert len(args.sizes) % 2 == 0, '--sizes must be H1 W1 H2 W2 ...'
    pairs = [(args.sizes[i], args.sizes[i + 1]) for i in range(0, len(args.sizes), 2)]
    print(f'extracting at resolutions: {pairs}  flip={args.with_flip}')

    # Build a test loader per (dataset, size) by overriding cfg.INPUT.SIZE_TEST.
    for ds in args.datasets:
        suffix = 'val' if 'val' in ds.lower() else 'test'
        for h, w in pairs:
            # rebuild cfg for this size (cfg is frozen, so use defrost-clone)
            cfg.defrost()
            cfg.INPUT.SIZE_TEST = [h, w]
            cfg.freeze()
            loader, num_query = build_reid_test_loader(cfg, ds)
            for flip in ([False, True] if args.with_flip else [False]):
                feats, paths, pids, camids = extract(backbone, loader, hflip=flip)
                qf, gf = feats[:num_query], feats[num_query:]
                meta = {
                    'query_paths':    paths[:num_query],
                    'gallery_paths':  paths[num_query:],
                    'query_pids':     pids[:num_query],
                    'gallery_pids':   pids[num_query:],
                    'query_camids':   camids[:num_query],
                    'gallery_camids': camids[num_query:],
                    'dataset': ds,
                    'weight':  args.weight,
                    'feat_dim': int(qf.shape[1]),
                    'size_test': [h, w],
                    'hflip': bool(flip),
                }
                tag = args.tag
                size_tag = f'{h}x{w}'
                flip_tag = '_flip' if flip else ''
                out = os.path.join(args.feat_root, f'{tag}_{size_tag}{flip_tag}_{suffix}')
                save_split(out, qf, gf, meta)


if __name__ == '__main__':
    main()
