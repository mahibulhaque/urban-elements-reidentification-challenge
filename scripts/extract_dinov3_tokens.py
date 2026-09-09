"""Extract multiple token-pool variants from a trained DinoV3ReID checkpoint.

Saves, for each of qf/gf:
  cls.npy        — final CLS token (current default)
  gap.npy        — global average pool of final-layer patch tokens
  mxp.npy        — global max  pool of final-layer patch tokens
  reg.npy        — mean of register tokens (DINOv3 has 4)
  stripe4.npy    — 4 horizontal stripes GAP'd then concatenated  (4 * C)

All features are post-norm (timm forward_intermediates norm=True), so they live
on the same scale as the trained CLS used by the existing pipeline.

Layout: feat_cache/<TAG>_<split>/{cls,gap,mxp,reg,stripe4}_{q,g}.npy + meta.json
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


def compute_tokens(timm_model, x, last_idx):
    """Returns dict of (B,*) tensors for every variant.

    Critical: timm's `forward()` applies BOTH self.norm (trunk) AND self.fc_norm
    (head pre-pool LayerNorm). `forward_intermediates(norm=True)` only applies
    self.norm. The trained ReID head/BN were learned on post-fc_norm features,
    so we must apply fc_norm here too — otherwise CLS here ≠ CLS used at train.
    """
    res = timm_model.forward_intermediates(
        x, indices=1, return_prefix_tokens=True,
        norm=True, stop_early=False, output_fmt='NLC',
        intermediates_only=True,
    )
    patches, prefix = res[-1]  # last block: (B, L, C), (B, P, C)

    fc_norm = getattr(timm_model, 'fc_norm', None)
    if fc_norm is not None and not isinstance(fc_norm, torch.nn.Identity):
        prefix = fc_norm(prefix)
        patches = fc_norm(patches)

    cls = prefix[:, 0]
    reg = prefix[:, 1:].mean(dim=1) if prefix.shape[1] > 1 else cls.clone()
    gap = patches.mean(dim=1)
    mxp = patches.max(dim=1).values

    B, L, C = patches.shape
    if L == 14 * 7:
        p = patches.view(B, 14, 7, C)
        rows = torch.tensor_split(p, 4, dim=1)
        stripe = torch.cat([r.reshape(B, -1, C).mean(dim=1) for r in rows], dim=1)
    else:
        chunks = list(torch.tensor_split(patches, 4, dim=1))
        stripe = torch.cat([c.mean(dim=1) for c in chunks], dim=1)

    return {'cls': cls, 'gap': gap, 'mxp': mxp, 'reg': reg, 'stripe4': stripe}


@torch.no_grad()
def extract(timm_model, loader, last_idx, tta=False):
    timm_model.eval()
    keys = ['cls', 'gap', 'mxp', 'reg', 'stripe4']
    bufs = {k: [] for k in keys}
    paths, pids, camids = [], [], []
    for batch in loader:
        img = batch['images'].cuda(non_blocking=True)
        feats = compute_tokens(timm_model, img, last_idx)
        if tta:
            feats_f = compute_tokens(timm_model, torch.flip(img, dims=[-1]), last_idx)
            for k in keys: feats[k] = (feats[k] + feats_f[k]) / 2.0
        for k in keys:
            bufs[k].append(feats[k].float().cpu())
        paths.extend(batch['img_path'])
        tgt = batch['targets']; cam = batch['camid']
        pids.extend(tgt.tolist() if torch.is_tensor(tgt) else list(tgt))
        camids.extend(cam.tolist() if torch.is_tensor(cam) else list(cam))
    return {k: torch.cat(bufs[k], 0).numpy().astype(np.float32) for k in keys}, \
           paths, pids, camids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weight', required=True,
                    help='Trained DinoV3ReID checkpoint (Hplus best).')
    ap.add_argument('--tta', action='store_true')
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
    timm_model = model.base.model
    last_idx = len(timm_model.blocks) - 1

    loader, num_query = build_reid_test_loader(cfg, args.dataset)
    feats, paths, pids, camids = extract(timm_model, loader, last_idx, tta=args.tta)

    os.makedirs(args.out, exist_ok=True)
    for k, v in feats.items():
        np.save(os.path.join(args.out, f'{k}_q.npy'), v[:num_query])
        np.save(os.path.join(args.out, f'{k}_g.npy'), v[num_query:])
        print(f'  [{k:8s}] q={v[:num_query].shape}  g={v[num_query:].shape}')

    meta = {
        'query_paths':   paths[:num_query],
        'gallery_paths': paths[num_query:],
        'query_pids':    pids[:num_query],
        'gallery_pids':  pids[num_query:],
        'query_camids':  camids[:num_query],
        'gallery_camids': camids[num_query:],
        'dataset': args.dataset,
        'weight': args.weight,
        'variants': list(feats.keys()),
        'feat_dims': {k: int(v.shape[1]) for k, v in feats.items()},
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f'[done] -> {args.out}')


if __name__ == '__main__':
    main()
