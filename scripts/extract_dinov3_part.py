"""Extract concat(CLS, parts) retrieval features from a DinoV3PartReID model.
Same I/O contract as extract_dinov3.py so eval_variants can be reused.
"""
import argparse, json, os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from configs import cfg                                            # noqa: E402
from urban_elements_reid_challenge.data.build_DG_dataloader import build_reid_test_loader        # noqa: E402
from urban_elements_reid_challenge.model.backbones.DINO_v3_part import DinoV3PartReID             # noqa: E402


@torch.no_grad()
def extract(model, loader, tta=False):
    model.eval()
    feats, paths, pids, camids = [], [], [], []
    for batch in loader:
        img = batch['images'].cuda(non_blocking=True)
        f = model(img).float()
        if tta:
            f_flip = model(torch.flip(img, dims=[-1])).float()
            f = (f + f_flip) / 2.0
        feats.append(f.cpu())
        paths.extend(batch['img_path'])
        tgt = batch['targets']; cam = batch['camid']
        pids.extend(tgt.tolist() if torch.is_tensor(tgt) else list(tgt))
        camids.extend(cam.tolist() if torch.is_tensor(cam) else list(cam))
    return torch.cat(feats, 0).numpy().astype(np.float32), paths, pids, camids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--weight', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tta', action='store_true')
    args = ap.parse_args()

    cfg.merge_from_file(args.config); cfg.freeze()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', cfg.MODEL.DEVICE_ID)

    state = torch.load(args.weight, map_location='cpu')
    if 'state_dict' in state: state = state['state_dict']
    if 'model' in state: state = state['model']
    # num_classes from classifiers.0.weight
    num_classes = int(state['classifiers.0.weight'].shape[0]) \
        if 'classifiers.0.weight' in state else 1
    model = DinoV3PartReID(num_classes=num_classes, cfg=cfg).cuda()
    model.load_state_dict(state, strict=False)

    loader, num_query = build_reid_test_loader(cfg, args.dataset)
    feat, paths, pids, camids = extract(model, loader, tta=args.tta)
    qf, gf = feat[:num_query], feat[num_query:]

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'qf.npy'), qf)
    np.save(os.path.join(args.out, 'gf.npy'), gf)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump({
            'query_paths':   paths[:num_query],
            'gallery_paths': paths[num_query:],
            'query_pids':    pids[:num_query],
            'gallery_pids':  pids[num_query:],
            'query_camids':  camids[:num_query],
            'gallery_camids': camids[num_query:],
            'dataset': args.dataset, 'weight': args.weight,
            'feat_dim': int(qf.shape[1]),
            'num_parts': int(model.num_parts),
        }, f)
    print(f'[done] qf={qf.shape}  gf={gf.shape}  -> {args.out}')


if __name__ == '__main__':
    main()
