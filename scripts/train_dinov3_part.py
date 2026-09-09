"""Trainer for the Part-Aware DINOv3 ReID model (PartDV3).

Differences from train_dinov3.py:
  - Builds DinoV3PartReID instead of DinoV3ReID.
  - Loss is summed over the CLS head + each part head (each gets its own CE
    against the same pid target), plus a triplet on the concatenated feature,
    plus a diversity penalty on the part queries.
  - Eval uses the concat(CLS, parts) retrieval feature returned by
    DinoV3PartReID in eval mode.

The staged-unfreeze logic and AMP are reused from the original trainer.
"""
import argparse
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from configs import cfg                                            # noqa: E402
from urban_elements_reid_challenge.data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader  # noqa: E402
from urban_elements_reid_challenge.utils.metrics import R1_mAP_eval                             # noqa: E402
from urban_elements_reid_challenge.model.backbones.DINO_v3_part import (
    DinoV3PartReID, diversity_loss, perimage_diversity_loss,
)  # noqa: E402
from loss.triplet_loss import TripletLoss                         # noqa: E402
from loss.ce_labelSmooth import CrossEntropyLabelSmooth           # noqa: E402


def setup_logger(log_path):
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path, mode='a'), logging.StreamHandler()])
    return logging.getLogger('PART')


def set_stage(model, stage, unfreeze_n):
    if stage == 1:
        model.base.freeze_all()
    elif stage == 2:
        model.base.unfreeze_last_n_blocks(unfreeze_n)
    else:
        model.base.unfreeze_all()


def make_optimizer(cfg, model, lr_scale):
    base_lr = cfg.SOLVER.BASE_LR * lr_scale
    wd = cfg.SOLVER.WEIGHT_DECAY
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    return torch.optim.AdamW(trainable, lr=base_lr, weight_decay=wd), n_params


@torch.no_grad()
def evaluate(model, val_loader, num_query, logger, device='cuda'):
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    model.eval()
    for batch in val_loader:
        img = batch['images'].to(device, non_blocking=True)
        vid = batch['targets']; camid = batch['camid']
        feat = model(img)                  # concat retrieval feature
        evaluator.update((feat.float(), vid, camid))
    cmc, mAP = evaluator.compute()[:2]
    logger.info(f'Validation: mAP={mAP*100:.2f}%  Rank-1={cmc[0]*100:.2f}%  '
                f'Rank-5={cmc[4]*100:.2f}%')
    return mAP


def train_one_epoch(model, loader, ce_loss, tri_loss, opt, scaler, epoch,
                    logger, clip_norm,
                    part_w: float, div_w: float, tri_on_concat: bool):
    model.train()
    t0 = time.time()
    tot_loss = tot_id = tot_tri = tot_div = 0.0
    n_batch = 0
    for batch in loader:
        img = batch['images'].cuda(non_blocking=True)
        target = batch['targets'].cuda(non_blocking=True)
        opt.zero_grad()
        with amp.autocast(enabled=True):
            scores, feats, parts, queries, _ = model(img)
            # ID loss: CE on every head (CLS + parts)
            id_losses = []
            for i, s in enumerate(scores):
                w = 1.0 if i == 0 else part_w
                id_losses.append(w * ce_loss(s, target))
            id_loss = sum(id_losses)

            # Per-stream triplet: each of CLS+parts gets its own metric loss.
            # Concatenated triplet collapses to 0 because the 5120-d concat
            # is trivially separable; per-stream stays informative.
            tri_streams = [tri_loss(f, target)[0] for f in feats]
            tri_per_stream = sum(tri_streams) / len(tri_streams)
            if tri_on_concat:
                # Light extra concat triplet for retrieval coherence
                tri_concat = tri_loss(torch.cat(feats, dim=-1), target)[0]
                tri = tri_per_stream + 0.3 * tri_concat
            else:
                tri = tri_per_stream

            # Per-image diversity: parts (B,P,D) - actually penalizes collapse
            div = perimage_diversity_loss(parts)
            loss = id_loss + tri + div_w * div
        scaler.scale(loss).backward()
        if clip_norm > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scaler.step(opt); scaler.update()
        tot_loss += float(loss.item()); tot_id += float(id_loss.item())
        tot_tri += float(tri.item()); tot_div += float(div.item())
        n_batch += 1
    dt = time.time() - t0
    logger.info(f'Epoch {epoch}: loss={tot_loss/n_batch:.4f}  '
                f'id={tot_id/n_batch:.4f}  tri={tot_tri/n_batch:.4f}  '
                f'div={tot_div/n_batch:.4f}  time={dt:.1f}s ({n_batch} batches)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', cfg.MODEL.DEVICE_ID)
    out_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(os.path.join(out_dir, 'train.log'))
    logger.info(f'out_dir = {out_dir}')

    train_loader = build_reid_train_loader(cfg)
    num_classes = len(train_loader.dataset.pids)
    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])
    logger.info(f'num_classes={num_classes}')

    model = DinoV3PartReID(num_classes=num_classes, cfg=cfg).cuda()
    logger.info(f'spatial_grid={model.spatial_grid}  num_parts={model.num_parts}  '
                f'embed_dim={model.in_planes}')

    # Optional warm-start
    warm = getattr(cfg.MODEL, 'WARM_START_PATH', '')
    if warm:
        state = torch.load(warm, map_location='cpu')
        if 'state_dict' in state: state = state['state_dict']
        if 'model' in state: state = state['model']
        # Map vanilla DinoV3ReID checkpoint into our part model: bottleneck.*
        # -> bottlenecks.0.*  ; classifier.* -> classifiers.0.*
        new_state = {}
        for k, v in state.items():
            if k.startswith('bottleneck.'):
                new_state['bottlenecks.0.' + k[len('bottleneck.'):]] = v
            elif k.startswith('classifier.'):
                old_n = state['classifier.weight'].shape[0]
                if old_n == num_classes:
                    new_state['classifiers.0.' + k[len('classifier.'):]] = v
                # else drop
            else:
                new_state[k] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        logger.info(f'warm-start: {warm}  missing={len(missing)} unexpected={len(unexpected)}')

    # --- losses ---
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        ce_loss = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        ce_loss = lambda s, t: F.cross_entropy(s, t)
    if cfg.MODEL.NO_MARGIN:
        tri_loss = TripletLoss()
    else:
        tri_loss = TripletLoss(float(cfg.SOLVER.MARGIN))

    part_w = float(cfg.MODEL.PART.PART_LOSS_W)
    div_w  = float(cfg.MODEL.PART.DIVERSITY_W)
    tri_concat = bool(cfg.MODEL.PART.TRIPLET_ON_CONCAT)
    logger.info(f'part_loss_w={part_w}  diversity_w={div_w}  '
                f'triplet_on_concat={tri_concat}')

    # --- staged training (mirrors train_dinov3.py) ---
    s1, s2, s3 = cfg.SOLVER.STAGE1_EPOCHS, cfg.SOLVER.STAGE2_EPOCHS, cfg.SOLVER.STAGE3_EPOCHS
    total_epochs = s1 + s2 + s3
    assert total_epochs > 0
    stage_specs = [(1, s1), (2, s2), (3, s3)]
    lr_scales = [cfg.SOLVER.STAGE1_LR_SCALE, cfg.SOLVER.STAGE2_LR_SCALE, cfg.SOLVER.STAGE3_LR_SCALE]

    best_map = 0.0
    epoch_global = 0
    for (stage_idx, ep_count), lr_sc in zip(stage_specs, lr_scales):
        if ep_count <= 0: continue
        logger.info(f'==== STAGE {stage_idx}: epochs={ep_count}  lr_scale={lr_sc} ====')
        set_stage(model, stage_idx, cfg.SOLVER.STAGE2_UNFREEZE_BLOCKS)
        opt, n_params = make_optimizer(cfg, model, lr_sc)
        logger.info(f'optimizer set; trainable params={n_params/1e6:.2f}M')
        sched = CosineAnnealingLR(opt, T_max=ep_count)
        warmup = cfg.SOLVER.WARMUP_EPOCHS
        scaler = amp.GradScaler()
        for ep in range(ep_count):
            epoch_global += 1
            # linear warmup within the stage
            if warmup > 0 and ep < warmup:
                lr = (cfg.SOLVER.BASE_LR * lr_sc) * (ep + 1) / warmup
                for pg in opt.param_groups: pg['lr'] = lr
            train_one_epoch(model, train_loader, ce_loss, tri_loss, opt, scaler,
                            epoch_global, logger, cfg.SOLVER.CLIP_GRAD_NORM,
                            part_w=part_w, div_w=div_w, tri_on_concat=tri_concat)
            if ep >= warmup: sched.step()
            if epoch_global % cfg.SOLVER.EVAL_PERIOD == 0:
                m = evaluate(model, val_loader, num_query, logger)
                if m > best_map:
                    best_map = m
                    torch.save(model.state_dict(), os.path.join(out_dir, 'dinov3_best.pth'))
                    logger.info(f'  -> new best mAP={m*100:.2f}% (saved best)')
            if epoch_global % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                torch.save(model.state_dict(),
                           os.path.join(out_dir, f'dinov3_ep{epoch_global}.pth'))

    torch.save(model.state_dict(), os.path.join(out_dir, 'dinov3_final.pth'))
    logger.info(f'Done. best mAP={best_map*100:.2f}%')


if __name__ == '__main__':
    main()
