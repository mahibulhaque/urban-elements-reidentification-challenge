"""Staged fine-tune trainer for DINOv3 ReID.

Stages (epoch counts from cfg.SOLVER.STAGE{1,2,3}_EPOCHS):
  1. head-only         (backbone frozen)
  2. last-N blocks     (head + cfg.SOLVER.STAGE2_UNFREEZE_BLOCKS blocks)
  3. full backbone     (everything)

Setting STAGE2 or STAGE3 epochs to 0 skips those stages — e.g. phase-1 probe
is STAGE1 only.
"""
import argparse
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from configs import cfg  # noqa: E402
from urban_elements_reid_challenge.data.build_DG_dataloader import build_reid_train_loader, build_reid_test_loader  # noqa: E402
from urban_elements_reid_challenge.losses.build_loss import build_loss  # noqa: E402
from urban_elements_reid_challenge.utils.metrics import R1_mAP_eval  # noqa: E402
from urban_elements_reid_challenge.model.backbones.DINO_v3_timm import DinoV3ReID  # noqa: E402


def setup_logger(log_path):
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path, mode='a'), logging.StreamHandler()])
    return logging.getLogger('DV3')


def set_stage(model, stage, unfreeze_n):
    if stage == 1:
        model.base.freeze_all()
    elif stage == 2:
        model.base.unfreeze_last_n_blocks(unfreeze_n)
    else:  # stage 3
        model.base.unfreeze_all()


def make_optimizer(cfg, model, lr_scale):
    base_lr = cfg.SOLVER.BASE_LR * lr_scale
    wd = cfg.SOLVER.WEIGHT_DECAY
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=wd)
    return opt, n_params


@torch.no_grad()
def evaluate(model, val_loader, num_query, logger, device='cuda'):
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    model.eval()
    for batch in val_loader:
        img = batch['images'].to(device, non_blocking=True)
        vid = batch['targets']
        camid = batch['camid']
        feat = model(img)
        evaluator.update((feat.float(), vid, camid))
    cmc, mAP = evaluator.compute()[:2]
    logger.info(f'Validation: mAP={mAP*100:.2f}%  Rank-1={cmc[0]*100:.2f}%  '
                f'Rank-5={cmc[4]*100:.2f}%')
    return mAP


def train_one_epoch(model, loader, loss_fn, opt, scaler, epoch, logger, clip_norm,
                    center_criterion=None, opt_center=None, center_weight=0.0,
                    cam_id_map=None, cam_loss_weight=0.0):
    model.train()
    t0 = time.time()
    total_loss = 0.0
    total_cam_loss = 0.0
    n_batch = 0
    use_cam_adv = cam_id_map is not None and cam_loss_weight > 0
    for batch in loader:
        img = batch['images'].cuda(non_blocking=True)
        target = batch['targets'].cuda(non_blocking=True)
        cam_target = None
        if use_cam_adv:
            cam_raw = batch['camid']
            if torch.is_tensor(cam_raw): cam_raw = cam_raw.tolist()
            cam_idx = torch.tensor([cam_id_map[int(c)] for c in cam_raw],
                                   dtype=torch.long, device='cuda')
            cam_target = cam_idx

        opt.zero_grad()
        if opt_center is not None:
            opt_center.zero_grad()
        with amp.autocast(enabled=True):
            out = model(img)
            if use_cam_adv and len(out) == 3:
                score, feat, cam_score = out
                cam_loss = nn.functional.cross_entropy(cam_score, cam_target)
            else:
                score, feat = out[:2]
                cam_loss = None
            loss = loss_fn(score, feat, target)
            if center_criterion is not None:
                loss = loss + center_weight * center_criterion(feat, target)
            if cam_loss is not None:
                loss = loss + cam_loss_weight * cam_loss
                total_cam_loss += float(cam_loss.item())
        scaler.scale(loss).backward()
        if clip_norm > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scaler.step(opt)
        if opt_center is not None:
            # Undo center-weight on center grads so the SGD update uses the
            # un-weighted gradient (standard ReID pattern).
            for p in center_criterion.parameters():
                if p.grad is not None and center_weight > 0:
                    p.grad.data *= (1.0 / center_weight)
            opt_center.step()
        scaler.update()
        total_loss += float(loss.item())
        n_batch += 1
    dt = time.time() - t0
    extra = f' cam_loss={total_cam_loss/max(n_batch,1):.4f}' if use_cam_adv else ''
    logger.info(f'Epoch {epoch} done. avg_loss={total_loss/max(n_batch,1):.4f}{extra}  '
                f'time={dt:.1f}s  ({n_batch} batches)')


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
    logger.info(f'cfg =\n{cfg}')

    # --- data ---
    train_loader = build_reid_train_loader(cfg)
    num_classes = len(train_loader.dataset.pids)
    val_loader, num_query = build_reid_test_loader(cfg, cfg.DATASETS.TEST[0])

    # --- camera mapping for camera-adversarial training ---
    cam_id_map = None
    num_cameras = 0
    cam_loss_weight = 0.0
    if cfg.MODEL.CAM_ADV.ENABLED:
        all_cams = sorted({int(rec[2]) for rec in train_loader.dataset.img_items})
        cam_id_map = {c: i for i, c in enumerate(all_cams)}
        num_cameras = len(all_cams)
        cam_loss_weight = float(cfg.MODEL.CAM_ADV.WEIGHT)
        logger.info(f'CAM_ADV: enabled, lambda={cfg.MODEL.CAM_ADV.LAMBDA}, weight={cam_loss_weight}, '
                    f'num_cameras={num_cameras}, mapping={cam_id_map}')

    # --- model + loss ---
    model = DinoV3ReID(num_classes=num_classes, cfg=cfg, num_cameras=num_cameras).cuda()

    # Optional warm-start from a previously trained DinoV3ReID checkpoint.
    # When the new num_classes differs from the checkpoint's classifier shape,
    # we drop the classifier and bottleneck rows to start the head fresh.
    warm = getattr(cfg.MODEL, 'WARM_START_PATH', '')
    if warm:
        state = torch.load(warm, map_location='cpu')
        if 'state_dict' in state: state = state['state_dict']
        if 'model' in state: state = state['model']
        # Strip classifier (and BN bottleneck weight stats if shapes differ)
        old_n = state['classifier.weight'].shape[0] if 'classifier.weight' in state else None
        if old_n is not None and old_n != num_classes:
            for k in list(state.keys()):
                if k.startswith('classifier.'):
                    state.pop(k)
            logger.info(f'warm-start: dropped classifier head ({old_n} -> {num_classes})')
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f'warm-start: loaded {warm}  missing={len(missing)} unexpected={len(unexpected)}')

    loss_fn, center_criterion = build_loss(cfg, num_classes=num_classes,
                                           feat_dim=model.in_planes)
    use_center = (cfg.MODEL.IF_WITH_CENTER == 'yes')
    opt_center = None
    if use_center:
        center_criterion = center_criterion.cuda()
        opt_center = torch.optim.SGD(center_criterion.parameters(),
                                     lr=cfg.SOLVER.CENTER_LR)
        logger.info(f'CenterLoss enabled  weight={cfg.SOLVER.CENTER_LOSS_WEIGHT}  '
                    f'center_lr={cfg.SOLVER.CENTER_LR}')
    logger.info(f'num_classes={num_classes}  embed_dim={model.in_planes}')

    s1, s2, s3 = cfg.SOLVER.STAGE1_EPOCHS, cfg.SOLVER.STAGE2_EPOCHS, cfg.SOLVER.STAGE3_EPOCHS
    total_epochs = s1 + s2 + s3
    assert total_epochs > 0, 'STAGE1+2+3 epochs sum must be > 0'

    lr_scales = [cfg.SOLVER.STAGE1_LR_SCALE, cfg.SOLVER.STAGE2_LR_SCALE, cfg.SOLVER.STAGE3_LR_SCALE]
    stage_epochs = [s1, s2, s3]
    scaler = amp.GradScaler(init_scale=512)
    clip_norm = cfg.SOLVER.CLIP_GRAD_NORM
    best_mAP, best_ep = 0.0, 0
    global_epoch = 0

    for stage_idx, (ep_count, lr_sc) in enumerate(zip(stage_epochs, lr_scales), start=1):
        if ep_count <= 0:
            continue
        logger.info(f'==== STAGE {stage_idx}: epochs={ep_count}  lr_scale={lr_sc} ====')
        set_stage(model, stage_idx, cfg.SOLVER.STAGE2_UNFREEZE_BLOCKS)
        opt, n_params = make_optimizer(cfg, model, lr_sc)
        logger.info(f'trainable params in stage {stage_idx}: {n_params/1e6:.2f}M')
        sched = CosineAnnealingLR(opt, T_max=max(1, ep_count),
                                  eta_min=cfg.SOLVER.BASE_LR * lr_sc * 0.01)
        for local_ep in range(1, ep_count + 1):
            global_epoch += 1
            train_one_epoch(model, train_loader, loss_fn, opt, scaler,
                            global_epoch, logger, clip_norm,
                            center_criterion=center_criterion if use_center else None,
                            opt_center=opt_center,
                            center_weight=cfg.SOLVER.CENTER_LOSS_WEIGHT if use_center else 0.0)
            sched.step()
            if global_epoch % cfg.SOLVER.EVAL_PERIOD == 0 or global_epoch == total_epochs:
                mAP = evaluate(model, val_loader, num_query, logger)
                if mAP > best_mAP:
                    best_mAP, best_ep = float(mAP), global_epoch
                    torch.save(model.state_dict(),
                               os.path.join(out_dir, 'dinov3_best.pth'))
                    logger.info(f'==== new best mAP={best_mAP*100:.2f}%  ep={best_ep} ====')
            if global_epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                torch.save(model.state_dict(),
                           os.path.join(out_dir, f'dinov3_ep{global_epoch}.pth'))

    # final snapshot
    torch.save(model.state_dict(), os.path.join(out_dir, f'dinov3_final.pth'))
    logger.info(f'training done. best_mAP={best_mAP*100:.2f}% at epoch {best_ep}')
    print(f'BEST_EPOCH {best_ep}')  # for sbatch scripts to grep
    print(f'BEST_MAP {best_mAP:.4f}')


if __name__ == '__main__':
    main()
