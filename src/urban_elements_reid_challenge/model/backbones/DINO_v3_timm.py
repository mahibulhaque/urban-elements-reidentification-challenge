"""DINOv3 ViT backbone wrapper.

Uses timm (>=1.0.20) which supports facebook/dinov3-* checkpoints. The loaded
timm model's forward returns the final CLS token (num_classes=0). We also
expose `forward_allblocks` for the last-N CLS experiment.
"""
import torch
import torch.nn as nn
from torch.autograd import Function

try:
    import timm  # type: ignore
except ImportError as e:
    raise RuntimeError("timm is required for DINOv3 backbone") from e


class _GradReverse(Function):
    """Gradient Reversal Layer (DANN, Ganin & Lempitsky 2015).

    Forward: identity. Backward: scale incoming gradient by -lambda.
    Used to make a feature extractor *minimize* its ability to predict a
    domain/camera label, so the features become invariant to that label.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return _GradReverse.apply(x, lambda_)


def _enable_lora_only(timm_vit):
    """Mark only LoRA A/B as trainable inside a (possibly LoRA-wrapped) ViT."""
    from .lora import LoRALinear
    for m in timm_vit.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True


class DinoV3Backbone(nn.Module):
    def __init__(self, timm_name: str, pretrained_path: str = "", img_size=None):
        super().__init__()
        # Build without downloading when we have a local checkpoint
        use_pretrained = not bool(pretrained_path)
        # Pass img_size only if specified — needed for models that strictly
        # check input size (e.g., SigLIP). DINOv3 models accept arbitrary sizes
        # via RoPE so img_size kwarg is harmless there.
        kwargs = dict(pretrained=use_pretrained, num_classes=0)
        if img_size is not None:
            kwargs['img_size'] = tuple(img_size)
        self.model = timm.create_model(timm_name, **kwargs)
        self.num_features = self.model.num_features
        if pretrained_path:
            state = torch.load(pretrained_path, map_location='cpu')
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if unexpected:
                print(f'[dinov3] unexpected keys (first 3): {unexpected[:3]}')
            if missing:
                print(f'[dinov3] missing keys (first 3): {missing[:3]}')
            print(f'[dinov3] loaded weights from {pretrained_path}')

    def forward(self, x):
        """Return final CLS token, shape (B, C)."""
        return self.model(x)

    @torch.no_grad()
    def forward_allblocks(self, x):
        """Return per-block CLS tokens, shape (B, depth, C).

        timm's forward_intermediates zips patches with prefix tokens:
            list[i] = (patch_tokens, prefix_tokens)
        where prefix holds CLS at index 0 (followed by registers for DINOv3).
        """
        out = self.model.forward_intermediates(
            x, indices=None, return_prefix_tokens=True,
            norm=True, stop_early=False, output_fmt='NLC',
            intermediates_only=False,
        )
        _, intermediates = out
        cls_per_block = []
        for item in intermediates:
            _, prefix = item  # patches, prefix
            cls_per_block.append(prefix[:, 0])
        return torch.stack(cls_per_block, dim=1)  # (B, depth, C)

    def _flat_blocks(self):
        """Flat list of all transformer/conv blocks across the backbone, in
        forward order. Handles ViT (.blocks), ConvNeXt (.stages[i].blocks),
        and Swin-V2 (.layers[i].blocks).
        """
        m = self.model
        if hasattr(m, 'blocks') and isinstance(m.blocks, (nn.Sequential, nn.ModuleList)):
            return list(m.blocks)
        for attr in ('stages', 'layers'):
            container = getattr(m, attr, None)
            if container is None: continue
            blocks = []
            for stage in container:
                if hasattr(stage, 'blocks'):
                    blocks.extend(list(stage.blocks))
            if blocks: return blocks
        return [m]   # fallback: treat whole model as one block

    def _trailing_norms(self):
        """Norms / pre-pool layers that come after the last block — must
        stay trainable when the last-N blocks are unfrozen."""
        m = self.model
        out = []
        for name in ('norm', 'norm_pre', 'fc_norm'):
            mod = getattr(m, name, None)
            if mod is not None and not isinstance(mod, nn.Identity):
                out.append(mod)
        return out

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int):
        self.freeze_all()
        blocks = self._flat_blocks()
        total = len(blocks)
        n = min(max(n, 0), total)
        for blk in blocks[total - n:]:
            for p in blk.parameters():
                p.requires_grad = True
        for norm in self._trailing_norms():
            for p in norm.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


class DinoV3ReID(nn.Module):
    """DINOv3 backbone + BN bottleneck + linear classifier (standard ReID head).

    Optional camera-adversarial head: a small MLP that predicts the camera
    label from the (pre-BN) feature, with a Gradient Reversal Layer in front.
    The CE loss on this head is added to the main loss with a positive sign,
    but GRL flips its gradient to the backbone — pushing the backbone to make
    features that are *un-predictive* of camera identity. Useful when val/test
    cameras differ (DANN-style domain adversarial training).
    """
    def __init__(self, num_classes: int, cfg, num_cameras: int = 0):
        super().__init__()
        self.cfg = cfg
        self.neck_feat = cfg.TEST.NECK_FEAT  # 'before' | 'after'
        timm_name = cfg.MODEL.DINOV3_VARIANT
        # SigLIP/CLIP/Swin-V2 have learned (non-RoPE) pos-embeds or window
        # attention with fixed grids — must pass img_size so timm builds the
        # right pos-embed / window mask for our non-default input shape.
        needs_imgsize = any(k in timm_name.lower() for k in ('siglip', 'clip', 'swinv2'))
        img_size = tuple(cfg.INPUT.SIZE_TRAIN) if needs_imgsize else None
        self.base = DinoV3Backbone(timm_name, pretrained_path=cfg.MODEL.PRETRAIN_PATH,
                                   img_size=img_size)

        # Optional LoRA: wrap last-N blocks, freeze the rest of the backbone.
        # When enabled, base.freeze_all/unfreeze_* must keep LoRA params hot.
        lora_cfg = getattr(cfg.MODEL, 'LORA', None)
        self._lora_enabled = bool(lora_cfg is not None and getattr(lora_cfg, 'ENABLED', False))
        if self._lora_enabled:
            from .lora import wrap_lora
            wrap_lora(self.base.model,
                      last_n_blocks=int(lora_cfg.LAST_N_BLOCKS),
                      r=int(lora_cfg.R),
                      alpha=int(lora_cfg.ALPHA),
                      targets=tuple(lora_cfg.TARGETS))
            # Patch backbone freeze methods so the trainer's stage-3 unfreeze
            # only re-enables LoRA params (originals stay frozen).
            base_obj = self.base
            def _lora_freeze_all(_self=base_obj):
                for p in _self.model.parameters(): p.requires_grad = False
                _enable_lora_only(_self.model)
            def _lora_unfreeze_last_n(n, _self=base_obj):
                _lora_freeze_all()
            def _lora_unfreeze_all(_self=base_obj):
                _lora_freeze_all()
            base_obj.freeze_all = _lora_freeze_all
            base_obj.unfreeze_last_n_blocks = _lora_unfreeze_last_n
            base_obj.unfreeze_all = _lora_unfreeze_all
            _enable_lora_only(self.base.model)
        self.in_planes = self.base.num_features
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)

        if num_classes > 0:
            self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)
        else:
            self.classifier = None

        # Camera adversarial head (DANN-style with GRL)
        self.num_cameras = num_cameras
        cam_cfg = getattr(cfg.MODEL, 'CAM_ADV', None)
        self.cam_lambda = float(cam_cfg.LAMBDA) if (cam_cfg is not None and cam_cfg.ENABLED and num_cameras > 0) else 0.0
        if self.cam_lambda > 0 and num_cameras > 0:
            hidden = int(getattr(cam_cfg, 'HIDDEN', 256))
            self.cam_classifier = nn.Sequential(
                nn.Linear(self.in_planes, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden, num_cameras),
            )
        else:
            self.cam_classifier = None

    def forward(self, x):
        feat = self.base(x)  # (B, C) — final CLS (already norm'd by timm)
        feat_bn = self.bottleneck(feat)
        if self.training:
            assert self.classifier is not None, 'num_classes > 0 required for training'
            score = self.classifier(feat_bn)
            cam_score = None
            if self.cam_classifier is not None:
                feat_rev = grad_reverse(feat, self.cam_lambda)
                cam_score = self.cam_classifier(feat_rev)
            if cam_score is not None:
                return score, feat, cam_score
            return score, feat
        else:
            return feat_bn if self.neck_feat == 'after' else feat

    # --- param-group helpers for staged fine-tuning ---
    def head_params(self):
        ps = list(self.bottleneck.parameters())
        if self.classifier is not None:
            ps += list(self.classifier.parameters())
        return [p for p in ps if p.requires_grad]

    def backbone_params(self):
        return [p for p in self.base.parameters() if p.requires_grad]

    def load_checkpoint(self, path: str, strict: bool = False):
        state = torch.load(path, map_location='cpu')
        if 'state_dict' in state: state = state['state_dict']
        if 'model' in state: state = state['model']
        missing, unexpected = self.load_state_dict(state, strict=strict)
        print(f'[dinov3_reid] loaded {path}  (missing={len(missing)}, unexpected={len(unexpected)})')
