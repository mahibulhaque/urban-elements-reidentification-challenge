"""Part-Aware DINOv3 ReID head (PartDV3).

Inspired by PAT (Part-Attention Transformer for Person ReID): take the
backbone's patch tokens, run cross-attention with a small set of learnable
part queries — each query biased toward a different spatial region of the
14x7 patch grid — and produce P part-embeddings alongside the CLS token.

Each part embedding gets its own BN bottleneck + ID classifier (trained
jointly with the CLS head), forcing the parts to be individually
discriminative. A diversity penalty on the part queries prevents collapse.

Test-time retrieval feature is the L2-normalized concatenation of CLS plus
the P part embeddings (5 * 1024 = 5120-d for DINOv3-L, 5 * 1280 = 6400-d
for H+).
"""
from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .DINO_v3_timm import DinoV3Backbone, grad_reverse


def _grid_gauss_bias(num_parts: int, H: int, W: int,
                     centers=((0.25, 0.5), (0.75, 0.5),
                              (0.5, 0.25), (0.5, 0.75)),
                     sigma2: float = 0.1) -> torch.Tensor:
    """(P, H*W) Gaussian bias logits — soft prior favoring different regions."""
    ys = torch.linspace(0, 1, H)
    xs = torch.linspace(0, 1, W)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    bias = torch.zeros(num_parts, H, W)
    for i in range(num_parts):
        cy, cx = centers[i % len(centers)]
        bias[i] = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / sigma2)
    return bias.view(num_parts, H * W) * 2.0   # ~2.0 magnitude added to logits


class PartAttentionHead(nn.Module):
    """Single-layer multi-head cross-attention with learnable part queries.

    Output: (B, P, D)  — P part embeddings per image.
    """
    def __init__(self, dim: int, num_parts: int = 4, num_heads: int = 8,
                 spatial_grid: Tuple[int, int] = (14, 7),
                 init_spatial_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} not divisible by heads {num_heads}'
        self.num_parts = num_parts
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.spatial_grid = spatial_grid
        L = spatial_grid[0] * spatial_grid[1]

        # Learnable part queries (P, D)
        self.queries = nn.Parameter(torch.empty(num_parts, dim))
        nn.init.normal_(self.queries, std=0.02)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        # Spatial bias logits (P, L) — added to attention scores, broadcast
        # over heads. Initialized to encourage region specialization.
        if init_spatial_bias:
            sb = _grid_gauss_bias(num_parts, *spatial_grid)
        else:
            sb = torch.zeros(num_parts, L)
        self.spatial_bias = nn.Parameter(sb)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, L, D) -> parts (B, P, D)."""
        B, L, D = patches.shape
        H = self.num_heads; d = self.head_dim
        P = self.num_parts

        q_in = self.norm_q(self.queries)                     # (P, D)
        kv_in = self.norm_kv(patches)                        # (B, L, D)

        q = self.q_proj(q_in).view(P, H, d)                  # (P, h, d)
        kv = self.kv_proj(kv_in).view(B, L, 2, H, d)
        k, v = kv.unbind(dim=2)                              # each (B, L, H, d)

        # to (B, h, *, d)
        q = q.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2)   # (B, h, P, d)
        k = k.transpose(1, 2)                                # (B, h, L, d)
        v = v.transpose(1, 2)                                # (B, h, L, d)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, h, P, L)
        attn = attn + self.spatial_bias.view(1, 1, P, L)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)                          # (B, h, P, d)
        out = out.transpose(1, 2).reshape(B, P, D)           # (B, P, D)
        out = self.out_proj(out)
        out = self.norm_out(out)
        return out, attn   # also return attn for logging


class DinoV3PartReID(nn.Module):
    """DINOv3 backbone + Part-Attention Head + per-part BN/classifier heads."""

    def __init__(self, num_classes: int, cfg, num_cameras: int = 0):
        super().__init__()
        self.cfg = cfg
        self.neck_feat = cfg.TEST.NECK_FEAT
        timm_name = cfg.MODEL.DINOV3_VARIANT
        needs_imgsize = any(k in timm_name.lower()
                            for k in ('siglip', 'clip', 'swinv2'))
        img_size = tuple(cfg.INPUT.SIZE_TRAIN) if needs_imgsize else None
        self.base = DinoV3Backbone(timm_name,
                                   pretrained_path=cfg.MODEL.PRETRAIN_PATH,
                                   img_size=img_size)
        self.in_planes = self.base.num_features
        self.num_classes = num_classes

        # Spatial grid for the part head — derived from input size & patch=16.
        H, W = cfg.INPUT.SIZE_TRAIN
        ph = pw = 16
        if 'patch14' in timm_name: ph = pw = 14
        gh, gw = H // ph, W // pw
        self.spatial_grid = (gh, gw)

        # Part config (defaults safe if cfg.MODEL.PART not set)
        part_cfg = getattr(cfg.MODEL, 'PART', None)
        self.num_parts = int(getattr(part_cfg, 'NUM', 4)) if part_cfg else 4
        nheads = int(getattr(part_cfg, 'HEADS', 8)) if part_cfg else 8
        # Auto-fix nheads if it doesn't divide dim
        while self.in_planes % nheads != 0 and nheads > 1:
            nheads //= 2

        self.part_head = PartAttentionHead(
            dim=self.in_planes, num_parts=self.num_parts,
            num_heads=nheads, spatial_grid=self.spatial_grid,
            init_spatial_bias=True, dropout=0.0,
        )

        # One BN bottleneck + classifier per stream (CLS + P parts)
        self.bottlenecks = nn.ModuleList([
            nn.BatchNorm1d(self.in_planes) for _ in range(self.num_parts + 1)
        ])
        for bn in self.bottlenecks:
            bn.bias.requires_grad_(False)
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)
        if num_classes > 0:
            self.classifiers = nn.ModuleList([
                nn.Linear(self.in_planes, num_classes, bias=False)
                for _ in range(self.num_parts + 1)
            ])
            for cl in self.classifiers:
                nn.init.normal_(cl.weight, std=0.001)
        else:
            self.classifiers = None

        # Camera-adversarial head (optional, kept for compatibility)
        self.num_cameras = num_cameras
        cam_cfg = getattr(cfg.MODEL, 'CAM_ADV', None)
        self.cam_lambda = (float(cam_cfg.LAMBDA)
                           if (cam_cfg is not None and cam_cfg.ENABLED and num_cameras > 0)
                           else 0.0)
        if self.cam_lambda > 0 and num_cameras > 0:
            hidden = int(getattr(cam_cfg, 'HIDDEN', 256))
            self.cam_classifier = nn.Sequential(
                nn.Linear(self.in_planes * (self.num_parts + 1), hidden),
                nn.ReLU(inplace=True), nn.Dropout(0.1),
                nn.Linear(hidden, num_cameras),
            )
        else:
            self.cam_classifier = None

    # --- backbone forward → CLS + patches ---
    def _backbone_tokens(self, x):
        """Return (cls_post_norm, patches_post_norm)."""
        m = self.base.model
        # Use forward_intermediates to grab last-block prefix + patches.
        res = m.forward_intermediates(
            x, indices=1, return_prefix_tokens=True,
            norm=True, stop_early=False, output_fmt='NLC',
            intermediates_only=True,
        )
        patches, prefix = res[-1]      # (B, L, D), (B, P_pref, D)
        # Apply fc_norm if present (DINOv3 = Identity; harmless elsewhere)
        fc_norm = getattr(m, 'fc_norm', None)
        if fc_norm is not None and not isinstance(fc_norm, nn.Identity):
            prefix = fc_norm(prefix); patches = fc_norm(patches)
        cls = prefix[:, 0]
        return cls, patches

    def forward(self, x):
        cls, patches = self._backbone_tokens(x)        # (B, D), (B, L, D)
        parts, attn = self.part_head(patches)          # (B, P, D)
        feats = [cls] + [parts[:, i] for i in range(self.num_parts)]   # list of (B, D)

        bn_feats = [self.bottlenecks[i](f) for i, f in enumerate(feats)]

        if self.training:
            assert self.classifiers is not None
            scores = [self.classifiers[i](bf) for i, bf in enumerate(bn_feats)]
            cam_score = None
            if self.cam_classifier is not None:
                cat_pre = torch.cat(feats, dim=-1)
                cat_rev = grad_reverse(cat_pre, self.cam_lambda)
                cam_score = self.cam_classifier(cat_rev)
            # Return: scores list, feats list (for triplet),
            # parts (B, P, D) for per-image diversity, queries (kept for compat),
            # cam_score
            return scores, feats, parts, self.part_head.queries, cam_score
        else:
            # Concat retrieval feature: pre-BN if NECK_FEAT='before', post-BN if 'after'
            use = bn_feats if self.neck_feat == 'after' else feats
            return torch.cat(use, dim=-1)               # (B, (P+1)*D)

    # --- param-group helpers (used by trainer) ---
    def head_params(self):
        ps = list(self.part_head.parameters())
        for bn in self.bottlenecks:
            ps += list(bn.parameters())
        if self.classifiers is not None:
            for cl in self.classifiers:
                ps += list(cl.parameters())
        if self.cam_classifier is not None:
            ps += list(self.cam_classifier.parameters())
        return [p for p in ps if p.requires_grad]

    def backbone_params(self):
        return [p for p in self.base.parameters() if p.requires_grad]

    def load_checkpoint(self, path: str, strict: bool = False):
        state = torch.load(path, map_location='cpu')
        if 'state_dict' in state: state = state['state_dict']
        if 'model' in state: state = state['model']
        missing, unexpected = self.load_state_dict(state, strict=strict)
        print(f'[part_dv3] loaded {path}  (missing={len(missing)}, unexpected={len(unexpected)})')


def diversity_loss(queries: torch.Tensor) -> torch.Tensor:
    """Query-side orthogonality. Saturates immediately at random init in
    high dim, so use per-image diversity in real training."""
    P = queries.shape[0]
    qn = F.normalize(queries, dim=-1)
    g = qn @ qn.t()
    I = torch.eye(P, device=g.device, dtype=g.dtype)
    return ((g - I) ** 2).sum() / (P * P)


def perimage_diversity_loss(parts: torch.Tensor) -> torch.Tensor:
    """Per-image part orthogonality.
       parts: (B, P, D). Forces each image's P part outputs to be cosine-
       orthogonal — actually fights collapse during training, unlike the
       query-only version which is ~constant."""
    B, P, _ = parts.shape
    pn = F.normalize(parts, dim=-1)                   # (B, P, D)
    g = torch.matmul(pn, pn.transpose(-2, -1))        # (B, P, P)
    I = torch.eye(P, device=g.device, dtype=g.dtype).expand(B, -1, -1)
    return ((g - I) ** 2).sum(dim=(-2, -1)).mean() / (P * P)
