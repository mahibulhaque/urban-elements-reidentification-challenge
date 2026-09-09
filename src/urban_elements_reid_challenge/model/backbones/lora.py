"""Minimal LoRA implementation: low-rank delta added to a frozen Linear.

  y = W x + b + scaling * B (A x),  scaling = alpha / r
  A: (r, in_f) Kaiming-init,  B: (out_f, r) zero-init  -> initial delta = 0.

Wrapping a model: walk the timm ViT, replace `attn.qkv`/`attn.proj` (and
optionally `mlp.fc1`/`mlp.fc2`) on the last N blocks with LoRALinear; freeze
all original parameters; return the list of trainable adapter params.
"""
from typing import Iterable, List
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        assert r > 0
        self.in_f = base.in_features
        self.out_f = base.out_features
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, self.in_f))
        self.B = nn.Parameter(torch.zeros(self.out_f, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        delta = self.dropout(x) @ self.A.t()
        delta = delta @ self.B.t()
        return out + self.scaling * delta


def _replace_linear(parent: nn.Module, attr: str, r: int, alpha: int):
    lin = getattr(parent, attr)
    if not isinstance(lin, nn.Linear):
        return False
    setattr(parent, attr, LoRALinear(lin, r=r, alpha=alpha))
    return True


def wrap_lora(timm_vit: nn.Module, last_n_blocks: int = 6,
              r: int = 8, alpha: int = 16,
              targets: Iterable[str] = ('qkv', 'proj')) -> List[nn.Parameter]:
    """Wrap target Linears in last-N transformer blocks with LoRA. Freezes
    all original params (including non-block params); only LoRA A/B remain
    trainable. Returns the list of trainable parameters.
    """
    for p in timm_vit.parameters():
        p.requires_grad = False

    blocks = timm_vit.blocks
    total = len(blocks)
    n = max(0, min(last_n_blocks, total))
    targets = set(targets)

    n_wrapped = 0
    for i in range(total - n, total):
        blk = blocks[i]
        if hasattr(blk, 'attn'):
            for tgt in ('qkv', 'proj'):
                if tgt in targets and hasattr(blk.attn, tgt):
                    n_wrapped += int(_replace_linear(blk.attn, tgt, r, alpha))
        if hasattr(blk, 'mlp'):
            for tgt in ('fc1', 'fc2'):
                if tgt in targets and hasattr(blk.mlp, tgt):
                    n_wrapped += int(_replace_linear(blk.mlp, tgt, r, alpha))

    trainable = [p for p in timm_vit.parameters() if p.requires_grad]
    print(f'[lora] wrapped {n_wrapped} Linears across last {n}/{total} blocks  '
          f'(r={r}, alpha={alpha}, targets={sorted(targets)})  '
          f'trainable params: {sum(p.numel() for p in trainable):,}')
    return trainable
