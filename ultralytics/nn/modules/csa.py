# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Center-Surround Attention (CSA) — bio-inspired small-bright-blob attention.
==========================================================================
Drop-in module for this fork. Goes in:  ultralytics/nn/modules/csa.py

Motivation
----------
The retina's first processing stage uses *center-surround* receptive fields:
a neuron fires for "bright center, dark surround", which is mathematically a
Difference-of-Gaussians (DoG / Laplacian-of-Gaussian) — the classic *blob
detector*. This is what lets a pigeon pop a microcalcification, or a raptor pop
a bright fish out of choppy water: small, high-contrast blobs on a busy
background. CSA bakes that prior into a learnable attention block.

Unlike CBAM (plain 7x7 conv for its spatial branch), CSA's spatial branch is a
bank of multi-scale DoG kernels, initialised to real DoG values then trained.

Design choices
--------------
* Channel-preserving (out C == in C) -> wraps cleanly after a C3k2 block,
  exactly like this fork's `_WMA` inside C3k2_WMA.
* The DoG prior is ACTIVE at init (not rediscovered by SGD), yet the block
  starts near-identity so it doesn't wreck a warm-started backbone.
* Mirrors the C3k2_WMA / C3k_WMA wrapper pattern so it slots into the same
  placement-study YAML positions.

This is a prior to TEST, not magic: always ablate vs the same model without CSA
(yolo26_A0_baseline) and report mAP-small.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k, C3k2

__all__ = ("CenterSurroundAttention", "C3k_CSA", "C3k2_CSA")


def _gaussian_kernel2d(sigma: float, kernel_size=None, max_size: int = 13):
    """Normalised 2D Gaussian kernel + its (odd) size (capped at max_size)."""
    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    kernel_size = min(kernel_size, max_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32) - half
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :], kernel_size


class CenterSurroundAttention(nn.Module):
    """Multi-scale center-surround (DoG) spatial attention + SE channel attention.

    Args:
        c1 (int):            input == output channels.
        probe (int):         number of 'retinal' probe maps features are reduced to.
        sigmas (tuple):      center scales in feature-map pixels (one DoG bank each).
        surround_ratio (f):  surround sigma = center sigma * surround_ratio (>1).
        reduction (int):     SE channel-attention reduction.
        bright_only (bool):  True keeps on-center (bright-blob) responses (ReLU);
                             False uses |response| (dark blobs count too).
    """

    def __init__(self, c1, probe=8, sigmas=(1.0, 2.0, 3.0), surround_ratio=1.6,
                 reduction=16, bright_only=True):
        super().__init__()
        self.c1 = c1
        self.bright_only = bright_only
        n = len(sigmas)

        # 'retinal' projection: compress C channels -> a few probe maps
        self.reduce = nn.Conv2d(c1, probe, kernel_size=1, bias=True)

        # one depthwise DoG conv per scale, initialised to a real center-surround kernel
        self.dog_convs = nn.ModuleList()
        for s in sigmas:
            gc, k = _gaussian_kernel2d(s)
            gs, _ = _gaussian_kernel2d(s * surround_ratio, kernel_size=k)
            dog = gc - gs                 # center - surround
            dog = dog - dog.mean()        # zero DC: flat regions -> ~0 response
            conv = nn.Conv2d(probe, probe, kernel_size=k, groups=probe,
                             padding=k // 2, bias=False)
            with torch.no_grad():
                conv.weight.copy_(dog[None, None].repeat(probe, 1, 1, 1))
            self.dog_convs.append(conv)

        # fuse all (scale x probe) blob responses -> single spatial attention map
        self.fuse = nn.Conv2d(probe * n, 1, kernel_size=1, bias=True)

        # SE-style channel attention
        hidden = max(c1 // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=True),
            nn.Sigmoid(),
        )

        # Active DoG prior + near-identity start:
        #   fuse(r) = gain * mean_channels(r) + bias
        #   flat -> sigmoid(bias) ~ 0.05 (near identity); blobs -> attention rises
        gain = 12.0
        nn.init.constant_(self.fuse.weight, gain / (probe * n))
        nn.init.constant_(self.fuse.bias, -3.0)

    def forward(self, x, return_attn: bool = False):
        p = self.reduce(x)                                   # B, probe, H, W
        responses = []
        for conv in self.dog_convs:
            r = conv(p)
            r = F.relu(r) if self.bright_only else r.abs()   # on-center / both polarities
            responses.append(r)
        r = torch.cat(responses, dim=1)                      # B, probe*n, H, W
        a = torch.sigmoid(self.fuse(r))                      # B, 1, H, W  (spatial)
        wc = self.se(self.gap(x))                            # B, C, 1, 1  (channel)
        out = x * (1.0 + a * wc)                             # residual gating
        return (out, a) if return_attn else out


class C3k_CSA(C3k):
    """C3k variant: center-surround attention applied after the C3k output.

    YAML args identical to C3k: [c2, shortcut, g, e, k]
    (Provided for parity with C3k_WMA; the head YAML uses C3k2_CSA.)
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3,
                 probe=8, sigmas=(1.0, 2.0, 3.0), surround_ratio=1.6,
                 reduction=16, bright_only=True):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        self.attn_csa = CenterSurroundAttention(
            c2, probe=probe, sigmas=sigmas, surround_ratio=surround_ratio,
            reduction=reduction, bright_only=bright_only,
        )

    def forward(self, x):
        return self.attn_csa(super().forward(x))


class C3k2_CSA(C3k2):
    """C3k2 variant: center-surround attention applied after the C3k2 output.

    YAML args identical to C3k2: [c2, c3k, e, attn, g, shortcut]
    The inner `attn` flag keeps its C3k2 meaning (PSABlock on/off); the outer
    center-surround attention is always applied. Mirrors C3k2_WMA exactly, so it
    can be swapped into any position where you'd place C3k2_WMA.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True,
                 probe=8, sigmas=(1.0, 2.0, 3.0), surround_ratio=1.6,
                 reduction=16, bright_only=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        # `attn_csa` avoids clashing with the inner C3k2 `attn` flag/attribute
        self.attn_csa = CenterSurroundAttention(
            c2, probe=probe, sigmas=sigmas, surround_ratio=surround_ratio,
            reduction=reduction, bright_only=bright_only,
        )

    def forward(self, x):
        return self.attn_csa(super().forward(x))
