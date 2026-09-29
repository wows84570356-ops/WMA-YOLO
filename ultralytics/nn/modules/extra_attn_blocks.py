# Ultralytics 🚀 AGPL-3.0 License
"""
C3k2 + baseline attention wrappers — for the WMA vs. baseline attention ablation.

Each wrapper subclasses C3k2 and applies a baseline attention module to the
output of C3k2.forward(), exactly like C3k2_WMA does. This makes the
comparison fair: WMA, CBAM, SE, GAM, GCT, ShuffleAttention, GlobalContext,
GatherExcite are all applied at the SAME position (after C3k2) with the
SAME structural pattern.

YAML signature matches C3k2/C3k2_WMA exactly:
    [-1, n, C3k2_CBAM, [c2, c3k, e, attn]]
where c3k/e/attn are optional (defaults: False, 0.5, False).
"""

from .block import C3k2
from .conv import (
    CBAM,
    ChannelAttention,
    GAM_Attention,
    GCT,
    ShuffleAttention,
    GlobalContext,
    GatherExcite,
)


class _C3k2_AttnBase(C3k2):
    """Base class: C3k2 followed by a subclass-defined attention module."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.attn_extra = self._make_attn(c2)

    def _make_attn(self, c2):
        raise NotImplementedError

    def forward(self, x):
        return self.attn_extra(super().forward(x))


class C3k2_CBAM(_C3k2_AttnBase):
    """C3k2 -> CBAM (channel + spatial attention, kernel=7)."""
    def _make_attn(self, c2):
        return CBAM(c2, kernel_size=7)


class C3k2_SE(_C3k2_AttnBase):
    """C3k2 -> SE-style channel attention (via ChannelAttention)."""
    def _make_attn(self, c2):
        return ChannelAttention(c2)


class C3k2_GAM(_C3k2_AttnBase):
    """C3k2 -> GAM (Global Attention Mechanism)."""
    def _make_attn(self, c2):
        return GAM_Attention(c2, c2, group=True, rate=4)


class C3k2_GCT(_C3k2_AttnBase):
    """C3k2 -> GCT (Gated Channel Transformation)."""
    def _make_attn(self, c2):
        return GCT(c2)


class C3k2_SA(_C3k2_AttnBase):
    """C3k2 -> ShuffleAttention. Needs c2 % 16 == 0 (G=8, *2 = 16).
    At YOLO26n scale (width=0.25) the smallest c2 is 64, which works."""
    def _make_attn(self, c2):
        return ShuffleAttention(c2, reduction=16, G=8)


class C3k2_GC(_C3k2_AttnBase):
    """C3k2 -> GlobalContext (timm-style GC block)."""
    def _make_attn(self, c2):
        return GlobalContext(c2)


class C3k2_GE(_C3k2_AttnBase):
    """C3k2 -> Gather-Excite (timm-style GE block)."""
    def _make_attn(self, c2):
        return GatherExcite(c2)


__all__ = (
    "C3k2_CBAM", "C3k2_SE", "C3k2_GAM", "C3k2_GCT",
    "C3k2_SA", "C3k2_GC", "C3k2_GE",
)
