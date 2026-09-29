# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
    # WMA additions:
    "C3k_WMA",
    "C3k2_WMA",
    # ADG + Sobel (DeepMine-Mamba inspired):
    "AntiDilutionGate",
    "SobelEdgePrior",
    "AdaptiveFusion",
    # Level-wise adaptive fusion (multi-scale global context):
    "LevelContextFusion",
    "ResidualAdd",
    "AntiDilutionGateYOLO",
    # Orientation-aware module (directional strip conv):
    "DirectionalStripBlock",
    # Mamba-inspired multi-directional state-space module:
    "DirectionalSSM",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and addition to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 4 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes image features through unchanged after fusing."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the fused RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, _, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
            v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semseg: bool = True) -> torch.Tensor:
        """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semseg:
            semseg = self.semseg(feat)
            return (p, semseg)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det


# ════════════════════════════════════════════════════════════════════════════
# === WMA Patch — Wavelet-domain Modulation Self-Attention for blocks ====
# ════════════════════════════════════════════════════════════════════════════
# C3k_WMA / C3k2_WMA
# Apply WaveletAttention AFTER the last conv of the block (cv3 for C3k,
# cv2 for C3k2). YAML args mirror the base classes; just swap the name.
#
# Adapted from DMNet (arXiv:2503.10047), §III-A, Fig. 4(c).
# WaveletAttention itself is defined in conv.py.
# ════════════════════════════════════════════════════════════════════════════

from .conv import WaveletAttention as _WMA


class C3k_WMA(C3k):
    """C3k variant: WaveletAttention applied after cv3.

    YAML args identical to C3k: [c2, shortcut, g, e, k]
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3,
                 num_heads=4, reduction=4):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        self.attn = _WMA(c2, num_heads=num_heads, reduction=reduction)

    def forward(self, x):
        return self.attn(super().forward(x))


class C3k2_WMA(C3k2):
    """C3k2 variant: WaveletAttention applied after cv2.

    YAML args identical to C3k2: [c2, c3k, e, attn, g, shortcut]
    The `attn` flag here keeps the same meaning as in C3k2 (inner PSABlock
    on/off); the outer wavelet attention is always applied.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True,
                 num_heads=4, reduction=4):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        # `attn_wma` to avoid clashing with the inner `attn` flag/attribute
        self.attn_wma = _WMA(c2, num_heads=num_heads, reduction=reduction)

    def forward(self, x):
        return self.attn_wma(super().forward(x))

import math


# ════════════════════════════════════════════════════════════════════════════
# === Anti-Dilution Gate (ADG) ===========================================
# ════════════════════════════════════════════════════════════════════════════
# Inspired by DeepMine-Mamba (arXiv:2606.08781) §3.3.
#
# Restores small-object / thin-structure detail diluted during FPN fusion
# (Upsample + Concat + C3k2). Takes the fused feature m and the backbone
# feature x; outputs y = m + I_hat ⊙ (x - m), where I_hat depends on the
# dilution magnitude |x - m| and an objectness prior.
#
# YAML: [[-1, K], 1, AntiDilutionGate, []]
#   -1 = fused branch (m), K = backbone layer index (x).
#   Output channel = m's channel.
# ════════════════════════════════════════════════════════════════════════════


class AntiDilutionGate(nn.Module):
    def __init__(self, c_m: int, c_x: int, alpha: float = 0.5, reduction: int = 4):
        super().__init__()
        self.c_m = int(c_m)
        self.c_x = int(c_x)
        self.alpha = float(alpha)
        hidden = max(self.c_m // int(reduction), 16)

        if self.c_x != self.c_m:
            self.align = nn.Sequential(
                nn.Conv2d(self.c_x, self.c_m, 1, bias=False),
                nn.BatchNorm2d(self.c_m),
                nn.SiLU(inplace=True),
            )
        else:
            self.align = nn.Identity()

        self.detector = nn.Sequential(
            nn.Conv2d(self.c_m * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        self.obj_head = nn.Sequential(
            nn.Conv2d(self.c_m, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

        # Gate-zero init: model starts as identity (y == m), then learns to deviate.
        nn.init.zeros_(self.detector[-1].weight)
        nn.init.constant_(self.detector[-1].bias, -4.0)
        nn.init.zeros_(self.obj_head[-1].weight)
        nn.init.constant_(self.obj_head[-1].bias, -4.0)

    def forward(self, inputs):
        m, x = inputs[0], inputs[1]
        x = self.align(x)
        if x.shape[-2:] != m.shape[-2:]:
            x = F.interpolate(x, size=m.shape[-2:], mode="bilinear", align_corners=False)
        diff = (x - m).abs()
        i_map = torch.sigmoid(self.detector(torch.cat([x, diff], dim=1)))
        p_obj = torch.sigmoid(self.obj_head(m))
        i_hat = self.alpha * p_obj * i_map
        return m + i_hat * (x - m)


# ════════════════════════════════════════════════════════════════════════════
# === Sobel Edge Prior ===================================================
# ════════════════════════════════════════════════════════════════════════════
# Inspired by DeepMine-Mamba (arXiv:2606.08781) §3.1 / §3.4.
#
# Extracts edge-rich features via fixed Sobel filters applied depthwise to
# an early backbone feature, then downsamples + projects to a small embedding.
# Designed to be fused near the small-object detection scale (P3) to
# explicitly inject "thin-edge" prior — analogous to the paper's stroke prior.
#
# YAML: [K, 1, SobelEdgePrior, [c_out, downsample_factor]]
#   K = source layer index (e.g., 1 for P2/4-stride feature)
#   c_out = output channels (will be width-scaled by parse_model)
#   downsample_factor = number of stride-2 steps to reach the target scale
#                       (e.g., 1 to go from stride 4 -> stride 8)
# ════════════════════════════════════════════════════════════════════════════


class SobelEdgePrior(nn.Module):
    """Sobel edge prior with learnable projection.

    Sobel filters are fixed (not trained). The downstream depthwise downsample
    and pointwise projection are learnable.
    """

    def __init__(self, c_in: int, c_out: int = 32, downsample_factor: int = 1):
        super().__init__()
        self.c_in = int(c_in)
        c_out = int(c_out)
        downsample_factor = int(downsample_factor)

        # Fixed Sobel filters, scaled by 1/4 so output stays in input range
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 4.0
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ) / 4.0
        kx = kx.view(1, 1, 3, 3).repeat(self.c_in, 1, 1, 1)
        ky = ky.view(1, 1, 3, 3).repeat(self.c_in, 1, 1, 1)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

        # After Sobel concat we have 2 * c_in channels (Gx + Gy).
        sobel_ch = 2 * self.c_in
        layers = []
        in_ch = sobel_ch

        # Depthwise stride-2 downsample(s) -- very cheap (c * 9 params per layer)
        for _ in range(downsample_factor):
            layers.append(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, groups=in_ch, bias=False)
            )
            layers.append(nn.BatchNorm2d(in_ch))
            layers.append(nn.SiLU(inplace=True))

        # Final 1x1 pointwise projection to c_out
        layers.append(nn.Conv2d(in_ch, c_out, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.SiLU(inplace=True))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, c_in, H, W) -- any backbone feature
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.c_in)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.c_in)
        # Polarity-invariant edge magnitudes. Bridge damage can be dark-on-light
        # (cracks) or light-on-dark (lime / efflorescence) -- we want both to
        # produce strong activations. Without |.|, SiLU in self.head would
        # suppress negative-polarity gradients toward zero.
        edges = torch.cat([gx.abs(), gy.abs()], dim=1)
        return self.head(edges)


# ════════════════════════════════════════════════════════════════════════════
# === Adaptive Fusion (AF) ===============================================
# ════════════════════════════════════════════════════════════════════════════
# Pixel-wise weighted fusion replacing a plain Concat in the FPN.
#
# Plain Concat:           y = cat([up, lat])              (c_up + c_lat ch)
# Adaptive Fusion:        y = cat([w_up * up, w_lat * lat])
#                         where w_up, w_lat are predicted per pixel from
#                         the joint feature via softmax (sum to 1, then * 2
#                         so average = 1.0 = identity at init).
#
# The output channel count equals c_up + c_lat (same as Concat) so the
# downstream C3k2 sees the same channel signature as before — partial
# weight transfer from pretrained checkpoints works unchanged.
#
# Lets the model decide pixel-by-pixel whether to emphasize the upsampled
# high-level (coarse semantic) feature or the lateral low-level (fine
# spatial) feature. Closer in spirit to ASFF than BiFPN — per-pixel rather
# than global scalar weights.
#
# YAML usage (replaces a Concat line):
#     [[-1, K], 1, AdaptiveFusion, []]
# ════════════════════════════════════════════════════════════════════════════


class AdaptiveFusion(nn.Module):
    """Per-pixel adaptive fusion of two features (replaces Concat).

    Args:
        c_up: channel count of the first input (typically upsampled high-level)
        c_lat: channel count of the second input (typically lateral backbone)
        reduction: reduction ratio for the weight predictor's hidden dim
    """

    def __init__(self, c_up: int, c_lat: int, reduction: int = 8):
        super().__init__()
        self.c_up = int(c_up)
        self.c_lat = int(c_lat)
        c_total = self.c_up + self.c_lat
        hidden = max(c_total // int(reduction), 16)

        self.weight_pred = nn.Sequential(
            nn.Conv2d(c_total, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=True),
        )

        # Identity init: weights & bias of last conv = 0
        #  -> logits = 0 -> softmax = [0.5, 0.5] -> *2 -> [1.0, 1.0]
        #  -> y = cat([1.0 * up, 1.0 * lat]) == plain Concat.
        # Model learns to deviate from concat behaviour as training proceeds.
        nn.init.zeros_(self.weight_pred[-1].weight)
        nn.init.zeros_(self.weight_pred[-1].bias)

    def forward(self, inputs):
        up, lat = inputs[0], inputs[1]

        # Safety: bilinear resize if spatial mismatch (shouldn't normally happen)
        if up.shape[-2:] != lat.shape[-2:]:
            up = F.interpolate(up, size=lat.shape[-2:], mode="bilinear", align_corners=False)

        joint = torch.cat([up, lat], dim=1)
        logits = self.weight_pred(joint)              # (B, 2, H, W)
        w = torch.softmax(logits, dim=1)              # weights sum to 1 along ch
        w_up = 2.0 * w[:, 0:1]                        # *2 -> avg weight = 1.0
        w_lat = 2.0 * w[:, 1:2]

        return torch.cat([up * w_up, lat * w_lat], dim=1)


# ════════════════════════════════════════════════════════════════════════════
# === Level-wise Adaptive Fusion =========================================
# ════════════════════════════════════════════════════════════════════════════
# A different mechanism from per-Concat AdaptiveFusion. Operates AFTER the
# entire neck on the multi-scale outputs (P3/P4/P5), producing a single
# global multi-scale summary that is residually added back to each level.
#
# Differs from AdaptiveFusion in three important ways:
#   1. Operates across all neck output levels at once (not within a single
#      Concat). Decides "P3 vs P4 vs P5" importance globally.
#   2. Per-image scalar weights (not per-pixel). Computed by an MLP over the
#      anchor (P3) global-pooled context. The whole image gets one weight
#      vector — coarse but stable.
#   3. Plugged in AFTER the neck, before the Detect head. Pretrained C3k2
#      weights inside the neck are completely untouched.
#
# Usage in YAML (two modules cooperating):
#   1. LevelContextFusion: takes [P3, P4, P5], produces a fused tensor at
#      the anchor (first input) resolution and channel count
#   2. ResidualAdd: takes [P_i, fused], adds the resized+projected fused
#      tensor to P_i as a residual scaled by a learnable alpha (init = 0,
#      so the model starts as identity — pretrained behavior preserved)
#
#   YAML snippet:
#     - [[P3_idx, P4_idx, P5_idx], 1, LevelContextFusion, []]   # fused
#     - [[P3_idx, fused_idx], 1, ResidualAdd, []]   # P3_f
#     - [[P4_idx, fused_idx], 1, ResidualAdd, []]   # P4_f
#     - [[P5_idx, fused_idx], 1, ResidualAdd, []]   # P5_f
#     - [[P3_f, P4_f, P5_f], 1, Detect, [nc]]
# ════════════════════════════════════════════════════════════════════════════


class LevelContextFusion(nn.Module):
    """Cross-level adaptive fusion at the anchor level's spatial size.

    Inputs:
        list of N feature maps at different spatial resolutions; the first
        is the "anchor" and defines the output channel count + spatial size

    Output:
        a single tensor with the anchor's spatial size and channels,
        computed as a softmax-weighted sum of (channel-projected, resized)
        versions of all input levels

    Args:
        c_anchor: channel count of the anchor (first) input
        c_others: list of channel counts for the remaining inputs
        reduction: MLP bottleneck reduction (default 4)
    """

    def __init__(self, c_anchor: int, c_others, reduction: int = 4):
        super().__init__()
        if not isinstance(c_others, (list, tuple)):
            c_others = [c_others]
        self.c_anchor = int(c_anchor)
        self.c_others = [int(c) for c in c_others]
        self.num_levels = 1 + len(self.c_others)

        # 1x1 projections aligning each non-anchor level to the anchor channels
        projs = []
        for c in self.c_others:
            if c == self.c_anchor:
                projs.append(nn.Identity())
            else:
                projs.append(
                    nn.Sequential(
                        nn.Conv2d(c, self.c_anchor, 1, bias=False),
                        nn.BatchNorm2d(self.c_anchor),
                        nn.SiLU(inplace=True),
                    )
                )
        self.projs = nn.ModuleList(projs)

        # MLP from anchor's global context -> per-level scalar logits
        hidden = max(self.c_anchor // int(reduction), 8)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.c_anchor, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.num_levels),
        )

        # Identity-flavoured init: zero out the final layer so softmax starts
        # uniform (1/N each level). The downstream ResidualAdd's alpha=0
        # makes the whole branch start as a no-op anyway, so any starting
        # weight distribution is fine — uniform is just the "fairest" prior.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, inputs):
        anchor = inputs[0]
        others = inputs[1:]
        B, _, H, W = anchor.shape

        # Project + resize others to anchor's channel count and spatial size
        aligned = [anchor]
        for i, f in enumerate(others):
            f = self.projs[i](f)
            if f.shape[-2:] != (H, W):
                f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
            aligned.append(f)

        # Per-image scalar weights from the anchor's global context
        logits = self.mlp(anchor)                           # (B, num_levels)
        weights = torch.softmax(logits, dim=1)              # (B, num_levels)

        # Weighted sum across levels
        fused = sum(
            weights[:, i : i + 1].view(B, 1, 1, 1) * aligned[i]
            for i in range(self.num_levels)
        )
        return fused


class ResidualAdd(nn.Module):
    """Add a (possibly differently-sized and -channelled) residual tensor to
    a main tensor, scaled by a learnable alpha that starts at zero.

    At init: alpha = 0 -> output = main exactly (identity, no disruption to
    pretrained features). The model gradually learns alpha > 0 if the
    residual signal helps.

    Args:
        c_main: channels of the main input
        c_residual: channels of the residual input
    """

    def __init__(self, c_main: int, c_residual: int):
        super().__init__()
        self.c_main = int(c_main)
        self.c_residual = int(c_residual)

        if self.c_main == self.c_residual:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(self.c_residual, self.c_main, 1, bias=False),
                nn.BatchNorm2d(self.c_main),
            )

        # Learnable residual gate, starts at 0 -> identity at init
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        main, residual = inputs[0], inputs[1]
        if residual.shape[-2:] != main.shape[-2:]:
            residual = F.interpolate(residual, size=main.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.proj(residual)
        return main + self.alpha * residual


# ════════════════════════════════════════════════════════════════════════════
# === Anti-Dilution Gate YOLO (cross-level differential gating) ==========
# ════════════════════════════════════════════════════════════════════════════
# Replacement for ResidualAdd in the level-wise adaptive fusion pipeline.
# Paper-faithful implementation of the ADG mechanism, but applied across
# levels in the YOLO neck (instead of within a single FPN top-down step).
#
# Drop-in YAML replacement for ResidualAdd:
#     [[K_main, K_fused], 1, AntiDilutionGateYOLO, []]
#
# Mapping vs the paper (DeepMine-Mamba, Eq. 4):
#     x = inputs[0] = "original" level feature (P_i from PAN output)
#     m = proj_resize(inputs[1]) = "fused" multi-scale feature
#                                   (from LevelContextFusion, projected and
#                                    resized to match P_i's channels + size)
#     y = m + I_hat * (x - m)
#     I_hat = alpha * P_obj * sigmoid(detector([x, |x - m|]))
#     P_obj = sigmoid(prior_conv(x))     <- objectness/edge prior, analog
#                                          of the paper's stroke prior P_s
#
# Identity init (different from paper's gate-closed init -- explained below):
#     Paper trains from random init, where I_hat ~ 0 -> y ~ m is natural.
#     For pretrained YOLO transfer, however, the Detect head was trained
#     on x (the standard PAN output). Starting with y ~ m means the Detect
#     head sees something it has never seen -> loss spikes, wastes epochs.
#     Instead we init with bias=+4 on both final layers -> sigmoid ~ 0.98
#     -> I_hat ~ 0.96 * alpha -> y ~ x. Pretrained Detect sees its expected
#     input. The model gradually learns to decrease I_hat (close the gate
#     and trust m) wherever that helps validation loss.
#
#     To follow the paper exactly: set bias=-4 in __init__ (gate closed,
#     y ~ m at start). Heads up: expect a larger initial loss with that
#     choice on a pretrained backbone.
# ════════════════════════════════════════════════════════════════════════════


class AntiDilutionGateYOLO(nn.Module):
    """Cross-level differential gating.

    Inputs (YAML: [[K_main, K_fused], 1, AntiDilutionGateYOLO, []]):
        main:   P_i from the neck (x in paper notation)
        fused:  global multi-scale feature, e.g., LevelContextFusion output
                (gets resized + 1x1-projected to P_i's spatial+channel space
                to form m)

    Output: y = m + I_hat * (x - m), with channels = main's channels.
    """

    def __init__(self, c_main: int, c_fused: int, alpha: float = 1.0, reduction: int = 16):
        super().__init__()
        self.c_main = int(c_main)
        self.c_fused = int(c_fused)
        self.alpha = float(alpha)
        hidden = max(self.c_main // int(reduction), 8)

        # 1x1 projection bringing the fused tensor's channels to c_main
        if self.c_fused == self.c_main:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(self.c_fused, self.c_main, 1, bias=False),
                nn.BatchNorm2d(self.c_main),
                nn.SiLU(inplace=True),
            )

        # phi_d : dilution detector takes [x, |x - m|] -> 1ch sigmoid map
        self.detector = nn.Sequential(
            nn.Conv2d(2 * self.c_main, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

        # P_obj : objectness/edge prior (paper's stroke prior analog).
        # Kept lightweight per user spec — single 3x3 conv.
        self.prior_conv = nn.Conv2d(self.c_main, 1, 3, padding=1)

        # Identity init (see header note for rationale):
        # bias=+4 on the final layers -> sigmoid ~ 0.98 -> I_hat ~ 0.96*alpha
        # With alpha=1: y ~ x at init (baseline behaviour preserved).
        nn.init.zeros_(self.detector[-1].weight)
        nn.init.constant_(self.detector[-1].bias, 4.0)
        nn.init.zeros_(self.prior_conv.weight)
        nn.init.constant_(self.prior_conv.bias, 4.0)

    def forward(self, inputs):
        main, fused = inputs[0], inputs[1]   # x and source-for-m

        # Spatial align + channel project: fused -> m at main's space
        if fused.shape[-2:] != main.shape[-2:]:
            fused = F.interpolate(fused, size=main.shape[-2:], mode="bilinear", align_corners=False)
        m = self.proj(fused)
        x = main

        # Dilution gate
        diff = (x - m).abs()
        i_map = torch.sigmoid(self.detector(torch.cat([x, diff], dim=1)))
        p_obj = torch.sigmoid(self.prior_conv(x))
        i_hat = self.alpha * p_obj * i_map

        # y = m + I_hat * (x - m)
        return m + i_hat * (x - m)


# ════════════════════════════════════════════════════════════════════════════
# === Directional Strip Block (DSB) — orientation-aware refinement =======
# ════════════════════════════════════════════════════════════════════════════
# Motivation:
#   Bridge damage classes split into two structural categories:
#     - Elongated (crack, lime along seams, rebar_corrosion along rebar):
#       high aspect ratio (10:1 to 20:1), strongly anisotropic
#     - Patch-like (delamination, floating):
#       roughly isotropic
#   Standard 3x3 convolutions have no orientation prior and waste capacity
#   on local symmetric receptive fields. This block injects an explicit
#   orientation inductive bias via three parallel branches:
#     - Horizontal strip: 1x7 depthwise conv -> captures horizontal lines
#     - Vertical strip:   7x1 depthwise conv -> captures vertical lines
#     - Identity:         no transformation -> preserves baseline behavior
#                                              (for patch-like / generic features)
#   A per-pixel softmax gate selects which branch to trust at each location.
#
# Theoretical novelty vs prior modules in this codebase:
#   ADG / AdaptiveFusion / LCF / ADGYOLO all operate on channel x spatial
#   gating — same mechanism family, just different scopes. DSB is orthogonal:
#   it operates on ORIENTATION as a first-class dimension. The gate is the
#   same softmax pattern but the BRANCHES are what make DSB novel.
#
# Identity init:
#   Gate bias = [0, 0, +10] on the identity branch.
#   At t=0: softmax([0, 0, 10]) ~ [0, 0, 1] -> output = identity branch = x.
#   Combined with the strip branches' near-zero contribution (gate weight
#   on them is ~0), the block is effectively a no-op at init -> pretrained
#   Detect head sees baseline input -> safe warm-up.
#
# Visualization potential (for paper figures):
#   At inference, plot gate[:, 0:1, :, :] (horizontal weight),
#   gate[:, 1:2, :, :] (vertical weight) as heatmaps overlaid on the image.
#   Expected: high horizontal weight on horizontal cracks, high vertical
#   weight on vertical cracks, high identity weight on delamination patches.
#   This DIRECTLY visualizes the model learning class-specific orientation
#   priors — strong figure material for the paper.
#
# YAML usage (drop-in refinement, single input, same channels out):
#   [-1, 1, DirectionalStripBlock, [kernel_size?]]
#   Default kernel_size = 7. Larger = more elongated receptive field.
# ════════════════════════════════════════════════════════════════════════════


class DirectionalStripBlock(nn.Module):
    """Four-branch orientation-aware refinement (v2).

    v2 changelog vs v1:
      - Added 4th branch: 3x3 depthwise 'diagonal' conv. Cracks along
        structural seams or shear planes are often oblique (not pure H/V).
        Inspired by EFDNet's DEConv (Zheng et al., 2026, DSP) which uses
        an angle-difference branch; the mechanism here is different (a
        learnable 3x3 depthwise instead of fixed differential operator).
      - Gate extended from 3 -> 4 outputs.

    Args:
        c: channel count of input/output (preserved)
        kernel_size: strip length for H/V branches (default 7).
                     Diagonal branch is fixed 3x3 (depthwise).
    """

    def __init__(self, c: int, kernel_size: int = 7):
        super().__init__()
        self.c = int(c)
        self.kernel_size = int(kernel_size)
        pad = self.kernel_size // 2

        # Strip branches H / V (depthwise) ----------------------------------
        self.strip_h = nn.Conv2d(
            self.c, self.c,
            kernel_size=(1, self.kernel_size),
            padding=(0, pad),
            groups=self.c,
            bias=False,
        )
        self.strip_v = nn.Conv2d(
            self.c, self.c,
            kernel_size=(self.kernel_size, 1),
            padding=(pad, 0),
            groups=self.c,
            bias=False,
        )
        self.bn_h = nn.BatchNorm2d(self.c)
        self.bn_v = nn.BatchNorm2d(self.c)

        # Diagonal branch (3x3 depthwise) -----------------------------------
        # 3x3 instead of longer kernel because diagonal patterns at scale of
        # one feature pixel are well captured at this receptive field, and
        # keeping it small holds parameter overhead in check.
        self.diag = nn.Conv2d(
            self.c, self.c,
            kernel_size=3, padding=1,
            groups=self.c, bias=False,
        )
        self.bn_diag = nn.BatchNorm2d(self.c)

        # Per-pixel softmax gate over 4 branches.
        # Input: concat of (h, v, diag, identity) -> 4*c channels.
        self.gate = nn.Conv2d(4 * self.c, 4, kernel_size=1, bias=True)

        # ── Identity init with gradient-conductive design ──────────────────
        # Same philosophy as v1:
        #   (1) Init each learnable branch as spatial-identity (center=1,
        #       others=0). At init: strip_h(x) = strip_v(x) = diag(x) = x.
        #   (2) Gate weights = 0; bias = [0, 0, 0, +3] on the identity slot.
        #       Softmax ~ [0.045, 0.045, 0.045, 0.865]. All four sum to x:
        #         out = 0.045*x + 0.045*x + 0.045*x + 0.865*x = x.
        # Result: identity at t=0, but every branch (strip_h, strip_v, diag)
        # receives gradient via its non-zero softmax weight, so all of them
        # can learn during training.
        with torch.no_grad():
            self.strip_h.weight.zero_()
            self.strip_h.weight[:, :, 0, pad] = 1.0

            self.strip_v.weight.zero_()
            self.strip_v.weight[:, :, pad, 0] = 1.0

            # Diag init: center of 3x3 = 1 (spatial identity)
            self.diag.weight.zero_()
            self.diag.weight[:, :, 1, 1] = 1.0

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        with torch.no_grad():
            self.gate.bias[3] = 3.0   # identity branch dominates at init

    def forward(self, x):
        h = self.bn_h(self.strip_h(x))      # horizontal-elongated context
        v = self.bn_v(self.strip_v(x))      # vertical-elongated context
        d = self.bn_diag(self.diag(x))      # diagonal / local context
        # identity = x

        gate_input = torch.cat([h, v, d, x], dim=1)         # (B, 4C, H, W)
        logits = self.gate(gate_input)                       # (B, 4, H, W)
        weights = torch.softmax(logits, dim=1)               # (B, 4, H, W)

        w_h  = weights[:, 0:1]
        w_v  = weights[:, 1:2]
        w_d  = weights[:, 2:3]
        w_id = weights[:, 3:4]

        return w_h * h + w_v * v + w_d * d + w_id * x


# ════════════════════════════════════════════════════════════════════════════
# === DirectionalSSM — Mamba-inspired 4-directional state-space module =====
# ════════════════════════════════════════════════════════════════════════════
# Pure-PyTorch approximation of VMamba's SS2D block. Does NOT require
# `mamba-ssm` or `causal-conv1d` packages (which need CUDA kernel compilation
# and are painful to install on Windows).
#
# Mechanism:
#   - Pre-norm via GroupNorm(1, c) (LayerNorm-equivalent for conv tensors)
#   - Input projection: c -> 2c, split into [main, gate]
#   - 4-directional depthwise scanning of `main`:
#       horizontal left->right   : 1xk depthwise conv
#       horizontal right->left   : 1xk depthwise conv on flipped input
#       vertical   top->bottom   : kx1 depthwise conv
#       vertical   bottom->top   : kx1 depthwise conv on flipped input
#   - Average the 4 directional outputs, then a 1x1 "direction mixing" conv
#   - Apply selective gate: scan * SiLU(gate)   (Mamba-style gating)
#   - Output projection (zero-init for identity warm-up) + residual
#
# Differences vs true Mamba:
#   - Replaces selective scan (S6) with depthwise conv. Receptive field is
#     limited to kernel size rather than unlimited via SSM state.
#   - "Selective" = input-dependent gate, not input-dependent A,B,C.
#   These are common simplifications in visual Mamba work; framed as
#   "Mamba-inspired" in the paper.
#
# YAML usage (single-input, same channels out):
#   [-1, 1, DirectionalSSM, [kernel_size?, expand?]]
#   Defaults: kernel_size=7, expand=1
# ════════════════════════════════════════════════════════════════════════════


class DirectionalSSM(nn.Module):
    """Mamba-inspired multi-directional state-space module (pure PyTorch)."""

    def __init__(self, c: int, kernel_size: int = 7, expand: int = 1):
        super().__init__()
        self.c = int(c)
        self.kernel_size = int(kernel_size)
        self.expand = int(expand)
        d_inner = self.c * self.expand
        pad = self.kernel_size // 2

        # Pre-norm
        self.norm = nn.GroupNorm(1, self.c)

        # Input projection: c -> 2 * d_inner (split into main + gate)
        self.in_proj = nn.Conv2d(self.c, 2 * d_inner, kernel_size=1, bias=False)

        # Depthwise convs for horizontal / vertical scans
        self.conv_h = nn.Conv2d(
            d_inner, d_inner,
            kernel_size=(1, self.kernel_size),
            padding=(0, pad),
            groups=d_inner, bias=False,
        )
        self.conv_v = nn.Conv2d(
            d_inner, d_inner,
            kernel_size=(self.kernel_size, 1),
            padding=(pad, 0),
            groups=d_inner, bias=False,
        )

        # Direction-mixing 1x1 conv (fuses 4 directional outputs)
        self.dir_mix = nn.Conv2d(d_inner, d_inner, kernel_size=1, bias=False)

        # Output projection back to c channels (zero-init for identity warm-up)
        self.out_proj = nn.Conv2d(d_inner, self.c, kernel_size=1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x):
        residual = x
        y = self.norm(x)

        # Input projection -> [main, gate]
        proj = self.in_proj(y)
        x_main, x_gate = proj.chunk(2, dim=1)
        x_gate = F.silu(x_gate)

        # 4-directional scanning via flip + conv
        h_lr = self.conv_h(x_main)
        h_rl = torch.flip(self.conv_h(torch.flip(x_main, dims=[-1])), dims=[-1])
        v_tb = self.conv_v(x_main)
        v_bt = torch.flip(self.conv_v(torch.flip(x_main, dims=[-2])), dims=[-2])

        # Mean of 4 directions + 1x1 mixing
        scan = (h_lr + h_rl + v_tb + v_bt) * 0.25
        scan = self.dir_mix(scan)

        # Selective gating
        gated = scan * x_gate

        # Output projection (zero-init -> residual dominates at start)
        return residual + self.out_proj(gated)
