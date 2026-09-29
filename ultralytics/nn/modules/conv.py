# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
    "GAM_Attention","GCT","ShuffleAttention","ResBlock_CBAM","GatherExcite","GlobalContext","C3k2","C3k",
    "WaveletAttention", "WMTBlock", "C3k_WMA", "C3k2_WMA",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
    
class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)

def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out
    
class GAM_Attention(nn.Module):
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )

        self.spatial_attention = nn.Sequential(

            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        # x_channel_att=channel_shuffle(x_channel_att,4) #last shuffle
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        # out=channel_shuffle(out,4) #last shuffle
        return out

class GCT(nn.Module):
    def __init__(self, channels, c=2, eps=1e-5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.eps = eps
        self.c = c

    def forward(self, x):
        y = self.avgpool(x)
        mean = y.mean(dim=1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_transform = torch.exp(-(y_norm ** 2 / 2 * self.c))
        return x * y_transform.expand_as(x)

    
class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out

class GlobalContext(nn.Module):

    def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
                 rd_ratio=1. / 8, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)

        self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None

        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        if fuse_add:
            self.mlp_add = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_scale = None

        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x


from timm.layers.create_conv2d import create_conv2d


class GatherExcite(nn.Module):
    def __init__(
            self, channels, feat_size=None, extra_params=False, extent=0, use_mlp=True,
            rd_ratio=1. / 16, rd_channels=None, rd_divisor=1, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, gate_layer='sigmoid'):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert feat_size is not None, 'spatial feature size must be specified for global extent w/ params'
                self.gather.add_module(
                    'conv1', create_conv2d(channels, channels, kernel_size=feat_size, stride=1, depthwise=True))
                if norm_layer:
                    self.gather.add_module(f'norm1', nn.BatchNorm2d(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_module(
                        f'conv{i + 1}',
                        create_conv2d(channels, channels, kernel_size=3, stride=2, depthwise=True))
                    if norm_layer:
                        self.gather.add_module(f'norm{i + 1}', nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.add_module(f'act{i + 1}', act_layer(inplace=True))
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.mlp = ConvMlp(channels, rd_channels, act_layer=act_layer) if use_mlp else nn.Identity()
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                # global extent
                x_ge = x.mean(dim=(2, 3), keepdims=True)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
            else:
                x_ge = F.avg_pool2d(
                    x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2, count_include_pad=False)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2)
        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)

class ResBlock_CBAM(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=1):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )
        # self.cbam = CBAM(c1=places * self.expansion, c2=places * self.expansion, )
        self.cbam = CBAM(c1=places * self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# =============================================================================
# Wavelet-domain Modulation Self-Attention (WMA), adapted from DMNet:
#   "Dual-domain Modulation Network for Lightweight Image Super-Resolution"
#   arXiv:2503.10047, §III-A, Fig. 4(c).
#
# Public classes:
#   - WaveletAttention : drop-in attention module, WaveletAttention(c1) -> same shape
#   - WMTBlock         : full pre-norm Transformer block (WMA + FFN), residual
#   - C3k_WMA          : C3k variant with WaveletAttention after the last conv
#   - C3k2_WMA         : C3k2 variant with WaveletAttention after the last conv
# =============================================================================

import torch.nn.functional as _F  # local alias, no effect on existing file


# ---------- Haar 2D DWT / IDWT (no learnable params, lossless) ---------------
def _haar_dwt(x):
    """Haar 2D DWT. (B, C, H, W) -> (B, 4C, H/2, W/2), channels = [LL, LH, HL, HH].

    Odd H/W are reflection-padded to even.
    """
    H, W = x.shape[-2], x.shape[-1]
    if H % 2 or W % 2:
        x = _F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
    x_e = x[..., 0::2, :]
    x_o = x[..., 1::2, :]
    L = (x_e + x_o) * 0.5
    Hh = (x_e - x_o) * 0.5
    LL = L[..., 0::2] + L[..., 1::2]
    LH = L[..., 0::2] - L[..., 1::2]
    HL = Hh[..., 0::2] + Hh[..., 1::2]
    HH = Hh[..., 0::2] - Hh[..., 1::2]
    return torch.cat([LL, LH, HL, HH], dim=1)


def _haar_idwt(x):
    """Inverse of _haar_dwt. (B, 4C, H, W) -> (B, C, 2H, 2W)."""
    C4 = x.shape[1]
    assert C4 % 4 == 0, f"channel count must be divisible by 4, got {C4}"
    C = C4 // 4
    LL, LH, HL, HH = x[:, 0:C], x[:, C:2 * C], x[:, 2 * C:3 * C], x[:, 3 * C:4 * C]
    L_e = (LL + LH) * 0.5
    L_o = (LL - LH) * 0.5
    H_e = (HL + HH) * 0.5
    H_o = (HL - HH) * 0.5
    B, Cn, Hh, Wh = L_e.shape
    L = torch.empty(B, Cn, Hh, 2 * Wh, device=x.device, dtype=x.dtype)
    L[..., 0::2] = L_e
    L[..., 1::2] = L_o
    Hi = torch.empty(B, Cn, Hh, 2 * Wh, device=x.device, dtype=x.dtype)
    Hi[..., 0::2] = H_e
    Hi[..., 1::2] = H_o
    x_e = L + Hi
    x_o = L - Hi
    out = torch.empty(B, Cn, 2 * Hh, 2 * Wh, device=x.device, dtype=x.dtype)
    out[..., 0::2, :] = x_e
    out[..., 1::2, :] = x_o
    return out


class _WaveletLayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensors. Local to this block to
    avoid clashing with any LayerNorm2d the rest of the file might define."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class WaveletAttention(nn.Module):
    """Wavelet-domain Modulation Self-Attention (WMA) from DMNet.

    Pipeline:
        x (B, C, H, W)
        -> LN + 1x1 reduce  : (B, C/reduction, H, W)
        -> Haar DWT         : (B, 4*C/reduction, H/2, W/2)   # [LL, LH, HL, HH]
        -> 1x1 + 3x3 dwconv -> Q', K', V'
        -> channel-wise multi-head self-attention (Restormer-style MDTA)
        -> elementwise * sigmoid(depthwise-k-conv on subbands)  # dynamic gate
        -> 1x1 proj
        -> IDWT             : (B, C/reduction, H, W)
        -> 1x1 expand       : (B, C, H, W)
        -> + residual (optional)

    Args:
        c1 (int): input/output channels.
        num_heads (int): heads for channel self-attention. Default 4.
        reduction (int): channel reduction before DWT. Default 4 (paper).
        dyn_kernel (int): kernel size of the dynamic depthwise branch. Default 5.
        use_residual (bool): if True, output = x + WMA(x). Default True.
    """

    def __init__(self, c1, num_heads=4, reduction=4, dyn_kernel=5, use_residual=True):
        super().__init__()
        assert c1 % reduction == 0, f"c1({c1}) must be divisible by reduction({reduction})"
        c_red = c1 // reduction
        c_dwt = 4 * c_red
        assert c_dwt % num_heads == 0, (
            f"4*c1/reduction ({c_dwt}) must be divisible by num_heads ({num_heads})"
        )
        self.c1 = c1
        self.c_red = c_red
        self.c_dwt = c_dwt
        self.num_heads = num_heads
        self.use_residual = use_residual

        self.norm = _WaveletLayerNorm2d(c1)
        self.reduce = nn.Conv2d(c1, c_red, 1, bias=False)

        self.qkv_pw = nn.Conv2d(c_dwt, c_dwt * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(c_dwt * 3, c_dwt * 3, 3, padding=1,
                                groups=c_dwt * 3, bias=False)

        self.temperature = nn.Parameter(torch.ones(num_heads))

        self.dyn = nn.Sequential(
            nn.Conv2d(c_dwt, c_dwt, dyn_kernel, padding=dyn_kernel // 2,
                      groups=c_dwt, bias=False),
            nn.Conv2d(c_dwt, c_dwt, 1, bias=True),
        )

        self.proj = nn.Conv2d(c_dwt, c_dwt, 1, bias=False)
        self.expand = nn.Conv2d(c_red, c1, 1, bias=False)

    def _attention(self, qkv):
        B, _, h, w = qkv.shape
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.num_heads, self.c_dwt // self.num_heads, h * w)
        k = k.view(B, self.num_heads, self.c_dwt // self.num_heads, h * w)
        v = v.view(B, self.num_heads, self.c_dwt // self.num_heads, h * w)
        q = _F.normalize(q, dim=-1)
        k = _F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature.view(-1, 1, 1)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out.view(B, self.c_dwt, h, w)

    def forward(self, x):
        identity = x
        H, W = x.shape[-2], x.shape[-1]
        y = self.norm(x)
        y = self.reduce(y)
        f_dwt = _haar_dwt(y)
        qkv = self.qkv_dw(self.qkv_pw(f_dwt))
        attn_out = self._attention(qkv)
        gate = torch.sigmoid(self.dyn(f_dwt))
        modulated = self.proj(attn_out * gate)
        y = _haar_idwt(modulated)
        y = y[..., :H, :W]                     # crop if odd-padded
        y = self.expand(y)
        return identity + y if self.use_residual else y


class WMTBlock(nn.Module):
    """Wavelet-domain Modulation Transformer block: WMA + FFN, both with residual.

    Heavier than WaveletAttention; use when you want a full Transformer-style block.
    """

    def __init__(self, c1, num_heads=4, reduction=4, ffn_expansion=2.0):
        super().__init__()
        self.attn = WaveletAttention(c1, num_heads=num_heads,
                                     reduction=reduction, use_residual=False)
        hidden = int(c1 * ffn_expansion)
        self.norm2 = _WaveletLayerNorm2d(c1)
        self.ffn = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, c1, 1, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------- Self-contained Bottleneck / C3 / C3k / C3k2 with WMA -------------
# Standard YOLO11 building blocks reproduced here so the WMA variants don't
# depend on `ultralytics.nn.modules.block`. Existing C3k / C3k2 you import
# elsewhere are untouched. Use C3k_WMA / C3k2_WMA in your YAML where you want
# wavelet attention at the end of the block.

class _Bottleneck(nn.Module):
    """Standard YOLO bottleneck (private copy, kept here for WMA blocks)."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class _C3(nn.Module):
    """CSP Bottleneck with 3 convs (private copy)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(_Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                                 for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class _C2f(nn.Module):
    """Faster CSP bottleneck with 2 convs (private copy)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(_Bottleneck(self.c, self.c, shortcut, g,
                                           k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k_WMA(_C3):
    """C3k variant: apply WaveletAttention after the last conv (cv3)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3,
                 num_heads=4, reduction=4):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # rebuild inner bottlenecks with the requested kernel k
        self.m = nn.Sequential(*(_Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0)
                                 for _ in range(n)))
        self.attn = WaveletAttention(c2, num_heads=num_heads, reduction=reduction)

    def forward(self, x):
        out = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        return self.attn(out)


class C3k2_WMA(_C2f):
    """C3k2 variant: apply WaveletAttention after the last conv (cv2).

    Args mirror standard C3k2: (c1, c2, n, c3k, e, g, shortcut).
    `c3k=True` uses C3k_WMA inside? -> No: inner blocks stay vanilla C3k/Bottleneck
    so we don't stack attention twice. Attention is added once, at the end.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True,
                 num_heads=4, reduction=4):
        super().__init__(c1, c2, n, shortcut, g, e)
        # inner modules: original C3k (no attention) or Bottleneck, matching YOLO11
        self.m = nn.ModuleList(
            _C3(self.c, self.c, 2, shortcut, g) if c3k
            else _Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
        self.attn = WaveletAttention(c2, num_heads=num_heads, reduction=reduction)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.attn(out)