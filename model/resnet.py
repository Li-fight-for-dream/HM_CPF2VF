import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Dict, List, Optional, Type, Union


# -----------------------------------BasicBlock, Bottleneck, conv1x1---------------------------------
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# --------------------------------------------ResNet + proj + heads--------------------------------------------
class ResNetVFMD(nn.Module):
    """
    标准ResNet骨架 + proj + vf/md 两个任务头。

    forward 返回 dict:
      VF_pred: (B, vf_dim)
      md_pred: (B, 1)
    """

    def __init__(
        self,
        block: Type[Union["BasicBlock", "Bottleneck"]],
        layers: List[int],
        vf_dim: int = 52,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        vf_hidden: int = 256,
        md_hidden: int = 128,
        proj_use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.vf_dim = vf_dim

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        # backbone
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 512 * block.expansion

        # proj
        if proj_use_layernorm:
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
            )

        # heads
        self.vf_head = nn.Sequential(
            nn.Linear(feat_dim, vf_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(vf_hidden, vf_dim),
        )
        self.md_head = nn.Sequential(
            nn.Linear(feat_dim, md_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(md_hidden, 1),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if "Bottleneck" in m.__class__.__name__ and hasattr(m, "bn3") and getattr(m.bn3, "weight", None) is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                if "BasicBlock" in m.__class__.__name__ and hasattr(m, "bn2") and getattr(m.bn2, "weight", None) is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union["BasicBlock", "Bottleneck"]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers_ = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers_.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers_)

    def _forward_impl(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        feat = self.proj(feat)

        vf_pred = self.vf_head(feat)
        md_pred = self.md_head(feat)

        return {"VF_pred": vf_pred, "md_pred": md_pred}

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self._forward_impl(x)


# ===== 统一接口：选择 resnet18 / resnet34 =====
def resnet18_vfmd(**kwargs) -> ResNetVFMD:
    return ResNetVFMD(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


def resnet34_vfmd(**kwargs) -> ResNetVFMD:
    return ResNetVFMD(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)


def build_resnet(model_name: str, **kwargs) -> ResNetVFMD:
    name = model_name.lower()
    if name in ["resnet18", "r18"]:
        return resnet18_vfmd(**kwargs)
    if name in ["resnet34", "r34"]:
        return resnet34_vfmd(**kwargs)
    raise ValueError(f"Unsupported model_name={model_name}, choose from resnet18/resnet34")
