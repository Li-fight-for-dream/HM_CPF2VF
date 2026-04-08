import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import joblib
from typing import Type, Union, List, Optional, Callable, Dict


# -----------------------------------BasicBlock, Bottleneck, conv1x1等resnet基础模块---------------------------------
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
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
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
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy.

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
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
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


# --------------------------------------------PCA 输出的 ResNet------------------------------------------------------

class ResNetPCA(nn.Module):
    """
    输出 PCA 系数作为主要预测目标，不再输出 AA / r。

    forward 返回 dict:
      pca_coef: (B, C)  # 归一化系数（原始系数除以 explained_variance）
      pca_coef_raw: (B, C)  # 原始 PCA 投影系数
      md_pred:  (B, 1)
      VF_pred/vf_pred:  (B, 52) = (pca_coef * variance) @ components + mean
    """

    def __init__(
        self,
        block: Type[Union["BasicBlock", "Bottleneck"]],
        layers: List[int],
        vf_dim: int = 52,
        pca_n_components: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        coef_hidden: int = 256,
        md_hidden: int = 128,
        proj_use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.vf_dim = vf_dim
        self.pca_n_components = pca_n_components

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

        # ===== backbone（与原始 ResNet 一致）=====
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

        # ===== proj（加 LayerNorm，仅此处加）=====
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

        # ===== heads =====
        self.coef_head = nn.Sequential(
            nn.Linear(feat_dim, coef_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(coef_hidden, pca_n_components),
        )
        self.md_head = nn.Sequential(
            nn.Linear(feat_dim, md_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(md_hidden, 1),
        )

        # ===== PCA 参数缓存（从 joblib 加载）=====
        self.register_buffer("pca_components", torch.zeros(pca_n_components, vf_dim), persistent=True)
        self.register_buffer("pca_variance", torch.zeros(pca_n_components), persistent=True)
        self.register_buffer("pca_mean", torch.zeros(vf_dim), persistent=True)
        self.register_buffer("pca_loaded", torch.tensor(False), persistent=False)

        # ===== 初始化（保持你原 ResNet 的 conv/bn init）=====
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize 残差分支逻辑
        if zero_init_residual:
            for m in self.modules():
                if "Bottleneck" in m.__class__.__name__ and hasattr(m, "bn3") and getattr(m.bn3, "weight", None) is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                if "BasicBlock" in m.__class__.__name__ and hasattr(m, "bn2") and getattr(m.bn2, "weight", None) is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def load_pca_from_joblib(self, model_path: str) -> None:
        """
        joblib 文件里保存的是 PCA 模型本体，或 PCAAdapter.model。
        """
        m = joblib.load(model_path)
        if hasattr(m, "model") and hasattr(m.model, "components_"):
            m = m.model

        if not hasattr(m, "components_"):
            raise ValueError("Loaded object does not have PCA components_.")
        if not hasattr(m, "explained_variance_"):
            raise ValueError("Loaded object does not have PCA explained_variance_.")

        components = m.components_
        mean = getattr(m, "mean_", None)
        variance = m.explained_variance_

        if not isinstance(components, np.ndarray):
            components = np.array(components)
        if mean is None:
            mean = np.zeros(components.shape[1], dtype=components.dtype)
        if not isinstance(mean, np.ndarray):
            mean = np.array(mean)
        if not isinstance(variance, np.ndarray):
            variance = np.array(variance)

        if components.ndim != 2:
            raise ValueError(f"Expected components_ to be 2D (C, vf_dim), got shape={components.shape}")

        C, D = components.shape
        if D != self.vf_dim:
            raise ValueError(f"PCA dim mismatch: components_ has D={D}, but model vf_dim={self.vf_dim}")
        if C != self.pca_n_components:
            raise ValueError(f"PCA component mismatch: components_ has C={C}, but pca_n_components={self.pca_n_components}")
        if mean.shape[0] != D:
            raise ValueError(f"PCA mean mismatch: mean_ has D={mean.shape[0]}, but components_ has D={D}")
        if variance.shape[0] != C:
            raise ValueError(f"PCA variance mismatch: variance_ has C={variance.shape[0]}, but pca_n_components={C}")

        self.pca_components.copy_(torch.from_numpy(components).float())
        self.pca_mean.copy_(torch.from_numpy(mean).float())
        self.pca_variance.copy_(torch.from_numpy(variance).float())
        self.pca_loaded.fill_(True)

    def pca_inverse_transform(self, coeff_norm: Tensor) -> Tensor:
        if not bool(self.pca_loaded.item()):
            raise RuntimeError("PCA components not loaded. Call load_pca_from_joblib first.")
        # 网络预测的是归一化系数（raw_coef / variance），逆变换前先恢复原始系数
        coeff_raw = coeff_norm * self.pca_variance
        return coeff_raw @ self.pca_components + self.pca_mean

    def pca_transform(self, vf: Tensor, normalized: bool = True, eps: float = 1e-12) -> Tensor:
        """
        将 52 维 VF 映射到 PCA 系数。
        normalized=True 时返回 raw_coef / explained_variance。
        """
        if not bool(self.pca_loaded.item()):
            raise RuntimeError("PCA components not loaded. Call load_pca_from_joblib first.")

        centered = vf - self.pca_mean
        coeff_raw = centered @ self.pca_components.t()
        if not normalized:
            return coeff_raw

        denom = torch.clamp(self.pca_variance, min=eps)
        return coeff_raw / denom

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

        layers_ = []
        layers_.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
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

        pca_coef = self.coef_head(feat)  # normalized coef: raw_coef / variance
        md_pred = self.md_head(feat)
        vf_pred = self.pca_inverse_transform(pca_coef)

        # 同时暴露 raw_coef，便于调试/分析
        pca_coef_raw = pca_coef * self.pca_variance
        return {
            "pca_coef": pca_coef,
            "pca_coef_raw": pca_coef_raw,
            "md_pred": md_pred,
            "VF_pred": vf_pred,
            "vf_pred": vf_pred,
        }

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self._forward_impl(x)


# ===== 统一接口：选择 resnet18 / resnet34 =====

def resnet18_pca(**kwargs) -> ResNetPCA:
    return ResNetPCA(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


def resnet34_pca(**kwargs) -> ResNetPCA:
    return ResNetPCA(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)


def build_resnet_pca(model_name: str, **kwargs) -> ResNetPCA:
    name = model_name.lower()
    if name in ["resnet18", "r18"]:
        return resnet18_pca(**kwargs)
    if name in ["resnet34", "r34"]:
        return resnet34_pca(**kwargs)
    raise ValueError(f"Unsupported model_name={model_name}, choose from resnet18/resnet34")
