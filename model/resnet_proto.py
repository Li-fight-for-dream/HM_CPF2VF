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
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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

# --------------------------------------------自定义的resnet网络------------------------------------------------------

class ResNetProtoResidual(nn.Module):
    """
    固定原型P (K, 52)，学习原型权重a + 残差r，并输出md。

    forward 返回 dict:
      a:       (B, K)
      VF1:     (B, 52)   = a @ P
      r:       (B, 52)
      VF_pred: (B, 52)   = VF1 + r
      r_norm:  (B, 1)    = ||r||_2
      md_pred: (B, 1)
    """

    def __init__(
        self,
        block: Type[Union["BasicBlock", "Bottleneck"]],
        layers: List[int],
        vf_dim: int = 52,
        num_prototypes: int = 9,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        a_hidden: int = 256,
        r_hidden: int = 256,
        md_hidden: int = 128,
        zero_init_residual_head: bool = True,
        proj_use_layernorm: bool = True,   # 默认只给 proj 加 norm
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.vf_dim = vf_dim
        self.num_prototypes = num_prototypes
        # 运行时缓存 AA 模型，用于将 VF 映射到 a_true（不参与 state_dict）
        self._aa_model = None

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

        # ===== 固定原型 P: (K, 52) =====
        P_init = torch.zeros(num_prototypes, vf_dim, dtype=torch.float32)
        self.register_buffer("P", P_init, persistent=True)

        # ===== heads（先不加 norm）=====
        self.a_head = nn.Sequential(
            nn.Linear(feat_dim, a_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(a_hidden, num_prototypes),
        )
        self.r_head = nn.Sequential(
            nn.Linear(feat_dim, r_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(r_hidden, vf_dim),
        )
        self.md_head = nn.Sequential(
            nn.Linear(feat_dim, md_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(md_hidden, 1),
        )

        # ===== 初始化（保持你原 ResNet 的 conv/bn init）=====
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # r_head 最后一层初始输出0（利于阶段1先学VF1）：把残差头 r_head 的最后一层初始化为“恒等于 0 的函数”
        if zero_init_residual_head:
            last = self.r_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.constant_(last.weight, 0.0)
                nn.init.constant_(last.bias, 0.0)

        # Zero-initialize 残差分支逻辑：把 ResNet 每个 residual block 的“残差分支”初始设为 0
        if zero_init_residual:
            for m in self.modules():
                if "Bottleneck" in m.__class__.__name__ and hasattr(m, "bn3") and getattr(m.bn3, "weight", None) is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                if "BasicBlock" in m.__class__.__name__ and hasattr(m, "bn2") and getattr(m.bn2, "weight", None) is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def load_prototypes_from_joblib(self, model_path: str) -> None:
        """
        joblib 模型里: model.archetypes_ -> (K, 52) numpy
        """
        m = joblib.load(model_path)
        # 兼容包装器对象（如 adapter.model）
        if hasattr(m, "model") and hasattr(m.model, "archetypes_"):
            m = m.model

        if not hasattr(m, "archetypes_"):
            raise ValueError("Loaded object does not contain `archetypes_`.")

        Z = m.archetypes_
        if not isinstance(Z, np.ndarray):
            Z = np.array(Z)

        if Z.ndim != 2:
            raise ValueError(f"Expected archetypes_ to be 2D (K, vf_dim), got shape={Z.shape}")

        K, D = Z.shape
        if D != self.vf_dim:
            raise ValueError(f"Prototype dim mismatch: Z has D={D}, but model vf_dim={self.vf_dim}")
        if K != self.num_prototypes:
            raise ValueError(f"Prototype count mismatch: Z has K={K}, but model num_prototypes={self.num_prototypes}")

        P = torch.from_numpy(Z).float()  # (K, 52)
        # buffer 用 copy_ 最稳，不会丢 register_buffer 属性
        self.P.copy_(P)
        self._aa_model = m

    def aa_transform(self, vf: Tensor, normalize_output: bool = True, eps: float = 1e-12) -> Tensor:
        """
        使用训练好的 AA 模型将 VF 映射为原型权重 a_true。
        推荐使用 clip + normalize 保证 a_true 落在单纯形：a>=0 且 sum(a)=1。
        """
        if self._aa_model is None or not hasattr(self._aa_model, "transform"):
            raise RuntimeError("AA model is not loaded. Call `load_prototypes_from_joblib` first.")

        vf_np = vf.detach().cpu().numpy()
        a_true = self._aa_model.transform(vf_np)
        a_true = np.asarray(a_true, dtype=np.float32)

        if a_true.ndim != 2:
            raise ValueError(f"Expected a_true to be 2D (B,K), got shape={a_true.shape}")
        if a_true.shape[1] != self.num_prototypes:
            raise ValueError(
                f"Prototype weight dim mismatch: a_true has K={a_true.shape[1]}, expected {self.num_prototypes}"
            )

        if normalize_output:
            a_true = np.clip(a_true, a_min=0.0, a_max=None)
            row_sum = np.sum(a_true, axis=1, keepdims=True)
            bad = row_sum <= eps
            if np.any(bad):
                a_true[bad[:, 0]] = 1.0 / float(self.num_prototypes)
                row_sum = np.sum(a_true, axis=1, keepdims=True)
            a_true = a_true / np.clip(row_sum, a_min=eps, a_max=None)

        return torch.from_numpy(a_true).to(device=vf.device, dtype=vf.dtype)

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
        feat = torch.flatten(x, 1)   # (B, feat_dim)
        feat = self.proj(feat)

        a_logits = self.a_head(feat)
        a = torch.softmax(a_logits, dim=1)  # (B, K)

        VF1 = a @ self.P              # (B, 52)
        r = self.r_head(feat)         # (B, 52)
        VF_pred = VF1 + r             # (B, 52)

        r_norm = torch.norm(r, p=2, dim=1, keepdim=True)  # (B,1)
        md_pred = self.md_head(feat)   # (B,1)

        return {"a": a, "VF1": VF1, "r": r, "VF_pred": VF_pred, "r_norm": r_norm, "md_pred": md_pred}

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self._forward_impl(x)


# ===== 统一接口：选择 resnet18 / resnet34 =====

def resnet18_proto_residual(**kwargs) -> ResNetProtoResidual:
    # resnet18: [2,2,2,2]
    return ResNetProtoResidual(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


def resnet34_proto_residual(**kwargs) -> ResNetProtoResidual:
    # resnet34: [3,4,6,3]
    return ResNetProtoResidual(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)


def build_resnet_proto(model_name: str, **kwargs) -> ResNetProtoResidual:
    name = model_name.lower()
    if name in ["resnet18", "r18"]:
        return resnet18_proto_residual(**kwargs)
    if name in ["resnet34", "r34"]:
        return resnet34_proto_residual(**kwargs)
    raise ValueError(f"Unsupported model_name={model_name}, choose from resnet18/resnet34")
