import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import joblib
from typing import Type, Union, List, Optional, Callable, Dict

"""
ResNet + NMF (Scheme A: fixed NMF basis/components)

- Precompute NMF components H on TRAIN VF only:
    X (n, vf_dim) >= 0
    sklearn.decomposition.NMF(n_components=K).fit(X_train)
    H = nmf.components_  # (K, vf_dim)

- In the network:
    backbone -> feat -> w_raw (B,K) -> softplus -> w (B,K) >= 0
    VF_pred = w @ H  # (B, vf_dim)

The model returns:
  w:       (B, K)     non-negative coefficients
  VF_pred: (B, vf_dim)
  md_pred: (B, 1)     optional auxiliary head (kept for parity with your AA model)
"""

# ----------------------------------- BasicBlock, Bottleneck, conv1x1 etc. ---------------------------------

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


# -------------------------------------------- ResNet + NMF ------------------------------------------------------

class ResNetNMF(nn.Module):
    """
    Fixed NMF components H (K, vf_dim). The network predicts non-negative coefficients w (B, K),
    then reconstructs VF_pred = w @ H.

    forward returns dict:
      w:       (B, K)     = softplus(w_raw)
      VF_pred: (B, vf_dim)
      md_pred: (B, 1)
    """

    def __init__(
        self,
        block: Type[Union["BasicBlock", "Bottleneck"]],
        layers: List[int],
        vf_dim: int = 52,
        num_components: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        w_hidden: int = 256,
        md_hidden: int = 128,
        proj_use_layernorm: bool = True,
        w_activation: str = "softplus",  # "softplus" or "relu"
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.vf_dim = vf_dim
        self.num_components = num_components

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

        # ===== backbone (same as standard ResNet) =====
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

        # ===== proj (optional LayerNorm only here) =====
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

        # ===== fixed NMF components H: (K, vf_dim) =====
        # For sklearn NMF, model.components_ is (K, vf_dim), non-negative.
        H_init = torch.zeros(num_components, vf_dim, dtype=torch.float32)
        self.register_buffer("H", H_init, persistent=True)

        # ===== heads =====
        self.w_head = nn.Sequential(
            nn.Linear(feat_dim, w_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(w_hidden, num_components),
        )
        self.md_head = nn.Sequential(
            nn.Linear(feat_dim, md_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(md_hidden, 1),
        )

        if w_activation.lower() not in ["softplus", "relu"]:
            raise ValueError(f"Unsupported w_activation={w_activation}, choose from softplus/relu")
        self.w_activation = w_activation.lower()
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

        # ===== init (keep original ResNet conv/bn init) =====
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize residual branch in each block (optional, same as torchvision)
        if zero_init_residual:
            for m in self.modules():
                if "Bottleneck" in m.__class__.__name__ and hasattr(m, "bn3") and getattr(m.bn3, "weight", None) is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                if "BasicBlock" in m.__class__.__name__ and hasattr(m, "bn2") and getattr(m.bn2, "weight", None) is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    # ----------------------------- load fixed components -----------------------------

    def load_components_from_joblib(self, model_path: str, attr_name: str = "components_") -> None:
        """
        Load sklearn NMF model saved by joblib. Default expects:
          model.components_ -> (K, vf_dim) numpy array (non-negative).
        """
        m = joblib.load(model_path)
        if not hasattr(m, attr_name):
            raise AttributeError(f"Loaded object has no attribute '{attr_name}'.")
        H = getattr(m, attr_name)

        if not isinstance(H, np.ndarray):
            H = np.array(H)

        if H.ndim != 2:
            raise ValueError(f"Expected components to be 2D (K, vf_dim), got shape={H.shape}")

        K, D = H.shape
        if D != self.vf_dim:
            raise ValueError(f"Component dim mismatch: H has D={D}, but model vf_dim={self.vf_dim}")
        if K != self.num_components:
            raise ValueError(f"Component count mismatch: H has K={K}, but model num_components={self.num_components}")

        H_t = torch.from_numpy(H).float()
        if torch.any(H_t < 0):
            raise ValueError("NMF components contain negative values. Ensure you fitted NMF on non-negative VF space.")
        self.H.copy_(H_t)

    def load_components_from_npy(self, npy_path: str) -> None:
        """
        Load components matrix from .npy file. Must be (K, vf_dim) and non-negative.
        """
        H = np.load(npy_path)
        if H.ndim != 2:
            raise ValueError(f"Expected H to be 2D (K, vf_dim), got shape={H.shape}")
        K, D = H.shape
        if D != self.vf_dim:
            raise ValueError(f"Component dim mismatch: H has D={D}, but model vf_dim={self.vf_dim}")
        if K != self.num_components:
            raise ValueError(f"Component count mismatch: H has K={K}, but model num_components={self.num_components}")

        H_t = torch.from_numpy(H).float()
        if torch.any(H_t < 0):
            raise ValueError("NMF components contain negative values. Ensure you saved the correct matrix.")
        self.H.copy_(H_t)

    # ----------------------------- resnet building blocks -----------------------------

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

    # ----------------------------- forward -----------------------------

    def _activate_w(self, w_raw: Tensor) -> Tensor:
        if self.w_activation == "relu":
            return torch.relu(w_raw)
        return self.softplus(w_raw)

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

        w_raw = self.w_head(feat)          # (B, K)
        w = self._activate_w(w_raw)        # (B, K), non-negative

        VF_pred = w @ self.H               # (B, vf_dim)

        md_pred = self.md_head(feat)       # (B, 1)

        return {"w": w, "VF_pred": VF_pred, "md_pred": md_pred}

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self._forward_impl(x)


# ===== unified interface: resnet18 / resnet34 =====

def resnet18_nmf(**kwargs) -> ResNetNMF:
    return ResNetNMF(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


def resnet34_nmf(**kwargs) -> ResNetNMF:
    return ResNetNMF(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)


def build_resnet_nmf(model_name: str, **kwargs) -> ResNetNMF:
    name = model_name.lower()
    if name in ["resnet18", "r18"]:
        return resnet18_nmf(**kwargs)
    if name in ["resnet34", "r34"]:
        return resnet34_nmf(**kwargs)
    raise ValueError(f"Unsupported model_name={model_name}, choose from resnet18/resnet34")
