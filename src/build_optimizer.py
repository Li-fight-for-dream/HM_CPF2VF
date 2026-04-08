import torch
import torch.nn as nn


OPTIMIZER_FACTORY = {
    "adam": torch.optim.Adam,
}


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def freeze_backbone(model):
    set_requires_grad(model.conv1, False)
    set_requires_grad(model.bn1, False)
    set_requires_grad(model.layer1, False)
    set_requires_grad(model.layer2, False)
    set_requires_grad(model.layer3, False)
    set_requires_grad(model.layer4, False)


def setup_stage1(model):
    set_requires_grad(model, True)
    set_requires_grad(model.r_head, False)
    set_requires_grad(model.md_head, False)


def setup_stage2_md_only(model):
    set_requires_grad(model, False)
    set_requires_grad(model.md_head, True)


def setup_stage3_joint(model):
    freeze_backbone(model)
    set_requires_grad(model.proj, True)
    set_requires_grad(model.a_head, True)
    set_requires_grad(model.r_head, True)
    set_requires_grad(model.md_head, True)


def _build_optimizer(params, optimizer_name: str, lr: float, weight_decay: float):
    name = optimizer_name.lower()
    if name not in OPTIMIZER_FACTORY:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return OPTIMIZER_FACTORY[name](params, lr=lr, weight_decay=weight_decay)


def build_optimizer_for_stage(model, cfg, stage: int):
    optim_cfg = cfg.optimizer
    wd = optim_cfg.weight_decay
    opt_name = optim_cfg.type

    if stage == 1:
        setup_stage1(model)
        params = [p for p in model.parameters() if p.requires_grad]
        return _build_optimizer(params, opt_name, cfg.training.lr.stage1, wd)

    if stage == 2:
        setup_stage2_md_only(model)
        params = [p for p in model.parameters() if p.requires_grad]
        return _build_optimizer(params, opt_name, cfg.training.lr.stage2_md, wd)

    if stage == 3:
        setup_stage3_joint(model)
        proj_params = [p for p in model.proj.parameters() if p.requires_grad]
        head_params = []
        for head in [model.a_head, model.r_head, model.md_head]:
            head_params.extend([p for p in head.parameters() if p.requires_grad])

        param_groups = []
        if proj_params:
            param_groups.append({"params": proj_params, "lr": cfg.training.lr.stage3_proj})
        if head_params:
            param_groups.append({"params": head_params, "lr": cfg.training.lr.stage3})

        if not param_groups:
            raise RuntimeError("No trainable parameters found for Stage3.")

        name = opt_name.lower()
        if name not in OPTIMIZER_FACTORY:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        return OPTIMIZER_FACTORY[name](param_groups, weight_decay=wd)

    raise ValueError(f"Unsupported stage: {stage}")


# ---------------------------------------------普通resnet------------------------------------------------

def setup_stage1_resnet(model):
    # 训练 backbone + proj + vf_head，冻结 md_head
    set_requires_grad(model, True)
    set_requires_grad(model.md_head, False)


def setup_stage2_resnet(model):
    # 训练 md_head，冻结 backbone + proj + vf_head
    set_requires_grad(model, False)
    set_requires_grad(model.md_head, True)


def build_optimizer_for_stage_resnet(model, cfg, stage: int):
    if stage == 1:
        setup_stage1_resnet(model)
        lr = float(cfg.training.lr.stage1)
    elif stage == 2:
        setup_stage2_resnet(model)
        lr = float(cfg.training.lr.stage2_md)
    else:
        raise ValueError(f"Unsupported stage={stage}, ResNet baseline has only stage1/2")

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable params for current stage")

    opt_name = str(cfg.optimizer.type).lower()
    wd = float(cfg.optimizer.weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    raise ValueError(f"Unsupported optimizer: {cfg.optimizer.type}")