import torch.nn as nn
import torch


LOSS_FACTORY = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
}


def build_losses(cfg):
    losses_cfg = cfg.losses
    vf_name = losses_cfg.vf.type.lower()
    md_name = losses_cfg.md.type.lower()
    a_name = getattr(getattr(losses_cfg, "a", None), "type", "mse").lower()
    coef_name = getattr(getattr(losses_cfg, "coef", None), "type", "mse").lower()
    w_name = getattr(getattr(losses_cfg, "w", None), "type", coef_name).lower()

    if vf_name not in LOSS_FACTORY:
        raise ValueError(f"Unsupported VF loss: {vf_name}")
    if md_name not in LOSS_FACTORY:
        raise ValueError(f"Unsupported MD loss: {md_name}")
    if a_name not in LOSS_FACTORY:
        raise ValueError(f"Unsupported A loss: {a_name}")
    if coef_name not in LOSS_FACTORY:
        raise ValueError(f"Unsupported coef loss: {coef_name}")
    if w_name not in LOSS_FACTORY:
        raise ValueError(f"Unsupported w loss: {w_name}")

    return {
        "vf": LOSS_FACTORY[vf_name](),
        "md": LOSS_FACTORY[md_name](),
        "a": LOSS_FACTORY[a_name](),
        "coef": LOSS_FACTORY[coef_name](),
        "w": LOSS_FACTORY[w_name](),
    }


def compute_stage_loss(stage, outputs, vf_true, md_true, losses, loss_weights, a_true=None):
    """
    用于构建三阶段AA+resnet的综合loss，用于engine.py
    """
    loss_vf = losses["vf"]
    loss_md = losses["md"]

    if stage == 1:  # 1阶段：训练VF1，并约束原型权重a
        l_vf1 = loss_vf(outputs["VF1"], vf_true)

        lambda_a = float(getattr(loss_weights, "lambda_a", 0.0))
        if a_true is None:
            if lambda_a > 0:
                raise ValueError("a_true must be provided when lambda_a > 0 in stage1")
            l_a = torch.tensor(0.0, device=vf_true.device)
        else:
            l_a = losses["a"](outputs["a"], a_true)

        total = l_vf1 + lambda_a * l_a
        return total, {
            "L_total": float(total.detach().item()),
            "L_vf1": float(l_vf1.detach().item()),
            "L_a": float(l_a.detach().item()) if a_true is not None else 0.0,
        }

    if stage == 2:  # 2阶段，只训练md
        if md_true is None:
            l_md = torch.tensor(0.0, device=outputs["md_pred"].device)
        else:
            l_md = loss_md(outputs["md_pred"], md_true)
        return l_md, {"L_md": float(l_md.detach().item()) if md_true is not None else 0.0}

    # stage=3时，三阶段，考虑VF、r和md的综合loss
    l_vf = loss_vf(outputs["VF_pred"], vf_true)
    l_vf1 = loss_vf(outputs["VF1"], vf_true)
    l_r = torch.mean(torch.abs(outputs["r"]))

    if md_true is None:
        l_md = torch.tensor(0.0, device=vf_true.device)
    else:
        l_md = loss_md(outputs["md_pred"], md_true)

    total = l_vf + loss_weights.lambda_vf1 * l_vf1 + loss_weights.beta_r * l_r + loss_weights.alpha_md * l_md
    return total, {
        "L_total": float(total.detach().item()),
        "L_vf": float(l_vf.detach().item()),
        "L_vf1": float(l_vf1.detach().item()),
        "L_r": float(l_r.detach().item()),
        "L_md": float(l_md.detach().item()) if md_true is not None else 0.0,
    }


def compute_pca_stage_loss(stage, outputs, vf_true, md_true, coef_true, losses, loss_weights):
    """
    用于构建 PCANet 两阶段 loss（stage1: VF + coef，stage2: MD）
    """
    loss_vf = losses["vf"]
    loss_md = losses["md"]
    loss_coef = losses["coef"]

    if stage == 1:
        if coef_true is None:
            raise ValueError("coef_true must be provided for PCANet stage1")

        l_vf = loss_vf(outputs["VF_pred"], vf_true)
        l_coef = loss_coef(outputs["pca_coef"], coef_true)
        lambda_coef = float(getattr(loss_weights, "lambda_coef", 1.0))
        total = l_vf + lambda_coef * l_coef
        return total, {
            "L_total": float(total.detach().item()),
            "L_vf": float(l_vf.detach().item()),
            "L_coef": float(l_coef.detach().item()),
        }

    if stage == 2:
        if md_true is None:
            l_md = torch.tensor(0.0, device=outputs["md_pred"].device)
        else:
            l_md = loss_md(outputs["md_pred"], md_true)
        return l_md, {"L_md": float(l_md.detach().item()) if md_true is not None else 0.0}

    raise ValueError(f"Unsupported stage={stage} for PCANet")


def compute_nmf_stage_loss(stage, outputs, vf_true, md_true, w_true, losses, loss_weights):
    """
    用于构建 NMFNet 两阶段 loss（stage1: VF + w，stage2: MD）
    """
    loss_vf = losses["vf"]
    loss_md = losses["md"]
    loss_w = losses.get("w", losses["coef"])

    if stage == 1:
        if w_true is None:
            raise ValueError("w_true must be provided for NMFNet stage1")

        l_vf = loss_vf(outputs["VF_pred"], vf_true)
        l_w = loss_w(outputs["w"], w_true)
        lambda_w = float(getattr(loss_weights, "lambda_w", 1.0))
        total = l_vf + lambda_w * l_w
        return total, {
            "L_total": float(total.detach().item()),
            "L_vf": float(l_vf.detach().item()),
            "L_w": float(l_w.detach().item()),
        }

    if stage == 2:
        if md_true is None:
            l_md = torch.tensor(0.0, device=outputs["md_pred"].device)
        else:
            l_md = loss_md(outputs["md_pred"], md_true)
        return l_md, {"L_md": float(l_md.detach().item()) if md_true is not None else 0.0}

    raise ValueError(f"Unsupported stage={stage} for NMFNet")
