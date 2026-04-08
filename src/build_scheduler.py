import torch


def build_scheduler(optimizer, cfg):
    sch_cfg = cfg.scheduler
    sch_type = sch_cfg.type.lower()
    if sch_type != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler type: {sch_cfg.type}")

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sch_cfg.mode,
        factor=sch_cfg.factor,
        patience=sch_cfg.patience,
    )
