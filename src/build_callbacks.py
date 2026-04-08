
class PlateauSwitch:
    def __init__(self, ema_alpha: float, min_delta: float, patience: int, min_epochs: int):
        self.ema_alpha = ema_alpha
        self.min_delta = min_delta
        self.patience = patience
        self.min_epochs = min_epochs
        self.ema = None
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, val_loss: float, epoch_idx: int) -> bool:
        if self.ema is None:
            self.ema = val_loss
        else:
            self.ema = self.ema_alpha * val_loss + (1 - self.ema_alpha) * self.ema

        improved = False
        if self.best == float("inf"):
            improved = True
        else:
            rel_impr = (self.best - self.ema) / max(1e-12, abs(self.best))
            if rel_impr > self.min_delta:
                improved = True

        if improved:
            self.best = self.ema
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if epoch_idx + 1 < self.min_epochs:
            return False
        return self.bad_epochs >= self.patience


class MultiMetricEarlyStop:
    def __init__(self, patience: int, min_epochs: int, min_delta_vf: float, min_delta_md: float):
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta_vf = min_delta_vf
        self.min_delta_md = min_delta_md

        self.best_vf = float("inf")
        self.best_md = float("inf")
        self.bad_epochs = 0

    def step(self, epoch_idx: int, val_vf_rmse: float, val_md_rmse):
        improved = False

        if val_vf_rmse is not None:
            if (self.best_vf - val_vf_rmse) > self.min_delta_vf or self.best_vf == float("inf"):
                self.best_vf = val_vf_rmse
                improved = True

        if val_md_rmse is not None:
            if (self.best_md - val_md_rmse) > self.min_delta_md or self.best_md == float("inf"):
                self.best_md = val_md_rmse
                improved = True

        if improved:
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if epoch_idx + 1 < self.min_epochs:
            return False

        return self.bad_epochs >= self.patience


def build_plateau_switch(cfg, stage: int):
    if stage == 1:
        sw = cfg.training.stage_switch.stage1
    elif stage == 2:
        sw = cfg.training.stage_switch.stage2
    else:
        raise ValueError(f"Unsupported stage for plateau switch: {stage}")

    return PlateauSwitch(
        ema_alpha=sw.ema_alpha,
        min_delta=sw.min_delta,
        patience=sw.patience,
        min_epochs=sw.min_epochs,
    )


def build_early_stop(cfg):
    es = cfg.training.early_stop
    return MultiMetricEarlyStop(
        patience=es.patience,
        min_epochs=es.min_stage3_epochs,
        min_delta_vf=es.min_delta_vf,
        min_delta_md=es.min_delta_md,
    )
