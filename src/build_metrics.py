import numpy as np
import torch


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor):
    mse = torch.mean((preds - targets) ** 2).item()
    rmse = float(np.sqrt(mse))
    mae = torch.mean(torch.abs(preds - targets)).item()
    return rmse, mae


def compute_scalar_metrics(preds: torch.Tensor, targets: torch.Tensor):
    preds = preds.view(-1)
    targets = targets.view(-1)
    mse = torch.mean((preds - targets) ** 2).item()
    rmse = float(np.sqrt(mse))
    mae = torch.mean(torch.abs(preds - targets)).item()
    return rmse, mae


def build_metrics(_cfg):
    return {
        "vf": compute_metrics,
        "md": compute_scalar_metrics,
    }
