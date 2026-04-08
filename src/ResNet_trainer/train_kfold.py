from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.resnet import build_resnet
from src.ResNet_trainer.engine import fit
from src.build_callbacks import build_plateau_switch
from src.build_losses import build_losses
from src.build_metrics import build_metrics
from src.build_optimizer import build_optimizer_for_stage_resnet
from src.build_scheduler import build_scheduler
from src.data_dealer.dataset_for_fundus import vf_with_fundus_dataset
from src.data_dealer.split_data_to_folds import make_kfold_splits
from src.utils.config import load_config, parse_train_args
from src.utils.create_save_path import create_save_path
from src.utils.logger import build_writers, close_writers, save_config_snapshot, save_split_dataframe
from src.utils.seed import set_seed


def _resolve_kfold_params(cfg):
    kfold_cfg = getattr(cfg, "kfold", None)
    if kfold_cfg is None:
        return 5, cfg.seed

    k_folds = int(getattr(kfold_cfg, "k_folds", 5))
    kfold_seed = int(getattr(kfold_cfg, "seed", cfg.seed))
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")
    return k_folds, kfold_seed


def _split_fold_indices(folds, fold_idx):
    val_idx = folds[fold_idx]
    train_idx = np.concatenate([folds[i] for i in range(len(folds)) if i != fold_idx])
    return train_idx, val_idx


def _build_fold_loaders(cfg, full_df, train_idx, val_idx):
    train_df = full_df.loc[train_idx].reset_index(drop=True)
    val_df = full_df.loc[val_idx].reset_index(drop=True)

    train_dataset = vf_with_fundus_dataset(train_df, image_root=getattr(cfg, "image_root", None))
    val_dataset = vf_with_fundus_dataset(val_df, image_root=getattr(cfg, "image_root", None))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=int(getattr(cfg, "num_workers", 0)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=int(getattr(cfg, "num_workers", 0)),
    )

    # fit 接口需要三元组；kfold场景不做独立测试集评估，第三个用占位即可
    return train_loader, val_loader, val_loader


def _run_one_fold(base_cfg, full_df, folds, fold_idx, base_save_path):
    fold_cfg = deepcopy(base_cfg)
    fold_cfg.fold_index = fold_idx + 1
    fold_cfg.fold_count = len(folds)
    fold_cfg.training.use_test = False

    # 训练准备
    # 每折重置随机种子，避免后续折受前一折随机状态影响
    fold_seed = int(base_cfg.seed)
    set_seed(fold_seed, deterministic=fold_cfg.deterministic)
    fold_save_path = str(Path(base_save_path) / f"fold_{fold_idx + 1}")
    Path(fold_save_path).mkdir(parents=True, exist_ok=True)
    fold_cfg.save_path = fold_save_path

    save_config_snapshot(fold_cfg, fold_save_path)

    train_idx, val_idx = _split_fold_indices(folds, fold_idx)
    print(f"[Fold {fold_idx + 1}] train={len(train_idx)} val={len(val_idx)}")

    # 记录训练前数据划分
    split_root = Path(fold_save_path) / "split_records"
    save_split_dataframe(full_df, train_idx, str(split_root / "train_split.xlsx"))
    save_split_dataframe(full_df, val_idx, str(split_root / "val_split.xlsx"))

    # 准备组件
    loaders = _build_fold_loaders(fold_cfg, full_df, train_idx, val_idx)
    model = build_resnet(
        fold_cfg.model.depth,
        vf_dim=int(fold_cfg.model.vf_dim),
        proj_use_layernorm=bool(fold_cfg.model.proj_use_layernorm),
    ).to(fold_cfg.device)

    losses = build_losses(fold_cfg)
    metrics = build_metrics(fold_cfg)

    optimizer = build_optimizer_for_stage_resnet(model, fold_cfg, stage=1)
    scheduler = build_scheduler(optimizer, fold_cfg)

    callbacks = {
        "switch_s1": build_plateau_switch(fold_cfg, stage=1),
        "build_optimizer_for_stage": build_optimizer_for_stage_resnet,
        "build_scheduler": build_scheduler,
    }

    writers = build_writers(fold_save_path)
    logger = {
        "writers": writers,
        "save_path": fold_save_path,
    }

    # 训练入口
    print(f"\n========== Start Fold {fold_idx + 1}/{len(folds)} ==========")
    try:
        fit(
            model=model,
            loaders=loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            losses=losses,
            metrics=metrics,
            cfg=fold_cfg,
            logger=logger,
        )
    finally:
        close_writers(writers)
    print(f"========== End Fold {fold_idx + 1}/{len(folds)} ==========")


def main():
    cli_args = parse_train_args()
    cfg = load_config(cli_args.config)

    cfg.task_name = cfg.task_name + "_5-fold-cv"

    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA is not available, fallback to CPU.")
        cfg.device = "cpu"

    set_seed(cfg.seed, deterministic=cfg.deterministic)

    k_folds, kfold_seed = _resolve_kfold_params(cfg)
    full_df = pd.read_excel(cfg.data_root)
    folds = make_kfold_splits(full_df, k_folds=k_folds, seed=kfold_seed)

    base_save_path = create_save_path(cfg.save_root, cfg.task_name)

    for fold_idx in range(k_folds):
        _run_one_fold(cfg, full_df, folds, fold_idx, base_save_path)


if __name__ == "__main__":
    main()
