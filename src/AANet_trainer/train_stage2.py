from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.AANet_trainer.AA_trainer.aa_train import fit_archetype
from src.AANet_trainer.engine_stage2 import fit
from src.build_callbacks import build_plateau_switch
from src.build_losses import build_losses
from src.build_metrics import build_metrics
from src.build_model import build_model
from src.build_optimizer import build_optimizer_for_stage
from src.build_scheduler import build_scheduler
from src.data_dealer.dataframe_dealer_for_fundus import stratified_split_by_md
from src.data_dealer.dataset_for_fundus import vf_with_fundus_dataset
from src.utils.config import load_config, parse_train_args
from src.utils.create_save_path import create_save_path
from src.utils.logger import (
    build_writers,
    close_writers,
    save_config_snapshot,
    save_split_dataframe,
)
from src.utils.seed import set_seed


def _build_loaders(cfg, full_df):
    train_df, val_df, test_df = stratified_split_by_md(
        full_df,
        ratios=cfg.dataset_ratio,
        seed=int(cfg.seed),
    )

    print("✅ 数据集划分结果：")
    print(f"   [训练集 Train] 数量: {len(train_df)}")
    print(f"   [验证集 Val  ] 数量: {len(val_df)}")
    print(f"   [测试集 Test ] 数量: {len(test_df)}")

    train_dataset = vf_with_fundus_dataset(train_df.reset_index(drop=True), image_root=getattr(cfg, "image_root", None))
    val_dataset = vf_with_fundus_dataset(val_df.reset_index(drop=True), image_root=getattr(cfg, "image_root", None))
    test_dataset = vf_with_fundus_dataset(test_df.reset_index(drop=True), image_root=getattr(cfg, "image_root", None))

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=int(getattr(cfg, "num_workers", 0)),
    )

    return (
        (train_loader, val_loader, test_loader),
        train_df.index.to_numpy(),
        val_df.index.to_numpy(),
        test_df.index.to_numpy(),
    )


def main():
    cli_args = parse_train_args()
    cfg = load_config(cli_args.config)

    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA is not available, fallback to CPU.")
        cfg.device = "cpu"

    set_seed(cfg.seed, deterministic=cfg.deterministic)

    save_path = create_save_path(cfg.save_root, cfg.task_name)
    cfg.save_path = save_path
    save_config_snapshot(cfg, save_path)

    full_df = pd.read_excel(cfg.data_root)
    loaders, train_idx, val_idx, test_idx = _build_loaders(cfg, full_df)

    split_root = Path(save_path) / "data_split_records"
    save_split_dataframe(full_df, train_idx, str(split_root / "train_split.xlsx"))
    save_split_dataframe(full_df, val_idx, str(split_root / "val_split.xlsx"))
    save_split_dataframe(full_df, test_idx, str(split_root / "test_split.xlsx"))

    cfg.model.num_prototypes = int(cfg.archetype.k)
    aa_result = fit_archetype(
        save_path=save_path,
        train_idx=train_idx,
        df=full_df,
        cfg=cfg,
        val_idx=val_idx,
    )
    cfg.model.prototype_model_path = aa_result["model_path"]

    model = build_model(cfg).to(cfg.device)
    losses = build_losses(cfg)
    metrics = build_metrics(cfg)

    optimizer = build_optimizer_for_stage(model, cfg, stage=1)
    scheduler = build_scheduler(optimizer, cfg)

    callbacks = {
        "switch_s1": build_plateau_switch(cfg, stage=1),
        "build_optimizer_for_stage": build_optimizer_for_stage,
        "build_scheduler": build_scheduler,
    }

    writers = build_writers(save_path)
    logger = {
        "writers": writers,
        "save_path": save_path,
    }

    try:
        fit(
            model=model,
            loaders=loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            losses=losses,
            metrics=metrics,
            cfg=cfg,
            logger=logger,
        )
    finally:
        close_writers(writers)


if __name__ == "__main__":
    main()
