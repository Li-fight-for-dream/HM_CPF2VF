import datetime
from pathlib import Path
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


def build_writers(save_path: str) -> Dict[str, SummaryWriter]:
    tb_root = Path(save_path) / "tensorboard"
    return {
        "train": SummaryWriter(str(tb_root / "train")),
        "val": SummaryWriter(str(tb_root / "val")),
        "test": SummaryWriter(str(tb_root / "test")),
    }


def save_config_snapshot(cfg, save_path: str) -> None:
    """
    保存当前配置
    """
    out = Path(save_path) / "args.txt"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"Arguments used at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for key, value in sorted(cfg.to_dict().items()):
            f.write(f"{key}: {value}\n")


def save_split_dataframe(df, idx, output_path: str) -> None:
    """按索引切片并保存为Excel，用于记录训练前数据划分。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    split_df = df.loc[idx].copy() if idx is not None else df.copy()
    # 保留原始索引，便于追溯划分来源
    split_df = split_df.reset_index(drop=False)
    split_df.to_excel(out, index=False)


def append_epoch_record(save_path: str, lines) -> None:
    out = Path(save_path) / "epochs_record.txt"
    with out.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def append_best_record(save_path: str, model_name: str, stage: int, epoch: int, monitor: str, monitor_value: float, metrics: dict) -> None:
    out = Path(save_path) / "best_models_record.txt"
    with out.open("a", encoding="utf-8") as f:
        f.write(f"epoch={epoch} stage={stage} model={model_name} monitor={monitor} value={monitor_value}\n")
        for k in sorted(metrics.keys()):
            f.write(f"  {k}: {metrics[k]}\n")
        f.write("\n")


def close_writers(writers: Dict[str, SummaryWriter]) -> None:
    for writer in writers.values():
        writer.close()
