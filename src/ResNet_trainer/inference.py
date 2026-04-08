import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.resnet import build_resnet
from src.data_dealer.dataset_for_fundus import vf_with_fundus_dataset
from src.utils.config import load_config, parse_train_args
from src.utils.logger import save_config_snapshot
from src.utils.seed import set_seed

# 命令行启动命令⬇
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# python -m src.ResNet_trainer.inference --config configs/recipe_ResNet.yaml


def _resolve_path(base_dir: str, path_or_name: str, must_exist: bool = True) -> str:
    p = Path(path_or_name)
    candidates = [p] if p.is_absolute() else [Path(path_or_name), Path(base_dir) / path_or_name]

    for c in candidates:
        if c.exists():
            return str(c)

    if must_exist:
        tried = " | ".join([str(c) for c in candidates])
        raise FileNotFoundError(f"Path not found: {path_or_name}. tried: {tried}")
    return str(candidates[-1])


def _load_state_dict_flexible(model, ckpt_path: str, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    if isinstance(state, dict):
        stripped = {}
        for k, v in state.items():
            stripped[k[len("module."):]] = v if k.startswith("module.") else v
        model.load_state_dict(stripped, strict=True)
        return

    raise RuntimeError(f"Unsupported checkpoint format in {ckpt_path}")


def _load_split_dataframes(path_for_eval: str) -> Dict[str, pd.DataFrame]:
    candidate_roots = [Path(path_for_eval) / "data_split_records", Path(path_for_eval) / "split_records"]
    split_root = None
    for r in candidate_roots:
        if r.exists():
            split_root = r
            break
    if split_root is None:
        raise FileNotFoundError(f"No split record folder found under {path_for_eval}")

    split_dfs: Dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        split_file = split_root / f"{split_name}_split.xlsx"
        if split_file.exists():
            split_dfs[split_name] = pd.read_excel(split_file)

    if "train" not in split_dfs or "val" not in split_dfs:
        raise RuntimeError(f"train/val split files are required under {split_root}")

    print("✅ 使用已有划分文件进行推理：")
    for split_name in ["train", "val", "test"]:
        if split_name in split_dfs:
            print(f"   [{split_name}] 样本数: {len(split_dfs[split_name])}")

    return split_dfs


def _build_loader(df: pd.DataFrame, cfg) -> DataLoader:
    ds = vf_with_fundus_dataset(df.reset_index(drop=True), image_root=getattr(cfg, "image_root", None))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=int(getattr(cfg, "num_workers", 0)))


def _to_json_list(arr: np.ndarray) -> str:
    return json.dumps([float(x) for x in arr.tolist()], ensure_ascii=False)


def _run_inference_and_attach(model, loader: DataLoader, source_df: pd.DataFrame, split_name: str, device) -> pd.DataFrame:
    model.eval()
    vf_pred_list, vf_true_list, md_pred_list, md_true_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"infer_{split_name}", leave=False):
            inputs = batch["fundus_img"].to(device)
            outputs = model(inputs)

            vf_pred = outputs["VF_pred"].detach().cpu().numpy()
            vf_true = batch["VF_tensor"].detach().cpu().numpy()
            md_pred = outputs["md_pred"].detach().cpu().view(-1).numpy()
            md_true = batch["md_tensor"].detach().cpu().view(-1).numpy()

            vf_pred_list.extend(vf_pred)
            vf_true_list.extend(vf_true)
            md_pred_list.extend(md_pred.tolist())
            md_true_list.extend(md_true.tolist())

    n = len(source_df)
    if not (len(vf_pred_list) == len(vf_true_list) == len(md_pred_list) == len(md_true_list) == n):
        raise RuntimeError(f"Inference size mismatch on split={split_name}")

    out_df = source_df.copy()
    out_df["split"] = split_name
    out_df["VF_true"] = [_to_json_list(v) for v in vf_true_list]
    out_df["VF_pred"] = [_to_json_list(v) for v in vf_pred_list]
    out_df["md_true"] = [float(v) for v in md_true_list]
    out_df["md_pred"] = [float(v) for v in md_pred_list]
    return out_df


def main():
    cli_args = parse_train_args()
    cfg = load_config(cli_args.config)

    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA is not available, fallback to CPU.")
        cfg.device = "cpu"

    set_seed(cfg.seed, deterministic=cfg.deterministic)

    test_cfg = getattr(cfg, "test", None)
    if test_cfg is None:
        raise ValueError("Missing `test` section in yaml config.")

    path_for_eval = str(getattr(test_cfg, "path_for_eval", "")).strip()
    eval_model_name = str(getattr(test_cfg, "eval_model", "")).strip()
    if len(path_for_eval) == 0:
        raise ValueError("test.path_for_eval is required.")
    if len(eval_model_name) == 0:
        raise ValueError("test.eval_model is required.")

    eval_model_path = _resolve_path(path_for_eval, eval_model_name, must_exist=True)
    inference_root = Path(path_for_eval) / "inference" / Path(eval_model_path).stem
    inference_root.mkdir(parents=True, exist_ok=True)

    cfg.save_path = str(inference_root)
    save_config_snapshot(cfg, str(inference_root))

    split_dfs = _load_split_dataframes(path_for_eval)

    model = build_resnet(
        cfg.model.depth,
        vf_dim=int(cfg.model.vf_dim),
        proj_use_layernorm=bool(cfg.model.proj_use_layernorm),
    ).to(cfg.device)
    _load_state_dict_flexible(model, eval_model_path, cfg.device)

    for split_name in ["train", "val", "test"]:
        if split_name not in split_dfs:
            continue
        split_df = split_dfs[split_name]
        loader = _build_loader(split_df, cfg)
        pred_df = _run_inference_and_attach(model, loader, split_df, split_name, cfg.device)
        out_xlsx = inference_root / f"{split_name}_with_preds.xlsx"
        pred_df.to_excel(out_xlsx, index=False)
        print(f"Saved: {out_xlsx}")


if __name__ == "__main__":
    main()
