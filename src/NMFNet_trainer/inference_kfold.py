import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.resnet_NMF import build_resnet_nmf
from src.data_dealer.dataset_for_fundus import vf_with_fundus_dataset
from src.utils.config import load_config, parse_train_args
from src.utils.logger import save_config_snapshot
from src.utils.seed import set_seed

# 命令行启动命令⬇
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# python -m src.NMFNet_trainer.inference_kfold --config configs/recipe_NMFNet.yaml

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


def _resolve_nmf_model_path(cfg, fold_dir: str) -> str:
    nmf_dir = Path(fold_dir) / "nmf"
    if not nmf_dir.exists():
        raise FileNotFoundError(f"NMF folder not found: {nmf_dir}")

    expected = nmf_dir / f"NMF_k{int(cfg.model.num_components)}_model.joblib"
    if expected.exists():
        return str(expected)

    candidates = sorted(nmf_dir.glob("*_model.joblib"))
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) == 0:
        raise FileNotFoundError(f"No *_model.joblib found under {nmf_dir}")
    raise RuntimeError(f"Multiple NMF model files found, please keep only one: {candidates}")


def _to_non_negative(X: np.ndarray, strategy: str) -> np.ndarray:
    s = str(strategy).lower()
    if s == "raise":
        if np.any(X < 0):
            mn = float(np.min(X))
            raise ValueError(f"NMF input contains negative values (min={mn}).")
        return X
    if s == "shift":
        mn = float(np.min(X))
        return X - mn if mn < 0 else X
    if s == "clip":
        return np.clip(X, a_min=0.0, a_max=None)
    raise ValueError(f"Unsupported non_negative strategy: {strategy}")


def _load_nmf_transformer(nmf_model_path: str):
    m = joblib.load(nmf_model_path)
    if hasattr(m, "model") and hasattr(m.model, "transform"):
        m = m.model
    if not hasattr(m, "transform"):
        raise ValueError("Loaded NMF object does not provide `transform`.")
    return m


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


def _discover_fold_dirs(path_for_eval: str) -> List[Path]:
    base = Path(path_for_eval)
    if not base.exists():
        raise FileNotFoundError(f"path_for_eval not found: {path_for_eval}")

    fold_dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    if len(fold_dirs) == 0:
        raise RuntimeError(f"No fold directories found under: {path_for_eval}")

    def _fold_key(p: Path):
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return 10**9

    return sorted(fold_dirs, key=_fold_key)


def _load_fold_split_dataframes(fold_dir: Path) -> Dict[str, pd.DataFrame]:
    candidate_roots = [fold_dir / "split_records", fold_dir / "data_split_records"]
    split_root = None
    for r in candidate_roots:
        if r.exists():
            split_root = r
            break

    if split_root is None:
        raise FileNotFoundError(f"No split record folder found in {fold_dir}")

    split_dfs: Dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        split_file = split_root / f"{split_name}_split.xlsx"
        if split_file.exists():
            split_dfs[split_name] = pd.read_excel(split_file)

    if "train" not in split_dfs or "val" not in split_dfs:
        raise RuntimeError(f"Fold {fold_dir.name} must contain train/val split files under {split_root}")
    return split_dfs


def _build_loader(df: pd.DataFrame, cfg) -> DataLoader:
    ds = vf_with_fundus_dataset(df.reset_index(drop=True), image_root=getattr(cfg, "image_root", None))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=int(getattr(cfg, "num_workers", 0)))


def _to_json_list(arr: np.ndarray) -> str:
    return json.dumps([float(x) for x in arr.tolist()], ensure_ascii=False)


def _run_inference_and_attach(
    model,
    loader: DataLoader,
    source_df: pd.DataFrame,
    split_name: str,
    device,
    fold_name: str,
    nmf_model,
    non_negative_strategy: str,
) -> pd.DataFrame:
    model.eval()
    vf_pred_list, vf_true_list, vf_recon_list = [], [], []
    md_pred_list, md_true_list = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"infer_{fold_name}_{split_name}", leave=False):
            inputs = batch["fundus_img"].to(device)
            outputs = model(inputs)

            vf_pred = outputs["VF_pred"].detach().cpu().numpy()
            vf_true = batch["VF_tensor"].detach().cpu().numpy()
            md_pred = outputs["md_pred"].detach().cpu().view(-1).numpy()
            md_true = batch["md_tensor"].detach().cpu().view(-1).numpy()
            core_model = model.module if hasattr(model, "module") else model
            H = core_model.H.detach().cpu().numpy()
            vf_nn = _to_non_negative(vf_true, non_negative_strategy)
            nmf_coef_true = np.clip(np.asarray(nmf_model.transform(vf_nn), dtype=np.float32), a_min=0.0, a_max=None)
            vf_recon = nmf_coef_true @ H

            vf_pred_list.extend(vf_pred)
            vf_true_list.extend(vf_true)
            vf_recon_list.extend(vf_recon)
            md_pred_list.extend(md_pred.tolist())
            md_true_list.extend(md_true.tolist())

    out_df = source_df.copy()
    out_df["split"] = split_name
    out_df["VF_true"] = [_to_json_list(v) for v in vf_true_list]
    out_df["VF_pred"] = [_to_json_list(v) for v in vf_pred_list]
    out_df["VF_recon"] = [_to_json_list(v) for v in vf_recon_list]
    out_df["md_true"] = [float(v) for v in md_true_list]
    out_df["md_pred"] = [float(v) for v in md_pred_list]
    return out_df


def _run_one_fold(cfg, fold_dir: Path, eval_model_name: str, device):
    fold_model_path = _resolve_path(str(fold_dir), eval_model_name, must_exist=True)
    fold_nmf_path = _resolve_nmf_model_path(cfg, str(fold_dir))

    inference_root = fold_dir / "inference" / Path(fold_model_path).stem
    inference_root.mkdir(parents=True, exist_ok=True)

    cfg.save_path = str(inference_root)
    save_config_snapshot(cfg, str(inference_root))

    split_dfs = _load_fold_split_dataframes(fold_dir)

    model = build_resnet_nmf(
        cfg.model.depth,
        vf_dim=int(cfg.model.vf_dim),
        num_components=int(cfg.model.num_components),
        proj_use_layernorm=bool(cfg.model.proj_use_layernorm),
        w_activation=str(getattr(cfg.model, "w_activation", "softplus")),
    ).to(device)
    model.load_components_from_joblib(fold_nmf_path)
    _load_state_dict_flexible(model, fold_model_path, device)
    nmf_model = _load_nmf_transformer(fold_nmf_path)
    non_negative_strategy = str(getattr(getattr(cfg, "nmf", None), "non_negative", "clip"))

    print(f"\n[Fold {fold_dir.name}] model={fold_model_path}")
    print(f"[Fold {fold_dir.name}] nmf={fold_nmf_path}")

    for split_name, split_df in split_dfs.items():
        loader = _build_loader(split_df, cfg)
        pred_df = _run_inference_and_attach(
            model,
            loader,
            split_df,
            split_name,
            device,
            fold_dir.name,
            nmf_model,
            non_negative_strategy,
        )
        out_xlsx = inference_root / f"{split_name}_with_preds.xlsx"
        pred_df.to_excel(out_xlsx, index=False)
        print(f"[Fold {fold_dir.name}] saved: {out_xlsx}")


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

    path_for_eval = str(getattr(test_cfg, "path_kfold_for_eval", getattr(test_cfg, "path_for_eval", ""))).strip()
    eval_model_name = str(getattr(test_cfg, "eval_model", "")).strip()
    if len(path_for_eval) == 0:
        raise ValueError("test.path_for_eval is required.")
    if len(eval_model_name) == 0:
        raise ValueError("test.eval_model is required.")

    fold_dirs = _discover_fold_dirs(path_for_eval)
    print(f"发现 {len(fold_dirs)} 个fold: {[p.name for p in fold_dirs]}")

    for fold_dir in fold_dirs:
        _run_one_fold(cfg, fold_dir, eval_model_name, cfg.device)


if __name__ == "__main__":
    main()
