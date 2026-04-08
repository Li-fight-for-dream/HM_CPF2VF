import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.resnet_PCA import build_resnet_pca
from src.data_dealer.dataset_for_fundus import vf_with_fundus_dataset
from src.utils.config import load_config, parse_train_args
from src.utils.logger import save_config_snapshot
from src.utils.seed import set_seed

# 命令行启动命令⬇
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# python -m src.PCANet_trainer.inference --config configs/recipe_PCANet.yaml

def _resolve_path(base_dir: str, path_or_name: str, must_exist: bool = True) -> str:
    p = Path(path_or_name)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path(path_or_name))
        candidates.append(Path(base_dir) / path_or_name)

    for c in candidates:
        if c.exists():
            return str(c)

    if must_exist:
        tried = " | ".join([str(c) for c in candidates])
        raise FileNotFoundError(f"Path not found: {path_or_name}. tried: {tried}")

    return str(candidates[-1])


def _resolve_pca_model_path(cfg, path_for_eval: str) -> str:
    configured = getattr(cfg.model, "pca_model_path", None)
    if configured:
        return _resolve_path(path_for_eval, configured, must_exist=True)

    default_path = Path(path_for_eval) / "pca" / f"PCA_k{int(cfg.model.pca_n_components)}_model.joblib"
    if default_path.exists():
        return str(default_path)

    candidates = sorted((Path(path_for_eval) / "pca").glob("*_model.joblib"))
    if len(candidates) == 1:
        return str(candidates[0])

    if len(candidates) == 0:
        raise FileNotFoundError("Cannot resolve PCA model path: no *_model.joblib found under <path_for_eval>/pca")

    raise RuntimeError(f"Multiple PCA model files found, please set model.pca_model_path explicitly: {candidates}")


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
            if k.startswith("module."):
                stripped[k[len("module."):]] = v
            else:
                stripped[k] = v
        model.load_state_dict(stripped, strict=True)
        return

    raise RuntimeError(f"Unsupported checkpoint format in {ckpt_path}")


def _load_split_dataframes(path_for_eval: str) -> Dict[str, pd.DataFrame]:
    split_root = Path(path_for_eval) / "data_split_records"
    split_files = {
        "train": split_root / "train_split.xlsx",
        "val": split_root / "val_split.xlsx",
        "test": split_root / "test_split.xlsx",
    }

    split_dfs: Dict[str, pd.DataFrame] = {}
    for split_name, split_file in split_files.items():
        if not split_file.exists():
            raise FileNotFoundError(
                f"Required split file not found: {split_file}. "
                "Please ensure training has saved data_split_records first."
            )
        split_dfs[split_name] = pd.read_excel(split_file)

    print("✅ 使用已有划分文件进行推理：")
    for split_name in ["train", "val", "test"]:
        print(f"   [{split_name}] 样本数: {len(split_dfs[split_name])}")

    return split_dfs


def _build_loader(df: pd.DataFrame, cfg) -> DataLoader:
    ds = vf_with_fundus_dataset(df.reset_index(drop=True), image_root=getattr(cfg, "image_root", None))
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=int(getattr(cfg, "num_workers", 0)),
    )


def _to_json_list(arr: np.ndarray) -> str:
    return json.dumps([float(x) for x in arr.tolist()], ensure_ascii=False)


def _run_inference_and_attach(
    model,
    loader: DataLoader,
    source_df: pd.DataFrame,
    split_name: str,
    device,
) -> pd.DataFrame:
    model.eval()

    vf_pred_list = []
    vf_true_list = []
    vf_recon_list = []
    md_pred_list = []
    md_true_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"infer_{split_name}", leave=False):
            inputs = batch["fundus_img"].to(device)
            outputs = model(inputs)

            vf_pred = outputs["VF_pred"].detach().cpu().numpy()
            vf_true_tensor = batch["VF_tensor"].to(device)
            vf_true = vf_true_tensor.detach().cpu().numpy()
            md_pred = outputs["md_pred"].detach().cpu().view(-1).numpy()
            md_true = batch["md_tensor"].detach().cpu().view(-1).numpy()
            core_model = model.module if hasattr(model, "module") else model
            pca_coef_true = core_model.pca_transform(vf_true_tensor, normalized=True).detach().cpu().numpy()
            vf_recon = core_model.pca_inverse_transform(
                torch.from_numpy(pca_coef_true).to(device=vf_true_tensor.device, dtype=vf_true_tensor.dtype)
            ).detach().cpu().numpy()

            vf_pred_list.extend(vf_pred)
            vf_true_list.extend(vf_true)
            vf_recon_list.extend(vf_recon)
            md_pred_list.extend(md_pred.tolist())
            md_true_list.extend(md_true.tolist())

    n = len(source_df)
    if not (
        len(vf_pred_list)
        == len(vf_true_list)
        == len(vf_recon_list)
        == len(md_pred_list)
        == len(md_true_list)
        == n
    ):
        raise RuntimeError(
            f"Inference size mismatch on split={split_name}: "
            f"pred_vf={len(vf_pred_list)}, true_vf={len(vf_true_list)}, "
            f"pred_md={len(md_pred_list)}, true_md={len(md_true_list)}, df={n}"
        )

    out_df = source_df.copy()
    out_df["split"] = split_name
    out_df["VF_true"] = [_to_json_list(v) for v in vf_true_list]
    out_df["VF_pred"] = [_to_json_list(v) for v in vf_pred_list]
    out_df["VF_recon"] = [_to_json_list(v) for v in vf_recon_list]
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
    pca_model_path = _resolve_pca_model_path(cfg, path_for_eval)

    eval_save_path = Path(path_for_eval) / "eval" / Path(eval_model_path).stem
    eval_save_path.mkdir(parents=True, exist_ok=True)

    cfg.save_path = str(eval_save_path)
    save_config_snapshot(cfg, str(eval_save_path))

    split_dfs = _load_split_dataframes(path_for_eval)

    model = build_resnet_pca(
        cfg.model.depth,
        vf_dim=int(cfg.model.vf_dim),
        pca_n_components=int(cfg.model.pca_n_components),
        proj_use_layernorm=bool(cfg.model.proj_use_layernorm),
    ).to(cfg.device)
    model.load_pca_from_joblib(pca_model_path)
    _load_state_dict_flexible(model, eval_model_path, cfg.device)

    saved_files = []
    for split_name in ["train", "val", "test"]:
        split_df = split_dfs[split_name]
        loader = _build_loader(split_df, cfg)
        pred_df = _run_inference_and_attach(
            model=model,
            loader=loader,
            source_df=split_df,
            split_name=split_name,
            device=cfg.device,
        )

        out_xlsx = eval_save_path / f"{split_name}_with_preds.xlsx"
        pred_df.to_excel(out_xlsx, index=False)
        saved_files.append(str(out_xlsx))

    print("\n========== Eval Export Done ==========")
    print(f"Model checkpoint: {eval_model_path}")
    print(f"PCA model: {pca_model_path}")
    for fp in saved_files:
        print(f"Saved: {fp}")


if __name__ == "__main__":
    main()
