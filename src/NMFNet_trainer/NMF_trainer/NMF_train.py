import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.decomposition import NMF

from src.data_dealer.dataset_for_fundus import COL_LATERALITY, vf_with_fundus_dataset


def _extract_vf_matrix(df):
    """从 dataframe 中提取 VF（52维）矩阵，不加载图像。"""
    ds = vf_with_fundus_dataset(df, image_root=None)
    vf_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        vf = ds.clean_VF_in_table_to_list(row)
        if row[COL_LATERALITY] == "L":
            vf = ds.trun_VF_from_left_to_right(vf)
        else:
            vf = ds.remove_right_blind_spots(vf)
        vf_list.append(ds.get_VF_tensor(vf).numpy())
    return np.stack(vf_list, axis=0)


def _to_non_negative(X: np.ndarray, strategy: str) -> np.ndarray:
    """
    NMF 要求输入非负。默认 clip 到 >=0，避免少量脏数据导致训练失败。
    """
    s = strategy.lower()
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


def _nmf_cfg_from_yaml(cfg) -> Dict[str, Any]:
    """
    NMF 配置优先从 cfg.nmf 读取；
    若不存在，则回退到 archetype.k 作为分量数。
    """
    nmf_cfg = getattr(cfg, "nmf", None)

    if nmf_cfg is not None:
        random_state = getattr(nmf_cfg, "random_state", None)
        if random_state is not None:
            random_state = int(random_state)
        return {
            "n_components": int(getattr(nmf_cfg, "n_components")),
            "init": getattr(nmf_cfg, "init", "nndsvda"),
            "solver": getattr(nmf_cfg, "solver", "cd"),
            "beta_loss": getattr(nmf_cfg, "beta_loss", "frobenius"),
            "max_iter": int(getattr(nmf_cfg, "max_iter", 500)),
            "tol": float(getattr(nmf_cfg, "tol", 1e-4)),
            "alpha_W": float(getattr(nmf_cfg, "alpha_w", 0.0)),
            "alpha_H": float(getattr(nmf_cfg, "alpha_h", 0.0)),
            "l1_ratio": float(getattr(nmf_cfg, "l1_ratio", 0.0)),
            "shuffle": bool(getattr(nmf_cfg, "shuffle", False)),
            "random_state": random_state,
            "non_negative": getattr(nmf_cfg, "non_negative", "clip"),
        }

    aa_cfg = getattr(cfg, "archetype", None)
    if aa_cfg is None:
        raise ValueError("Missing `nmf` section in yaml config")

    return {
        "n_components": int(aa_cfg.k),
        "init": "nndsvda",
        "solver": "cd",
        "beta_loss": "frobenius",
        "max_iter": 500,
        "tol": 1e-4,
        "alpha_W": 0.0,
        "alpha_H": 0.0,
        "l1_ratio": 0.0,
        "shuffle": False,
        "random_state": int(getattr(aa_cfg, "random_state", 2026)),
        "non_negative": "clip",
    }


def train_nmf_model(X_train: np.ndarray, cfg: Dict[str, Any]):
    """仅在训练集上拟合 NMF，返回训练产物。"""
    X_train_nn = _to_non_negative(X_train, cfg["non_negative"])

    model = NMF(
        n_components=cfg["n_components"],
        init=cfg["init"],
        solver=cfg["solver"],
        beta_loss=cfg["beta_loss"],
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
        alpha_W=cfg["alpha_W"],
        alpha_H=cfg["alpha_H"],
        l1_ratio=cfg["l1_ratio"],
        shuffle=cfg["shuffle"],
        random_state=cfg["random_state"],
    )

    train_code = model.fit_transform(X_train_nn)  # W: (n_train, k)
    components = model.components_  # H: (k, 52)
    X_train_hat = train_code @ components  # (n_train, 52)
    return model, components, train_code, X_train_hat, X_train_nn


def evaluate_nmf_model(X_true: np.ndarray, X_hat: np.ndarray, prefix: str) -> Dict[str, float]:
    """评估重构误差。"""
    diff = X_hat - X_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae": mae,
    }


def fit_nmf(
    save_path: str,
    train_idx,
    df,
    cfg,
    val_idx=None,
) -> Dict[str, Any]:
    """拟合 NMF 并保存训练结果。"""
    save_dir = Path(save_path) / "nmf"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df = df.loc[train_idx].reset_index(drop=True)
    X_train = _extract_vf_matrix(train_df)

    # 训练
    nmf_cfg = _nmf_cfg_from_yaml(cfg)
    model, components, train_code, X_train_hat, X_train_nn = train_nmf_model(X_train, nmf_cfg)

    # 评估
    metrics: Dict[str, Any] = {}
    metrics.update(evaluate_nmf_model(X_train_nn, X_train_hat, prefix="train"))
    metrics["reconstruction_err"] = float(getattr(model, "reconstruction_err_", np.nan))
    metrics["n_iter"] = int(getattr(model, "n_iter_", 0))

    val_code = None
    X_val_hat = None
    if val_idx is not None and len(val_idx) > 0:
        val_df = df.loc[val_idx].reset_index(drop=True)
        X_val = _extract_vf_matrix(val_df)
        X_val_nn = _to_non_negative(X_val, nmf_cfg["non_negative"])
        val_code = model.transform(X_val_nn)
        X_val_hat = val_code @ components
        metrics.update(evaluate_nmf_model(X_val_nn, X_val_hat, prefix="val"))

    # 结果保存
    k = nmf_cfg["n_components"]
    model_path = save_dir / f"NMF_k{k}_model.joblib"
    joblib.dump(model, model_path)
    np.save(save_dir / "components.npy", components)
    np.save(save_dir / "train_code.npy", train_code)
    np.save(save_dir / "train_recon.npy", X_train_hat)
    if val_code is not None:
        np.save(save_dir / "val_code.npy", val_code)
    if X_val_hat is not None:
        np.save(save_dir / "val_recon.npy", X_val_hat)

    summary = {
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(0 if val_idx is None else len(val_idx)),
        "model_path": str(model_path),
        "metrics": metrics,
        "nmf_cfg": nmf_cfg,
    }
    with (save_dir / "nmf_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[NMF] saved model={model_path} | "
        f"train_rmse={metrics.get('train_rmse')} | "
        f"val_rmse={metrics.get('val_rmse')} | "
        f"recon_err={metrics.get('reconstruction_err')}"
    )

    return {
        "model_path": str(model_path),
        "metrics": metrics,
        "save_dir": str(save_dir),
    }
