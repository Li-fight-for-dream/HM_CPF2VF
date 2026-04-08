import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from archetypes import AA

from src.data_dealer.dataset_for_fundus import COL_LATERALITY, vf_with_fundus_dataset


def _extract_vf_matrix(df):
    """从dataframe中提取VF（52维）矩阵，不加载图像。"""
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


def _aa_cfg_from_yaml(cfg) -> Dict[str, Any]:
    """AA配置统一从 cfg.archetype 读取。"""
    if not hasattr(cfg, "archetype"):
        raise ValueError("Missing `archetype` section in yaml config")

    aa = cfg.archetype
    return {
        "k": int(aa.k),
        "method": aa.method,
        "init": aa.init,
        "n_init": int(aa.n_init),
        "max_iter": int(aa.max_iter),
        "tol": float(aa.tol),
        "random_state": int(aa.random_state),
        "method_params": getattr(aa, "method_params", None),
    }


def train_archetype_model(X_train: np.ndarray, cfg: Dict[str, Any]):
    """仅在训练集上拟合原型，返回训练产物。"""
    model = AA(
        n_archetypes=cfg["k"],
        method=cfg["method"],
        init=cfg["init"],
        n_init=cfg["n_init"],
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
        random_state=cfg["random_state"],
        method_params=cfg["method_params"],
    )

    train_coeff = model.fit_transform(X_train)  # (n_train, k)
    archetypes = model.archetypes_  # (k, 52)
    X_train_hat = train_coeff @ archetypes  # (n_train, 52)
    return model, archetypes, train_coeff, X_train_hat


def evaluate_archetype_model(X_true: np.ndarray, X_hat: np.ndarray, prefix: str) -> Dict[str, float]:
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


def fit_archetype(
    save_path: str,
    train_idx,
    df,
    cfg,
    val_idx=None,
) -> Dict[str, Any]:
    """拟合AA原型并保存训练结果。"""
    save_dir = Path(save_path) / "archetype"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df = df.loc[train_idx].reset_index(drop=True)
    X_train = _extract_vf_matrix(train_df)

    # 训练
    aa_cfg = _aa_cfg_from_yaml(cfg)
    model, archetypes, train_coeff, X_train_hat = train_archetype_model(X_train, aa_cfg)

    # 评估
    metrics: Dict[str, Any] = {}
    metrics.update(evaluate_archetype_model(X_train, X_train_hat, prefix="train"))

    val_coeff = None
    X_val_hat = None
    if val_idx is not None and len(val_idx) > 0:
        val_df = df.loc[val_idx].reset_index(drop=True)
        X_val = _extract_vf_matrix(val_df)
        val_coeff = model.transform(X_val)
        X_val_hat = val_coeff @ archetypes
        metrics.update(evaluate_archetype_model(X_val, X_val_hat, prefix="val"))

    # 结果保存
    k = aa_cfg["k"]
    model_path = save_dir / f"AA_k{k}_model.joblib"
    joblib.dump(model, model_path)
    np.save(save_dir / "archetypes.npy", archetypes)
    np.save(save_dir / "train_coeff.npy", train_coeff)
    np.save(save_dir / "train_recon.npy", X_train_hat)
    if val_coeff is not None:
        np.save(save_dir / "val_coeff.npy", val_coeff)
    if X_val_hat is not None:
        np.save(save_dir / "val_recon.npy", X_val_hat)

    summary = {
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(0 if val_idx is None else len(val_idx)),
        "model_path": str(model_path),
        "metrics": metrics,
    }

    with (save_dir / "aa_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[AA] saved model={model_path} | "
        f"train_rmse={metrics.get('train_rmse')} | val_rmse={metrics.get('val_rmse')}"
    )

    return {
        "model_path": str(model_path),
        "metrics": metrics,
        "save_dir": str(save_dir),
    }
