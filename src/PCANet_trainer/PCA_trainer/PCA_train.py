import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.decomposition import PCA

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


def _pca_cfg_from_yaml(cfg) -> Dict[str, Any]:
    """
    PCA 配置优先从 cfg.pca 读取；
    若不存在，则回退到 archetype.k 作为主成分数。
    """
    pca_cfg = getattr(cfg, "pca", None)

    if pca_cfg is not None:
        n_components = int(getattr(pca_cfg, "n_components"))
        svd_solver = getattr(pca_cfg, "svd_solver", "auto")
        whiten = bool(getattr(pca_cfg, "whiten", False))
        random_state = getattr(pca_cfg, "random_state", None)
        if random_state is not None:
            random_state = int(random_state)
        return {
            "n_components": n_components,
            "svd_solver": svd_solver,
            "whiten": whiten,
            "random_state": random_state,
        }

    aa_cfg = getattr(cfg, "archetype", None)
    if aa_cfg is None:
        raise ValueError("Missing `pca` section in yaml config")

    return {
        "n_components": int(aa_cfg.k),
        "svd_solver": "auto",
        "whiten": False,
        "random_state": int(getattr(aa_cfg, "random_state", 2026)),
    }


def train_pca_model(X_train: np.ndarray, cfg: Dict[str, Any]):
    """仅在训练集上拟合 PCA，返回训练产物。"""
    model = PCA(
        n_components=cfg["n_components"],
        svd_solver=cfg["svd_solver"],
        whiten=cfg["whiten"],
        random_state=cfg["random_state"],
    )
    train_code = model.fit_transform(X_train)  # (n_train, k)
    components = model.components_  # (k, 52)
    X_train_hat = model.inverse_transform(train_code)  # (n_train, 52)
    return model, components, train_code, X_train_hat


def evaluate_pca_model(X_true: np.ndarray, X_hat: np.ndarray, prefix: str) -> Dict[str, float]:
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


def fit_pca(
    save_path: str,
    train_idx,
    df,
    cfg,
    val_idx=None,
) -> Dict[str, Any]:
    """拟合 PCA 并保存训练结果。"""
    save_dir = Path(save_path) / "pca"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df = df.loc[train_idx].reset_index(drop=True)
    X_train = _extract_vf_matrix(train_df)

    # 训练
    pca_cfg = _pca_cfg_from_yaml(cfg)
    model, components, train_code, X_train_hat = train_pca_model(X_train, pca_cfg)

    # 评估
    metrics: Dict[str, Any] = {}
    metrics.update(evaluate_pca_model(X_train, X_train_hat, prefix="train"))
    metrics["explained_variance_ratio_sum"] = float(np.sum(model.explained_variance_ratio_))  # 保留信息量（0，1）之间

    val_code = None
    X_val_hat = None
    if val_idx is not None and len(val_idx) > 0:
        val_df = df.loc[val_idx].reset_index(drop=True)
        X_val = _extract_vf_matrix(val_df)
        val_code = model.transform(X_val)
        X_val_hat = model.inverse_transform(val_code)
        metrics.update(evaluate_pca_model(X_val, X_val_hat, prefix="val"))

    # 结果保存
    k = pca_cfg["n_components"]
    model_path = save_dir / f"PCA_k{k}_model.joblib"
    joblib.dump(model, model_path)
    np.save(save_dir / "components.npy", components)
    np.save(save_dir / "explained_variance.npy", model.explained_variance_)
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
        "pca_cfg": pca_cfg,
    }
    with (save_dir / "pca_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[PCA] saved model={model_path} | "
        f"train_rmse={metrics.get('train_rmse')} | "
        f"val_rmse={metrics.get('val_rmse')} | "
        f"evr_sum={metrics.get('explained_variance_ratio_sum')}"
    )

    return {
        "model_path": str(model_path),
        "metrics": metrics,
        "save_dir": str(save_dir),
    }
