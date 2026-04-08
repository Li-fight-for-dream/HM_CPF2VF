import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    # python -m src.excel_xlsx
    from src.plot import get_display_cmap, prepare_display_grid, vf52_to_vf54
except ModuleNotFoundError:
    # python src/excel_xlsx.py
    CUR_DIR = Path(__file__).resolve().parent
    if str(CUR_DIR) not in sys.path:
        sys.path.insert(0, str(CUR_DIR))
    from plot import get_display_cmap, prepare_display_grid, vf52_to_vf54


PATIENT_ID_CANDIDATES = ["vf_id", "pid", "patient_id", "PID", "id"]
REQUIRED_COLS = ["VF_true", "VF_pred", "md_true", "md_pred"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model inference xlsx by patient/point levels.")
    parser.add_argument("--xlsx", type=str, 
                        default="save_result_fundus/resnet34_PCANet_pa=3_5-fold-cv/fold_2/inference/best_stage2_md_model/val_with_preds.xlsx", help="Path to inference xlsx.")
    parser.add_argument(
        "--splits",
        type=str,
        default="all",
        help="Comma-separated splits to evaluate, e.g. train,val,test. default=all",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _safe_parse_vector(v) -> Optional[np.ndarray]:
    if isinstance(v, np.ndarray):
        arr = v.astype(np.float32)
    elif isinstance(v, (list, tuple)):
        arr = np.asarray(v, dtype=np.float32)
    elif pd.isna(v):
        return None
    elif isinstance(v, str):
        s = v.strip()
        if len(s) == 0:
            return None
        try:
            obj = json.loads(s)
        except Exception:
            obj = ast.literal_eval(s)
        arr = np.asarray(obj, dtype=np.float32)
    else:
        return None

    arr = arr.reshape(-1)
    if arr.shape[0] == 52:
        return arr
    if arr.shape[0] == 54:
        # 若误传54维，删去右眼盲点位(0-based: 25,34)
        return np.delete(arr, [25, 34]).astype(np.float32)
    return None


def _detect_patient_col(df: pd.DataFrame) -> Optional[str]:
    for c in PATIENT_ID_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _parse_splits_arg(splits_arg: str, available_splits: Sequence[str]) -> List[str]:
    if splits_arg.strip().lower() == "all":
        return sorted(list({str(s) for s in available_splits}))
    asked = [x.strip() for x in splits_arg.split(",") if x.strip()]
    asked_set = set(asked)
    avail_set = set(str(s) for s in available_splits)
    selected = [s for s in asked if s in avail_set]
    if len(selected) == 0:
        raise ValueError(f"No requested splits found. requested={asked}, available={sorted(avail_set)}")
    missing = sorted(list(asked_set - avail_set))
    if missing:
        print(f"[Warn] Missing splits ignored: {missing}")
    return selected


def _patient_level_records(df: pd.DataFrame, patient_col: Optional[str]) -> pd.DataFrame:
    group_key = patient_col if patient_col is not None else "__row_id__"
    if group_key == "__row_id__":
        tmp = df.copy()
        tmp[group_key] = np.arange(len(tmp))
    else:
        tmp = df

    rows = []
    for pid, g in tmp.groupby(group_key, dropna=False):
        vf_true_stack = np.stack(g["VF_true_arr"].to_list(), axis=0)  # (n,52)
        vf_pred_stack = np.stack(g["VF_pred_arr"].to_list(), axis=0)  # (n,52)
        vf_diff = vf_pred_stack - vf_true_stack

        vf_rmse = float(np.sqrt(np.mean(vf_diff ** 2)))
        vf_mae = float(np.mean(np.abs(vf_diff)))

        md_true_mean = float(np.nanmean(g["md_true"].to_numpy(dtype=np.float32)))
        md_pred_mean = float(np.nanmean(g["md_pred"].to_numpy(dtype=np.float32)))

        row = {
            "patient_id": pid,
            "n_samples": int(len(g)),
            "split": str(g["split"].iloc[0]),
            "md_true_mean": md_true_mean,
            "md_pred_mean": md_pred_mean,
            "vf_rmse": vf_rmse,
            "vf_mae": vf_mae,
        }

        has_recon = bool(g["VF_recon_arr"].notna().all())
        if has_recon:
            vf_recon_stack = np.stack(g["VF_recon_arr"].to_list(), axis=0)
            vf_recon_diff = vf_recon_stack - vf_true_stack
            row["vf_recon_rmse"] = float(np.sqrt(np.mean(vf_recon_diff ** 2)))
            row["vf_recon_mae"] = float(np.mean(np.abs(vf_recon_diff)))
        else:
            row["vf_recon_rmse"] = np.nan
            row["vf_recon_mae"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    out["md_group"] = np.where(
        out["md_true_mean"] < -6.0,
        "md<-6",
        np.where(out["md_true_mean"] > -6.0, "md>-6", "md==-6"),
    )
    return out


def _build_patient_group_metrics_json(
    patient_df: pd.DataFrame, work_df: pd.DataFrame, patient_col: Optional[str]
) -> dict:
    group_key = patient_col if patient_col is not None else "__row_id__"
    patient_to_group = dict(zip(patient_df["patient_id"].tolist(), patient_df["md_group"].tolist()))
    df = work_df.copy()
    df["patient_id"] = df[group_key]
    df["md_group"] = df["patient_id"].map(patient_to_group)

    def _metrics_for_group(group_name: str) -> dict:
        p = patient_df[patient_df["md_group"] == group_name].copy()
        s = df[df["md_group"] == group_name].copy()
        if len(p) == 0:
            return {
                "n_patients": 0,
                "n_samples": 0,
                "vf_rmse": None,
                "vf_mae": None,
                "md_rmse": None,
                "md_mae": None,
                "vf_recon_rmse": None,
                "vf_recon_mae": None,
            }

        md_diff = p["md_pred_mean"].to_numpy(dtype=np.float32) - p["md_true_mean"].to_numpy(dtype=np.float32)
        md_rmse = float(np.sqrt(np.mean(md_diff ** 2)))
        md_mae = float(np.mean(np.abs(md_diff)))

        vf_true = np.stack(s["VF_true_arr"].to_list(), axis=0)
        vf_pred = np.stack(s["VF_pred_arr"].to_list(), axis=0)
        vf_err = vf_pred - vf_true
        vf_rmse = float(np.sqrt(np.mean(vf_err ** 2)))
        vf_mae = float(np.mean(np.abs(vf_err)))

        has_recon_all = bool(s["VF_recon_arr"].notna().all())
        if has_recon_all:
            vf_recon = np.stack(s["VF_recon_arr"].to_list(), axis=0)
            vf_recon_err = vf_recon - vf_true
            vf_recon_rmse = float(np.sqrt(np.mean(vf_recon_err ** 2)))
            vf_recon_mae = float(np.mean(np.abs(vf_recon_err)))
        else:
            vf_recon_rmse = None
            vf_recon_mae = None

        return {
            "n_patients": int(len(p)),
            "n_samples": int(len(s)),
            "vf_rmse": vf_rmse,
            "vf_mae": vf_mae,
            "md_rmse": md_rmse,
            "md_mae": md_mae,
            "vf_recon_rmse": vf_recon_rmse,
            "vf_recon_mae": vf_recon_mae,
        }

    return {
        "group_rule": "md_true_mean < -6 vs md_true_mean > -6 (md_true_mean == -6 excluded)",
        "n_patients_total": int(len(patient_df)),
        "n_patients_md_equal_minus6_excluded": int((patient_df["md_group"] == "md==-6").sum()),
        "groups": {
            "md<-6": _metrics_for_group("md<-6"),
            "md>-6": _metrics_for_group("md>-6"),
        },
    }


def _plot_md_scatter(patient_df: pd.DataFrame, save_path: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(
        patient_df["md_group"] == "md<-6",
        "tab:red",
        np.where(patient_df["md_group"] == "md>-6", "tab:blue", "tab:gray"),
    )
    ax.scatter(
        patient_df["md_true_mean"].to_numpy(dtype=np.float32),
        patient_df["md_pred_mean"].to_numpy(dtype=np.float32),
        s=18,
        c=colors,
        alpha=0.65,
        edgecolors="none",
    )

    all_vals = np.concatenate(
        [
            patient_df["md_true_mean"].to_numpy(dtype=np.float32),
            patient_df["md_pred_mean"].to_numpy(dtype=np.float32),
        ]
    )
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = 0.5
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], "k--", lw=1.0)
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlabel("MD True (Patient Mean)")
    ax.set_ylabel("MD Pred (Patient Mean)")
    ax.set_title("MD Scatter (Patient Level)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _vector_to_display_grid(vf52: np.ndarray) -> np.ndarray:
    vf54 = vf52_to_vf54(vf52.astype(np.float32))
    return prepare_display_grid(vf54)


def _sanitize_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))


def _plot_vf_maps_per_sample(
    split_df: pd.DataFrame, split_name: str, images_dir: Path, dpi: int, patient_col: Optional[str]
) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    cmap = get_display_cmap()
    saved = 0
    pid_col = patient_col if patient_col in split_df.columns else None

    use_tqdm = sys.stdout.isatty()
    pbar = tqdm(
        split_df.iterrows(),
        total=len(split_df),
        desc=f"[{split_name}] plotting VF maps",
        unit="sample",
        leave=False,
        disable=not use_tqdm,
    )
    for row_idx, row in pbar:
        vf_true = row["VF_true_arr"]
        vf_pred = row["VF_pred_arr"]
        if vf_true is None or vf_pred is None:
            continue

        has_recon = isinstance(row["VF_recon_arr"], np.ndarray)
        pid_val = row[pid_col] if pid_col is not None else row_idx
        prefix = f"{int(saved):06d}_idx{row_idx}_{_sanitize_name(pid_val)}"

        plots = [
            ("VF True", _vector_to_display_grid(vf_true), 0, 40),
            ("VF Pred", _vector_to_display_grid(vf_pred), 0, 40),
            ("|Err Pred-True|", _vector_to_display_grid(np.abs(vf_pred - vf_true)), 0, 15),
        ]
        if has_recon:
            vf_recon = row["VF_recon_arr"]
            plots = [
                ("VF True", _vector_to_display_grid(vf_true), 0, 40),
                ("VF Pred", _vector_to_display_grid(vf_pred), 0, 40),
                ("VF Recon", _vector_to_display_grid(vf_recon), 0, 40),
                ("|Err Pred-True|", _vector_to_display_grid(np.abs(vf_pred - vf_true)), 0, 15),
                ("|Err Recon-True|", _vector_to_display_grid(np.abs(vf_recon - vf_true)), 0, 15),
            ]

        ncols = len(plots)
        fig, axes = plt.subplots(1, ncols, figsize=(4.1 * ncols, 4.2))
        if ncols == 1:
            axes = [axes]
        for ax, (title, grid, vmin, vmax) in zip(axes, plots):
            im = ax.imshow(grid, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"split={split_name} | idx={row_idx}")
        fig.tight_layout()
        fig.savefig(images_dir / f"{prefix}.png", dpi=dpi)
        plt.close(fig)
        saved += 1
        pbar.set_postfix(saved=saved)
        if (not use_tqdm) and (saved % 20 == 0 or saved == len(split_df)):
            print(f"[Info] split={split_name} plotted {saved}/{len(split_df)} samples")
    pbar.close()
    return saved


def _plot_pointwise_boxplot(split_df: pd.DataFrame, split_name: str, save_path: Path, dpi: int):
    vf_true = np.concatenate(split_df["VF_true_arr"].to_list(), axis=0).astype(np.float32)
    vf_pred = np.concatenate(split_df["VF_pred_arr"].to_list(), axis=0).astype(np.float32)

    m = np.isfinite(vf_true) & np.isfinite(vf_pred)
    vf_true = vf_true[m]
    vf_pred = vf_pred[m]

    true_bins = np.clip(np.rint(vf_true), 0, 40).astype(np.int32)
    grouped = [vf_pred[true_bins == i] for i in range(41)]
    if all(arr.size == 0 for arr in grouped):
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    data = [arr if arr.size > 0 else np.array([np.nan], dtype=np.float32) for arr in grouped]
    ax.boxplot(
        data,
        positions=np.arange(41),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#8ecae6", "alpha": 0.75},
        medianprops={"color": "#023047", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    ax.set_xlim(-0.5, 40.5)
    ax.set_xticks(np.arange(0, 41, 2))
    ax.set_xlabel("True VF Value (dB, rounded to integer)")
    ax.set_ylabel("Pred VF Value (dB)")
    ax.set_title(f"Pred Distribution by True VF Value | split={split_name}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def main():
    t0 = time.time()
    args = parse_args()
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"xlsx file not found: {xlsx_path}")

    out_dir = xlsx_path.parent / f"{xlsx_path.stem}_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_excel = out_dir / "patient_level_metrics.xlsx"
    if legacy_excel.exists():
        legacy_excel.unlink()
    for legacy_png in out_dir.glob("vf_maps_*.png"):
        legacy_png.unlink()

    df = pd.read_excel(xlsx_path)
    print(f"[Info] Loaded xlsx rows: {len(df)}")
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "split" not in df.columns:
        df["split"] = "all"
    df["split"] = df["split"].astype(str)

    patient_col = _detect_patient_col(df)
    if patient_col is None:
        print("[Warn] patient id column not found; fallback to per-row as patient unit.")
        df["__row_id__"] = np.arange(len(df))

    df["VF_true_arr"] = df["VF_true"].apply(_safe_parse_vector)
    df["VF_pred_arr"] = df["VF_pred"].apply(_safe_parse_vector)
    if "VF_recon" in df.columns:
        df["VF_recon_arr"] = df["VF_recon"].apply(_safe_parse_vector)
    else:
        df["VF_recon_arr"] = None

    df["md_true"] = pd.to_numeric(df["md_true"], errors="coerce")
    df["md_pred"] = pd.to_numeric(df["md_pred"], errors="coerce")

    valid = (
        df["VF_true_arr"].notna()
        & df["VF_pred_arr"].notna()
        & df["md_true"].notna()
        & df["md_pred"].notna()
    )
    work_df = df.loc[valid].copy()
    if len(work_df) == 0:
        raise RuntimeError("No valid rows after parsing VF/md fields.")
    print(f"[Info] Valid rows after parsing: {len(work_df)}")

    available_splits = work_df["split"].unique().tolist()
    selected_splits = _parse_splits_arg(args.splits, available_splits)
    work_df = work_df[work_df["split"].isin(selected_splits)].copy()
    print(f"[Info] Selected splits: {selected_splits}")

    # ========== patient-level metrics ==========
    patient_df = _patient_level_records(work_df, patient_col if patient_col in work_df.columns else "__row_id__")
    group_json = _build_patient_group_metrics_json(
        patient_df,
        work_df,
        patient_col if patient_col in work_df.columns else "__row_id__",
    )
    with (out_dir / "md_group_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(group_json, f, ensure_ascii=False, indent=2)

    _plot_md_scatter(patient_df, out_dir / "md_scatter_patient_level.png", dpi=args.dpi)
    print("[Info] Saved md scatter.")

    # ========== split-level plots ==========
    per_split_image_counts = {}
    split_use_tqdm = sys.stdout.isatty()
    split_pbar = tqdm(selected_splits, desc="Processing splits", unit="split", disable=not split_use_tqdm)
    for split_name in split_pbar:
        split_df = work_df[work_df["split"] == split_name].copy()
        if len(split_df) == 0:
            per_split_image_counts[split_name] = 0
            continue
        if not split_use_tqdm:
            print(f"[Info] Start split={split_name}, n_samples={len(split_df)}")
        images_dir = out_dir / f"{split_name}_images"
        for old_img in images_dir.glob("*.png"):
            old_img.unlink()
        n_saved = _plot_vf_maps_per_sample(
            split_df,
            split_name=split_name,
            images_dir=images_dir,
            dpi=args.dpi,
            patient_col=patient_col if patient_col in work_df.columns else "__row_id__",
        )
        per_split_image_counts[split_name] = int(n_saved)
        split_pbar.set_postfix(split=split_name, images=n_saved)
        _plot_pointwise_boxplot(
            split_df,
            split_name=split_name,
            save_path=out_dir / f"vf_boxplot_true_vs_pred_{split_name}.png",
            dpi=args.dpi,
        )
        if not split_use_tqdm:
            print(f"[Info] Finished split={split_name}")
    split_pbar.close()

    # ========== run summary ==========
    summary = {
        "input_xlsx": str(xlsx_path),
        "output_dir": str(out_dir),
        "selected_splits": selected_splits,
        "patient_id_column": patient_col if patient_col is not None else "__row_id__",
        "n_rows_total": int(len(df)),
        "n_rows_valid": int(len(valid[valid])),
        "n_rows_used": int(len(work_df)),
        "per_split_saved_images": per_split_image_counts,
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Input:  {xlsx_path}")
    print(f"Output: {out_dir}")
    print(f"Splits: {selected_splits}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
