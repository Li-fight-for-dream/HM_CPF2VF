import numpy as np
import matplotlib.pyplot as plt
import os

SEGMENTS_24_2 = [4, 6, 8, 9, 9, 8, 6, 4]
RIGHT_EYE_ROW_START_COLS = [4, 3, 2, 1, 1, 2, 3, 4]  # 每行起始列（1-based）
BLINDSPOT_IDXS_54 = (25, 34)  # 第26、35个值（1-based）对应的0-based索引


def blindspot_mask_grid(n_cols: int = 9) -> np.ndarray:
    """返回右眼8x9网格上的盲点mask（盲点位置为True）。"""
    mask = np.zeros((len(SEGMENTS_24_2), n_cols), dtype=bool)
    blindspot_set = set(BLINDSPOT_IDXS_54)

    start = 0
    for r, seg_len in enumerate(SEGMENTS_24_2):
        left_pad = RIGHT_EYE_ROW_START_COLS[r] - 1
        for c in range(seg_len):
            if (start + c) in blindspot_set:
                mask[r, left_pad + c] = True
        start += seg_len

    return mask

def prepare_display_grid(vf54: np.ndarray, n_cols: int = 9) -> np.ndarray:
    """用于显示的网格：将盲点位置设为NaN。"""
    grid = vf54_to_grid(vf54, n_cols=n_cols).copy()
    grid[blindspot_mask_grid(n_cols=n_cols)] = np.nan
    return grid

def vf52_to_vf54(vf52: np.ndarray) -> np.ndarray:
    """将52维VF恢复为54维（插入盲点位置的0）。"""
    if vf52.shape[0] != 52:
        raise ValueError(f"vf52 must have length 52, got {vf52.shape[0]}")
    vf54 = np.zeros(54, dtype=np.float32)
    mask = np.ones(54, dtype=bool)
    mask[list(BLINDSPOT_IDXS_54)] = False
    vf54[mask] = vf52.astype(np.float32)
    vf54[list(BLINDSPOT_IDXS_54)] = 0.0
    return vf54


def vf54_to_grid(vf54: np.ndarray, n_cols: int = 9) -> np.ndarray:
    """将54维VF映射到右眼8x9网格，按固定行起始列放置。"""
    if vf54.shape[0] != 54:
        raise ValueError(f"vf54 must have length 54, got {vf54.shape[0]}")
    if len(RIGHT_EYE_ROW_START_COLS) != len(SEGMENTS_24_2):
        raise ValueError("RIGHT_EYE_ROW_START_COLS长度必须与SEGMENTS_24_2一致。")

    grid = np.full((len(SEGMENTS_24_2), n_cols), np.nan, dtype=np.float32)
    start = 0
    for r, seg_len in enumerate(SEGMENTS_24_2):
        row_vals = vf54[start:start + seg_len]
        start += seg_len

        left_pad = RIGHT_EYE_ROW_START_COLS[r] - 1
        if left_pad < 0 or left_pad + seg_len > n_cols:
            raise ValueError("行放置超出网格边界。")
        grid[r, left_pad:left_pad + seg_len] = row_vals

    return grid

def get_display_cmap():
    """显示风格：viridis，NaN（盲点）显示为白色。"""
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    return cmap

def plot_vf_compare_2x3(
    vf52_gt: np.ndarray,
    vf52_dl: np.ndarray,
    vf52_proto: np.ndarray,
    save_path: str,
    title: str = "",
):
    """
    保存一张 2x3 图：
      第一行：GT | DL Pred | Diff (GT - DL)
      第二行：GT | Proto Recon | Diff (GT - Proto)
    """
    # ---- to vf54 ----
    vf54_gt = vf52_to_vf54(vf52_gt)
    vf54_dl = vf52_to_vf54(vf52_dl)
    vf54_pr = vf52_to_vf54(vf52_proto)

    # ---- diffs (保持你原函数的习惯：Diff = GT - Recon) ----
    vf54_df_dl = vf54_gt - vf54_dl
    vf54_df_pr = vf54_gt - vf54_pr

    # ---- to grid ----
    g_gt = prepare_display_grid(vf54_gt)
    g_dl = prepare_display_grid(vf54_dl)
    g_pr = prepare_display_grid(vf54_pr)
    g_df_dl = prepare_display_grid(vf54_df_dl)
    g_df_pr = prepare_display_grid(vf54_df_pr)
    cmap = get_display_cmap()

    fig = plt.figure(figsize=(12, 7))
    if title:
        fig.suptitle(title)

    # ========== Row 1 ==========
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(g_gt, interpolation="nearest", cmap=cmap, vmin=0, vmax=40)
    ax1.set_title("GT")
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(g_dl, interpolation="nearest", cmap=cmap, vmin=0, vmax=40)
    ax2.set_title("DL Pred")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(g_df_dl, interpolation="nearest", cmap=cmap, vmin=0, vmax=40)
    ax3.set_title("Diff (GT - DL)")
    ax3.axis("off")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # ========== Row 2 ==========
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(g_gt, interpolation="nearest", cmap=cmap, vmin=0, vmax=40)
    ax4.set_title("GT")
    ax4.axis("off")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(g_pr, interpolation="nearest", cmap=cmap, vmin=0, vmax=40)
    ax5.set_title("Proto Recon")
    ax5.axis("off")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.imshow(g_df_pr, interpolation="nearest", cmap=cmap, vmin=0, vmax=40)
    ax6.set_title("Diff (GT - Proto)")
    ax6.axis("off")
    fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)