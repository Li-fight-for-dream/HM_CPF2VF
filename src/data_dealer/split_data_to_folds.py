from src.data_dealer.dataset_for_fundus import COL_MD, COL_PID, COL_TEST_DATE
import pandas as pd
import numpy as np


def _md_to_severity(md_value: float) -> str:
    # 分级标准与 dataframe_dealer_for_fundus.py 保持一致
    if pd.isna(md_value):
        return "Unknown"
    if md_value > -6:
        return "Mild"
    if md_value >= -12:
        return "Moderate"
    return "Severe"


def make_kfold_splits(df, k_folds, seed=42):
    # 病人-日期级近似分层K折：
    # 1) 同一病人同一天(vf_id + vf_test_date)的样本必须在同一折；
    # 2) 先按(病人, 日期)聚合平均MD，再按严重程度(Mild/Moderate/Severe)分层；
    # 3) 每个严重程度层内分别做k折，再把同一fold编号的各层样本拼接成最终fold。
    if COL_PID not in df.columns:
        raise ValueError(f"缺少 '{COL_PID}' 列，无法按病人进行K折划分")
    if COL_TEST_DATE not in df.columns:
        raise ValueError(f"缺少 '{COL_TEST_DATE}' 列，无法按病人+日期进行K折划分")
    if COL_MD not in df.columns:
        raise ValueError(f"缺少 '{COL_MD}' 列，无法按MD严重程度分层")

    data = df.copy()

    # vf_id 为空时无法保证“同一病人同一天不被拆分”，因此直接报错
    if data[COL_PID].isna().any():
        raise ValueError(f"'{COL_PID}' 存在缺失值，无法保证同一病人同一天进入同一折")

    # 转换 test_date 到“天”粒度；同一天不同时间将视为同一visit
    data[COL_TEST_DATE] = pd.to_datetime(data[COL_TEST_DATE], errors="coerce")
    if data[COL_TEST_DATE].isna().any():
        raise ValueError(f"'{COL_TEST_DATE}' 存在缺失/非法日期，无法按病人+日期稳定分折")
    data["__visit_day"] = data[COL_TEST_DATE].dt.normalize()

    # 将MD转为数值；非法值会变成NaN，后续会被归到 Unknown 严重程度
    data[COL_MD] = pd.to_numeric(data[COL_MD], errors="coerce")

    # 以“病人+日期”为单位聚合平均MD（当前分层核心单位）
    visit_stats = (
        data.groupby([COL_PID, "__visit_day"], as_index=False)[COL_MD]
        .mean()
        .rename(columns={COL_MD: "visit_md"})
    )
    visit_stats["severity_label"] = visit_stats["visit_md"].apply(_md_to_severity)

    rng = np.random.default_rng(seed)
    fold_parts = [[] for _ in range(k_folds)]
    # 记录每个(病人, 日期)visit被分配到哪个fold，最后用于一致性校验
    visit_to_fold = {}
    all_visit_keys = pd.MultiIndex.from_frame(data[[COL_PID, "__visit_day"]])

    # 在每个严重程度层内独立做k折，保证各fold严重程度分布尽量接近
    for severity, severity_group in visit_stats.groupby("severity_label", sort=False):
        visit_row_indices = severity_group.index.to_numpy().copy()
        rng.shuffle(visit_row_indices)
        severity_folds = np.array_split(visit_row_indices, k_folds)

        for fold_idx, visit_rows_in_fold in enumerate(severity_folds):
            if len(visit_rows_in_fold) == 0:
                continue

            selected_visits = visit_stats.loc[visit_rows_in_fold, [COL_PID, "__visit_day"]]
            for visit in selected_visits.itertuples(index=False):
                visit_to_fold[(visit[0], visit[1])] = fold_idx

            # 把当前fold里这些(病人, 日期)对应的所有样本索引取出来
            selected_visit_keys = pd.MultiIndex.from_frame(selected_visits)
            mask = all_visit_keys.isin(selected_visit_keys)
            sample_idx = data.index[mask].to_numpy(dtype=np.int64)
            if sample_idx.size > 0:
                fold_parts[fold_idx].append(sample_idx)

        print(f"[KFold Split] severity={severity}, visits={len(visit_row_indices)}")

    folds = []
    for fold_idx in range(k_folds):
        if len(fold_parts[fold_idx]) == 0:
            fold_idx_array = np.array([], dtype=np.int64)
        else:
            fold_idx_array = np.concatenate(fold_parts[fold_idx])
            rng.shuffle(fold_idx_array)
        folds.append(fold_idx_array)
        print(f"[KFold Split] fold={fold_idx + 1}, samples={len(fold_idx_array)}")

    # 最终校验：每个(病人, 日期)visit只能映射到唯一fold
    for (pid, visit_day), _ in data.groupby([COL_PID, "__visit_day"]):
        assigned_fold = visit_to_fold.get((pid, visit_day), None)
        if assigned_fold is None:
            raise RuntimeError(f"visit ({pid}, {visit_day}) 未被分配到任何fold")

    return folds
