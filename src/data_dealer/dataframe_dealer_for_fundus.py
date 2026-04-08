data_path = "data_table_with_fundus.xlsx"
dataset_ratio = [0.8, 0.1, 0.1]  # 训练集、验证集、测试集比例

import pandas as pd


def get_df_length(df):
    """
    获取 DataFrame 的行数。
    """
    return len(df)


def analyze_all_md_ratios(df):
    """
    分析 Excel 文件中所有样本的 vf_md 属性，统计轻度、中度、重度损伤的比例。
    
    分级标准:
    - 轻度 (Mild): MD > -6
    - 中度 (Moderate): -12 <= MD <= -6
    - 重度 (Severe): MD < -12
    """
    try:
        if "vf_md" not in df.columns:
            return {"error": "文件中缺少 'vf_md' 列"}

        df["vf_md"] = pd.to_numeric(df["vf_md"], errors="coerce")
        valid_df = df.dropna(subset=["vf_md"])
        total_count = len(valid_df)
        if total_count == 0:
            return {"info": "文件中没有有效的 vf_md 数值"}

        mild_count = len(valid_df[valid_df["vf_md"] > -6])
        severe_count = len(valid_df[valid_df["vf_md"] < -12])
        moderate_count = total_count - mild_count - severe_count

        stats = {
            "total_valid_samples": total_count,
            "distribution": {
                "mild": {
                    "count": mild_count,
                    "ratio": round(mild_count / total_count, 4),
                    "label": "轻度 (> -6)"
                },
                "moderate": {
                    "count": moderate_count,
                    "ratio": round(moderate_count / total_count, 4),
                    "label": "中度 (-12 ~ -6)"
                },
                "severe": {
                    "count": severe_count,
                    "ratio": round(severe_count / total_count, 4),
                    "label": "重度 (< -12)"
                }
            }
        }
        return stats
    except FileNotFoundError:
        return {"error": "文件未找到，请检查路径"}
    except Exception as e:
        return {"error": f"分析过程中发生错误: {str(e)}"}


def stratified_split_by_md(df, ratios=[0.8, 0.1, 0.1], seed=42):
    """
    将 DataFrame 按 vf_md 严重程度分层，并按比例划分为训练集、验证集、测试集。
    原则: 同一病人 (vf_id) 的所有样本必须进入同一集合。
    """
    data = df.copy()
    if "vf_id" not in data.columns:
        raise ValueError("缺少 'vf_id' 列，无法按病人分组划分数据集")

    data["vf_md"] = pd.to_numeric(data["vf_md"], errors="coerce")
    data = data.dropna(subset=["vf_md", "vf_id"])

    def get_severity(md):
        if md > -6:
            return "Mild"
        elif md >= -12:
            return "Moderate"
        else:
            return "Severe"

    # 以病人为单位做近似分层：先聚合出病人的平均 MD，再按严重程度划分
    patient_stats = (
        data.groupby("vf_id", as_index=False)["vf_md"]
        .mean()
        .rename(columns={"vf_md": "patient_md"})
    )
    patient_stats["Severity_Label"] = patient_stats["patient_md"].apply(get_severity)

    train_patient_ids = []
    val_patient_ids = []
    test_patient_ids = []

    for _, group in patient_stats.groupby("Severity_Label"):
        shuffled_group = group.sample(frac=1, random_state=seed)
        n_total = len(shuffled_group)
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])

        train_ids = shuffled_group.iloc[:n_train]["vf_id"].tolist()
        val_ids = shuffled_group.iloc[n_train : n_train + n_val]["vf_id"].tolist()
        test_ids = shuffled_group.iloc[n_train + n_val :]["vf_id"].tolist()

        train_patient_ids.extend(train_ids)
        val_patient_ids.extend(val_ids)
        test_patient_ids.extend(test_ids)

    train_df = data[data["vf_id"].isin(train_patient_ids)].sample(frac=1, random_state=seed)
    val_df = data[data["vf_id"].isin(val_patient_ids)].sample(frac=1, random_state=seed)
    test_df = data[data["vf_id"].isin(test_patient_ids)].sample(frac=1, random_state=seed)
    return train_df, val_df, test_df


if __name__ == "__main__":
    df = pd.read_excel(data_path)
    all_length = get_df_length(df)
    train_len, val_len, test_len = [int(r * all_length) for r in dataset_ratio]

    print("开始分析...")
    all_data_result = analyze_all_md_ratios(df)

    if "error" in all_data_result:
        print(f"❌ 错误: {all_data_result['error']}")
    elif "info" in all_data_result:
        print(f"⚠️ 提示: {all_data_result['info']}")
    else:
        print(f"✅ 对总体数据分析完成 (总有效样本: {all_data_result['total_valid_samples']})")
        dist = all_data_result["distribution"]
        print(f"   - 轻度: {dist['mild']['ratio']:.2%} ({dist['mild']['count']}只眼睛)")
        print(f"   - 中度: {dist['moderate']['ratio']:.2%} ({dist['moderate']['count']}只眼睛)")
        print(f"   - 重度: {dist['severe']['ratio']:.2%} ({dist['severe']['count']}只眼睛)")

    train_df, val_df, test_df = stratified_split_by_md(df, ratios=dataset_ratio, seed=42)

    print("✅ 数据集划分结果：")
    print(f"   [训练集 Train] 数量: {len(train_df)} (实际占比: {dataset_ratio[0]:.2%})")
    print(f"   [验证集 Val  ] 数量: {len(val_df)} (实际占比: {dataset_ratio[1]:.2%})")
    print(f"   [测试集 Test ] 数量: {len(test_df)} (实际占比: {dataset_ratio[2]:.2%})")
