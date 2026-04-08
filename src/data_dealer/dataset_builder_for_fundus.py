from src.data_dealer.dataframe_dealer_for_fundus import stratified_split_by_md, get_df_length
from torch.utils.data import DataLoader
from src.data_dealer.dataset_for_fundus import vf_with_fundus_dataset
import pandas as pd


def devide_dataset(args):
    df = pd.read_excel(args.data_root)
    all_length = get_df_length(df)
    train_len, val_len, test_len = [int(r * all_length) for r in args.dataset_ratio]

    print("开始分析...")
    train_df, val_df, test_df = stratified_split_by_md(df, ratios=args.dataset_ratio, seed=42)

    print("✅ 数据集划分结果：")
    print(f"   [训练集 Train] 数量: {len(train_df)} (实际占比: {args.dataset_ratio[0]:.2%})")
    print(f"   [验证集 Val  ] 数量: {len(val_df)} (实际占比: {args.dataset_ratio[1]:.2%})")
    print(f"   [测试集 Test ] 数量: {len(test_df)} (实际占比: {args.dataset_ratio[2]:.2%})")

    return train_df, val_df, test_df


def make_dataloaders(args):
    train_df, val_df, test_df = devide_dataset(args)
    train_dataset = vf_with_fundus_dataset(train_df, image_root=getattr(args, "image_root", None))
    val_dataset = vf_with_fundus_dataset(val_df, image_root=getattr(args, "image_root", None))
    test_dataset = vf_with_fundus_dataset(test_df, image_root=getattr(args, "image_root", None))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
