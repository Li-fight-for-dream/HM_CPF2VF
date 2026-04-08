from pathlib import Path

import torch
from tqdm import tqdm

from src.utils.logger import append_epoch_record, append_best_record
from src.build_losses import compute_stage_loss


def _format_lr(optimizer):
    lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
    return ",".join([f"{v:.6g}" for v in lrs if v is not None])


def _get_md_true(batch, device, alpha_md: float, md_missing_warned: bool):
    md_true = None
    if "md_tensor" in batch:
        md_true = batch["md_tensor"].to(device).view(-1, 1)
    else:
        if (not md_missing_warned) and alpha_md != 0:
            print("[Warn] batch does not contain 'md_tensor'. md loss will be 0.0.")
            md_missing_warned = True
    return md_true, md_missing_warned


def _get_a_true_from_vf(model, vf_true):
    core_model = model.module if hasattr(model, "module") else model
    if not hasattr(core_model, "aa_transform"):
        raise AttributeError("AANet model must implement `aa_transform` for prototype-weight supervision.")

    with torch.no_grad():
        a_true = core_model.aa_transform(vf_true, normalize_output=True)
    return a_true

def _tb_log_stage(writer, split, stage, epoch, m):
    # 按阶段来记录和画图
    writer.add_scalar("Stage", stage, epoch)

    if stage == 1:
        writer.add_scalar("stage1/Loss_Total", m["loss"], epoch)
        writer.add_scalar("stage1/Loss_VF1", m["loss_parts"]["L_vf1"], epoch)
        writer.add_scalar("stage1/Loss_A", m["loss_parts"]["L_a"], epoch)
        writer.add_scalars(
            "stage1/VF_Metrics",
            {
                "RMSE_VFpred": m["vf_rmse"],
                "MAE_VFpred": m["vf_mae"],
            },
            epoch,
        )
        return

    if stage == 2:
        writer.add_scalar("stage2/Loss_MD", m["loss"], epoch)
        if m["md_rmse"] is not None:
            writer.add_scalars("stage2/MD_Metrics", {"RMSE": m["md_rmse"], "MAE": m["md_mae"]}, epoch)
        return

    writer.add_scalar("stage3/Loss_Total", m["loss"], epoch)
    for k, v in m["loss_parts"].items():
        writer.add_scalar(f"stage3/LossParts_{k}", v, epoch)

    writer.add_scalars(
        "stage3/VF_Metrics",
        {
            "RMSE_VF1": m["vf1_rmse"],
            "MAE_VF1": m["vf1_mae"],
            "RMSE_VFpred": m["vf_rmse"],
            "MAE_VFpred": m["vf_mae"],
        },
        epoch,
    )
    writer.add_scalar("stage3/r_norm_mean", m["r_norm"], epoch)
    if m["md_rmse"] is not None:
        writer.add_scalars("stage3/MD_Metrics", {"RMSE": m["md_rmse"], "MAE": m["md_mae"]}, epoch)


def _save_model(model, save_path: str, name: str):
    path = Path(save_path) / name
    torch.save(model.state_dict(), path)

# ==========================================  主函数  =================================================

def run_one_epoch(model, loader, device, stage, cfg, losses, metrics, optimizer=None):
    """
    一个epoch所经历的大部分事情:
    -> 数据经过权重，计算loss，反向传播计算梯度
    -> 更新
    -> 计算metrics[iter累计，epoch完成再计算]
    [注意另外两个步骤：④schedule/阶段切换/早停的判断、更新 ⑤结果保存放在了函数外面]

    函数返回：该epoch的loss、子loss、VF或MD指标
    """
    do_backward = optimizer is not None
    model.train() if do_backward else model.eval()

    total_loss = 0.0
    all_vf1, all_vf_pred, all_vf_true = [], [], []
    all_r_norm = []
    all_md_pred, all_md_true = [], []
    md_missing_warned = False

    stage1_sums = {"L_total": 0.0, "L_vf1": 0.0, "L_a": 0.0}
    loss_sums = {"L_total": 0.0, "L_vf": 0.0, "L_vf1": 0.0, "L_r": 0.0, "L_md": 0.0}

    with (torch.enable_grad() if do_backward else torch.no_grad()):
        for batch in tqdm(loader, desc="iter", leave=False):
            inputs = batch["fundus_img"].to(device)
            vf_true = batch["VF_tensor"].to(device)
            md_true, md_missing_warned = _get_md_true(
                batch,
                device,
                cfg.training.loss_weights.alpha_md,
                md_missing_warned,
            )

            a_true = None
            if stage == 1:
                a_true = _get_a_true_from_vf(model, vf_true)

            outputs = model(inputs)
            loss, parts = compute_stage_loss(
                stage,
                outputs,
                vf_true,
                md_true,
                losses,
                cfg.training.loss_weights,
                a_true=a_true,
            )

            if do_backward:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            # 为metrics收集好所有样本信息
            if stage == 1:
                stage1_sums["L_total"] += float(parts["L_total"])
                stage1_sums["L_vf1"] += float(parts["L_vf1"])
                stage1_sums["L_a"] += float(parts["L_a"])

                all_vf1.append(outputs["VF1"].detach())
                all_vf_pred.append(outputs["VF_pred"].detach())
                all_vf_true.append(vf_true.detach())
            elif stage == 2:
                all_md_pred.append(outputs["md_pred"].detach())
                if md_true is not None:
                    all_md_true.append(md_true.detach())
            else:
                for k in loss_sums:
                    loss_sums[k] += float(parts[k])
                all_vf1.append(outputs["VF1"].detach())
                all_vf_pred.append(outputs["VF_pred"].detach())
                all_vf_true.append(vf_true.detach())
                all_r_norm.append(outputs["r_norm"].detach())
                all_md_pred.append(outputs["md_pred"].detach())
                if md_true is not None:
                    all_md_true.append(md_true.detach())

    n = max(1, len(loader))
    total_loss /= n

    # 正式评估metrics
    if stage == 1:
        for k in stage1_sums:
            stage1_sums[k] /= n

        vf1 = torch.cat(all_vf1, dim=0)
        vf_pred = torch.cat(all_vf_pred, dim=0)
        vf_true = torch.cat(all_vf_true, dim=0)
        rmse_vf1, mae_vf1 = metrics["vf"](vf1, vf_true)
        rmse_vf, mae_vf = metrics["vf"](vf_pred, vf_true)
        return {
            "loss": float(stage1_sums["L_total"]),
            "vf1_rmse": rmse_vf1,
            "vf1_mae": mae_vf1,
            "vf_rmse": rmse_vf,
            "vf_mae": mae_vf,
            "loss_parts": {
                "L_total": float(stage1_sums["L_total"]),
                "L_vf1": float(stage1_sums["L_vf1"]),
                "L_a": float(stage1_sums["L_a"]),
            },
        }

    if stage == 2:
        md_rmse = md_mae = None
        if all_md_true:
            md_pred = torch.cat(all_md_pred, dim=0)
            md_true = torch.cat(all_md_true, dim=0)
            md_rmse, md_mae = metrics["md"](md_pred, md_true)
        return {
            "loss": float(total_loss),
            "md_rmse": md_rmse,
            "md_mae": md_mae,
            "loss_parts": {"L_md": float(total_loss)},
        }

    # stage=3的情况评估metrics
    for k in loss_sums:
        loss_sums[k] /= n

    vf1 = torch.cat(all_vf1, dim=0)
    vf_pred = torch.cat(all_vf_pred, dim=0)
    vf_true = torch.cat(all_vf_true, dim=0)
    r_norm = torch.cat(all_r_norm, dim=0).mean().item()

    rmse_vf1, mae_vf1 = metrics["vf"](vf1, vf_true)
    rmse_vf, mae_vf = metrics["vf"](vf_pred, vf_true)

    md_rmse = md_mae = None
    if all_md_true:
        md_pred = torch.cat(all_md_pred, dim=0)
        md_true = torch.cat(all_md_true, dim=0)
        md_rmse, md_mae = metrics["md"](md_pred, md_true)

    return {
        "loss": float(loss_sums["L_total"]),
        "vf1_rmse": rmse_vf1,
        "vf1_mae": mae_vf1,
        "vf_rmse": rmse_vf,
        "vf_mae": mae_vf,
        "r_norm": float(r_norm),
        "md_rmse": md_rmse,
        "md_mae": md_mae,
        "loss_parts": loss_sums,
    }


def fit(
    model,
    loaders,
    optimizer,
    scheduler,
    callbacks,
    losses,
    metrics,
    cfg,
    logger,
):
    train_loader, val_loader, test_loader = loaders
    writers = logger["writers"]
    save_path = logger["save_path"]

    best_stage1_vf = float("inf")
    best_stage2_md = float("inf")
    best_stage3_vf = float("inf")
    best_stage3_md = float("inf")

    stage = 1

    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs} | stage={stage} | lr={_format_lr(optimizer)}")

        train_m = run_one_epoch(model, train_loader, cfg.device, stage, cfg, losses, metrics, optimizer=optimizer)
        val_m = run_one_epoch(model, val_loader, cfg.device, stage, cfg, losses, metrics, optimizer=None)
        scheduler.step(val_m["loss"])

        _tb_log_stage(writers["train"], "train", stage, epoch, train_m)
        _tb_log_stage(writers["val"], "val", stage, epoch, val_m)

        if stage == 1:
            print(
                f"Train: Total {train_m['loss']:.6f} | VF1_loss {train_m['loss_parts']['L_vf1']:.6f} | "
                f"A_loss {train_m['loss_parts']['L_a']:.6f} | VF_RMSE {train_m['vf_rmse']:.4f} MAE {train_m['vf_mae']:.4f}"
            )
            print(
                f"Val:   Total {val_m['loss']:.6f} | VF1_loss {val_m['loss_parts']['L_vf1']:.6f} | "
                f"A_loss {val_m['loss_parts']['L_a']:.6f} | VF_RMSE {val_m['vf_rmse']:.4f} MAE {val_m['vf_mae']:.4f}"
            )

            if val_m["vf_rmse"] < best_stage1_vf:
                best_stage1_vf = val_m["vf_rmse"]
                best_model_name = "best_vf_on_stage1_model.pth"
                _save_model(model, save_path, best_model_name)
                append_best_record(
                    save_path=save_path,
                    model_name=best_model_name,
                    stage=1,
                    epoch=epoch + 1,
                    monitor="val_vf_rmse",
                    monitor_value=best_stage1_vf,
                    metrics=val_m,
                )
                print(f"[Best] {best_model_name} | stage=1 | val_vf_rmse={best_stage1_vf:.4f}")

            append_epoch_record(
                save_path,
                [
                    f"Epoch {epoch+1}/{cfg.training.epochs} stage=1",
                    (
                        f"Train: total {train_m['loss']:.6f}, vf1_loss {train_m['loss_parts']['L_vf1']:.6f}, "
                        f"a_loss {train_m['loss_parts']['L_a']:.6f}, VF_RMSE {train_m['vf_rmse']:.4f}, VF_MAE {train_m['vf_mae']:.4f}"
                    ),
                    (
                        f"Val:   total {val_m['loss']:.6f}, vf1_loss {val_m['loss_parts']['L_vf1']:.6f}, "
                        f"a_loss {val_m['loss_parts']['L_a']:.6f}, VF_RMSE {val_m['vf_rmse']:.4f}, VF_MAE {val_m['vf_mae']:.4f}"
                    ),
                ],
            )

            if callbacks["switch_s1"].step(val_m["loss"], epoch):
                print(f"Switch to Stage2 at epoch {epoch+1}.")
                stage = 2
                optimizer = callbacks["build_optimizer_for_stage"](model, cfg, stage)
                scheduler = callbacks["build_scheduler"](optimizer, cfg)

        elif stage == 2:
            print(f"Train: MD_loss {train_m['loss']:.6f} | MD_RMSE {train_m['md_rmse']} MAE {train_m['md_mae']}")
            print(f"Val:   MD_loss {val_m['loss']:.6f} | MD_RMSE {val_m['md_rmse']} MAE {val_m['md_mae']}")

            score = val_m["md_rmse"] if val_m["md_rmse"] is not None else val_m["loss"]
            if score < best_stage2_md:
                best_stage2_md = score
                best_model_name = "best_stage2_md_model.pth"
                _save_model(model, save_path, best_model_name)
                monitor_name = "val_md_rmse" if val_m["md_rmse"] is not None else "val_loss"
                append_best_record(
                    save_path=save_path,
                    model_name=best_model_name,
                    stage=2,
                    epoch=epoch + 1,
                    monitor=monitor_name,
                    monitor_value=best_stage2_md,
                    metrics=val_m,
                )
                print(f"[Best] {best_model_name} | stage=2 | {monitor_name}={best_stage2_md}")

            append_epoch_record(
                save_path,
                [
                    f"Epoch {epoch+1}/{cfg.training.epochs} stage=2",
                    f"Train: MD_loss {train_m['loss']:.6f}, MD_RMSE {train_m['md_rmse']}, MD_MAE {train_m['md_mae']}",
                    f"Val:   MD_loss {val_m['loss']:.6f}, MD_RMSE {val_m['md_rmse']}, MD_MAE {val_m['md_mae']}",
                ],
            )

            if callbacks["switch_s2"].step(val_m["loss"], epoch):
                print(f"Switch to Stage3 at epoch {epoch+1}.")
                stage = 3
                optimizer = callbacks["build_optimizer_for_stage"](model, cfg, stage)
                scheduler = callbacks["build_scheduler"](optimizer, cfg)

        else:
            lp_t = train_m["loss_parts"]
            lp_v = val_m["loss_parts"]
            print(
                "Train:",
                f"Total {train_m['loss']:.6f} | VF {train_m['vf_rmse']:.4f}/{train_m['vf_mae']:.4f} | "
                f"r_norm {train_m['r_norm']:.4f} | "
                f"parts: vf={lp_t['L_vf']:.6f}, vf1={lp_t['L_vf1']:.6f}, r={lp_t['L_r']:.6f}, md={lp_t['L_md']:.6f}",
            )
            print(
                "Val:  ",
                f"Total {val_m['loss']:.6f} | VF {val_m['vf_rmse']:.4f}/{val_m['vf_mae']:.4f} | "
                f"r_norm {val_m['r_norm']:.4f} | "
                f"parts: vf={lp_v['L_vf']:.6f}, vf1={lp_v['L_vf1']:.6f}, r={lp_v['L_r']:.6f}, md={lp_v['L_md']:.6f}",
                f"| MD_RMSE {val_m['md_rmse']}",
            )

            if val_m["vf_rmse"] < best_stage3_vf:
                best_stage3_vf = val_m["vf_rmse"]
                best_model_name = "best_stage3_by_vf.pth"
                _save_model(model, save_path, best_model_name)
                append_best_record(
                    save_path=save_path,
                    model_name=best_model_name,
                    stage=3,
                    epoch=epoch + 1,
                    monitor="val_vf_rmse",
                    monitor_value=best_stage3_vf,
                    metrics=val_m,
                )
                print(f"[Best] {best_model_name} | stage=3 | val_vf_rmse={best_stage3_vf:.4f}")

            if val_m["md_rmse"] is not None and val_m["md_rmse"] < best_stage3_md:
                best_stage3_md = val_m["md_rmse"]
                best_model_name = "best_stage3_by_md.pth"
                _save_model(model, save_path, best_model_name)
                append_best_record(
                    save_path=save_path,
                    model_name=best_model_name,
                    stage=3,
                    epoch=epoch + 1,
                    monitor="val_md_rmse",
                    monitor_value=best_stage3_md,
                    metrics=val_m,
                )
                print(f"[Best] {best_model_name} | stage=3 | val_md_rmse={best_stage3_md}")

            append_epoch_record(
                save_path,
                [
                    f"Epoch {epoch+1}/{cfg.training.epochs} stage=3",
                    f"Train: total {train_m['loss']:.6f}, VF_RMSE {train_m['vf_rmse']:.4f}, VF_MAE {train_m['vf_mae']:.4f}, "
                    f"r_norm {train_m['r_norm']:.4f}, parts {train_m['loss_parts']}, MD_RMSE {train_m['md_rmse']}",
                    f"Val:   total {val_m['loss']:.6f}, VF_RMSE {val_m['vf_rmse']:.4f}, VF_MAE {val_m['vf_mae']:.4f}, "
                    f"r_norm {val_m['r_norm']:.4f}, parts {val_m['loss_parts']}, MD_RMSE {val_m['md_rmse']}",
                ],
            )

            # if epoch % cfg.training.checkpoint_interval == 0:
            #     _save_model(model, save_path, "checkpoint_model.pth")

            if cfg.training.early_stop.enabled:
                if callbacks["early_s3"].step(epoch_idx=epoch, val_vf_rmse=val_m["vf_rmse"], val_md_rmse=val_m["md_rmse"]):
                    print(f"Early stop at epoch {epoch+1}.")
                    break

        if cfg.training.use_test:
            test_m = run_one_epoch(model, test_loader, cfg.device, stage, cfg, losses, metrics, optimizer=None)
            _tb_log_stage(writers["test"], "test", stage, epoch, test_m)

            if stage == 1:
                print(
                    f"Test:  Total {test_m['loss']:.6f} | VF1_loss {test_m['loss_parts']['L_vf1']:.6f} | "
                    f"A_loss {test_m['loss_parts']['L_a']:.6f} | VF_RMSE {test_m['vf_rmse']:.4f} MAE {test_m['vf_mae']:.4f}"
                )
                append_epoch_record(
                    save_path,
                    [
                        f"Epoch {epoch+1}/{cfg.training.epochs} stage=1",
                        (
                            f"Test:  total {test_m['loss']:.6f}, vf1_loss {test_m['loss_parts']['L_vf1']:.6f}, "
                            f"a_loss {test_m['loss_parts']['L_a']:.6f}, VF_RMSE {test_m['vf_rmse']:.4f}, VF_MAE {test_m['vf_mae']:.4f}"
                        ),
                    ],
                )
            elif stage == 2:
                print(f"Test:  MD_loss {test_m['loss']:.6f} | MD_RMSE {test_m['md_rmse']} MAE {test_m['md_mae']}")
                append_epoch_record(
                    save_path,
                    [
                        f"Epoch {epoch+1}/{cfg.training.epochs} stage=2",
                        f"Test:  MD_loss {test_m['loss']:.6f}, MD_RMSE {test_m['md_rmse']}, MD_MAE {test_m['md_mae']}",
                    ],
                )
            else:
                print(f"Test:  total {test_m['loss']:.6f} | VF {test_m['vf_rmse']:.4f}/{test_m['vf_mae']:.4f} | r_norm {test_m['r_norm']:.4f} | MD_RMSE {test_m['md_rmse']}")
                append_epoch_record(
                    save_path,
                    [
                        f"Epoch {epoch+1}/{cfg.training.epochs} stage=3",
                        f"Test:  total {test_m['loss']:.6f}, VF_RMSE {test_m['vf_rmse']:.4f}, VF_MAE {test_m['vf_mae']:.4f}, "
                        f"r_norm {test_m['r_norm']:.4f}, parts {test_m['loss_parts']}, MD_RMSE {test_m['md_rmse']}",
                    ],
                )
