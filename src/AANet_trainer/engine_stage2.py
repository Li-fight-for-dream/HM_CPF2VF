from pathlib import Path

import torch
from tqdm import tqdm

from src.utils.logger import append_best_record, append_epoch_record
from src.build_losses import compute_stage_loss


def _format_lr(optimizer):
    lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
    return ",".join([f"{v:.6g}" for v in lrs if v is not None])


def _save_model(model, save_path: str, name: str):
    path = Path(save_path) / name
    torch.save(model.state_dict(), path)


def _get_md_true(batch, device, md_missing_warned: bool):
    md_true = None
    if "md_tensor" in batch:
        md_true = batch["md_tensor"].to(device).view(-1, 1)
    else:
        if not md_missing_warned:
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


def _tb_log_stage(writer, stage, epoch, m):
    writer.add_scalar("Stage", stage, epoch)

    if stage == 1:
        writer.add_scalar("stage1/Loss_Total", m["loss"], epoch)
        writer.add_scalar("stage1/Loss_VF1", m["loss_parts"]["L_vf1"], epoch)
        writer.add_scalar("stage1/Loss_A", m["loss_parts"]["L_a"], epoch)
        writer.add_scalars(
            "stage1/VF_Metrics",
            {
                "RMSE_VF1": m["vf1_rmse"],
                "MAE_VF1": m["vf1_mae"],
                "RMSE_VFpred": m["vf_rmse"],
                "MAE_VFpred": m["vf_mae"],
            },
            epoch,
        )
    else:
        writer.add_scalar("stage2/Loss_MD", m["loss"], epoch)
        if m["md_rmse"] is not None:
            writer.add_scalars(
                "stage2/MD_Metrics",
                {
                    "RMSE": m["md_rmse"],
                    "MAE": m["md_mae"],
                },
                epoch,
            )


def run_one_epoch(model, loader, device, stage, cfg, losses, metrics, optimizer=None):
    do_backward = optimizer is not None
    model.train() if do_backward else model.eval()

    total_loss = 0.0
    md_missing_warned = False

    stage1_sums = {"L_total": 0.0, "L_vf1": 0.0, "L_a": 0.0}

    all_vf1, all_vf_pred, all_vf_true = [], [], []
    all_md_pred, all_md_true = [], []

    with (torch.enable_grad() if do_backward else torch.no_grad()):
        for batch in tqdm(loader, desc="iter", leave=False):
            inputs = batch["fundus_img"].to(device)
            outputs = model(inputs)

            if stage == 1:
                vf_true = batch["VF_tensor"].to(device)
                a_true = _get_a_true_from_vf(model, vf_true)

                loss, parts = compute_stage_loss(
                    stage=1,
                    outputs=outputs,
                    vf_true=vf_true,
                    md_true=None,
                    losses=losses,
                    loss_weights=cfg.training.loss_weights,
                    a_true=a_true,
                )

                stage1_sums["L_total"] += float(parts["L_total"])
                stage1_sums["L_vf1"] += float(parts["L_vf1"])
                stage1_sums["L_a"] += float(parts["L_a"])

                all_vf1.append(outputs["VF1"].detach())
                all_vf_pred.append(outputs["VF_pred"].detach())
                all_vf_true.append(vf_true.detach())
            else:
                md_true, md_missing_warned = _get_md_true(batch, device, md_missing_warned)
                loss, _ = compute_stage_loss(
                    stage=2,
                    outputs=outputs,
                    vf_true=batch["VF_tensor"].to(device),
                    md_true=md_true,
                    losses=losses,
                    loss_weights=cfg.training.loss_weights,
                )
                all_md_pred.append(outputs["md_pred"].detach())
                if md_true is not None:
                    all_md_true.append(md_true.detach())

            if do_backward:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())

    n = max(1, len(loader))

    if stage == 1:
        for k in stage1_sums:
            stage1_sums[k] /= n

        vf1 = torch.cat(all_vf1, dim=0)
        vf_pred = torch.cat(all_vf_pred, dim=0)
        vf_true = torch.cat(all_vf_true, dim=0)

        vf1_rmse, vf1_mae = metrics["vf"](vf1, vf_true)
        vf_rmse, vf_mae = metrics["vf"](vf_pred, vf_true)

        return {
            "loss": float(stage1_sums["L_total"]),
            "vf1_rmse": vf1_rmse,
            "vf1_mae": vf1_mae,
            "vf_rmse": vf_rmse,
            "vf_mae": vf_mae,
            "loss_parts": {
                "L_total": float(stage1_sums["L_total"]),
                "L_vf1": float(stage1_sums["L_vf1"]),
                "L_a": float(stage1_sums["L_a"]),
            },
        }

    total_loss /= n
    md_rmse = md_mae = None
    if len(all_md_true) > 0:
        md_pred = torch.cat(all_md_pred, dim=0)
        md_true = torch.cat(all_md_true, dim=0)
        md_rmse, md_mae = metrics["md"](md_pred, md_true)

    return {
        "loss": float(total_loss),
        "md_rmse": md_rmse,
        "md_mae": md_mae,
        "loss_parts": {"L_md": float(total_loss)},
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

    stage = 1
    stage2_bad_epochs = 0

    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs} | stage={stage} | lr={_format_lr(optimizer)}")

        train_m = run_one_epoch(model, train_loader, cfg.device, stage, cfg, losses, metrics, optimizer=optimizer)
        val_m = run_one_epoch(model, val_loader, cfg.device, stage, cfg, losses, metrics, optimizer=None)
        scheduler.step(val_m["loss"])

        _tb_log_stage(writers["train"], stage, epoch, train_m)
        _tb_log_stage(writers["val"], stage, epoch, val_m)

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
                best_model_name = "best_stage1_vf_model.pth"
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

        else:
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
                stage2_bad_epochs = 0
            else:
                min_delta = float(getattr(cfg.training.early_stop, "min_delta_md", 0.0))
                if best_stage2_md - score <= min_delta:
                    stage2_bad_epochs += 1
                else:
                    stage2_bad_epochs = 0

            append_epoch_record(
                save_path,
                [
                    f"Epoch {epoch+1}/{cfg.training.epochs} stage=2",
                    f"Train: MD_loss {train_m['loss']:.6f}, MD_RMSE {train_m['md_rmse']}, MD_MAE {train_m['md_mae']}",
                    f"Val:   MD_loss {val_m['loss']:.6f}, MD_RMSE {val_m['md_rmse']}, MD_MAE {val_m['md_mae']}",
                ],
            )

            if cfg.training.early_stop.enabled and stage2_bad_epochs >= int(cfg.training.early_stop.patience):
                print(f"Early stop at epoch {epoch+1} (stage2 patience={cfg.training.early_stop.patience}).")
                break

        if cfg.training.use_test:
            test_m = run_one_epoch(model, test_loader, cfg.device, stage, cfg, losses, metrics, optimizer=None)
            _tb_log_stage(writers["test"], stage, epoch, test_m)

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
            else:
                print(f"Test:  MD_loss {test_m['loss']:.6f} | MD_RMSE {test_m['md_rmse']} MAE {test_m['md_mae']}")
                append_epoch_record(
                    save_path,
                    [
                        f"Epoch {epoch+1}/{cfg.training.epochs} stage=2",
                        f"Test:  MD_loss {test_m['loss']:.6f}, MD_RMSE {test_m['md_rmse']}, MD_MAE {test_m['md_mae']}",
                    ],
                )
