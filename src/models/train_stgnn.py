"""轻量 STGCN 训练（成员 A：RTX 4060 8G）。

用法::
    python -m src.models.train_stgnn --config configs/default.yaml
    python -m src.models.train_stgnn --epochs 10 --batch-size 8 --plot
    python -m src.models.train_stgnn --tensorboard --plot
    # 快速冒烟：不覆盖正式 pred_test / latest 产物
    python -m src.models.train_stgnn --panel data/processed/sample_day.parquet \
        --epochs 2 --batch-size 8 --device cpu --no-features \
        --no-predict --no-canonical --tag smoke

输入选择:
    - 若 ``paths.feature_tensor`` 存在且与面板时间/区域对齐，默认使用 [T,N,F] 多变量输入；
    - ``--no-features`` 强制只用历史 demand，输入退化为 [T,N]（内部 [T,N,1]）。

输出::
    data/processed/models/stgnn_<tag>_best.pt
    data/processed/models/stgnn_<tag>.metrics.json
    data/processed/models/stgnn_<tag>_history.parquet / .csv
    data/processed/models/stgnn_<tag>_loss.png          # 加 --plot
    data/processed/logs/stgnn_<tag>/                    # 加 --tensorboard
    data/processed/predictions/pred_test_stgnn_<tag>.parquet
    data/processed/predictions/pred_test.parquet        # 非 --no-canonical
"""
from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import ensure_dir, load_config, make_logger, resolve
from src.models.common import (
    make_prediction_frame,
    regression_metrics,
    save_json,
    save_training_curve,
    time_split_bounds,
)
from src.models.stgnn import (
    STGCN,
    TensorWindowDataset,
    WindowDataset,
    apply_scaler,
    fit_scaler,
    inverse_scaler,
    load_feature_tensor,
    load_or_build_adjacency,
    load_zone_matrix,
    predict_matrix,
    split_indices,
    window_starts,
)

log = make_logger("train_stgnn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train lightweight STGCN.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--panel", default=None)
    p.add_argument("--features", default=None, help="特征张量 .npy；默认读取 paths.feature_tensor")
    p.add_argument("--no-features", action="store_true", help="只用 demand，不使用多变量张量")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--tensorboard", dest="tensorboard", action="store_true", default=None)
    p.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    p.add_argument("--log-dir", default=None)
    p.add_argument("--plot", dest="plot", action="store_true", default=None, help="保存 Loss/MAE 曲线 PNG")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.add_argument("--plot-out", default=None)
    p.add_argument("--no-predict", action="store_true", help="跳过测试集推理（冒烟/只验验证集时用）")
    p.add_argument("--no-canonical", action="store_true", help="不覆盖 pred_test.parquet / stgnn_latest.*")
    return p.parse_args()


def evaluate(model: STGCN, loader: DataLoader, device: torch.device, pred_len: int) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, model._adj)  # type: ignore[attr-defined]
            if pred_len == 1:
                pred = pred[:, :, 0]
                total += F.l1_loss(pred, y[:, 0], reduction="sum").item()
            else:
                total += F.l1_loss(pred, y.permute(0, 2, 1), reduction="sum").item()
            n += y.numel() if pred_len > 1 else y[:, 0].numel()
    return total / max(n, 1)


def _resolve_feature_input(
    args: argparse.Namespace,
    cfg: dict,
    zm,
    n_t: int,
):
    """返回 (feature_tensor, feature_meta, feature_path, use_features)。"""
    want = not args.no_features and (
        args.features is not None or bool(cfg.get("stgnn", {}).get("use_features", True))
    )
    if not want:
        if args.features:
            log.warning("--no-features 已指定，忽略 --features %s", args.features)
        return None, {}, None, False

    feature_arg = args.features or cfg.get("paths", {}).get("feature_tensor")
    if not feature_arg:
        log.warning("未配置 paths.feature_tensor，回退到仅 demand 输入")
        return None, {}, None, False

    feature_path = resolve(feature_arg)
    if not feature_path.exists():
        if args.features:
            raise FileNotFoundError(f"特征张量不存在: {feature_path}")
        log.warning("特征张量不存在 %s，回退到仅 demand 输入；可先运行 build_features.py", feature_path)
        return None, {}, None, False

    tensor, meta = load_feature_tensor(feature_path, zones=zm.zones, timestamps=zm.timestamps)
    if tensor.shape[0] != n_t:
        raise ValueError(
            f"特征张量时间片 {tensor.shape[0]} != 面板 {n_t}；"
            "请用同一面板重新运行 build_features.py，或加 --no-features"
        )
    return tensor, meta, feature_path, True


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    mcfg = dict(cfg.get("stgnn", {}))
    scfg = cfg.get("split", {})

    seq_len = int(args.seq_len or mcfg.get("seq_len", 12))
    pred_len = int(mcfg.get("pred_len", 1))
    batch_size = int(args.batch_size or mcfg.get("batch_size", 16))
    epochs = int(args.epochs or mcfg.get("epochs", 50))
    hidden = int(args.hidden or mcfg.get("hidden_channels", 32))
    lr = float(mcfg.get("lr", 1e-3))
    weight_decay = float(mcfg.get("weight_decay", 1e-4))
    num_workers = int(mcfg.get("num_workers", 0))
    grad_clip = float(mcfg.get("grad_clip", 5.0))
    if epochs < 1:
        raise ValueError("epochs 必须 >= 1")

    requested = args.device or mcfg.get("device", "cuda")
    device = torch.device(requested if (requested != "cuda" or torch.cuda.is_available()) else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA 不可用，回退到 CPU")

    panel_path = resolve(args.panel or cfg["paths"]["panel"])
    if not panel_path.exists():
        raise FileNotFoundError(f"需求面板不存在: {panel_path}")
    log.info("读取面板 %s", panel_path)
    zm = load_zone_matrix(panel_path, cfg["data"].get("zone_min", 1), cfg["data"].get("zone_max", 263))
    n_t = len(zm.timestamps)
    n_zones = len(zm.zones)
    log.info("矩阵: %d 时间片 × %d 区域", n_t, n_zones)

    n_train, n_val = split_indices(n_t, scfg.get("train", 0.70), scfg.get("val", 0.15))
    train_end, val_end = time_split_bounds(
        zm.timestamps, scfg.get("train", 0.70), scfg.get("val", 0.15)
    )
    log.info("时间切分: train < %s | val < %s | test >= %s", train_end, val_end, val_end)

    mean, std = fit_scaler(zm.values, n_train)
    normalized = apply_scaler(zm.values, mean, std)

    s_train = window_starts(n_t, seq_len, pred_len, 0, n_train)
    s_val = window_starts(n_t, seq_len, pred_len, n_train, n_val)
    s_test = window_starts(n_t, seq_len, pred_len, n_val, n_t)
    if min(len(s_train), len(s_val), len(s_test)) == 0:
        raise ValueError("窗口数量为 0，请检查 seq_len / 数据长度")
    log.info("窗口数: train=%d, val=%d, test=%d", len(s_train), len(s_val), len(s_test))

    feature_tensor, feature_meta, feature_path, use_features = _resolve_feature_input(
        args, cfg, zm, n_t
    )
    in_features = int(feature_tensor.shape[2]) if use_features else 1
    train_input = feature_tensor if use_features else normalized
    test_input = feature_tensor if use_features else normalized
    if use_features:
        log.info(
            "输入: feature tensor %s, shape=%s, F=%d, batch=[B,%d,%d,%d]",
            feature_path, feature_tensor.shape, in_features, seq_len, n_zones, in_features,
        )
    else:
        log.info("输入: 仅 demand, shape=%s, 内部广播为 [B,%d,%d,1]", zm.values.shape, seq_len, n_zones)

    adj, adj_src = load_or_build_adjacency(
        cfg["paths"]["adjacency"], cfg["paths"]["zone_centroids"], zm.zones
    )
    log.info("邻接矩阵: %s, shape=%s", adj_src, adj.shape)
    if adj_src == "identity_fallback":
        log.warning("未找到质心/邻接文件，使用单位矩阵；空间建模能力退化为逐区域模型")

    if use_features:
        train_ds = TensorWindowDataset(train_input, normalized, s_train, seq_len, pred_len)
        val_ds = TensorWindowDataset(train_input, normalized, s_val, seq_len, pred_len)
        test_ds = TensorWindowDataset(train_input, normalized, s_test, seq_len, pred_len)
    else:
        train_ds = WindowDataset(train_input, s_train, seq_len, pred_len)
        val_ds = WindowDataset(train_input, s_val, seq_len, pred_len)
        test_ds = WindowDataset(train_input, s_test, seq_len, pred_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = STGCN(
        n_zones=n_zones,
        hidden_channels=hidden,
        pred_len=pred_len,
        num_layers=mcfg.get("num_layers", 2),
        kernel_size=mcfg.get("kernel_size", 3),
        dropout=mcfg.get("dropout", 0.1),
        in_features=in_features,
    ).to(device)
    model._adj = torch.from_numpy(adj).to(device)  # type: ignore[attr-defined]

    n_params = sum(p.numel() for p in model.parameters())
    log.info("模型参数量: %.2f K, device=%s, in_features=%d", n_params / 1e3, device, in_features)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    model_dir = ensure_dir(cfg["paths"]["model_dir"])
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = model_dir / f"stgnn_{tag}_best.pt"
    best_val = float("inf")
    history: list[dict] = []

    # ---- 可选 TensorBoard ----
    tb_enabled = bool(mcfg.get("tensorboard", False)) if args.tensorboard is None else bool(args.tensorboard)
    writer = None
    if tb_enabled:
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = ensure_dir(args.log_dir or cfg["paths"].get("log_dir", "data/processed/logs"))
            writer = SummaryWriter(log_dir=str(Path(log_dir) / f"stgnn_{tag}"))
            log.info("TensorBoard -> %s", writer.log_dir)
        except Exception as exc:
            log.warning("TensorBoard 初始化失败（可 pip install tensorboard）: %s", exc)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, model._adj)  # type: ignore[attr-defined]
            if pred_len == 1:
                loss = F.smooth_l1_loss(pred[:, :, 0], y[:, 0], beta=0.5)
            else:
                loss = F.smooth_l1_loss(pred, y.permute(0, 2, 1), beta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total / max(n, 1)
        val_mae = evaluate(model, val_loader, device, pred_len)
        scheduler.step(val_mae)
        elapsed = time.time() - t0
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_mae_norm": float(val_mae),
                "lr": current_lr,
                "seconds": float(elapsed),
            }
        )
        log.info(
            "epoch %03d | train_loss %.4f | val_MAE(norm) %.4f | lr %.2e | %.1fs",
            epoch, train_loss, val_mae, current_lr, elapsed,
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("mae/val_norm", val_mae, epoch)
            writer.add_scalar("lr", current_lr, epoch)
            writer.add_scalar("time/epoch_sec", elapsed, epoch)

        if val_mae < best_val:
            best_val = val_mae
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "n_zones": n_zones,
                        "hidden_channels": hidden,
                        "pred_len": pred_len,
                        "num_layers": mcfg.get("num_layers", 2),
                        "kernel_size": mcfg.get("kernel_size", 3),
                        "dropout": mcfg.get("dropout", 0.1),
                        "seq_len": seq_len,
                        "in_features": in_features,
                        "feature_tensor": str(feature_path) if use_features else None,
                    },
                    "mean": mean,
                    "std": std,
                    "zones": zm.zones,
                    "adjacency": adj,
                    "split": {"train_end": str(train_end), "val_end": str(val_end)},
                },
                best_path,
            )
            log.info("  ↳ 保存最佳模型 %s", best_path)

    # ---- 测试集评估 + 标准预测输出 ----
    metrics: dict[str, float] = {}
    if not args.no_predict:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        preds = predict_matrix(model, test_input, s_test, seq_len, batch_size, device, pred_len)
        if pred_len != 1:
            raise ValueError("当前 predict 输出契约只支持 pred_len=1")
        pred_counts = inverse_scaler(preds[:, :, 0], mean, std)
        target_idx = s_test + seq_len
        y_true = zm.values[target_idx]

        ts = zm.timestamps[target_idx].repeat(n_zones)
        zid = np.tile(zm.zones, len(target_idx))
        pred_df = make_prediction_frame(ts, zid, pred_counts.reshape(-1), "stgnn")
        pred_df = pred_df.sort_values(["timestamp", "zone_id"], ignore_index=True)

        metrics = regression_metrics(y_true.reshape(-1), pred_counts.reshape(-1))
        log.info(
            "test: MAE=%.4f RMSE=%.4f WAPE=%.2f%%",
            metrics["mae"], metrics["rmse"], metrics["wape"] * 100,
        )

        pred_dir = ensure_dir(cfg["paths"]["prediction_dir"])
        pred_df.to_parquet(
            pred_dir / f"pred_test_stgnn_{tag}.parquet", compression="zstd", index=False
        )
        if not args.no_canonical:
            pred_df.to_parquet(pred_dir / "pred_test.parquet", compression="zstd", index=False)
            log.info("正式预测输出 -> %s", pred_dir / "pred_test.parquet")

    if not args.no_canonical:
        shutil.copy2(best_path, model_dir / "stgnn_latest.pt")

    meta = {
        "model": "stgnn",
        "tag": tag,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_val_mae_norm": float(best_val),
        "params": {
            "seq_len": seq_len,
            "pred_len": pred_len,
            "batch_size": batch_size,
            "hidden_channels": hidden,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "in_features": in_features,
        },
        "n_parameters": int(n_params),
        "input": "features" if use_features else "demand",
        "n_features": int(in_features),
        "feature_tensor": str(feature_path) if use_features else None,
        "adjacency": adj_src,
        "split": {"train_end": str(train_end), "val_end": str(val_end)},
        "metrics": metrics,
        "history": history,
    }
    save_json(meta, model_dir / f"stgnn_{tag}.metrics.json")
    if not args.no_canonical:
        save_json(meta, model_dir / "stgnn_latest.metrics.json")

    hist_df = pd.DataFrame(history)
    hist_df.to_parquet(model_dir / f"stgnn_{tag}_history.parquet", index=False)
    hist_df.to_csv(model_dir / f"stgnn_{tag}_history.csv", index=False, encoding="utf-8-sig")
    if not args.no_canonical:
        hist_df.to_parquet(model_dir / "stgnn_history.parquet", index=False)

    # ---- 可选 Loss 曲线 ----
    plot_enabled = bool(mcfg.get("plot", False)) if args.plot is None else bool(args.plot)
    if plot_enabled:
        plot_path = (
            resolve(args.plot_out)
            if args.plot_out
            else model_dir / f"stgnn_{tag}_loss.png"
        )
        saved = save_training_curve(history, plot_path, title=f"STGCN {tag}")
        if saved is not None:
            log.info("训练曲线 -> %s", saved)

    if writer is not None:
        writer.flush()
        writer.close()

    log.info("完成：%s", best_path)


if __name__ == "__main__":
    main()