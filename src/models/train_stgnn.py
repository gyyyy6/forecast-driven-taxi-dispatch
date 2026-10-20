"""轻量 STGCN 训练（成员 A：RTX 4060 8G）。

用法:
    python -m src.models.train_stgnn --config configs/default.yaml
    python -m src.models.train_stgnn --epochs 10 --batch-size 8   # 快速冒烟

输出:
    data/processed/models/stgnn_best.pt
    data/processed/models/stgnn_<tag>.metrics.json
    data/processed/predictions/pred_test.parquet
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import ensure_dir, load_config, make_logger, resolve
from src.models.common import make_prediction_frame, regression_metrics, save_json, time_split_bounds
from src.models.stgnn import (
    STGCN,
    WindowDataset,
    apply_scaler,
    fit_scaler,
    inverse_scaler,
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
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default=None)
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    mcfg = dict(cfg.get("stgnn", {}))
    scfg = cfg.get("split", {})

    seq_len = args.seq_len or mcfg.get("seq_len", 12)
    pred_len = mcfg.get("pred_len", 1)
    batch_size = args.batch_size or mcfg.get("batch_size", 16)
    epochs = args.epochs or mcfg.get("epochs", 50)
    hidden = args.hidden or mcfg.get("hidden_channels", 32)
    lr = mcfg.get("lr", 1e-3)
    weight_decay = mcfg.get("weight_decay", 1e-4)
    num_workers = mcfg.get("num_workers", 0)
    grad_clip = mcfg.get("grad_clip", 5.0)

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

    adj, adj_src = load_or_build_adjacency(
        cfg["paths"]["adjacency"], cfg["paths"]["zone_centroids"], zm.zones
    )
    log.info("邻接矩阵: %s, shape=%s", adj_src, adj.shape)
    if adj_src == "identity_fallback":
        log.warning("未找到质心/邻接文件，使用单位矩阵；空间建模能力退化为逐区域模型")

    train_ds = WindowDataset(normalized, s_train, seq_len, pred_len)
    val_ds = WindowDataset(normalized, s_val, seq_len, pred_len)
    test_ds = WindowDataset(normalized, s_test, seq_len, pred_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = STGCN(
        n_zones=n_zones,
        hidden_channels=hidden,
        pred_len=pred_len,
        num_layers=mcfg.get("num_layers", 2),
        kernel_size=mcfg.get("kernel_size", 3),
        dropout=mcfg.get("dropout", 0.1),
    ).to(device)
    model._adj = torch.from_numpy(adj).to(device)  # type: ignore[attr-defined]

    n_params = sum(p.numel() for p in model.parameters())
    log.info("模型参数量: %.2f K, device=%s", n_params / 1e3, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    model_dir = ensure_dir(cfg["paths"]["model_dir"])
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = model_dir / f"stgnn_{tag}_best.pt"
    best_val = float("inf")
    history: list[dict] = []

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
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mae_norm": val_mae, "seconds": elapsed})
        log.info("epoch %03d | train_loss %.4f | val_MAE(norm) %.4f | %.1fs", epoch, train_loss, val_mae, elapsed)

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

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # ---- 测试集评估 + 标准预测输出 ----
    preds = predict_matrix(model, normalized, s_test, seq_len, batch_size, device, pred_len)
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
    log.info("test: MAE=%.4f RMSE=%.4f WAPE=%.2f%%", metrics["mae"], metrics["rmse"], metrics["wape"] * 100)

    pred_dir = ensure_dir(cfg["paths"]["prediction_dir"])
    pred_df.to_parquet(pred_dir / f"pred_test_stgnn_{tag}.parquet", compression="zstd", index=False)
    pred_df.to_parquet(pred_dir / "pred_test.parquet", compression="zstd", index=False)

    latest = model_dir / "stgnn_latest.pt"
    shutil.copy2(best_path, latest)

    meta = {
        "model": "stgnn",
        "tag": tag,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_val_mae_norm": best_val,
        "params": {
            "seq_len": seq_len,
            "pred_len": pred_len,
            "batch_size": batch_size,
            "hidden_channels": hidden,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        "n_parameters": int(n_params),
        "adjacency": adj_src,
        "split": {"train_end": str(train_end), "val_end": str(val_end)},
        "metrics": metrics,
        "history": history,
    }
    save_json(meta, model_dir / f"stgnn_{tag}.metrics.json")
    save_json(meta, model_dir / "stgnn_latest.metrics.json")
    pd.DataFrame(history).to_parquet(model_dir / "stgnn_history.parquet", index=False)
    log.info("完成：%s", best_path)


if __name__ == "__main__":
    main()