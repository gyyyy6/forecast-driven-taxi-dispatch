"""统一推理入口：LightGBM / STGCN → 标准 prediction parquet。

用法::
    python -m src.models.predict --config configs/default.yaml --model lgbm --split test
    python -m src.models.predict --config configs/default.yaml --model stgnn --split test
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src.config import ensure_dir, load_config, make_logger, resolve
from src.models.common import make_prediction_frame, regression_metrics, save_json, time_split_bounds
from src.models.stgnn import (
    STGCN,
    apply_scaler,
    inverse_scaler,
    load_feature_tensor,
    load_zone_matrix,
    predict_matrix,
    split_indices,
    window_starts,
)

log = make_logger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified demand inference.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="lgbm", choices=["lgbm", "stgnn"])
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out", default=None)
    return p.parse_args()


def split_mask(ts: pd.Series, split: str, cfg: dict) -> pd.Series:
    scfg = cfg.get("split", {})
    train_end, val_end = time_split_bounds(ts, scfg.get("train", 0.70), scfg.get("val", 0.15))
    if split == "val":
        return (ts >= train_end) & (ts < val_end)
    return ts >= val_end


def predict_lgbm(cfg: dict, split: str) -> pd.DataFrame:
    feature_list_path = resolve(cfg["paths"]["feature_list"])
    meta = json.loads(feature_list_path.read_text(encoding="utf-8"))
    features = meta["features"]
    model_path = resolve(cfg["paths"]["model_dir"]) / "lgbm_latest.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    df = pd.read_parquet(resolve(cfg["paths"]["features"]))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    mask = split_mask(df["timestamp"], split, cfg)
    sub = df.loc[mask].copy()
    sub["zone_id"] = sub["zone_id"].astype("category")

    booster = lgb.Booster(model_file=str(model_path))
    pred = booster.predict(sub[features])
    out = make_prediction_frame(sub["timestamp"].to_numpy(), sub["zone_id"].astype("int16").to_numpy(), pred, "lgbm")
    if "demand" in sub.columns:
        metrics = regression_metrics(sub["demand"].to_numpy(), pred)
        log.info("lgbm %s: MAE=%.4f WAPE=%.2f%%", split, metrics["mae"], metrics["wape"] * 100)
    return out


def predict_stgnn(cfg: dict, split: str) -> pd.DataFrame:
    ckpt_path = resolve(cfg["paths"]["model_dir"]) / "stgnn_latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型不存在: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ckpt["config"]
    in_features = int(mcfg.get("in_features", 1))

    model = STGCN(
        n_zones=mcfg["n_zones"],
        hidden_channels=mcfg["hidden_channels"],
        pred_len=mcfg["pred_len"],
        num_layers=mcfg["num_layers"],
        kernel_size=mcfg["kernel_size"],
        dropout=mcfg.get("dropout", 0.1),
        in_features=in_features,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model._adj = torch.from_numpy(ckpt["adjacency"].astype(np.float32)).to(device)  # type: ignore[attr-defined]
    model.eval()

    zm = load_zone_matrix(resolve(cfg["paths"]["panel"]), cfg["data"].get("zone_min", 1), cfg["data"].get("zone_max", 263))
    n_t = len(zm.timestamps)
    n_train, n_val = split_indices(n_t, cfg["split"].get("train", 0.70), cfg["split"].get("val", 0.15))
    seq_len, pred_len = mcfg["seq_len"], mcfg["pred_len"]
    target_start, target_end = (n_train, n_val) if split == "val" else (n_val, n_t)
    starts = window_starts(n_t, seq_len, pred_len, target_start, target_end)
    if len(starts) == 0:
        raise ValueError("窗口数量为 0，请检查 seq_len 与数据长度")

    if in_features > 1:
        candidates = [mcfg.get("feature_tensor"), cfg["paths"].get("feature_tensor")]
        feature_path = next(
            (candidate for candidate in candidates if candidate and resolve(candidate).exists()),
            None,
        )
        if not feature_path:
            raise ValueError(
                "checkpoint 需要多变量输入，但找不到 feature_tensor；"
                "请确认 paths.feature_tensor 指向有效文件"
            )
        input_matrix, _ = load_feature_tensor(
            resolve(feature_path), zones=zm.zones, timestamps=zm.timestamps
        )
        if input_matrix.shape[2] != in_features:
            raise ValueError(
                f"特征张量 F={input_matrix.shape[2]} 与 checkpoint in_features={in_features} 不一致"
            )
        log.info("stgnn 输入: feature tensor %s, shape=%s", feature_path, input_matrix.shape)
    else:
        input_matrix = apply_scaler(zm.values, ckpt["mean"], ckpt["std"])
        log.info("stgnn 输入: demand, shape=%s", zm.values.shape)

    preds = predict_matrix(model, input_matrix, starts, seq_len, batch_size=32, device=device, pred_len=pred_len)
    if pred_len != 1:
        raise ValueError("预测输出契约当前只支持 pred_len=1")
    pred_counts = inverse_scaler(preds[:, :, 0], ckpt["mean"], ckpt["std"])
    target_idx = starts + seq_len
    y_true = zm.values[target_idx].reshape(-1)

    out = make_prediction_frame(
        zm.timestamps[target_idx].repeat(len(zm.zones)),
        np.tile(zm.zones, len(target_idx)),
        pred_counts.reshape(-1),
        "stgnn",
    )
    metrics = regression_metrics(y_true, pred_counts.reshape(-1))
    log.info("stgnn %s: MAE=%.4f WAPE=%.2f%%", split, metrics["mae"], metrics["wape"] * 100)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = predict_lgbm(cfg, args.split) if args.model == "lgbm" else predict_stgnn(cfg, args.split)

    pred_dir = ensure_dir(cfg["paths"]["prediction_dir"])
    path = resolve(args.out) if args.out else pred_dir / f"pred_{args.split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, compression="zstd", index=False)
    log.info("预测输出 -> %s（%d 行）", path, len(out))

    meta = {
        "model": args.model,
        "split": args.split,
        "rows": int(len(out)),
        "zones": int(out["zone_id"].nunique()),
        "timestamps": int(out["timestamp"].nunique()),
        "predicted_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(meta, path.with_suffix(".meta.json"))


if __name__ == "__main__":
    main()