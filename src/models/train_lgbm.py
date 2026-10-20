"""LightGBM 需求预测基线（CPU / GPU）。

用法:
    python -m src.models.train_lgbm --config configs/default.yaml
    python -m src.models.train_lgbm --device gpu --n-estimators 3000

输出:
    data/processed/models/lgbm_<tag>.txt
    data/processed/models/lgbm_latest.txt
    data/processed/models/lgbm_<tag>.metrics.json
    data/processed/predictions/pred_test.parquet
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ensure_parent, load_config, make_logger, resolve
from src.models.common import (
    make_prediction_frame,
    regression_metrics,
    save_json,
    time_split_bounds,
)

log = make_logger("train_lgbm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM demand baseline.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--features", default=None)
    p.add_argument("--device", default=None, choices=["cpu", "gpu"])
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--num-leaves", type=int, default=None)
    p.add_argument("--tag", default=None, help="实验标签，默认时间戳")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    lcfg = dict(cfg.get("lgbm", {}))
    scfg = cfg.get("split", {})

    if args.device:
        lcfg["device"] = args.device
    if args.n_estimators:
        lcfg["n_estimators"] = args.n_estimators
    if args.learning_rate:
        lcfg["learning_rate"] = args.learning_rate
    if args.num_leaves:
        lcfg["num_leaves"] = args.num_leaves

    features_path = resolve(args.features or cfg["paths"]["features"])
    feature_list_path = resolve(cfg["paths"]["feature_list"])
    if not features_path.exists():
        raise FileNotFoundError(f"特征表不存在: {features_path}，请先运行 build_features.py")
    if not feature_list_path.exists():
        raise FileNotFoundError(f"特征清单不存在: {feature_list_path}")

    feature_meta = json.loads(feature_list_path.read_text(encoding="utf-8"))
    feature_cols = [c for c in feature_meta["features"] if c != "demand"]
    target = "demand"

    log.info("读取特征表 %s", features_path)
    cols = ["timestamp", "zone_id", target] + [c for c in feature_cols if c != "zone_id"]
    df = pd.read_parquet(features_path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["zone_id"] = df["zone_id"].astype("category")

    train_end, val_end = time_split_bounds(
        df["timestamp"],
        scfg.get("train", 0.70),
        scfg.get("val", 0.15),
    )
    train_mask = df["timestamp"] < train_end
    val_mask = (df["timestamp"] >= train_end) & (df["timestamp"] < val_end)
    test_mask = df["timestamp"] >= val_end
    log.info("时间切分: train < %s | val < %s | test >= %s", train_end, val_end, val_end)
    log.info("行数: train=%d, val=%d, test=%d", train_mask.sum(), val_mask.sum(), test_mask.sum())

    X = df[feature_cols]
    y = df[target].astype("float32")
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    params = {
        "objective": lcfg.get("objective", "regression_l1"),
        "metric": lcfg.get("metric", "mae"),
        "learning_rate": lcfg.get("learning_rate", 0.05),
        "num_leaves": lcfg.get("num_leaves", 63),
        "min_child_samples": lcfg.get("min_child_samples", 50),
        "subsample": lcfg.get("subsample", 0.8),
        "subsample_freq": lcfg.get("subsample_freq", 1),
        "colsample_bytree": lcfg.get("colsample_bytree", 0.8),
        "reg_alpha": lcfg.get("reg_alpha", 0.0),
        "reg_lambda": lcfg.get("reg_lambda", 1.0),
        "seed": cfg.get("project", {}).get("seed", 42),
        "n_jobs": lcfg.get("n_jobs", -1),
        "verbose": -1,
        "device_type": lcfg.get("device", "cpu"),
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_cols, free_raw_data=True)
    log.info("开始训练 LightGBM: %s, device=%s", params["objective"], params["device_type"])
    callbacks = [
        lgb.early_stopping(lcfg.get("early_stopping_rounds", 100), verbose=True),
        lgb.log_evaluation(period=100),
    ]
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=lcfg.get("n_estimators", 2000),
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )
    best_iter = int(booster.best_iteration or booster.current_iteration())
    log.info("训练完成，best_iteration=%d", best_iter)

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = resolve(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"lgbm_{tag}.txt"
    booster.save_model(str(model_path), num_iteration=best_iter)
    latest_path = model_dir / "lgbm_latest.txt"
    shutil.copy2(model_path, latest_path)
    log.info("模型 -> %s", model_path)

    metrics: dict[str, dict[str, float]] = {}
    pred_frames: dict[str, pd.DataFrame] = {}
    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    for name, (Xs, ys) in splits.items():
        pred = booster.predict(Xs, num_iteration=best_iter)
        metrics[name] = regression_metrics(ys.to_numpy(), pred)
        log.info("%s: MAE=%.4f RMSE=%.4f WAPE=%.2f%%",
                 name, metrics[name]["mae"], metrics[name]["rmse"], metrics[name]["wape"] * 100)
        if name == "test":
            ts = df.loc[test_mask, "timestamp"].reset_index(drop=True)
            zid = df.loc[test_mask, "zone_id"].astype("int16").reset_index(drop=True)
            pred_frames["test"] = make_prediction_frame(ts, zid, pred, "lgbm")

    meta = {
        "model": "lightgbm",
        "tag": tag,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_iteration": best_iter,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "params": params,
        "split": {"train_end": str(train_end), "val_end": str(val_end)},
        "metrics": metrics,
        "data": str(features_path),
    }
    save_json(meta, model_dir / f"lgbm_{tag}.metrics.json")
    save_json(meta, model_dir / "lgbm_latest.metrics.json")

    imp = pd.DataFrame(
        {"feature": feature_cols, "gain": booster.feature_importance(importance_type="gain", iteration=best_iter)}
    ).sort_values("gain", ascending=False)
    imp["gain"] = imp["gain"].astype("float64")
    imp.to_parquet(model_dir / "lgbm_feature_importance.parquet", index=False)

    pred_dir = resolve(cfg["paths"]["prediction_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_test = pred_frames["test"]
    pred_test.to_parquet(pred_dir / f"pred_test_lgbm_{tag}.parquet", compression="zstd", index=False)
    pred_test.to_parquet(pred_dir / "pred_test.parquet", compression="zstd", index=False)
    log.info("测试集预测 -> %s（%d 行）", pred_dir / "pred_test.parquet", len(pred_test))


if __name__ == "__main__":
    main()