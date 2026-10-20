"""时间切分、评估指标与通用 IO。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def time_split_bounds(
    timestamps: pd.Series | np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """按唯一时间戳排序后返回 (train_end, val_end)。

    训练: ts < train_end
    验证: train_end <= ts < val_end
    测试: ts >= val_end
    """
    ts = pd.Index(pd.to_datetime(pd.Series(timestamps).unique())).sort_values()
    n = len(ts)
    if n < 10:
        raise ValueError(f"时间片过少（{n}），无法切分")
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n_train <= 0 or n_val <= 0 or n_train + n_val >= n:
        raise ValueError("切分比例导致空集合，请检查 split 配置")
    return pd.Timestamp(ts[n_train]), pd.Timestamp(ts[n_train + n_val])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    abs_err = np.abs(err)
    denom = np.abs(y_true).sum()
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(abs_err.sum() / denom) if denom > 0 else float("nan"),
        "smape": float(
            np.mean(2 * abs_err / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        ),
        "n": int(len(y_true)),
    }


def save_json(payload: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def make_prediction_frame(
    timestamp: pd.Series | np.ndarray,
    zone_id: pd.Series | np.ndarray,
    predicted_demand: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """按 docs/api_contract.md 生成标准预测表。"""
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamp),
            "zone_id": pd.array(zone_id, dtype="int16"),
            "predicted_demand": np.clip(
                np.asarray(predicted_demand, dtype=np.float32), 0, None
            ).astype("float32"),
            "model_name": model_name,
        }
    )
    return out.sort_values(["timestamp", "zone_id"], ignore_index=True)