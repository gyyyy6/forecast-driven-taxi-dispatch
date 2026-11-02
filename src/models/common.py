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


def save_training_curve(
    history: list[dict],
    path: str | Path,
    title: str = "STGCN training",
) -> Path | None:
    """把训练历史画成 train loss / val MAE 双轴曲线（无显示器环境也可用）。"""
    if not history:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib 未安装时不影响训练
        print(f"[warn] 无法绘制训练曲线: {exc}")
        return None

    epochs = [int(h["epoch"]) for h in history]
    train_loss = [float(h.get("train_loss", np.nan)) for h in history]
    val_mae = [float(h.get("val_mae_norm", np.nan)) for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax1.plot(epochs, train_loss, color="#2563eb", linewidth=1.8, label="train loss")
    ax1.set_xlabel("epoch")
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylabel("train loss (smooth L1, normalized)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_mae, color="#ea580c", linewidth=1.8, label="val MAE")
    ax2.set_ylabel("val MAE (normalized)", color="#ea580c")
    ax2.tick_params(axis="y", labelcolor="#ea580c")

    best_idx = int(np.nanargmin(val_mae)) if np.isfinite(val_mae).any() else -1
    if best_idx >= 0:
        best_epoch = epochs[best_idx]
        best_val = val_mae[best_idx]
        ax2.axvline(best_epoch, color="#9ca3af", linestyle="--", linewidth=1.0)
        ax2.scatter([best_epoch], [best_val], color="#ea580c", s=28, zorder=5)
        span = max(epochs[-1] - epochs[0], 1)
        near_right = best_epoch >= epochs[-1] - 0.15 * span
        ax2.annotate(
            f"best {best_val:.4f}",
            (best_epoch, best_val),
            textcoords="offset points",
            xytext=(-58, 10) if near_right else (6, 8),
            ha="right" if near_right else "left",
            fontsize=8,
            color="#9a3412",
        )

    ax1.set_title(title)
    fig.tight_layout()

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


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