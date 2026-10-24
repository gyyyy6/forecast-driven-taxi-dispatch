"""成员 C / C3：LightGBM 特征组与损失函数消融。

8 组实验共用同一切分、同一 seed、同一超参数，唯一变量是“特征列集合 / 损失函数”：

    A0  zone + calendar                         （仅区域 + 日历）
    A1  A0 + lag_1/2/4/8                        （加短期滞后）
    A2  A1 + lag_96                             （加日周期滞后）
    A3  A2 + lag_672                            （加周周期滞后）
    A4  A3 + roll_mean_4/96                     （完整 18 特征，等价于 C2 基线）
    A5  A4 同特征，objective=regression_l2      （损失对比）
    A6  A4 去掉 zone_id                         （任务书可选组）
    A7  A4 去掉日历                             （补充：日历特征贡献）

用法:
    python -m src.models.run_ablation --config configs/experiments/c_ablation.yaml --dry-run
    python -m src.models.run_ablation --config configs/experiments/c_ablation.yaml
    python -m src.models.run_ablation --config configs/experiments/c_ablation.yaml --only A4,A5

产出:
    data/processed/experiments/lgbm_ablation.parquet        长表（一行一组）
    data/processed/experiments/lgbm_ablation.manifest.json  运行元信息 + 推荐组
    data/processed/experiments/predictions/pred_test_lgbm_<id>.parquet

指标口径:
    长表中的 mae / rmse / wape 为**验证集**指标（任务书 C3“记录验证集指标”的语义，
    也是推荐组的唯一选择依据）；test_* 为留出评估，train_* 仅用于看拟合程度。

注意:
    - 列顺序统一按 feature_list.json 的规范顺序，保证 A4 与 c_baseline 逐位可比；
    - 特征只加载一次，逐组按列切分，各组特征取值完全相同；
    - 不改 build_features.py，也不覆盖 B 消费的 data/processed/predictions/pred_test.parquet。
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ensure_dir, ensure_parent, load_config, make_logger, resolve
from src.models.common import make_prediction_frame, regression_metrics, save_json, time_split_bounds

log = make_logger("run_ablation")

FORBIDDEN = {"demand", "timestamp", "time_idx"}
RESULT_COLUMNS = [
    "experiment_id",
    "feature_set",
    "objective",
    "n_features",
    "mae",
    "rmse",
    "wape",
    "test_mae",
    "test_rmse",
    "test_wape",
    "train_mae",
    "train_rmse",
    "train_wape",
    "best_iteration",
    "train_seconds",
    "n_train",
    "n_val",
    "n_test",
    "seed",
]


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    groups: tuple[str, ...]
    objective: str
    note: str
    columns: tuple[str, ...]

    @property
    def feature_set(self) -> str:
        return "+".join(self.groups)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LightGBM feature-group / objective ablation.")
    p.add_argument("--config", default="configs/experiments/c_ablation.yaml")
    p.add_argument("--only", default=None, help="只跑指定实验，逗号分隔，如 A4,A5")
    p.add_argument("--out", default=None, help="长表输出路径；默认 out_dir/lgbm_ablation.parquet")
    p.add_argument("--n-estimators", type=int, default=None, help="覆盖轮数（仅用于快速验证，正式结果不要用）")
    p.add_argument("--save-models", action="store_true", help="额外保存每组 booster .txt（约 60 MB）")
    p.add_argument("--no-save-predictions", action="store_true", help="不写各组测试集预测")
    p.add_argument("--dry-run", action="store_true", help="只打印实验矩阵与特征列，不训练")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 实验矩阵
# --------------------------------------------------------------------------- #
def build_matrix(cfg: dict, feature_meta: dict) -> list[Experiment]:
    groups: dict[str, list[str]] = cfg.get("feature_groups") or {}
    if not groups:
        raise SystemExit("配置缺少 feature_groups")
    canonical: list[str] = list(feature_meta.get("features") or [])
    if not canonical:
        raise SystemExit("feature_list.json 里没有 features")
    canonical_set = set(canonical)
    order = {c: i for i, c in enumerate(canonical)}

    for name, cols in groups.items():
        if not cols:
            raise SystemExit(f"特征组 {name} 为空")
        missing = [c for c in cols if c not in canonical_set]
        if missing:
            raise SystemExit(f"特征组 {name} 引用了不存在的列 {missing}；可用列: {canonical}")
        forbidden = [c for c in cols if c in FORBIDDEN]
        if forbidden:
            raise SystemExit(f"特征组 {name} 含标签/键/索引列 {forbidden}")

    specs = cfg.get("experiments") or []
    if not specs:
        raise SystemExit("配置缺少 experiments")

    default_objective = (cfg.get("lgbm") or {}).get("objective", "regression_l1")
    matrix: list[Experiment] = []
    for spec in specs:
        exp_id = str(spec.get("id") or "").strip()
        if not exp_id:
            raise SystemExit(f"实验缺少 id: {spec}")
        names = list(spec.get("groups") or [])
        unknown = [g for g in names if g not in groups]
        if unknown:
            raise SystemExit(f"实验 {exp_id} 引用了未知特征组 {unknown}；可用组: {sorted(groups)}")
        if not names:
            raise SystemExit(f"实验 {exp_id} 没有特征组")
        seen: set[str] = set()
        cols: list[str] = []
        for g in names:
            for c in groups[g]:
                if c in seen:
                    log.warning("实验 %s 的特征组 %s 有重复列 %s，已去重", exp_id, g, c)
                    continue
                seen.add(c)
                cols.append(c)
        with_objective = spec.get("objective") or default_objective
        matrix.append(
            Experiment(
                experiment_id=exp_id,
                groups=tuple(names),
                objective=str(with_objective),
                note=str(spec.get("note") or ""),
                columns=tuple(sorted(cols, key=order.__getitem__)),
            )
        )

    ids = [e.experiment_id for e in matrix]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"实验 id 重复: {ids}")

    _validate_matrix(matrix, canonical_set, groups)
    return matrix


def _validate_matrix(matrix: list[Experiment], canonical_set: set[str], groups: dict) -> None:
    """守住“累积消融 + 单项剥离”的结构，防止配置漂移。"""
    by_id = {e.experiment_id: e for e in matrix}

    def cols(exp_id: str) -> set[str]:
        return set(by_id[exp_id].columns)

    if "A0" in by_id:
        want = set(groups.get("zone", [])) | set(groups.get("calendar", []))
        if cols("A0") != want:
            raise SystemExit(f"A0 应等于 zone + calendar，实际 {sorted(cols('A0'))}")
    chain = [f"A{i}" for i in range(5)]
    if all(i in by_id for i in chain):
        for i in range(4):
            prev, cur = f"A{i}", f"A{i + 1}"
            if not cols(prev) < cols(cur):
                raise SystemExit(f"{prev} 必须是 {cur} 的真子集（累积消融），实际 {prev}={len(cols(prev))}, {cur}={len(cols(cur))}")
    if "A4" in by_id and cols("A4") != canonical_set:
        raise SystemExit(f"A4 必须等于 feature_list.json 的全部特征（{len(canonical_set)} 个），实际 {len(cols('A4'))} 个")
    if "A5" in by_id and "A4" in by_id and by_id["A5"].columns != by_id["A4"].columns:
        raise SystemExit("A5 必须与 A4 同特征（只改 objective）")
    if "A6" in by_id and "A4" in by_id and cols("A6") != cols("A4") - set(groups.get("zone", [])):
        raise SystemExit("A6 必须等于 A4 去掉 zone_id")
    if "A7" in by_id and "A4" in by_id and cols("A7") != cols("A4") - set(groups.get("calendar", [])):
        raise SystemExit("A7 必须等于 A4 去掉日历")


# --------------------------------------------------------------------------- #
# 数据与训练
# --------------------------------------------------------------------------- #
def build_params(lcfg: dict, seed: int, objective: str) -> dict:
    """与 src/models/train_lgbm.py 完全一致的参数映射，保证 A4 可复现基线。"""
    return {
        "objective": objective,
        "metric": lcfg.get("metric", "mae"),
        "learning_rate": lcfg.get("learning_rate", 0.05),
        "num_leaves": lcfg.get("num_leaves", 63),
        "min_child_samples": lcfg.get("min_child_samples", 50),
        "subsample": lcfg.get("subsample", 0.8),
        "subsample_freq": lcfg.get("subsample_freq", 1),
        "colsample_bytree": lcfg.get("colsample_bytree", 0.8),
        "reg_alpha": lcfg.get("reg_alpha", 0.0),
        "reg_lambda": lcfg.get("reg_lambda", 1.0),
        "seed": seed,
        "n_jobs": lcfg.get("n_jobs", -1),
        "verbose": -1,
        "device_type": lcfg.get("device", "cpu"),
    }


def train_one(
    exp: Experiment,
    df: pd.DataFrame,
    masks: dict[str, pd.Series],
    lcfg: dict,
    seed: int,
    n_estimators: int,
    pred_dir: Path | None,
    model_dir: Path | None,
) -> dict:
    y = df["demand"].astype("float32")
    X = df[list(exp.columns)]

    train_mask, val_mask, test_mask = masks["train"], masks["val"], masks["test"]
    params = build_params(lcfg, seed, exp.objective)
    dtrain = lgb.Dataset(
        X[train_mask], label=y[train_mask], feature_name=list(exp.columns), free_raw_data=True
    )
    dval = lgb.Dataset(
        X[val_mask], label=y[val_mask], reference=dtrain, feature_name=list(exp.columns), free_raw_data=True
    )

    log.info("[%s] 开始训练：%s，%d 个特征，objective=%s", exp.experiment_id, exp.feature_set, len(exp.columns), exp.objective)
    t0 = time.perf_counter()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(int(lcfg.get("early_stopping_rounds", 100)), verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    train_seconds = time.perf_counter() - t0
    best_iter = int(booster.best_iteration or booster.current_iteration())
    if best_iter >= n_estimators:
        log.warning("[%s] 早停未触发，best_iteration 顶到 n_estimators=%d", exp.experiment_id, n_estimators)
    log.info("[%s] 训练完成：best_iteration=%d，耗时 %.1f 秒", exp.experiment_id, best_iter, train_seconds)

    row: dict = {
        "experiment_id": exp.experiment_id,
        "feature_set": exp.feature_set,
        "objective": exp.objective,
        "n_features": len(exp.columns),
        "best_iteration": best_iter,
        "train_seconds": round(float(train_seconds), 3),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "seed": seed,
    }
    preds: dict[str, np.ndarray] = {}
    for split, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        pred = booster.predict(X[mask], num_iteration=best_iter)
        preds[split] = pred
        m = regression_metrics(y[mask].to_numpy(), pred)
        prefix = "" if split == "val" else f"{split}_"
        row[f"{prefix}mae"] = m["mae"]
        row[f"{prefix}rmse"] = m["rmse"]
        row[f"{prefix}wape"] = m["wape"]
        log.info(
            "[%s] %-5s MAE=%.4f RMSE=%.4f WAPE=%.2f%%",
            exp.experiment_id, split, m["mae"], m["rmse"], m["wape"] * 100,
        )

    if model_dir is not None:
        booster.save_model(str(model_dir / f"lgbm_{exp.experiment_id}.txt"), num_iteration=best_iter)

    if pred_dir is not None:
        ts = df.loc[test_mask, "timestamp"].reset_index(drop=True)
        zid = df.loc[test_mask, "zone_id"].astype("int16").reset_index(drop=True)
        frame = make_prediction_frame(ts, zid, preds["test"], f"lgbm_{exp.experiment_id}")
        frame.to_parquet(pred_dir / f"pred_test_lgbm_{exp.experiment_id}.parquet", compression="zstd", index=False)

    return row


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def markdown_table(table: pd.DataFrame) -> str:
    cols = ["experiment_id", "n_features", "objective", "mae", "wape", "test_mae", "test_wape", "best_iteration", "train_seconds"]
    head = "| 组 | 特征数 | objective | val MAE | val WAPE | test MAE | test WAPE | best_iter | 训练秒 |"
    sep = "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [head, sep]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['experiment_id']} | {int(r['n_features'])} | {r['objective']} | "
            f"{r['mae']:.4f} | {r['wape'] * 100:.2f}% | {r['test_mae']:.4f} | "
            f"{r['test_wape'] * 100:.2f}% | {int(r['best_iteration'])} | {r['train_seconds']:.1f} |"
        )
    return "\n".join(lines)


def pick_recommendation(table: pd.DataFrame) -> dict:
    """预登记规则：验证集 WAPE 最低；并列取特征更少者；再并列按 id 顺序。"""
    ordered = table.sort_values(["wape", "n_features", "experiment_id"], ascending=[True, True, True])
    best = ordered.iloc[0]
    return {
        "experiment_id": str(best["experiment_id"]),
        "rule": "min val WAPE, tie -> fewer features, then experiment_id",
        "n_features": int(best["n_features"]),
        "objective": str(best["objective"]),
        "val_mae": float(best["mae"]),
        "val_rmse": float(best["rmse"]),
        "val_wape": float(best["wape"]),
        "test_mae": float(best["test_mae"]),
        "test_rmse": float(best["test_rmse"]),
        "test_wape": float(best["test_wape"]),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    lcfg = dict(cfg.get("lgbm") or {})
    seed = int((cfg.get("project") or {}).get("seed", 42))
    n_estimators = int(args.n_estimators or lcfg.get("n_estimators", 2000))

    features_path = resolve(cfg["paths"]["features"])
    feature_list_path = resolve(cfg["paths"]["feature_list"])
    if not features_path.exists():
        raise FileNotFoundError(f"特征表不存在: {features_path}，请先运行 build_features.py")
    if not feature_list_path.exists():
        raise FileNotFoundError(f"特征清单不存在: {feature_list_path}")

    feature_meta = json.loads(feature_list_path.read_text(encoding="utf-8"))
    matrix = build_matrix(cfg, feature_meta)

    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [w for w in wanted if w not in {e.experiment_id for e in matrix}]
        if unknown:
            raise SystemExit(f"--only 指定了不存在的实验 {unknown}")
        matrix = [e for e in matrix if e.experiment_id in wanted]

    if args.dry_run:
        log.info("dry-run：共 %d 组实验，数据 %s", len(matrix), features_path)
        for e in matrix:
            log.info("  %-3s %-4d 特征 %-9s objective=%-14s %s", e.experiment_id, len(e.columns), e.feature_set, e.objective, e.note)
            log.info("       %s", ", ".join(e.columns))
        return

    log.info("读取特征表 %s", features_path)
    df = pd.read_parquet(features_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["zone_id"] = df["zone_id"].astype("category")

    scfg = cfg.get("split") or {}
    train_end, val_end = time_split_bounds(
        df["timestamp"], scfg.get("train", 0.70), scfg.get("val", 0.15)
    )
    masks = {
        "train": df["timestamp"] < train_end,
        "val": (df["timestamp"] >= train_end) & (df["timestamp"] < val_end),
        "test": df["timestamp"] >= val_end,
    }
    log.info("时间切分: train < %s | val < %s | test >= %s", train_end, val_end, val_end)
    log.info(
        "行数: train=%d, val=%d, test=%d；共 %d 组实验",
        masks["train"].sum(), masks["val"].sum(), masks["test"].sum(), len(matrix),
    )

    ocfg = cfg.get("output") or {}
    save_predictions = bool(ocfg.get("save_predictions", True)) and not args.no_save_predictions
    save_models = bool(ocfg.get("save_models", False)) or args.save_models
    pred_dir = ensure_dir(cfg["paths"].get("prediction_dir", "data/processed/experiments/predictions")) if save_predictions else None
    model_dir = ensure_dir(cfg["paths"].get("model_dir", "data/processed/experiments/models")) if save_models else None

    rows: list[dict] = []
    t_all = time.perf_counter()
    for exp in matrix:
        rows.append(train_one(exp, df, masks, lcfg, seed, n_estimators, pred_dir, model_dir))
    total_seconds = time.perf_counter() - t_all

    table = pd.DataFrame(rows)[RESULT_COLUMNS].sort_values("experiment_id", ignore_index=True)
    if args.out:
        out_path = ensure_parent(args.out)
    elif args.only:
        out_path = ensure_parent(
            Path(cfg["paths"].get("out_dir", "data/processed/experiments"))
            / f"lgbm_ablation_{'_'.join(table['experiment_id'])}.parquet"
        )
    else:
        out_path = ensure_parent(
            Path(cfg["paths"].get("out_dir", "data/processed/experiments")) / "lgbm_ablation.parquet"
        )
    table.to_parquet(out_path, compression="zstd", index=False)

    recommendation = pick_recommendation(table)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "data": {"features": str(features_path), "feature_list": str(feature_list_path)},
        "seed": seed,
        "n_estimators": n_estimators,
        "early_stopping_rounds": int(lcfg.get("early_stopping_rounds", 100)),
        "metric_for_early_stopping": lcfg.get("metric", "mae"),
        "split": {
            "train_end": str(train_end),
            "val_end": str(val_end),
            "n_train": int(masks["train"].sum()),
            "n_val": int(masks["val"].sum()),
            "n_test": int(masks["test"].sum()),
        },
        "versions": {"lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "total_seconds": round(float(total_seconds), 3),
        "experiments": [
            {
                "experiment_id": e.experiment_id,
                "feature_set": e.feature_set,
                "groups": list(e.groups),
                "objective": e.objective,
                "n_features": len(e.columns),
                "columns": list(e.columns),
                "note": e.note,
            }
            for e in matrix
        ],
        "metrics_split": "mae/rmse/wape = validation; test_* = holdout",
        "recommendation": recommendation,
        "out": str(out_path),
    }
    manifest_path = out_path.with_suffix(".manifest.json") if args.out else ensure_parent(
        Path(cfg["paths"].get("out_dir", "data/processed/experiments")) / "lgbm_ablation.manifest.json"
    )
    save_json(manifest, manifest_path)

    print("\n" + markdown_table(table) + "\n")
    log.info("推荐特征集（规则：%s）：%s（%d 特征，%s），val WAPE=%.2f%%，test WAPE=%.2f%%",
             recommendation["rule"], recommendation["experiment_id"], recommendation["n_features"],
             recommendation["objective"], recommendation["val_wape"] * 100, recommendation["test_wape"] * 100)
    log.info("长表 -> %s（%d 行）", out_path, len(table))
    log.info("清单 -> %s", manifest_path)
    log.info("全部实验总耗时 %.1f 秒", total_seconds)


if __name__ == "__main__":
    main()
