"""成员 C / C4：LightGBM 小规模超参数调优（随机搜索）与最佳配置产出。

约束（docs/tasks/member_c.md C4）:
    - 特征集固定为 C3 推荐的 A2（15 个特征），由 c_ablation.yaml 解析，不重新选特征；
    - 时间切分与 C1/C2/C3 完全一致（按时间 70/15/15）；
    - **只用验证集**（WAPE → MAE）选择，测试集只在最佳组确定后评估一次。

用法:
    python -m src.models.run_tuning --config configs/experiments/c_tuning.yaml --dry-run
    python -m src.models.run_tuning --config configs/experiments/c_tuning.yaml
    python -m src.models.run_tuning --config configs/experiments/c_tuning.yaml --resume      # 断点续跑
    python -m src.models.run_tuning --config configs/experiments/c_tuning.yaml --refit       # 只重跑最佳组终评
    python -m src.models.run_tuning --config configs/experiments/c_tuning.yaml --only-trial 3

产出:
    data/processed/experiments/lgbm_tuning.parquet        搜索长表（一行一组，只含验证集指标）
    data/processed/experiments/lgbm_tuning.manifest.json  搜索元信息 + 最佳组 + 一次性测试指标
    data/processed/models/c_best/lgbm_best.txt            最佳模型（另有 lgbm_latest.txt 供 predict.py）
    data/processed/models/c_best/lgbm_best.metrics.json   最佳模型指标（含 best_iteration / train_seconds）
    data/processed/feature_list_best.json                 A2 的 15 特征清单（供 lgbm_best.yaml 引用）
    data/processed/predictions/pred_test_lgbm_best.parquet 测试集预测（契约三列）
    configs/experiments/lgbm_best.yaml                    最佳配置（自动生成，带搜索元信息）

约定:
    - 调参阶段不会计算任何测试集指标；test 只出现在 fit_best() 里一次；
    - 默认不覆盖 B 消费的 data/processed/predictions/pred_test.parquet，需 --promote 才提升。
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from datetime import date, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.config import ensure_dir, ensure_parent, load_config, make_logger, resolve
from src.models.common import (
    make_prediction_frame,
    regression_metrics,
    save_json,
    time_split_bounds,
)
from src.models.run_ablation import build_matrix

log = make_logger("run_tuning")

PARAM_KEYS = [
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "colsample_bytree",
    "subsample",
    "reg_lambda",
]
RESULT_COLUMNS = [
    "trial_id",
    "source",
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "colsample_bytree",
    "subsample",
    "reg_lambda",
    "objective",
    "n_features",
    "val_mae",
    "val_rmse",
    "val_wape",
    "best_iteration",
    "train_seconds",
    "is_best",
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-search tuning for the LightGBM baseline.")
    p.add_argument("--config", default="configs/experiments/c_tuning.yaml")
    p.add_argument("--trials", type=int, default=None, help="覆盖 search.n_trials")
    p.add_argument("--seed", type=int, default=None, help="覆盖 search.seed（默认取 config）")
    p.add_argument("--resume", action="store_true", help="复用已有长表，跳过已完成的 trial")
    p.add_argument("--only-trial", type=int, default=None, help="只跑指定 trial（用于复现检查）")
    p.add_argument("--refit", action="store_true", help="跳过搜索，直接用长表里的最佳组重跑终评")
    p.add_argument("--dry-run", action="store_true", help="只打印抽样计划，不训练")
    p.add_argument("--promote", action="store_true", help="把最佳预测复制为 pred_test.parquet（会覆盖 B 的契约文件）")
    p.add_argument("--out", default=None, help="搜索长表输出路径")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 特征与抽样
# --------------------------------------------------------------------------- #
def resolve_features(cfg: dict) -> tuple[list[str], dict]:
    """按 C3 的推荐组解析特征列，避免在调参配置里重复抄写特征组。"""
    src = cfg.get("feature_source") or {}
    if src and src.get("config"):
        abl_cfg = load_config(src["config"])
        feature_meta = json.loads(resolve(cfg["paths"]["feature_list"]).read_text(encoding="utf-8"))
        matrix = build_matrix(abl_cfg, feature_meta)
        by_id = {e.experiment_id: e for e in matrix}
        exp_id = str(src.get("experiment_id") or "")
        if exp_id not in by_id:
            raise SystemExit(f"feature_source 指定的实验 {exp_id} 不存在；可用 {sorted(by_id)}")
        exp = by_id[exp_id]
        log.info("特征集固定为 C3 的 %s：%d 个特征（%s）", exp.experiment_id, len(exp.columns), exp.feature_set)
        return list(exp.columns), {
            "config": str(src["config"]),
            "experiment_id": exp.experiment_id,
            "feature_set": exp.feature_set,
        }
    cols = list(cfg.get("features", {}).get("columns") or [])
    if not cols:
        raise SystemExit("feature_source 与 features.columns 均为空，无法确定特征集")
    return cols, {"config": None, "experiment_id": None, "feature_set": "+".join(cols)}


def build_trials(cfg: dict, n_trials: int, seed: int) -> list[dict]:
    search = cfg.get("search") or {}
    space = search.get("space") or {}
    anchor = search.get("anchor") or {}
    if not space:
        raise SystemExit("配置缺少 search.space")
    for key in PARAM_KEYS:
        if key not in space:
            raise SystemExit(f"search.space 缺少 {key}")
        if key not in anchor:
            raise SystemExit(f"search.anchor 缺少 {key}")
        if anchor[key] not in space[key]:
            raise SystemExit(f"锚点的 {key}={anchor[key]} 不在搜索空间 {space[key]} 里")
    rng = random.Random(seed)
    trials: list[dict] = [
        {"trial_id": 0, "source": "anchor", **{k: anchor[k] for k in PARAM_KEYS}}
    ]
    seen = {tuple(sorted(anchor.items()))}
    attempts = 0
    while len(trials) < n_trials and attempts < 10000:
        attempts += 1
        cand = {k: rng.choice(list(space[k])) for k in PARAM_KEYS}
        key = tuple(sorted(cand.items()))
        if key in seen:
            continue
        seen.add(key)
        trials.append({"trial_id": len(trials), "source": "random", **cand})
    if len(trials) < n_trials:
        raise SystemExit(f"搜索空间不足以抽满 {n_trials} 组（实际 {len(trials)}）")
    return trials


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def build_params(lcfg: dict, trial: dict, seed: int) -> dict:
    return {
        "objective": lcfg.get("objective", "regression_l1"),
        "metric": lcfg.get("metric", "mae"),
        "learning_rate": trial["learning_rate"],
        "num_leaves": int(trial["num_leaves"]),
        "min_child_samples": int(trial["min_child_samples"]),
        "subsample": trial["subsample"],
        "subsample_freq": lcfg.get("subsample_freq", 1),
        "colsample_bytree": trial["colsample_bytree"],
        "reg_alpha": lcfg.get("reg_alpha", 0.0),
        "reg_lambda": trial["reg_lambda"],
        "seed": seed,
        "n_jobs": lcfg.get("n_jobs", -1),
        "verbose": -1,
        "device_type": lcfg.get("device", "cpu"),
    }


def train_and_score(
    trial: dict,
    X: pd.DataFrame,
    y: pd.Series,
    masks: dict[str, pd.Series],
    lcfg: dict,
    seed: int,
    n_estimators: int,
    evaluate_test: bool = False,
) -> tuple[dict, lgb.Booster, dict[str, np.ndarray]]:
    """训练一组参数；默认只评估验证集，evaluate_test=True 时才计算测试集指标。"""
    params = build_params(lcfg, trial, seed)
    train_mask, val_mask = masks["train"], masks["val"]
    dtrain = lgb.Dataset(X[train_mask], label=y[train_mask], free_raw_data=True)
    dval = lgb.Dataset(X[val_mask], label=y[val_mask], reference=dtrain, free_raw_data=True)

    t0 = time.perf_counter()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(int(lcfg.get("early_stopping_rounds", 100)), verbose=False),
            lgb.log_evaluation(period=1000),
        ],
    )
    train_seconds = time.perf_counter() - t0
    best_iter = int(booster.best_iteration or booster.current_iteration())
    if best_iter >= n_estimators:
        log.warning("trial %s 早停未触发，best_iteration 顶到 n_estimators=%d", trial.get("trial_id"), n_estimators)

    preds: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    splits = ("train", "val", "test") if evaluate_test else ("val",)
    for split in splits:
        mask = masks[split]
        pred = booster.predict(X[mask], num_iteration=best_iter)
        preds[split] = pred
        metrics[split] = regression_metrics(y[mask].to_numpy(), pred)

    row = {k: trial[k] for k in ("trial_id", "source", *PARAM_KEYS)}
    row.update(
        {
            "objective": params["objective"],
            "n_features": int(X.shape[1]),
            "val_mae": metrics["val"]["mae"],
            "val_rmse": metrics["val"]["rmse"],
            "val_wape": metrics["val"]["wape"],
            "best_iteration": best_iter,
            "train_seconds": round(float(train_seconds), 3),
        }
    )
    return row, booster, preds


def pick_best(table: pd.DataFrame) -> pd.Series:
    """预登记规则：val WAPE 最小 → val MAE → trial_id。"""
    ordered = table.sort_values(["val_wape", "val_mae", "trial_id"], ignore_index=True)
    return ordered.iloc[0]


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def save_table(rows: list[dict], path: Path) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    for col in RESULT_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan
    table = table[RESULT_COLUMNS].sort_values("trial_id", ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, compression="zstd", index=False)
    return table


def write_best_yaml(
    cfg: dict,
    path: Path,
    best: pd.Series,
    cols: list[str],
    feature_source: dict,
    metrics: dict[str, dict],
    n_trials: int,
    seed: int,
) -> None:
    """用 default.yaml 作骨架，覆盖路径/特征/超参，并写入搜索元信息。"""
    best_cfg = copy.deepcopy(load_config("configs/default.yaml"))
    best_cfg["paths"]["feature_list"] = cfg["paths"]["feature_list_best"]
    best_cfg["paths"]["model_dir"] = cfg["paths"]["model_dir"]
    best_cfg["paths"]["prediction_dir"] = cfg["paths"]["prediction_dir"]
    best_cfg["features"]["columns"] = list(cols)
    best_cfg["lgbm"] = {
        "objective": str(best["objective"]),
        "metric": cfg["lgbm"].get("metric", "mae"),
        "n_estimators": int(cfg["lgbm"].get("n_estimators", 6000)),
        "early_stopping_rounds": int(cfg["lgbm"].get("early_stopping_rounds", 100)),
        "learning_rate": float(best["learning_rate"]),
        "num_leaves": int(best["num_leaves"]),
        "min_child_samples": int(best["min_child_samples"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "subsample": float(best["subsample"]),
        "subsample_freq": int(cfg["lgbm"].get("subsample_freq", 1)),
        "reg_alpha": float(cfg["lgbm"].get("reg_alpha", 0.0)),
        "reg_lambda": float(best["reg_lambda"]),
        "n_jobs": int(cfg["lgbm"].get("n_jobs", -1)),
        "device": str(cfg["lgbm"].get("device", "cpu")),
    }
    best_cfg["search"] = {
        "date": date.today().isoformat(),
        "seed": int(seed),
        "n_trials": int(n_trials),
        "selection": "val WAPE -> val MAE -> trial_id",
        "space": cfg["search"]["space"],
        "best_trial": int(best["trial_id"]),
        "feature_source": feature_source,
        "best_val": {
            "mae": float(best["val_mae"]),
            "rmse": float(best["val_rmse"]),
            "wape": float(best["val_wape"]),
            "best_iteration": int(best["best_iteration"]),
        },
        # 测试集只用于报告，不参与任何选择
        "test": {
            "mae": float(metrics["test"]["mae"]),
            "rmse": float(metrics["test"]["rmse"]),
            "wape": float(metrics["test"]["wape"]),
        },
    }
    header = (
        "# =============================================================================\n"
        "# 成员 C / C4：LightGBM 最佳配置\n"
        f"# 由 src/models/run_tuning.py 自动生成于 {datetime.now().isoformat(timespec='seconds')}，请勿手工修改\n"
        f"# 随机搜索：seed={seed}，共 {n_trials} 组（1 锚点 + {n_trials - 1} 随机），按 val WAPE 选择\n"
        f"# 特征集：C3 推荐 {feature_source.get('experiment_id')}（{feature_source.get('feature_set')}，{len(cols)} 个特征）\n"
        f"# 训练预算：{best_cfg['lgbm']['n_estimators']} 轮 + {best_cfg['lgbm']['early_stopping_rounds']} 轮早停，"
        "metric=mae，CPU\n"
        "# =============================================================================\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(best_cfg, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(header + body, encoding="utf-8")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    lcfg = dict(cfg.get("lgbm") or {})
    search = dict(cfg.get("search") or {})
    seed = int(args.seed if args.seed is not None else search.get("seed", 42))
    n_trials = int(args.trials if args.trials is not None else search.get("n_trials", 18))
    n_estimators = int(lcfg.get("n_estimators", 6000))

    cols, feature_source = resolve_features(cfg)
    tests_dir = resolve(cfg["paths"].get("out_dir", "data/processed/experiments"))
    default_out = tests_dir / "lgbm_tuning.parquet"
    out_path = resolve(args.out) if args.out else default_out

    trials = build_trials(cfg, n_trials, seed)
    if args.only_trial is not None:
        trials = [t for t in trials if t["trial_id"] == args.only_trial]
        if not trials:
            raise SystemExit(f"trial {args.only_trial} 不存在（共 {n_trials} 组）")
        if not args.out:
            out_path = tests_dir / f"lgbm_tuning_trial{args.only_trial}.parquet"

    if args.dry_run:
        log.info("dry-run：%d 组试验，特征 %d 个（%s）", len(trials), len(cols), feature_source.get("experiment_id"))
        for t in trials:
            log.info(
                "  #%02d %-6s lr=%.2f leaves=%3d mcs=%3d colsample=%.1f subsample=%.1f lambda=%.1f",
                t["trial_id"], t["source"], t["learning_rate"], t["num_leaves"],
                t["min_child_samples"], t["colsample_bytree"], t["subsample"], t["reg_lambda"],
            )
        return

    features_path = resolve(cfg["paths"]["features"])
    if not features_path.exists():
        raise FileNotFoundError(f"特征表不存在: {features_path}")
    log.info("读取特征表 %s", features_path)
    df = pd.read_parquet(features_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["zone_id"] = df["zone_id"].astype("category")
    y = df["demand"].astype("float32")
    X = df[list(cols)]

    scfg = cfg.get("split") or {}
    train_end, val_end = time_split_bounds(df["timestamp"], scfg.get("train", 0.70), scfg.get("val", 0.15))
    masks = {
        "train": df["timestamp"] < train_end,
        "val": (df["timestamp"] >= train_end) & (df["timestamp"] < val_end),
        "test": df["timestamp"] >= val_end,
    }
    log.info("时间切分: train < %s | val < %s | test >= %s", train_end, val_end, val_end)
    log.info("行数: train=%d, val=%d, test=%d", masks["train"].sum(), masks["val"].sum(), masks["test"].sum())

    # ---------- 搜索或复现 ----------
    if args.refit:
        if not out_path.exists():
            raise FileNotFoundError(f"找不到搜索长表 {out_path}，无法 --refit")
        table = pd.read_parquet(out_path)
        best = pick_best(table)
        log.info("--refit：直接使用长表中的最佳组 trial %s", int(best["trial_id"]))
        trial = {k: best[k] for k in ("trial_id", "source", *PARAM_KEYS)}
        n_trials = int(table["trial_id"].max()) + 1
    else:
        rows: list[dict] = []
        done: set[int] = set()
        if args.resume and out_path.exists():
            prev = pd.read_parquet(out_path)
            rows = prev.to_dict("records")
            done = {int(x) for x in prev["trial_id"]}
            log.info("--resume：已有 %d 组结果，跳过它们", len(done))
        for trial in trials:
            if int(trial["trial_id"]) in done:
                continue
            log.info(
                "trial %02d/%d（%s）lr=%.2f leaves=%d mcs=%d colsample=%.2f subsample=%.2f lambda=%.1f",
                trial["trial_id"], n_trials, trial["source"], trial["learning_rate"],
                trial["num_leaves"], trial["min_child_samples"], trial["colsample_bytree"],
                trial["subsample"], trial["reg_lambda"],
            )
            row, _, _ = train_and_score(trial, X, y, masks, lcfg, seed, n_estimators, evaluate_test=False)
            log.info(
                "trial %02d 完成：val MAE=%.4f RMSE=%.4f WAPE=%.2f%% best_iter=%d %.1fs",
                row["trial_id"], row["val_mae"], row["val_rmse"], row["val_wape"] * 100,
                row["best_iteration"], row["train_seconds"],
            )
            rows.append(row)
            save_table(rows, out_path)
        table = save_table(rows, out_path)
        best = pick_best(table)
        log.info(
            "最佳组：trial %s（val WAPE=%.2f%%，MAE=%.4f）",
            int(best["trial_id"]), best["val_wape"] * 100, best["val_mae"],
        )
        if args.only_trial is not None:
            log.info("--only-trial 模式：不写最佳模型与配置")
            return
        # 注意：此处不要复用上面的循环变量 trial（那是最后一组），必须按最佳行重建参数
        trial = {k: best[k] for k in ("trial_id", "source", *PARAM_KEYS)}

    # ---------- 最佳组终评（唯一一次触碰测试集）----------
    row, booster, preds = train_and_score(trial, X, y, masks, lcfg, seed, n_estimators, evaluate_test=True)
    metrics = {s: regression_metrics(y[masks[s]].to_numpy(), preds[s]) for s in ("train", "val", "test")}
    # 同一组参数、同一切分、同一 seed：重跑必须与搜索阶段的验证集指标逐位一致
    diffs = {
        "mae": abs(row["val_mae"] - float(best["val_mae"])),
        "rmse": abs(row["val_rmse"] - float(best["val_rmse"])),
        "wape": abs(row["val_wape"] - float(best["val_wape"])),
        "best_iteration": abs(int(row["best_iteration"]) - int(best["best_iteration"])),
    }
    if max(diffs.values()) > 1e-9:
        raise SystemExit(f"终评与搜索阶段结果不一致（不可复现）：{diffs}")
    log.info("终评复现自检通过：val 指标与搜索阶段逐位一致")

    table = table.copy()
    table["is_best"] = (table["trial_id"] == int(best["trial_id"])).astype(bool)
    save_table(table.to_dict("records"), out_path)

    # 模型与指标
    model_dir = ensure_dir(cfg["paths"]["model_dir"])
    model_path = model_dir / "lgbm_best.txt"
    booster.save_model(str(model_path), num_iteration=int(row["best_iteration"]))
    shutil.copy2(model_path, model_dir / "lgbm_latest.txt")
    log.info("最佳模型 -> %s", model_path)

    meta = {
        "model": "lightgbm",
        "tag": "lgbm_best",
        "kind": "tuned_best",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_trial": int(best["trial_id"]),
        "selection_rule": "val WAPE -> val MAE -> trial_id",
        "search": {
            "date": date.today().isoformat(),
            "seed": seed,
            "n_trials": int(n_trials),
            "space": search.get("space"),
            "feature_source": feature_source,
        },
        "best_params": {k: (int(best[k]) if k in ("num_leaves", "min_child_samples") else float(best[k]))
                        for k in PARAM_KEYS},
        "best_iteration": int(row["best_iteration"]),
        "train_seconds": round(float(row["train_seconds"]), 3),
        "n_features": len(cols),
        "features": list(cols),
        "split": {"train_end": str(train_end), "val_end": str(val_end)},
        "rows": {k: int(v.sum()) for k, v in masks.items()},
        "metrics": metrics,
        "versions": {"lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "data": str(features_path),
    }
    save_json(meta, model_dir / "lgbm_best.metrics.json")
    save_json(meta, model_dir / "lgbm_latest.metrics.json")

    # 特征清单（15 个），供 lgbm_best.yaml 的 predict/check 流程引用
    feature_list_path = ensure_parent(cfg["paths"]["feature_list_best"])
    save_json(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": feature_source,
            "features": list(cols),
            "target": "demand",
            "keys": ["timestamp", "zone_id"],
            "n_rows": int(len(df)),
            "note": "C4 调优最终模型使用的特征子集（C3 推荐 A2）",
        },
        feature_list_path,
    )

    # 预测（契约三列 + model_name）
    pred_dir = ensure_dir(cfg["paths"]["prediction_dir"])
    pred_frame = make_prediction_frame(
        df.loc[masks["test"], "timestamp"].reset_index(drop=True),
        df.loc[masks["test"], "zone_id"].astype("int16").reset_index(drop=True),
        preds["test"],
        "lgbm_best",
    )
    pred_path = pred_dir / "pred_test_lgbm_best.parquet"
    pred_frame.to_parquet(pred_path, compression="zstd", index=False)
    log.info("最佳模型测试集预测 -> %s（%d 行）", pred_path, len(pred_frame))
    promote = bool(args.promote or (cfg.get("output") or {}).get("promote", False))
    if promote:
        shutil.copy2(pred_path, pred_dir / "pred_test.parquet")
        log.warning("已 --promote：覆盖 %s（B 的契约文件）", pred_dir / "pred_test.parquet")

    # 最佳配置
    best_cfg_path = ensure_parent("configs/experiments/lgbm_best.yaml")
    write_best_yaml(cfg, best_cfg_path, best, cols, feature_source, metrics, n_trials, seed)
    log.info("最佳配置 -> %s", best_cfg_path)

    # 清单
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "data": {"features": str(features_path), "feature_list": str(feature_list_path)},
        "features": {"experiment_id": feature_source.get("experiment_id"), "columns": list(cols)},
        "split": {
            "train_end": str(train_end),
            "val_end": str(val_end),
            "n_train": int(masks["train"].sum()),
            "n_val": int(masks["val"].sum()),
            "n_test": int(masks["test"].sum()),
        },
        "search": {
            "seed": seed,
            "n_trials": int(n_trials),
            "space": search.get("space"),
            "anchor_trial": 0,
            "selection_rule": "val WAPE -> val MAE -> trial_id",
            "best_trial": int(best["trial_id"]),
        },
        "metrics": {
            "val": {"mae": float(best["val_mae"]), "rmse": float(best["val_rmse"]), "wape": float(best["val_wape"])},
            "test": metrics["test"],
            "train": metrics["train"],
            "note": "搜索阶段只用验证集；test 仅在最佳组确定后评估一次",
        },
        "versions": {"lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "out": {"table": str(out_path), "model": str(model_path), "prediction": str(pred_path),
                "config": str(best_cfg_path), "promoted": promote},
    }
    save_json(manifest, tests_dir / "lgbm_tuning.manifest.json")

    log.info("搜索长表 -> %s（%d 行）", out_path, len(table))
    log.info(
        "最佳组 test：MAE=%.4f RMSE=%.4f WAPE=%.2f%%",
        metrics["test"]["mae"], metrics["test"]["rmse"], metrics["test"]["wape"] * 100,
    )


if __name__ == "__main__":
    main()
