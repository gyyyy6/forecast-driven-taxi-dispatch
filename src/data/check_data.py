"""需求面板 / 特征表 / 预测文件的契约校验（成员 C，任务书 C1）。

在校验集合上做四类检查，任何一项 FAIL 都以退出码 1 结束，便于 CI 或人工复核：

    panel        行数 == 时间片 × 区域、键唯一、15min 步长无缺口、区域 1..263、
                 demand 非负整型无 NaN、dtype 符合 docs/api_contract.md
    features     feature_list.json 声明的特征列齐全、无 inf、
                 NaN 只出现在滞后/滚动列的暖机前缀（按区域计）
    split        70/15/15 时间切分边界、三段互不重叠、各段网格完整
    leak         防泄漏断言：lag_k == demand.shift(k)、
                 roll_mean_w == shift(1).rolling(w).mean()
    predictions  契约三列 + 类型、键唯一、无负值无 NaN、覆盖测试期全网格，
                 并能与特征表对齐复算 MAE/RMSE/WAPE

用法:
    python -m src.data.check_data --config configs/experiments/c_baseline.yaml
    python -m src.data.check_data --config configs/experiments/c_baseline.yaml --what panel,split
    python -m src.data.check_data --config configs/experiments/c_baseline.yaml \
        --what predictions --predictions data/processed/predictions/pred_test_lgbm_c_baseline.parquet
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import ensure_parent, load_config, make_logger, resolve
from src.models.common import regression_metrics, time_split_bounds

log = make_logger("check_data")

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
SECTIONS = ("panel", "features", "leak", "split", "predictions")


# --------------------------------------------------------------------------- #
# 检查结果收集
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    section: str
    check_id: str
    description: str
    status: str
    detail: str = ""


@dataclass
class Checker:
    results: list[Result] = field(default_factory=list)

    def run(self, section: str, check_id: str, description: str, fn: Callable[[], tuple[str, str]]) -> Result:
        try:
            status, detail = fn()
        except Exception as exc:  # 检查本身出错也算 FAIL，避免静默跳过
            status, detail = FAIL, f"检查抛出异常: {type(exc).__name__}: {exc}"
        status = status.upper()
        if status not in (PASS, FAIL, WARN, SKIP):
            status, detail = FAIL, f"未知状态 {status}: {detail}"
        res = Result(section, check_id, description, status, detail)
        self.results.append(res)
        log.info("[%s] %-4s %s | %s", section, status, description, detail)
        return res

    @property
    def failures(self) -> list[Result]:
        return [r for r in self.results if r.status == FAIL]

    def summary(self) -> dict[str, int]:
        out = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
        for r in self.results:
            out[r.status] += 1
        return out


def _fmt(value: float, ndigits: int = 4) -> str:
    return f"{value:.{ndigits}f}"


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
def load_panel(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rename = {}
    if "timestamp" not in df.columns:
        for cand in ("ts", "datetime", "pickup_datetime"):
            if cand in df.columns:
                rename[cand] = "timestamp"
                break
    df = df.rename(columns=rename)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["timestamp", "zone_id"], ignore_index=True)


def load_features(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
def check_panel(chk: Checker, panel: pd.DataFrame, cfg: dict[str, Any]) -> None:
    dcfg = cfg.get("data", {})
    freq_str = str(dcfg.get("freq", "15min"))
    # 用 to_timedelta 而非 Timedelta(str)，避免 pandas 2.3 + numpy 2.x 的
    # "generic unit for NumPy timedelta" DeprecationWarning
    freq = pd.to_timedelta(freq_str)
    zone_min = int(dcfg.get("zone_min", 1))
    zone_max = int(dcfg.get("zone_max", 263))
    n_ts = panel["timestamp"].nunique()
    n_zone = panel["zone_id"].nunique()

    def schema():
        missing = {"timestamp", "zone_id", "demand"} - set(panel.columns)
        if missing:
            return FAIL, f"缺少列 {sorted(missing)}"
        bad = []
        if not pd.api.types.is_datetime64_any_dtype(panel["timestamp"]):
            bad.append(f"timestamp={panel['timestamp'].dtype}")
        if not pd.api.types.is_integer_dtype(panel["zone_id"]):
            bad.append(f"zone_id={panel['zone_id'].dtype}")
        if not pd.api.types.is_integer_dtype(panel["demand"]):
            bad.append(f"demand={panel['demand'].dtype}")
        if bad:
            return FAIL, "；".join(bad)
        return PASS, (f"timestamp={panel['timestamp'].dtype}, zone_id={panel['zone_id'].dtype}, "
                      f"demand={panel['demand'].dtype}")

    def unique_keys():
        dup = int(panel.duplicated(subset=["timestamp", "zone_id"]).sum())
        return (PASS, "0 条重复键") if dup == 0 else (FAIL, f"{dup} 条重复键")

    def full_grid():
        expect = n_ts * n_zone
        if len(panel) != expect:
            return FAIL, f"行数 {len(panel)} != {n_ts} 时间片 × {n_zone} 区域 = {expect}"
        return PASS, f"{len(panel)} 行 = {n_ts} × {n_zone}"

    def cadence():
        uniq = pd.Index(panel["timestamp"].unique()).sort_values()
        if len(uniq) < 2:
            return FAIL, f"时间片只有 {len(uniq)} 个"
        diffs = pd.Series(uniq).diff().dropna()
        bad = diffs[diffs != freq]
        if len(bad):
            return FAIL, f"{len(bad)} 处步长不是 {freq_str}（例如 {bad.iloc[0]}）"
        off_grid = (uniq.minute % int(freq.total_seconds() // 60)) != 0
        if off_grid.any():
            return FAIL, "存在未对齐到时间片起点的 timestamp"
        return PASS, f"步长恒为 {freq_str}，无缺口（{len(uniq)} 个时间片，{uniq.min()} ~ {uniq.max()}）"

    def zone_domain():
        zmin, zmax, n = int(panel["zone_id"].min()), int(panel["zone_id"].max()), int(panel["zone_id"].nunique())
        expect_n = zone_max - zone_min + 1
        if zmin < zone_min or zmax > zone_max:
            return FAIL, f"区域越界 {zmin}~{zmax}，约定 {zone_min}~{zone_max}"
        if n != expect_n:
            return FAIL, f"区域数 {n} != {expect_n}"
        return PASS, f"zone_id 覆盖 {zmin}~{zmax}，共 {n} 个"

    def demand_values():
        d = panel["demand"]
        problems = []
        if d.isna().any():
            problems.append(f"NaN={int(d.isna().sum())}")
        if (d < 0).any():
            problems.append(f"负值={int((d < 0).sum())}")
        if not np.all(np.equal(np.mod(d.dropna().to_numpy(), 1), 0)):
            problems.append("存在非整数取值")
        if problems:
            return FAIL, "；".join(problems)
        return PASS, (f"min={int(d.min())} max={int(d.max())} sum={int(d.sum())} "
                      f"mean={_fmt(float(d.mean()), 2)} zero_ratio={_fmt(float((d == 0).mean()))}")

    chk.run("panel", "P1", "表结构与 dtype 符合接口契约", schema)
    chk.run("panel", "P2", "(timestamp, zone_id) 唯一", unique_keys)
    chk.run("panel", "P3", "行数 = 时间片 × 区域（完整网格）", full_grid)
    chk.run("panel", "P4", f"时间步长恒为 {freq_str} 且无缺口", cadence)
    chk.run("panel", "P5", f"区域取值落在 {zone_min}~{zone_max}", zone_domain)
    chk.run("panel", "P6", "demand 非负整型、无 NaN", demand_values)


# --------------------------------------------------------------------------- #
# features / leak
# --------------------------------------------------------------------------- #
def check_features(chk: Checker, feats: pd.DataFrame, feature_meta: dict[str, Any], cfg: dict[str, Any]) -> None:
    declared = list(feature_meta.get("features", []))
    fcfg = cfg.get("features", {})
    lags = [int(x) for x in fcfg.get("lags", [])]
    windows = [int(x) for x in fcfg.get("rolling_windows", [])]
    ordered = feats.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    pos = ordered.groupby("zone_id", sort=False).cumcount()

    def declared_present():
        missing = [c for c in declared if c not in feats.columns]
        if missing:
            return FAIL, f"缺少特征列 {missing}"
        if "demand" in declared:
            return FAIL, "demand（标签）被声明为特征"
        return PASS, f"{len(declared)} 个特征全部存在: {', '.join(declared)}"

    def finite():
        num = ordered[[c for c in declared if c != "zone_id"]]
        inf_cols = [c for c in num.columns if np.isinf(num[c].to_numpy()).any()]
        if inf_cols:
            return FAIL, f"含 inf 的列: {inf_cols}"
        return PASS, "无非有限值（inf）"

    def warmup_nan():
        problems, notes = [], []
        for lag in lags:
            col = f"lag_{lag}"
            if col not in ordered.columns:
                continue
            isna = ordered[col].isna()
            expect = pos < lag
            if not bool((isna == expect).all()):
                problems.append(f"{col} 的 NaN 不满足“每区域前 {lag} 行”")
            notes.append(f"{col}:{_fmt(float(isna.mean())*100, 2)}%")
        for w in windows:
            col = f"roll_mean_{w}"
            if col not in ordered.columns:
                continue
            isna = ordered[col].isna()
            expect = pos < w
            if not bool((isna == expect).all()):
                problems.append(f"{col} 的 NaN 不满足“每区域前 {w} 行”")
            notes.append(f"{col}:{_fmt(float(isna.mean())*100, 2)}%")
        if problems:
            return FAIL, "；".join(problems)
        return PASS, "NaN 只出现在暖机前缀（" + ", ".join(notes) + "）"

    def constant_cols():
        num = ordered[[c for c in declared if c != "zone_id"]]
        const = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
        if const:
            return WARN, f"常量特征（无信息量）: {const}"
        return PASS, "所有特征都有变化"

    def leak_lag():
        demand = ordered["demand"].astype("float64")
        g = demand.groupby(ordered["zone_id"], sort=False)
        bad = []
        for lag in lags:
            col = f"lag_{lag}"
            if col not in ordered.columns:
                continue
            expect = g.shift(lag).to_numpy()
            got = ordered[col].to_numpy(dtype="float64")
            if not np.allclose(got, expect, equal_nan=True, rtol=0, atol=1e-5):
                bad.append(col)
        if bad:
            return FAIL, f"与 demand.shift(k) 不一致（存在泄漏或错位）: {bad}"
        return PASS, f"{', '.join('lag_' + str(k) for k in lags)} 与 demand.shift(k) 完全一致"

    def leak_roll():
        demand = ordered["demand"].astype("float64")
        g = demand.groupby(ordered["zone_id"], sort=False)
        bad = []
        for w in windows:
            col = f"roll_mean_{w}"
            if col not in ordered.columns:
                continue
            expect = g.transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=w).mean()
            ).to_numpy()
            got = ordered[col].to_numpy(dtype="float64")
            if not np.allclose(got, expect, equal_nan=True, rtol=0, atol=1e-4):
                bad.append(col)
        if bad:
            return FAIL, f"与 shift(1).rolling(w).mean 不一致: {bad}"
        return PASS, f"{', '.join('roll_mean_' + str(w) for w in windows)} 均先 shift(1) 再滚动"

    chk.run("features", "F1", "feature_list.json 声明的特征列齐全", declared_present)
    chk.run("features", "F2", "特征无 inf", finite)
    chk.run("features", "F3", "NaN 仅来自暖机前缀", warmup_nan)
    chk.run("features", "F4", "特征不是常量列", constant_cols)
    chk.run("leak", "L1", "滞后特征 = demand.shift(k)", leak_lag)
    chk.run("leak", "L2", "滚动均值 = shift(1) 后 rolling(w).mean", leak_roll)


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def check_split(chk: Checker, panel: pd.DataFrame, cfg: dict[str, Any]) -> None:
    scfg = cfg.get("split", {})
    train_ratio = float(scfg.get("train", 0.70))
    val_ratio = float(scfg.get("val", 0.15))

    def bounds():
        train_end, val_end = time_split_bounds(panel["timestamp"], train_ratio, val_ratio)
        tr = panel[panel["timestamp"] < train_end]
        va = panel[(panel["timestamp"] >= train_end) & (panel["timestamp"] < val_end)]
        te = panel[panel["timestamp"] >= val_end]
        parts = {"train": tr, "val": va, "test": te}
        if min(len(v) for v in parts.values()) == 0:
            return FAIL, "存在空切分"
        desc = [];
        for name, part in parts.items():
            ts = part["timestamp"]
            desc.append(f"{name}: {part['timestamp'].nunique()} 片/{len(part)} 行 "
                        f"[{ts.min()} ~ {ts.max()}]")
        log.info("切分边界: train < %s | val < %s | test >= %s", train_end, val_end, val_end)
        return PASS, "；".join(desc)

    def no_overlap():
        t = pd.Index(panel["timestamp"].unique()).sort_values()
        n = len(t)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        s_tr, s_va, s_te = set(t[:n_train]), set(t[n_train:n_train + n_val]), set(t[n_train + n_val:])
        ov = (len(s_tr & s_va), len(s_va & s_te), len(s_tr & s_te))
        if any(ov):
            return FAIL, f"时间重叠 (train,val)={ov[0]} (val,test)={ov[1]} (train,test)={ov[2]}"
        return PASS, "三段时间戳零重叠"

    def ordered():
        train_end, val_end = time_split_bounds(panel["timestamp"], train_ratio, val_ratio)
        tr_max = panel.loc[panel["timestamp"] < train_end, "timestamp"].max()
        va = panel[(panel["timestamp"] >= train_end) & (panel["timestamp"] < val_end)]["timestamp"]
        te_min = panel.loc[panel["timestamp"] >= val_end, "timestamp"].min()
        if not (tr_max < va.min() <= va.max() < te_min):
            return FAIL, f"时间顺序异常: train_max={tr_max}, val=[{va.min()}, {va.max()}], test_min={te_min}"
        return PASS, f"train_max={tr_max} < val[{va.min()} ~ {va.max()}] < test_min={te_min}"

    def per_split_grid():
        train_end, val_end = time_split_bounds(panel["timestamp"], train_ratio, val_ratio)
        n_zone = panel["zone_id"].nunique()
        parts = {
            "train": panel[panel["timestamp"] < train_end],
            "val": panel[(panel["timestamp"] >= train_end) & (panel["timestamp"] < val_end)],
            "test": panel[panel["timestamp"] >= val_end],
        }
        bad = []
        for name, part in parts.items():
            expect = part["timestamp"].nunique() * n_zone
            if len(part) != expect:
                bad.append(f"{name}: {len(part)} != {expect}")
        if bad:
            return FAIL, "；".join(bad)
        return PASS, "；".join(f"{k}: {len(v)} 行" for k, v in parts.items())

    chk.run("split", "S1", "按时间戳 70/15/15 切分且三段非空", bounds)
    chk.run("split", "S2", "三段时间戳互不重叠", no_overlap)
    chk.run("split", "S3", "切分严格按时间先后顺序", ordered)
    chk.run("split", "S4", "每段都是完整网格", per_split_grid)


# --------------------------------------------------------------------------- #
# predictions
# --------------------------------------------------------------------------- #
def check_predictions(
    chk: Checker,
    pred: pd.DataFrame,
    panel: pd.DataFrame,
    feats: pd.DataFrame | None,
    cfg: dict[str, Any],
) -> None:
    scfg = cfg.get("split", {})
    train_end, val_end = time_split_bounds(panel["timestamp"], scfg.get("train", 0.70), scfg.get("val", 0.15))
    test = panel[panel["timestamp"] >= val_end]
    expect_rows = test["timestamp"].nunique() * panel["zone_id"].nunique()

    def schema():
        missing = {"timestamp", "zone_id", "predicted_demand"} - set(pred.columns)
        if missing:
            return FAIL, f"缺少列 {sorted(missing)}（实际列 {list(pred.columns)}）"
        try:
            ts = pd.to_datetime(pred["timestamp"])
        except Exception as exc:
            return FAIL, f"timestamp 无法解析: {exc}"
        if ts.isna().any():
            return FAIL, f"{int(ts.isna().sum())} 行 timestamp 解析失败"
        bad = []
        if not pd.api.types.is_integer_dtype(pred["zone_id"]):
            bad.append(f"zone_id={pred['zone_id'].dtype}（应为整数）")
        if not pd.api.types.is_float_dtype(pred["predicted_demand"]):
            bad.append(f"predicted_demand={pred['predicted_demand'].dtype}（应为 float）")
        if bad:
            return FAIL, "；".join(bad)
        return PASS, (f"timestamp={pred['timestamp'].dtype}, zone_id={pred['zone_id'].dtype}, "
                      f"predicted_demand={pred['predicted_demand'].dtype}")

    def values():
        p = pred["predicted_demand"]
        problems = []
        if p.isna().any():
            problems.append(f"NaN={int(p.isna().sum())}")
        if (p < 0).any():
            problems.append(f"负值={int((p < 0).sum())}")
        if problems:
            return FAIL, "；".join(problems)
        return PASS, (f"min={_fmt(float(p.min()))} max={_fmt(float(p.max()))} "
                      f"mean={_fmt(float(p.mean()))}")

    def unique_keys():
        dup = int(pred.duplicated(subset=["timestamp", "zone_id"]).sum())
        return (PASS, "0 条重复键") if dup == 0 else (FAIL, f"{dup} 条重复键")

    def coverage():
        n_ts = pred["timestamp"].nunique()
        n_zone = pred["zone_id"].nunique()
        expect_ts = test["timestamp"].nunique()
        if len(pred) != expect_rows or n_ts != expect_ts or n_zone != panel["zone_id"].nunique():
            return FAIL, (f"覆盖不全: {len(pred)} 行 / {n_ts} 片 × {n_zone} 区域，"
                          f"应为 {expect_rows} 行 / {expect_ts} 片 × {panel['zone_id'].nunique()} 区域")
        if pred["timestamp"].min() != test["timestamp"].min():
            return FAIL, f"测试期起点不符: {pred['timestamp'].min()} != {test['timestamp'].min()}"
        return PASS, f"{len(pred)} 行 = {n_ts} 片 × {n_zone} 区域，{pred['timestamp'].min()} ~ {pred['timestamp'].max()}"

    def align_and_score():
        if feats is None or "demand" not in feats.columns:
            return SKIP, "未提供含 demand 的特征表，跳过指标复核"
        truth = feats[["timestamp", "zone_id", "demand"]].copy()
        truth["timestamp"] = pd.to_datetime(truth["timestamp"])
        truth = truth[truth["timestamp"] >= val_end]
        merged = pred[["timestamp", "zone_id", "predicted_demand"]].merge(
            truth, on=["timestamp", "zone_id"], how="left", validate="one_to_one"
        )
        if merged["demand"].isna().any():
            return FAIL, f"{int(merged['demand'].isna().sum())} 行预测对不上真实 demand"
        m = regression_metrics(merged["demand"].to_numpy(), merged["predicted_demand"].to_numpy())
        return PASS, (f"test MAE={_fmt(m['mae'])} RMSE={_fmt(m['rmse'])} "
                      f"WAPE={_fmt(m['wape'] * 100, 2)}% n={m['n']}")

    chk.run("predictions", "R1", "契约列与 dtype 正确", schema)
    chk.run("predictions", "R2", "预测值非负且无 NaN", values)
    chk.run("predictions", "R3", "(timestamp, zone_id) 唯一", unique_keys)
    chk.run("predictions", "R4", "覆盖测试期全部 timestamp × 区域", coverage)
    chk.run("predictions", "R5", "与特征表对齐并复算测试集指标", align_and_score)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate panel / features / predictions against the API contract.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--what", default="all", help=f"逗号分隔: {', '.join(SECTIONS)} 或 all")
    p.add_argument("--panel", default=None, help="默认读取 paths.panel")
    p.add_argument("--features", default=None, help="默认读取 paths.features")
    p.add_argument("--predictions", default=None, help="默认读取 prediction_dir/pred_<split>.parquet")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out", default=None, help="JSON 报告路径，默认 log_dir/data_checks.json")
    return p.parse_args()


def resolve_sections(what: str) -> list[str]:
    if what.strip().lower() == "all":
        return list(SECTIONS)
    wanted = [w.strip().lower() for w in what.split(",") if w.strip()]
    unknown = [w for w in wanted if w not in SECTIONS]
    if unknown:
        raise SystemExit(f"未知的检查分区: {unknown}，可选 {SECTIONS} 或 all")
    return [s for s in SECTIONS if s in wanted]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    sections = resolve_sections(args.what)
    panel_path = resolve(args.panel or cfg["paths"]["panel"])
    features_path = resolve(args.features or cfg["paths"]["features"])
    feature_list_path = resolve(cfg["paths"]["feature_list"])
    pred_path = resolve(
        args.predictions
        or (resolve(cfg["paths"]["prediction_dir"]) / f"pred_{args.split}.parquet")
    )

    need_features = bool({"features", "leak"} & set(sections))
    need_panel = bool({"panel", "split", "predictions"} & set(sections))

    panel = load_panel(panel_path) if need_panel else None
    feats = None
    feature_meta: dict[str, Any] = {}
    if need_features or "predictions" in sections:
        if not features_path.exists():
            raise FileNotFoundError(f"特征表不存在: {features_path}")
        feats = load_features(features_path)
    if need_features:
        if not feature_list_path.exists():
            raise FileNotFoundError(f"特征清单不存在: {feature_list_path}")
        feature_meta = json.loads(feature_list_path.read_text(encoding="utf-8"))

    log.info("配置文件 %s", args.config)
    log.info("面板 %s（%d 行）", panel_path, len(panel) if panel is not None else 0)
    if feats is not None:
        log.info("特征 %s（%d 行 × %d 列）", features_path, len(feats), len(feats.columns))

    chk = Checker()
    if "panel" in sections:
        check_panel(chk, panel, cfg)
    if "features" in sections:
        check_features(chk, feats, feature_meta, cfg)
    if "split" in sections:
        check_split(chk, panel, cfg)
    if "predictions" in sections:
        if not pred_path.exists():
            chk.run("predictions", "R0", f"预测文件存在: {pred_path}", lambda: (FAIL, "文件不存在"))
        else:
            pred = pd.read_parquet(pred_path)
            log.info("预测 %s（%d 行，列 %s）", pred_path, len(pred), list(pred.columns))
            check_predictions(chk, pred, panel, feats, cfg)

    summary = chk.summary()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "sections": sections,
        "files": {
            "panel": str(panel_path),
            "features": str(features_path),
            "predictions": str(pred_path) if "predictions" in sections else None,
        },
        "summary": summary,
        "ok": summary[FAIL] == 0,
        "results": [r.__dict__ for r in chk.results],
    }
    out_path = resolve(args.out) if args.out else resolve(cfg["paths"].get("log_dir", "data/processed/logs")) / "data_checks.json"
    out_path = ensure_parent(out_path)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("检查汇总: PASS=%d FAIL=%d WARN=%d SKIP=%d", summary[PASS], summary[FAIL], summary[WARN], summary[SKIP])
    log.info("报告 -> %s", out_path)
    if summary[FAIL] > 0:
        for r in chk.failures:
            log.error("FAIL %s/%s: %s | %s", r.section, r.check_id, r.description, r.detail)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
