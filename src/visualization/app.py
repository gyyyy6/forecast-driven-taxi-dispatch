"""Streamlit 看板：需求热点 + 预测趋势 + 调度仿真 + 场景对比。

运行:
    streamlit run src/visualization/app.py

兼容性:
    plotly >= 5.24 用 scatter_map + map_style；更旧的版本回退 scatter_mapbox。
    streamlit 1.49+ 用 width="stretch"；更旧的版本回退 use_container_width。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许 `streamlit run src/visualization/app.py` 直接导入 src 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import load_config, resolve

st.set_page_config(page_title="网约车需求预测与调度", page_icon="🚕", layout="wide")


# ---------------------------------------------------------------- 兼容层
def scatter_map(data_frame=None, **kwargs):
    """plotly >=5.24 用 scatter_map，旧版回退 scatter_mapbox。"""
    if hasattr(px, "scatter_map"):
        return px.scatter_map(
            data_frame=data_frame, map_style="carto-positron", **kwargs
        )
    return px.scatter_mapbox(
        data_frame=data_frame, mapbox_style="carto-positron", **kwargs
    )


def show_fig(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- 读取
@st.cache_data(show_spinner=False)
def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def read_coords(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    low = {c.lower(): c for c in df.columns}
    id_col = low.get("locationid") or low.get("location_id") or low.get("zone_id")
    lat_col, lon_col = low.get("lat"), low.get("lon") or low.get("lng")
    if id_col is None or lat_col is None or lon_col is None:
        return None
    return df.rename(columns={id_col: "zone_id", lat_col: "lat", lon_col: "lon"})[
        ["zone_id", "lat", "lon"]
    ]


def candidate_predictions(cfg: dict) -> list[Path]:
    """正式预测优先；没有就退回本地 fixture。"""
    out: list[Path] = []
    official = resolve(cfg["visualization"]["prediction_path"])
    if official.exists():
        out.append(official)
    sim_dir = resolve(cfg["paths"]["simulation_dir"])
    if sim_dir.exists():
        out += sorted(sim_dir.glob("fixture_*.parquet"))
    return out


def main() -> None:
    cfg = load_config()

    scen_summary = resolve(
        cfg.get("visualization", {}).get(
            "simulation_path",
            "data/processed/simulation/scenarios/scenarios_summary.parquet",
        )
    )
    scen_dir = scen_summary.parent
    coords_path = resolve(cfg["paths"]["zone_centroids"])

    st.title("🚕 网约车需求热点预测与轻量级闭环调度")
    st.caption("NYC TLC HVFHV · 15 分钟 × 263 Taxi Zone · SimPy 调度仿真")

    preds = candidate_predictions(cfg)
    with st.sidebar:
        st.header("数据源")
        if preds:
            pred_path = st.selectbox("预测文件", preds, format_func=lambda p: p.name)
            st.code(str(pred_path), language=None)
        else:
            pred_path = None
            st.error("未找到预测文件")
        st.code(str(scen_summary), language=None)
        if st.button("刷新缓存"):
            st.cache_data.clear()

    if pred_path is None:
        st.warning(
            "尚未生成任何预测文件。先用 fixture 跑通流程：\n\n"
            "`python -m src.simulation.make_fixture --config configs/default.yaml`"
        )
        return

    pred = read_parquet(str(pred_path))
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    coords = read_coords(str(coords_path))

    scale_tag = pred["model_name"].iloc[0] if "model_name" in pred.columns else "unknown"
    st.caption(f"数据口径：`{pred_path.name}` → {scale_tag}")

    # ---------------- 时间选择 ----------------
    tsl = pred["timestamp"].drop_duplicates().sort_values()
    if len(tsl) > 1:
        picked = st.slider(
            "选择时间片",
            min_value=tsl.min().to_pydatetime(),
            max_value=tsl.max().to_pydatetime(),
            value=tsl.min().to_pydatetime(),
            step=pd.Timedelta(minutes=15).to_pytimedelta(),
        )
    else:
        picked = tsl.iloc[0].to_pydatetime()

    snapshot = pred[pred["timestamp"] == pd.Timestamp(picked)].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("预测总需求", f"{snapshot['predicted_demand'].sum():,.0f}")
    c2.metric("活跃区域数", f"{(snapshot['predicted_demand'] > 0).sum():,}")
    c3.metric("单区峰值", f"{snapshot['predicted_demand'].max():,.1f}")
    c4.metric("覆盖时间片", f"{pred['timestamp'].nunique():,}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ 需求热点", "📈 时间趋势", "🚦 调度仿真", "🧪 场景对比"]
    )

    # ---------------- Tab1 需求热点 ----------------
    with tab1:
        if coords is not None:
            plot_df = snapshot.merge(coords, on="zone_id", how="left").dropna(
                subset=["lat", "lon"]
            )
            fig = scatter_map(
                plot_df,
                lat="lat",
                lon="lon",
                size="predicted_demand",
                color="predicted_demand",
                hover_name="zone_id",
                labels={"predicted_demand": "预测需求（单/15分钟）"},
                color_continuous_scale="Turbo",
                size_max=40,
                zoom=9.5,
                center={"lat": 40.73, "lon": -73.94},
                title=f"{picked} 需求热点",
            )
            fig.update_layout(height=560, margin=dict(l=0, r=0, t=40, b=0))
            show_fig(fig)
        else:
            top = snapshot.nlargest(25, "predicted_demand")
            fig = px.bar(
                top,
                x="zone_id",
                y="predicted_demand",
                color="predicted_demand",
                labels={"zone_id": "区域 ID", "predicted_demand": "预测需求（单/15分钟）"},
                color_continuous_scale="Turbo",
                title=f"{picked} Top 25 区域",
            )
            fig.update_layout(height=520)
            show_fig(fig)
            st.info("未找到含 lat/lon 的质心文件，暂以柱状图展示。")

    # ---------------- Tab2 时间趋势 ----------------
    with tab2:
        by_ts = pred.groupby("timestamp", as_index=False)["predicted_demand"].sum()
        fig = px.line(
            by_ts,
            x="timestamp",
            y="predicted_demand",
            labels={"timestamp": "时间", "predicted_demand": "全网预测需求（单/15分钟）"},
            title="全网预测需求（15 分钟粒度）",
        )
        fig.update_layout(height=460)
        show_fig(fig)

    # ---------------- Tab3 调度仿真 ----------------
    with tab3:
        if not scen_summary.exists():
            st.info(
                "尚未生成调度结果。运行：\n\n"
                '`python -m src.simulation.scenarios --start "2023-01-01 00:00" --end "2023-01-02 00:00"`'
            )
        else:
            summ = read_parquet(str(scen_summary))
            summ["max_move_ratio"] = summ["max_move_ratio"].fillna(-1.0)

            combos = (
                summ.groupby(["n_vehicles", "max_wait_min"])["policy"]
                .nunique()
                .reset_index()
            )
            combos = combos[combos["policy"] >= 2]
            if combos.empty:
                st.warning("场景结果里没有同时包含 greedy 与 forecast 的工况。")
            else:
                opts = [
                    f"{int(r.n_vehicles)} 辆 / 等待上限 {int(r.max_wait_min)} 分钟"
                    for r in combos.itertuples()
                ]
                pick = st.radio("选择工况", opts, horizontal=True)
                idx = opts.index(pick)
                n_sel = combos.iloc[idx]["n_vehicles"]
                w_sel = combos.iloc[idx]["max_wait_min"]

                sub = summ[(summ.n_vehicles == n_sel) & (summ.max_wait_min == w_sel)]
                show_df(
                    sub[
                        [
                            "scenario_id",
                            "policy",
                            "served_rate",
                            "avg_wait_min",
                            "p90_wait_min",
                            "deadhead_km",
                        ]
                    ]
                )

                fig = px.bar(
                    sub,
                    x="policy",
                    y="served_rate",
                    color="policy",
                    labels={"policy": "策略", "served_rate": "服务率"},
                    text_auto=".2%",
                    title="服务率",
                )
                fig.update_yaxes(tickformat=".0%")
                fig.update_layout(height=340, showlegend=False)
                show_fig(fig)

                frames = []
                for sid in sub["scenario_id"]:
                    p = scen_dir / f"scenario_{sid}_requests.parquet"
                    if p.exists():
                        r = read_parquet(str(p))
                        frames.append(r[r["served"] == 1][["policy", "wait_min"]])
                if frames:
                    req = pd.concat(frames, ignore_index=True)
                    fig2 = px.box(
                        req,
                        x="policy",
                        y="wait_min",
                        color="policy",
                        labels={"policy": "策略", "wait_min": "接驾等待（分钟）"},
                        title="接驾等待时间分布（已服务订单）",
                    )
                    fig2.update_layout(height=400, showlegend=False)
                    show_fig(fig2)

                fig3 = px.bar(
                    sub,
                    x="policy",
                    y="deadhead_km",
                    color="policy",
                    labels={"policy": "策略", "deadhead_km": "空驶里程（km）"},
                    text_auto=",.0f",
                    title="空驶里程",
                )
                fig3.update_layout(height=340, showlegend=False)
                show_fig(fig3)

    # ---------------- Tab4 场景对比 ----------------
    with tab4:
        if not scen_summary.exists():
            st.info("场景汇总不存在，先运行 `python -m src.simulation.scenarios`。")
        else:
            summ = read_parquet(str(scen_summary))
            tmp = summ.copy()
            tmp["max_move_ratio"] = tmp["max_move_ratio"].fillna(0.0)

            g = tmp[tmp.policy == "greedy"]
            fig = px.line(
                g,
                x="n_vehicles",
                y="served_rate",
                color="max_wait_min",
                labels={
                    "n_vehicles": "车辆数",
                    "served_rate": "服务率",
                    "max_wait_min": "等待上限（分钟）",
                },
                markers=True,
                title="Greedy：服务率 vs 车队规模（按匹配半径）",
            )
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=420)
            show_fig(fig)

            base = g.set_index(["n_vehicles", "max_wait_min"])["served_rate"]
            fc = tmp[tmp.policy == "forecast"].copy()
            if not fc.empty and not base.empty:
                fc["greedy_rate"] = [
                    base.get((n, w), float("nan"))
                    for n, w in zip(fc.n_vehicles, fc.max_wait_min)
                ]
                fc["diff_pp"] = (fc.served_rate - fc.greedy_rate) * 100
                fc["配置"] = (
                    fc["n_vehicles"].astype(int).astype(str)
                    + " 辆 / w="
                    + fc["max_wait_min"].astype(int).astype(str)
                    + " / r="
                    + fc["max_move_ratio"].round(2).astype(str)
                )
                fig2 = px.bar(
                    fc,
                    x="配置",
                    y="diff_pp",
                    color="diff_pp",
                    color_continuous_scale="RdYlGn",
                    labels={"配置": "场景配置", "diff_pp": "服务率差（百分点）"},
                    title="Forecast 相对 Greedy 的服务率差（百分点）",
                )
                fig2.update_layout(height=420, showlegend=False)
                show_fig(fig2)

                fr = fc[(fc.n_vehicles == 200) & (fc.max_wait_min == 30)].sort_values(
                    "max_move_ratio"
                )
                if len(fr):
                    fig3 = px.line(
                        fr,
                        x="max_move_ratio",
                        y="served_rate",
                        markers=True,
                        labels={
                            "max_move_ratio": "预调度比例",
                            "served_rate": "服务率",
                        },
                        title="预调度比例 vs 服务率（200 辆 / 上限 30 分钟）",
                    )
                    fig3.update_yaxes(tickformat=".0%")
                    fig3.update_layout(height=400)
                    show_fig(fig3)

                show_df(
                    fc[
                        [
                            "scenario_id",
                            "配置",
                            "served_rate",
                            "greedy_rate",
                            "diff_pp",
                            "avg_wait_min",
                            "p90_wait_min",
                            "deadhead_km",
                        ]
                    ]
                )


if __name__ == "__main__":
    main()