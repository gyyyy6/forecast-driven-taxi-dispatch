"""Streamlit 看板：需求预测 + 调度仿真结果。

运行:
    streamlit run src/visualization/app.py
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
    return df.rename(columns={id_col: "zone_id", lat_col: "lat", lon_col: "lon"})[["zone_id", "lat", "lon"]]


def main() -> None:
    cfg = load_config()
    st.title("🚕 网约车需求热点预测与轻量级闭环调度")
    st.caption("NYC TLC HVFHV · 15 分钟 × Taxi Zone · 预测驱动调度仿真")

    pred_path = resolve(cfg["visualization"]["prediction_path"])
    sim_path = resolve(cfg["visualization"]["simulation_path"])
    coords_path = resolve(cfg["paths"]["zone_centroids"])

    with st.sidebar:
        st.header("数据源")
        st.code(str(pred_path), language=None)
        st.code(str(sim_path), language=None)
        if st.button("刷新缓存"):
            st.cache_data.clear()

    if not pred_path.exists():
        st.warning("尚未生成预测结果。请先运行 `python -m src.models.train_lgbm` 与 `python -m src.models.predict`。")
        return

    pred = read_parquet(str(pred_path))
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    coords = read_coords(str(coords_path))

    ts_min, ts_max = pred["timestamp"].min(), pred["timestamp"].max()
    tsl = pred["timestamp"].drop_duplicates().sort_values()
    picked = st.slider(
        "选择时间片",
        min_value=ts_min.to_pydatetime(),
        max_value=ts_max.to_pydatetime(),
        value=ts_min.to_pydatetime(),
        step=pd.Timedelta(minutes=15).to_pytimedelta(),
    )
    snapshot = pred[pred["timestamp"] == pd.Timestamp(picked)].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("预测总需求", f"{snapshot['predicted_demand'].sum():,.0f}")
    c2.metric("活跃区域数", f"{(snapshot['predicted_demand'] > 0).sum():,}")
    c3.metric("单区峰值", f"{snapshot['predicted_demand'].max():,.1f}")
    c4.metric("覆盖时间片", f"{pred['timestamp'].nunique():,}")

    tab1, tab2, tab3 = st.tabs(["🗺️ 需求热点", "📈 时间趋势", "🚦 调度仿真"])

    with tab1:
        if coords is not None:
            plot_df = snapshot.merge(coords, on="zone_id", how="left")
            fig = px.scatter_mapbox(
                plot_df,
                lat="lat",
                lon="lon",
                size="predicted_demand",
                color="predicted_demand",
                hover_name="zone_id",
                color_continuous_scale="Turbo",
                size_max=40,
                zoom=9.5,
                center={"lat": 40.73, "lon": -73.94},
                mapbox_style="carto-positron",
                title=f"{picked} 需求热点",
            )
            fig.update_layout(height=560, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            top = snapshot.nlargest(25, "predicted_demand")
            fig = px.bar(top, x="zone_id", y="predicted_demand", color="predicted_demand",
                         color_continuous_scale="Turbo", title=f"{picked} Top 25 区域")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
            st.info("未找到含 lat/lon 的质心文件，暂以柱状图展示；补充经纬度后自动切换地图。")

    with tab2:
        by_ts = pred.groupby("timestamp", as_index=False)["predicted_demand"].sum()
        fig = px.line(by_ts, x="timestamp", y="predicted_demand", title="全网预测需求（15 分钟）")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if not sim_path.exists():
            st.info("尚未运行仿真。请执行 `python -m src.simulation.env --policy both`。")
        else:
            summary = read_parquet(str(sim_path))
            st.dataframe(summary, use_container_width=True, hide_index=True)
            req_path = sim_path.parent / "sim_requests.parquet"
            if req_path.exists():
                req = read_parquet(str(req_path))
                fig = px.box(req[req["served"] == 1], x="policy", y="wait_min",
                             color="policy", title="接驾等待时间分布（已服务订单）")
                fig.update_layout(height=420, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                if "served" in req.columns:
                    rate = req.groupby("policy", as_index=False)["served"].mean()
                    rate["served_rate"] = rate["served"] * 100
                    fig2 = px.bar(rate, x="policy", y="served_rate", color="policy",
                                  title="服务率（%）", text_auto=".1f")
                    fig2.update_layout(height=360, showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()