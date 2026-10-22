"""核对 A 的 release_2023-01 解压结果（成员 B 只读小样本）。"""
from pathlib import Path

import pandas as pd

DAY = Path("data/processed/sample_day.parquet")
CEN = Path("data/raw/taxi_zone_centroids.csv")

ok = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if extra else ""))
    ok = ok and bool(cond)


# ---------------- sample_day ----------------
print("=" * 60)
print("1) sample_day.parquet")
print("=" * 60)
d = pd.read_parquet(DAY, columns=["timestamp", "zone_id", "demand"])
d["timestamp"] = pd.to_datetime(d["timestamp"])
print("shape      :", d.shape)
print("time range :", d.timestamp.min(), "~", d.timestamp.max())
print("dtypes     :", d.dtypes.to_dict())

check("dtypes 正确（datetime / 整数 / 整数）",
      str(d.timestamp.dtype).startswith("datetime64") and d.zone_id.dtype.kind in "iu"
      and d.demand.dtype.kind in "iu",
      f"{d.timestamp.dtype}, {d.zone_id.dtype}, {d.demand.dtype}")
check("行数 = 96 x 263 = 25248", len(d) == 25248, str(len(d)))

ts = d.timestamp.drop_duplicates().sort_values()
check("时间片数 = 96", len(ts) == 96, str(len(ts)))
check("区域数 = 263", d.zone_id.nunique() == 263, str(d.zone_id.nunique()))
check("区域覆盖 1..263", set(d.zone_id.unique()) == set(range(1, 264)))
check("(timestamp, zone_id) 唯一", not d.duplicated(["timestamp", "zone_id"]).any())
steps = set(ts.diff().dropna().unique())
check("时间步长全是 15 分钟", steps == {pd.Timedelta("15min")}, str(steps))
check("时间连续无缺口", len(ts) == 96 and (ts.max() - ts.min()) == pd.Timedelta("23h45m"))
check("demand 非负", d.demand.min() >= 0, f"min={d.demand.min()}")

total = int(d.demand.sum())
print(f"\n自然日需求基数 R = {total:,} 单/天")
print("zero_ratio = %.2f%%  mean = %.2f  p99 = %.1f  max = %d"
      % ((d.demand == 0).mean() * 100, d.demand.mean(),
         d.demand.quantile(0.99), d.demand.max()))

per_slice = d.groupby("timestamp")["demand"].sum()
print("每 15 分钟全网需求：min=%d / 中位=%d / 峰值=%d（%s）"
      % (per_slice.min(), per_slice.median(), per_slice.max(), per_slice.idxmax()))

# ---------------- 质心 ----------------
print("\n" + "=" * 60)
print("2) taxi_zone_centroids.csv")
print("=" * 60)
c = pd.read_csv(CEN)
cols = {x.lower(): x for x in c.columns}
print("shape  :", c.shape)
print("columns:", list(c.columns))

idc = cols.get("locationid") or cols.get("location_id") or cols.get("zone_id")
lat_c, lon_c = cols.get("lat"), cols.get("lon") or cols.get("lng")
x_c, y_c = cols.get("x"), cols.get("y")

check("263 行", len(c) == 263, str(len(c)))
check("有区域 ID 列", idc is not None, str(idc))
check("有 lat/lon 或 X/Y", bool((lat_c and lon_c) or (x_c and y_c)))
if idc:
    ids = set(c[idc].astype(int))
    check("覆盖 1..263", ids == set(range(1, 264)),
          f"缺 {sorted(set(range(1, 264)) - ids)[:10]}")
if lat_c and lon_c:
    check("经纬度在 NYC 范围内",
          c[lat_c].between(40.4, 41.0).all() and c[lon_c].between(-74.3, -73.6).all(),
          f"lat {c[lat_c].min():.3f}~{c[lat_c].max():.3f}, "
          f"lon {c[lon_c].min():.3f}~{c[lon_c].max():.3f}")

print("\n" + "=" * 60)
print("结论:", "全部 PASS，可以进入 B1 造 fixture" if ok else "有 FAIL，先别往下走")
print("=" * 60)