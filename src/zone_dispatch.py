"""康炜奕（任务 3）：结合需求预测与无桩电单车热点，输出分区预投放配额与跨区调度路线。

输入三份数据：

1. ``data/processed/station_trips.csv``：站点有效集，用于计算各区域分时段净流量；
2. ``data/derived/core_station_predictions.csv``：陈宇涵的站点级逐小时预测
   （``historical_mean`` 模型，2024-01-22 至 01-28 的 test 窗口，含 actual 列），
   取自 ``feat/temporal-and-modeling`` 分支 ``reports/tables/station_weekly_predictions.csv``，
   按冻结推荐记录自动导出核心站点的逐小时预测与实测；
3. ``data/processed/clean_trips.csv``：全量有效集，用于提取无桩电单车（缺站点 ID 的电单车）
   的空间热点。

本模块是整月历史流量叠加测试周模型偏差的回顾性情景；修正量用到测试周实测，
不能作为预测起点之前可用的未来调度决策，也不参与成员 B 的预测评估。
净流量 ``F = 出发量 − 到达量``，正值表示净流出压力、负值表示净流入压力；分时段统计中到达量按
还车时间归属（与第 8 节一致）。区域按站点坐标做加权 k-means 划分为 7 个片区，再按方位命名。
预投放配额 = 预测净流量 + 无桩热点净需求，再按 15% 安全库存上浮；跨区路线在同一时段内
按“最近邻优先”贪心匹配，量纲为辆/日。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .data_cleaning import read_clean_data
from .spatial_analysis import build_station_summary, load_station_trips, shorten
from .utils import (
    FIGURE_DIR, PROCESSED_DIR, PROJECT_ROOT, SECTION_DIR, TABLE_DIR, save_figure_html, set_plot_style,
)

DATA_DIR = PROJECT_ROOT / "data"
CLEAN_TRIPS_FILE = PROCESSED_DIR / "clean_trips.csv"
CORE_PREDICTION_FILE = DATA_DIR / "derived" / "core_station_predictions.csv"
STATION_PREDICTION_FILE = TABLE_DIR / "station_weekly_predictions.csv"
ZONE_PROFILE_FILE = TABLE_DIR / "zone_profile.csv"
ZONE_QUOTA_FILE = TABLE_DIR / "zone_preallocation_quota.csv"
ROUTE_FILE = TABLE_DIR / "zone_dispatch_routes.csv"
HOTSPOT_FILE = TABLE_DIR / "dockless_hotspots.csv"
SECTION_FILE = SECTION_DIR / "03_spatial_and_dispatch.md"

SEGMENTS = ("早高峰", "日间平峰", "晚高峰", "夜间")
SEGMENT_LABELS = {
    "早高峰": "早高峰 7:00–9:59",
    "日间平峰": "日间平峰 10:00–16:59",
    "晚高峰": "晚高峰 17:00–19:59",
    "夜间": "夜间 20:00–6:59",
}
SAFETY_STOCK = 0.15
ZONE_COUNT = 7
ZONE_SEED = 7
GRID_DEG = 0.0025
HOTSPOT_TOP_N = 15
HOTSPOT_MAP_N = 10
ROUTE_COLUMNS = ["segment", "source_zone", "target_zone", "volume_per_day",
                 "distance_km", "target_quota", "share_of_target"]

ZONE_COLORS = ["#315D8A", "#159A85", "#D78538", "#AE5F67", "#7A6CA8", "#5B8C5A", "#B58B5A", "#8093A3"]
SINK_COLOR = "#AE5F67"
SOURCE_COLOR = "#2F6E8F"
ROUTE_COLOR = "#B03A2E"
HOTSPOT_COLOR = "#E08A3C"


def segment_of(hours: pd.Series) -> pd.Series:
    """把 0–23 时映射到四个时段；夜间跨零点，单独判定。"""
    values = pd.Series("夜间", index=hours.index, dtype="object")
    values = values.mask(hours.between(7, 9), "早高峰")
    values = values.mask(hours.between(10, 16), "日间平峰")
    values = values.mask(hours.between(17, 19), "晚高峰")
    return values


def weekday_days(trips: pd.DataFrame) -> int:
    """工作日天数（周一至周五，含 1/1 与 1/15），与第 1、8 节口径一致。"""
    weekdays = trips.loc[trips["weekday"].lt(5), "started_at"]
    return int(pd.to_datetime(weekdays, errors="coerce").dt.normalize().nunique())


def build_zones(summary: pd.DataFrame, k: int = ZONE_COUNT, seed: int = ZONE_SEED
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按站点坐标做加权 k-means（权重为站点总流量），返回带 zone 的站点表与区域汇总表。"""
    features = summary[["lat", "lng"]].to_numpy(dtype=float)
    weights = summary["total_trips"].to_numpy(dtype=float)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=30).fit_predict(features, sample_weight=weights)
    stations = summary.assign(zone_id=labels)
    center_lat = float(np.average(stations["lat"], weights=weights))
    center_lng = float(np.average(stations["lng"], weights=weights))
    zones = (
        stations.groupby("zone_id")
        .agg(stations=("station_id", "size"), lat=("lat", "mean"), lng=("lng", "mean"),
             trips=("total_trips", "sum"))
        .reset_index()
    )
    zones["zone"] = [zone_name(row.lat, row.lng, center_lat, center_lng) for row in zones.itertuples()]
    zones["trips_share"] = zones["trips"] / zones["trips"].sum()
    stations["zone"] = stations["zone_id"].map(zones.set_index("zone_id")["zone"])
    return stations, zones.sort_values("trips", ascending=False).reset_index(drop=True)


def zone_name(lat: float, lng: float, center_lat: float, center_lng: float) -> str:
    """按区域重心相对城市中心的方位命名，便于报告叙述。"""
    north = lat - center_lat
    east = (lng - center_lng) * np.cos(np.radians(center_lat))
    distance = float(np.hypot(north, east))
    if distance < 0.012:
        return "中心商务区"
    angle = (np.degrees(np.arctan2(east, north)) + 360) % 360
    sectors = ((22.5, "北部"), (67.5, "东北部"), (112.5, "东部"), (157.5, "东南部"),
               (202.5, "南部"), (247.5, "西南部"), (292.5, "西部"), (337.5, "西北部"),
               (360.0, "北部"))
    label = next(name for limit, name in sectors if angle < limit)
    return f"{label}（外围）" if distance > 0.10 else label


def assign_nearest_zone(points: pd.DataFrame, stations: pd.DataFrame, lat_col: str = "lat",
                        lng_col: str = "lng") -> pd.Series:
    """按最近区域重心给点赋区域。"""
    centers = stations.groupby("zone")[["lat", "lng"]].mean()
    lat = points[lat_col].to_numpy(dtype=float)[:, None]
    lng = points[lng_col].to_numpy(dtype=float)[:, None]
    dlat = lat - centers["lat"].to_numpy()[None, :]
    dlng = (lng - centers["lng"].to_numpy()[None, :]) * np.cos(np.radians(lat))
    distance = np.hypot(dlat, dlng)
    return pd.Series(centers.index[distance.argmin(axis=1)], index=points.index)


def zone_segment_flows(station_trips: pd.DataFrame, stations: pd.DataFrame, days: int) -> pd.DataFrame:
    """各区域分时段的日均出发、到达与净流量（辆/日）；到达量按还车时间归属。"""
    data = station_trips.copy()
    returned = pd.to_datetime(data["ended_at"], errors="coerce")
    data["end_hour"] = returned.dt.hour
    data["end_weekday"] = returned.dt.dayofweek

    start = data.loc[data["weekday"].lt(5)].copy()
    start["segment"] = segment_of(start["hour"])
    departures = start.groupby(["start_station_id", "segment"]).size().rename("departures")

    end = data.loc[data["end_weekday"].lt(5)].copy()
    end["segment"] = segment_of(end["end_hour"])
    arrivals = end.groupby(["end_station_id", "segment"]).size().rename("arrivals")

    flows = pd.concat([departures, arrivals], axis=1).fillna(0)
    flows.index.names = ["station_id", "segment"]
    flows = flows.reset_index()
    flows["station_id"] = flows["station_id"].astype("string")
    flows = flows.merge(stations[["station_id", "zone"]], on="station_id", how="left")
    grouped = flows.groupby(["zone", "segment"]).agg(
        departures=("departures", "sum"), arrivals=("arrivals", "sum")
    ).reset_index()
    grouped[["departures", "arrivals"]] = grouped[["departures", "arrivals"]] / days
    grouped["observed_net"] = grouped["departures"] - grouped["arrivals"]
    return grouped


def export_core_predictions() -> Path:
    """从统一的预测表导出冻结推荐模型的测试周子集，避免维护两份不同精度的输入。"""
    selection = json.loads((TABLE_DIR / "station_model_selection.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(STATION_PREDICTION_FILE, dtype={"station_id": "string"})
    columns = ["station_id", "timestamp", "target", "actual", "prediction", "is_peak"]
    subset = raw.loc[raw["phase"].eq("test") & raw["model"].eq(selection["recommended_model"]), columns]
    if subset.empty or subset.duplicated(["station_id", "timestamp", "target"]).any():
        raise ValueError("核心站点预测缺失或键重复，请先运行 src.station_model。")
    CORE_PREDICTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(CORE_PREDICTION_FILE, index=False, encoding="utf-8-sig", float_format="%.10f")
    return CORE_PREDICTION_FILE


def load_core_predictions(path: Path = CORE_PREDICTION_FILE) -> pd.DataFrame:
    """读取成员 B 的核心站点逐小时预测，整理为 (站点, 时段) 的日均值。"""
    raw = pd.read_csv(path)
    raw["station_id"] = raw["station_id"].astype("string")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["segment"] = segment_of(raw["timestamp"].dt.hour)
    raw["weekday"] = raw["timestamp"].dt.dayofweek
    weekday = raw.loc[raw["weekday"].lt(5)]
    days = int(weekday["timestamp"].dt.normalize().nunique())
    grouped = weekday.groupby(["station_id", "segment", "target"]).agg(
        actual=("actual", "sum"), prediction=("prediction", "sum")
    ).reset_index()
    grouped[["actual", "prediction"]] = grouped[["actual", "prediction"]] / days
    table = grouped.pivot_table(index=["station_id", "segment"], columns="target",
                                values=["actual", "prediction"], aggfunc="sum")
    table.columns = [f"{level}_{target}" for level, target in table.columns]
    table = table.reset_index().fillna(0.0)
    table["days"] = days
    return table


def prediction_correction(core: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """把核心站点的“预测 − 实测”差额汇总到区域 × 时段。"""
    merged = core.merge(stations[["station_id", "zone"]], on="station_id", how="left")
    merged["delta_net"] = merged["prediction_net_outflow"] - merged["actual_net_outflow"]
    return merged.groupby(["zone", "segment"]).agg(
        delta_net=("delta_net", "sum"), core_stations=("station_id", "nunique")
    ).reset_index()


def dockless_trips(clean_trips: pd.DataFrame) -> pd.DataFrame:
    """保留主城出发的缺站点电单车队列；工作日分别按借出、归还事件的时间筛选。"""
    data = clean_trips.loc[clean_trips["is_dockless_ebike"].fillna(False)].copy()
    inside = (data["start_lat"].between(38.79, 39.00) & data["start_lng"].between(-77.13, -76.89))
    return data.loc[inside]


def dockless_hotspots(dockless: pd.DataFrame, stations: pd.DataFrame, days: int,
                      grid: float = GRID_DEG, top_n: int = HOTSPOT_TOP_N) -> pd.DataFrame:
    """无桩电单车的方格热点；net = 还车 − 取车，正值为可回收车辆。"""
    start = dockless.loc[pd.to_datetime(dockless["started_at"]).dt.dayofweek.lt(5)]
    end = dockless.loc[pd.to_datetime(dockless["ended_at"]).dt.dayofweek.lt(5)]
    origin = pd.DataFrame({
        "cell_lat": (start["start_lat"] / grid).round().astype(int),
        "cell_lng": (start["start_lng"] / grid).round().astype(int),
    }).groupby(["cell_lat", "cell_lng"]).size().rename("departures")
    destination = pd.DataFrame({
        "cell_lat": (end["end_lat"] / grid).round().astype(int),
        "cell_lng": (end["end_lng"] / grid).round().astype(int),
    }).groupby(["cell_lat", "cell_lng"]).size().rename("arrivals")
    cells = pd.concat([origin, destination], axis=1).fillna(0).reset_index()
    cells[["departures", "arrivals"]] = cells[["departures", "arrivals"]] / days
    cells["activity"] = cells["departures"] + cells["arrivals"]
    cells["net"] = cells["arrivals"] - cells["departures"]
    cells["lat"] = cells["cell_lat"] * grid
    cells["lng"] = cells["cell_lng"] * grid
    cells = cells.nlargest(top_n, "activity").reset_index(drop=True)
    cells["hotspot_id"] = [f"H{index + 1:02d}" for index in range(len(cells))]
    cells["zone"] = assign_nearest_zone(cells, stations).to_numpy()
    return cells[["hotspot_id", "lat", "lng", "zone", "departures", "arrivals", "activity", "net"]]


def zone_hotspot_balance(dockless: pd.DataFrame, stations: pd.DataFrame, days: int) -> pd.DataFrame:
    """按区域 × 时段汇总无桩电单车的日均取车、还车与净流入。"""
    data = dockless.copy()
    started = pd.to_datetime(data["started_at"])
    returned = pd.to_datetime(data["ended_at"])
    data["start_segment"] = segment_of(started.dt.hour)
    data["end_segment"] = segment_of(returned.dt.hour)
    data["zone_start"] = assign_nearest_zone(data, stations, "start_lat", "start_lng").to_numpy()
    data["zone_end"] = assign_nearest_zone(data, stations, "end_lat", "end_lng").to_numpy()
    starts = data.loc[started.dt.dayofweek.lt(5)].groupby(["zone_start", "start_segment"]).size().rename("departures").reset_index()
    ends = data.loc[returned.dt.dayofweek.lt(5)].groupby(["zone_end", "end_segment"]).size().rename("arrivals").reset_index()
    starts = starts.rename(columns={"zone_start": "zone", "start_segment": "segment"})
    ends = ends.rename(columns={"zone_end": "zone", "end_segment": "segment"})
    merged = starts.merge(ends, on=["zone", "segment"], how="outer").fillna(0)
    merged[["departures", "arrivals"]] = merged[["departures", "arrivals"]] / days
    merged["hotspot_net"] = merged["arrivals"] - merged["departures"]
    return merged


def build_quota(flows: pd.DataFrame, correction: pd.DataFrame, hotspots: pd.DataFrame,
                zones: pd.DataFrame) -> pd.DataFrame:
    """区域 × 时段的预测缺口、预投放配额与可调出量。"""
    base = flows.merge(correction, on=["zone", "segment"], how="left").fillna(
        {"delta_net": 0.0, "core_stations": 0}
    )
    base = base.merge(hotspots, on=["zone", "segment"], how="left").fillna(
        {"departures": 0.0, "arrivals": 0.0, "hotspot_net": 0.0}
    )
    base["predicted_net"] = base["observed_net"] + base["delta_net"]
    # 站点净流量为正（缺车）时，无桩热点回收的车辆可以抵扣；热点自身缺车则加需求。
    base["gap"] = base["predicted_net"] - base["hotspot_net"]
    base["quota"] = np.where(base["gap"] > 0, np.ceil(base["gap"] * (1 + SAFETY_STOCK)), 0.0)
    # 需求向上取整、供给向下取整；不能靠舍入制造可调出车辆。
    base["available"] = np.floor((-base["gap"]).clip(lower=0) + 1e-9)
    base = base.merge(zones[["zone", "lat", "lng", "stations", "trips_share"]], on="zone", how="left")
    order = {name: index for index, name in enumerate(zones["zone"])}
    base["zone_order"] = base["zone"].map(order)
    base["segment_order"] = base["segment"].map({name: index for index, name in enumerate(SEGMENTS)})
    return base.sort_values(["segment_order", "zone_order"]).reset_index(drop=True)


def build_routes(quota: pd.DataFrame, max_distance_km: float = 25.0) -> pd.DataFrame:
    """同一时段内按最近邻贪心把可调出区域的车辆分配给需投放区域。"""
    rows = []
    for segment in SEGMENTS:
        block = quota.loc[quota["segment"].eq(segment)]
        targets = block.loc[block["quota"].gt(0), ["zone", "quota", "lat", "lng"]].copy()
        sources = block.loc[block["available"].gt(0), ["zone", "available", "lat", "lng"]].copy()
        remaining = dict(zip(sources["zone"], sources["available"]))
        for target in targets.sort_values("quota", ascending=False).itertuples():
            need = float(target.quota)
            while need >= 1.0:
                candidates = [zone for zone, volume in remaining.items() if volume >= 1.0]
                if not candidates:
                    break
                source_rows = sources.set_index("zone").loc[candidates]
                distances = _distance_km(target.lat, target.lng, source_rows["lat"], source_rows["lng"])
                distances = distances.loc[distances.le(max_distance_km)]
                if distances.empty:
                    break
                source = distances.idxmin()
                volume = min(need, remaining[source])
                rows.append({
                    "segment": segment,
                    "source_zone": source,
                    "target_zone": target.zone,
                    "volume_per_day": round(volume, 1),
                    "distance_km": round(float(distances.loc[source]), 2),
                    "target_quota": round(float(target.quota), 1),
                    "share_of_target": round(volume / float(target.quota), 3),
                })
                need -= volume
                remaining[source] -= volume
    plan = pd.DataFrame(rows, columns=ROUTE_COLUMNS)
    if plan.empty:
        return plan
    return plan.sort_values(["segment", "volume_per_day"], ascending=[True, False]).reset_index(drop=True)


def _distance_km(lat: float, lng: float, other_lat: pd.Series, other_lng: pd.Series) -> pd.Series:
    radius = 6371.0
    lat1, lng1 = np.radians(lat), np.radians(lng)
    lat2, lng2 = np.radians(other_lat.to_numpy()), np.radians(other_lng.to_numpy())
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return pd.Series(2 * radius * np.arctan2(np.sqrt(a), np.sqrt(1 - a)), index=other_lat.index)


def attach_coverage(quota: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """把调度路线的调运量回填到目标区域，计算缺口覆盖率与处置建议。"""
    delivered = (routes.groupby(["segment", "target_zone"])["volume_per_day"].sum()
                 .rename("delivered").reset_index())
    quota = quota.merge(delivered, left_on=["segment", "zone"],
                        right_on=["segment", "target_zone"], how="left")
    quota["delivered"] = quota["delivered"].fillna(0.0)
    quota["uncovered"] = (quota["quota"] - quota["delivered"]).clip(lower=0).round(1)
    quota["coverage"] = np.where(quota["quota"] > 0, quota["delivered"] / quota["quota"], np.nan)
    quota["remark"] = [
        "跨区调运已覆盖" if row.quota > 0 and row.delivered >= row.quota - 0.5
        else f"缺口 {row.uncovered:.0f} 辆/日，建议提前预投放" if row.quota > 0
        else "可调出" if row.available >= 1
        else "基本平衡"
        for row in quota.itertuples()
    ]
    return quota.drop(columns=["target_zone"])


def plot_route_map(quota: pd.DataFrame, routes: pd.DataFrame, stations: pd.DataFrame,
                   hotspots: pd.DataFrame) -> plt.Figure:
    """图13：四个时段的区域角色、跨区调度路线与无桩热点。"""
    figure = plt.figure(figsize=(15.2, 12.2))
    figure.text(0.045, 0.968, "图13  分时段跨区调度路线与配额（历史情景）",
                fontsize=21, weight="bold", va="top")
    figure.text(0.045, 0.936,
                "圆点大小＝预投放配额或可调出量（辆/日），红色为需投放区域、蓝色为可调出区域；"
                "箭头为跨区路线，菱形为无桩电单车热点",
                fontsize=10.6, color="#657585", va="top")

    positions = [(0.055, 0.520, 0.400, 0.365), (0.545, 0.520, 0.400, 0.365),
                 (0.055, 0.090, 0.400, 0.365), (0.545, 0.090, 0.400, 0.365)]
    map_hotspots = hotspots.head(HOTSPOT_MAP_N)
    # 展示全部片区重心，避免外围路线被固定的主城窗口裁掉。
    centers = quota.drop_duplicates("zone")
    x_limits = (float(centers.lng.min()) - 0.04, float(centers.lng.max()) + 0.04)
    y_limits = (float(centers.lat.min()) - 0.035, float(centers.lat.max()) + 0.035)

    for index, (segment, (left, bottom, width, height)) in enumerate(zip(SEGMENTS, positions)):
        axis = figure.add_axes([left, bottom, width, height])
        block = quota.loc[quota["segment"].eq(segment)]
        segment_routes = routes.loc[routes["segment"].eq(segment)]
        route_zones = set(segment_routes["source_zone"]) | set(segment_routes["target_zone"])
        axis.scatter(stations["lng"], stations["lat"], s=3.5, c="#C9D4DC", linewidths=0, zorder=1)
        axis.scatter(map_hotspots["lng"], map_hotspots["lat"],
                     s=16 + 80 * (map_hotspots["activity"] / hotspots["activity"].max()),
                     marker="D", c=HOTSPOT_COLOR, alpha=0.5, edgecolors="white",
                     linewidths=0.6, zorder=2)
        scale = max(float(block["quota"].max()), float(block["available"].max()), 1.0)
        for row in block.itertuples():
            value = max(row.quota, row.available)
            if value < 2 and row.zone not in route_zones:
                continue
            sink = row.quota > 0
            axis.scatter(row.lng, row.lat, s=40 + 900 * (value / scale),
                         c=SINK_COLOR if sink else SOURCE_COLOR, alpha=0.72,
                         edgecolors="white", linewidths=1.2, zorder=4)
            if row.zone == "中心商务区":
                offset, horizontal = (-52, -8), "right"
            elif row.zone == "西部":
                offset, horizontal = (0, -20), "center"
            elif 38.85 <= row.lat <= 38.99:
                offset, horizontal = (0, 16), "center"
            else:
                offset, horizontal = (0, -16), "center"
            axis.annotate(f"{row.zone}\n{'投放' if sink else '调出'} {value:.0f}",
                          xy=(row.lng, row.lat), xytext=offset, textcoords="offset points",
                          ha=horizontal, va="center" if horizontal == "right" else
                          ("bottom" if offset[1] > 0 else "top"),
                          fontsize=7.8, color="#203548", zorder=6,
                          bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                                "edgecolor": "#D3DDE5", "alpha": 0.9})
        lookup = block.set_index("zone")
        biggest = max(float(routes["volume_per_day"].max()) if not routes.empty else 1.0, 1.0)
        for route in routes.loc[routes["segment"].eq(segment)].itertuples():
            source = lookup.loc[route.source_zone]
            target = lookup.loc[route.target_zone]
            axis.annotate("", xy=(target.lng, target.lat), xytext=(source.lng, source.lat),
                          arrowprops={"arrowstyle": "-|>", "color": ROUTE_COLOR,
                                      "lw": 0.9 + 1.6 * route.volume_per_day / biggest,
                                      "alpha": 0.85, "mutation_scale": 13,
                                      "connectionstyle": "arc3,rad=0.16"}, zorder=5)
            dx = float(target.lng - source.lng)
            dy = float(target.lat - source.lat)
            length = float(np.hypot(dx, dy)) or 1.0
            mid_x = (source.lng + target.lng) / 2 - dy / length * 0.013
            mid_y = (source.lat + target.lat) / 2 + dx / length * 0.013
            axis.text(mid_x, mid_y, f"{route.volume_per_day:.0f} 辆/日", fontsize=7.0,
                      color=ROUTE_COLOR, ha="center", va="center", zorder=6,
                      bbox={"boxstyle": "round,pad=0.16", "facecolor": "white",
                            "edgecolor": "none", "alpha": 0.85})
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect(1 / np.cos(np.radians(38.90)))
        axis.ticklabel_format(useOffset=False, style="plain")
        axis.tick_params(labelsize=7.6)
        if index < 2:
            axis.tick_params(labelbottom=False)
        axis.grid(alpha=0.3, linestyle=":")
        axis.set_title(SEGMENT_LABELS[segment], loc="left", pad=8, fontsize=11)

    figure.text(0.045, 0.052,
                "分区为站点坐标的加权 k-means（k=7，权重＝站点月流量）；跨区路线只在同一时段内匹配，"
                "距离为区域重心间的球面直线距离；小于 2 辆/日且未参与路线的区域不标注。",
                fontsize=10)
    figure.text(0.045, 0.022,
                "缺站点电单车为代理标记；取车、还车按各自时刻统计；供给向下取整。"
                "热点仅显示活动量前 10 格；情景修正含测试周实测，不能作为未来调度决策。",
                fontsize=10, color="#657585")
    return figure


def plot_quota_heatmap(quota: pd.DataFrame, hotspots: pd.DataFrame) -> plt.Figure:
    """图14：分区预投放配额热力图与无桩热点排行。"""
    figure = plt.figure(figsize=(14.8, 8.8))
    figure.text(0.05, 0.955, "图14  分区配额与无桩电单车热点（历史情景）",
                fontsize=21, weight="bold", va="top")
    figure.text(0.05, 0.910,
                "矩阵数值为“净缺口”（辆/日）：正值需预投放，负值可调出；右侧为无桩电单车热点排行",
                fontsize=11, color="#657585", va="top")

    zones = list(dict.fromkeys(quota["zone"]))
    matrix = (quota.pivot_table(index="zone", columns="segment", values="gap", aggfunc="sum")
              .reindex(index=zones, columns=list(SEGMENTS)).fillna(0.0))
    axis = figure.add_axes([0.135, 0.195, 0.355, 0.655])
    limit = float(np.nanmax(np.abs(matrix.to_numpy())))
    image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(SEGMENTS)), labels=list(SEGMENTS), fontsize=9.4)
    axis.set_yticks(range(len(zones)), labels=zones, fontsize=9)
    axis.set_title("(a) 区域 × 时段净缺口（辆/日）", loc="left", pad=10, fontsize=11.5)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix.iloc[row, column])
            axis.text(column, row, f"{value:+.0f}", ha="center", va="center", fontsize=8.8,
                      color="white" if abs(value) > limit * 0.55 else "#203548")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.042, pad=0.02)
    colorbar.set_label("净缺口（辆/日）：正＝需预投放，负＝可调出", fontsize=8.6, labelpad=4)
    colorbar.ax.tick_params(labelsize=7.8)

    bar_axis = figure.add_axes([0.700, 0.195, 0.265, 0.655])
    top = hotspots.nlargest(10, "activity").iloc[::-1]
    colors = [SINK_COLOR if value > 0 else SOURCE_COLOR for value in top["net"]]
    bar_axis.barh(np.arange(len(top)), top["activity"], color=colors, alpha=0.85, height=0.62)
    bar_axis.set_yticks(np.arange(len(top)),
                        labels=[f"{row.hotspot_id} · {shorten(row.zone, 8)}" for row in top.itertuples()],
                        fontsize=8.4)
    bar_axis.set_xlim(0, top["activity"].max() * 1.32)
    bar_axis.set_xlabel("日均取车＋还车（次/日）")
    bar_axis.set_title("(b) 无桩电单车热点 Top10", loc="left", pad=12, fontsize=11.5)
    bar_axis.grid(axis="y", visible=False)
    for position, row in enumerate(top.itertuples()):
        bar_axis.text(row.activity + top["activity"].max() * 0.015, position,
                      f"{row.activity:.0f}（净 {row.net:+.0f}）", va="center", fontsize=7.8,
                      color="#203548")

    figure.text(0.05, 0.062,
                "净缺口 = 情景净流量 − 无桩热点净流入；情景净流量 = 分区观测净流量 + 核心站点的模型偏差修正，"
                "预投放配额在此基础上上浮 15% 安全库存。",
                fontsize=10)
    figure.text(0.05, 0.022,
                "热点颜色：红＝还车多于取车（车辆堆积，可回收），蓝＝取车多于还车（车辆流失，需回补）；"
                "取车、还车按各自工作日统计；情景修正含测试周实测，不是未来库存预测。",
                fontsize=10, color="#657585")
    return figure


def write_tables(zones: pd.DataFrame, quota: pd.DataFrame, routes: pd.DataFrame,
                 hotspots: pd.DataFrame) -> dict[str, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "zones": (zones[["zone", "stations", "trips_share", "lat", "lng"]].round(6), ZONE_PROFILE_FILE),
        "quota": (quota[["segment", "zone", "observed_net", "delta_net", "predicted_net", "hotspot_net",
                         "gap", "quota", "available", "delivered", "uncovered", "coverage",
                         "core_stations", "stations", "trips_share", "remark"]].round(3), ZONE_QUOTA_FILE),
        "routes": (routes.round(3), ROUTE_FILE),
        "hotspots": (hotspots.round(4), HOTSPOT_FILE),
    }
    paths = {}
    for name, (frame, path) in outputs.items():
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
    return paths


def write_scenario_report(zones: pd.DataFrame, quota: pd.DataFrame, routes: pd.DataFrame,
                          hotspots: pd.DataFrame, dockless: pd.DataFrame, days: int) -> None:
    """更新报告第 9 节，使配额、路线与重新计算的表格保持一致。"""
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |",
                          "| " + " | ".join(["---"] * len(headers)) + " |",
                          *["| " + " | ".join(map(str, row)) + " |" for row in rows]])

    totals = quota.groupby("segment")[["quota", "available", "delivered", "uncovered"]].sum().reindex(SEGMENTS)
    zone_table = table(["区域", "站点数", "月流量份额", "区域重心"],
                       [[r.zone, r.stations, f"{r.trips_share:.1%}", f"{r.lat:.3f}, {r.lng:.3f}"] for r in zones.itertuples()])
    quota_table = table(["时段", "配额", "可调出", "已匹配调运", "未覆盖"],
                        [[segment, *[f"{v:.0f}" for v in totals.loc[segment]]] for segment in SEGMENTS])
    route_table = table(["时段", "调出区域", "投放区域", "辆/日", "直线距离 km", "目标覆盖率"],
                        [[r.segment, r.source_zone, r.target_zone, f"{r.volume_per_day:.0f}",
                          f"{r.distance_km:.2f}", f"{r.share_of_target:.1%}"] for r in routes.itertuples()])
    section = f"""## 9 分时段跨区调度路线与预投放配额

### 9.1 数据、分区与情景边界

本节把 {int(zones.stations.sum())} 个站点按坐标与月流量权重划为 {len(zones)} 个片区。按早高峰 7:00–9:59、日间平峰 10:00–16:59、晚高峰 17:00–19:59、夜间 20:00–6:59 汇总，分母为 {days} 个工作日；这里的时段是历史数据分析窗口。

**这是回顾性情景测算。** 在站点整月日均净流量上，叠加陈宇涵冻结推荐模型在测试周（1 月 22–28 日）的“预测减实测”差额，再加入缺站点电单车的日均净流量。差额使用了测试周实测，事前不可获得；本节不参与预测模型的测试评分，也不能解释为预测起点之前可执行的未来调度方案。

预测子集由 `station_weekly_predictions.csv` 与 `station_model_selection.json` 自动导出到 `data/derived/core_station_predictions.csv`，保留原预测精度。陈宇涵按每个已知端点预测，空间模块使用双端站点完整队列，两者覆盖不同，因此此处只把 10 个核心站点的差额作为模型偏差敏感性修正，其余站点沿用历史流量，不能称为全城站点预测。

{zone_table}

### 9.2 缺站点电单车与还车时刻

无桩电单车沿用 `is_dockless_ebike` 代理标记，限定主城坐标窗口内出发的队列，共 {len(dockless):,} 条。主城窗口为纬度 38.79–39.00、经度 −77.13–−76.89；目的地随队列保留。借出按 `started_at`、归还按 `ended_at` 各自的工作日和时段统计，周日出发、周一还车的记录只计工作日归还，跨时段记录也不再错记为出发时段到达。队列按出发月份选取，缺少前一月出发订单，不能作为独立完整的运营事件台账。

热点按 0.0025° 网格的取还总活动量排序，保留前 {len(hotspots)} 格；其日均活动量合计 {hotspots.activity.sum():.1f} 次，净流入合计 {hotspots.net.sum():+.1f} 辆/日。配额计算使用全部入选队列的分区净流量，图中仅展示前 10 个热点，不能把前 15 格的净流入当作全区供给。

### 9.3 配额与供给约束

令 F 为站点日均净流出，Δ 为核心站点模型偏差修正，H 为缺站点电单车净流入（归还减借出）。情景缺口 G = F + Δ − H。正值只表示流出压力，不证明站点实际缺车。需求配额为 `ceil(max(G, 0) × 1.15)`，供给上限为 `floor(max(−G, 0))`：需求加入 15% 余量后向上取整，供给向下取整，避免把日均 1.1 辆当成可调出 2 辆。

**表 9-2 分时段情景配额（辆/日）**

{quota_table}

四个时段的配额合计 {quota.quota.sum():.0f} 辆/日、可调出合计 {quota.available.sum():.0f} 辆/日，路线合计匹配 {routes.volume_per_day.sum():.0f} 辆/日；这是各时段的流量求和，不能当作全日独立车辆数，同一辆车可能跨时段周转。

### 9.4 跨区候选路线

同一时段按最近邻匹配，并限制每个来源区域的累计调出量和每个目标区域的累计接收量。恰好 1 辆也参与匹配，没有供给或距离内没有来源时保留零覆盖结果。本次得到 {len(routes)} 条候选路线，区域重心间直线距离上限为 25 km。

{route_table}

### 9.5 解释与局限

净流量不是库存；缺少期初存量、容量、人员、车辆与路网通行时间，不能证明可调出车辆实际存在，也不能把未覆盖配额直接称作必须新增投放的数量。分时段差额同时受途中车辆、队列边界、区域间流动与模型偏差影响，未逐项分解前不将其归因于单一因素。有效行程时长不超过 180 分钟，跨时段事件不能单独解释平峰的全部缺口。

缺站点标记不能确认运营形态，格网活动位置也不是可回收车辆的实时停放位置。区域划分权重和普通站点流量使用整月历史，核心修正使用测试周实测，两者只适合本节的历史敏感性情景。图13–14、配额表与路线表用于比较区域方向和规模；真正的事前方案需以当时可获得的信息重建输入，并加入实时库存与执行约束。

"""
    text = SECTION_FILE.read_text(encoding="utf-8")
    start = text.index("## 9 分时段跨区调度路线与预投放配额")
    end = text.index("## 复核资料", start)
    text = text[:start] + section + text[end:]
    text = "\n".join(
        f"- [跨区调度路线表](../tables/zone_dispatch_routes.csv)：{len(routes)} 条候选路线的调出/投放区域、调运量与距离。"
        if line.startswith("- [跨区调度路线表]") else line
        for line in text.splitlines()
    ) + "\n"
    SECTION_FILE.write_text(text, encoding="utf-8")


def run(figure_dir: Path = FIGURE_DIR) -> dict[str, float]:
    """生成图13、图14、《分区预投放配额表》与《跨区调度路线表》。"""
    set_plot_style()
    stations_raw = load_station_trips()
    # 与 Notebook 共用原始精度，避免读回展示表时的坐标舍入影响分区和图片。
    summary = build_station_summary(stations_raw)
    clean = read_clean_data(CLEAN_TRIPS_FILE)

    days = weekday_days(stations_raw)
    stations, zones = build_zones(summary)
    flows = zone_segment_flows(stations_raw, stations, days)
    core = load_core_predictions(export_core_predictions())
    correction = prediction_correction(core, stations)
    dockless = dockless_trips(clean)
    hotspots = dockless_hotspots(dockless, stations, days)
    hotspot_balance = zone_hotspot_balance(dockless, stations, days)
    quota = build_quota(flows, correction, hotspot_balance, zones)
    routes = build_routes(quota)
    quota = attach_coverage(quota, routes)
    write_tables(zones, quota, routes, hotspots)
    write_scenario_report(zones, quota, routes, hotspots, dockless, days)

    print(f"区域 {len(zones)} 个；工作日 {days} 天；核心站点预测 {core['station_id'].nunique()} 个。")
    for segment in SEGMENTS:
        block = quota.loc[quota["segment"].eq(segment)]
        print(f"{segment}：需投放 {block['quota'].sum():.0f} 辆/日、可调出 {block['available'].sum():.0f} 辆/日、"
              f"路线 {int((routes['segment'] == segment).sum())} 条、调运 {routes.loc[routes.segment.eq(segment), 'volume_per_day'].sum():.0f} 辆/日")

    top_quota = quota.loc[quota["quota"].gt(0)].nlargest(12, "quota")[
        ["segment", "zone", "predicted_net", "hotspot_net", "gap", "quota", "delivered", "uncovered"]
    ].rename(columns={"segment": "时段", "zone": "区域", "predicted_net": "情景净流量",
                      "hotspot_net": "无桩热点净流入", "gap": "净缺口", "quota": "预投放配额",
                      "delivered": "已匹配调运", "uncovered": "未覆盖"})
    top_routes = routes.nlargest(12, "volume_per_day")[
        ["segment", "source_zone", "target_zone", "volume_per_day", "distance_km"]
    ].rename(columns={"segment": "时段", "source_zone": "调出区域", "target_zone": "投放区域",
                      "volume_per_day": "调运量(辆/日)", "distance_km": "直线距离(km)"})

    figures = (
        (
            plot_route_map(quota, routes, stations, hotspots),
            "fig13_zone_dispatch_routes",
            "图13  分时段跨区调度路线与配额（历史情景）",
            f"7 个片区、{len(routes)} 条跨区路线；四个时段合计预投放 "
            f"{quota['quota'].sum():.0f} 辆/日、可调出 {quota['available'].sum():.0f} 辆/日。",
            "历史情景：仅对 10 个核心站点叠加测试周模型偏差，其余站点沿用整月流量；"
            "缺站点电单车按各自借还时刻统计，供给向下取整。",
            top_routes,
            "表13-1  调运量 Top12 的跨区调度路线",
        ),
        (
            plot_quota_heatmap(quota, hotspots),
            "fig14_zone_quota_hotspots",
            "图14  分区配额与无桩电单车热点（历史情景）",
            f"净缺口＝情景净流量−热点净流入；需求另加 {SAFETY_STOCK:.0%} 余量，供给向下取整。",
            "矩阵是历史流量与模型偏差的合成情景；正负表示流出与流入压力，不能确认实际缺车或可回收库存。",
            top_quota,
            "表14-1  预投放配额 Top12（区域 × 时段）",
        ),
    )

    for figure, filename, title, subtitle, note, table, caption in figures:
        try:
            png_path, html_path = save_figure_html(
                figure, filename, title=title, subtitle=subtitle, note=note,
                table=table, table_caption=caption, directory=figure_dir,
                footer="数据来源：station_trips.csv、clean_trips.csv 与核心站点预测子集。"
                       "回顾性情景含测试周实测，不作为未来调度决策；图片与表格可离线查看。",
            )
            print(f"已保存：{png_path}")
            print(f"已保存：{html_path}")
        finally:
            plt.close(figure)

    best = quota.loc[quota["quota"].gt(0)].nlargest(1, "quota").iloc[0]
    print(f"最大单区配额：{best['zone']} {best['segment']} {best['quota']:.0f} 辆/日；"
          f"路线 {len(routes)} 条，合计 {routes['volume_per_day'].sum():.0f} 辆/日。")
    return {
        "zones": len(zones),
        "routes": len(routes),
        "quota_total": float(quota["quota"].sum()),
        "available_total": float(quota["available"].sum()),
        "volume_total": float(routes["volume_per_day"].sum()),
        "hotspots": len(hotspots),
        "core_stations": int(core["station_id"].nunique()),
        "coverage_am": float(quota.loc[quota["segment"].eq("早高峰"), "coverage"].mean(skipna=True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    run(parser.parse_args().figure_dir)


if __name__ == "__main__":
    main()
