"""成员 C（任务 3）：结合需求预测与无桩电单车热点，输出分区预投放配额与跨区调度路线。

输入三份数据：

1. ``data/processed/station_trips.csv``：站点有效集，用于计算各区域分时段净流量；
2. ``data/derived/core_station_predictions.csv``：成员 B 的站点级逐小时预测
   （``historical_mean`` 模型，2024-01-22 至 01-28 的 test 窗口，含 actual 列），
   取自 ``feat/temporal-and-modeling`` 分支 ``reports/tables/station_weekly_predictions.csv``，
   只保留 10 个核心站点的逐小时预测与实测，避免 6.9 MB 原表随本分支提交；
3. ``data/processed/clean_trips.csv``：全量有效集，用于提取无桩电单车（缺站点 ID 的电单车）
   的空间热点。

净流量 ``F = 出发量 − 到达量``，正值表示站点缺车、负值表示积车；分时段统计中到达量按
还车时间归属（与第 8 节一致）。区域按站点坐标做加权 k-means 划分为 7 个片区，再按方位命名。
预投放配额 = 预测净流量 + 无桩热点净需求，再按 15% 安全库存上浮；跨区路线在同一时段内
按“最近邻优先”贪心匹配，量纲为辆/日。
"""

from __future__ import annotations

import argparse
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
from .spatial_analysis import load_station_trips, shorten
from .utils import (
    FIGURE_DIR, PROCESSED_DIR, PROJECT_ROOT, TABLE_DIR, save_figure_html, set_plot_style,
)

DATA_DIR = PROJECT_ROOT / "data"
CLEAN_TRIPS_FILE = PROCESSED_DIR / "clean_trips.csv"
CORE_PREDICTION_FILE = DATA_DIR / "derived" / "core_station_predictions.csv"
ZONE_PROFILE_FILE = TABLE_DIR / "zone_profile.csv"
ZONE_QUOTA_FILE = TABLE_DIR / "zone_preallocation_quota.csv"
ROUTE_FILE = TABLE_DIR / "zone_dispatch_routes.csv"
HOTSPOT_FILE = TABLE_DIR / "dockless_hotspots.csv"

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
    """无桩电单车（缺站点 ID 的电单车）的工作日记录，限定在主城窗口内。"""
    data = clean_trips.loc[~clean_trips["has_station_info"].fillna(False)].copy()
    data = data.loc[data["weekday"].lt(5)]
    inside = (data["start_lat"].between(38.79, 39.00) & data["start_lng"].between(-77.13, -76.89))
    return data.loc[inside]


def dockless_hotspots(dockless: pd.DataFrame, stations: pd.DataFrame, days: int,
                      grid: float = GRID_DEG, top_n: int = HOTSPOT_TOP_N) -> pd.DataFrame:
    """无桩电单车的方格热点；net = 还车 − 取车，正值为可回收车辆。"""
    origin = pd.DataFrame({
        "cell_lat": (dockless["start_lat"] / grid).round().astype(int),
        "cell_lng": (dockless["start_lng"] / grid).round().astype(int),
    }).groupby(["cell_lat", "cell_lng"]).size().rename("departures")
    destination = pd.DataFrame({
        "cell_lat": (dockless["end_lat"] / grid).round().astype(int),
        "cell_lng": (dockless["end_lng"] / grid).round().astype(int),
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
    data["segment"] = segment_of(data["hour"])
    data["zone_start"] = assign_nearest_zone(data, stations, "start_lat", "start_lng").to_numpy()
    data["zone_end"] = assign_nearest_zone(data, stations, "end_lat", "end_lng").to_numpy()
    starts = data.groupby(["zone_start", "segment"]).size().rename("departures").reset_index()
    ends = data.groupby(["zone_end", "segment"]).size().rename("arrivals").reset_index()
    starts = starts.rename(columns={"zone_start": "zone"})
    ends = ends.rename(columns={"zone_end": "zone"})
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
    base["available"] = np.where(base["gap"] < 0, np.ceil(-base["gap"]), 0.0)
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
            while need > 1.0:
                candidates = [zone for zone, volume in remaining.items() if volume > 1.0]
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
    plan = pd.DataFrame(rows)
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
        else "可调出" if row.available > 1
        else "基本平衡"
        for row in quota.itertuples()
    ]
    return quota.drop(columns=["target_zone"])


def plot_route_map(quota: pd.DataFrame, routes: pd.DataFrame, stations: pd.DataFrame,
                   hotspots: pd.DataFrame) -> plt.Figure:
    """图13：四个时段的区域角色、跨区调度路线与无桩热点。"""
    figure = plt.figure(figsize=(15.2, 12.2))
    figure.text(0.045, 0.968, "图13  分时段跨区调度路线与预投放配额",
                fontsize=21, weight="bold", va="top")
    figure.text(0.045, 0.936,
                "圆点大小＝预投放配额或可调出量（辆/日），红色为需投放区域、蓝色为可调出区域；"
                "箭头为跨区路线，菱形为无桩电单车热点",
                fontsize=10.6, color="#657585", va="top")

    positions = [(0.055, 0.520, 0.400, 0.365), (0.545, 0.520, 0.400, 0.365),
                 (0.055, 0.090, 0.400, 0.365), (0.545, 0.090, 0.400, 0.365)]
    map_hotspots = hotspots.head(HOTSPOT_MAP_N)

    for index, (segment, (left, bottom, width, height)) in enumerate(zip(SEGMENTS, positions)):
        axis = figure.add_axes([left, bottom, width, height])
        block = quota.loc[quota["segment"].eq(segment)]
        axis.scatter(stations["lng"], stations["lat"], s=3.5, c="#C9D4DC", linewidths=0, zorder=1)
        axis.scatter(map_hotspots["lng"], map_hotspots["lat"],
                     s=16 + 80 * (map_hotspots["activity"] / hotspots["activity"].max()),
                     marker="D", c=HOTSPOT_COLOR, alpha=0.5, edgecolors="white",
                     linewidths=0.6, zorder=2)
        scale = max(float(block["quota"].max()), float(block["available"].max()), 1.0)
        for row in block.itertuples():
            value = max(row.quota, row.available)
            if value < 2:
                continue
            sink = row.quota > 0
            axis.scatter(row.lng, row.lat, s=40 + 900 * (value / scale),
                         c=SINK_COLOR if sink else SOURCE_COLOR, alpha=0.72,
                         edgecolors="white", linewidths=1.2, zorder=4)
            if row.zone == "中心商务区":
                offset, horizontal = (-52, -8), "right"
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
        axis.set_xlim(-77.16, -76.86)
        axis.set_ylim(38.775, 39.065)
        axis.set_aspect(1 / np.cos(np.radians(38.90)))
        axis.ticklabel_format(useOffset=False, style="plain")
        axis.tick_params(labelsize=7.6)
        if index < 2:
            axis.tick_params(labelbottom=False)
        axis.grid(alpha=0.3, linestyle=":")
        axis.set_title(SEGMENT_LABELS[segment], loc="left", pad=8, fontsize=11)

    figure.text(0.045, 0.052,
                "分区为站点坐标的加权 k-means（k=7，权重＝站点月流量）；跨区路线只在同一时段内匹配，"
                "距离为区域重心间的球面直线距离；配额小于 2 辆/日的区域不标注。",
                fontsize=10)
    figure.text(0.045, 0.022,
                "无桩电单车指缺站点 ID 的电单车订单，是运营形态的代理标记，不代表实际车辆归属；"
                "热点方格 0.0025°（约 250 m），仅显示活动量前 10 的格子。",
                fontsize=10, color="#657585")
    return figure


def plot_quota_heatmap(quota: pd.DataFrame, hotspots: pd.DataFrame) -> plt.Figure:
    """图14：分区预投放配额热力图与无桩热点排行。"""
    figure = plt.figure(figsize=(14.8, 8.8))
    figure.text(0.05, 0.955, "图14  分区预投放配额与无桩电单车热点",
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
                "净缺口 = 预测净流量 − 无桩热点净流入；预测净流量 = 分区观测净流量 + 核心站点的预测修正量，"
                "预投放配额在此基础上上浮 15% 安全库存。",
                fontsize=10)
    figure.text(0.05, 0.022,
                "热点颜色：红＝还车多于取车（车辆堆积，可回收），蓝＝取车多于还车（车辆流失，需回补）；"
                "本图中热点净流入均为正。",
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


def run(figure_dir: Path = FIGURE_DIR) -> dict[str, float]:
    """生成图13、图14、《分区预投放配额表》与《跨区调度路线表》。"""
    set_plot_style()
    stations_raw = load_station_trips()
    summary = pd.read_csv(TABLE_DIR / "station_flow_summary.csv", dtype={"station_id": "string"})
    clean = read_clean_data(CLEAN_TRIPS_FILE)

    days = weekday_days(stations_raw)
    stations, zones = build_zones(summary)
    flows = zone_segment_flows(stations_raw, stations, days)
    core = load_core_predictions()
    correction = prediction_correction(core, stations)
    dockless = dockless_trips(clean)
    hotspots = dockless_hotspots(dockless, stations, days)
    hotspot_balance = zone_hotspot_balance(dockless, stations, days)
    quota = build_quota(flows, correction, hotspot_balance, zones)
    routes = build_routes(quota)
    quota = attach_coverage(quota, routes)
    write_tables(zones, quota, routes, hotspots)

    print(f"区域 {len(zones)} 个；工作日 {days} 天；核心站点预测 {core['station_id'].nunique()} 个。")
    for segment in SEGMENTS:
        block = quota.loc[quota["segment"].eq(segment)]
        print(f"{segment}：需投放 {block['quota'].sum():.0f} 辆/日、可调出 {block['available'].sum():.0f} 辆/日、"
              f"路线 {int((routes['segment'] == segment).sum())} 条、调运 {routes.loc[routes.segment.eq(segment), 'volume_per_day'].sum():.0f} 辆/日")

    top_quota = quota.loc[quota["quota"].gt(0)].nlargest(12, "quota")[
        ["segment", "zone", "predicted_net", "hotspot_net", "gap", "quota", "delivered", "uncovered"]
    ].rename(columns={"segment": "时段", "zone": "区域", "predicted_net": "预测净流量",
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
            "图13  分时段跨区调度路线与预投放配额",
            f"7 个片区、{len(routes)} 条跨区路线；四个时段合计预投放 "
            f"{quota['quota'].sum():.0f} 辆/日、可调出 {quota['available'].sum():.0f} 辆/日。",
            "结合成员 B 的站点级预测与无桩电单车热点：预测修正量只作用于 10 个核心站点，"
            "其余站点沿用 1 月观测；无桩热点提供额外的回收来源。",
            top_routes,
            "表13-1  调运量 Top12 的跨区调度路线",
        ),
        (
            plot_quota_heatmap(quota, hotspots),
            "fig14_zone_quota_hotspots",
            "图14  分区预投放配额与无桩电单车热点",
            f"净缺口＝预测净流量−热点净流入；预投放配额另加 {SAFETY_STOCK:.0%} 安全库存。",
            "早高峰缺口集中在北部、东北部居住片区，中心商务区与西部片区可调出；"
            "夜间方向反转但量级较小。无桩热点以还车堆积为主，可作为回收来源。",
            top_quota,
            "表14-1  预投放配额 Top12（区域 × 时段）",
        ),
    )

    for figure, filename, title, subtitle, note, table, caption in figures:
        try:
            png_path, html_path = save_figure_html(
                figure, filename, title=title, subtitle=subtitle, note=note,
                table=table, table_caption=caption, directory=figure_dir,
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
