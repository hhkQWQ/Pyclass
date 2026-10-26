"""康炜奕：站点空间分布、热门站点与高频 OD 走廊分析（图9、图10）。

输入为 ``data/processed/station_trips.csv``（双端站点 ID 完整的 180,914 条记录）。
输出三张统计表（站点流量、OD 走廊、站点坐标）和两张图，每张图同时导出
300 DPI PNG 与自包含 HTML，供报告排版和网页预览共用。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

from .data_cleaning import read_clean_data
from .utils import (
    FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, save_figure_html, set_plot_style,
)


STATION_FLOW_FILE = TABLE_DIR / "station_flow_summary.csv"
OD_CORRIDOR_FILE = TABLE_DIR / "od_corridor_summary.csv"
STATION_COORD_FILE = TABLE_DIR / "station_coordinates.csv"

# 主城取景窗口：用于地图聚焦，窗口外站点仍参与全部流量统计。
CORE_LNG = (-77.125, -76.895)
CORE_LAT = (38.790, 39.000)
NAME_WIDTH = 26

FLOW_COLORS = {"departures": "#315D8A", "arrivals": "#159A85"}
NET_COLORS = {"deficit": "#AE5F67", "surplus": "#315D8A"}
MAP_COLORS = LinearSegmentedColormap.from_list(
    "station_net", ["#1F4E79", "#6E9BC0", "#EEF3F6", "#D08C86", "#A03B37"],
)


def load_station_trips(path: str | Path = PROCESSED_DIR / "station_trips.csv") -> pd.DataFrame:
    """读取站点子集并检查接口约定。"""
    data = read_clean_data(path)
    if data.empty:
        raise ValueError("站点子集为空，请先运行 python -m src.data_cleaning。")
    if not data["has_station_info"].fillna(False).all():
        raise ValueError("站点子集包含缺站点记录，接口不一致。")
    return data


def shorten(name: str, width: int = NAME_WIDTH) -> str:
    """站点名过长时截断，保证坐标轴可读；完整名称见统计表。"""
    text = str(name)
    return text if len(text) <= width else text[: width - 1] + "…"


def attach_coordinates(station: pd.DataFrame) -> pd.DataFrame:
    """起终点分别取首个可用坐标，按站点 ID 汇总为 (lat, lng)。"""
    start = station.groupby("start_station_id").agg(
        start_lat=("start_lat", "first"), start_lng=("start_lng", "first")
    )
    end = station.groupby("end_station_id").agg(
        end_lat=("end_lat", "first"), end_lng=("end_lng", "first")
    )
    coordinates = start.join(end, how="outer")
    coordinates["lat"] = coordinates["start_lat"].fillna(coordinates["end_lat"])
    coordinates["lng"] = coordinates["start_lng"].fillna(coordinates["end_lng"])
    counts = pd.concat([
        station.groupby("start_station_id").size().rename("departures"),
        station.groupby("end_station_id").size().rename("arrivals"),
    ], axis=1).fillna(0)
    names = pd.concat([
        station.groupby("start_station_id")["start_station_name"].first(),
        station.groupby("end_station_id")["end_station_name"].first(),
    ]).groupby(level=0).first()
    table = counts.join(coordinates[["lat", "lng"]]).join(names.rename("station_name"))
    table.index.name = "station_id"
    table[["departures", "arrivals"]] = table[["departures", "arrivals"]].astype(int)
    if table["lat"].isna().any() or table["lng"].isna().any():
        missing = table.loc[table["lat"].isna() | table["lng"].isna(), "station_name"]
        raise ValueError(f"存在缺少坐标的站点：{missing.tolist()}")
    return table.reset_index()


def build_station_summary(station: pd.DataFrame) -> pd.DataFrame:
    """站点总出发量、总到达量、净流量与失衡指数；指数加 1 平滑避免分母为 0。"""
    summary = attach_coordinates(station)
    summary["total_trips"] = summary["departures"] + summary["arrivals"]
    summary["net_flow"] = summary["departures"] - summary["arrivals"]
    summary["imbalance_index"] = summary["net_flow"].abs() / (summary["total_trips"] + 1)
    summary["rank_total"] = summary["total_trips"].rank(ascending=False, method="first").astype(int)
    summary["rank_abs_net"] = summary["net_flow"].abs().rank(ascending=False, method="first").astype(int)
    return summary.sort_values("total_trips", ascending=False).reset_index(drop=True)


def build_od_corridors(station: pd.DataFrame) -> pd.DataFrame:
    """把 A→B 与 B→A 合并为一条无向走廊，并保留两个方向的次数。"""
    trips = station.loc[station["start_station_id"].ne(station["end_station_id"])]
    direct = (
        trips.groupby(["start_station_id", "end_station_id", "start_station_name", "end_station_name"])
        .size()
        .rename("trips")
        .reset_index()
    )
    direct["station_a"] = direct[["start_station_id", "end_station_id"]].min(axis=1)
    direct["station_b"] = direct[["start_station_id", "end_station_id"]].max(axis=1)
    direct["name_a"] = direct["start_station_name"].where(
        direct["start_station_id"].eq(direct["station_a"]), direct["end_station_name"]
    )
    direct["name_b"] = direct["start_station_name"].where(
        direct["start_station_id"].eq(direct["station_b"]), direct["end_station_name"]
    )
    forward = (
        direct.loc[direct["start_station_id"].eq(direct["station_a"])]
        .groupby(["station_a", "station_b"])["trips"].sum().rename("trips_a_to_b")
    )
    backward = (
        direct.loc[direct["start_station_id"].eq(direct["station_b"])]
        .groupby(["station_a", "station_b"])["trips"].sum().rename("trips_b_to_a")
    )
    corridors = pd.concat([forward, backward], axis=1).fillna(0).astype(int).reset_index()
    names = direct.groupby(["station_a", "station_b"])[["name_a", "name_b"]].first().reset_index()
    corridors = corridors.merge(names, on=["station_a", "station_b"], how="left")
    corridors["total_trips"] = corridors["trips_a_to_b"] + corridors["trips_b_to_a"]
    corridors["dominant_trips"] = corridors[["trips_a_to_b", "trips_b_to_a"]].max(axis=1)
    corridors["dominant_share"] = corridors["dominant_trips"] / corridors["total_trips"]
    corridors["dominant_direction"] = np.where(
        corridors["trips_a_to_b"].ge(corridors["trips_b_to_a"]),
        corridors["name_a"] + " → " + corridors["name_b"],
        corridors["name_b"] + " → " + corridors["name_a"],
    )
    return corridors.sort_values("total_trips", ascending=False).reset_index(drop=True)


def corridor_concentration(corridors: pd.DataFrame) -> pd.Series:
    """走廊集中度：Top1/Top10/Top50 占全部非环线骑行的比例。"""
    total = corridors["total_trips"].sum()
    return pd.Series({
        "station_pairs": len(corridors),
        "station_pairs_both_directions": int(corridors[["trips_a_to_b", "trips_b_to_a"]].gt(0).all(axis=1).sum()),
        "non_loop_trips": int(total),
        "top1_share": corridors["total_trips"].iloc[:1].sum() / total,
        "top10_share": corridors["total_trips"].iloc[:10].sum() / total,
        "top50_share": corridors["total_trips"].iloc[:50].sum() / total,
        "top_station_pair": f"{corridors.iloc[0].name_a} ↔ {corridors.iloc[0].name_b}",
        "top_station_pair_trips": int(corridors["total_trips"].iloc[0]),
    })


def plot_top_stations_od(summary: pd.DataFrame, corridors: pd.DataFrame, station: pd.DataFrame,
                         top_n: int = 15) -> plt.Figure:
    """图9：Top15 热门站点总流量（出发/到达）与 Top15 高频 OD 走廊。"""
    top_stations = summary.nsmallest(top_n, "rank_total")
    top_od = corridors.head(top_n).iloc[::-1]

    figure = plt.figure(figsize=(16.4, 9.2))
    figure.text(0.045, 0.958, "图9  Top15 热门骑行站点与高频 OD 走廊",
                fontsize=21, weight="bold", va="top")
    figure.text(
        0.045, 0.908,
        f"站点有效集 n={len(station):,}  ·  站点 {len(summary)} 个  ·  "
        f"非环线 OD {int(corridors['total_trips'].sum()):,} 次 / {len(corridors):,} 个站点对",
        fontsize=11, color="#657585", va="top",
    )

    left = figure.add_axes([0.180, 0.190, 0.300, 0.640])
    positions = np.arange(len(top_stations))
    height = 0.38
    left.barh(positions + height / 2, top_stations["departures"], height=height * 0.92,
              color=FLOW_COLORS["departures"], label="总出发量")
    left.barh(positions - height / 2, top_stations["arrivals"], height=height * 0.92,
              color=FLOW_COLORS["arrivals"], label="总到达量")
    left.set_yticks(positions, labels=[shorten(name, 24) for name in top_stations["station_name"]])
    left.invert_yaxis()
    left.set_xlabel("月累计骑行次数")
    left.set_title(f"(a) 总流量 Top{top_n} 站点（深蓝＝出发，绿色＝到达）\n按出发量与到达量合计排序",
                   loc="left", pad=14)
    left.grid(axis="y", visible=False)
    scale = top_stations["departures"].max()
    left.set_xlim(0, scale * 1.16)
    for position, row in zip(positions, top_stations.itertuples()):
        left.text(row.departures + scale * 0.012, position, f"{row.net_flow:+.0f}",
                  va="center", ha="left", fontsize=9,
                  color=NET_COLORS["deficit"] if row.net_flow > 0 else NET_COLORS["surplus"])

    right = figure.add_axes([0.605, 0.190, 0.360, 0.640])
    positions_od = np.arange(len(top_od))
    right.barh(positions_od, top_od["total_trips"], height=0.62, color="#315D8A", alpha=0.88)
    labels = []
    for row in top_od.itertuples():
        origin, target = (row.name_a, row.name_b) if row.trips_a_to_b >= row.trips_b_to_a else (row.name_b, row.name_a)
        labels.append(f"{shorten(origin, 19)} → {shorten(target, 19)}")
    right.set_yticks(positions_od, labels=labels, fontsize=8.8)
    right.invert_yaxis()
    right.set_xlabel("双向合计骑行次数")
    right.set_title(f"(b) 高频 OD 走廊 Top{top_n}\nA→B 与 B→A 合并统计，标签为主向", loc="left", pad=14)
    right.grid(axis="y", visible=False)
    right.set_xlim(0, top_od["total_trips"].max() * 1.30)
    for position, row in zip(positions_od, top_od.itertuples()):
        right.text(row.total_trips + top_od["total_trips"].max() * 0.015, position,
                   f"{row.total_trips:,} 次 · 主向 {row.dominant_share:.0%}",
                   va="center", ha="left", fontsize=8.6, color="#526271")

    figure.text(0.045, 0.108,
                "站点流量为 1 月全月累计；条尾数字为净流量（出发−到达），正值为净流出，负值为净流入。",
                fontsize=10)
    figure.text(0.045, 0.072,
                "走廊按无向合并后排序，主向占比＝较大方向次数 ÷ 双向合计；环线（起终点相同）不计入走廊。",
                fontsize=10, color="#657585")
    figure.text(0.045, 0.036,
                "站点名称按站点 ID 统一；名称过长时截断，完整名称见 reports/tables/station_flow_summary.csv。",
                fontsize=10, color="#657585")
    return figure


def plot_station_heatmap(summary: pd.DataFrame, station: pd.DataFrame) -> plt.Figure:
    """图10：站点骑行热度空间分布，大小表示总流量，颜色表示净流量方向。"""
    core = summary["lng"].between(*CORE_LNG) & summary["lat"].between(*CORE_LAT)
    outside = summary.loc[~core]
    outside_trips = int(outside["total_trips"].sum())
    plotted = summary.loc[core]

    figure = plt.figure(figsize=(12.8, 10.2))
    figure.text(0.05, 0.955, "图10  城市站点骑行热度空间分布",
                fontsize=21, weight="bold", va="top")
    figure.text(
        0.05, 0.910,
        f"站点有效集 n={len(station):,}  ·  主城 {len(plotted)} 个站点占 "
        f"{1 - outside_trips / summary['total_trips'].sum():.2%} 的骑行量",
        fontsize=11, color="#657585", va="top",
    )
    axis = figure.add_axes([0.085, 0.115, 0.760, 0.755])

    bounds = float(np.percentile(summary["net_flow"].abs(), 96))
    norm = TwoSlopeNorm(vmin=-bounds, vcenter=0, vmax=bounds)
    sizes = 14 + 520 * (summary["total_trips"] / summary["total_trips"].max()) ** 0.72
    scatter = axis.scatter(
        summary["lng"], summary["lat"], s=sizes, c=summary["net_flow"], cmap=MAP_COLORS,
        norm=norm, alpha=0.82, linewidths=0.5, edgecolors="white", zorder=3,
    )

    axis.set_xlim(*CORE_LNG)
    axis.set_ylim(*CORE_LAT)
    axis.set_aspect(1 / np.cos(np.radians(float(plotted["lat"].mean()))))
    axis.set_xlabel("经度")
    axis.set_ylabel("纬度")
    axis.ticklabel_format(useOffset=False, style="plain")
    axis.grid(alpha=0.35, linestyle=":")

    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.030, pad=0.015)
    colorbar.set_label("月净流量（出发−到达，辆）", fontsize=10, labelpad=6)
    colorbar.ax.tick_params(labelsize=9)

    corners = {
        "NW": ((-64, 26), "right", "bottom"),
        "NE": ((34, 30), "left", "bottom"),
        "SW": ((-58, -30), "right", "top"),
        "SE": ((32, -34), "left", "top"),
    }
    extremes = [
        ("NW", summary.nlargest(1, "net_flow").iloc[0]),
        ("NE", summary.nlargest(2, "net_flow").iloc[1]),
        ("SE", summary.nsmallest(1, "net_flow").iloc[0]),
        ("SW", summary.nsmallest(2, "net_flow").iloc[1]),
    ]
    for corner, row in extremes:
        (offset_x, offset_y), horizontal, vertical = corners[corner]
        axis.annotate(
            f"{shorten(row['station_name'], 26)}\n{row['net_flow']:+.0f} 辆 · I={row['imbalance_index']:.2f}",
            xy=(row["lng"], row["lat"]), xycoords="data",
            xytext=(offset_x, offset_y), textcoords="offset points",
            fontsize=8.6, ha=horizontal, va=vertical, color="#203548",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white",
                  "edgecolor": "#D3DDE5", "alpha": 0.95},
            arrowprops={"arrowstyle": "-", "color": "#8093A3", "lw": 0.8,
                        "shrinkA": 3, "shrinkB": 7},
            zorder=6,
        )

    axis.text(0.62, 0.035, "站点月度流量分布\n时段差异另行分析", transform=axis.transAxes,
              fontsize=9, color="#2F5478", ha="center")

    figure.text(0.05, 0.070,
                f"颜色截断于净流量绝对值 P96（±{bounds:.0f} 辆）；接近浅灰的站点月度收支接近平衡。",
                fontsize=10)
    figure.text(0.05, 0.040,
                f"为突出主城结构，取景窗口外另有 {len(outside)} 个外围站点（合计 {outside_trips:,} 次，"
                f"{outside_trips / summary['total_trips'].sum():.2%}）未绘出，它们仍计入全部统计。",
                fontsize=10, color="#657585")
    return figure


def write_tables(summary: pd.DataFrame, corridors: pd.DataFrame) -> dict[str, Path]:
    """导出站点流量、OD 走廊和坐标表，供报告与后续调度模块复用。"""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "station_flow": summary,
        "od_corridor": corridors,
        "coordinates": summary[["station_id", "station_name", "lat", "lng"]],
    }
    paths = {}
    for name, frame in outputs.items():
        path = {"station_flow": STATION_FLOW_FILE, "od_corridor": OD_CORRIDOR_FILE,
                "coordinates": STATION_COORD_FILE}[name]
        frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
        paths[name] = path
    return paths


def run(figure_dir: Path = FIGURE_DIR) -> dict[str, float]:
    """生成图9、图10 与配套统计表；Notebook 与命令行共用同一入口。"""
    set_plot_style()
    station = load_station_trips()
    summary = build_station_summary(station)
    corridors = build_od_corridors(station)
    concentration = corridor_concentration(corridors)
    write_tables(summary, corridors)

    top_stations = summary.nsmallest(15, "rank_total")[
        ["station_name", "departures", "arrivals", "total_trips", "net_flow", "imbalance_index"]
    ].copy()
    top_corridors = corridors.head(15)[
        ["name_a", "name_b", "trips_a_to_b", "trips_b_to_a", "total_trips", "dominant_share"]
    ].copy()

    figures = (
        (
            plot_top_stations_od(summary, corridors, station),
            "fig09_top_stations_od",
            "图9  Top15 热门骑行站点与高频 OD 走廊",
            f"站点有效集 n={len(station):,}，站点 {len(summary)} 个；"
            f"非环线骑行 {int(concentration['non_loop_trips']):,} 次，构成 {len(corridors):,} 个站点对。",
            "左图气泡条为月累计出发量与到达量，条尾数字为净流量；右图为双向合并后的高频走廊。"
            "Top10 走廊合计占非环线骑行的 "
            f"{concentration['top10_share']:.2%}，说明少量走廊的覆盖比例有限。",
            top_stations.rename(columns={
                "station_name": "站点", "departures": "出发量", "arrivals": "到达量",
                "total_trips": "总流量", "net_flow": "净流量", "imbalance_index": "失衡指数",
            }),
            "表9-1  总流量 Top15 站点（月累计）",
        ),
        (
            plot_station_heatmap(summary, station),
            "fig10_station_heatmap",
            "图10  城市站点骑行热度空间分布",
            f"站点 {len(summary)} 个，月累计 {int(station.shape[0]):,} 次骑行；"
            "气泡面积表示站点总流量，颜色冷暖表示净流出与净流入。",
            "图中展示出发月份队列的月度站点总流量与净流量，不用于直接判断早晚高峰的方向。"
            "净流量为订单流量指标，不能据此确认实际缺车或满桩。",
            summary.nsmallest(10, "rank_abs_net")[
                ["station_name", "departures", "arrivals", "net_flow", "imbalance_index"]
            ].rename(columns={
                "station_name": "站点", "departures": "出发量", "arrivals": "到达量",
                "net_flow": "净流量", "imbalance_index": "失衡指数",
            }),
            "表10-1  净流量绝对值 Top10 站点（月累计）",
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

    print(f"站点 {len(summary)} 个；非环线 OD {len(corridors):,} 个站点对；"
          f"Top10 走廊占非环线骑行 {concentration['top10_share']:.2%}。")
    return {
        "stations": len(summary),
        "trips": int(len(station)),
        "corridors": len(corridors),
        "top10_share": float(concentration["top10_share"]),
        "top_corridor_trips": int(concentration["top_station_pair_trips"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    run(parser.parse_args().figure_dir)


if __name__ == "__main__":
    main()
