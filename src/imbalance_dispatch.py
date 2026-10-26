"""成员 C：站点潮汐失衡、失衡指数与车辆调度策略（图11、图12）。

净流量定义为 ``F = 出发量 − 到达量``（正值＝净流出，站点缺车；负值＝净流入，站点积车），
失衡指数 ``I = |F| / (出发量 + 到达量 + 1)``，分母加 1 为防零平滑项。统计口径为
工作日早晚高峰（早 7:00–9:59、晚 17:00–19:59），与图9、图10 共用站点有效集。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_cleaning import read_clean_data
from .spatial_analysis import load_station_trips, shorten
from .utils import FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, save_figure_html, set_plot_style


PEAK_WINDOWS = {"早高峰": (7, 9), "晚高峰": (17, 19)}
PEAK_LABELS = {"早高峰": "早高峰 7:00–9:59", "晚高峰": "晚高峰 17:00–19:59"}
GRADE_RULES = ((0.30, 60), (0.20, 30))  # (失衡指数下限, 净流量绝对值下限)，由高到低匹配
DISPATCH_COLORS = {"居住区": "#315D8A", "商务区": "#159A85"}
FLOW_COLORS = {"deficit": "#AE5F67", "surplus": "#2F6E8F"}

PEAK_TABLE_FILE = TABLE_DIR / "peak_imbalance_summary.csv"
HOURLY_TABLE_FILE = TABLE_DIR / "hourly_net_flow.csv"
DISPATCH_TABLE_FILE = TABLE_DIR / "dispatch_plan.csv"


def weekday_count(station: pd.DataFrame) -> int:
    """统计口径中的工作日天数（周一至周五，1/1 与 1/15 也算工作日，与章节一一致）。"""
    weekdays = station.loc[station["weekday"].lt(5), "started_at"]
    days = pd.to_datetime(weekdays, errors="coerce").dt.normalize().nunique()
    return int(days)


def with_return_time(station: pd.DataFrame) -> pd.DataFrame:
    """补充按还车时间统计所需的列；出发小时仍用 hour（started_at）。"""
    data = station.copy()
    returned = pd.to_datetime(data["ended_at"], errors="coerce")
    data["end_hour"] = returned.dt.hour.astype("int8")
    data["end_weekday"] = returned.dt.dayofweek.astype("int8")
    return data


def build_peak_imbalance(station: pd.DataFrame) -> pd.DataFrame:
    """按早/晚高峰汇总各站点的出发、到达、净流量与失衡等级。

    出发量按出发时间归属，到达量按还车时间归属：一次 6:50 出发、7:10 还车的骑行
    计入 7 时的到达而不是 7 时的出发。两种归属对应不同的订单集合，因此同一窗口内
    各站点净流量之和不再为 0，差额是跨越窗口或跨日边界的骑行（见 ``window_boundary_flow``）。
    """
    data = with_return_time(station)
    weekday_start = data.loc[data["weekday"].lt(5)]
    weekday_end = data.loc[data["end_weekday"].lt(5)]
    frames = []
    for period, (start, end) in PEAK_WINDOWS.items():
        departures = (
            weekday_start.loc[weekday_start["hour"].between(start, end)]
            .groupby("start_station_id").size().rename("departures")
        )
        arrivals = (
            weekday_end.loc[weekday_end["end_hour"].between(start, end)]
            .groupby("end_station_id").size().rename("arrivals")
        )
        table = pd.concat([departures, arrivals], axis=1).fillna(0).astype(int)
        table.index.name = "station_id"
        table = table.reset_index()
        table["period"] = period
        table["net_flow"] = table["departures"] - table["arrivals"]
        table["imbalance_index"] = table["net_flow"].abs() / (table["departures"] + table["arrivals"] + 1)
        table["grade"] = [grade_of(row) for row in table.itertuples()]
        table["state"] = np.where(table["net_flow"].gt(0), "缺车",
                                  np.where(table["net_flow"].lt(0), "积车", "平衡"))
        frames.append(table)
    result = pd.concat(frames, ignore_index=True)
    names = pd.concat([
        station.groupby("start_station_id")["start_station_name"].first(),
        station.groupby("end_station_id")["end_station_name"].first(),
    ]).groupby(level=0).first().rename("station_name")
    result = result.merge(names, left_on="station_id", right_index=True, how="left")
    order = {"严重失衡": 0, "中度失衡": 1, "轻度失衡": 2, "基本均衡": 3}
    result["grade_order"] = result["grade"].map(order)
    return result.sort_values(["period", "net_flow"], ascending=[True, False]).reset_index(drop=True)


def window_boundary_flow(peak: pd.DataFrame) -> pd.DataFrame:
    """窗口内“出发总量 − 到达总量”：正值表示净有车辆骑出该窗口对应的时段。"""
    grouped = peak.groupby("period").agg(
        departures=("departures", "sum"),
        arrivals=("arrivals", "sum"),
        stations=("station_id", "nunique"),
        deficit=("net_flow", lambda values: values.clip(lower=0).sum()),
        surplus=("net_flow", lambda values: -values.clip(upper=0).sum()),
    )
    grouped["boundary_flow"] = grouped["departures"] - grouped["arrivals"]
    return grouped


def grade_of(row) -> str:
    """按失衡指数与净流量规模给出可执行的分级；小样本站点不会仅因指数高被列为严重。"""
    magnitude = abs(int(row.net_flow))
    for threshold, minimum in GRADE_RULES:
        if row.imbalance_index >= threshold and magnitude >= minimum:
            return "严重失衡" if threshold >= 0.30 else "中度失衡"
    return "轻度失衡" if magnitude >= 10 else "基本均衡"


def build_hourly_panel(station: pd.DataFrame) -> pd.DataFrame:
    """站点 × 小时的工作日净流量面板；到达量按还车时间归入对应小时。"""
    data = with_return_time(station)
    weekday_start = data.loc[data["weekday"].lt(5)]
    weekday_end = data.loc[data["end_weekday"].lt(5)]
    departures = weekday_start.groupby(["start_station_id", "hour"]).size().rename("departures")
    arrivals = weekday_end.groupby(["end_station_id", "end_hour"]).size().rename("arrivals")
    panel = pd.concat([departures, arrivals], axis=1).fillna(0).astype(int)
    panel.index.names = ["station_id", "hour"]
    panel = panel.reset_index()
    panel["net_flow"] = panel["departures"] - panel["arrivals"]
    names = station.groupby("start_station_id")["start_station_name"].first()
    panel["station_name"] = names.reindex(panel["station_id"]).to_numpy()
    return panel


def peak_profile(station: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """单个站点在工作日的 0–23 时净流量与累计净流量；到达量按还车小时归属。"""
    data = with_return_time(station)
    weekday_start = data.loc[data["weekday"].lt(5)]
    weekday_end = data.loc[data["end_weekday"].lt(5)]
    departures = weekday_start.loc[weekday_start["start_station_name"].eq(station_name)].groupby("hour").size()
    arrivals = weekday_end.loc[weekday_end["end_station_name"].eq(station_name)].groupby("end_hour").size()
    hours = pd.RangeIndex(24, name="hour")
    profile = pd.DataFrame({
        "hour": hours,
        "departures": departures.reindex(hours, fill_value=0).to_numpy(),
        "arrivals": arrivals.reindex(hours, fill_value=0).to_numpy(),
    })
    profile["net_flow"] = profile["departures"] - profile["arrivals"]
    profile["cumulative_net"] = profile["net_flow"].cumsum()
    profile["station_name"] = station_name
    return profile


def choose_representatives(peak: pd.DataFrame, summary: pd.DataFrame,
                           min_trips: int = 800) -> list[tuple[str, str]]:
    """从早高峰挑出净流出最大（居住区）与净流入最大（商务区）的高流量站点。"""
    am = peak.loc[peak["period"].eq("早高峰")].merge(
        summary[["station_id", "total_trips"]], on="station_id", how="left"
    )
    am = am.loc[am["total_trips"].ge(min_trips)]
    deficit = am.nlargest(1, "net_flow").iloc[0]
    surplus = am.nsmallest(1, "net_flow").iloc[0]
    return [("居住区", deficit["station_name"]), ("商务区", surplus["station_name"])]


def build_dispatch_plan(peak: pd.DataFrame, summary: pd.DataFrame, days: int,
                        max_distance_km: float = 6.0, top_targets: int = 30) -> pd.DataFrame:
    """按“最近可供给站点优先”的贪心匹配生成调运方案，量纲为辆/日。"""
    supply_geo = summary.set_index("station_id")[["lat", "lng"]]
    names = summary.set_index("station_id")["station_name"]
    rows = []
    for period in ("早高峰", "晚高峰"):
        block = peak.loc[peak["period"].eq(period)]
        demand = block.loc[block["net_flow"].gt(0)].nlargest(top_targets, "net_flow").copy()
        supply = block.loc[block["net_flow"].lt(0)].copy()
        demand["daily_need"] = demand["net_flow"] / days
        supply["daily_supply"] = -supply["net_flow"] / days
        available = dict(zip(supply["station_id"], supply["daily_supply"]))
        for target in demand.itertuples():
            remaining = float(target.daily_need)
            while remaining > 1.0:
                candidates = [sid for sid, volume in available.items() if volume > 1.0]
                if not candidates:
                    break
                target_point = supply_geo.loc[target.station_id]
                distances = _haversine(target_point, supply_geo.loc[candidates])
                distances = distances.loc[distances.le(max_distance_km)]
                if distances.empty:
                    break
                source = distances.idxmin()
                volume = min(remaining, available[source])
                rows.append({
                    "period": period,
                    "target_station": target.station_name,
                    "target_need_per_day": round(float(target.daily_need), 1),
                    "source_station": names.get(source, source),
                    "source_capacity_per_day": round(float(available[source]), 1),
                    "volume_per_day": round(float(volume), 1),
                    "distance_km": round(float(distances.loc[source]), 2),
                    "priority": "高" if float(target.daily_need) >= 8 else "中",
                })
                remaining -= volume
                available[source] -= volume
    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan
    plan["target_need_per_day"] = plan["target_need_per_day"].round(1)
    return plan.sort_values(["period", "volume_per_day"], ascending=[True, False]).reset_index(drop=True)


def _haversine(point: pd.Series, others: pd.DataFrame) -> pd.Series:
    """球面直线距离（km），用于调度线路的邻近匹配。"""
    radius = 6371.0
    lat1, lng1 = np.radians(point["lat"]), np.radians(point["lng"])
    lat2, lng2 = np.radians(others["lat"].to_numpy()), np.radians(others["lng"].to_numpy())
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return pd.Series(2 * radius * np.arctan2(np.sqrt(a), np.sqrt(1 - a)), index=others.index)


def plot_peak_divergence(peak: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """图11：早晚高峰核心站点的净流量正负双向对比。"""
    figure = plt.figure(figsize=(16.0, 9.4))
    figure.text(0.045, 0.958, "图11  早晚高峰核心潮汐站点净流量双向对比",
                fontsize=21, weight="bold", va="top")
    figure.text(0.045, 0.910,
                "净流量 F = 出发量 − 到达量（出发量按出发时间、到达量按还车时间归属）：正值（红，向右）为净流出即缺车，负值（蓝，向左）为净流入即积车",
                fontsize=10.6, color="#657585", va="top")

    for index, (period, position) in enumerate(zip(("早高峰", "晚高峰"),
                                                   ([0.255, 0.195, 0.245, 0.620], [0.625, 0.195, 0.245, 0.620]))):
        block = peak.loc[peak["period"].eq(period)]
        deficit = block.nlargest(top_n, "net_flow")
        surplus = block.nsmallest(top_n, "net_flow")
        selection = pd.concat([deficit, surplus.iloc[::-1]])
        axis = figure.add_axes(position)
        values = selection["net_flow"].to_numpy(dtype=float)
        positions = np.arange(len(selection))
        colors = [FLOW_COLORS["deficit"] if value > 0 else FLOW_COLORS["surplus"] for value in values]
        axis.barh(positions, values, height=0.66, color=colors, alpha=0.9)
        axis.set_yticks(positions, labels=[shorten(name, 27) for name in selection["station_name"]],
                        fontsize=8.8)
        axis.axvline(0, color="#203548", linewidth=1.1)
        axis.set_xlabel("月累计净流量（辆）")
        axis.set_title(f"({'ab'[index]}) {PEAK_LABELS[period]}\n蓝＝积车（净流入）　红＝缺车（净流出）",
                       loc="left", pad=14)
        axis.grid(axis="y", visible=False)
        limit = float(np.abs(values).max()) * 1.30
        axis.set_xlim(-limit, limit)
        for position_y, row in zip(positions, selection.itertuples()):
            offset = limit * 0.014
            inside = abs(row.net_flow) > limit * 0.42
            if row.net_flow > 0:
                x_text, horizontal = (row.net_flow - offset, "right") if inside else (row.net_flow + offset, "left")
            else:
                x_text, horizontal = (row.net_flow + offset, "left") if inside else (row.net_flow - offset, "right")
            axis.text(x_text, position_y, f"{row.net_flow:+.0f}（I={row.imbalance_index:.2f}）",
                      va="center", ha=horizontal, fontsize=8.2,
                      color="white" if inside else "#203548")

    boundary = {
        period: int(block["departures"].sum() - block["arrivals"].sum())
        for period, block in peak.groupby("period")
    }
    figure.text(0.045, 0.120,
                "统计口径：工作日（周一至周五，含 1/1 与 1/15）各高峰 3 小时；出发量按出发时间、到达量按还车时间归属，"
                "每根条为该站点的月累计净流量，括号内为失衡指数 I。",
                fontsize=10)
    figure.text(0.045, 0.078,
                f"同一窗口内各站净流量之和不为 0，差额是跨窗口边界的骑行（早高峰 {boundary['早高峰']:+,} 辆、"
                f"晚高峰 {boundary['晚高峰']:+,} 辆），因此不能用“缺车合计 = 积车合计”校验。",
                fontsize=10)
    figure.text(0.045, 0.036,
                "I = |净流量| ÷ (出发量 + 到达量 + 1)，分母加 1 为防零平滑项；完整分级见 reports/tables/peak_imbalance_summary.csv。",
                fontsize=10, color="#657585")
    return figure


def plot_tidal_curves(profiles: dict[str, pd.DataFrame], days: int) -> plt.Figure:
    """图12：居住区与商务区站点 24 小时净流量与累计净流量潮汐曲线。"""
    figure = plt.figure(figsize=(14.0, 9.8))
    figure.text(0.05, 0.960, "图12  典型居住区与商务区站点的 24 小时潮汐曲线",
                fontsize=21, weight="bold", va="top")
    figure.text(0.05, 0.916,
                f"工作日 n={days} 天，纵轴为全月累计口径；曲线形状在逐日平均后不变",
                fontsize=11, color="#657585", va="top")

    hourly = figure.add_axes([0.085, 0.575, 0.790, 0.270])
    cumulative = figure.add_axes([0.085, 0.190, 0.790, 0.270], sharex=hourly)

    for label, profile in profiles.items():
        color = DISPATCH_COLORS[label]
        hourly.plot(profile["hour"], profile["net_flow"], marker="o", markersize=3.4,
                    linewidth=1.6, color=color, label=f"{label}：{shorten(profile['station_name'].iloc[0], 26)}")
        cumulative.plot(profile["hour"], profile["cumulative_net"], marker="o", markersize=3.4,
                        linewidth=1.9, color=color,
                        label=f"{label}：{shorten(profile['station_name'].iloc[0], 26)}")

    for axis in (hourly, cumulative):
        for start, end in PEAK_WINDOWS.values():
            axis.axvspan(start - 0.5, end + 0.5, color="#DDE6ED", alpha=0.55, zorder=0)
        axis.set_xlim(-0.8, 24.3)
        axis.set_xticks(range(0, 24, 2))
        axis.grid(axis="x", visible=False)

    hourly.axhline(0, color="#203548", linewidth=1.0)
    cumulative.axhline(0, color="#203548", linewidth=1.0)
    hourly.set_ylabel("逐小时净流量（辆/月）")
    cumulative.set_ylabel("累计净流量（辆/月）")
    cumulative.set_xlabel("小时（出发与还车按各自时间归属）")
    hourly.set_title("(a) 逐小时净流量：早晚高峰方向相反", loc="left", pad=12)
    cumulative.set_title("(b) 累计净流量：居住区先“失血”后回补，商务区先“积压”后释放", loc="left", pad=12)
    hourly.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

    stacked = pd.concat([profile["cumulative_net"] for profile in profiles.values()])
    span = float(stacked.max() - stacked.min()) or 1.0
    cumulative.set_ylim(float(stacked.min()) - span * 0.20, float(stacked.max()) + span * 0.18)

    for label, profile in profiles.items():
        color = DISPATCH_COLORS[label]
        peak_row = profile.loc[profile["cumulative_net"].idxmax() if label == "居住区"
                               else profile["cumulative_net"].idxmin()]
        anchor_hour = min(float(peak_row["hour"]) + 0.6, 21.5)
        value = float(peak_row["cumulative_net"]) + (span * 0.085 if label == "居住区" else -span * 0.085)
        cumulative.annotate(f"{profile['station_name'].iloc[0]}（{label}）",
                            xy=(anchor_hour, value), xytext=(anchor_hour, value),
                            textcoords="data", ha="left", va="center", fontsize=9.2, color=color)
        row = profile.iloc[-1]
        cumulative.annotate(f"{row['cumulative_net']:+.0f}", xy=(23, row["cumulative_net"]),
                            xytext=(9, 0), textcoords="offset points", fontsize=9,
                            color=color, va="center", clip_on=False)

    figure.text(0.05, 0.118,
                "浅灰竖带为早晚高峰（7–9 时、17–19 时）。累计值为正表示当日累计净流出（账面车辆减少），为负表示累计净流入（账面增加）。",
                fontsize=10)
    figure.text(0.05, 0.074,
                "逐小时出发量按出发时间、到达量按还车时间统计；累计值以 0 时为起点，正负号即库存变化方向，不可反向解释。",
                fontsize=10, color="#657585")
    figure.text(0.05, 0.032,
                "站点选择规则：在早高峰且月流量 ≥ 800 次的站点中，分别取净流出与净流入最大者；不代表该站点的全部功能属性。",
                fontsize=10, color="#657585")
    return figure


def write_tables(peak: pd.DataFrame, panel: pd.DataFrame, plan: pd.DataFrame) -> dict[str, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "peak": (peak.drop(columns=["grade_order"]), PEAK_TABLE_FILE),
        "hourly": (panel, HOURLY_TABLE_FILE),
        "dispatch": (plan, DISPATCH_TABLE_FILE),
    }
    paths = {}
    for name, (frame, path) in outputs.items():
        frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
        paths[name] = path
    return paths


def run(figure_dir: Path = FIGURE_DIR) -> dict[str, float]:
    """生成图11、图12 与失衡、调运统计表；Notebook 与命令行共用同一入口。"""
    set_plot_style()
    station = load_station_trips(PROCESSED_DIR / "station_trips.csv")
    summary = pd.read_csv(TABLE_DIR / "station_flow_summary.csv", dtype={"station_id": "string"})
    days = weekday_count(station)

    peak = build_peak_imbalance(station)
    panel = build_hourly_panel(station)
    plan = build_dispatch_plan(peak, summary, days)
    write_tables(peak, panel, plan)

    am = peak.loc[peak["period"].eq("早高峰")]
    pm = peak.loc[peak["period"].eq("晚高峰")]
    flows = window_boundary_flow(peak)
    print(f"工作日 {days} 天；早高峰出发 {int(am['departures'].sum()):,} 次 / 到达 "
          f"{int(am['arrivals'].sum()):,} 次，晚高峰出发 {int(pm['departures'].sum()):,} 次 / 到达 "
          f"{int(pm['arrivals'].sum()):,} 次（到达按还车时间统计）。")
    for label, block in (("早高峰", am), ("晚高峰", pm)):
        deficit = block.loc[block["net_flow"].gt(0), "net_flow"].sum()
        surplus = -block.loc[block["net_flow"].lt(0), "net_flow"].sum()
        print(f"{label}：缺车合计 {deficit:,.0f} 辆、积车合计 {surplus:,.0f} 辆、"
              f"窗口边界净流量 {flows.loc[label, 'boundary_flow']:+,} 辆；"
              f"严重失衡 {int(block['grade'].eq('严重失衡').sum())} 个站点。")

    representatives = choose_representatives(peak, summary)
    profiles = {label: peak_profile(station, name) for label, name in representatives}
    print("代表站点：" + "；".join(f"{label}={name}" for label, name in representatives))

    top_plan = plan.sort_values("volume_per_day", ascending=False).head(14)
    figures = (
        (
            plot_peak_divergence(peak),
            "fig11_peak_divergence",
            "图11  早晚高峰核心潮汐站点净流量双向对比",
            f"早高峰出发 {int(am['departures'].sum()):,} 次、到达 {int(am['arrivals'].sum()):,} 次；"
            "红色向右为净流出（缺车），蓝色向左为净流入（积车）。",
            "出发量按出发时间、到达量按还车时间归属，因此同一窗口内净流量合计不为 0；"
            "同一批站点在早晚高峰方向相反，正是不平衡调度需要覆盖的时段。",
            am.nlargest(10, "net_flow")[["station_name", "departures", "arrivals", "net_flow", "imbalance_index", "grade"]]
            .rename(columns={"station_name": "站点", "departures": "出发量", "arrivals": "到达量",
                             "net_flow": "净流量", "imbalance_index": "失衡指数", "grade": "分级"}),
            "表11-1  早高峰缺车 Top10（月累计）",
        ),
        (
            plot_tidal_curves(profiles, days),
            "fig12_tidal_curves",
            "图12  典型居住区与商务区站点的 24 小时潮汐曲线",
            "；".join(f"{label}站点 {profiles[label]['station_name'].iloc[0]}" for label in profiles),
            "累计净流量为正表示当日账面车辆净减少（缺车方向），为负表示净增加（积车方向）。"
            "居住区站点先失血后回补，商务区站点先积压后释放，二者方向相反但幅度不对等。",
            top_plan[["period", "target_station", "source_station", "volume_per_day", "distance_km", "priority"]]
            .rename(columns={"period": "时段", "target_station": "补车站点", "source_station": "取车站点",
                             "volume_per_day": "建议调运量(辆/日)", "distance_km": "直线距离(km)",
                             "priority": "优先级"}),
            "表12-1  最近邻调运方案示例（按调运量排序）",
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

    print(f"调运方案 {len(plan)} 条线路，合计 {plan['volume_per_day'].sum():.0f} 辆/日。")
    return {
        "weekdays": days,
        "am_trips": int(am["departures"].sum()),
        "pm_trips": int(pm["departures"].sum()),
        "am_deficit": float(am.loc[am["net_flow"].gt(0), "net_flow"].sum()),
        "pm_deficit": float(pm.loc[pm["net_flow"].gt(0), "net_flow"].sum()),
        "am_severe": int(am["grade"].eq("严重失衡").sum()),
        "pm_severe": int(pm["grade"].eq("严重失衡").sum()),
        "routes": len(plan),
        "volume_per_day": float(plan["volume_per_day"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    run(parser.parse_args().figure_dir)


if __name__ == "__main__":
    main()
