"""生成马文博负责的图 1–3 和基础探索统计表。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .data_audit import CleaningRules, load_rules
from .data_cleaning import read_clean_data
from .utils import (
    FIGURE_DIR, TABLE_DIR, PROCESSED_DIR, RIDEABLE_COLORS, MEMBER_COLORS,
    STRUCTURE_COLORS, save_figure, set_plot_style,
)


GROUP_LABELS = {
    "classic_bike": "经典车", "electric_bike": "电单车",
    "member": "会员", "casual": "散客",
}


def build_summary_tables(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    statistics = []
    for dimension, groups in (("rideable_type", RIDEABLE_COLORS), ("member_casual", MEMBER_COLORS)):
        for group in groups:
            subset = clean.loc[clean[dimension].eq(group)]
            for metric in ("duration_min", "dist_km"):
                values = subset[metric].dropna()
                quartiles = values.quantile([0.25, 0.5, 0.75, 0.95])
                statistics.append({
                    "dimension": dimension, "group": group, "group_label": GROUP_LABELS[group],
                    "metric": metric, "n_total": len(subset), "n_valid": len(values),
                    "n_missing": len(subset) - len(values), "mean": values.mean(),
                    "q25": quartiles.loc[0.25], "median": quartiles.loc[0.5],
                    "q75": quartiles.loc[0.75], "p95": quartiles.loc[0.95],
                    "min": values.min(), "max": values.max(),
                })
    known = clean["has_station_info"].fillna(False)
    proxy = clean["is_dockless_ebike"].fillna(False)
    loops = clean["is_loop_trip"].fillna(False)
    counts = [
        ("station_structure", "station_complete", "双端站点完整", int(known.sum()), len(clean)),
        ("station_structure", "dockless_ebike_proxy", "疑似无桩电单车", int(proxy.sum()), len(clean)),
        ("station_structure", "other_missing_station", "其他缺站点", int((~known & ~proxy).sum()), len(clean)),
        ("loop_availability", "known", "环线可判定", int(known.sum()), len(clean)),
        ("loop_availability", "unknown", "环线未知", int((~known).sum()), len(clean)),
        ("loop_structure", "different_station", "异站骑行", int((known & ~loops).sum()), int(known.sum())),
        ("loop_structure", "loop_trip", "同站环线", int((known & loops).sum()), int(known.sum())),
    ]
    structure = pd.DataFrame([
        {"dimension": dimension, "category": category, "label": label, "count": count,
         "denominator": denominator, "percent": count / denominator * 100 if denominator else np.nan}
        for dimension, category, label, count, denominator in counts
    ])
    return pd.DataFrame(statistics), structure


def plot_cleaning_funnel(funnel: pd.DataFrame, clean: pd.DataFrame,
                         station: pd.DataFrame, rules: CleaningRules) -> plt.Figure:
    titles = {
        "missing_ride_id": "订单号缺失", "duplicate_ride_id": "重复订单号",
        "invalid_timestamp": "无效时间", "outside_analysis_month": "分析月份之外",
        "unsupported_rideable_type": "未知车型", "unsupported_member_type": "未知用户类型",
        "nonpositive_duration": "非正时长",
        "short_duration": f"不足 {rules.duration_min_inclusive:g} 分钟",
        "long_duration": f"超过 {rules.duration_max_inclusive:g} 分钟",
    }
    full = funnel.loc[funnel.scope.eq("full")]
    raw_count = int(full.iloc[0].remaining_rows)
    stages = full.loc[full.removed_rows.gt(0)]
    nodes = [("原始骑行记录", raw_count)] + [
        (f"去除{titles[row.stage]}后", int(row.remaining_rows)) for row in stages.itertuples()
    ]
    nodes[-1] = ("全量有效集" if len(nodes) > 1 else "全量有效集（原始记录均满足规则）", len(clean))
    figure = plt.figure(figsize=(12, max(8.8, len(nodes) * 1.6 + 2.4)))
    figure.text(0.06, 0.965, "图1  数据清洗与双子集分流", fontsize=21, weight="bold", va="top")
    figure.text(0.06, 0.905,
                f"原始 {raw_count:,} 条  ·  排除 {raw_count - len(clean):,} 条  ·  保留率 {len(clean) / raw_count:.2%}",
                fontsize=11, color="#657585", va="top")
    axis = figure.add_axes([0.04, 0.10, 0.92, 0.79])
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set_axis_off()
    centers = np.linspace(0.87, 0.35, len(nodes))
    height = min(0.115, 0.58 * (0.87 - 0.35) / max(len(nodes) - 1, 1))
    for index, ((title, count), center) in enumerate(zip(nodes, centers)):
        final = index == len(nodes) - 1
        fill, foreground = ("#315D8A", "white") if final else ("#F0F4F7", "#203548")
        axis.add_patch(FancyBboxPatch((0.30, center - height / 2), 0.40, height,
                                     boxstyle="round,pad=0.008,rounding_size=0.012",
                                     facecolor=fill, edgecolor="#D3DDE5", linewidth=0.8))
        axis.text(0.5, center + height * 0.21, title, ha="center", va="center", fontsize=11, color=foreground)
        axis.text(0.5, center - height * 0.17, f"{count:,}", ha="center", va="center", fontsize=23, weight="bold", color=foreground)
        if index:
            upper = centers[index - 1] - height / 2 - 0.01
            lower = center + height / 2 + 0.01
            axis.annotate("", xy=(0.5, lower), xytext=(0.5, upper),
                          arrowprops={"arrowstyle": "->", "color": "#8093A3", "lw": 1.4})
            row = stages.iloc[index - 1]
            axis.text(0.73, (upper + lower) / 2,
                      f"排除 {int(row.removed_rows):,} 条\n{titles[row.stage]}",
                      ha="left", va="center", fontsize=10, color="#A45145")
    axis.plot([0.5, 0.5], [centers[-1] - height / 2 - 0.01, 0.24], color="#8093A3", lw=1.4)
    axis.plot([0.25, 0.75], [0.24, 0.24], color="#8093A3", lw=1.4)
    for center, count, title, subtitle, color in (
        (0.25, len(station), "站点有效子集", "双端站点 ID 完整", "#315D8A"),
        (0.75, len(clean) - len(station), "缺站点分支", "仍保留在全量有效集中", "#159A85"),
    ):
        axis.annotate("", xy=(center, 0.188), xytext=(center, 0.24),
                      arrowprops={"arrowstyle": "->", "color": "#8093A3", "lw": 1.4})
        axis.add_patch(FancyBboxPatch((center - 0.21, 0.04), 0.42, 0.14,
                                     boxstyle="round,pad=0.008,rounding_size=0.012",
                                     facecolor="#F6F9FA", edgecolor=color, linewidth=1.3))
        axis.text(center, 0.147, title, ha="center", va="center", color=color, fontsize=12, weight="bold")
        axis.text(center, 0.105, f"{count:,}", ha="center", va="center", color=color, fontsize=24, weight="bold")
        axis.text(center, 0.063, subtitle, ha="center", va="center", fontsize=10, color="#657585")
    figure.text(0.06, 0.053, "分流后的两支互斥，合计为全量有效集；站点子集不与全量有效集相加。", fontsize=10)
    figure.text(0.06, 0.025,
                f"时长保留 {rules.duration_min_inclusive:g}–{rules.duration_max_inclusive:g} 分钟（含边界）；站点缺失不作为全城记录的删除条件。",
                fontsize=10, color="#657585")
    return figure


def plot_duration_distance(clean: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="row")
    figure.subplots_adjust(left=0.15, right=0.97, bottom=0.16, top=0.83, hspace=0.48, wspace=0.40)
    figure.text(0.06, 0.963, "图2  不同车型与用户类型的时长和距离分布", fontsize=21, weight="bold", va="top")
    figure.text(0.06, 0.905, f"全量有效集 n={len(clean):,}  ·  分组统计使用全部有效观测，不抽样", fontsize=11, color="#657585", va="top")
    dimensions = (("rideable_type", RIDEABLE_COLORS, "车型"), ("member_casual", MEMBER_COLORS, "用户类型"))
    row_limits = [0.0, 0.0]
    for column, (dimension, palette, group_title) in enumerate(dimensions):
        for row, (metric, label) in enumerate((("duration_min", "骑行时长（分钟）"), ("dist_km", "起终点直线距离（km）"))):
            axis = axes[row, column]
            values = [clean.loc[clean[dimension].eq(group), metric].dropna().to_numpy(dtype=float) for group in palette]
            labels = [f"{GROUP_LABELS[group]}\nn={len(sample):,}" for group, sample in zip(palette, values)]
            boxes = axis.boxplot(values, orientation="horizontal", tick_labels=labels, patch_artist=True,
                                 showfliers=False, whis=1.5, widths=0.50,
                                 medianprops={"color": "#142C3F", "linewidth": 2.1},
                                 boxprops={"linewidth": 1.1}, whiskerprops={"color": "#708595"},
                                 capprops={"color": "#708595"})
            for box, color in zip(boxes["boxes"], palette.values()):
                box.set(facecolor=color, edgecolor=color, alpha=0.82)
            for cap in boxes["caps"]:
                endpoints = np.asarray(cap.get_xdata(), dtype=float)
                if np.isfinite(endpoints).any():
                    row_limits[row] = max(row_limits[row], float(np.nanmax(endpoints)))
            axis.invert_yaxis()
            axis.set_title(f"({chr(97 + row * 2 + column)}) {group_title}与{'时长' if row == 0 else '距离'}", loc="left", pad=13)
            axis.set_xlabel(label)
            axis.grid(axis="y", visible=False)
    for row, maximum in enumerate(row_limits):
        axes[row, 0].set_xlim(0, maximum * 1.07 if maximum else 1)
    figure.text(0.06, 0.084, "箱体为 Q1–Q3，箱内线为中位数，须按 1.5×IQR 延伸；离群点未绘出，但参与全部分位数计算。", fontsize=10)
    figure.text(0.06, 0.049, "距离使用坐标可用子样本，并分别标注样本量；直线距离不等同实际骑行路程。", fontsize=10, color="#657585")
    return figure


def plot_trip_structure(clean: pd.DataFrame, structure: pd.DataFrame) -> plt.Figure:
    figure = plt.figure(figsize=(14, 7.8))
    figure.text(0.06, 0.963, "图3  站点记录与环线骑行结构", fontsize=21, weight="bold", va="top")
    figure.text(0.06, 0.902, "每个面板单独标注统计分母，站点信息不足的记录不直接归为非环线。", fontsize=11, color="#657585")
    known = int(clean.has_station_info.sum())
    unknown = len(clean) - known
    for position, dimension, title in (
        ([0.19, 0.31, 0.31, 0.43], "station_structure", f"(a) 站点记录结构\n全量有效集 n={len(clean):,}"),
        ([0.68, 0.31, 0.27, 0.43], "loop_structure", f"(b) 已知站点中的环线结构\n站点子集 n={known:,}"),
    ):
        axis = figure.add_axes(position)
        subset = structure.loc[structure.dimension.eq(dimension)]
        if not int(subset.denominator.iloc[0]):
            axis.text(0.5, 0.5, "站点信息不足，无法判断环线结构", ha="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        axis.barh(np.arange(len(subset)), subset.percent, height=0.52,
                  color=[STRUCTURE_COLORS[category] for category in subset.category])
        axis.set_yticks(np.arange(len(subset)), labels=subset.label)
        axis.invert_yaxis()
        axis.set_xlim(0, 100)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_xlabel("占该面板样本的比例（%）")
        axis.set_title(title, loc="left", pad=20)
        axis.grid(axis="y", visible=False)
        for position_y, item in enumerate(subset.itertuples()):
            inside = item.percent >= 55
            axis.text(item.percent - 2 if inside else item.percent + 1.8, position_y,
                      f"{item.percent:.2f}% · {item.count:,} 次", va="center",
                      ha="right" if inside else "left", color="white" if inside else "#203548", fontsize=10)
    figure.text(0.06, 0.18,
                f"环线可判定：{known:,} / {len(clean):,}（{known / len(clean):.2%}）；未知：{unknown:,}（{unknown / len(clean):.2%}）。",
                fontsize=11)
    figure.text(0.06, 0.112, "“疑似无桩电单车”指电单车且至少一端站点 ID 缺失，是代理标记，不能证明实际无桩运营形态。", fontsize=10, color="#657585")
    figure.text(0.06, 0.068, "异站骑行仅表示起终点站点 ID 不同，不直接等同通勤。", fontsize=10, color="#657585")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    arguments = parser.parse_args()
    font = set_plot_style()
    clean = read_clean_data(PROCESSED_DIR / "clean_trips.csv")
    station = read_clean_data(PROCESSED_DIR / "station_trips.csv")
    if clean.empty:
        raise ValueError("全量有效集为空，无法生成基础探索图表。")
    if not clean.loc[clean.has_station_info].reset_index(drop=True).equals(station):
        raise ValueError("两份清洗子集不一致，请先重新运行数据清洗。")
    if not clean.is_loop_trip.notna().eq(clean.has_station_info).all():
        raise ValueError("环线未知标记与站点完整性不一致。")
    if not clean.is_dockless_ebike.eq(clean.rideable_type.eq("electric_bike") & ~clean.has_station_info).all():
        raise ValueError("疑似无桩电单车标记不符合约定。")
    funnel = pd.read_csv(TABLE_DIR / "cleaning_funnel.csv")
    full_funnel = funnel.loc[funnel.scope.eq("full")]
    branch = funnel.loc[funnel.scope.eq("station_branch")]
    if (int(full_funnel.iloc[-1].remaining_rows) != len(clean)
            or int(branch.iloc[-1].remaining_rows) != len(station)
            or int(full_funnel.removed_rows.sum()) + len(clean) != int(full_funnel.iloc[0].remaining_rows)):
        raise ValueError("清洗漏斗与数据不一致，请先重新运行数据审查及清洗。")
    statistics, structure = build_summary_tables(clean)
    generators = (
        ("fig01_cleaning_funnel.png", lambda: plot_cleaning_funnel(funnel, clean, station, load_rules())),
        ("fig02_duration_distance_distribution.png", lambda: plot_duration_distance(clean)),
        ("fig03_trip_structure.png", lambda: plot_trip_structure(clean, structure)),
    )
    for filename, make_figure in generators:
        figure = make_figure()
        try:
            path = save_figure(figure, filename, directory=arguments.figure_dir)
            print(f"已保存：{path}")
        finally:
            plt.close(figure)
    statistics.to_csv(TABLE_DIR / "basic_eda_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    structure.to_csv(TABLE_DIR / "trip_structure_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"中文字体：{font}；图片分辨率：300 DPI；统计表已保存。")


if __name__ == "__main__":
    main()
