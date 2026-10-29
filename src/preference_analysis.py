"""陈宇涵 任务 2：会员、车型与小时需求的分组比较。

所有比例从同一份完整小时计数计算；百分比只在展示时乘以 100。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

from src.utils import save_figure


DAY_TYPES = ("工作日", "周末")
MEMBERS = ("member", "casual")
VEHICLES = ("classic_bike", "electric_bike")
MEMBER_NAMES = {"member": "会员", "casual": "散客"}
VEHICLE_NAMES = {"classic_bike": "经典车", "electric_bike": "电单车"}
VEHICLE_COLORS = {"classic_bike": "#315D8A", "electric_bike": "#159A85"}


@dataclass
class PreferenceResult:
    grid: pd.DataFrame
    member: pd.DataFrame
    vehicle: pd.DataFrame
    cross: pd.DataFrame
    mix: pd.DataFrame
    summary: pd.DataFrame
    validation: pd.DataFrame


def analyze_preferences(data: pd.DataFrame, hourly: pd.DataFrame,
                        day_counts: dict[str, int], base_day_type: pd.DataFrame) -> PreferenceResult:
    """先补齐小时×两个人群×两种车型，再统一聚合，保留零计数组合。"""
    for column, categories in (("member_casual", MEMBERS), ("rideable_type", VEHICLES)):
        if data[column].isna().any() or not data[column].isin(categories).all():
            raise ValueError(f"{column} 存在缺失或未识别类别，请先核对数据接口。")

    rides = data[["member_casual", "rideable_type"]].copy()
    rides["timestamp"] = data["started_at"].dt.floor("h")
    index = pd.MultiIndex.from_product(
        [hourly["timestamp"], MEMBERS, VEHICLES],
        names=["timestamp", "member_casual", "rideable_type"],
    )
    grid = rides.groupby(index.names, observed=True).size().reindex(index, fill_value=0)
    grid = grid.rename("ride_count").reset_index()
    grid["date"] = grid["timestamp"].dt.normalize()
    grid["hour"] = grid["timestamp"].dt.hour
    grid["day_type"] = np.where(grid["timestamp"].dt.dayofweek.ge(5), "周末", "工作日")

    def aggregate(categories: list[str], values: list[tuple[str, ...]]) -> pd.DataFrame:
        keys = ["day_type", "hour", *categories]
        complete = pd.MultiIndex.from_product([DAY_TYPES, range(24), *values], names=keys)
        table = grid.groupby(keys, observed=True)["ride_count"].sum().reindex(complete)
        table = table.rename("total_rides").reset_index()
        table["days"] = table["day_type"].map(day_counts).astype(int)
        table["daily_mean_rides"] = table["total_rides"] / table["days"]
        table["group_total_rides"] = table.groupby(["day_type", *categories])["total_rides"].transform("sum")
        # 分母为该类日期内该组的全天总次数；没有骑行的整组保持 NaN。
        denominator = table["group_total_rides"].where(table["group_total_rides"].gt(0))
        table["time_share"] = table["total_rides"] / denominator
        return table

    member = aggregate(["member_casual"], [MEMBERS])
    vehicle = aggregate(["rideable_type"], [VEHICLES])
    cross = aggregate(["member_casual", "rideable_type"], [MEMBERS, VEHICLES])
    mix = cross[["day_type", "hour", "member_casual", "rideable_type", "total_rides"]].copy()
    mix["member_hour_total"] = mix.groupby(["day_type", "hour", "member_casual"])["total_rides"].transform("sum")
    mix["vehicle_share"] = mix["total_rides"] / mix["member_hour_total"].where(mix["member_hour_total"].gt(0))

    rows = []
    for dimension, table, categories in (
        ("会员类型", member, ["member_casual"]),
        ("车辆类型", vehicle, ["rideable_type"]),
        ("会员×车型", cross, ["member_casual", "rideable_type"]),
    ):
        for keys, group in table.groupby(["day_type", *categories], sort=False):
            day_type, *levels = keys
            names = [MEMBER_NAMES[value] if column == "member_casual" else VEHICLE_NAMES[value]
                     for column, value in zip(categories, levels)]
            peak = float(group["daily_mean_rides"].max())
            peak_rows = group.loc[group["daily_mean_rides"].eq(peak)]
            total = int(group["total_rides"].sum())
            rows.append({
                "dimension": dimension, "day_type": day_type, "group": " · ".join(names),
                "total_rides": total, "days": int(group["days"].iloc[0]),
                "daily_mean_total": total / int(group["days"].iloc[0]),
                "peak_hours": "/".join(f"{h:02d}" for h in peak_rows["hour"]) if total else "无骑行",
                "peak_daily_mean": peak,
                "peak_time_share": float(peak_rows["time_share"].iloc[0]),
            })
    summary = pd.DataFrame(rows)
    validation = validate_preferences(data, hourly, base_day_type, day_counts,
                                      grid, member, vehicle, cross, mix)
    return PreferenceResult(grid, member, vehicle, cross, mix, summary, validation)


def validate_preferences(data: pd.DataFrame, hourly: pd.DataFrame, base_day_type: pd.DataFrame,
                         day_counts: dict[str, int], grid: pd.DataFrame, member: pd.DataFrame,
                         vehicle: pd.DataFrame, cross: pd.DataFrame, mix: pd.DataFrame) -> pd.DataFrame:
    """同时检查原始记录、交叉表边际、日历分母和两种不同的比例分母。"""
    checks = []

    def record(label: str, observed: int, expected: int = 0) -> None:
        checks.append({"核验项": label, "实际值": str(observed), "预期值": str(expected),
                       "结果": "通过" if observed == expected else "未通过"})

    record("会员与车型字段缺失或未知类别数", int((~data["member_casual"].isin(MEMBERS)).sum()
                                                   + (~data["rideable_type"].isin(VEHICLES)).sum()))
    record("小时×会员×车型网格数", len(grid), len(hourly) * 4)
    record("交叉小时网格骑行总次数", int(grid["ride_count"].sum()), len(data))
    calendar_counts = pd.Series(grid["date"].unique()).dt.dayofweek
    expected_days = {"工作日": int(calendar_counts.lt(5).sum()), "周末": int(calendar_counts.ge(5).sum())}
    record("分类天数与完整日历不一致数", sum(day_counts[key] != expected_days[key] for key in DAY_TYPES))
    aligned = grid.groupby("timestamp")["ride_count"].sum().reindex(hourly["timestamp"])
    record("交叉计数与任务1不一致的小时数", int((aligned.to_numpy() != hourly["ride_count"].to_numpy()).sum()))

    for title, table, categories, expected_rows in (
        ("会员", member, ["member_casual"], 96),
        ("车型", vehicle, ["rideable_type"], 96),
        ("四组合", cross, ["member_casual", "rideable_type"], 192),
    ):
        keys = ["day_type", "hour", *categories]
        record(f"{title}汇总行数", len(table), expected_rows)
        record(f"{title}汇总骑行总次数", int(table["total_rides"].sum()), len(data))
        # 用未补齐的原始骑行直接计数，与补齐后再聚合的结果逐格比较。
        direct = data.assign(day_type=np.where(data["started_at"].dt.dayofweek.ge(5), "周末", "工作日"))
        direct = direct.groupby(keys, observed=True).size().reindex(pd.MultiIndex.from_frame(table[keys]), fill_value=0)
        record(f"{title}汇总与原始记录不一致格数", int((direct.to_numpy() != table["total_rides"].to_numpy()).sum()))
        expected_denominator = table["day_type"].map(expected_days)
        invalid = table["days"].ne(expected_denominator) | ~np.isclose(
            table["daily_mean_rides"], table["total_rides"] / expected_denominator)
        record(f"{title}日均分母或结果异常格数", int(invalid.sum()))
        groups = ["day_type", *categories]
        totals = table.groupby(groups)["total_rides"].transform("sum")
        expected_share = table["total_rides"] / totals.where(totals.gt(0))
        valid = np.isclose(table["time_share"], expected_share, equal_nan=True)
        valid &= table["group_total_rides"].eq(totals)
        record(f"{title}时段占比分母或结果异常格数", int((~valid).sum()))
        sums = table.groupby(groups)["time_share"].sum(min_count=1)
        positive = table.groupby(groups)["total_rides"].sum().gt(0)
        record(f"{title}全天时段占比合计异常组数", int((~np.isclose(sums[positive], 1)).sum()))

    for title, table, categories in (("会员", member, ["member_casual"]),
                                     ("车型", vehicle, ["rideable_type"]),
                                     ("任务1日期类型", base_day_type.reset_index(), [])):
        keys = ["day_type", "hour", *categories]
        margin = cross.groupby(keys)["total_rides"].sum().reindex(pd.MultiIndex.from_frame(table[keys]))
        record(f"交叉表还原{title}异常格数", int((margin.to_numpy() != table["total_rides"].to_numpy()).sum()))

    record("车型构成汇总行数", len(mix), 192)
    membership_counts = member.set_index(["day_type", "hour", "member_casual"])["total_rides"]
    expected = membership_counts.reindex(pd.MultiIndex.from_frame(mix[["day_type", "hour", "member_casual"]]))
    record("车型构成分母与会员小时数不一致格数", int((expected.to_numpy() != mix["member_hour_total"].to_numpy()).sum()))
    expected_share = mix["total_rides"] / mix["member_hour_total"].where(mix["member_hour_total"].gt(0))
    record("车型构成比例异常格数", int((~np.isclose(mix["vehicle_share"], expected_share, equal_nan=True)).sum()))
    sums = mix.groupby(["day_type", "hour", "member_casual"])["vehicle_share"].sum(min_count=1)
    positive = mix.groupby(["day_type", "hour", "member_casual"])["total_rides"].sum().gt(0)
    record("两种车型占比合计异常组数", int((~np.isclose(sums[positive], 1)).sum()))
    record("零分母被显示为数值的格数", int(mix.loc[mix["member_hour_total"].eq(0), "vehicle_share"].notna().sum()))
    validation = pd.DataFrame(checks)
    if validation["结果"].ne("通过").any():
        raise ValueError("任务2统计核验未通过：" + "、".join(validation.loc[validation["结果"].ne("通过"), "核验项"]))
    return validation


def _plot_comparison(table: pd.DataFrame, series: list[tuple[dict, str, str, str]],
                     title: str, filename: str, directory: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.9), sharex=True)
    max_count = max(1.0, float(table["daily_mean_rides"].max()))
    shares = table["time_share"].dropna()
    max_share = max(0.01, float(shares.max())) if not shares.empty else 0.01
    handles = []
    for row, day_type in enumerate(DAY_TYPES):
        for column, metric in enumerate(("daily_mean_rides", "time_share")):
            ax = axes[row, column]
            for filters, label, color, linestyle in series:
                selected = table.loc[table["day_type"].eq(day_type)]
                for key, value in filters.items():
                    selected = selected.loc[selected[key].eq(value)]
                line, = ax.plot(selected["hour"], selected[metric], label=label,
                                color=color, linestyle=linestyle, linewidth=2.2)
                if row == 0 and column == 0:
                    handles.append(line)
            ax.set(title=f"{day_type} · {'日均次数' if column == 0 else '组内时段占比'}",
                   xlabel="小时", ylabel="次/天" if column == 0 else "时段占比",
                   xlim=(0, 23), xticks=[0, 3, 6, 9, 12, 15, 18, 21, 23],
                   ylim=(0, max_count * 1.12 if column == 0 else max_share * 1.12))
            ax.tick_params(labelbottom=True)
            ax.grid(axis="x", visible=False)
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}") if column == 0 else PercentFormatter(1, decimals=0))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.07, right=0.98, hspace=0.43, wspace=0.21)
    path = save_figure(fig, filename, directory=directory)
    plt.close(fig)
    return path


def _plot_mix(result: PreferenceResult, directory: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.3), sharex=True, sharey=True)
    for row, member in enumerate(MEMBERS):
        for column, day_type in enumerate(DAY_TYPES):
            ax = axes[row, column]
            table = result.mix.loc[result.mix["day_type"].eq(day_type) & result.mix["member_casual"].eq(member)]
            matrix = table.pivot(index="hour", columns="rideable_type", values="vehicle_share").reindex(range(24))
            bottom = np.zeros(24)
            for vehicle in VEHICLES:
                values = matrix[vehicle].to_numpy()
                ax.bar(range(24), values, bottom=bottom, width=0.9, label=VEHICLE_NAMES[vehicle],
                       color=VEHICLE_COLORS[vehicle], edgecolor="white", linewidth=0.4)
                bottom = bottom + values
            ax.set(title=f"{day_type} · {MEMBER_NAMES[member]}", xlabel="小时", ylabel="车型占比",
                   ylim=(0, 1), xlim=(-0.7, 23.7), xticks=[0, 3, 6, 9, 12, 15, 18, 21, 23])
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.grid(axis="x", visible=False)
    fig.suptitle("图 6C-2  各人群的逐小时车型构成", fontsize=16, fontweight="bold", y=0.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.07, right=0.98, hspace=0.43, wspace=0.17)
    path = save_figure(fig, "fig06c2_vehicle_mix_by_member.png", directory=directory)
    plt.close(fig)
    return path


def export_preferences(result: PreferenceResult, output: Path) -> dict[str, Path]:
    assets = {}
    for key, table, filename in (
        ("member", result.member, "fig06_membership_hourly.csv"),
        ("vehicle", result.vehicle, "fig06_vehicle_hourly.csv"),
        ("cross", result.cross, "fig06_member_vehicle_hourly.csv"),
        ("mix", result.mix, "fig06_vehicle_mix_by_member.csv"),
        ("summary", result.summary, "fig06_preference_summary.csv"),
    ):
        path = output / "tables" / filename
        table.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.8f", lineterminator="\n")
        assets[f"{key}_table"] = path
    assets["member_figure"] = _plot_comparison(result.member, [
        ({"member_casual": "member"}, "会员", "#315D8A", "-"),
        ({"member_casual": "casual"}, "散客", "#D78538", "--"),
    ], "图 6A  会员与散客的小时需求", "fig06a_membership_hourly.png", output / "figures")
    assets["vehicle_figure"] = _plot_comparison(result.vehicle, [
        ({"rideable_type": vehicle}, VEHICLE_NAMES[vehicle], VEHICLE_COLORS[vehicle], "-")
        for vehicle in VEHICLES
    ], "图 6B  经典车与电单车的小时需求", "fig06b_vehicle_hourly.png", output / "figures")
    assets["cross_figure"] = _plot_comparison(result.cross, [
        ({"member_casual": member, "rideable_type": vehicle},
         f"{MEMBER_NAMES[member]} · {VEHICLE_NAMES[vehicle]}", VEHICLE_COLORS[vehicle],
         "-" if member == "member" else "--") for member in MEMBERS for vehicle in VEHICLES
    ], "图 6C-1  会员与车型的交叉比较", "fig06c1_member_vehicle_hourly.png", output / "figures")
    assets["mix_figure"] = _plot_mix(result, output / "figures")
    return assets


def _group_row(result: PreferenceResult, dimension: str, day_type: str, group: str) -> pd.Series:
    return result.summary.loc[result.summary["dimension"].eq(dimension)
                              & result.summary["day_type"].eq(day_type)
                              & result.summary["group"].eq(group)].iloc[0]


def _electric_share(result: PreferenceResult, day_type: str, member: str) -> float:
    group = result.cross.loc[result.cross["day_type"].eq(day_type) & result.cross["member_casual"].eq(member)]
    total = int(group["total_rides"].sum())
    return float(group.loc[group["rideable_type"].eq("electric_bike"), "total_rides"].sum() / total) if total else np.nan


def preference_notes(result: PreferenceResult) -> dict[str, list[str]]:
    notes = {key: [] for key in ("member", "vehicle", "cross", "mix")}
    for day_type in DAY_TYPES:
        for key, dimension, names in (("member", "会员类型", ("会员", "散客")),
                                       ("vehicle", "车辆类型", ("经典车", "电单车"))):
            first, second = [_group_row(result, dimension, day_type, name) for name in names]
            notes[key].append(
                f"{day_type}：{names[0]}、{names[1]}日均分别为 {first.daily_mean_total:,.1f}、{second.daily_mean_total:,.1f} 次，"
                f"最高小时分别为 {first.peak_hours} 时、{second.peak_hours} 时。")
        groups = result.summary.loc[result.summary["dimension"].eq("会员×车型") & result.summary["day_type"].eq(day_type)]
        largest = groups.loc[groups["daily_mean_total"].idxmax()]
        notes["cross"].append(f"{day_type}：四种组合中，{largest['group']}的日均总量最高，为 {largest.daily_mean_total:,.1f} 次；最高小时为 {largest.peak_hours} 时。")
        member_share, casual_share = [_electric_share(result, day_type, member) for member in MEMBERS]
        difference = (casual_share - member_share) * 100
        notes["mix"].append(f"{day_type}：会员、散客骑行中的电单车占比分别为 {member_share:.1%}、{casual_share:.1%}，"
                            f"散客比会员{'高' if difference >= 0 else '低'} {abs(difference):.1f} 个百分点。")
    return notes


def _distribution_note(table: pd.DataFrame, category: str, first: str, second: str,
                       names: tuple[str, str]) -> str:
    pieces = []
    for day_type in DAY_TYPES:
        matrix = table.loc[table["day_type"].eq(day_type)].pivot(index="hour", columns=category, values="time_share")
        if matrix[[first, second]].isna().any().any():
            continue
        gap = matrix[first] - matrix[second]
        hour = int(gap.abs().idxmax())
        pieces.append(f"{day_type}差异最大在 {hour:02d} 时，{names[0]}与{names[1]}的时段占比分别为 "
                      f"{matrix.loc[hour, first]:.1%}、{matrix.loc[hour, second]:.1%}（相差 {abs(gap.loc[hour]) * 100:.1f} 个百分点）")
    return "；".join(pieces) + "。" if pieces else "该组无骑行时，时段占比不计算。"


def preference_highlights(result: PreferenceResult) -> list[str]:
    totals = result.member.groupby("member_casual")["total_rides"].sum()
    distribution = result.member.loc[result.member["day_type"].eq("工作日")].pivot(
        index="hour", columns="member_casual", values="time_share")
    hour = int((distribution["member"] - distribution["casual"]).abs().idxmax())
    vehicles = result.vehicle.groupby("rideable_type")["total_rides"].sum()
    classic = _group_row(result, "车辆类型", "周末", "经典车")
    electric = _group_row(result, "车辆类型", "周末", "电单车")
    weekday_difference = (_electric_share(result, "工作日", "casual") - _electric_share(result, "工作日", "member")) * 100
    return [
        f"会员占全部骑行的 {totals['member'] / totals.sum():.1%}；工作日 {hour:02d} 时在会员、散客全天骑行中的占比分别为 {distribution.loc[hour, 'member']:.1%}、{distribution.loc[hour, 'casual']:.1%}。",
        f"两种车型总量接近（电单车占 {vehicles['electric_bike'] / vehicles.sum():.1%}）；周末经典车最高在 {classic.peak_hours} 时，电单车最高在 {electric.peak_hours} 时。",
        f"工作日，散客骑行中的电单车占比比会员{'高' if weekday_difference >= 0 else '低'} {abs(weekday_difference):.1f} 个百分点。",
    ]


def _data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _download(path: Path, label: str = "下载 CSV") -> str:
    return f'<a class="download" href="{_data_url(path, "text/csv")}" download="{escape(path.name)}">{escape(label)}</a>'


def _display_table(table: pd.DataFrame) -> str:
    display = table.copy()
    if "member_casual" in display:
        display["member_casual"] = display["member_casual"].map(MEMBER_NAMES)
    if "rideable_type" in display:
        display["rideable_type"] = display["rideable_type"].map(VEHICLE_NAMES)
    for column in ("time_share", "vehicle_share", "peak_time_share"):
        if column in display:
            display[column] = display[column].map(lambda value: "无数据" if pd.isna(value) else f"{value:.1%}")
    display = display.rename(columns={
        "day_type": "日期类型", "hour": "小时", "member_casual": "会员类型", "rideable_type": "车型",
        "total_rides": "总次数", "days": "天数", "daily_mean_rides": "日均次数",
        "group_total_rides": "该组全天总次数", "time_share": "组内时段占比",
        "member_hour_total": "该人群该小时总次数", "vehicle_share": "该小时车型占比",
        "dimension": "分析维度", "group": "分组", "daily_mean_total": "日均总量",
        "peak_hours": "最高小时", "peak_daily_mean": "最高小时日均次数", "peak_time_share": "最高小时时段占比",
    })
    return display.to_html(index=False, border=0, float_format=lambda value: f"{value:,.1f}", na_rep="无数据")


def preference_html(result: PreferenceResult, assets: dict[str, Path]) -> str:
    notes = preference_notes(result)
    sections = []
    specifications = (
        ("member", "members", "图 6A · 会员与散客", "会员类型", result.member,
         "日均次数看需求规模，组内占比看一天中的使用分布。",
         _distribution_note(result.member, "member_casual", "member", "casual", ("会员", "散客"))),
        ("vehicle", "vehicles", "图 6B · 经典车与电单车", "车辆类型", result.vehicle,
         "两种车型分别归一化，比较时段分布时兼顾总量差异。",
         _distribution_note(result.vehicle, "rideable_type", "classic_bike", "electric_bike", ("经典车", "电单车"))),
        ("cross", "cross", "图 6C · 会员与车型交叉分析", "会员×车型", result.cross,
         "将人群与车型交叉分组，进一步查看总体差异来自哪些组合。", ""),
    )
    caption = "注：按出发时间计数；工作日、周末分别除以对应天数。右列占比的分母是该类日期内该组的全天总次数，每条曲线的 24 小时合计为 100%；上下行采用相同纵轴范围。"
    for key, anchor, title, dimension, table, introduction, distribution in specifications:
        peak_table = result.summary.loc[result.summary["dimension"].eq(dimension)]
        findings = "".join(f"<p>{escape(note)}</p>" for note in notes[key])
        extra = ""
        if key == "cross":
            mix_notes = "".join(f"<p>{escape(note)}</p>" for note in notes["mix"])
            extra = f'''<div class="insight">{mix_notes}</div>
<details><summary>展开人群内部的车型构成 · 图 6C-2</summary>
<figure><img src="{_data_url(assets['mix_figure'], 'image/png')}" alt="工作日与周末，会员及散客各小时的两种车型构成"><figcaption>注：每根柱表示某类人群在该小时的全部骑行，两种车型合计为 100%。分母为零时留空；本图描述骑行记录的车型构成。</figcaption></figure>
<p class="muted">上面的整体占比按总次数计算，未对 24 个小时的百分比作简单平均。</p>
<details><summary>车型构成明细与下载</summary><div class="scroll">{_display_table(result.mix)}</div>{_download(assets['mix_table'])}</details></details>'''
        sections.append(f'''<section id="{anchor}"><h2>{title}</h2><p class="muted">{introduction}</p>
<figure><img src="{_data_url(assets[key + '_figure'], 'image/png')}" alt="{escape(title)}，工作日和周末的日均次数及组内时段占比"><figcaption>{caption}{' 交叉图用颜色区分车型，实线为会员，虚线为散客。' if key == 'cross' else ''}</figcaption></figure>
<div class="insight">{findings}</div>
{f'<details><summary>时段分布的量化比较</summary><p>{escape(distribution)}</p></details>' if distribution else ''}
{extra}
<details><summary>峰值摘要、逐小时统计与下载</summary><div class="scroll">{_display_table(peak_table)}</div><div class="scroll">{_display_table(table)}</div>{_download(assets[key + '_table'])}</details></section>''')
    return "\n".join(sections)


def preference_markdown(result: PreferenceResult) -> str:
    notes = preference_notes(result)
    lines = ["## 5.5 会员与车型的时段差异", "",
             "使用 `member_casual`（member=会员、casual=散客）与 `rideable_type`（classic_bike=经典车、electric_bike=电单车）。两个字段均通过缺失与类别核对；沿用全量有效骑行，分别分析工作日和周末。", "",
             "### 统计方法与三个分母", "",
             "先用批量日期运算与 `groupby` 计数，再用完整索引 `reindex` 对齐日期×小时×会员×车型。31×24×2×2=2,976 个组合全部保留，零记录组合计为 0。日均分母来自完整日历，而不是某组有骑行的日期数。", "",
             "- **日均每小时次数**＝该组该小时总次数÷对应日期类型的天数，比较需求规模。",
             "- **组内时段占比**＝该组该小时总次数÷该组在对应日期类型内的全天总次数，比较一天中的分布；各组 24 小时合计为 100%。",
             "- **该小时车型占比**＝该人群该小时某车型次数÷该人群该小时两种车型总次数，比较车型构成；每个人群、每个小时的两车型合计为 100%。",
             "- 分母为零时保留缺失值，HTML 显示“无数据”，图中留空；不把无法计算的比例填成 0。", "",
             "会员与散客总量不同，日均次数和组内时段占比需要配合阅读。整体车型占比用总次数之比计算，避免低需求小时与高需求小时被赋予相同权重。", ""]
    for key, title, filename, reason in (
        ("member", "图 6A：会员与散客", "fig06a_membership_hourly.png", "折线保留小时顺序，左右分开展示次数与占比；上下分别为工作日和周末，使用相同的纵轴范围。"),
        ("vehicle", "图 6B：经典车与电单车", "fig06b_vehicle_hourly.png", "沿用相同布局，观察需求规模与归一化曲线是否给出不同的信息。"),
        ("cross", "图 6C-1：四种组合", "fig06c1_member_vehicle_hourly.png", "交叉分组检查总体差异；颜色区分车型，线型区分会员与散客，减少同时辨认四种颜色的负担。"),
        ("mix", "图 6C-2：人群内部的车型构成", "fig06c2_vehicle_mix_by_member.png", "百分比堆叠柱展示每小时的整体与部分关系；按人群、日期类型拆成四个子图。"),
    ):
        lines += [f"### {title}", "", reason, "", f"![{title}](../figures/{filename})", ""]
        lines += [f"- {note}" for note in notes[key]]
        if key in ("member", "vehicle"):
            lines += ["", _distribution_note(result.member, "member_casual", "member", "casual", ("会员", "散客"))
                      if key == "member" else _distribution_note(result.vehicle, "rideable_type", "classic_bike", "electric_bike", ("经典车", "电单车"))]
        lines += [""]
    lines += ["### 导出与结果核验", "",
              "逐小时 CSV 保留 `total_rides`、`days`、`daily_mean_rides`、`group_total_rides`、`time_share`；车型构成表保留 `member_hour_total`、`vehicle_share`。比例保存为 0–1 小数，展示时转换成百分比。`fig06_preference_summary.csv` 给出各组日均总量、最高小时、最高小时次数与占比；并列最高小时全部保留。", "",
              "核验原始骑行与补齐计数逐格一致、四组合边际可还原会员/车型/任务1汇总、日历分母正确、占比合计正确。图、表和文字共用同一份统计结果。", "",
              "这些差异描述本月骑行记录的使用分布。记录不含用户唯一标识，无法计算独立用户人数；日均曲线也会掩盖不同日期间的波动。", ""]
    return "\n".join(lines)
