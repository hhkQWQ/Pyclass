"""陈宇涵 任务 1、2：时间规律、会员与车型差异，以及可离线阅读的 HTML。

在项目根目录运行：python -m src.temporal_analysis
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from html import escape
from pathlib import Path
from string import Template
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

# 同时支持 python -m src.temporal_analysis 和 python src/temporal_analysis.py。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import read_clean_data
from src.utils import PROJECT_ROOT, PROCESSED_DIR, REPORT_DIR, save_figure, set_plot_style
from src.preference_analysis import (
    PreferenceResult, analyze_preferences, export_preferences,
    preference_highlights, preference_html, preference_markdown,
)


MONTH_START = pd.Timestamp("2024-01-01")
MONTH_END = pd.Timestamp("2024-02-01")
WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
DAY_TYPES = ("工作日", "周末")
SOURCE_COMMIT = "777c65867a129d6f0dba03c0acfefbbce1e98cae"
SOURCE_URL = f"https://github.com/hhkQWQ/Pyclass/tree/{SOURCE_COMMIT}"


@dataclass
class TemporalResult:
    data: pd.DataFrame
    hourly: pd.DataFrame
    day_type: pd.DataFrame
    weekday: pd.DataFrame
    counts: dict[str, int]


def analyze_temporal(data: pd.DataFrame) -> TemporalResult:
    """核对时间分析输入，补齐日历后求均值，保留全部有效骑行。"""
    if data.empty:
        raise ValueError("有效集为空，无法开展时间分析。")
    if data["ride_id"].isna().any() or not data["ride_id"].is_unique:
        raise ValueError("订单编号缺失或重复，请先核对输入有效集。")
    started = data["started_at"]
    if started.isna().any() or not started.between(MONTH_START, MONTH_END, inclusive="left").all():
        raise ValueError("开始时间缺失或超出 2024 年 1 月，请核对数据范围。")
    derived = {
        "hour": started.dt.hour,
        "weekday": started.dt.dayofweek,
        "is_weekend": started.dt.dayofweek.ge(5),
    }
    for name, expected in derived.items():
        if data[name].isna().any() or not data[name].eq(expected).all():
            raise ValueError(f"{name} 与 started_at 不一致，请先核对清洗接口。")

    dates = pd.date_range(MONTH_START, MONTH_END, inclusive="left", freq="D")
    observed_dates = pd.DatetimeIndex(started.dt.normalize().unique())
    missing_dates = dates.difference(observed_dates)
    if len(missing_dates):
        missing = ", ".join(missing_dates.strftime("%Y-%m-%d"))
        raise ValueError(f"以下日期整天没有记录，需先确认是否缺数据，不能直接当作零需求：{missing}")

    # 按时间标签对齐到整月小时格；这里的 0 是订单计数，不用于填补站点或坐标。
    calendar = pd.date_range(MONTH_START, MONTH_END, inclusive="left", freq="h")
    counts = started.dt.floor("h").value_counts().reindex(calendar, fill_value=0).sort_index()
    hourly = counts.rename_axis("timestamp").rename("ride_count").reset_index()
    hourly["date"] = hourly["timestamp"].dt.normalize()
    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly["weekday"] = hourly["timestamp"].dt.dayofweek
    hourly["day_type"] = np.where(hourly["weekday"].ge(5), "周末", "工作日")

    # 明确分组键、计数列与聚合方式，分母来自完整日历而非有骑行的天数。
    day_type = hourly.groupby(["day_type", "hour"], sort=False).agg(
        total_rides=("ride_count", "sum"), days=("date", "nunique")
    ).reindex(pd.MultiIndex.from_product([DAY_TYPES, range(24)], names=["day_type", "hour"]))
    weekday = hourly.groupby(["weekday", "hour"], sort=False).agg(
        total_rides=("ride_count", "sum"), days=("date", "nunique")
    ).reindex(pd.MultiIndex.from_product([range(7), range(24)], names=["weekday", "hour"]))
    for table in (day_type, weekday):
        table["daily_mean_rides"] = table["total_rides"] / table["days"]

    # 独立从骑行记录计数，检查补零和两个分组过程是否守恒。
    direct_hours = started.dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
    if not np.array_equal(hourly.groupby("hour")["ride_count"].sum(), direct_hours):
        raise ValueError("逐日补齐后，小时计数与原始记录不一致。")
    for table, expected_shape in ((day_type, (48, 3)), (weekday, (168, 3))):
        if table.shape != expected_shape or int(table["total_rides"].sum()) != len(data):
            raise ValueError("分组表形状或骑行总数不一致。")
        if not np.isfinite(table["daily_mean_rides"]).all() or (table["days"] <= 0).any():
            raise ValueError("日均值或日历分母无效。")
    day_counts = {name: int(day_type.loc[name, "days"].iloc[0]) for name in DAY_TYPES}
    if len(hourly) != 744 or sum(day_counts.values()) != 31:
        raise ValueError("2024 年 1 月日历应为 31 天、744 小时。")
    return TemporalResult(data, hourly, day_type, weekday, day_counts)


def build_validation_summary(result: TemporalResult) -> pd.DataFrame:
    """把本步骤的输入与统计检查保存为可复核记录；不以参考行数决定通过。"""
    rows = []

    def record(label: str, observed: int | str, expected: int | str) -> None:
        rows.append({"核验项": label, "实际值": str(observed), "预期值": str(expected),
                     "结果": "通过" if observed == expected else "未通过"})

    data, hourly = result.data, result.hourly
    started = data["started_at"]
    dates = pd.date_range(MONTH_START, MONTH_END, inclusive="left", freq="D")
    record("关键订单与时间字段缺失数", int(data[["ride_id", "started_at", "hour", "weekday", "is_weekend"]].isna().sum().sum()), 0)
    record("不重复的订单编号数", int(data["ride_id"].nunique()), len(data))
    record("分析月份外的记录数", int((~started.between(MONTH_START, MONTH_END, inclusive="left")).sum()), 0)
    mismatch = sum(int(data[name].ne(expected).sum()) for name, expected in {
        "hour": started.dt.hour, "weekday": started.dt.dayofweek, "is_weekend": started.dt.dayofweek.ge(5),
    }.items())
    record("时间派生字段不一致数", mismatch, 0)
    record("有记录的日期数", int(started.dt.normalize().nunique()), len(dates))
    record("整月小时格数", len(hourly), len(dates) * 24)
    expected_types = [int((dates.dayofweek < 5).sum()), int((dates.dayofweek >= 5).sum())]
    type_days = [int(result.day_type.loc[name, "days"].iloc[0]) for name in DAY_TYPES]
    record("工作日/周末天数", "/".join(map(str, type_days)), "/".join(map(str, expected_types)))
    weekday_days = [int(result.weekday.loc[weekday, "days"].iloc[0]) for weekday in range(7)]
    expected_weekdays = [int((dates.dayofweek == weekday).sum()) for weekday in range(7)]
    record("周一至周日的天数", "/".join(map(str, weekday_days)), "/".join(map(str, expected_weekdays)))
    record("图4汇总行数", len(result.day_type), 2 * 24)
    record("图5汇总行数", len(result.weekday), 7 * 24)
    record("小时表的骑行总次数", int(hourly["ride_count"].sum()), len(data))
    record("图4的骑行总次数", int(result.day_type["total_rides"].sum()), len(data))
    record("图5的骑行总次数", int(result.weekday["total_rides"].sum()), len(data))
    direct = started.dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()
    record("补齐前后计数不同的小时数", int(hourly.groupby("hour")["ride_count"].sum().ne(direct).sum()), 0)
    invalid_means = 0
    for table, calendar_days in ((result.day_type, dict(zip(DAY_TYPES, expected_types))),
                                 (result.weekday, dict(enumerate(expected_weekdays)))):
        expected_denominators = np.array([calendar_days[key] for key in table.index.get_level_values(0)])
        valid = (table["days"].to_numpy() == expected_denominators)
        valid &= np.isfinite(table["daily_mean_rides"].to_numpy())
        valid &= np.isclose(table["daily_mean_rides"], table["total_rides"] / expected_denominators)
        invalid_means += int((~valid).sum())
    record("分母或日均值异常的汇总行数", invalid_means, 0)
    summary = pd.DataFrame(rows)
    failed = summary.loc[summary["结果"].ne("通过"), "核验项"]
    if not failed.empty:
        raise ValueError("统计核验未通过：" + "、".join(failed))
    return summary


def hour_span(hour: int) -> str:
    return f"{hour:02d}:00–{hour + 1:02d}:00"


def main_peak_hours(means: pd.Series) -> list[int]:
    """标注最高的两个严格局部峰值；始终包含全日最高小时。"""
    local = [h for h in range(1, 23) if means.loc[h] > means.loc[h - 1] and means.loc[h] > means.loc[h + 1]]
    candidates = set(local) | {int(means.idxmax())}
    return sorted(sorted(candidates, key=lambda h: (-float(means.loc[h]), h))[:2])


def findings(result: TemporalResult) -> list[str]:
    means = result.day_type["daily_mean_rides"]
    work, weekend = means.loc["工作日"], means.loc["周末"]
    weekend_peak = int(weekend.idxmax())
    work_peaks = "、".join(f"{hour:02d} 时（{work.loc[hour]:,.1f} 次/天）" for hour in main_peak_hours(work))
    weekday_peak = result.weekday["daily_mean_rides"].idxmax()
    weekday_value = float(result.weekday.loc[weekday_peak, "daily_mean_rides"])
    return [
        f"工作日主要峰值为 {work_peaks}；"
        f"周末最高小时为 {hour_span(weekend_peak)}，日均 {weekend.loc[weekend_peak]:,.1f} 次。",
        f"工作日日均总量 {work.sum():,.1f} 次，周末 {weekend.sum():,.1f} 次；"
        f"周末约为工作日的 {weekend.sum() / work.sum():.1%}。",
        f"星期与小时组合中，{WEEKDAYS[weekday_peak[0]]} {hour_span(weekday_peak[1])} 最强，"
        f"日均 {weekday_value:,.1f} 次。",
    ]


def plot_day_types(result: TemporalResult, directory: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 4.9), layout="constrained")
    colors = ("#315D8A", "#159A85")
    for name, color in zip(DAY_TYPES, colors):
        means = result.day_type.loc[name, "daily_mean_rides"]
        ax.plot(means.index, means, color=color, linewidth=2.5, marker="o", markersize=4,
                label=f"{name}（{result.counts[name]} 天）")
        peaks = main_peak_hours(means) if name == "工作日" else [int(means.idxmax())]
        for peak in peaks:
            ax.scatter([peak], [means.loc[peak]], color=color, s=60, zorder=4)
            ax.annotate(f"{peak:02d} 时 · {means.loc[peak]:,.1f}", xy=(peak, means.loc[peak]),
                        xytext=(8, 15 if name == "工作日" else -24), textcoords="offset points",
                        color=color, fontsize=10, fontweight="bold")
    ax.set(title="图 4  工作日与周末的小时骑行需求", xlabel="小时", ylabel="日均骑行次数（次/天）",
           xticks=range(24), xlim=(-0.4, 23.7), ylim=(0, None))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.margins(y=0.17)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False)
    path = save_figure(fig, "fig04_workday_weekend_hourly.png", directory=directory)
    plt.close(fig)
    return path


def plot_weekday_heatmap(result: TemporalResult, directory: Path) -> Path:
    matrix = result.weekday["daily_mean_rides"].unstack("hour")
    fig, ax = plt.subplots(figsize=(12.2, 4.7), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("demand", ["#F0F6FA", "#B2D8D2", "#438E9C", "#214D78"])
    image = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=0, aspect="auto", interpolation="nearest")
    ax.grid(False)
    ax.set(title="图 5  星期 × 小时骑行需求", xlabel="小时", ylabel="星期",
           xticks=range(24), yticks=range(7), yticklabels=WEEKDAYS)
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.032)
    bar.set_label("日均骑行次数（次/天）")
    peak_weekday, peak_hour = result.weekday["daily_mean_rides"].idxmax()
    ax.add_patch(plt.Rectangle((peak_hour - 0.5, peak_weekday - 0.5), 1, 1,
                               fill=False, edgecolor="#D78538", linewidth=2))
    path = save_figure(fig, "fig05_weekday_hour_heatmap.png", directory=directory)
    plt.close(fig)
    return path


def csv_data_url(path: Path) -> str:
    return "data:text/csv;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def input_fingerprint(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "未读取到本地提交号"


def write_markdown(result: TemporalResult, validation: pd.DataFrame, path: Path,
                   source: Path, fingerprint: str, commit: str,
                   preferences: PreferenceResult) -> None:
    """生成简要方法说明；图表相对路径适用于 reports/sections。"""
    notes = "\n".join(f"- {text}" for text in findings(result))
    min_time, max_time = result.data["started_at"].min(), result.data["started_at"].max()
    station_missing = int((~result.data["has_station_info"]).sum())
    text = f"""# 5 时间规律与会员、车型差异

**负责人：陈宇涵**

## 5.1 数据是什么

本部分研究一天和一周内的骑行需求变化。输入为 `data/processed/clean_trips.csv`：**{len(result.data):,} 行、{len(result.data.columns)} 列**，一行代表一次骑行。开始时间从 {min_time} 到 {max_time}，覆盖 2024 年 1 月的 31 天。

| 字段分组 | 字段 | 简单说明 |
|---|---|---|
| 骑行标识 | `ride_id` | 文本，每次骑行的唯一编号 |
| 车辆与用户 | `rideable_type`、`member_casual` | 文本，车辆类型和会员/临时用户类型 |
| 起止时间 | `started_at`、`ended_at` | 日期时间，何时出发和结束 |
| 站点 | `start_station_name`、`start_station_id`、`end_station_name`、`end_station_id` | 文本，起终点站名和编号；编号保留为文本 |
| 坐标 | `start_lat`、`start_lng`、`end_lat`、`end_lng` | 数值，起终点经纬度，部分可缺失 |
| 时长与距离 | `duration_min`、`dist_km` | 数值，分钟和起终点直线距离（公里），距离不是实际骑行路线长度 |
| 时间特征 | `hour`、`weekday`、`is_weekend` | 小时 0–23、星期 0–6（周一=0）、是否周末 |
| 其他标记 | `has_station_info`、`is_dockless_ebike`、`is_loop_trip` | 是否具备起终点站 ID、缺站点电单车代理标记、是否同站往返 |

**数据选择与检查**：分析对象是全城有效骑行，统计使用订单和出发时间字段。保留站点 ID 不完整但时间有效的 {station_missing:,} 次骑行（占 {station_missing / len(result.data):.1%}），避免因无关字段缺失而低估需求。读入后核对字段类型、关键字段缺失、订单唯一性和日期范围；站点、坐标等字段不参与本步需求计数，也不统一填零。订单次数不代表独立用户人数。

## 5.2 怎么统计，为什么这样做

采用描述性统计与分组对比，依次回答“工作日和周末有什么差异”“一周哪些时段更繁忙”。用日期时间列的批量运算提取时间信息，`groupby` 分组汇总，`reindex` 对齐完整日历，计算过程可重复、可核对。

- **按出发时间计数**：一次骑行计入 `started_at` 所在小时，衡量出发需求；沿用数据时间，不转换时区。
- **按星期分类**：周一至周五为工作日，周六、周日为周末；未额外处理当地公共假日。
- **区分零记录与缺数据**：核对 31 天都有记录后，构建 31 × 24 = 744 个小时格；按订单计数口径，无记录的小时记 0，并纳入均值分母。若整天无记录，先停止检查数据完整性，避免直接把缺数据当作零需求。
- **比较日均量**：工作日 {result.counts['工作日']} 天、周末 {result.counts['周末']} 天，不能直接用总量比较。日均每小时次数 = 该类日期该小时的总次数 ÷ 该类日期天数。
- **星期分别计算**：周一至周三各 5 天，周四至周日各 4 天；热力图每格也除以对应星期的天数。

## 5.3 为什么选这两种图

**图 4 用折线图**：小时有先后顺序，折线便于对比日内变化和两类日期的差异。横轴为 0–23 时，纵轴为日均骑行次数，两组使用相同单位和坐标。直接绘制分组均值，保留实际峰值；标注工作日主要峰值及周末最高小时。

![图 4：工作日与周末小时需求](../figures/fig04_workday_weekend_hourly.png)

**图 5 用热力图**：同时查看星期与小时两个维度，比逐条比较曲线更容易定位繁忙时段。行是星期、列是小时，每格为日均需求次数；色标从 0 起，颜色越深需求越高，橙框标记最高格。两张图都配有明确单位和可核对的读数。

![图 5：星期与小时需求热力图](../figures/fig05_weekday_hour_heatmap.png)

## 5.4 实际结果与限制

{notes}

这些结果为后续需求预测和调度讨论提供重点观察时段。分组差异只能说明分布不同，通勤、天气或假日等原因仍需其他资料验证。结果仅代表 2024 年 1 月，日均值也会掩盖日期间的波动；每天有记录并不能完全排除局部漏采。

{preference_markdown(preferences)}

## 5.6 如何复现与核对

在项目根目录、已按 `requirements.txt` 配置的 Python 环境中运行：

```powershell
python -m src.temporal_analysis
```

直接运行 `python src/temporal_analysis.py` 也可；确保使用 Pyclass 的 Python。脚本将统计、核验、绘图和导出分为独立函数，图与表共用一份统计结果。输出包括离线 HTML、图 4/5、图 6A/6B/6C-1/6C-2，以及对应 CSV。图 4/5 的数值保留四位小数，图 6 的数值保留八位小数；百分比在 CSV 中保存为 0–1 小数。

本次 **{len(validation)} 项统计核验全部通过**：关键字段、日期覆盖、小时补齐、分组分母及总次数均一致。实际值与预期值保存在 [统计核验表](../tables/temporal_validation_summary.csv)，HTML 也可展开查看和下载。两张汇总表的总次数均为 {len(result.data):,}；核对依据是实际输入行数，不以固定参考行数作为筛选目标。

方法对应教材中的向量化、分组聚合、索引对齐、函数复用与按问题选图；核验用于检查本步统计结果的可靠性。

输入版本：[项目数据版本]({SOURCE_URL})；本地基线提交：`{commit}`。输入文件：`{source.name}`；SHA-256：`{fingerprint}`。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


HTML_TEMPLATE = Template("""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>共享单车 · 时间与使用差异</title><style>
:root{--ink:#203548;--muted:#596e7c;--blue:#315d8a;--green:#159a85;--line:#dbe4e9}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f6f8;color:var(--ink);font:16px/1.75 "Microsoft YaHei","PingFang SC",sans-serif}
main{max-width:1180px;margin:0 auto;padding:38px 28px 24px}header{margin-bottom:22px}h1{font-size:32px;line-height:1.35;margin:0 0 10px}h2{font-size:22px;line-height:1.45;margin:0 0 10px}h3{font-size:17px;margin:18px 0 8px}p{margin:8px 0}.muted{color:var(--muted)}.meta{font-size:13px;color:var(--muted)}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 0}nav a{background:white;border:1px solid var(--line);border-radius:20px;padding:5px 15px;text-decoration:none;font-size:14px}
section{background:white;border:1px solid var(--line);border-radius:14px;padding:25px;margin:18px 0;scroll-margin-top:16px}.findings{display:grid;grid-template-columns:1fr 1fr;gap:12px}.finding{padding:14px 17px;border-radius:9px;background:#f0f7f6;border-left:3px solid var(--green);font-size:15px}.insight{margin:15px 0 8px;padding:10px 16px;background:#f0f7f6;border-left:3px solid var(--green);font-size:14px}
figure{margin:16px 0 8px}figure img{display:block;width:100%;height:auto}figcaption{color:var(--muted);font-size:12px;line-height:1.6;margin-top:9px}
details{border-top:1px solid var(--line);margin-top:16px;padding-top:12px}summary{cursor:pointer;color:var(--blue);font-weight:600;font-size:14px}details[open]>summary{margin-bottom:12px}.scroll{overflow:auto;max-height:430px;margin:14px 0;overscroll-behavior:contain}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;text-align:right}th,td{padding:7px 10px;border-bottom:1px solid #e7edf1;white-space:nowrap}th{background:#eef4f7;color:var(--blue);position:sticky;top:0}td:first-child,th:first-child{text-align:left}
a{color:var(--blue);text-underline-offset:3px}.download{display:inline-block;border:1px solid var(--line);padding:4px 12px;border-radius:6px;font-size:13px;text-decoration:none;margin:2px 6px 2px 0}ul{padding-left:22px}dt{font-weight:600;margin-top:12px}dd{margin:3px 0 0;color:var(--muted);font-size:14px}.source{font-size:12px;overflow-wrap:anywhere}footer{font-size:12px;color:var(--muted);padding:10px 3px}
figure img{cursor:zoom-in}figure img:focus-visible{outline:3px solid var(--green);outline-offset:3px}dialog{width:96vw;max-width:1500px;max-height:94vh;border:1px solid var(--line);border-radius:12px;padding:0;color:var(--ink)}dialog::backdrop{background:#142b40aa}.viewer-bar{display:flex;align-items:center;justify-content:space-between;padding:10px 15px;border-bottom:1px solid var(--line)}.viewer-bar button{font:inherit;font-size:14px;border:1px solid var(--line);border-radius:6px;padding:4px 12px;background:white;color:var(--blue);cursor:pointer}.zoom-scroll{overflow:auto;max-height:82vh}.zoom-scroll img{display:block;width:max(1100px,90vw);max-width:none}
@media(max-width:640px){main{padding:24px 12px}h1{font-size:26px}h2{font-size:20px}.findings{grid-template-columns:1fr;gap:9px}section{padding:18px 12px}.finding{padding:12px 14px}nav a{padding:4px 11px}nav{gap:6px}}
@media print{body{background:white}main{padding:0;max-width:none}nav,.download{display:none}section{break-inside:auto}figure{break-inside:avoid}.scroll{max-height:none}.findings{grid-template-columns:1fr 1fr}}
</style></head><body><main>
<header><h1>共享单车时间与使用差异</h1><p class="muted">从整体时间规律，进一步观察会员、散客与两种车型的使用分布。</p><p class="meta">2024 年 1 月 · $rides 次有效骑行 · 工作日 $workdays 天 / 周末 $weekends 天 · 图表可点开放大</p><nav aria-label="报告导航"><a href="#time">整体时间</a><a href="#members">会员与散客</a><a href="#vehicles">车型对比</a><a href="#cross">交叉分析</a><a href="#method">方法与核验</a></nav></header>
<section aria-labelledby="findings"><h2 id="findings">先看四个发现</h2><div class="findings">$findings</div></section>
<section id="time"><h2>整体时间规律</h2><p class="muted">工作日与周末分别计算日均次数，避免天数不同影响比较。</p>
<figure><img src="$fig4" alt="图4，工作日与周末的日均每小时骑行次数"><figcaption>图 4 · 按出发时间计数。周一至周五为工作日，周六、日为周末；单位为次/天，标注点为主要峰值。</figcaption></figure>
<details><summary>展开星期 × 小时热力图 · 图 5</summary><figure><img src="$fig5" alt="图5，星期与小时的日均骑行需求热力图"><figcaption>每格总次数分别除以该星期的天数。颜色越深，日均需求越高，橙框为最高格；周一至周三各 5 天，其余各 4 天。</figcaption></figure><div class="scroll">$table5</div><a class="download" href="$csv5" download="fig05_weekday_hourly.csv">下载图 5 CSV</a></details>
<details><summary>工作日与周末逐小时统计</summary><div class="scroll">$table4</div><a class="download" href="$csv4" download="fig04_hourly_by_day_type.csv">下载图 4 CSV</a></details></section>
$preference_sections
<section id="method"><h2>方法与核验</h2><p>选择全城有效骑行，按日期、小时、人群和车型分组统计。用日均次数比较规模，用占比比较分布；图、表和结论共用一份统计结果。</p><p class="muted">已通过 $validation_count 项核验：覆盖完整、分母正确、分组计数可相互还原。</p>
<details><summary>展开计算公式、数据选择与核验明细</summary><dl><dt>日均每小时次数</dt><dd>该组该小时总次数 ÷ 对应日期类型天数。</dd><dt>组内时段占比</dt><dd>该组该小时总次数 ÷ 该组在对应日期类型内的全天总次数。每组 24 小时合计 100%。</dd><dt>该小时车型占比</dt><dd>该人群该小时某车型次数 ÷ 该人群该小时全部骑行次数。两车型合计 100%。</dd></dl><p>保留时间有效、站点信息不完整的骑行，避免无关字段缺失造成需求低估。确认每天有记录后补齐小时与分类组合，零记录记为 0；比例分母为零时标为无数据。</p><p class="muted">通过日期列批量运算、分组聚合和完整索引对齐得到结果，再核对原始记录、交叉表边际及占比合计。</p><div class="scroll">$validation_table</div><a class="download" href="$validation_csv" download="temporal_validation_summary.csv">下载统计核验 CSV</a>$summary_download</details>
<details class="source"><summary>数据版本与来源</summary><p>输入：$input · $columns 个字段 · 实际开始时间：$range<br>版本：<a href="$source">项目数据版本</a> · 本地提交：$commit<br>输入文件 SHA-256：$fingerprint</p></details></section>
<footer>结果描述本月骑行记录的使用分布，骑行次数不等于独立用户人数；日均值会掩盖日期间波动。图表、统计表和 CSV 已内嵌，可离线阅读与下载。</footer>
</main><dialog id="chart-viewer" aria-labelledby="viewer-title"><div class="viewer-bar"><strong id="viewer-title">图表放大查看</strong><button id="close-viewer" type="button">关闭</button></div><div class="zoom-scroll"><img id="zoom-image" alt=""></div></dialog>
<script>
const viewer=document.getElementById('chart-viewer');
const zoom=document.getElementById('zoom-image');
function openChart(img){zoom.src=img.src;zoom.alt=img.alt;viewer.showModal();viewer.querySelector('.zoom-scroll').scrollTo(0,0);}
document.querySelectorAll('figure img').forEach(img=>{img.tabIndex=0;img.setAttribute('role','button');img.setAttribute('aria-label',img.alt+'，放大查看');img.addEventListener('click',()=>openChart(img));img.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();openChart(img);}});});
document.getElementById('close-viewer').addEventListener('click',()=>viewer.close());
</script></body></html>
""")


def write_html(result: TemporalResult, validation: pd.DataFrame, validation_path: Path,
               path: Path, figures: list[Path], tables: list[Path],
               source: Path, fingerprint: str, commit: str,
               preferences: PreferenceResult, preference_assets: dict[str, Path]) -> None:
    table4 = result.day_type["daily_mean_rides"].unstack("day_type").reindex(columns=DAY_TYPES)
    table4.index = [f"{hour:02d}:00–{hour + 1:02d}:00" for hour in table4.index]
    table4.index.name = "开始时段"
    table4.columns.name = None
    table5 = result.weekday["daily_mean_rides"].unstack("hour")
    table5.index = WEEKDAYS
    table5.index.name = "星期"
    table5.columns = [f"{hour:02d} 时" for hour in table5.columns]
    work_peaks = "、".join(f"{hour:02d} 时" for hour in main_peak_hours(result.day_type.loc["工作日", "daily_mean_rides"]))
    weekend_peak = int(result.day_type.loc["周末", "daily_mean_rides"].idxmax())
    highlights = [f"工作日主要峰值在 {work_peaks}；周末最高在 {weekend_peak:02d} 时。", *preference_highlights(preferences)]
    summary_path = preference_assets["summary_table"]
    content = HTML_TEMPLATE.substitute(
        rides=f"{len(result.data):,}", columns=len(result.data.columns),
        workdays=result.counts["工作日"], weekends=result.counts["周末"],
        findings="".join(f'<div class="finding">{escape(text)}</div>' for text in highlights),
        preference_sections=preference_html(preferences, preference_assets),
        summary_download=f'<a class="download" href="{csv_data_url(summary_path)}" download="{summary_path.name}">下载峰值与总量摘要 CSV</a>',
        fig4=image_data_url(figures[0]), fig5=image_data_url(figures[1]),
        table4=table4.to_html(border=0, float_format=lambda x: f"{x:,.1f}"),
        table5=table5.to_html(border=0, float_format=lambda x: f"{x:,.1f}"),
        csv4=csv_data_url(tables[0]), csv5=csv_data_url(tables[1]), source=SOURCE_URL,
        validation_count=len(validation), validation_table=validation.to_html(index=False, border=0),
        validation_csv=csv_data_url(validation_path),
        range=f"{result.data['started_at'].min()} 至 {result.data['started_at'].max()}",
        input=escape(source.name), fingerprint=fingerprint, commit=escape(commit),
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "clean_trips.csv", help="时间分析使用的全量有效集")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR, help="报告、图片和汇总表的输出目录")
    arguments = parser.parse_args()
    source = arguments.input.resolve()
    result = analyze_temporal(read_clean_data(source))
    preferences = analyze_preferences(result.data, result.hourly, result.counts, result.day_type)
    validation = pd.concat([build_validation_summary(result), preferences.validation], ignore_index=True)
    output = arguments.output_dir.resolve()
    for directory in (output, output / "figures", output / "tables", output / "sections"):
        directory.mkdir(parents=True, exist_ok=True)
    font = set_plot_style()
    figures = [plot_day_types(result, output / "figures"), plot_weekday_heatmap(result, output / "figures")]
    preference_assets = export_preferences(preferences, output)
    tables = [output / "tables" / "fig04_hourly_by_day_type.csv", output / "tables" / "fig05_weekday_hourly.csv"]
    result.day_type.reset_index().to_csv(tables[0], index=False, encoding="utf-8-sig", float_format="%.4f", lineterminator="\n")
    weekday = result.weekday.reset_index()
    weekday.insert(1, "weekday_name", weekday["weekday"].map(dict(enumerate(WEEKDAYS))))
    weekday.to_csv(tables[1], index=False, encoding="utf-8-sig", float_format="%.4f", lineterminator="\n")
    validation_path = output / "tables" / "temporal_validation_summary.csv"
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    fingerprint, commit = input_fingerprint(source), repository_commit()
    markdown = output / "sections" / "02_temporal_and_prediction.md"
    html = output / "temporal_analysis.html"
    write_markdown(result, validation, markdown, source, fingerprint, commit, preferences)
    write_html(result, validation, validation_path, html, figures, tables, source, fingerprint, commit,
               preferences, preference_assets)
    print(f"有效骑行：{len(result.data):,} 次，{len(result.data.columns)} 列；744 小时；工作日 {result.counts['工作日']} 天，周末 {result.counts['周末']} 天。")
    print(f"{len(validation)} 项统计核验通过：订单与时间字段、日期覆盖、小时补齐、分母及总次数。")
    print(f"中文字体：{font}")
    for text in findings(result):
        print(text)
    for text in preference_highlights(preferences):
        print(text)
    for path in (*figures, *tables, *preference_assets.values(), validation_path, markdown, html):
        print(path)


if __name__ == "__main__":
    main()
