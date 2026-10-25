"""审查原始数据，输出质量表和清洗口径；本模块不导出清洗后的骑行数据。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import CLEANING_RULES_FILE, RAW_DATA_FILE, TABLE_DIR


RAW_COLUMNS = (
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "start_station_id", "end_station_name", "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng", "member_casual",
)
COORDINATE_COLUMNS = ("start_lat", "start_lng", "end_lat", "end_lng")
FLAG_DETAILS = {
    "missing_ride_id": ("订单号缺失", "排除"),
    "duplicate_ride_id": ("重复订单号中首次出现之后的记录", "排除"),
    "invalid_timestamp": ("起止时间缺失或解析失败", "排除"),
    "outside_analysis_month": ("出发时间不在分析月份", "排除"),
    "unsupported_rideable_type": ("车型缺失或不在约定枚举内", "排除"),
    "unsupported_member_type": ("用户类型缺失或不在约定枚举内", "排除"),
    "nonpositive_duration": ("骑行时长小于或等于零", "排除"),
    "short_duration": ("正时长但小于最短时长阈值", "排除"),
    "long_duration": ("时长大于最长时长阈值", "排除"),
    "overdue_duration": ("超过超时审查阈值，仅凭时长不能确认丢车", "审查标记"),
    "missing_station": ("至少一端站点 ID 缺失", "只影响站点子集"),
    "station_id_name_mismatch": ("同一端站点 ID 与名称的缺失状态不一致", "审查标记"),
    "missing_coordinate": ("至少一个坐标缺失", "只影响空间或距离分析"),
    "nonnumeric_coordinate": ("非空坐标无法转为数值", "只影响空间或距离分析"),
    "invalid_coordinate_range": ("数值坐标非有限数或超出经纬度范围", "只影响空间或距离分析"),
    "station_name_variant": ("同一站点 ID 的名称与规范名称不同", "按 ID 统一名称"),
    "classic_missing_station": ("经典车至少一端站点 ID 缺失", "审查标记"),
}
EXCLUSION_ORDER = (
    "missing_ride_id", "duplicate_ride_id", "invalid_timestamp", "outside_analysis_month",
    "unsupported_rideable_type", "unsupported_member_type", "nonpositive_duration",
    "short_duration", "long_duration",
)


@dataclass(frozen=True)
class CleaningRules:
    analysis_start: str
    analysis_end_exclusive: str
    duration_min_inclusive: float
    duration_max_inclusive: float
    overdue_min_exclusive: float
    allowed_rideable_types: tuple[str, ...]
    allowed_member_types: tuple[str, ...]
    reference_counts: dict[str, int]

    def __post_init__(self) -> None:
        if pd.Timestamp(self.analysis_start) >= pd.Timestamp(self.analysis_end_exclusive):
            raise ValueError("分析起点必须早于终点。")
        if not 0 < self.duration_min_inclusive <= self.duration_max_inclusive:
            raise ValueError("时长阈值必须满足 0 < 最短值 <= 最长值。")
        if self.overdue_min_exclusive <= self.duration_max_inclusive:
            raise ValueError("超时审查阈值必须大于基础清洗的最长时长。")
        if not self.allowed_rideable_types or not self.allowed_member_types:
            raise ValueError("车型与用户类型的枚举不能为空。")


@dataclass
class AuditResult:
    summary: dict[str, Any]
    flags: pd.DataFrame
    valid_mask: pd.Series
    station_mask: pd.Series
    metrics: pd.DataFrame
    missing_values: pd.DataFrame
    funnel: pd.DataFrame
    station_mapping: pd.DataFrame
    sensitivity: pd.DataFrame


def load_rules(path: Path = CLEANING_RULES_FILE) -> CleaningRules:
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("allowed_rideable_types", "allowed_member_types"):
        settings[key] = tuple(settings[key])
    return CleaningRules(**settings)


def load_raw_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """所有列先按字符串读取，保留站点 ID 的前导零和原始空值语义。"""
    raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    missing = sorted(set(RAW_COLUMNS) - set(raw.columns))
    unexpected = sorted(set(raw.columns) - set(RAW_COLUMNS))
    if missing or unexpected:
        raise ValueError(f"原始字段不符合接口：缺失={missing}，额外={unexpected}")
    return raw.loc[:, list(RAW_COLUMNS)]


def build_station_mapping(data: pd.DataFrame) -> pd.DataFrame:
    """合并两端观测，取同 ID 下频数最多的名称；同频时按名称排序。"""
    observations = pd.concat([
        data[[f"{side}_station_id", f"{side}_station_name"]].rename(columns={
            f"{side}_station_id": "station_id", f"{side}_station_name": "observed_name",
        }) for side in ("start", "end")
    ], ignore_index=True).dropna()
    counts = observations.groupby(["station_id", "observed_name"]).size().rename(
        "occurrence_count"
    ).reset_index()
    counts = counts.sort_values(
        ["station_id", "occurrence_count", "observed_name"],
        ascending=[True, False, True], kind="stable",
    ).reset_index(drop=True)
    counts["canonical_name"] = counts.groupby("station_id")["observed_name"].transform("first")
    counts["is_canonical"] = counts["observed_name"].eq(counts["canonical_name"])
    return counts[["station_id", "canonical_name", "observed_name", "occurrence_count", "is_canonical"]]


def normalize_raw_data(raw: pd.DataFrame) -> pd.DataFrame:
    """统一字符串与空值，供质量审查和清洗导出共用。"""
    return raw.loc[:, list(RAW_COLUMNS)].astype("string").apply(
        lambda column: column.str.strip().replace("", pd.NA)
    )


def audit_data(raw: pd.DataFrame, rules: CleaningRules) -> AuditResult:
    """保留全部审查标记；漏斗按固定顺序统计本步新增排除的数量。"""
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"原始字段缺失：{sorted(missing)}")
    if not raw.index.is_unique:
        raise ValueError("输入数据必须有唯一行索引。")
    data = normalize_raw_data(raw)
    starts = pd.to_datetime(data["started_at"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ends = pd.to_datetime(data["ended_at"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    duration = (ends - starts).dt.total_seconds() / 60
    coordinates = data[list(COORDINATE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    missing_coordinate = data[list(COORDINATE_COLUMNS)].isna().any(axis=1)
    nonnumeric = (data[list(COORDINATE_COLUMNS)].notna() & coordinates.isna()).any(axis=1)
    bad_range = pd.Series(False, index=data.index)
    for column in COORDINATE_COLUMNS:
        limit = 90 if column.endswith("lat") else 180
        bad_range |= coordinates[column].notna() & (
            ~np.isfinite(coordinates[column]) | ~coordinates[column].between(-limit, limit)
        )
    station_complete = data[["start_station_id", "end_station_id"]].notna().all(axis=1)
    mapping = build_station_mapping(data)
    canonical = mapping.drop_duplicates("station_id").set_index("station_id")["canonical_name"]
    name_variant = pd.Series(False, index=data.index)
    id_name_mismatch = pd.Series(False, index=data.index)
    renamed_endpoints = 0
    for side in ("start", "end"):
        ids, names = data[f"{side}_station_id"], data[f"{side}_station_name"]
        chosen = ids.map(canonical)
        variant = (names.notna() & chosen.notna() & names.ne(chosen)).fillna(False)
        name_variant |= variant
        renamed_endpoints += int(variant.sum())
        id_name_mismatch |= ids.isna().ne(names.isna())
    flags = pd.DataFrame({
        "missing_ride_id": data["ride_id"].isna(),
        "duplicate_ride_id": data["ride_id"].notna() & data["ride_id"].duplicated(keep="first"),
        "invalid_timestamp": starts.isna() | ends.isna(),
        "outside_analysis_month": starts.notna() & ~(
            starts.ge(rules.analysis_start) & starts.lt(rules.analysis_end_exclusive)
        ),
        "unsupported_rideable_type": ~data["rideable_type"].isin(rules.allowed_rideable_types),
        "unsupported_member_type": ~data["member_casual"].isin(rules.allowed_member_types),
        "nonpositive_duration": duration.le(0),
        "short_duration": duration.gt(0) & duration.lt(rules.duration_min_inclusive),
        "long_duration": duration.gt(rules.duration_max_inclusive),
        "overdue_duration": duration.gt(rules.overdue_min_exclusive),
        "missing_station": ~station_complete,
        "station_id_name_mismatch": id_name_mismatch,
        "missing_coordinate": missing_coordinate,
        "nonnumeric_coordinate": nonnumeric,
        "invalid_coordinate_range": bad_range,
        "station_name_variant": name_variant,
        "classic_missing_station": data["rideable_type"].eq("classic_bike") & ~station_complete,
    }, index=data.index).fillna(False).astype(bool)
    alive = pd.Series(True, index=data.index)
    funnel = [{"scope": "full", "stage": "raw", "removed_rows": 0, "remaining_rows": len(data)}]
    for flag in EXCLUSION_ORDER:
        removed = int((alive & flags[flag]).sum())
        alive &= ~flags[flag]
        funnel.append({"scope": "full", "stage": flag, "removed_rows": removed, "remaining_rows": int(alive.sum())})
    station_mask = alive & station_complete
    funnel.append({"scope": "station_branch", "stage": "station_complete",
                   "removed_rows": int((alive & ~station_complete).sum()), "remaining_rows": int(station_mask.sum())})
    percentages = lambda count: round(count / len(data) * 100, 6) if len(data) else 0.0
    metrics = pd.DataFrame([
        {"flag": flag, "description": description, "handling": handling,
         "row_count": int(flags[flag].sum()), "percent_of_raw": percentages(int(flags[flag].sum()))}
        for flag, (description, handling) in FLAG_DETAILS.items()
    ])
    missing_values = pd.DataFrame([
        {"field": field, "missing_rows": int(data[field].isna().sum()),
         "percent_of_raw": percentages(int(data[field].isna().sum()))}
        for field in RAW_COLUMNS
    ])
    nonduration = ~flags[[f for f in EXCLUSION_ORDER if "duration" not in f]].any(axis=1)
    sensitivity = []
    for maximum in sorted({60.0, 120.0, rules.duration_max_inclusive, 240.0, 1440.0}):
        mask = nonduration & duration.ge(rules.duration_min_inclusive) & duration.le(maximum)
        sensitivity.append({"min_inclusive": rules.duration_min_inclusive, "max_inclusive": maximum,
                            "clean_trips": int(mask.sum()), "station_trips": int((mask & station_complete).sum())})
    whitespace = raw.loc[:, list(RAW_COLUMNS)].astype("string").fillna("").ne(data.fillna(""))
    valid_coordinate = ~(missing_coordinate | nonnumeric | bad_range)
    id_name_counts = mapping.groupby("station_id")["observed_name"].nunique()
    name_id_counts = mapping.groupby("observed_name")["station_id"].nunique()
    summary = {
        "raw_rows": len(data), "raw_columns": len(data.columns),
        "clean_trips": int(alive.sum()), "station_trips": int(station_mask.sum()),
        "excluded_rows": int((~alive).sum()), "whitespace_rows": int(whitespace.any(axis=1).sum()),
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "station_ids": int(mapping["station_id"].nunique()),
        "station_ids_with_multiple_names": int(id_name_counts.gt(1).sum()),
        "station_names_with_multiple_ids": int(name_id_counts.gt(1).sum()),
        "station_name_repair_rows": int(name_variant.sum()), "station_name_repair_endpoints": renamed_endpoints,
        "cross_month_end_rows": int(ends.ge(rules.analysis_end_exclusive).sum()),
        "valid_cross_month_end_rows": int((alive & ends.ge(rules.analysis_end_exclusive)).sum()),
        "candidate_rows_without_valid_coordinates": int((alive & ~valid_coordinate).sum()),
        "started_at_min": str(starts.min()), "started_at_max": str(starts.max()),
        "ended_at_min": str(ends.min()), "ended_at_max": str(ends.max()),
    }
    return AuditResult(summary, flags, alive, station_mask, metrics, missing_values,
                       pd.DataFrame(funnel), mapping, pd.DataFrame(sensitivity))


def markdown_table(frame: pd.DataFrame) -> str:
    """不依赖 tabulate，生成 GitHub 可阅读的 Markdown 表格。"""
    escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(escape, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(map(escape, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_audit_outputs(result: AuditResult, rules: CleaningRules, input_path: Path,
                        output_dir: Path = TABLE_DIR) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "data_quality_summary.csv": result.metrics,
        "missing_values.csv": result.missing_values,
        "cleaning_funnel.csv": result.funnel,
        "station_name_mapping.csv": result.station_mapping,
        "duration_threshold_sensitivity.csv": result.sensitivity,
    }.items():
        table.to_csv(output_dir / name, index=False, encoding="utf-8-sig", lineterminator="\n")
    summary = result.summary
    references = pd.DataFrame([
        {"子集": name, "审查推算行数": summary[name], "分工方案参考行数": reference,
         "差异": summary[name] - reference}
        for name, reference in rules.reference_counts.items()
    ])
    variants = result.station_mapping[
        result.station_mapping["station_id"].isin(
            result.station_mapping.groupby("station_id").size().loc[lambda values: values.gt(1)].index
        )
    ]
    with Path(input_path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = f"""# 数据质量审查结果与清洗口径

原始文件：`{Path(input_path).name}`；{summary['raw_rows']:,} 行、{summary['raw_columns']} 列。SHA-256：`{digest}`。

本报告记录原始数据审查及按规则推算的筛选结果。实际清洗数据由 `python -m src.data_cleaning` 生成，本报告不用于确认导出任务是否已运行。下面是本地项目约定，仍需陈宇涵、康炜奕在使用接口前核对。

## 清洗规则

- 使用原始出发时间筛选 `{rules.analysis_start}`（含）至 `{rules.analysis_end_exclusive}`（不含）的记录；保留原始时间语义，不做时区转换。
- 从完整起止时间重算分钟数。保留 **{rules.duration_min_inclusive:g} <= duration_min <= {rules.duration_max_inclusive:g}**，两端均包含。短于下限和长于上限的记录按本项目短途骑行分析口径排除；该约定不代表这些行程在运营意义上无效。阈值变更需重跑审查并同步所有模块。
- 超过 {rules.overdue_min_exclusive:g} 分钟额外标记为超时；该标记与超长时长重叠，不重复扣减，不能据此确认丢车。
- 订单号缺失、重复订单的后续出现项、非法时间、月份范围外记录、未知车型与未知用户类型按固定顺序排除。重复订单保留原文件首次出现项，审查过程不改动原始文件。
- 文本去首尾空白，空字符串视为缺失；ID 始终按字符串保存，不转数字，不因名称相同合并不同 ID。
- 站点或坐标缺失本身不删除全城时间分析记录；站点分析另取起终点 ID 完整的子集。空间图和距离分析使用有效坐标；坐标不填 0、不凭空插补。
- 同 ID 多名称按两端合并观测的最多出现名称统一；并列时按名称排序选取，保留映射与频数以便追溯。

## 原始数据审查

出发时间：{summary['started_at_min']} 至 {summary['started_at_max']}；还车时间：{summary['ended_at_min']} 至 {summary['ended_at_max']}。

共有 {summary['whitespace_rows']:,} 行包含需去除的首尾空白，规范化后整行重复 {summary['exact_duplicate_rows']:,} 行。跨月还车 {summary['cross_month_end_rows']:,} 行，其中 {summary['valid_cross_month_end_rows']:,} 行满足当前基础有效性条件并保留。

各类异常可以重叠，**不能直接相加作为删除行数**。

{markdown_table(result.metrics)}

## 缺失值

{markdown_table(result.missing_values)}

## 顺序清洗漏斗与双子集

`full` 中各步只统计首次被该规则排除的记录；`station_branch` 是从全量有效集筛选站点子集，不代表再次删除全量数据。站点子集包含于全量有效集，两者行数不能相加。

{markdown_table(result.funnel)}

本次基础排除合计 {summary['excluded_rows']:,} 行。全量候选集 {summary['clean_trips']:,} 行，站点候选集 {summary['station_trips']:,} 行。候选集中坐标不可用于空间分析的记录为 {summary['candidate_rows_without_valid_coordinates']:,} 行。

{markdown_table(references)}

参考行数仅作对照，不参与筛选；代码没有按目标行数截取、补齐或抽样。

## 站点名称

共有 {summary['station_ids']:,} 个已知 ID，其中 {summary['station_ids_with_multiple_names']:,} 个 ID 有多个名称，{summary['station_names_with_multiple_ids']:,} 个名称对应多个 ID。按上述映射预计统一 {summary['station_name_repair_rows']:,} 条记录、{summary['station_name_repair_endpoints']:,} 个起终点名称。

{markdown_table(variants)}

完整的 ID、规范名、观测名及频数见 `station_name_mapping.csv`。映射用于清洗导出时统一站点名称；质量审查过程不改写原始骑行数据。

## 阈值敏感性

保持其他基础规则及最短时长不变，比较不同最长时长的候选行数，说明长时骑行排除对结果的影响。

{markdown_table(result.sensitivity)}

## 共享字段约定

原始 13 列保留字段名。`ride_id`、站点 ID、站点名及分类字段使用字符串；`started_at`、`ended_at` 解析为无时区 datetime；坐标为可缺失数值。CSV 本身不保存这些类型，读取时必须显式恢复。

以下为清洗数据导出的派生字段接口；本审查报告本身不生成这些导出列。

| 字段 | 类型 | 口径 |
| --- | --- | --- |
| duration_min | float | 完整起止时间之差的总分钟数 |
| dist_km | nullable float | 有效经纬度计算的 Haversine 直线距离，单位 km；无效坐标记缺失，不代表实际骑行路程 |
| hour | int | 出发小时，0–23 |
| weekday | int | 周一为 0、周日为 6 |
| is_weekend | bool | weekday >= 5，不等同于节假日标记 |
| has_station_info | bool | 两端站点 ID 均非空 |
| is_dockless_ebike | bool | 电单车且至少一端站点 ID 缺失的代理标记，不能据此证明真实无桩运营状态 |
| is_loop_trip | nullable bool | 两端 ID 已知时判断是否相等；站点不足时为未知，不能归为非环线 |

`clean_trips.csv` 供全城时间分析与基础探索使用；`station_trips.csv` 为其站点 ID 完整子集，供站点、OD 与站点需求分析使用。站点缺失的经典车不归为无桩电单车。CSV 读取需声明 ID 字符串、时间解析和可缺失布尔类型，不将字符串 `"False"` 直接强转为 bool。

缺少实时库存，不能从订单数据确定实际缺车、满桩或丢车。周末按星期计算；节假日未另作识别。
"""
    (output_dir / "data_quality_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA_FILE)
    parser.add_argument("--rules", type=Path, default=CLEANING_RULES_FILE)
    parser.add_argument("--output-dir", type=Path, default=TABLE_DIR)
    arguments = parser.parse_args()
    rules = load_rules(arguments.rules)
    result = audit_data(load_raw_data(arguments.input), rules)
    write_audit_outputs(result, rules, arguments.input, arguments.output_dir)
    print(f"原始数据：{result.summary['raw_rows']:,} 行")
    print(f"全量候选集：{result.summary['clean_trips']:,} 行；站点候选集：{result.summary['station_trips']:,} 行")
    print(f"审查结果：{arguments.output_dir / 'data_quality_summary.md'}")
    print("审查完成；清洗数据通过 python -m src.data_cleaning 导出。")


if __name__ == "__main__":
    main()
