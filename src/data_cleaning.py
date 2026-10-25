"""按统一规则清洗骑行数据、生成基础特征并导出两个 CSV 子集。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .data_audit import (
    AuditResult, CleaningRules, COORDINATE_COLUMNS, RAW_COLUMNS,
    audit_data, load_raw_data, load_rules, normalize_raw_data,
)
from .utils import CLEANING_RULES_FILE, PROCESSED_DIR, RAW_DATA_FILE


EARTH_RADIUS_KM = 6371.0
TIME_COLUMNS = ("started_at", "ended_at")
BOOLEAN_COLUMNS = ("is_weekend", "has_station_info", "is_dockless_ebike", "is_loop_trip")
FLOAT_COLUMNS = (*COORDINATE_COLUMNS, "duration_min", "dist_km")
DERIVED_COLUMNS = (
    "duration_min", "dist_km", "hour", "weekday", "is_weekend",
    "has_station_info", "is_dockless_ebike", "is_loop_trip",
)
OUTPUT_COLUMNS = (*RAW_COLUMNS, *DERIVED_COLUMNS)


@dataclass
class CleaningResult:
    clean_trips: pd.DataFrame
    station_trips: pd.DataFrame
    audit: AuditResult
    renamed_endpoints: int


def haversine_km(coordinates: pd.DataFrame) -> pd.Series:
    """使用半径 6371 km 的球面近似；坐标缺失或无效时返回缺失值。"""
    values = coordinates.loc[:, list(COORDINATE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values).all(axis=1)
    valid &= (np.abs(values[:, [0, 2]]) <= 90).all(axis=1)
    valid &= (np.abs(values[:, [1, 3]]) <= 180).all(axis=1)
    distances = np.full(len(values), np.nan)
    start_lat, start_lng, end_lat, end_lng = np.radians(values[valid]).T
    a = (
        np.sin((end_lat - start_lat) / 2) ** 2
        + np.cos(start_lat) * np.cos(end_lat) * np.sin((end_lng - start_lng) / 2) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    distances[valid] = 2 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return pd.Series(distances, index=coordinates.index, dtype="Float64", name="dist_km")


def clean_data(raw: pd.DataFrame, rules: CleaningRules) -> CleaningResult:
    """复用审查筛选掩码；缺站点或缺坐标不额外删除全城时间分析记录。"""
    audit = audit_data(raw, rules)
    data = normalize_raw_data(raw).loc[audit.valid_mask].copy()
    canonical = audit.station_mapping.drop_duplicates("station_id").set_index("station_id")["canonical_name"]
    renamed_endpoints = 0
    for side in ("start", "end"):
        column = f"{side}_station_name"
        chosen = data[f"{side}_station_id"].map(canonical).astype("string")
        changed = chosen.notna() & data[column].notna() & chosen.ne(data[column])
        renamed_endpoints += int(changed.fillna(False).sum())
        data[column] = chosen.combine_first(data[column])
    for column in TIME_COLUMNS:
        data[column] = pd.to_datetime(data[column], format="%Y-%m-%d %H:%M:%S", errors="raise")
    for column in COORDINATE_COLUMNS:
        numeric = pd.to_numeric(data[column], errors="coerce").astype("Float64")
        limit = 90 if column.endswith("lat") else 180
        data[column] = numeric.where(np.isfinite(numeric) & numeric.between(-limit, limit), pd.NA)
    data["duration_min"] = (
        (data["ended_at"] - data["started_at"]).dt.total_seconds() / 60
    ).astype("Float64")
    data["dist_km"] = haversine_km(data)
    data["hour"] = data["started_at"].dt.hour.astype("int8")
    data["weekday"] = data["started_at"].dt.dayofweek.astype("int8")
    data["is_weekend"] = data["weekday"].ge(5).astype("boolean")
    data["has_station_info"] = data[["start_station_id", "end_station_id"]].notna().all(axis=1).astype("boolean")
    data["is_dockless_ebike"] = (
        data["rideable_type"].eq("electric_bike") & ~data["has_station_info"]
    ).astype("boolean")
    data["is_loop_trip"] = data["start_station_id"].eq(data["end_station_id"]).astype("boolean").where(
        data["has_station_info"], pd.NA
    )
    clean = data.loc[:, list(OUTPUT_COLUMNS)].reset_index(drop=True)
    station = clean.loc[clean["has_station_info"]].reset_index(drop=True)
    result = CleaningResult(clean, station, audit, renamed_endpoints)
    validate_datasets(result, rules)
    return result


def validate_datasets(result: CleaningResult, rules: CleaningRules) -> None:
    """检查影响下游分析的导出约束，不使用参考行数控制结果。"""
    clean, station = result.clean_trips, result.station_trips
    for name, frame in (("clean_trips", clean), ("station_trips", station)):
        if tuple(frame.columns) != OUTPUT_COLUMNS:
            raise ValueError(f"{name} 的导出字段不符合约定。")
        if frame["ride_id"].isna().any() or not frame["ride_id"].is_unique:
            raise ValueError(f"{name} 存在缺失或重复订单号。")
        if len(frame) != result.audit.summary[name]:
            raise ValueError(f"{name} 行数与实际审查筛选结果不一致。")
        valid_duration = frame["duration_min"].between(
            rules.duration_min_inclusive, rules.duration_max_inclusive
        )
        if not valid_duration.fillna(False).all():
            raise ValueError(f"{name} 中存在范围外或缺失时长。")
        valid_month = frame["started_at"].ge(rules.analysis_start) & frame["started_at"].lt(rules.analysis_end_exclusive)
        if not valid_month.all():
            raise ValueError(f"{name} 中存在分析月份以外的出发记录。")
        distance = frame["dist_km"].dropna()
        if not (np.isfinite(distance) & distance.ge(0)).all():
            raise ValueError(f"{name} 中存在非法距离。")
    if not station["has_station_info"].all():
        raise ValueError("站点子集含缺失站点 ID。")
    if not clean.loc[clean["has_station_info"]].reset_index(drop=True).equals(station):
        raise ValueError("站点子集与全量有效集的对应记录不一致。")


def read_clean_data(path: str | Path) -> pd.DataFrame:
    """恢复 CSV 中的 ID、时间、可缺失布尔和数值类型。"""
    dtypes = {column: "string" for column in RAW_COLUMNS if column not in COORDINATE_COLUMNS}
    dtypes.update({column: "Float64" for column in FLOAT_COLUMNS})
    dtypes.update({column: "boolean" for column in BOOLEAN_COLUMNS})
    dtypes.update({"hour": "int8", "weekday": "int8"})
    data = pd.read_csv(
        path, dtype=dtypes, encoding="utf-8-sig", keep_default_na=False,
        na_values=[""], true_values=["True"], false_values=["False"],
        float_precision="round_trip",
    )
    if tuple(data.columns) != OUTPUT_COLUMNS:
        raise ValueError("CSV 字段不符合清洗数据接口。")
    for column in TIME_COLUMNS:
        data[column] = pd.to_datetime(data[column], format="%Y-%m-%d %H:%M:%S", errors="raise")
    return data


def export_datasets(result: CleaningResult, output_dir: Path = PROCESSED_DIR) -> dict[str, Path]:
    """先写临时文件，完成后替换目标 CSV，避免留下半写入文件。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, frame in (("clean_trips", result.clean_trips), ("station_trips", result.station_trips)):
        target = output_dir / f"{name}.csv"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8-sig", newline="", suffix=".tmp",
                prefix=f".{name}-", dir=output_dir, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                frame.to_csv(stream, index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n")
            temporary.replace(target)
            outputs[name] = target
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA_FILE)
    parser.add_argument("--rules", type=Path, default=CLEANING_RULES_FILE)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    arguments = parser.parse_args()
    result = clean_data(load_raw_data(arguments.input), load_rules(arguments.rules))
    outputs = export_datasets(result, arguments.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {len(getattr(result, name)):,} 行，{len(OUTPUT_COLUMNS)} 列 -> {path}")
    print(f"已统一 {result.renamed_endpoints} 个起终点站点名称；原始数据未改动。")


if __name__ == "__main__":
    main()
