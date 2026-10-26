"""成员 B 的数据读取接口：恢复字段类型，不执行清洗或筛选。

读取规则沿用共享项目 read_clean_data 接口，原版本：
https://github.com/hhkQWQ/Pyclass/blob/777c65867a129d6f0dba03c0acfefbbce1e98cae/src/data_cleaning.py
"""

from pathlib import Path

import pandas as pd


RAW_COLUMNS = (
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "start_station_id", "end_station_name", "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng", "member_casual",
)
COORDINATE_COLUMNS = ("start_lat", "start_lng", "end_lat", "end_lng")
TIME_COLUMNS = ("started_at", "ended_at")
BOOLEAN_COLUMNS = ("is_weekend", "has_station_info", "is_dockless_ebike", "is_loop_trip")
FLOAT_COLUMNS = (*COORDINATE_COLUMNS, "duration_min", "dist_km")
DERIVED_COLUMNS = (
    "duration_min", "dist_km", "hour", "weekday", "is_weekend",
    "has_station_info", "is_dockless_ebike", "is_loop_trip",
)
OUTPUT_COLUMNS = (*RAW_COLUMNS, *DERIVED_COLUMNS)


def read_clean_data(path: str | Path) -> pd.DataFrame:
    """读取 21 列有效集，保持 ID 文本、日期时间和可缺失类型。"""
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
        raise ValueError("CSV 字段不符合时间分析的数据接口。")
    for column in TIME_COLUMNS:
        data[column] = pd.to_datetime(data[column], format="%Y-%m-%d %H:%M:%S", errors="raise")
    return data
