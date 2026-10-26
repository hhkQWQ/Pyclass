# 数据质量审查结果与清洗口径

原始文件：`题5_共享单车骑行_202401.csv`；238,834 行、13 列。SHA-256：`bb189909da35620f98191dab9a4ba6ca36162d92917bfee174d91ad5704336b9`。

本报告记录原始数据审查及按规则推算的筛选结果。实际清洗数据由 `python -m src.data_cleaning` 生成，本报告不用于确认导出任务是否已运行。下面是本地项目约定，仍需陈宇涵、康炜奕在使用接口前核对。

## 清洗规则

- 使用原始出发时间筛选 `2024-01-01 00:00:00`（含）至 `2024-02-01 00:00:00`（不含）的记录；保留原始时间语义，不做时区转换。
- 从完整起止时间重算分钟数。保留 **1 <= duration_min <= 180**，两端均包含。短于下限和长于上限的记录按本项目短途骑行分析口径排除；该约定不代表这些行程在运营意义上无效。阈值变更需重跑审查并同步所有模块。
- 超过 1440 分钟额外标记为超时；该标记与超长时长重叠，不重复扣减，不能据此确认丢车。
- 订单号缺失、重复订单的后续出现项、非法时间、月份范围外记录、未知车型与未知用户类型按固定顺序排除。重复订单保留原文件首次出现项，审查过程不改动原始文件。
- 文本去首尾空白，空字符串视为缺失；ID 始终按字符串保存，不转数字，不因名称相同合并不同 ID。
- 站点或坐标缺失本身不删除全城时间分析记录；站点分析另取起终点 ID 完整的子集。空间图和距离分析使用有效坐标；坐标不填 0、不凭空插补。
- 同 ID 多名称按两端合并观测的最多出现名称统一；并列时按名称排序选取，保留映射与频数以便追溯。

## 原始数据审查

出发时间：2024-01-01 00:00:15 至 2024-01-31 23:59:51；还车时间：2024-01-01 00:04:53 至 2024-02-01 23:35:46。

共有 3,213 行包含需去除的首尾空白，规范化后整行重复 0 行。跨月还车 35 行，其中 19 行满足当前基础有效性条件并保留。

各类异常可以重叠，**不能直接相加作为删除行数**。

| flag | description | handling | row_count | percent_of_raw |
| --- | --- | --- | --- | --- |
| missing_ride_id | 订单号缺失 | 排除 | 0 | 0.0 |
| duplicate_ride_id | 重复订单号中首次出现之后的记录 | 排除 | 0 | 0.0 |
| invalid_timestamp | 起止时间缺失或解析失败 | 排除 | 0 | 0.0 |
| outside_analysis_month | 出发时间不在分析月份 | 排除 | 0 | 0.0 |
| unsupported_rideable_type | 车型缺失或不在约定枚举内 | 排除 | 0 | 0.0 |
| unsupported_member_type | 用户类型缺失或不在约定枚举内 | 排除 | 0 | 0.0 |
| nonpositive_duration | 骑行时长小于或等于零 | 排除 | 135 | 0.056525 |
| short_duration | 正时长但小于最短时长阈值 | 排除 | 4722 | 1.977105 |
| long_duration | 时长大于最长时长阈值 | 排除 | 797 | 0.333705 |
| overdue_duration | 超过超时审查阈值，仅凭时长不能确认丢车 | 审查标记 | 228 | 0.095464 |
| missing_station | 至少一端站点 ID 缺失 | 只影响站点子集 | 54946 | 23.005937 |
| station_id_name_mismatch | 同一端站点 ID 与名称的缺失状态不一致 | 审查标记 | 0 | 0.0 |
| missing_coordinate | 至少一个坐标缺失 | 只影响空间或距离分析 | 203 | 0.084996 |
| nonnumeric_coordinate | 非空坐标无法转为数值 | 只影响空间或距离分析 | 0 | 0.0 |
| invalid_coordinate_range | 数值坐标非有限数或超出经纬度范围 | 只影响空间或距离分析 | 0 | 0.0 |
| station_name_variant | 同一站点 ID 的名称与规范名称不同 | 按 ID 统一名称 | 5 | 0.002094 |
| classic_missing_station | 经典车至少一端站点 ID 缺失 | 审查标记 | 222 | 0.092952 |

## 缺失值

| field | missing_rows | percent_of_raw |
| --- | --- | --- |
| ride_id | 0 | 0.0 |
| rideable_type | 0 | 0.0 |
| started_at | 0 | 0.0 |
| ended_at | 0 | 0.0 |
| start_station_name | 37220 | 15.584046 |
| start_station_id | 37220 | 15.584046 |
| end_station_name | 39741 | 16.639591 |
| end_station_id | 39741 | 16.639591 |
| start_lat | 0 | 0.0 |
| start_lng | 0 | 0.0 |
| end_lat | 203 | 0.084996 |
| end_lng | 203 | 0.084996 |
| member_casual | 0 | 0.0 |

## 顺序清洗漏斗与双子集

`full` 中各步只统计首次被该规则排除的记录；`station_branch` 是从全量有效集筛选站点子集，不代表再次删除全量数据。站点子集包含于全量有效集，两者行数不能相加。

| scope | stage | removed_rows | remaining_rows |
| --- | --- | --- | --- |
| full | raw | 0 | 238834 |
| full | missing_ride_id | 0 | 238834 |
| full | duplicate_ride_id | 0 | 238834 |
| full | invalid_timestamp | 0 | 238834 |
| full | outside_analysis_month | 0 | 238834 |
| full | unsupported_rideable_type | 0 | 238834 |
| full | unsupported_member_type | 0 | 238834 |
| full | nonpositive_duration | 135 | 238699 |
| full | short_duration | 4722 | 233977 |
| full | long_duration | 797 | 233180 |
| station_branch | station_complete | 52266 | 180914 |

本次基础排除合计 5,654 行。全量候选集 233,180 行，站点候选集 180,914 行。候选集中坐标不可用于空间分析的记录为 0 行。

| 子集 | 审查推算行数 | 分工方案参考行数 | 差异 |
| --- | --- | --- | --- |
| clean_trips | 233180 | 233180 | 0 |
| station_trips | 180914 | 180914 | 0 |

参考行数仅作对照，不参与筛选；代码没有按目标行数截取、补齐或抽样。

## 站点名称

共有 761 个已知 ID，其中 1 个 ID 有多个名称，0 个名称对应多个 ID。按上述映射预计统一 5 条记录、5 个起终点名称。

| station_id | canonical_name | observed_name | occurrence_count | is_canonical |
| --- | --- | --- | --- | --- |
| 31380 | Sunrise Valley Dr & Roland Clarke Pl | Sunrise Valley Dr & Roland Clarke Pl | 10 | True |
| 31380 | Sunrise Valley Dr & Roland Clarke Pl | Roland Clarke Pl & Sunrise Valley Dr | 5 | False |

完整的 ID、规范名、观测名及频数见 `station_name_mapping.csv`。映射用于清洗导出时统一站点名称；质量审查过程不改写原始骑行数据。

## 阈值敏感性

保持其他基础规则及最短时长不变，比较不同最长时长的候选行数，说明长时骑行排除对结果的影响。

| min_inclusive | max_inclusive | clean_trips | station_trips |
| --- | --- | --- | --- |
| 1.0 | 60.0 | 229884 | 178237 |
| 1.0 | 120.0 | 232662 | 180470 |
| 1.0 | 180.0 | 233180 | 180914 |
| 1.0 | 240.0 | 233363 | 181054 |
| 1.0 | 1440.0 | 233749 | 181409 |

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
