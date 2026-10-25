# Pyclass

Final Group Assignment for Python Programming Course in the Second Year

## 共享单车数据分析

基于 2024 年 1 月共享单车骑行数据，开展数据清洗、骑行规律分析、需求预测和空间调度研究。

## 成员分工

| 成员 | 主要职责 |
| --- | --- |
| 马文博 | 数据清洗、基础探索与项目统筹 |
| 陈宇涵 | 时间规律分析与需求预测 |
| 康炜奕 | 空间分布、站点失衡与调度优化 |

## 本地环境

使用 Python 3.12，虚拟环境位于 `.venv/`，依赖版本记录在 `requirements.txt`。

在已安装 Python 3.12 的电脑上首次配置时，先执行 `python -m venv .venv`；已有环境可跳过创建步骤。

在项目根目录使用 PowerShell 运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.utils
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name pyclass --display-name "Python (Pyclass)"
.\.venv\Scripts\jupyter.exe lab
```

Notebook 选择 `Python (Pyclass)` 内核；上述命令会将内核配置到项目虚拟环境中。

## 项目目录

```text
Pyclass/
├── config/            # 清洗规则与阈值
├── data/raw/          # 原始 CSV，随仓库提供
├── data/processed/    # 两份清洗后的 CSV，随仓库提供
├── src/               # 路径工具、数据审查与清洗导出代码
├── notebooks/         # 分析 Notebook
├── reports/figures/   # 图表
├── reports/sections/  # 报告章节
├── reports/tables/    # 数据质量报告、统计表与站点名称映射
├── tests/             # 数据保留边界与计数规则测试
├── requirements.txt
├── .gitignore
└── README.md
```

仓库提供原始文件 `data/raw/题5_共享单车骑行_202401.csv`（238,834 行、13 列）及两份清洗后的 CSV。`.gitignore` 仅放行这三份约定数据，其他临时数据、虚拟环境和缓存继续忽略；`.gitattributes` 保留 CSV 原始换行和字节，保证原始数据校验值可复核。

## 数据质量审查

```powershell
.\.venv\Scripts\python.exe -m src.data_audit
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

清洗约定记录在 [cleaning_rules.json](config/cleaning_rules.json)，审查结果见 [数据质量报告](reports/tables/data_quality_summary.md)。报告包含缺失值、异常时长、顺序清洗漏斗、阈值敏感性和站点名称映射。

当前时长口径为 **1–180 分钟，两端均包含**，审查推算全量有效集 233,180 行、站点子集 180,914 行，与实际导出结果一致。站点缺失不直接删除全城时间分析记录，坐标问题单独标记。

## 清洗数据导出

```powershell
.\.venv\Scripts\python.exe -m src.data_cleaning
```

| 文件 | 行数 | 用途 |
| --- | --- | --- |
| [clean_trips.csv](data/processed/clean_trips.csv) | 233,180 | 全城时间规律、车型和用户类型分析 |
| [station_trips.csv](data/processed/station_trips.csv) | 180,914 | 站点、OD 和站点需求分析，是全量有效集的子集 |

两份文件均为 UTF-8 BOM 编码，包含原始 13 列和 8 个派生字段：`duration_min`、`dist_km`、`hour`、`weekday`、`is_weekend`、`has_station_info`、`is_dockless_ebike`、`is_loop_trip`。站点名称按同 ID 的规范名称统一；`dist_km` 采用半径 6371 km 的球面 Haversine 直线距离，不能作为实际骑行路程。

读取时使用项目函数，保留 ID 字符串、时间及可缺失布尔类型：

```python
from src.data_cleaning import read_clean_data

trips = read_clean_data("data/processed/clean_trips.csv")
station_trips = read_clean_data("data/processed/station_trips.csv")
```

站点信息不足时，`is_loop_trip` 为未知，不能直接当作非环线。两份 CSV 随代码共享，也可运行清洗脚本重新生成，原始文件保持不变。

已完成数据审查和清洗数据导出，后续图表与建模分析尚未完成。
