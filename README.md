# Pyclass

Final Group Assignment for Python Programming Course in the Second Year

## 共享单车数据分析

基于 2024 年 1 月共享单车骑行数据，开展数据清洗、骑行规律分析、需求预测和空间调度研究。

当前为**首次整合阶段版**：已纳入数据清洗与基础探索、时间规律分析、空间分布及调度分析。**需求预测与模型评估尚未交付**，对应图6–8及预测 Notebook 尚未纳入，本版本不代表全组最终成果。

| 模块 | 当前成果 | 主要入口 |
| --- | --- | --- |
| 数据清洗与基础探索 | 原始数据、双子集、图1–3、报告与 Notebook | `src.data_audit`、`src.data_cleaning`、`src.basic_eda` |
| 时间规律分析 | 图4–5、统计表、核验表、报告与离线 HTML | `src.temporal_analysis` |
| 空间与调度分析 | 图9–12、统计表、报告与 Notebook；已纳入到达时间归属修正 | `src.spatial_analysis`、`src.imbalance_dispatch` |
| 需求预测与模型评估 | 待补充 | 尚无运行入口 |

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
├── src/               # 公共绘图工具、数据审查、清洗、基础探索与空间调度分析代码
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

## 基础探索图表

```powershell
.\.venv\Scripts\python.exe -m src.basic_eda
```

| 图表 | 内容 |
| --- | --- |
| [图1](reports/figures/fig01_cleaning_funnel.png) | 数据清洗与双子集分流 |
| [图2](reports/figures/fig02_duration_distance_distribution.png) | 车型、用户类型的时长与直线距离箱线图 |
| [图3](reports/figures/fig03_trip_structure.png) | 站点记录结构与已知站点中的环线占比 |

三张 PNG 均为 300 DPI，并标注样本量与统计口径。图2不显示离群点，全部有效观测仍参与分位数计算；图3将站点信息不足的环线状态记为未知。

配套数值见 [分组描述统计](reports/tables/basic_eda_summary.csv) 和 [结构占比统计](reports/tables/trip_structure_summary.csv)。

公共绘图接口位于 `src/utils.py`：`set_plot_style()` 自动选择可用中文字体并统一样式，`save_figure(fig, filename)` 保存图片到 `reports/figures/`。车型与用户类型使用固定的 `RIDEABLE_COLORS`、`MEMBER_COLORS`，供全组复用。没有可用中文字体时会提示安装字体。`save_figure_html()` 在保存 300 DPI PNG 的同时输出自包含 HTML（图片以 base64 内嵌、附带数据表），图9–图12 使用该接口。

## 时间规律分析

```powershell
.\.venv\Scripts\python.exe -m src.temporal_analysis
```

| 图表 | 内容 |
| --- | --- |
| [图4](reports/figures/fig04_workday_weekend_hourly.png) | 工作日与周末的日均每小时骑行需求 |
| [图5](reports/figures/fig05_weekday_hour_heatmap.png) | 星期 × 小时日均需求热力图 |

该模块使用全量有效集 233,180 条记录，不重复清洗。按出发时间补齐 31 天、744 个小时格，工作日与周末分别除以 23 天和 8 天；星期热力图按各星期在本月实际出现的天数计算均值。周末只按星期定义，没有另外处理节假日。

配套成果包括 [离线 HTML](reports/temporal_analysis.html)、[图4统计表](reports/tables/fig04_hourly_by_day_type.csv)、[图5统计表](reports/tables/fig05_weekday_hourly.csv)和[15项统计核验](reports/tables/temporal_validation_summary.csv)。HTML 内嵌图片和表格，下载后可离线打开；GitHub 的文件页可查看源码，PNG 可直接预览。

`src/data_io.py` 作为时间模块的兼容入口，调用公共的 `src.data_cleaning.read_clean_data`；全组使用同一份数据和一套字段类型规则。

## 空间分布、站点失衡与调度分析

```powershell
.\.venv\Scripts\python.exe -m src.spatial_analysis
.\.venv\Scripts\python.exe -m src.imbalance_dispatch
```

| 图表 PNG | 离线 HTML | 内容 |
| --- | --- | --- |
| [图9](reports/figures/fig09_top_stations_od.png) | [下载页](reports/figures/fig09_top_stations_od.html) | Top15 热门站点总流量与双向合并后的高频 OD 走廊 |
| [图10](reports/figures/fig10_station_heatmap.png) | [下载页](reports/figures/fig10_station_heatmap.html) | 站点骑行热度空间分布：面积表流量、颜色表净流量方向 |
| [图11](reports/figures/fig11_peak_divergence.png) | [下载页](reports/figures/fig11_peak_divergence.html) | 早晚高峰核心站点净流量正负双向对比 |
| [图12](reports/figures/fig12_tidal_curves.png) | [下载页](reports/figures/fig12_tidal_curves.html) | 居住区与商务区站点 24 小时净流量与累计净流量曲线 |

四张图同时提供 300 DPI PNG（`fig09`–`fig12`）与自包含 HTML。HTML 内嵌 150 DPI 缩图，可离线直接打开，页面附表与正文数值一致。

净流量定义为 `F = 出发量 − 到达量`，正值表示净流出，负值表示净流入；忽略运维干预时，库存变化方向与 F 相反。失衡指数为 `I = |F| / (出发量 + 到达量 + 1)`，分母加 1 为防零平滑项。分时段统计中出发量按出发时间、到达量按还车时间（`ended_at`）归属：一次 6:50 出发、7:10 还车的骑行计入 7 时到达。同一窗口内出发与到达不必配平，差额反映跨窗口骑行（早高峰 +498、晚高峰 −1,394）。时段口径为周一至周五的早 7:00–9:59、晚 17:00–19:59，样本沿用 `data/processed/station_trips.csv` 的 180,914 条记录，不重复定义清洗规则。

空间与调度分析的样本按**出发月份**选取，包含这些行程完整的还车事件（跨月还车也随原行程保留），不等于覆盖 1 月所有真实还车事件的独立事件表。小时曲线把多个工作日按小时汇总，不能当作某一天的库存轨迹；净流量只用于描述流出／流入压力，缺少实时库存和容量时不能确认实际缺车或满桩。

调运方案按“最近邻优先”在同一高峰内匹配净流入点与净流出点，单程上限 6 km，量纲为辆/日（月累计 ÷ 23 个工作日），用于提供候选配对和调运规模；实际执行还需要库存、容量、载重和路网约束。配套表格：

- [站点流量汇总](reports/tables/station_flow_summary.csv)：760 个站点的出发量、到达量、净流量与失衡指数。
- [OD 走廊汇总](reports/tables/od_corridor_summary.csv)：29,005 个站点对的双向次数与主向占比。
- [高峰失衡汇总](reports/tables/peak_imbalance_summary.csv)：早晚高峰的净流量、失衡指数与分级。
- [逐小时净流量面板](reports/tables/hourly_net_flow.csv)：站点 × 小时的工作日净流量。
- [调运方案](reports/tables/dispatch_plan.csv)：130 条线路的目标站、取车站、调运量与直线距离。

新增测试覆盖净流量与失衡指数公式、OD 双向合并与环线剔除、分级阈值、高峰时段筛选和调运方案的容量约束：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_spatial_dispatch -v
```

## Notebook 复现

[01_cleaning_and_eda.ipynb](notebooks/01_cleaning_and_eda.ipynb) 串联原始数据概览、清洗规则、质量审查、双子集导出、图1–3、动态数值结论和数据接口检查。

[03_spatial_and_dispatch.ipynb](notebooks/03_spatial_and_dispatch.ipynb) 串联站点与 OD 汇总、图9–图12、失衡分级、调运方案和产物校验，最后调用 `src.spatial_analysis` 与 `src.imbalance_dispatch` 重写全部图表与统计表。

在 Jupyter 中打开该文件，选择 `Python (Pyclass)` 内核，执行“重启内核并运行全部单元格”。Notebook 可从项目根目录或 `notebooks/` 目录启动，直接调用 `src/` 中的现有函数，不维护另一套清洗规则。

运行会重新生成两份 CSV、审查与统计表和三张 300 DPI 图片，并检查原始文件未改动、子集一致性和导出类型。分析结论从本次运行数据计算；保存后的 Notebook 内嵌图表，方便预览。

该 Notebook 已在独立新内核中从头执行通过，并保存全部运行输出。03 Notebook 同样已在独立新内核中执行通过。

## 报告正文

[引言、数据介绍、数据清洗与基础探索](reports/sections/01_intro_and_cleaning.md)由马文博撰写，包含统一的数据处理口径、图1–3、分组统计、结果解释与分析局限。正文数值可通过配套统计表和 Notebook 复核，可作为小组报告对应章节的内容。

[时间规律分析](reports/sections/02_temporal_and_prediction.md)由陈宇涵撰写，对应第5章与图4–5；文件名保留既定接口，当前正文尚不包含需求预测结果。

[空间分布、站点潮汐失衡与车辆调度优化](reports/sections/03_spatial_and_dispatch.md)由康炜奕撰写，对应第7–8章与图9–12。第6章的需求预测内容待补齐后统一编排。

## 整合版本复现

完成环境配置后，在项目根目录依次运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.data_audit
.\.venv\Scripts\python.exe -m src.data_cleaning
.\.venv\Scripts\python.exe -m src.basic_eda
.\.venv\Scripts\python.exe -m src.temporal_analysis
.\.venv\Scripts\python.exe -m src.spatial_analysis
.\.venv\Scripts\python.exe -m src.imbalance_dispatch
```

测试、统计校验和生成过程使用同一套 `requirements.txt`。所有原作者提交记录保留在合并历史中，公共环境和接口在整合时统一。

本版整合验证：27 项单元测试、时间模块 15 项统计核验均通过；两份 Notebook 已分别在独立新内核中从头执行通过（18、14 个代码单元格），输出不含执行错误。三份共享 CSV 与上游数据版本字节一致，九张 PNG 均为 300 DPI，Notebook 的七张内嵌图片与导出文件一致。

整合来源：[马文博的数据基础](https://github.com/hhkQWQ/Pyclass/commit/777c65867a129d6f0dba03c0acfefbbce1e98cae)、[陈宇涵的时间分析](https://github.com/hhkQWQ/Pyclass/commit/94622f080d8b6dadf7715eccfa9dae3a5e0fec24)、[康炜奕的空间调度修正](https://github.com/hhkQWQ/Pyclass/commit/dc42053ae0b818fa9346f60dee4b92692f1a815c)。
