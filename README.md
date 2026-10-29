# Pyclass · 共享单车骑行分析

我们以 2024 年 1 月的共享单车骑行订单为基础，从数据清洗出发，分析骑行的时间规律、用户偏好和空间分布，再尝试预测全城及核心站点的需求，结合站点流量讨论车辆调度。

这个项目主要围绕四个问题展开：

1. 原始订单中有哪些异常，怎样得到适合不同分析任务的数据？
2. 工作日和周末、会员和散客、不同车型的骑行规律有什么差别？
3. 只使用预测起点之前的数据，能否预测下一周的全城与站点需求？
4. 哪些站点和区域存在明显的潮汐失衡，可以提出怎样的调运候选方案？

我们把分析代码、三份 Notebook、数据表和图表保存在这个仓库中，方便一起修改、核对结果和复现。调度部分使用历史流量与测试周模型偏差做情景测算，具体的计算口径和适用范围见下文。

## 我们的分工

| 成员 | 主要职责 | 代码入口 |
| --- | --- | --- |
| 马文博 | 数据审查、清洗、基础探索与整合复核 | `src.data_audit`、`src.data_cleaning`、`src.basic_eda` |
| 陈宇涵 | 时间规律、使用偏好、全城与站点预测、模型比较 | `src.temporal_analysis`、`src.demand_model`、`src.station_model`、`src.model_comparison` |
| 康炜奕 | 站点与 OD、潮汐失衡、站点调运和分区情景 | `src.spatial_analysis`、`src.imbalance_dispatch`、`src.zone_dispatch` |

我们按模块分工，统一使用同一套清洗规则和数据接口。`src.data_io` 复用 `src.data_cleaning.read_clean_data`，项目路径和绘图样式统一放在 `src.utils` 中。各自的开发记录保留在 Git 提交历史里。

## 环境与运行

我们使用 Python 3.12，依赖版本记录在 [requirements.txt](requirements.txt) 中。下面以 Windows PowerShell 为例，首次运行时在项目根目录创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name pyclass --display-name "Python (Pyclass)"
.\.venv\Scripts\jupyter.exe lab
```

已有环境可以跳过创建步骤。打开 Notebook 后选择 `Python (Pyclass)` 内核。我们使用中文图表，运行环境需要安装微软雅黑、黑体或 Noto Sans CJK SC 等字体，程序会自动选择可用字体。

### 按 Notebook 完整复现

我们按分析顺序整理了三份 Notebook。完整复现时，依次打开并选择“重启内核、运行全部单元格”：

1. [01 清洗与基础探索](notebooks/01_cleaning_and_eda.ipynb)：审查与清洗、双子集导出、图1–3。
2. [02 时间规律与双层预测](notebooks/02_temporal_and_model.ipynb)：重新训练模型，生成时间与偏好分析、预测、比较报告及图4–8、15–18。
3. [03 空间失衡与调度](notebooks/03_spatial_and_dispatch.ipynb)：站点与 OD、潮汐失衡、调运与分区情景，生成图9–14。

可以从项目根目录或 `notebooks/` 启动。第二份 Notebook 会重新训练模型并生成逐站预测图，运行时间较长。

### 按脚本完整复现

也可以直接运行各个模块。我们建议先运行测试，再按数据依赖顺序生成结果：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.data_audit
.\.venv\Scripts\python.exe -m src.data_cleaning
.\.venv\Scripts\python.exe -m src.basic_eda
.\.venv\Scripts\python.exe -m src.temporal_analysis
.\.venv\Scripts\python.exe -m src.demand_model
.\.venv\Scripts\python.exe -m src.station_model
.\.venv\Scripts\python.exe -m src.model_comparison
.\.venv\Scripts\python.exe -m src.spatial_analysis
.\.venv\Scripts\python.exe -m src.imbalance_dispatch
.\.venv\Scripts\python.exe -m src.zone_dispatch
```

`src.zone_dispatch` 会根据站点预测总表和验证阶段选定的模型，自动更新核心预测子集，再生成分区配额、路线、热点和报告第9节。我们在时间、预测与空间分析中都复用清洗后的数据。

## 我们使用的数据

原始数据共有 238,834 条订单。清洗后，我们保留全量有效集用于时间分析和需求预测，同时提取双端站点完整的子集，用于站点空间分布、OD 和历史调运分析。

| 文件 | 规模 | 用途 |
| --- | --- | --- |
| [原始订单](data/raw/题5_共享单车骑行_202401.csv) | 238,834 行、13 列 | 清洗输入 |
| [clean_trips.csv](data/processed/clean_trips.csv) | 233,180 行、21 列 | 全城规律、预测、端点事件与缺站点电单车分析 |
| [station_trips.csv](data/processed/station_trips.csv) | 180,914 行、21 列 | 双端站点完整的空间、OD 与历史调运分析 |
| [core_station_predictions.csv](data/derived/core_station_predictions.csv) | 冻结推荐模型的测试周子集 | 分区情景的模型偏差输入，由站点预测表自动生成 |

我们将原始订单和两份清洗后的主 CSV 一起保存在仓库中，临时文件、缓存和 `.venv/` 不纳入版本管理。CSV 的原始字节与换行由 `.gitattributes` 保留，便于核对文件指纹。

我们按**出发月份**选取 2024 年 1 月的订单，保留时长在 **1–180 分钟（含边界）** 内的有效行程，也保留这些行程的跨月归还。具体规则见 [cleaning_rules.json](config/cleaning_rules.json)，数据检查结果见 [数据质量报告](reports/tables/data_quality_summary.md)。

对于缺少站点信息、但时间等字段有效的订单，我们仍保留其全城时间分析价值。几个派生字段的含义如下：

| 字段 | 我们采用的定义 |
| --- | --- |
| `dist_km` | 用 Haversine 公式计算的球面直线距离，不能当作实际骑行里程 |
| `has_station_info` | 起点和终点的站点 ID 是否都完整 |
| `is_dockless_ebike` | 缺少一端或两端站点 ID 的电单车，作为疑似无桩骑行的代理标记 |
| `is_loop_trip` | 起终点站点 ID 是否相同；站点不完整时记为未知 |

仅凭站点缺失，我们无法确认一辆电单车的真实运营形态，因此在解释热点和调度结果时保留这一限制。

读取清洗后的 CSV 时，我们统一使用下面的函数恢复站点 ID、日期时间和可缺失布尔类型：

```python
from src.data_cleaning import read_clean_data
trips = read_clean_data("data/processed/clean_trips.csv")
```

我们将周一至周五划为工作日，其中包含样本月份的公共假日。全城需求按出发时刻统计；站点归还和缺站点电单车归还，则按 `ended_at` 对应的星期、小时和时段统计。

## 我们怎样做预测与调度分析

### 下一周需求预测

我们比较历史同期均值、上周同期、Ridge 岭回归和随机森林，采用固定的自然周起点，一次预测未来 168 小时：

- 用前两周训练，在第三周验证并确定模型与参数。
- 用前三周重新训练，在第四周进行最终测试，周内不补入新的实测值。
- 核心 Top10 站点只用训练期的数据筛选；1 月 29–31 日不参与这轮选型和评价。

全城预测在四种方法中选型；站点预测在历史同期均值、Ridge 和随机森林中确定统一推荐方法，并把上周同期作为补充对照。

站点预测中，我们分别统计每个已知端点的借出和归还事件，因此它的覆盖范围与双端站点完整的空间子集不同。月初三小时的归还可能缺少前一个月出发的订单，我们将这部分标记为未知。全城和站点的需求量级也不同，我们分别解释其误差，不直接用原始 MAE 判断哪个预测任务更容易。

### 潮汐失衡与历史调度情景

我们用净流出 `F = 出发量 − 到达量` 描述流动方向，用失衡指数 `I = |F| / (出发量 + 到达量 + 1)` 描述相对失衡程度。净流出可以提示补车压力，但不能直接说明库存短缺。小时曲线合并了多个工作日的数据，分时段差额也受到途中车辆和样本边界影响，因此不能将曲线解释成某一天的库存轨迹。

在站点层面，我们根据历史净流量做最近邻候选配对；在区域层面，我们把站点划为 7 个片区，叠加“测试周预测 − 实测”的偏差修正和缺站点电单车流量，测算四个时段的投放配额。配额增加 15% 余量后向上取整，可调出量向下取整，并分别约束每个来源和目标的累计调运量。

当前站点候选配对共 131 条，合计约 380.17 辆/日；分区情景得到 6 条跨区路线，四个时段合计匹配 482 辆/日。两者属于不同层级的测算，不能相加，也不能把跨时段流量合计理解为独立车辆数。

我们将这些结果用于讨论历史流量的方向、规模和空间配对关系。分区修正使用了测试周实测，且我们尚未加入实时库存、站点容量、人员和路网约束，所以**这些结果是回顾性情景，不能直接作为未来调度或排班方案**。

## 图表索引

| 图号 | 内容 | 主要文件 |
| --- | --- | --- |
| 1–3 | 清洗漏斗、时长与距离、骑行结构 | [图1](reports/figures/fig01_cleaning_funnel.png)、[图2](reports/figures/fig02_duration_distance_distribution.png)、[图3](reports/figures/fig03_trip_structure.png) |
| 4–5 | 日均小时需求与星期热力图 | [图4](reports/figures/fig04_workday_weekend_hourly.png)、[图5](reports/figures/fig05_weekday_hour_heatmap.png) |
| 6A–6C | 会员、车型及交叉偏好 | [6A](reports/figures/fig06a_membership_hourly.png)、[6B](reports/figures/fig06b_vehicle_hourly.png)、[6C-1](reports/figures/fig06c1_member_vehicle_hourly.png)、[6C-2](reports/figures/fig06c2_vehicle_mix_by_member.png) |
| 7–8 | 全城预测与特征重要性 | [图7](reports/figures/fig07_city_weekly_prediction.png)、[图8](reports/figures/fig08_city_feature_importance.png) |
| 9–10 | 热门站点、OD 与空间热度 | [图9](reports/figures/fig09_top_stations_od.png)、[图10](reports/figures/fig10_station_heatmap.png) |
| 11–12 | 高峰失衡与潮汐曲线 | [图11](reports/figures/fig11_peak_divergence.png)、[图12](reports/figures/fig12_tidal_curves.png) |
| 13–14 | 分区路线、配额与热点情景 | [图13](reports/figures/fig13_zone_dispatch_routes.png)、[图14](reports/figures/fig14_zone_quota_hotspots.png) |
| 15–16 | 核心站点预测与共享模型重要性 | [图15（逐站预测系列）](reports/figures/stations)、[图16](reports/figures/fig16_station_feature_importance.png) |
| 17–18 | 模型相对误差与逐站误差 | [图17](reports/figures/fig17_model_comparison.png)、[图18](reports/figures/fig18_station_error_comparison.png) |

我们将图15整理为逐站预测系列，共 10 个站点 × 3 个预测目标 × 2 个评价窗口＝60 张图，保存在 `reports/figures/stations/`；文件名中的 `station_01` 至 `station_10` 表示站点筛选排名。图7的验证周曲线另列为补充图，不占用新的主图号。所有 PNG 均按 300 DPI 输出，图9–14 还提供内嵌图片和统计表的 HTML，下载后可直接在浏览器中离线查看。

## 我们如何核对结果

我们通过单元测试、统计守恒检查、预测边界检查和产物一致性检查来核对结果。目前已通过：

| 检查内容 | 结果 |
| --- | --- |
| 清洗、预测与调度单元测试 | 71 项通过 |
| 时间规律与用户偏好统计核验 | 46 项通过 |
| 已保存预测的模型比较核验 | 230 项通过 |
| 数据指纹、分区供需约束、报告链接和内嵌图等产物核验 | 135 项通过 |

我们在整合时分别用独立新内核运行了三份 Notebook，其代码单元格数量为 18、9、18。最近一次空间分析和图号整理后，我们重新完整运行了第二、第三份 Notebook，同步更新调运表、图12、报告引用和代码指纹；重新训练所得的预测数值与此前一致。图1–18的文件、图内标题和正文引用也已核对。清洗和调度的边界情况已加入回归测试，包括恰好 1 辆的匹配、空方案、缺坐标、站点编号保留和导出精度。

## 报告与数据表

我们按分析流程整理了报告章节，可以从数据处理一路读到预测和调度：

- [第1–4章：引言、清洗与基础探索](reports/sections/01_intro_and_cleaning.md)
- [第5章：时间规律与使用偏好](reports/sections/02_temporal_and_prediction.md)
- [第6.1节：全城需求预测](reports/sections/03_city_demand_prediction.md)
- [第6.2节：核心站点需求预测](reports/sections/04_station_demand_prediction.md)
- [第6.3节：模型比较与误差分析](reports/sections/05_model_comparison.md)
- [第7–9章：空间分布、失衡、调运与分区情景](reports/sections/03_spatial_and_dispatch.md)

需要直接查看图表和结果时，可以打开我们整理的离线报告：[时间与偏好](reports/temporal_analysis.html)、[全城预测](reports/demand_prediction.html)、[站点预测](reports/station_prediction.html)、[模型比较](reports/model_comparison.html)。

主要调度表包括：[站点历史候选配对](reports/tables/dispatch_plan.csv)、[区域信息](reports/tables/zone_profile.csv)、[分区配额](reports/tables/zone_preallocation_quota.csv)、[跨区路线](reports/tables/zone_dispatch_routes.csv)、[缺站点电单车热点](reports/tables/dockless_hotspots.csv)。我们把预测明细、验证阶段确定的模型与参数，以及各项核验结果一起保存在 `reports/tables/`，便于回查每个结论的依据。
