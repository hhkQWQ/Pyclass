# Pyclass · 共享单车骑行分析

基于 2024 年 1 月共享单车订单，完成数据清洗、骑行规律分析、全城与核心站点需求预测，以及空间失衡与调度情景测算。

当前为**整合修正版**：已接入三人的分析模块，并补齐预测 Notebook。调度结果使用历史流量和测试周模型偏差，属于回顾性情景，不能直接作为未来库存或实际调度承诺。最终 Word 报告、PPT 和汇报视频仍需另行整理。

## 成员与模块

| 成员 | 主要职责 | 代码入口 |
| --- | --- | --- |
| 马文博 | 数据审查、清洗、基础探索与整合复核 | `src.data_audit`、`src.data_cleaning`、`src.basic_eda` |
| 陈宇涵 | 时间规律、使用偏好、全城与站点预测、模型比较 | `src.temporal_analysis`、`src.demand_model`、`src.station_model`、`src.model_comparison` |
| 康炜奕 | 站点与 OD、潮汐失衡、站点调运和分区情景 | `src.spatial_analysis`、`src.imbalance_dispatch`、`src.zone_dispatch` |

原作者提交记录保留在合并历史中。公共数据读取入口 `src.data_io` 复用 `src.data_cleaning.read_clean_data`，绘图与目录共用 `src.utils`。

## 环境与运行

使用 Python 3.12，依赖固定在 [requirements.txt](requirements.txt)，包括预测与分区需要的 `scikit-learn`。
首次配置时，在项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name pyclass --display-name "Python (Pyclass)"
.\.venv\Scripts\jupyter.exe lab
```

已有环境可跳过创建步骤。图表需要微软雅黑、黑体或 Noto Sans CJK SC 等中文字体，程序会自动选择可用字体。Notebook 选择 `Python (Pyclass)` 内核。

### 按 Notebook 完整复现

依次打开并“重启内核、运行全部单元格”：

1. [01 清洗与基础探索](notebooks/01_cleaning_and_eda.ipynb)：审查与清洗、双子集导出、图1–3。
2. [02 时间规律与双层预测](notebooks/02_temporal_and_model.ipynb)：重新训练模型，生成时间与偏好分析、预测、比较报告及图4–8、15–18。
3. [03 空间失衡与调度](notebooks/03_spatial_and_dispatch.ipynb)：站点与 OD、潮汐失衡、调运与分区情景，生成图9–14。

Notebook 可从项目根目录或 `notebooks/` 启动。02 需要训练模型并生成逐站图，运行时间比纯统计单元格长。

### 按脚本完整复现

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

最后一个模块会从站点预测总表及冻结推荐记录自动更新核心预测子集，再生成配额、路线、热点和报告第9节。时间、模型与空间分析都读取既有清洗数据，不另设一套清洗规则。

## 数据与统一口径

| 文件 | 规模 | 用途 |
| --- | --- | --- |
| [原始订单](data/raw/题5_共享单车骑行_202401.csv) | 238,834 行、13 列 | 清洗输入 |
| [clean_trips.csv](data/processed/clean_trips.csv) | 233,180 行、21 列 | 全城规律、预测、端点事件与缺站点电单车分析 |
| [station_trips.csv](data/processed/station_trips.csv) | 180,914 行、21 列 | 双端站点完整的空间、OD 与历史调运分析 |
| [core_station_predictions.csv](data/derived/core_station_predictions.csv) | 冻结推荐模型的测试周子集 | 分区情景的模型偏差输入，由站点预测表自动生成 |

三份主 CSV 随仓库共享；临时数据、缓存与 `.venv/` 忽略。`.gitattributes` 保留 CSV 的原始字节与换行。

清洗按**出发月份**选取 2024 年 1 月订单，时长为 **1–180 分钟，两端包含**，保留合规行程的跨月归还。清洗规则见 [cleaning_rules.json](config/cleaning_rules.json)，审查结果见 [数据质量报告](reports/tables/data_quality_summary.md)。

站点缺失不直接删除全城时间分析记录。`dist_km` 是球面 Haversine 直线距离，不是路网里程。`is_dockless_ebike` 表示缺少一端或两端站点 ID 的电单车，是运营形态的代理标记；不能证明真实无桩运营。站点不全时 `is_loop_trip` 为未知。

使用公共读取函数可恢复 ID、日期时间和可缺失布尔类型：

```python
from src.data_cleaning import read_clean_data
trips = read_clean_data("data/processed/clean_trips.csv")
```

工作日只按周一至周五划分，包含样本中的公共假日。全城需求按出发时刻统计；站点归还与缺站点电单车归还均按 `ended_at` 自己的星期、小时和时段统计。

## 预测与调度的边界

预测采用自然周固定起点：前两周训练、第三周验证；前三周重训、第四周测试，一次预测 168 小时，周内不读取新实测作为输入。推荐模型与参数在验证后冻结，Top10 站点也仅用训练期筛选。1 月 29–31 日不参与本轮选型和评价。

站点预测按每个已知端点生成事件，和双端完整的空间子集覆盖不同。月初三小时归还可能缺前月订单，按未知处理。全城与站点指标分别解释，不能直接用原始 MAE 比较难度。

空间模块定义净流出 `F = 出发量 − 到达量`，失衡指数 `I = |F| / (出发量 + 到达量 + 1)`。净流出只表示压力，不等于库存短缺。分时段差额包含途中车辆与队列边界，小时曲线合并多个工作日，不能当作某一天的库存轨迹。

分区情景叠加“测试周预测 − 实测”修正，所以**不属于事前可用的未来调度方案**。缺站点电单车按各自借还事件统计；配额加 15% 余量后向上取整，可调出量向下取整，恰好 1 辆也参与匹配。来源和目标区域分别受累计供给、需求上限约束；没有可行路线时保留零覆盖结果。方案还缺实时库存、容量、人员与路网约束。

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
| 15–16 | 核心站点预测与共享模型重要性 | [逐站预测图](reports/figures/stations)、[图16](reports/figures/fig16_station_feature_importance.png) |
| 17–18 | 模型相对误差与逐站误差 | [图17](reports/figures/fig17_model_comparison.png)、[图18](reports/figures/fig18_station_error_comparison.png) |

图15是逐站预测系列，按站点、目标和评价窗口保存在 `reports/figures/stations/`。全部 PNG 按 300 DPI 输出。
图9–14 同时有内嵌图片和统计表的离线 HTML；GitHub 文件页展示源码，下载后用浏览器打开即可。

## 整合验证

59 项单元测试、46 项时间与偏好统计核验、230 项模型比较核验均通过。三份 Notebook 已分别在独立新内核中执行通过（18、9、18 个代码单元格）；135 项产物一致性检查通过，包括共享数据字节、预测输入与代码指纹、逐区域供需约束、报告链接和内嵌图。81 张 PNG 均为 300 DPI，整合前后的预测数值一致。

## 报告与数据表

报告按以下顺序阅读：

- [第1–4章：引言、清洗与基础探索](reports/sections/01_intro_and_cleaning.md)
- [第5章：时间规律与使用偏好](reports/sections/02_temporal_and_prediction.md)
- [第6.1节：全城需求预测](reports/sections/03_city_demand_prediction.md)
- [第6.2节：核心站点需求预测](reports/sections/04_station_demand_prediction.md)
- [第6.3节：模型比较与误差分析](reports/sections/05_model_comparison.md)
- [第7–9章：空间分布、失衡、调运与分区情景](reports/sections/03_spatial_and_dispatch.md)

离线报告：[时间与偏好](reports/temporal_analysis.html)、[全城预测](reports/demand_prediction.html)、[站点预测](reports/station_prediction.html)、[模型比较](reports/model_comparison.html)。

主要调度表：[站点历史候选配对](reports/tables/dispatch_plan.csv)、[区域信息](reports/tables/zone_profile.csv)、[分区配额](reports/tables/zone_preallocation_quota.csv)、[跨区路线](reports/tables/zone_dispatch_routes.csv)、[缺站点电单车热点](reports/tables/dockless_hotspots.csv)。预测明细、冻结推荐与核验结果保存在 `reports/tables/`。

整合来源：[马文博的数据基础](https://github.com/hhkQWQ/Pyclass/commit/777c65867a129d6f0dba03c0acfefbbce1e98cae)、[陈宇涵的预测与比较](https://github.com/hhkQWQ/Pyclass/commit/4f4774a12bda62b6ffebeffbd175856634b0913a)、[康炜奕的分区调度](https://github.com/hhkQWQ/Pyclass/commit/5f496ffce1f6c4d09fa669bbdb08e0324420083f)。
