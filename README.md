# 成员 B：时间分析与需求预测

当前已完成任务 1、2：整体时间规律（图 4/5）、会员与车型的分时段差异（图 6A/6B/6C-1/6C-2），以及统计核验和离线 HTML。新增任务 3 的第一步：全城小时需求的整周预测（图 7/8）。站点筛选、站点预测及综合报告等待本步骤验收后另行开展。

## 环境与运行

使用已有 Conda 环境 `Pyclass`，本地验证版本为 Python 3.13.5、NumPy 2.5.2、pandas 3.0.5、Matplotlib 3.11.0、scikit-learn 1.9.1。当前环境已可直接运行，所需包记录在 `requirements.txt`。

在 PowerShell 中运行：

```powershell
Set-Location 'D:\Pyclass\terminal\temporal_modeling'
conda run -n Pyclass python -m src.temporal_analysis
```

也可使用 Pyclass 的 Python 直接运行 `src/temporal_analysis.py`。默认输入、输出位置均根据本项目所在目录确定。

## 项目目录

```text
temporal_modeling/
├── data/processed/clean_trips.csv       # 本步需要的输入数据副本
├── src/
│   ├── __init__.py
│   ├── data_io.py                      # 读取有效集并恢复字段类型
│   ├── utils.py                        # 路径、公共绘图样式与图片保存
│   ├── preference_analysis.py          # 会员、车型、交叉统计与绘图
│   └── temporal_analysis.py            # 统一入口、时间统计、核验和报告生成
├── reports/
│   ├── figures/                        # 图 4/5、图 6 的三个图组（共六张 PNG）
│   ├── sections/                       # 成员 B 的简明方法与结论
│   ├── tables/                         # 逐小时统计、峰值摘要与核验表
│   └── temporal_analysis.html          # 可单独复制、离线打开
├── requirements.txt
├── .gitattributes
├── .gitignore
└── README.md
```

目录命名与数据清洗项目保持一致，成员 B 的代码、报告及图表集中放在本目录。

## 输入数据与复用接口

`data/processed/clean_trips.csv` 是清洗模块交付的分析输入副本，共 233,180 行、21 列。该数据不属于成员 B 新增的清洗成果；本步读取后不再次筛选。上游目录可能调整，以以下版本与文件校验值核对输入。

数据版本：`777c65867a129d6f0dba03c0acfefbbce1e98cae`；SHA-256：`1ea4c2d6ccc091e73f3633ec2f58b551fc9c3d1aec5e58f235995c426879d7c0`。上游输入更新时，先核对版本与字段，再更新本目录的副本，避免两份数据混用。

`src/data_io.py` 沿用共享项目的字段类型读取规则；`src/utils.py` 沿用公共中文字体、配色和 300 DPI 保存接口，并已在代码中保留复用来源。运行本项目无需导入同学 A 的审查、清洗或基础探索模块。

## 统计口径

- 按骑行开始时间计数；周一至周五为工作日，周六、周日为周末。
- 核对月份覆盖后补齐 744 个小时格，无记录小时按订单计数口径记为 0。
- 图 4 分别除以工作日、周末天数；图 5 分别除以各星期出现的天数。
- 图表使用日均每小时次数，汇总表同时保留总次数和天数。
- 会员、车型与四种交叉组合分别计算工作日、周末的日均次数和组内时段占比。时段占比分母为该组在对应日期类型内的全天总次数，24 小时合计 100%。
- 人群内部的逐小时车型占比，以该人群该小时的全部骑行为分母，两种车型合计 100%。整体车型占比按次数加权计算，不简单平均小时百分比。
- 补齐 744 小时 × 2 类人群 × 2 种车型，共 2,976 个计数组合；比例分母为零时保留缺失值，图中留空、HTML 显示无数据。

## 输出与核验

| 文件 | 内容 |
| --- | --- |
| [图 4](reports/figures/fig04_workday_weekend_hourly.png) | 工作日与周末小时需求曲线 |
| [图 5](reports/figures/fig05_weekday_hour_heatmap.png) | 星期 × 小时日均需求热力图 |
| [图 6A](reports/figures/fig06a_membership_hourly.png) | 会员与散客：日均次数和组内时段占比 |
| [图 6B](reports/figures/fig06b_vehicle_hourly.png) | 经典车与电单车：日均次数和组内时段占比 |
| [图 6C-1](reports/figures/fig06c1_member_vehicle_hourly.png) | 四种会员 × 车型组合的小时分布 |
| [图 6C-2](reports/figures/fig06c2_vehicle_mix_by_member.png) | 各人群每小时的两种车型构成 |
| [报告说明](reports/sections/02_temporal_and_prediction.md) | 数据结构、分析选择、绘图理由与结论 |
| [离线 HTML](reports/temporal_analysis.html) | 内嵌图片、汇总表和 CSV 下载 |
| [图 4 汇总表](reports/tables/fig04_hourly_by_day_type.csv) | 总次数、天数、日均值 |
| [图 5 汇总表](reports/tables/fig05_weekday_hourly.csv) | 各星期各小时的总次数、天数、日均值 |
| [会员汇总表](reports/tables/fig06_membership_hourly.csv) | 96 行：日期类型 × 小时 × 人群 |
| [车型汇总表](reports/tables/fig06_vehicle_hourly.csv) | 96 行：日期类型 × 小时 × 车型 |
| [交叉汇总表](reports/tables/fig06_member_vehicle_hourly.csv) | 192 行：日期类型 × 小时 × 人群 × 车型 |
| [车型构成表](reports/tables/fig06_vehicle_mix_by_member.csv) | 192 行：各人群各小时的车型占比及分母 |
| [峰值与总量摘要](reports/tables/fig06_preference_summary.csv) | 16 行：日均总量、最高小时、最高小时次数与占比 |
| [统计核验表](reports/tables/temporal_validation_summary.csv) | 两项任务各项检查的实际值、预期值和结果 |

运行时检查关键字段、订单唯一性、日期与小时覆盖、时间派生字段、分母和计数一致性，并核对交叉表边际与两类占比合计；核验失败会停止。正常重复运行会更新本目录下同名输出。

图 6 CSV 中的 `time_share`、`vehicle_share`、`peak_time_share` 均为 0–1 小数，页面展示时转换成百分比。最高小时并列时全部保留。HTML 将关键发现和主图直接呈现，明细、补充图、公式及核验结果可展开查看；复制单个 HTML 文件即可离线阅读与下载 CSV。

## Git 管理

沿用本地现有 Git 仓库和历史；本步骤不提交、更改分支或清理已有修改。

## 全城整周预测：任务 3 / 第一步

```powershell
conda run -n Pyclass python -m src.demand_model
conda run -n Pyclass python -B -m unittest discover -s tests -p test_demand_model.py -v
```

独立入口 `src/demand_model.py` 支持 `--input` 和 `--output-dir`。它不调用时间分析的旧入口，也不覆盖图 4/5/6 或前两项报告。

- 每周一 00:00 一次预测未来 168 小时，周内不回填真实需求。
- 用 1 月 1–14 日预测 15–21 日作验证，按 MAE 选择参数及推荐方法；再用 1–21 日重训，预测 22–28 日作最终测试，不反向调参。
- 比较历史同期均值、上周同期、Ridge、随机森林。学习模型共用日历信息和上周同期需求；去掉前 168 小时无滞后的训练行后，分别拟合 168 / 336 行。
- 补充 RMSE、R²、高峰 MAE 和日／周总量偏差。高峰仅按训练期确定。骑行次数负预测截为 0，评估保留小数。
- 1 月 29–31 日不参与本轮选型与评估，不输出跨月预测。只有一次验证与一次最终测试，不能保证长期稳定性。

| 新增位置 | 内容 |
|---|---|
| [预测 HTML](reports/demand_prediction.html) | 先介绍模型与指标，再展示结果；图片内嵌，可独立离线阅读 |
| [方法与结果](reports/sections/03_city_demand_prediction.md) | 四种方法的原理、拟合、理由、优点、局限与有效性证据 |
| [图 7](reports/figures/fig07_city_weekly_prediction.png) | 最终测试周：真实值、历史均值、Ridge和随机森林 |
| [验证周补充图](reports/figures/fig07_city_validation_prediction.png) | 相同三种预测与真实值，HTML中展开查看 |
| [图 8](reports/figures/fig08_city_feature_importance.png) | 随机森林内部特征重要性（非因果解释） |
| `reports/tables/city_hourly_demand.csv` | 744 小时原始计数 |
| `reports/tables/city_weekly_predictions.csv` | 1,344 行：两周 × 四种方法 × 168 小时，附预测起点和滞后来源 |
| `reports/tables/city_model_metrics.csv` | 两阶段四种方法的误差、拟合行数与截断次数 |
| `reports/tables/city_parameter_candidates.csv` | 所有预定参数组合的验证表现与选择记录 |
| `reports/tables/city_daily_weekly_totals.csv` | 每种方法的日、周总量及偏差 |
| `reports/tables/city_model_selection.json` | 测试前冻结的选择、参数网格、特征、输入和代码指纹 |
| `reports/tables/city_feature_importance.csv` | 随机森林逐特征重要性及分组 |
| `reports/tables/city_prediction_validation.csv` | 运行时的输入、训练和预测边界核验 |
| `tests/test_demand_model.py` | 手算基线、信息泄漏、边界与指标等独立测试 |

CSV 保留 10 位小数；重复运行更新本步骤同名输出。交付检查另记录原成果校验值、重复运行一致性、独立重算及页面离线检查。

HTML 按“认识模型 → 理解 MAE/RMSE/R² → 预测结果 → 结论与局限”组织。模型基础介绍直接显示；验证周曲线与随机森林特征重要性可展开。页面不显示上周同期预测曲线和结果行，也不显示核验、环境、运行命令或工程下载。完整四方法评估和工程记录仍保存在CSV、JSON、MD及成果包中。
