# 成员 B：时间分析与需求预测

当前已完成任务 1：工作日与周末的小时需求对比（图 4）、星期 × 小时需求热力图（图 5），以及统计核验和离线 HTML。后续任务继续在本目录扩展。

## 环境与运行

使用已有 Conda 环境 `Pyclass`，本地验证版本为 Python 3.13.5、NumPy 2.5.2、pandas 3.0.5、Matplotlib 3.11.0。当前环境已可直接运行，所需包记录在 `requirements.txt`。

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
│   └── temporal_analysis.py            # 时间统计、核验和报告生成
├── reports/
│   ├── figures/                        # 图 4、图 5
│   ├── sections/                       # 成员 B 的简明方法与结论
│   ├── tables/                         # 两份统计表与核验表
│   └── temporal_analysis.html          # 可单独复制、离线打开
├── requirements.txt
├── .gitattributes
├── .gitignore
└── README.md
```

目录命名与数据清洗项目保持一致，成员 B 的代码、报告及图表集中放在本目录。

## 输入数据与复用接口

`data/processed/clean_trips.csv` 是从 `../data-cleaning-eda/data/processed/clean_trips.csv` 复制的分析输入，共 233,180 行、21 列。该数据来源于清洗模块，不属于成员 B 新增的清洗成果；本步读取后不再次筛选。

数据版本：`777c65867a129d6f0dba03c0acfefbbce1e98cae`；SHA-256：`1ea4c2d6ccc091e73f3633ec2f58b551fc9c3d1aec5e58f235995c426879d7c0`。上游输入更新时，先核对版本与字段，再更新本目录的副本，避免两份数据混用。

`src/data_io.py` 沿用共享项目的字段类型读取规则；`src/utils.py` 沿用公共中文字体、配色和 300 DPI 保存接口，并已在代码中保留复用来源。运行本项目无需导入同学 A 的审查、清洗或基础探索模块。

## 统计口径

- 按骑行开始时间计数；周一至周五为工作日，周六、周日为周末。
- 核对月份覆盖后补齐 744 个小时格，无记录小时按订单计数口径记为 0。
- 图 4 分别除以工作日、周末天数；图 5 分别除以各星期出现的天数。
- 图表使用日均每小时次数，汇总表同时保留总次数和天数。

## 输出与核验

| 文件 | 内容 |
| --- | --- |
| [图 4](reports/figures/fig04_workday_weekend_hourly.png) | 工作日与周末小时需求曲线 |
| [图 5](reports/figures/fig05_weekday_hour_heatmap.png) | 星期 × 小时日均需求热力图 |
| [报告说明](reports/sections/02_temporal_and_prediction.md) | 数据结构、分析选择、绘图理由与结论 |
| [离线 HTML](reports/temporal_analysis.html) | 内嵌图片、汇总表和 CSV 下载 |
| [图 4 汇总表](reports/tables/fig04_hourly_by_day_type.csv) | 总次数、天数、日均值 |
| [图 5 汇总表](reports/tables/fig05_weekday_hourly.csv) | 各星期各小时的总次数、天数、日均值 |
| [统计核验表](reports/tables/temporal_validation_summary.csv) | 15 项检查的实际值、预期值和结果 |

运行时检查关键字段、订单唯一性、日期与小时覆盖、时间派生字段、分母和计数一致性；核验失败会停止。正常重复运行会更新本目录下同名输出。

## Git 管理

本目录和 `data-cleaning-eda` 由父目录 `D:\Pyclass\terminal` 的同一 Git 仓库管理，历史保留在父目录 `.git` 中。
