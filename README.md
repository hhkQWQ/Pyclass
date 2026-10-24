# Pyclass

Final Group Assignment for Python Programming Course in the Second Year

## 共享单车数据分析项目：马文博的分步完成计划

基于 2024 年 1 月共享单车骑行数据，完成数据清洗、基础特征探索，并为时间预测和空间调度分析提供统一的数据接口。

**仓库：[hhkQWQ/Pyclass](https://github.com/hhkQWQ/Pyclass)｜负责人：马文博｜开发分支：`feat/data-cleaning-eda`**

## 当前进度

- [x] 根据《共享单车项目分工方案》整理马文博的任务、交付物和验收标准。
- [x] 核对本地原始 CSV：238,834 行、13 个字段。
- [ ] 完成数据清洗与双子集导出。
- [ ] 完成基础探索图 1–3、Notebook 和报告前四节。
- [ ] 完成全组联调、PPT、视频和最终报告统稿。

当前仓库内容为执行计划与上传说明，分析代码、图表和报告将在后续步骤中逐项完成。

## 从这里开始

1. 阅读 [马文博的分步完成计划](docs/mawenbo-plan.md)，按步骤完成代码、图表与报告。
2. 阅读 [现有仓库的 GitHub 协作说明](docs/github-upload.md)，克隆 `Pyclass` 并随实际进度提交成果。
3. 将原始数据仅保存在本地 `data/raw/题5_共享单车骑行_202401.csv`，清洗结果由脚本生成。

## 成员分工

| 成员 | 主要职责 | 文件和图表范围 |
| --- | --- | --- |
| 马文博（本人） | 数据工程、基础探索、项目统筹 | `src/utils.py`、`src/data_cleaning.py`、Notebook 01、报告前四节、图 1–3 |
| 陈宇涵 | 时间规律与需求预测 | `src/temporal_analysis.py`、`src/demand_model.py`、Notebook 02、图 4–8 |
| 康炜奕 | 空间分布、潮汐失衡与调度 | `src/spatial_analysis.py`、`src/imbalance_dispatch.py`、Notebook 03、图 9–12 |

## 数据与成果提交约定

- 上传代码、依赖清单、Notebook、报告草稿和图表。
- `data/raw/`、`data/processed/`、虚拟环境和缓存已列入 `.gitignore`。
- CSV 数据由组员在本地保存或生成，不进入 Git 提交。
- `main` 保存可交付版本，`dev` 用于联调，成员在各自功能分支开发。
- 每完成一个真实阶段再提交一次；提交记录应能对应代码、图表和报告贡献。

计划依据：用户提供的《共享单车项目分工方案.pdf》。清洗阈值、字段扩展和验收细节中新增的建议，已在详细计划中标明。
