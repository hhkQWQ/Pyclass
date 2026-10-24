# Pyclass 仓库的 GitHub 协作说明

使用小组现有仓库：**[hhkQWQ/Pyclass](https://github.com/hhkQWQ/Pyclass)**。

本次计划提交到 `feat/data-cleaning-eda`，通过 Pull Request 合入 `dev`。后续完成的分析成果继续按阶段提交；全组联调通过后，再将 `dev` 合入 `main`。

## 1. 克隆仓库并切换到 A 的分支

首次在自己的电脑上使用仓库时，在 PowerShell 中执行：

```powershell
git clone https://github.com/hhkQWQ/Pyclass.git
Set-Location -LiteralPath '.\Pyclass'
git switch feat/data-cleaning-eda
git remote -v
git status
```

已克隆仓库的同学，进入已有的 `Pyclass` 目录后执行 `git fetch origin`，再切换到对应分支，无需重新初始化仓库。

分支用途：

| 分支 | 用途 |
| --- | --- |
| `main` | 可交付的稳定版本 |
| `dev` | 三人模块联调 |
| `feat/data-cleaning-eda` | A 的数据清洗、基础探索、计划与报告草稿 |
| `feat/temporal-and-modeling` | B 开发时从 `dev` 创建 |
| `feat/spatial-and-dispatch` | C 开发时从 `dev` 创建 |

## 2. 后续每完成一步就提交一次

每次开始工作前，在 A 分支获取最新联调内容：

```powershell
git switch feat/data-cleaning-eda
git fetch origin
git merge origin/dev
```

先提交或妥善保存本地未完成改动，再执行合并。若团队已采用其他同步方式，遵循既有约定。

例如完成公共工具后，只暂存相关文件并检查：

```powershell
git status --short
git add src/utils.py
git diff --cached
git commit -m "feat: add shared plotting utilities"
git push -u origin feat/data-cleaning-eda
```

如果 Git 提示缺少作者身份，按提示配置自己的姓名和 GitHub 已验证邮箱；不要使用他人的身份。推送时按 Git 提示完成登录。

完成一阶段后在 GitHub 建立 Pull Request：来源为 `feat/data-cleaning-eda`，目标为 `dev`。描述本次交付、复现方式和检查结果。若该分支已有未合并的 Pull Request，后续推送会更新它，无需重复创建。

## 3. 放置本地数据并检查忽略规则

原始文件保存在本地 `data/raw/题5_共享单车骑行_202401.csv`。例如，将仓库克隆到本次工作目录后，在 `Pyclass` 根目录执行：

```powershell
New-Item -ItemType Directory -Force -Path '.\data\raw' | Out-Null
Copy-Item -LiteralPath '..\题5_共享单车骑行_202401.csv' -Destination '.\data\raw\题5_共享单车骑行_202401.csv'
git check-ignore data/raw/题5_共享单车骑行_202401.csv
git status --short
```

`git check-ignore` 应输出该文件路径，`git status` 中不应出现该 CSV。其他组员需要将复制命令的源路径替换为自己电脑上的数据位置。

清洗结果保存在 `data/processed/`，由代码生成；目录尚不存在时由脚本创建。原始数据、清洗数据、缓存和虚拟环境已加入 `.gitignore`。

不要将原始 CSV 通过网页直接上传；网页上传不替你执行本地 `.gitignore` 检查。如果数据先前已被 Git 跟踪，新增忽略规则也不会自动移除跟踪，需要单独处理后再提交。

## 4. 提交前后检查

- [ ] 文件只包含本次实际完成的改动。
- [ ] 提交作者对应本人账号，提交说明对应实际工作。
- [ ] 仓库中没有原始或清洗后的 CSV、虚拟环境和缓存。
- [ ] README 和文档之间的链接可以打开。
- [ ] A 的 Pull Request 指向 `dev`。
- [ ] 后续成果按阶段提交，未完成事项继续保持未勾选状态。

## 5. 这次提交包含什么

本次包含 README 的项目入口、A 的 8 步执行计划、本协作说明和 `.gitignore`，保留仓库原有英文介绍。清洗脚本、Notebook、图表和报告在后续阶段完成。
