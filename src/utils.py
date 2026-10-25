"""项目路径与目录准备工具。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
CLEANING_RULES_FILE = CONFIG_DIR / "cleaning_rules.json"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
SECTION_DIR = REPORT_DIR / "sections"
TABLE_DIR = REPORT_DIR / "tables"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
RAW_DATA_FILE = RAW_DIR / "题5_共享单车骑行_202401.csv"


def ensure_project_dirs() -> None:
    """创建本地数据、Notebook 和报告目录，可重复调用。"""
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        NOTEBOOK_DIR,
        FIGURE_DIR,
        SECTION_DIR,
        TABLE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_project_dirs()
    print("项目目录已准备好。")
