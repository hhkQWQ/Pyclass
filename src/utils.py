"""项目路径、公共绘图样式和图片保存工具。"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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

RIDEABLE_COLORS = {"classic_bike": "#315D8A", "electric_bike": "#159A85"}
MEMBER_COLORS = {"member": "#315D8A", "casual": "#D78538"}
STRUCTURE_COLORS = {
    "station_complete": "#315D8A",
    "dockless_ebike_proxy": "#159A85",
    "other_missing_station": "#D78538",
    "different_station": "#315D8A",
    "loop_trip": "#159A85",
}
FONT_CANDIDATES = (
    "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", "SimHei",
    "PingFang SC", "WenQuanYi Micro Hei", "Arial Unicode MS",
)


def set_plot_style() -> str:
    """选择可显示中文的字体并设置统一样式，返回实际使用的字体名。"""
    import matplotlib as mpl
    from matplotlib import font_manager
    from cycler import cycler

    required = set(map(ord, "共享单车骑行时长距离会员站点清洗分流结构未知"))
    selected = None
    for family in FONT_CANDIDATES:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=[family]), fallback_to_default=False
            )
            if required.issubset(font_manager.get_font(path).get_charmap()):
                selected = family
                break
        except (ValueError, RuntimeError, OSError):
            continue
    if selected is None:
        for font in font_manager.fontManager.ttflist:
            try:
                if required.issubset(font_manager.get_font(font.fname).get_charmap()):
                    selected = font.name
                    break
            except (ValueError, RuntimeError, OSError):
                continue
    if selected is None:
        raise RuntimeError("未找到可显示中文的字体，请安装 Noto Sans CJK SC、微软雅黑或黑体后重试。")
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [selected, "DejaVu Sans"],
        "font.size": 11, "axes.unicode_minus": False,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "text.color": "#203548", "axes.labelcolor": "#203548",
        "xtick.color": "#526271", "ytick.color": "#526271",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#CAD3DA", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#DFE6EB",
        "grid.alpha": 0.8, "grid.linewidth": 0.7,
        "axes.prop_cycle": cycler(color=["#315D8A", "#159A85", "#D78538", "#AE5F67"]),
        "figure.dpi": 120, "savefig.dpi": 300, "pdf.fonttype": 42,
    })
    return selected


def save_figure(figure: "Figure", filename: str | Path, *, directory: Path = FIGURE_DIR) -> Path:
    """统一保存为白底、300 DPI 的 PNG；保留图对象供 Notebook 继续显示。"""
    path = Path(filename)
    if not path.suffix:
        path = path.with_suffix(".png")
    if path.suffix.lower() != ".png":
        raise ValueError("统一图片导出使用 .png 格式。")
    if not path.is_absolute():
        path = Path(directory) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    return path


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
