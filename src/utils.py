"""成员 B 的项目路径与公共绘图接口。

字体、配色与 300 DPI 保存逻辑复用共享项目 src/utils.py，原版本：
https://github.com/hhkQWQ/Pyclass/blob/777c65867a129d6f0dba03c0acfefbbce1e98cae/src/utils.py
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
SECTION_DIR = REPORT_DIR / "sections"
TABLE_DIR = REPORT_DIR / "tables"


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
