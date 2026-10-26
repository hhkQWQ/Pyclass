"""项目路径、公共绘图样式和图片保存工具。"""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    import pandas as pd


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


HTML_STYLE = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 18px 56px; background: #F4F7F9;
       font-family: "Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", sans-serif;
       color: #203548; }
main { max-width: 1180px; margin: 0 auto; background: #FFFFFF;
       border: 1px solid #DFE6EB; border-radius: 14px; padding: 30px 34px 34px;
       box-shadow: 0 10px 30px rgba(32, 53, 72, 0.06); }
h1 { font-size: 22px; margin: 0 0 6px; letter-spacing: 0.4px; }
.subtitle { margin: 0 0 22px; color: #657585; font-size: 13.5px; line-height: 1.7; }
figure { margin: 0; }
img { display: block; width: 100%; height: auto; border-radius: 10px; border: 1px solid #E8EEF2; }
figcaption { margin-top: 14px; font-size: 12.8px; color: #526271; line-height: 1.8; }
table { width: 100%; border-collapse: collapse; margin-top: 22px; font-size: 12.6px; }
caption { caption-side: top; text-align: left; font-weight: 600; padding-bottom: 8px; }
th, td { padding: 7px 10px; border-bottom: 1px solid #E8EEF2; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; white-space: normal; }
thead th { background: #F0F4F7; color: #203548; border-bottom: 1px solid #D3DDE5; }
tbody tr:nth-child(even) { background: #FAFCFD; }
footer { margin-top: 26px; padding-top: 14px; border-top: 1px dashed #DFE6EB;
         color: #8093A3; font-size: 11.8px; line-height: 1.8; }
"""


def _html_table(frame: "pd.DataFrame", caption: str, digits: int = 2) -> str:
    """把统计表渲染成静态 HTML 表格；离线打开也能看到与正文一致的数值。"""
    header = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for record in frame.itertuples(index=False):
        cells = []
        for value in record:
            text = f"{value:,.{digits}f}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><caption>{escape(caption)}</caption>"
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def save_figure_html(
    figure: "Figure",
    filename: str | Path,
    *,
    title: str,
    subtitle: str = "",
    note: str = "",
    table: "pd.DataFrame | None" = None,
    table_caption: str = "",
    footer: str = "数据来源：data/processed/station_trips.csv（站点有效集）。"
                  "图片由 matplotlib 生成并以 base64 内嵌，单文件离线可看。",
    directory: Path = FIGURE_DIR,
) -> tuple[Path, Path]:
    """同时导出 300 DPI PNG 与自包含 HTML；HTML 内嵌 150 DPI 缩图，便于直接分发。"""
    stem = Path(filename)
    if stem.suffix.lower() in {".html", ".png"}:
        stem = stem.with_suffix("")
    png_path = save_figure(figure, stem.with_suffix(".png"), directory=directory)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight",
                   pad_inches=0.25, facecolor="white")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    caption = f"<figcaption>{escape(note)}</figcaption>" if note else ""
    table_html = (
        _html_table(table, table_caption) if table is not None and not table.empty else ""
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>{HTML_STYLE}</style>
</head>
<body>
<main>
<h1>{escape(title)}</h1>
<p class="subtitle">{escape(subtitle)}</p>
<figure>
<img src="data:image/png;base64,{encoded}" alt="{escape(title)}">
{caption}
</figure>
{table_html}
<footer>{escape(footer)}</footer>
</main>
</body>
</html>
"""
    html_path = Path(directory) / stem.with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    return png_path, html_path


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
