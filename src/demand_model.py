"""陈宇涵 第三项 / 步骤一：在自然周开始前预测未来 168 小时。

运行：python -m src.demand_model
所有预测输入均在 forecast_origin 前可获得；验证选型，最终测试只评估。
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_io import read_clean_data
from src.utils import PROCESSED_DIR, REPORT_DIR, save_figure, set_plot_style

MONTH_START = pd.Timestamp("2024-01-01")
MONTH_END = pd.Timestamp("2024-02-01")
PHASES = {
    "validation": (pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-22")),
    "test": (pd.Timestamp("2024-01-22"), pd.Timestamp("2024-01-29")),
}
PHASE_NAMES = {"validation": "验证周", "test": "最终测试周"}
MODEL_NAMES = {"historical_mean": "历史同期均值", "last_week": "上周同期",
               "ridge": "Ridge 岭回归", "random_forest": "随机森林"}
MODEL_ORDER = tuple(MODEL_NAMES)
DISPLAY_MODELS = ("historical_mean", "ridge", "random_forest")
PARAMETER_GRIDS = {
    "historical_mean": [{}], "last_week": [{}],
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
    "random_forest": [{"n_estimators": 200, "max_depth": d,
                       "min_samples_leaf": n, "random_state": 42, "n_jobs": 1}
                      for d in (3, 5) for n in (8, 16)],
}
COLORS = {"historical_mean": "#D18B42", "last_week": "#A16989",
          "ridge": "#168D83", "random_forest": "#3267A5"}


def build_hourly(data: pd.DataFrame) -> pd.Series:
    """只按有效订单开始时间计数；不因站点信息缺失删除订单。"""
    if data.empty or data.ride_id.isna().any() or not data.ride_id.is_unique:
        raise ValueError("输入为空，或订单编号缺失/重复。")
    started = data.started_at
    if started.isna().any() or not started.between(MONTH_START, MONTH_END, inclusive="left").all():
        raise ValueError("开始时间缺失或超出 2024 年 1 月。")
    for name, values in {"hour": started.dt.hour, "weekday": started.dt.dayofweek,
                         "is_weekend": started.dt.dayofweek.ge(5)}.items():
        if data[name].isna().any() or not data[name].eq(values).all():
            raise ValueError(f"{name} 与开始时间不一致。")
    dates = pd.date_range(MONTH_START, MONTH_END, inclusive="left", freq="D")
    missing = dates.difference(pd.DatetimeIndex(started.dt.normalize().unique()))
    if len(missing):
        raise ValueError("整天没有记录，不能直接当作零需求：" + str(list(missing)))
    hours = pd.date_range(MONTH_START, MONTH_END, inclusive="left", freq="h")
    y = started.dt.floor("h").value_counts().reindex(hours, fill_value=0).astype("int64")
    y.index.name, y.name = "timestamp", "ride_count"
    if int(y.sum()) != len(data):
        raise ValueError("小时计数不守恒。")
    return y


def features(history: pd.Series, index: pd.DatetimeIndex,
             forecast_origin: pd.Timestamp | None = None) -> pd.DataFrame:
    """日历 + 上周同期。训练行无周滞后时保留 NaN，拟合时删除。

    预测必须传入截止前的历史；整个 168 小时矩阵一次构建，不回填周内真实值。
    """
    index = pd.DatetimeIndex(index)
    if not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("历史时间索引必须唯一且递增。")
    sources = index - pd.Timedelta(hours=168)
    if forecast_origin is not None:
        if (history.index >= forecast_origin).any():
            raise ValueError("预测历史包含预测起点之后的数据。")
        if (index < forecast_origin).any() or (sources >= forecast_origin).any():
            raise ValueError("预测区间或上周同期来源超出周初可知范围。")
    x = pd.DataFrame(index=index)
    weekend = (index.dayofweek >= 5).astype(float)
    for period in (24, 12):
        angle = index.hour.to_numpy() * (2 * np.pi / period)
        x[f"hour_sin_{period}"] = np.sin(angle)
        x[f"hour_cos_{period}"] = np.cos(angle)
    for day in range(1, 7):
        x[f"weekday_{day}"] = (index.dayofweek == day).astype(float)
    x["is_weekend"] = weekend
    for name in list(x.columns[:4]):
        x[f"weekend_x_{name}"] = weekend * x[name]
    x["lag_168h"] = history.reindex(sources).to_numpy(dtype=float)
    return x


@dataclass
class Fitted:
    model_id: str
    history: pd.Series
    estimator: object | None
    fit_rows: int
    feature_names: list[str]

    def predict(self, index: pd.DatetimeIndex, origin: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        x = features(self.history, index, origin)
        if x.isna().any().any():
            raise ValueError("预测所需的历史同期数据不完整。")
        if self.model_id == "historical_mean":
            means = self.history.groupby([self.history.index.dayofweek, self.history.index.hour]).mean()
            raw = np.array([means.loc[(d, h)] for d, h in zip(index.dayofweek, index.hour)], dtype=float)
        elif self.model_id == "last_week":
            raw = x.lag_168h.to_numpy()
        else:
            raw = np.asarray(self.estimator.predict(x), dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("模型生成了非有限预测值。")
        return raw, np.maximum(raw, 0.0)


def fit_model(model_id: str, history: pd.Series, parameters: dict) -> Fitted:
    if len(history) < 168 or history.isna().any() or (history < 0).any():
        raise ValueError("至少需要完整一周的非负历史计数。")
    expected = pd.date_range(history.index.min(), history.index.max(), freq="h")
    if not history.index.equals(expected):
        raise ValueError("历史小时网格不连续。")
    if model_id in ("historical_mean", "last_week"):
        return Fitted(model_id, history.copy(), None, len(history), [])
    x = features(history, history.index).dropna()
    if x.empty:
        raise ValueError("没有可用于学习上周同期关系的样本。")
    if model_id == "ridge":
        estimator = make_pipeline(StandardScaler(), Ridge(**parameters))
    elif model_id == "random_forest":
        estimator = RandomForestRegressor(**parameters)
    else:
        raise ValueError(f"未知模型：{model_id}")
    estimator.fit(x, history.loc[x.index])
    return Fitted(model_id, history.copy(), estimator, len(x), list(x.columns))


def peak_mask(history: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """根据预测前历史，分别取工作日/周末平均需求最高的两个小时。"""
    means = history.groupby([history.index.dayofweek >= 5, history.index.hour]).mean()
    result = np.zeros(len(index), dtype=bool)
    for weekend in (False, True):
        group = means.loc[weekend]
        chosen = sorted(group.index, key=lambda h: (-float(group.loc[h]), h))[:2]
        result |= ((index.dayofweek >= 5) == weekend) & np.isin(index.hour, chosen)
    return result


def scores(actual: np.ndarray, predicted: np.ndarray, peaks: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if np.var(actual) > 0 else None,
        "peak_mae": float(mean_absolute_error(actual[peaks], predicted[peaks])) if peaks.any() else None,
        "total_actual": float(np.sum(actual)), "total_predicted": float(np.sum(predicted)),
        "total_bias": float(np.sum(predicted) - np.sum(actual)),
    }


@dataclass
class Experiment:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    candidates: pd.DataFrame
    selection: dict
    fits: dict[str, Fitted]
    audit: pd.DataFrame


def run_experiment(y: pd.Series, grids: dict | None = None) -> Experiment:
    """先完成验证选型，冻结后才读取测试标签。"""
    grids = PARAMETER_GRIDS if grids is None else grids
    val_origin, val_end = PHASES["validation"]
    train = y.loc[(y.index >= MONTH_START) & (y.index < val_origin)]
    val_index = pd.date_range(val_origin, val_end, inclusive="left", freq="h")
    val_actual = y.reindex(val_index).to_numpy(dtype=float)
    if not np.isfinite(val_actual).all():
        raise ValueError("验证标签不完整。")
    val_peaks = peak_mask(train, val_index)
    candidates, chosen, val_fits, val_values = [], {}, {}, {}
    for model_id in MODEL_ORDER:
        ranked = []
        for candidate_id, parameters in enumerate(grids[model_id]):
            fitted = fit_model(model_id, train, parameters)
            raw, pred = fitted.predict(val_index, val_origin)
            metric = scores(val_actual, pred, val_peaks)
            candidates.append({"model": model_id, "candidate_id": candidate_id,
                               "parameters": json.dumps(parameters, sort_keys=True), **metric})
            ranked.append((metric["mae"], metric["rmse"], candidate_id, fitted, raw, pred))
        best = min(ranked, key=lambda item: item[:3])
        chosen[model_id] = dict(grids[model_id][best[2]])
        val_fits[model_id], val_values[model_id] = best[3], (best[4], best[5])
    candidate_table = pd.DataFrame(candidates)
    candidate_table["selected_within_family"] = candidate_table.apply(
        lambda r: r.parameters == json.dumps(chosen[r.model], sort_keys=True), axis=1)
    best_rows = candidate_table.loc[candidate_table.selected_within_family]
    recommended = min(MODEL_ORDER, key=lambda key: (
        float(best_rows.loc[best_rows.model.eq(key), "mae"].iloc[0]),
        float(best_rows.loc[best_rows.model.eq(key), "rmse"].iloc[0]), MODEL_ORDER.index(key)))
    # 此对象在读取最终测试标签前已经确定；测试结果不参与选型。
    selection = {"selection_metric": "validation_mae", "tie_break": "validation_rmse_then_predefined_order",
                 "recommended_model": recommended, "selected_parameters": chosen,
                 "parameter_grids": grids,
                 "validation_origin": str(val_origin), "test_origin": str(PHASES["test"][0]),
                 "within_week_actual_updates": False, "post_test_tuning": False,
                 "forecast_horizon_hours": 168}
    predictions, metric_rows, audits, test_fits = [], [], [], {}

    def check(phase: str, item: str, observed, expected) -> None:
        audits.append({"phase": phase, "check": item, "observed": str(observed),
                       "expected": str(expected), "passed": bool(observed == expected)})

    for phase, (origin, end) in PHASES.items():
        history = y.loc[(y.index >= MONTH_START) & (y.index < origin)]
        index = pd.date_range(origin, end, inclusive="left", freq="h")
        actual = y.reindex(index).to_numpy(dtype=float)
        if not np.isfinite(actual).all():
            raise ValueError("评估标签不完整。")
        peaks = peak_mask(history, index)
        check(phase, "forecast_hours", len(index), 168)
        check(phase, "forecast_starts_monday_00", (origin.dayofweek, origin.hour), (0, 0))
        check(phase, "history_hours", len(history), 336 if phase == "validation" else 504)
        check(phase, "latest_history_before_origin", bool(history.index.max() < origin), True)
        check(phase, "latest_lag_source_before_origin", bool((index - pd.Timedelta(hours=168)).max() < origin), True)
        for model_id in MODEL_ORDER:
            if phase == "validation":
                fitted, (raw, pred) = val_fits[model_id], val_values[model_id]
            else:
                fitted = fit_model(model_id, history, chosen[model_id])
                raw, pred = fitted.predict(index, origin)
                test_fits[model_id] = fitted
            metric_rows.append({"phase": phase, "model": model_id,
                                "recommended_frozen": model_id == recommended,
                                "history_hours": len(history), "fit_rows": fitted.fit_rows,
                                "negative_raw_predictions": int((raw < 0).sum()),
                                **scores(actual, pred, peaks)})
            predictions.append(pd.DataFrame({
                "phase": phase, "forecast_origin": origin, "timestamp": index,
                "model": model_id, "actual": actual, "raw_prediction": raw,
                "prediction": pred, "lag_168h": history.reindex(index - pd.Timedelta(hours=168)).to_numpy(),
                "lag_source_time": index - pd.Timedelta(hours=168), "is_peak": peaks,
                "recommended_frozen": model_id == recommended,
            }))
            check(phase, f"{model_id}:finite_nonnegative", bool(np.isfinite(pred).all() and (pred >= 0).all()), True)
            if model_id in ("ridge", "random_forest"):
                check(phase, f"{model_id}:fit_rows_after_week_lag", fitted.fit_rows, len(history) - 168)
    audit = pd.DataFrame(audits)
    if not audit.passed.all():
        raise ValueError("预测边界核验失败：" + audit.loc[~audit.passed, "check"].to_string(index=False))
    return Experiment(pd.concat(predictions, ignore_index=True), pd.DataFrame(metric_rows),
                      candidate_table, selection, test_fits, audit)


def aggregate_totals(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["date"] = frame.timestamp.dt.normalize()
    daily = frame.groupby(["phase", "model", "date"], sort=False)[["actual", "prediction"]].sum().reset_index()
    daily["period"] = "day"
    weekly = frame.groupby(["phase", "model"], sort=False).agg(
        date=("timestamp", "min"), actual=("actual", "sum"), prediction=("prediction", "sum")).reset_index()
    weekly["period"] = "week"
    result = pd.concat([daily, weekly], ignore_index=True)
    result["bias"] = result.prediction - result.actual
    result["absolute_percentage_error"] = np.where(result.actual > 0, abs(result.bias) / result.actual, np.nan)
    return result


def feature_importances(experiment: Experiment) -> pd.DataFrame:
    fitted = experiment.fits["random_forest"]
    group_names = {"hour": "小时周期", "weekday": "星期类别", "weekend": "周末标记",
                   "interaction": "周末与小时的交互", "lag": "上周同小时需求"}
    def group(name):
        if name == "lag_168h": return "lag"
        if name.startswith("weekend_x_"): return "interaction"
        if name.startswith("hour_"): return "hour"
        if name.startswith("weekday_"): return "weekday"
        return "weekend"
    details = pd.DataFrame({"feature": fitted.feature_names,
                            "importance": fitted.estimator.feature_importances_})
    details["group"] = details.feature.map(group)
    details["group_label"] = details.group.map(group_names)
    return details


def plot_results(experiment: Experiment, directory: Path) -> list[Path]:
    recommended = experiment.selection["recommended_model"]
    curves = {}
    for phase in ("test", "validation"):
        fig, ax = plt.subplots(figsize=(12.5, 4.8), layout="constrained")
        frame = experiment.predictions.loc[experiment.predictions.phase.eq(phase)]
        actual = frame.loc[frame.model.eq("historical_mean")]
        ax.plot(actual.timestamp, actual.actual, color="#253342", linewidth=2.2, label="真实骑行量", zorder=6)
        for model_id in DISPLAY_MODELS:
            part = frame.loc[frame.model.eq(model_id)]
            label = MODEL_NAMES[model_id] + ("（验证选定）" if model_id == recommended else "")
            ax.plot(part.timestamp, part.prediction, color=COLORS[model_id],
                    linewidth=1.9 if model_id == recommended else 1.2,
                    alpha=0.95 if model_id == recommended else 0.78,
                    linestyle="-" if model_id == recommended else "--", label=label)
        prefix = "图 7  最终测试周" if phase == "test" else "补充图  验证周"
        ax.set(title=f"{prefix} · {actual.timestamp.min():%m/%d}–{actual.timestamp.max():%m/%d}",
               xlabel="日期", ylabel="骑行次数", ylim=(0, None))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.set_xlim(actual.timestamp.min(), actual.timestamp.max())
        ax.legend(ncol=2, loc="upper left", fontsize=9, frameon=False)
        ax.margins(y=0.22)
        filename = "fig07_city_weekly_prediction.png" if phase == "test" else "fig07_city_validation_prediction.png"
        curves[phase] = save_figure(fig, filename, directory=directory)
        plt.close(fig)
    importance = feature_importances(experiment).groupby("group_label").importance.sum().sort_values()
    fig, ax = plt.subplots(figsize=(10.5, 4.2), layout="constrained")
    ax.barh(importance.index, importance.values, color="#328F89", height=0.62)
    for i, value in enumerate(importance.values):
        ax.text(value + 0.008, i, f"{value:.1%}", va="center", fontsize=10)
    ax.set(title="图 8  随机森林使用哪些信息", xlabel="内部重要性占比", xlim=(0, max(0.15, float(importance.max()) * 1.25)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    explanation = save_figure(fig, "fig08_city_feature_importance.png", directory=directory)
    plt.close(fig)
    return [curves["test"], explanation, curves["validation"]]


def data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def display_table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, escape=True, na_rep="无定义",
                         float_format=lambda v: f"{v:,.2f}", classes="data-table")


def metric_display(experiment: Experiment, phase: str) -> pd.DataFrame:
    frame = experiment.metrics.loc[experiment.metrics.phase.eq(phase) & experiment.metrics.model.isin(DISPLAY_MODELS)].copy()
    frame["model"] = frame.model.map(MODEL_NAMES)
    return frame[["model", "mae", "rmse", "r2"]].rename(columns={
        "model": "方法", "mae": "MAE ↓", "rmse": "RMSE ↓", "r2": "R² ↑"})


def narrative(experiment: Experiment) -> dict:
    recommended = experiment.selection["recommended_model"]
    test = experiment.metrics.loc[experiment.metrics.phase.eq("test")].set_index("model")
    validation = experiment.metrics.loc[experiment.metrics.phase.eq("validation")].set_index("model")
    selected = test.loc[recommended]
    improvements = {key: ((float(test.loc[key, "mae"]) - float(selected.mae)) / float(test.loc[key, "mae"])
                          if float(test.loc[key, "mae"]) > 0 else None)
                    for key in ("historical_mean", "last_week")}
    if recommended in ("historical_mean", "last_week"):
        verdict = "验证周选中了简单对照方法；本次没有预先选出优于它的学习模型。"
    elif all(value is not None and value > 0 for value in improvements.values()):
        verdict = "验证选定的学习模型在最终测试周也优于两种简单对照，获得本次周预测的初步改进证据。"
    else:
        verdict = "验证选定的学习模型未在最终测试周同时优于两种简单对照，尚无稳定改进证据。"
    return {"recommended": recommended, "name": MODEL_NAMES[recommended], "selected": selected,
            "validation": validation, "test": test, "improvements": improvements, "verdict": verdict,
            "week_bias_percentage": (float(selected.total_bias) / float(selected.total_actual)
                                     if float(selected.total_actual) > 0 else None)}


def format_value(value, spec: str) -> str:
    if value is None or not np.isfinite(value):
        return "无定义"
    return format(value, spec)


def json_safe(value):
    """JSON 中无定义指标使用 null，不输出非标准 NaN。"""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


METHODS_HTML = """
<h3>历史同期均值：按过去的星期与小时求平均</h3>
<p>例如预测周二 8 时，就平均训练期所有周二 8 时的骑行次数。验证时每个组合只有 2 次历史观测，测试时为 3 次。无需迭代拟合，容易复核；会平滑短期变化，也可能落后于近期需求变化。</p>
<h3>上周同期：重复上一周</h3>
<p>公式为 <code>预测值(t) = 实际值(t−168小时)</code>。整周对应值在周初都已知。它是周预测的直接对照，但上周的异常变化也会被带入下一周。</p>
<h3>Ridge 岭回归：带系数约束的线性拟合</h3>
<p>模型用 <code>预测值 = 截距 + 各特征×对应系数之和</code> 拟合需求，同时惩罚过大的系数（L2 正则化）。小时用 24 小时和 12 小时正余弦表示；星期按类别编码，加入周末与小时的交互及上周同期需求。特征仅按训练集计算均值和标准差，再使用同样规则处理未来周。优点是结构简单、可控制复杂度；局限是依赖特征表达，无法自动覆盖任意复杂关系。</p>
<h3>随机森林：多棵受限决策树的平均</h3>
<p>每棵树根据特征逐步划分样本，叶节点给出需求估计，最终取 200 棵树的平均。它使用与 Ridge 相同的信息，可表达非线性关系和交互。限制树深与叶节点样本数以控制复杂度。当前学习样本很少，仍有过拟合风险，对历史之外的突然变化也缺乏可靠依据。</p>
<p>四种方法都不能仅凭原理证明有效。有效性由提前整周预测的验证、最终测试以及简单对照共同检验。</p>
"""


def write_markdown(experiment: Experiment, totals: pd.DataFrame, metadata: dict, path: Path) -> None:
    n = narrative(experiment)
    selected = n["selected"]
    rows = []
    for _, row in experiment.metrics.iterrows():
        rows.append(f"| {PHASE_NAMES[row.phase]} | {MODEL_NAMES[row.model]} | {row.mae:.2f} | {row.rmse:.2f} | {format_value(row.r2, '.3f')} | {format_value(row.peak_mae, '.2f')} |")
    params = json.dumps(experiment.selection["selected_parameters"], ensure_ascii=False, indent=2)
    text = f"""# 6.1 全城小时需求预测

**负责人：陈宇涵**

### 6.1.1 本步骤回答什么

在每周一 00:00，用此前已知的订单信息，一次预测接下来周一至周日的 168 个小时。周内不使用新发生的真实需求更新输入。本节为全城预测；核心站点预测与模型比较见配套章节。不生成跨月预测示例。

### 6.1.2 数据与样本

输入为本项目 `data/processed/clean_trips.csv`：{metadata['input_rows']:,} 条有效骑行、21 个字段。每行是一次骑行，以 `started_at` 所在小时计数；保留站点信息不完整但时间有效的记录。先检查订单唯一、月份与时间派生字段一致、31 天均有记录，再补齐 744 个小时；确认完整日期内无订单的小时计为 0。小时计数之和必须等于输入订单数。

23 万条订单只形成 744 个时间样本，不能当作 23 万个独立预测样本。1 月 29–31 日只保留在输入覆盖记录中，不参与选型或评估。月份和范围是数据自身的历史日期，不是对当前日期的实时预测。

| 阶段 | 可用历史 | 一次预测的目标周 | 学习模型可拟合行数 |
|---|---|---|---:|
| 验证 | 1 月 1–14 日，336 小时 | 1 月 15–21 日，168 小时 | 168 |
| 最终测试 | 1 月 1–21 日，504 小时 | 1 月 22–28 日，168 小时 | 336 |

学习模型需要上周同期值，所以删除最初 168 个缺少周滞后的训练行，不用未来数据填补。两种简单对照可使用截止前全部历史。各方法使用相同预测起点和目标时段，但有效拟合行数不同，CSV 会记录差异。

### 6.1.3 特征与预测边界

- 小时：24 小时、12 小时的正余弦周期项，允许表达日内多峰变化。
- 星期：周二至周日的类别指示，周一为参照；另含周末标记。
- 周末与四个小时周期项的交互，允许周末具有不同的小时曲线。
- `lag_168h`：上周同一小时的实际需求。

共 16 个特征；Ridge 与随机森林使用相同特征。周末标记与星期类别存在确定关系，Ridge 通过系数约束处理冗余；系数只用于说明模型，不解释为独立的因果效应。预测周最后一个小时的滞后来源，最多到预测起点前一小时。整周输入可以在周初构造，不用短滞后递推，也不读取目标周实际需求。导出的逐小时预测保留 `forecast_origin`、`lag_source_time` 与 `lag_168h`，便于审查。

### 6.1.4 四种预测与拟合方法

### 1. 历史同期均值

对训练期相同“星期 × 小时”的次数求算术平均。例如周二 8 时的预测，等于历史周二 8 时次数的均值；验证阶段平均 2 个值，测试阶段平均 3 个值。无需迭代拟合，优点是透明、便于复核；局限是平滑变化较大时可能滞后。

### 2. 上周同期

`预测值(t) = 实际值(t − 168 小时)`。无需参数拟合，能直接检验重复最近一周的表现；局限是容易延续上周的异常波动。

### 3. Ridge 岭回归

用 `截距 + 各特征与其系数乘积之和` 拟合需求，优化“平方误差之和 + α × 系数平方和”。后一项约束系数过大。所有特征的标准化只在训练行上拟合，预测周沿用这些训练均值和标准差。比较 α = 0.1、1、10、100，按验证 MAE 选择。优点是可解释、复杂度可控制；局限是预测能力取决于选定特征，且少量周样本无法充分验证稳定性。

### 4. 随机森林

对训练样本进行自助抽样，建立多棵决策树；每棵树通过特征阈值逐步划分样本，叶节点估计需求，再取 200 棵树的平均。比较最大深度 3/5、叶节点最少样本数 8/16 的四种组合；固定随机种子 42。优点是可学习非线性关系；局限是少量周样本仍可能过拟合，对突发变化和范围外需求缺乏可靠外推能力。

预测骑行次数若为负，统一截为 0；评估保留小数，不提前取整。原始预测值和截断次数均保留，便于检查。

### 6.1.5 选型与评估

先在验证周为每种方法选择参数，再以验证 MAE 最低的方法作为推荐；若并列，先比较 RMSE，再按预定方法/参数顺序取值。参数及推荐方法在最终测试前冻结。测试只评估，不据此重新调参、改特征或改推荐方法。

- MAE：平均每小时差多少次，作为主要指标，越小越好。
- RMSE：对大误差更敏感，作为辅助指标，越小越好。
- R²：相对目标周均值的误差参照；可能为负，不将其解释为准确率。目标完全不变时记为无定义。
- 高峰 MAE：分别按训练期工作日/周末日均次数选出最高两个小时，评价目标周这些小时的误差；不根据目标周挑高峰。
- 日、周总量偏差：预测总量减真实总量；同时给出绝对百分比误差。总量接近不能证明每小时都准确。

| 阶段 | 方法 | MAE | RMSE | R² | 高峰 MAE |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

验证选定：**{n['name']}**。其最终测试 MAE 为 **{selected.mae:.2f} 次/小时**，RMSE 为 **{selected.rmse:.2f}**。相对历史同期均值的 MAE 改善为 **{format_value(n['improvements']['historical_mean'], '.1%')}**，相对上周同期为 **{format_value(n['improvements']['last_week'], '.1%')}**；负值表示更差，基线 MAE 为 0 时相对改善无定义。

**{n['verdict']}** 不用最终测试的排名回头替换推荐方案。

误差解读：验证周实际总量为 {n['validation'].loc[n['recommended'], 'total_actual']:,.0f} 次，选定方法预测 {n['validation'].loc[n['recommended'], 'total_predicted']:,.0f} 次；测试周实际 {selected.total_actual:,.0f} 次，预测 {selected.total_predicted:,.0f} 次，总量偏差为 **{format_value(n['week_bias_percentage'], '+.1%')}**。这说明日内曲线与整周规模需要分开判断，当前方法未充分捕捉周与周之间的规模变化，不能仅因对照排名较好就称为高精度预测。

选定参数：

```json
{params}
```

### 6.1.6 图表、有效性与限制

![图7：验证周与最终测试周的实际值和预测值](../figures/fig07_city_weekly_prediction.png)

图 7 只呈现最终测试周，保留真实值、历史同期均值、Ridge、随机森林四条曲线。用折线观察日内形状、峰值和偏差；指标表用于精确比较，不能仅凭曲线接近认定有效。上周同期仍保留在完整评估中，不在图表和 HTML 结果表中展示。

![补充图：验证周的实际值和三种预测](../figures/fig07_city_validation_prediction.png)

验证周单独绘图，在 HTML 中可展开查看。HTML 先介绍 Ridge、随机森林及 MAE、RMSE、R²，再展示结果；数据复核和复现说明保留在工程文件中。

![图8：随机森林特征重要性](../figures/fig08_city_feature_importance.png)

图 8 将随机森林的内部重要性按信息类型合并，解释这一个模型如何使用特征。相关特征可能分摊权重；它不是因果分析，也不是预测有效性的证明，即使随机森林未被选中仍明确标注模型名称。

有效性证据限于一次验证周和一次最终测试周。周总量有明显变化，当前没有天气、节假日、活动或库存等解释变量，不能确定变化原因，也不能保证未来周稳定表现。模型预测的是已观测有效订单量，不是全部潜在需求或未满足需求。

### 6.1.7 复现与产物

在项目目录运行：

```powershell
python -m src.demand_model
python -B -m unittest discover -s tests -p test_demand_model.py
```

默认独立生成 `reports/demand_prediction.html`，不调用前两项入口。对应表格位于 `reports/tables/city_*.csv`，参数和来源位于 `city_model_selection.json`。CSV 保留 10 位小数，HTML 为阅读而适当四舍五入。统计核验表与单元测试共同检查时间边界、计数和信息泄漏；运行命令会更新本步骤同名产物。

输入 SHA-256：`{metadata['input_sha256']}`。代码 SHA-256：`{metadata['code_sha256']}`。环境：Python {metadata['versions']['python']}、NumPy {metadata['versions']['numpy']}、pandas {metadata['versions']['pandas']}、scikit-learn {metadata['versions']['scikit_learn']}。

方法参考：[简单预测与季节朴素法](https://otexts.com/fpp3/toolbox.html)、[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)、[随机森林](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)、[周期时间特征](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)。参考方法说明不构成本数据上的有效性结论。
"""
    path.write_text(text, encoding="utf-8")


def write_html(experiment: Experiment, totals: pd.DataFrame, metadata: dict,
               figures: list[Path], downloads: list[Path], path: Path) -> None:
    """按学习顺序呈现：模型 → 指标 → 结果 → 局限；工程记录不进入页面。"""
    n = narrative(experiment)
    selected = n["selected"]
    daily = totals.loc[totals.phase.eq("test") & totals.model.eq(n["recommended"])].copy()
    daily["period"] = daily.period.map({"day": "日", "week": "周"})
    daily["date"] = daily.date.dt.strftime("%m/%d")
    daily["absolute_percentage_error"] *= 100
    daily = daily[["period", "date", "actual", "prediction", "bias", "absolute_percentage_error"]].rename(columns={
        "period": "单位", "date": "起始日", "actual": "真实次数", "prediction": "预测次数", "bias": "总量偏差", "absolute_percentage_error": "绝对误差 %"})

    def figure_tag(figure_path: Path, label: str) -> str:
        return (f'<button class="figure" type="button" aria-label="放大{escape(label)}">'
                f'<img src="{data_url(figure_path, "image/png")}" alt="{escape(label)}"></button>')

    curve = figure_tag(figures[0], "最终测试周预测对比")
    importance = figure_tag(figures[1], "随机森林特征重要性")
    validation = figure_tag(figures[2] if len(figures) > 2 else figures[0], "验证周预测对比")
    css = """
    :root{font-family:"Microsoft YaHei","Segoe UI",sans-serif;color:#23394a;background:#f3f6f7;font-size:15px;line-height:1.8}*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:34px 24px 50px}header{padding:28px 32px;background:#183f50;color:white;border-radius:16px}h1{font-size:30px;line-height:1.4;margin:0 0 10px}header p{margin:0;color:#d7e7ec}nav{display:flex;gap:22px;flex-wrap:wrap;margin:20px 0}a{color:#13766f;text-decoration:none}nav a{font-weight:600}section{background:white;border:1px solid #e0e8eb;border-radius:14px;padding:24px;margin-top:20px}h2{font-size:21px;margin:0 0 14px}h3{font-size:17px;margin:0 0 12px}p{margin:10px 0}.step{color:#16867c;font-size:12px;letter-spacing:.08em;display:block;margin-bottom:5px}.model-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.model{padding:20px;background:#f5f9f9;border-radius:10px;border-top:3px solid #2a9187}.model p{font-size:14px}.model strong{color:#215c66}.baseline{border-left:3px solid #d18b42;padding:8px 14px;background:#fff9f0;font-size:14px;margin-top:18px}.note{color:#677981;font-size:13px}.table-scroll{max-width:100%;overflow-x:auto}.data-table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}.data-table th{background:#eef4f5;color:#365967}.data-table td,.data-table th{padding:11px 14px;border-bottom:1px solid #e5ecee;text-align:right}.data-table th:first-child,.data-table td:first-child{text-align:left}.metric-guide{white-space:normal;min-width:540px}.metric-guide td,.metric-guide th{text-align:left;vertical-align:top}.metric-guide td:first-child{width:190px;font-weight:600;color:#245d64}.metric-guide th:last-child{width:200px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:16px 0}.card{border:1px solid #dce8e8;border-radius:10px;padding:16px}.card small{display:block;color:#60727b}.card strong{display:block;font-size:22px;color:#156d69;line-height:1.55}.card span{font-size:12px;color:#61727c}.verdict{padding:12px 16px;border-left:4px solid #288f86;background:#edf6f4}.figure{display:block;width:100%;border:0;padding:0;background:white;cursor:zoom-in}.figure img{width:100%;height:auto;display:block}details{border-top:1px solid #e3eaed;margin-top:18px;padding-top:14px}summary{cursor:pointer;color:#176f6a;font-weight:600}.example{padding:12px 16px;background:#f5f8f9;border-radius:8px;font-size:14px}.sources{font-size:12px;color:#6c7e85;margin-top:20px}footer{font-size:12px;color:#687c84;margin-top:20px}dialog{max-width:98vw;border:0;border-radius:12px;padding:16px}dialog::backdrop{background:#112632d9}dialog img{display:block;max-width:94vw;max-height:85vh;object-fit:contain}dialog button{display:block;margin:0 0 10px auto;border:0;border-radius:6px;padding:6px 16px;cursor:pointer;background:#e8f2f0;color:#205e5b}@media(max-width:680px){main{padding:18px 12px 32px}header{padding:24px 20px}h1{font-size:25px}section{padding:20px 15px}.model-grid,.cards{grid-template-columns:1fr}.model{padding:17px}.card strong{font-size:22px}nav{gap:16px}.data-table td,.data-table th{padding:9px 11px}.metric-guide{min-width:0;font-size:13px}.metric-guide th:last-child{width:auto}.metric-guide td:first-child{width:105px}.metric-guide td,.metric-guide th{padding:9px 7px}}
    """
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>全城周需求预测</title><style>{css}</style></head><body><main>
    <header><h1>未来一周的小时需求预测</h1><p>基于 2024 年 1 月 {metadata['input_rows']:,} 条有效骑行，研究每周初一次预测未来 168 小时的表现。</p></header>
    <nav aria-label="页内导航"><a href="#methods">认识模型</a><a href="#metrics">理解指标</a><a href="#results">预测结果</a><a href="#conclusion">结论与局限</a></nav>

    <section id="methods"><span class="step">01 · 认识模型</span><h2>Ridge 和随机森林分别是什么？</h2>
    <div class="model-grid"><article class="model"><h3>Ridge 岭回归</h3>
    <p><strong>是什么：</strong>一种带系数约束的线性回归。它把小时、星期、历史需求等信息的影响相加，得到骑行量预测。</p>
    <p><strong>怎样拟合：</strong>让预测尽量接近历史真实值，同时限制系数过大，控制对少量样本的过度拟合。</p>
    <p><strong>为什么用：</strong>结构简单、便于解释，适合检验这些时间因素是否具有可预测的关系。</p>
    <p><strong>局限：</strong>复杂变化需要合适的特征才能表达，不能保证跟上突然的需求变化。</p></article>
    <article class="model"><h3>随机森林</h3>
    <p><strong>是什么：</strong>由多棵决策树组成的模型。每棵树通过“是否周末”“历史需求是否较高”等条件逐步划分样本。</p>
    <p><strong>怎样拟合：</strong>各棵树学习历史样本并分别预测，再对它们的结果取平均；本项目使用 200 棵树。</p>
    <p><strong>为什么用：</strong>可以学习非线性关系，以及日期、小时和历史需求之间的组合影响。</p>
    <p><strong>局限：</strong>样本较少时仍可能过拟合，需要控制树的复杂度；对未见过的突发变化也可能预测不准。</p></article></div>
    <p class="baseline"><strong>对照方法：历史同期均值。</strong>例如预测周二 8 时，就平均此前各周二 8 时的实际次数，用它检查两个学习模型是否带来改善。</p>
    <details><summary>查看模型输入与比较安排</summary><p>输入包括小时周期、星期、工作日／周末及其小时差异，以及预测开始前已经知道的历史需求。每周一生成未来七天的预测，周内不更新真实需求。</p><p>先用 1 月 1–14 日预测 15–21 日，选择参数和方法；再用 1–21 日重新拟合，预测 22–28 日作为最终测试。最终测试用于检验效果。</p></details></section>

    <section id="metrics"><span class="step">02 · 理解指标</span><h2>MAE、RMSE、R² 怎么看？</h2>
    <div class="table-scroll"><table class="data-table metric-guide"><thead><tr><th>指标</th><th>它衡量什么？</th><th>怎样判断？</th></tr></thead><tbody>
    <tr><td>MAE<br>平均绝对误差</td><td>把每小时预测与真实次数的差取绝对值，再求平均。回答“平均每小时差多少次”。</td><td>越小越好。MAE＝20 表示平均每小时差 20 次。</td></tr>
    <tr><td>RMSE<br>均方根误差</td><td>将误差平方后求平均，再开平方。大误差会受到更重的惩罚，单位仍是骑行次数。</td><td>越小越好，用来补充检查是否出现较大的预测偏差。</td></tr>
    <tr><td>R²<br>决定系数</td><td>比较模型与“始终预测目标周实际均值”的平方误差。这个均值是评价参照。</td><td>1 表示完全吻合；0 表示与均值参照相当；负数表示更差。它不是准确率。</td></tr>
    </tbody></table></div>
    <p class="note">本项目以 MAE 为主要选型依据，RMSE 和 R² 为辅助。R² 也可能无定义，例如真实需求完全不变时。</p>
    <details><summary>用两个小时的小例子理解</summary><div class="example"><p>实际次数为 100、200，预测为 110、170，两个误差的绝对值分别为 10、30。</p><p><strong>MAE＝(10＋30)÷2＝20 次。</strong><br><strong>RMSE＝√[(10²＋30²)÷2]≈22.36 次。</strong></p><p>实际均值为 150；模型的平方误差之和为 1,000，用 150 预测的平方误差之和为 5,000，因此 <strong>R²＝1－1,000÷5,000＝0.80</strong>。这不表示有 80% 的小时预测正确。</p></div></details></section>

    <section id="results"><span class="step">03 · 预测结果</span><h2>最终测试周：哪种方法表现更好？</h2>
    <div class="cards"><div class="card"><small>验证周选定的方法</small><strong>{n['name']}</strong><span>先选定，再检验最终测试周</span></div>
    <div class="card"><small>该方法最终测试 MAE</small><strong>{selected.mae:,.2f} 次/小时</strong><span>平均每小时的绝对误差</span></div>
    <div class="card"><small>该方法测试周总量偏差</small><strong>{selected.total_bias:+,.0f} 次</strong><span>相对真实总量 {format_value(n['week_bias_percentage'], '+.1%')}</span></div></div>
    {curve}<p class="note">图 7：1 月 22–28 日最终测试周。保留真实骑行量和三种预测曲线。预测在周初生成，点击图表可放大。</p>
    <div class="table-scroll">{display_table(metric_display(experiment, 'test'))}</div>
    <p class="note">MAE、RMSE 越低越好，R² 越接近 1 越好。推荐方法依据验证周确定。</p>
    <details id="validation-results"><summary>查看验证周的曲线与表现</summary>{validation}<p class="note">补充图：1 月 15–21 日验证周，用于选择参数及方法。</p><div class="table-scroll">{display_table(metric_display(experiment, 'validation'))}</div></details>
    <details><summary>查看选定方法的每日与整周总量</summary><div class="table-scroll">{display_table(daily)}</div><p class="note">总量偏差＝预测总量－真实总量。总量接近时，每小时仍可能存在较大误差，因此两种尺度分开查看。</p></details></section>

    <section id="conclusion"><span class="step">04 · 结论与局限</span><h2>怎样理解这次预测结果？</h2><p class="verdict">{n['verdict']}</p>
    <p>最终测试周实际 {selected.total_actual:,.0f} 次，选定方法预测 {selected.total_predicted:,.0f} 次，总量偏差为 {format_value(n['week_bias_percentage'], '+.1%')}。当前方法没有充分捕捉周与周之间的规模变化，对照表现较好也不等于高精度预测。</p>
    <p>数据只有四个完整自然周，且周需求波动明显。当前只有一次验证周、一次最终测试周，也没有天气、活动、库存等信息，不能保证长期稳定表现。预测对象是实际观测的有效骑行次数。</p>
    <details id="feature-explanation"><summary>补充：随机森林主要使用了哪些信息？</summary>{importance}<p class="note">图 8 仅解释随机森林的内部特征重要性。相关特征可能分摊权重，重要性不能证明因果或预测有效。</p></details>
    <p class="sources">方法定义参考：<a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html">Ridge</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">随机森林</a> · <a href="https://scikit-learn.org/stable/modules/model_evaluation.html">评价指标</a></p></section>
    <footer>图片和表格已内嵌，可单独复制此页面离线阅读。</footer><dialog id="zoom"><button type="button">关闭</button><img alt="图表放大"></dialog>
    """
    page += """<script>const dialog=document.getElementById('zoom');document.querySelectorAll('.figure').forEach(button=>button.addEventListener('click',()=>{dialog.querySelector('img').src=button.querySelector('img').src;dialog.showModal();}));dialog.querySelector('button').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});</script></main></body></html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "clean_trips.csv")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    print("读取有效订单并核对小时序列…", flush=True)
    data = read_clean_data(args.input)
    y = build_hourly(data)
    print("验证选型，再评估最终测试周…", flush=True)
    experiment = run_experiment(y)
    metadata = {"input_rows": len(data), "input_columns": len(data.columns),
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                             "scikit_learn": sklearn.__version__}}
    experiment.selection["provenance"] = metadata
    experiment.selection["feature_names"] = experiment.fits["ridge"].feature_names
    rows = [{"phase":"input", "check": label, "observed": str(observed), "expected": str(expected), "passed": observed == expected}
            for label, observed, expected in [("hour_count", len(y), 744), ("observed_dates", data.started_at.dt.normalize().nunique(), 31),
                                              ("count_conservation", int(y.sum()), len(data)), ("unique_orders", data.ride_id.nunique(), len(data))]]
    experiment.audit = pd.concat([pd.DataFrame(rows), experiment.audit], ignore_index=True)
    if not experiment.audit.passed.all():
        raise ValueError("输入核验失败。")
    print("生成图表、明细和离线报告…", flush=True)
    font = set_plot_style()
    out = args.output_dir
    for folder in (out / "tables", out / "figures", out / "sections"):
        folder.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"
    totals = aggregate_totals(experiment.predictions)
    exports = {
        "city_hourly_demand.csv": y.reset_index(),
        "city_weekly_predictions.csv": experiment.predictions,
        "city_model_metrics.csv": experiment.metrics,
        "city_parameter_candidates.csv": experiment.candidates,
        "city_daily_weekly_totals.csv": totals,
        "city_feature_importance.csv": feature_importances(experiment),
        "city_prediction_validation.csv": experiment.audit,
    }
    for name, frame in exports.items():
        frame.to_csv(tables / name, index=False, encoding="utf-8-sig", float_format="%.10f", date_format="%Y-%m-%d %H:%M:%S")
    selection_path = tables / "city_model_selection.json"
    selection_path.write_text(json.dumps(json_safe(experiment.selection), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figures = plot_results(experiment, out / "figures")
    markdown = out / "sections" / "03_city_demand_prediction.md"
    write_markdown(experiment, totals, metadata, markdown)
    write_html(experiment, totals, metadata, figures,
               [tables / name for name in exports] + [selection_path, markdown], out / "demand_prediction.html")
    print(json.dumps(json_safe({"orders": len(data), "hours": len(y), "checks_passed": len(experiment.audit),
                      "font": font, "recommended_frozen": experiment.selection["recommended_model"],
                      "metrics": experiment.metrics.to_dict(orient="records"), "report": str(out / "demand_prediction.html")}),
                     ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
