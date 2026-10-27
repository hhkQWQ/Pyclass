"""成员 B / 第二步：冻结核心站点，共享建模预测下一自然周的借出、归还与净流出。

运行：python -m src.station_model。只生成 station_* 成果，不调用全城预测入口。
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_io import read_clean_data
from src.utils import PROCESSED_DIR, REPORT_DIR, set_plot_style, save_figure
from src.demand_model import (
    MONTH_START, MONTH_END, PHASES, PHASE_NAMES, MODEL_NAMES, MODEL_ORDER,
    DISPLAY_MODELS, PARAMETER_GRIDS, COLORS, build_hourly, features,
    peak_mask, scores, json_safe, format_value,
)

SELECTION_START = MONTH_START + pd.Timedelta(hours=3)
SELECTION_END = PHASES["validation"][0]
TARGETS = ("outflow", "inflow", "net_outflow")
TARGET_NAMES = {"outflow": "借出", "inflow": "归还", "net_outflow": "净流出"}
DIRECTIONS = TARGETS[:2]
_HTML_IMAGES: dict[str, str] = {}


def remember_html_image(figure, file: Path) -> None:
    """HTML直接从同一图对象渲染轻量无损图；300DPI PNG另存，数值和曲线不变。"""
    buffer = BytesIO()
    figure.savefig(buffer, format="webp", dpi=120, bbox_inches="tight", pad_inches=.25,
                   facecolor="white", pil_kwargs={"lossless": True, "method": 4})
    _HTML_IMAGES[str(file.resolve())] = "data:image/webp;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")


def extract_events(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """每个已知端点各产生一个事件；完整性按端点判断，不整行丢弃。"""
    build_hourly(data)  # 复用有效集、时间派生与完整月份检查，不产生旧任务输出。
    duration = (data.ended_at - data.started_at).dt.total_seconds() / 60
    if duration.isna().any() or not duration.between(1, 180).all():
        raise ValueError("有效集骑行时长不在已约定的 1–180 分钟内。")
    if not np.allclose(duration.to_numpy(), data.duration_min.to_numpy(dtype=float), atol=1e-6):
        raise ValueError("骑行时长与起止时间不一致。")
    frames, known = [], {}
    for direction, endpoint, time_column in (
        ("outflow", "start", "started_at"), ("inflow", "end", "ended_at")
    ):
        ids = data[f"{endpoint}_station_id"]
        valid = ids.notna() & ids.fillna("").str.strip().ne("")
        known[direction] = valid
        frame = pd.DataFrame({
            "ride_id": data.loc[valid, "ride_id"].astype(str),
            "station_id": ids.loc[valid].str.strip().astype(str),
            "station_name": data.loc[valid, f"{endpoint}_station_name"].fillna("").astype(str),
            "event_at": data.loc[valid, time_column],
        })
        frame["timestamp"] = frame.event_at.dt.floor("h")
        frame["direction"] = direction
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True)
    if events.duplicated(["ride_id", "direction"]).any():
        raise ValueError("同一订单在同一方向出现重复事件。")
    in_month = events.timestamp.between(MONTH_START, MONTH_END, inclusive="left")
    info = {"input_rows": len(data), "both_endpoints_known": int((known["outflow"] & known["inflow"]).sum()),
            "only_start_known": int((known["outflow"] & ~known["inflow"]).sum()),
            "only_end_known": int((~known["outflow"] & known["inflow"]).sum()),
            "neither_endpoint_known": int((~known["outflow"] & ~known["inflow"]).sum()),
            "max_duration_min": float(duration.max()),
            "outflow_events_in_january": int((in_month & events.direction.eq("outflow")).sum()),
            "inflow_events_in_january": int((in_month & events.direction.eq("inflow")).sum()),
            "inflow_events_after_january": int((events.timestamp.ge(MONTH_END) & events.direction.eq("inflow")).sum()),
            "observed_inflow_first_3h": int((events.timestamp.lt(SELECTION_START) & events.direction.eq("inflow")).sum())}
    return events, info


def select_stations(events: pd.DataFrame, top_n: int = 10,
                    min_daily_activity: float = 20, min_active_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只在固定的第一、二周筛选。稀疏事件按完整、共同可观测的小时分母计算。"""
    window = events.loc[events.timestamp.ge(SELECTION_START) & events.timestamp.lt(SELECTION_END)].copy()
    if window.empty:
        raise ValueError("站点筛选期没有可用事件。")
    window["is_weekend"] = window.timestamp.dt.dayofweek.ge(5)
    window["hour"] = window.timestamp.dt.hour
    window["date"] = window.timestamp.dt.normalize()
    ids = sorted(window.station_id.unique())
    grid = pd.MultiIndex.from_product([ids, [False, True], range(24)], names=["station_id", "is_weekend", "hour"])
    counts = window.groupby(["station_id", "is_weekend", "hour", "direction"]).size().unstack("direction", fill_value=0)
    counts = counts.reindex(index=grid, columns=DIRECTIONS, fill_value=0)
    calendar = pd.date_range(SELECTION_START, SELECTION_END, inclusive="left", freq="h")
    denominators = pd.DataFrame({"is_weekend": calendar.dayofweek >= 5, "hour": calendar.hour}).value_counts()
    div = np.array([denominators.loc[(weekend, hour)] for _, weekend, hour in grid])
    counts["mean_net"] = (counts.outflow - counts.inflow).to_numpy() / div
    imbalance = counts.mean_net.abs().groupby(level=["station_id", "is_weekend"]).sum().unstack("is_weekend", fill_value=0)
    activity = window.groupby("station_id").agg(activity_total=("direction", "size"), active_days=("date", "nunique"))
    activity["daily_activity"] = activity.activity_total / 14.0
    activity["workday_imbalance"] = imbalance[False]
    activity["weekend_imbalance"] = imbalance[True]
    activity["tidal_score"] = activity.workday_imbalance * (10 / 14) + activity.weekend_imbalance * (4 / 14)
    direction_totals = window.groupby(["station_id", "direction"]).size().unstack("direction", fill_value=0)
    for direction in DIRECTIONS:
        activity[f"selection_{direction}"] = direction_totals.reindex(columns=DIRECTIONS, fill_value=0)[direction]
    names = window.loc[window.station_name.str.strip().ne("")].groupby(["station_id", "station_name"]).size().rename("name_events").reset_index()
    names = names.sort_values(["station_id", "name_events", "station_name"], ascending=[True, False, True], kind="stable").drop_duplicates("station_id")
    name_map = names.set_index("station_id").station_name.to_dict()
    candidates = activity.reset_index()
    candidates["station_name"] = candidates.station_id.map(name_map).fillna("")
    candidates["eligible"] = candidates.daily_activity.ge(min_daily_activity) & candidates.active_days.ge(min_active_days)
    eligible = candidates.loc[candidates.eligible].sort_values(
        ["tidal_score", "activity_total", "station_id"], ascending=[False, False, True], kind="stable").copy()
    if len(eligible) < top_n:
        raise ValueError(f"只有 {len(eligible)} 个站点满足活动门槛，不足 {top_n} 个；需更新方案，不能自动降低门槛。")
    eligible["selection_rank"] = np.arange(1, len(eligible) + 1)
    ranks = eligible.set_index("station_id").selection_rank
    candidates["selection_rank"] = candidates.station_id.map(ranks).astype("Int64")
    candidates["selected"] = candidates.selection_rank.le(top_n).fillna(False)
    selected = candidates.loc[candidates.selected].sort_values("selection_rank").reset_index(drop=True)
    candidates = candidates.sort_values(["eligible", "tidal_score", "activity_total", "station_id"], ascending=[False, False, False, True], kind="stable")
    return candidates.reset_index(drop=True), selected


def build_panel(events: pd.DataFrame, station_ids: list[str]) -> pd.DataFrame:
    calendar = pd.date_range(MONTH_START, MONTH_END, inclusive="left", freq="h")
    index = pd.MultiIndex.from_product([station_ids, calendar], names=["station_id", "timestamp"])
    month = events.loc[events.station_id.isin(station_ids) & events.timestamp.ge(MONTH_START) & events.timestamp.lt(MONTH_END)]
    counts = month.groupby(["station_id", "timestamp", "direction"]).size().unstack("direction", fill_value=0)
    panel = counts.reindex(index=index, columns=DIRECTIONS, fill_value=0).astype(float)
    panel.loc[panel.index.get_level_values("timestamp") < SELECTION_START, "inflow"] = np.nan
    panel["net_outflow"] = panel.outflow - panel.inflow
    panel["inflow_complete"] = panel.index.get_level_values("timestamp") >= SELECTION_START
    return panel


def before(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    return panel.loc[panel.index.get_level_values("timestamp") < origin].copy()


def design(history: pd.DataFrame, station_ids: list[str], direction: str,
           forecast_origin: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
    """站点内构造周滞后，再合并；站点编号用训练期冻结的类别编码。"""
    all_x, all_y = [], []
    for station_id in station_ids:
        series = history.xs(station_id, level="station_id")[direction]
        if forecast_origin is None:
            index = pd.DatetimeIndex(series.index)
            x = features(series, index)
            valid = series.notna() & x.notna().all(axis=1)
            x, y = x.loc[valid].copy(), series.loc[valid].copy()
        else:
            index = pd.date_range(forecast_origin, periods=168, freq="h")
            x = features(series, index, forecast_origin)
            if x.isna().any().any():
                raise ValueError("整周预测缺少上周同期数据。")
            y = None
        for code, category in enumerate(station_ids[1:], start=1):
            x[f"station_{code}"] = float(station_id == category)
        x.index = pd.MultiIndex.from_arrays([[station_id] * len(x), x.index], names=["station_id", "timestamp"])
        all_x.append(x)
        if y is not None:
            y.index = x.index
            all_y.append(y)
    return pd.concat(all_x), pd.concat(all_y) if all_y else None


@dataclass
class SharedFit:
    model: str
    direction: str
    station_ids: list[str]
    history: pd.DataFrame
    estimator: object | None
    fit_rows: int
    feature_names: list[str]

    def predict(self, origin: pd.Timestamp) -> pd.DataFrame:
        x, _ = design(self.history, self.station_ids, self.direction, origin)
        if self.model == "historical_mean":
            output = []
            for station_id in self.station_ids:
                h = self.history.xs(station_id, level="station_id")[self.direction].dropna()
                means = h.groupby([h.index.dayofweek, h.index.hour]).mean()
                index = x.xs(station_id, level="station_id").index
                output.extend(float(means.loc[(d, hour)]) for d, hour in zip(index.dayofweek, index.hour))
            raw = np.asarray(output)
        elif self.model == "last_week":
            raw = x.lag_168h.to_numpy()
        else:
            raw = np.asarray(self.estimator.predict(x), dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("共享模型输出非有限数值。")
        return pd.DataFrame({"raw_prediction": raw, "prediction": np.maximum(raw, 0), "lag_168h": x.lag_168h}, index=x.index)


def fit_shared(model: str, direction: str, history: pd.DataFrame,
               station_ids: list[str], parameters: dict) -> SharedFit:
    if direction not in DIRECTIONS or model not in MODEL_ORDER:
        raise ValueError("未知方向或模型。")
    for station_id in station_ids:
        h = history.xs(station_id, level="station_id")
        if not h.index.is_unique or not h.index.equals(pd.date_range(MONTH_START, h.index.max(), freq="h")):
            raise ValueError("站点历史小时必须唯一、连续并从月初开始。")
        observed = h[direction].dropna()
        if len(h) < 168 or (observed < 0).any():
            raise ValueError("站点历史不足或包含负事件数。")
    if model in ("historical_mean", "last_week"):
        return SharedFit(model, direction, station_ids, history.copy(), None, int(history[direction].notna().sum()), [])
    x, y = design(history, station_ids, direction)
    if x.empty:
        raise ValueError("共享模型没有可拟合的周滞后样本。")
    estimator = make_pipeline(StandardScaler(), Ridge(**parameters)) if model == "ridge" else RandomForestRegressor(**parameters)
    estimator.fit(x, y)
    return SharedFit(model, direction, station_ids, history.copy(), estimator, len(x), list(x.columns))


def target_values(panel: pd.DataFrame, index: pd.MultiIndex, target: str) -> np.ndarray:
    values = panel[target].reindex(index).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("目标周没有完整的站点标签。")
    return values


def station_peaks(history: pd.DataFrame, station_ids: list[str], origin: pd.Timestamp, target: str) -> np.ndarray:
    index = pd.date_range(origin, periods=168, freq="h")
    masks = []
    for station_id in station_ids:
        h = history.xs(station_id, level="station_id")[target]
        masks.append(peak_mask(h.abs() if target == "net_outflow" else h, index))
    return np.concatenate(masks)


def mean_station_mae(actual: np.ndarray, prediction: np.ndarray, station_ids: list[str]) -> float:
    return float(np.mean([np.mean(np.abs(actual[i*168:(i+1)*168] - prediction[i*168:(i+1)*168])) for i in range(len(station_ids))]))


@dataclass
class StationExperiment:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    overall: pd.DataFrame
    candidates: pd.DataFrame
    method_scores: pd.DataFrame
    selection: dict
    fits: dict[tuple[str, str], SharedFit]
    audit: pd.DataFrame


def run_experiment(panel: pd.DataFrame, selected: pd.DataFrame, grids: dict | None = None,
                   progress=None) -> StationExperiment:
    grids = PARAMETER_GRIDS if grids is None else grids
    ids = selected.station_id.tolist()
    origin = PHASES["validation"][0]
    history = before(panel, origin)
    candidate_rows, parameters, val_outputs = [], {}, {}
    val_fit_objects = {}
    for direction in DIRECTIONS:
        parameters[direction] = {}
        for model in MODEL_ORDER:
            if progress:
                progress(f"验证：{TARGET_NAMES[direction]} / {MODEL_NAMES[model]}")
            ranked = []
            for candidate_id, params in enumerate(grids[model]):
                fitted = fit_shared(model, direction, history, ids, params)
                predicted = fitted.predict(origin)
                actual = target_values(panel, predicted.index, direction)
                primary = mean_station_mae(actual, predicted.prediction.to_numpy(), ids)
                rmse = float(np.sqrt(np.mean((actual - predicted.prediction.to_numpy()) ** 2)))
                candidate_rows.append({"direction": direction, "model": model, "candidate_id": candidate_id,
                                       "parameters": json.dumps(params, sort_keys=True),
                                       "mean_station_mae": primary, "rmse": rmse, "fit_rows": fitted.fit_rows})
                ranked.append((primary, rmse, candidate_id, fitted, predicted))
            best = min(ranked, key=lambda item: item[:3])
            parameters[direction][model] = dict(grids[model][best[2]])
            val_fit_objects[(model, direction)] = best[3]
            val_outputs[(model, direction)] = best[4]
    candidates = pd.DataFrame(candidate_rows)
    candidates["selected_within_family_direction"] = candidates.apply(
        lambda row: row.parameters == json.dumps(parameters[row.direction][row.model], sort_keys=True), axis=1)
    chosen = candidates.loc[candidates.selected_within_family_direction]
    method_scores = chosen.groupby("model", sort=False).agg(
        combined_mae=("mean_station_mae", "mean"), combined_rmse=("rmse", "mean")).reset_index()
    method_scores["eligible_for_recommendation"] = method_scores.model.isin(DISPLAY_MODELS)
    recommendation = min(DISPLAY_MODELS, key=lambda model: (
        float(method_scores.loc[method_scores.model.eq(model), "combined_mae"].iloc[0]),
        float(method_scores.loc[method_scores.model.eq(model), "combined_rmse"].iloc[0]), DISPLAY_MODELS.index(model)))
    # 统一推荐与各方向参数在最终测试标签用于评价之前冻结。
    selection = {"recommended_model": recommendation, "recommendation_candidates": list(DISPLAY_MODELS),
                 "supplementary_only": ["last_week"], "selected_parameters": parameters,
                 "parameter_grids": grids, "station_ids": ids,
                 "selection_period": {"start": str(SELECTION_START), "end_exclusive": str(SELECTION_END)},
                 "screening": {"min_daily_activity": 20, "days_denominator": 14, "min_active_days": 7,
                               "tidal_weights": {"workday": 10/14, "weekend": 4/14},
                               "tie_break": ["tidal_score_desc", "activity_total_desc", "station_id_lexical_asc"]},
                 "recommendation_score": "mean(mean_station_outflow_MAE, mean_station_inflow_MAE)",
                 "tie_break": "combined_validation_RMSE_then_predefined_model_order",
                 "shared_models": True, "directions_fitted_separately": True,
                 "within_week_actual_updates": False, "post_test_tuning": False,
                 "forecast_horizon_hours": 168}
    predictions, audits, final_fits = [], [], {}

    def check(phase, label, observed, expected):
        audits.append({"phase": phase, "check": label, "observed": str(observed),
                       "expected": str(expected), "passed": bool(observed == expected)})

    for phase, (forecast_origin, end) in PHASES.items():
        h = before(panel, forecast_origin)
        expected_history = int((forecast_origin - MONTH_START) / pd.Timedelta(hours=1))
        check(phase, "forecast_hours_per_station", int((end-forecast_origin)/pd.Timedelta(hours=1)), 168)
        check(phase, "latest_history_before_origin", bool(h.index.get_level_values("timestamp").max() < forecast_origin), True)
        check(phase, "station_count", len(ids), len(selected))
        for model in MODEL_ORDER:
            paired = {}
            for direction in DIRECTIONS:
                if phase == "validation":
                    fitted = val_fit_objects[(model, direction)]
                    predicted = val_outputs[(model, direction)]
                else:
                    if progress:
                        progress(f"最终测试：{TARGET_NAMES[direction]} / {MODEL_NAMES[model]}")
                    fitted = fit_shared(model, direction, h, ids, parameters[direction][model])
                    predicted = fitted.predict(forecast_origin)
                    final_fits[(model, direction)] = fitted
                paired[direction] = predicted
                check(phase, f"{model}/{direction}:finite_nonnegative", bool(np.isfinite(predicted.prediction).all() and predicted.prediction.ge(0).all()), True)
                if model in ("ridge", "random_forest"):
                    expected_fit = (expected_history - 168 - (3 if direction == "inflow" else 0)) * len(ids)
                    check(phase, f"{model}/{direction}:effective_fit_rows", fitted.fit_rows, expected_fit)
            left, right = paired["outflow"], paired["inflow"]
            if not left.index.equals(right.index):
                raise ValueError("借出和归还预测索引错位。")
            index = left.index
            timestamps = index.get_level_values("timestamp")
            check(phase, f"{model}:latest_lag_source_before_origin", bool((timestamps-pd.Timedelta(hours=168)).max() < forecast_origin), True)
            for target in TARGETS:
                if target == "net_outflow":
                    raw = left.raw_prediction.to_numpy() - right.raw_prediction.to_numpy()
                    pred = left.prediction.to_numpy() - right.prediction.to_numpy()
                else:
                    raw, pred = paired[target].raw_prediction.to_numpy(), paired[target].prediction.to_numpy()
                predictions.append(pd.DataFrame({
                    "phase": phase, "forecast_origin": forecast_origin,
                    "station_id": index.get_level_values("station_id"), "timestamp": timestamps,
                    "model": model, "target": target, "actual": target_values(panel, index, target),
                    "raw_prediction": raw, "prediction": pred,
                    "lag_source_time": timestamps-pd.Timedelta(hours=168),
                    "lag_out_168h": left.lag_168h.to_numpy(), "lag_in_168h": right.lag_168h.to_numpy(),
                    "is_peak": station_peaks(h, ids, forecast_origin, target),
                    "recommended_frozen": model == recommendation,
                }))
    predicted_all = pd.concat(predictions, ignore_index=True)
    metric_rows = []
    for (phase, model, target, station_id), frame in predicted_all.groupby(["phase", "model", "target", "station_id"], sort=False):
        metric_rows.append({"phase": phase, "model": model, "target": target, "station_id": station_id,
                            "recommended_frozen": model == recommendation,
                            **scores(frame.actual.to_numpy(), frame.prediction.to_numpy(), frame.is_peak.to_numpy())})
    metrics = pd.DataFrame(metric_rows)
    overall_rows = []
    for (phase, model, target), frame in predicted_all.groupby(["phase", "model", "target"], sort=False):
        per_station = metrics.loc[metrics.phase.eq(phase) & metrics.model.eq(model) & metrics.target.eq(target)]
        overall_rows.append({"phase": phase, "model": model, "target": target,
                             "mean_station_mae": float(per_station.mae.mean()),
                             "pooled_rmse": float(np.sqrt(np.mean((frame.actual-frame.prediction)**2))),
                             "mean_station_peak_mae": float(per_station.peak_mae.mean()),
                             "total_actual": float(frame.actual.sum()), "total_predicted": float(frame.prediction.sum()),
                             "total_bias": float(frame.prediction.sum()-frame.actual.sum())})
    audit = pd.DataFrame(audits)
    if not audit.passed.all():
        raise ValueError("站点预测边界检查失败。")
    return StationExperiment(predicted_all, metrics, pd.DataFrame(overall_rows), candidates,
                             method_scores, selection, final_fits, audit)


def daily_weekly_totals(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["date"] = frame.timestamp.dt.normalize()
    keys = ["phase", "model", "target", "station_id"]
    daily = frame.groupby(keys+["date"], sort=False)[["actual", "prediction"]].sum().reset_index()
    daily["period"] = "day"
    weekly = frame.groupby(keys, sort=False).agg(date=("timestamp", "min"), actual=("actual", "sum"), prediction=("prediction", "sum")).reset_index()
    weekly["period"] = "week"
    result = pd.concat([daily, weekly], ignore_index=True)
    result["bias"] = result.prediction-result.actual
    result["absolute_percentage_error"] = np.where(result.target.ne("net_outflow") & result.actual.gt(0), abs(result.bias)/result.actual, np.nan)
    return result


def importances(experiment: StationExperiment) -> pd.DataFrame:
    parts = []
    for direction in DIRECTIONS:
        fitted = experiment.fits[("random_forest", direction)]
        frame = pd.DataFrame({"direction": direction, "feature": fitted.feature_names,
                              "importance": fitted.estimator.feature_importances_})
        def group(feature):
            if feature.startswith("station_"): return "站点类别"
            if feature.startswith("weekend_x_"): return "周末与小时的交互"
            if feature.startswith("hour_"): return "小时周期"
            if feature.startswith("weekday_"): return "星期类别"
            if feature == "lag_168h": return "该站上周同小时需求"
            return "周末标记"
        frame["group_label"] = frame.feature.map(group)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def plot_results(experiment: StationExperiment, selected: pd.DataFrame, directory: Path, progress=None) -> tuple[dict, Path]:
    charts = {}
    recommended = experiment.selection["recommended_model"]
    for site in selected.itertuples():
        station_id = site.station_id
        charts[station_id] = {}
        for target in TARGETS:
            charts[station_id][target] = {}
            for phase in ("test", "validation"):
                frame = experiment.predictions.loc[experiment.predictions.station_id.eq(station_id) & experiment.predictions.target.eq(target) & experiment.predictions.phase.eq(phase)]
                actual = frame.loc[frame.model.eq("historical_mean")]
                fig, ax = plt.subplots(figsize=(11.5, 4.3), layout="constrained")
                ax.plot(actual.timestamp, actual.actual, color="#253342", linewidth=2, label=f"真实{TARGET_NAMES[target]}", zorder=6)
                for model in DISPLAY_MODELS:
                    part = frame.loc[frame.model.eq(model)]
                    ax.plot(part.timestamp, part.prediction, color=COLORS[model],
                            linewidth=1.9 if model == recommended else 1.25,
                            linestyle="-" if model == recommended else "--", alpha=.95 if model == recommended else .8,
                            label=MODEL_NAMES[model]+("（统一推荐）" if model == recommended else ""))
                if target == "net_outflow":
                    ax.axhline(0, color="#a9b8c0", linewidth=.8)
                else:
                    ax.set_ylim(bottom=0)
                ax.set(xlabel="日期", ylabel="净流出次数" if target == "net_outflow" else "骑行次数",
                       title=f"图 9  {TARGET_NAMES[target]} · {PHASE_NAMES[phase]}（{actual.timestamp.min():%m/%d}–{actual.timestamp.max():%m/%d}）")
                ax.xaxis.set_major_locator(mdates.DayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
                ax.set_xlim(actual.timestamp.min(), actual.timestamp.max())
                ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02),
                          borderaxespad=0, frameon=False, fontsize=8.2)
                ax.set_title(ax.get_title(), pad=26)
                ax.margins(y=.25)
                name = textwrap.shorten(site.station_name or station_id, width=60, placeholder="…")
                fig.suptitle(f"{int(site.selection_rank):02d} · {name}  [{station_id}]", fontsize=11)
                charts[station_id][target][phase] = save_figure(fig, f"station_{int(site.selection_rank):02d}_{target}_{phase}.png", directory=directory/"stations")
                remember_html_image(fig, charts[station_id][target][phase])
                plt.close(fig)
        if progress:
            progress(f"已生成站点 {int(site.selection_rank)}/{len(selected)} 的预测图")
    grouped = importances(experiment).groupby(["direction", "group_label"]).importance.sum()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), layout="constrained")
    for ax, direction in zip(axes, DIRECTIONS):
        values = grouped.loc[direction].sort_values()
        ax.barh(values.index, values.values, color="#328f89", height=.62)
        for i, value in enumerate(values.values): ax.text(value+.006, i, f"{value:.1%}", va="center", fontsize=9)
        ax.set(title=f"随机森林 · {TARGET_NAMES[direction]}", xlabel="内部重要性占比", xlim=(0, max(.15, float(values.max())*1.25)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis="y", visible=False)
    fig.suptitle("图 10  共享随机森林使用哪些信息", fontsize=14, fontweight="bold")
    importance_path = save_figure(fig, "fig10_station_feature_importance.png", directory=directory)
    remember_html_image(fig, importance_path)
    plt.close(fig)
    return charts, importance_path


def summary(experiment: StationExperiment) -> dict:
    model = experiment.selection["recommended_model"]
    grouped = experiment.overall.set_index(["phase", "model", "target"])
    def combined(phase, family):
        return float(np.mean([grouped.loc[(phase, family, direction), "mean_station_mae"] for direction in DIRECTIONS]))
    test_score, baseline, supplementary = combined("test", model), combined("test", "historical_mean"), combined("test", "last_week")
    if model == "historical_mean":
        verdict = "验证周统一推荐历史同期均值；两种学习模型没有在验证周获得更低的借还综合误差。"
    elif test_score < baseline and test_score < supplementary:
        verdict = "验证选定的学习模型在最终测试周也优于两种简单对照，获得本次站点周预测的初步改进证据。"
    elif test_score < baseline:
        verdict = "验证选定的学习模型在最终测试周优于历史均值，但未超过全部简单对照，改进证据仍有限。"
    else:
        verdict = "验证选定的学习模型未在最终测试周保持相对历史均值的优势，尚无稳定改进证据。"
    volume = {}
    for direction in DIRECTIONS:
        row = grouped.loc[("test", model, direction)]
        volume[direction] = {"actual": float(row.total_actual), "predicted": float(row.total_predicted),
                             "relative_bias": float(row.total_bias/row.total_actual) if row.total_actual > 0 else None}
    return {"model": model, "name": MODEL_NAMES[model], "test_combined_mae": test_score,
            "validation_combined_mae": combined("validation", model),
            "test_mean_net_mae": float(grouped.loc[("test", model, "net_outflow"), "mean_station_mae"]),
            "baseline_test_combined_mae": baseline, "supplementary_test_combined_mae": supplementary,
            "verdict": verdict, "test_volume": volume}


def png_data_url(path: Path) -> str:
    cached = _HTML_IMAGES.get(str(path.resolve()))
    return cached if cached is not None else "data:image/png;base64,"+base64.b64encode(path.read_bytes()).decode("ascii")


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value): return "无定义"
        if isinstance(value, (float, np.floating)): return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| "+" | ".join(map(str, frame.columns))+" |", "| "+" | ".join(["---"]*len(frame.columns))+" |"]
    lines += ["| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_markdown(experiment: StationExperiment, selected: pd.DataFrame, candidates: pd.DataFrame,
                   metadata: dict, charts: dict, path: Path) -> None:
    result = summary(experiment)
    top = selected[["selection_rank", "station_id", "station_name", "tidal_score", "daily_activity", "active_days"]].rename(columns={
        "selection_rank":"排名", "station_id":"站点ID", "station_name":"站点名称", "tidal_score":"潮汐分数", "daily_activity":"日均活动", "active_days":"活动天数"})
    aggregate = experiment.overall.loc[experiment.overall.phase.eq("test"), ["model", "target", "mean_station_mae", "pooled_rmse", "mean_station_peak_mae"]].copy()
    aggregate["model"] = aggregate.model.map(MODEL_NAMES);aggregate["target"] = aggregate.target.map(TARGET_NAMES)
    aggregate.columns = ["方法", "目标", "逐站平均MAE", "合并RMSE", "逐站平均高峰MAE"]
    params = json.dumps(experiment.selection["selected_parameters"], ensure_ascii=False, indent=2)
    first = selected.iloc[0]
    example = f"../figures/stations/{charts[first.station_id]['outflow']['test'].name}"
    text = f"""# 核心站点筛选与整周需求预测（第二步）

## 1. 研究对象与数据选择

输入为成员B自己的clean_trips.csv，共{metadata['input_rows']:,}条有效骑行。按目标分别保留已知端点：起点已知就产生借出事件，终点已知就产生归还事件；缺另一端不删除已经可归属的事件。

起终点都已知{metadata['both_endpoints_known']:,}条，仅起点已知{metadata['only_start_known']:,}条，仅终点已知{metadata['only_end_known']:,}条，两端都缺失{metadata['neither_endpoint_known']:,}条。1月可归属借出{metadata['outflow_events_in_january']:,}次、归还{metadata['inflow_events_in_january']:,}次，另有{metadata['inflow_events_after_january']}次归还发生在2月，不计入1月。

借出按start_station_id与started_at归属；归还按end_station_id与ended_at归属，不能使用属于出发时刻的hour/weekday代替归还时间。事件按自己的发生时间截断，已借出但尚未归还的订单只能进入此前借出统计。

有效骑行最长{metadata['max_duration_min']:.4f}分钟，小于180分钟。缺少12月出发订单可能使1月1日00–02时的归还记录不完整，即使已有{metadata['observed_inflow_first_3h']}次已观测归还，也不将其当作完整目标。这3小时归还和净流出记为缺失；其余已覆盖小时没有事件时计0。借出模型保留月初已知借出；站点筛选为统一借还口径，将最初3小时双方事件都排除。

这是固定清洗有效集中的历史事件预测，不能保证实时系统中数据已经全部到齐；未知端点未被补全。净流出是可归属观测流量的差，不等于库存变化、缺车数或未满足需求。

## 2. 仅训练期筛选Top10并冻结

筛选只用1月1日03:00至1月15日00:00前的事件。候选门槛为（借出＋归还）÷14个日历日≥20次/日，且至少7个活动日。最初3小时不计入活动量，仍除14天，采用保守口径。共有{int(candidates.eligible.sum())}个站点满足条件。

分别求工作日、周末各小时的平均净流出（借出−归还），对24个小时逐项取绝对值后相加，再按10/14与4/14加权：

`潮汐分数 = (10/14) × Σ小时|工作日平均净流出| + (4/14) × Σ小时|周末平均净流出|`

单位为典型一天的时段流量差次数。先平均再取绝对值，强调反复出现的时段方向；不将全天净流出先相加，避免早出晚归相互抵消。普通工作日小时分母10天，最初三个小时仅9个完整工作日；周末每小时4天，未观察完整的小时不补0进入均值。

按分数降序选前10；并列先比较活动总量降序，再按站点ID文本升序。名称只取筛选期出现最多的名称，并列按名称文本排序，不用测试期选择显示名称。名单在1月15日之前冻结，最终测试不重新筛选。

{markdown_table(top)}

## 3. 共享模型如何拟合，为什么适用

10站合并训练，但借出与归还分别拟合。原有16个特征为24/12小时正余弦、星期类别、周末及其小时交互、该站上周同小时需求；加入9个站点类别指示，共25个特征。站点ID不是连续数值。跨站共享时间规律，同时通过站点类别和本站历史需求表达站点差异。

共享模型减少分别为短序列拟合大量独立模型的需要；它也可能不完全贴合某站，因此保留逐站评估。跨站数据增加并没有增加独立历史周数，不能据样本行数声称充分验证。

| 方法 | 拟合与理由 | 优点和局限 |
|---|---|---|
| 历史同期均值 | 按站点、星期、小时平均截至预测前的完整观测 | 透明易复核；对周规模变化反应慢 |
| 上周同期 | 预测值直接等于该站上周对应小时 | 简单的补充对照；容易延续上周异常；只保留在完整评估文件 |
| Ridge | 标准化后拟合线性关系，优化平方误差＋α×系数平方和 | 系数受约束、复杂度可控；依赖给定特征表达 |
| 随机森林 | 自助抽样训练多棵受限决策树，取200棵树的平均 | 可表达非线性和交互；小量周历史仍可能过拟合、难预测突变 |

Ridge比较α=0.1/1/10/100；随机森林比较最大深度3/5、叶节点最少样本8/16，固定200棵树和随机种子42。两个学习模型使用相同特征；标准化只用训练行拟合。借还分别选择各家族的参数，统一方法家族可使用不同方向参数。

## 4. 整周预测与统一推荐

| 阶段 | 使用历史 | 一次预测 | 借出学习行数 | 归还学习行数 |
|---|---|---|---:|---:|
| 验证 | 1月1–14日 | 1月15–21日，168小时 | 1680 | 1650 |
| 最终测试 | 1月1–21日 | 1月22–28日，168小时 | 3360 | 3330 |

周滞后消耗最初168小时；归还还需剔除由月初3个不完整小时引起的滞后缺失。目标周168小时的输入全部在周初已知。周内不回填实际需求；1月29–31日不参与选型或评价，不生成跨月预测。

验证先为借出、归还各自选择参数，再在历史均值、Ridge、随机森林中按以下分数统一推荐：

`统一分数 = (10站借出MAE平均 + 10站归还MAE平均) / 2`

并列先比对应两方向合并RMSE的平均，再按上述方法顺序。上周同期只作完整评估中的补充对照，不进入页面或正式推荐候选。推荐及参数在最终测试前冻结。最终测试仅检验，不回头调参或更改站点。

借出、归还的负预测各自截为0后，再相减形成预测净流出。净流出允许为负；原始预测亦保留。净流出误差可能因借还偏差抵消而较小，因此选型仍以借还平均MAE为主。

验证统一选定：**{result['name']}**。验证借还综合MAE为{result['validation_combined_mae']:.4f}，最终测试为{result['test_combined_mae']:.4f}；最终测试逐站平均净流出MAE为{result['test_mean_net_mae']:.4f}。

**{result['verdict']}** 推荐是针对10站整体的统一选择，不保证每个站点、每个目标都是单独最优。

10站测试周借出实际{result['test_volume']['outflow']['actual']:,.0f}次，预测{result['test_volume']['outflow']['predicted']:,.0f}次，总量偏差{format_value(result['test_volume']['outflow']['relative_bias'], '+.1%')}；归还实际{result['test_volume']['inflow']['actual']:,.0f}次，预测{result['test_volume']['inflow']['predicted']:,.0f}次，偏差{format_value(result['test_volume']['inflow']['relative_bias'], '+.1%')}。需要结合总量偏差判断效果，不能仅凭对照排名或较小的净流出误差认定高精度。

## 5. 误差如何评价

MAE是每小时绝对误差的平均，越低越好；RMSE是平方误差平均后开平方，更重视大误差；R²用目标序列自身均值作误差参照，可能为负，不是准确率。站点目标完全不变时R²记为无定义，不填0。

逐站分别计算借出、归还、净流出的MAE/RMSE/R²与高峰MAE。总体主要看逐站MAE平均，不只看站点总量相加后的误差。高峰分别按预测前工作日/周末每小时均值最高两个小时确定；净流出按历史绝对净流出强度确定。日／周总量偏差为预测减真实；净流出总量可正可负、可能接近0，因此不报告其总量百分比误差。

最终测试整体指标如下，完整逐站指标见CSV：

{markdown_table(aggregate)}

## 6. 展示与结论边界

![排名第一站点的借出预测示例]({example})

HTML先解释筛选、模型与指标，再通过站点和目标选择器，每次展示一张最终测试周图。曲线为真实值、历史均值、Ridge、随机森林；验证图、日周总量、RF重要性折叠显示。默认站点按训练期潮汐排名选择，不按测试效果挑选。

图10只是借出与归还两个随机森林的内部重要性，按信息类型合并。类别或相关特征可能分摊权重，不能解释为因果，也不能证明模型有效。

只有一次验证周与一次最终测试周；样本仅为本次冻结Top10和可归属事件，不能推广成所有站点需求、真实库存或缺车量。没有天气、活动、容量等解释变量，不能确定需求波动原因。

## 7. 工程与复现（不进入HTML）

```powershell
conda run -n Pyclass python -m src.station_model
conda run -n Pyclass python -B -m unittest discover -s tests -p test_station_model.py -v
```

独立生成reports/station_prediction.html；参数--input和--output-dir可指定输入和输出。本步骤不调用全城预测入口，不覆盖已验收成果。

- station_candidates.csv / station_top10.csv：候选活动量、潮汐分数、资格、排序与冻结名单。
- station_hourly_events.csv：10站×744小时，含借出、归还、净流出和归还完整性标记。
- station_weekly_predictions.csv：40,320行，2周×10站×168小时×4方法×3目标；含预测起点、历史来源、原始与最终预测。
- station_model_metrics.csv / station_overall_metrics.csv：逐站240行与总体24行。
- station_parameter_candidates.csv / station_method_scores.csv：20个参数候选及验证统一选型分数。
- station_daily_weekly_totals.csv：日周总量与偏差，净流出百分比不定义。
- station_feature_importance.csv / station_prediction_validation.csv：RF内部重要性与运行核验。
- station_model_selection.json：冻结名单、参数、输入、特征与来源说明。

CSV保留10位小数，HTML为阅读适当舍入。输入SHA-256：`{metadata['input_sha256']}`；站点代码SHA-256：`{metadata['code_sha256']}`；复用时间特征代码SHA-256：`{metadata['shared_code_sha256']}`。

选定参数：

```json
{params}
```

参考：[跨序列共享预测研究](https://arxiv.org/abs/2008.00444)、[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)、[随机森林](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)、[评价指标](https://scikit-learn.org/stable/modules/model_evaluation.html)。方法原理不替代本数据上的实测证据。
"""
    path.write_text(text, encoding="utf-8")


def write_html(experiment: StationExperiment, selected: pd.DataFrame, candidates: pd.DataFrame,
               totals: pd.DataFrame, metadata: dict, charts: dict, importance_path: Path, path: Path) -> None:
    result = summary(experiment)
    chart_data = {site: {target: {phase: png_data_url(file) for phase, file in phases.items()} for target, phases in targets.items()} for site, targets in charts.items()}
    metrics = experiment.metrics.loc[experiment.metrics.model.isin(DISPLAY_MODELS)]
    chosen_totals = totals.loc[totals.model.eq(result["model"]) & totals.phase.eq("test")]
    payload = {"stations": selected[["station_id", "station_name", "selection_rank"]].to_dict(orient="records"),
               "charts": chart_data, "metrics": metrics.to_dict(orient="records"),
               "totals": chosen_totals.to_dict(orient="records"), "recommended": result["model"],
               "modelNames": {key: MODEL_NAMES[key] for key in DISPLAY_MODELS}, "targetNames": TARGET_NAMES,
               "chart_png_sha256": {site: {target: {phase: hashlib.sha256(file.read_bytes()).hexdigest() for phase,file in phases.items()} for target,phases in targets.items()} for site,targets in charts.items()},
               "importance_png_sha256": hashlib.sha256(importance_path.read_bytes()).hexdigest()}
    # 将日期转换为普通字符串，再安全放入JSON数据区；站点名称不作为代码或HTML执行。
    for row in payload["totals"]: row["date"] = pd.Timestamp(row["date"]).strftime("%m/%d")
    bundle = json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")
    top_rows = "".join(f"<tr><td>{int(row.selection_rank)}</td><td>{escape(row.station_name or row.station_id)}<small>{escape(row.station_id)}</small></td><td>{row.tidal_score:.2f}</td><td>{row.daily_activity:.1f}</td></tr>" for row in selected.itertuples())
    css = """
    :root{font-family:"Microsoft YaHei","Segoe UI",sans-serif;font-size:15px;line-height:1.8;color:#23394a;background:#f3f6f7}*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:32px 24px 50px}header{padding:28px 32px;border-radius:16px;color:white;background:#183f50}h1{font-size:30px;line-height:1.4;margin:0 0 10px}header p{color:#d7e7ec;margin:0}nav{display:flex;gap:22px;flex-wrap:wrap;margin:20px 0}a{color:#13766f;text-decoration:none}nav a{font-weight:600}section{margin-top:20px;background:white;border:1px solid #e0e8eb;border-radius:14px;padding:24px}h2{font-size:21px;margin:0 0 14px}h3{font-size:17px;margin:0 0 10px}p{margin:10px 0}.step{font-size:12px;color:#16867c;letter-spacing:.08em}.note{color:#677981;font-size:13px}.table-scroll{max-width:100%;overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 13px;border-bottom:1px solid #e3ecee;text-align:right}th{background:#eef4f5;color:#365967}td:first-child,th:first-child{text-align:left}td:nth-child(2),th:nth-child(2){text-align:left}td small{display:block;color:#74858d;font-size:11px}.model-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.model{padding:18px;background:#f5f9f9;border-radius:10px;border-top:3px solid #2a9187}.model p{font-size:14px}.metric-guide th,.metric-guide td{text-align:left;vertical-align:top}.metric-guide td:first-child{width:130px;font-weight:600;color:#245d64}.baseline{border-left:3px solid #d18b42;background:#fff9f0;padding:10px 14px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:16px 0}.card{border:1px solid #dce8e8;border-radius:10px;padding:16px}.card small{display:block;color:#60727b}.card strong{display:block;color:#156d69;font-size:22px;line-height:1.5}.card span{font-size:12px;color:#61727c}.controls{display:grid;grid-template-columns:2fr 1fr;gap:18px;margin:18px 0;padding:16px;background:#f3f8f8;border-radius:10px}label{display:block;font-size:13px;font-weight:600;color:#37626a}select{width:100%;min-width:0;margin-top:6px;border:1px solid #c6d9dc;background:white;color:#23394a;border-radius:7px;padding:10px;font-family:inherit;font-size:14px}.figure{display:block;width:100%;border:0;padding:0;background:white;cursor:zoom-in}.figure img{display:block;width:100%;height:auto}.verdict{padding:12px 16px;border-left:4px solid #288f86;background:#edf6f4}details{border-top:1px solid #e3eaed;margin-top:18px;padding-top:14px}summary{cursor:pointer;color:#176f6a;font-weight:600}.sources,footer{font-size:12px;color:#687c84;margin-top:18px}dialog{max-width:98vw;border:0;border-radius:12px;padding:16px}dialog::backdrop{background:#112632d9}dialog img{display:block;max-width:94vw;max-height:85vh;object-fit:contain}dialog button{display:block;margin:0 0 10px auto;padding:6px 16px;background:#e8f2f0;color:#205e5b;border:0;border-radius:6px;cursor:pointer}@media(max-width:680px){main{padding:18px 12px 32px}header{padding:24px 20px}h1{font-size:25px}section{padding:20px 14px}.model-grid,.cards,.controls{grid-template-columns:1fr}nav{gap:15px}th,td{padding:9px 8px;font-size:12px}.metric-guide td:first-child{width:90px}.card strong{font-size:22px}}
    """
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>核心站点周需求预测</title><style>{css}</style></head><body><main>
    <header><h1>核心站点：未来一周的借出与归还</h1><p>用预测前的潮汐规律选出10个站点，再检验每周初一次预测未来168小时的表现。</p></header>
    <nav aria-label="页内导航"><a href="#selection">为什么选这些站</a><a href="#methods">模型与指标</a><a href="#results">逐站预测</a><a href="#conclusion">结论与局限</a></nav>
    <section id="selection"><span class="step">01 · 筛选核心站点</span><h2>关注反复出现的时段失衡</h2><p>先用1月1–14日的已知端点事件筛选：日均借出＋归还至少20次，且至少7个活动日，共有{int(candidates.eligible.sum())}个候选站点。</p><p>分别计算工作日、周末各小时的平均“借出−归还”，取各小时差值的绝对值后相加，再按天数加权。分数越高，典型一天的时段流量差越明显；早出晚归不会因全天抵消而漏选。</p>
    <div class="table-scroll"><table id="top10"><thead><tr><th>排名</th><th>站点</th><th>潮汐分数</th><th>日均活动</th></tr></thead><tbody>{top_rows}</tbody></table></div>
    <p class="note">分数与日均活动均按次数计；排名只使用预测前两周，后续不因测试表现更换站点。</p>
    <details><summary>查看数据范围与边界</summary><p>起点已知就计入借出，终点已知就计入归还，缺另一端不删除可用事件。1月可归属借出{metadata['outflow_events_in_january']:,}次、归还{metadata['inflow_events_in_january']:,}次。</p><p>月初前三小时可能缺少12月出发的归还，因此不当作完整归还目标；筛选时双方都排除这些小时。其余已覆盖小时无事件计0。结果描述可归属的观测事件，不是全部潜在需求。</p></details></section>

    <section id="methods"><span class="step">02 · 模型与指标</span><h2>为什么让10个站点共享模型？</h2><p>每站只有几周历史。把10站数据合并，可以共同学习时间规律，再用站点类别与本站历史需求区分差异；借出和归还分别拟合。共享不保证每站都更准，因此仍逐站检验。</p>
    <div class="model-grid"><article class="model"><h3>Ridge 岭回归</h3><p>带系数约束的线性回归。把小时、星期、站点和历史需求的影响相加，在拟合历史的同时限制系数过大。</p><p><strong>优点：</strong>结构简单、复杂度可控。<strong>局限：</strong>复杂变化依赖特征表达，不能保证跟上突发需求。</p></article><article class="model"><h3>随机森林</h3><p>多棵决策树分别从历史样本学习，再将200棵树的预测取平均；树的复杂度由验证周选择。</p><p><strong>优点：</strong>可表达非线性和组合影响。<strong>局限：</strong>短期历史仍可能过拟合，未见过的变化难以预测。</p></article></div>
    <p class="baseline"><strong>对照：历史同期均值。</strong>分别平均该站过去相同星期、相同小时的次数。借出、归还先分别预测，再相减得到净流出；净流出允许为负。</p>
    <div class="table-scroll"><table class="metric-guide"><thead><tr><th>指标</th><th>含义与读法</th></tr></thead><tbody><tr><td>MAE<br>平均绝对误差</td><td>每小时预测与真实值相差次数的绝对值，再求平均。越小越好，是本次选型的主要依据。</td></tr><tr><td>RMSE<br>均方根误差</td><td>误差平方后平均，再开平方；大误差的惩罚更重，单位仍为次数。越小越好。</td></tr><tr><td>R²<br>决定系数</td><td>与始终预测该目标实际均值相比，平方误差改善多少。1表示吻合，0表示相当，负数表示更差；它不是准确率，真实值完全不变时记为无定义。</td></tr></tbody></table></div>
    <details><summary>查看选型与整周预测安排</summary><p>用1月1–14日预测15–21日，分别选择借出和归还参数，再用“10站借出MAE平均＋10站归还MAE平均”的一半，统一推荐一类方法。推荐与参数冻结后，用1–21日重训，预测22–28日作为最终测试。</p><p>输入为日历信息、站点类别和本站上周对应小时的需求，整周都在周初可知，周内不更新实际需求。净流出误差可能互相抵消，因此选型以借还两方向为主。</p></details></section>

    <section id="results"><span class="step">03 · 逐站预测</span><h2>一次查看一个站点、一个目标</h2>
    <div class="cards"><div class="card"><small>10站统一推荐方法</small><strong>{result['name']}</strong><span>依据验证周借还综合表现选择</span></div><div class="card"><small>最终测试借还综合MAE</small><strong>{result['test_combined_mae']:.2f} 次/小时</strong><span>20个借出／归还序列的平均MAE</span></div><div class="card"><small>最终测试净流出平均MAE</small><strong>{result['test_mean_net_mae']:.2f} 次/小时</strong><span>逐站先算误差，再取平均</span></div></div>
    <div class="controls"><label>站点<select id="station-select" aria-label="选择站点"></select></label><label>预测对象<select id="target-select" aria-label="预测对象"><option value="outflow">借出</option><option value="inflow">归还</option><option value="net_outflow">净流出</option></select></label></div>
    <h3 id="current-title"></h3><button class="figure" type="button" aria-label="放大最终测试图"><img id="test-image" alt="最终测试周预测"></button><p class="note">1月22–28日最终测试周。真实值与三种方法同图，点击可放大。统一推荐针对整体，不保证每个站点或目标单独最优。</p>
    <div class="table-scroll"><table><thead><tr><th>方法</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th></tr></thead><tbody id="test-metrics"></tbody></table></div>
    <p class="note" id="target-note"></p>
    <details id="validation-results"><summary>查看当前站点、目标的验证周表现</summary><button class="figure" type="button" aria-label="放大验证图"><img id="validation-image" alt="验证周预测"></button><div class="table-scroll"><table><thead><tr><th>方法</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th></tr></thead><tbody id="validation-metrics"></tbody></table></div></details>
    <details><summary>查看统一推荐方法的每日与整周总量</summary><div class="table-scroll"><table><thead><tr><th>单位</th><th>起始日</th><th>真实次数</th><th>预测次数</th><th>总量偏差</th></tr></thead><tbody id="daily-totals"></tbody></table></div><p class="note">偏差＝预测−真实。净流出为有正负的流量差，总量偏差与逐小时误差分开看。</p></details></section>

    <section id="conclusion"><span class="step">04 · 结论与局限</span><h2>哪些结论有依据？</h2><p class="verdict">{result['verdict']}</p>
    <p>10站测试周借出实际{result['test_volume']['outflow']['actual']:,.0f}次，预测{result['test_volume']['outflow']['predicted']:,.0f}次，偏差{format_value(result['test_volume']['outflow']['relative_bias'], '+.1%')}；归还实际{result['test_volume']['inflow']['actual']:,.0f}次，预测{result['test_volume']['inflow']['predicted']:,.0f}次，偏差{format_value(result['test_volume']['inflow']['relative_bias'], '+.1%')}。总量变化仍未充分捕捉，对照排名较好不等于高精度预测。</p>
    <p>站点名单在验证前冻结，方法依据验证周选择；最终测试只检验。当前只有一次验证与一次最终测试，站点间共享数据并没有增加独立历史周数，不能保证未来长期稳定。</p><p>净流出为正表示借出多于归还，为负表示归还更多；它不是库存变化或缺车数量。未知站点没有补全，也没有天气、活动、容量等信息。</p>
    <details id="feature-explanation"><summary>补充：共享随机森林主要使用哪些信息？</summary><button class="figure" type="button" aria-label="放大特征解释图"><img src="{png_data_url(importance_path)}" alt="借出与归还随机森林的内部特征重要性"></button><p class="note">分别解释两个随机森林，按信息类型合并。类别和相关特征可能分摊权重，重要性不代表因果或预测有效。</p></details>
    <p class="sources">方法参考：<a href="https://arxiv.org/abs/2008.00444">共享预测研究</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html">Ridge</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">随机森林</a> · <a href="https://scikit-learn.org/stable/modules/model_evaluation.html">评价指标</a></p></section>
    <footer>站点数据、图片和表格已内嵌，可单独复制此页面离线查看。</footer><dialog id="zoom"><button type="button">关闭</button><img alt="图表放大"></dialog><script id="station-data" type="application/json">{bundle}</script>
    """
    page += """<script>
    const data=JSON.parse(document.getElementById('station-data').textContent);
    const stationSelect=document.getElementById('station-select'),targetSelect=document.getElementById('target-select');
    const number=value=>value===null||value===undefined?'无定义':Number(value).toLocaleString('zh-CN',{minimumFractionDigits:2,maximumFractionDigits:2});
    data.stations.forEach(site=>{const option=document.createElement('option');option.value=site.station_id;option.textContent=site.selection_rank+'. '+(site.station_name||site.station_id)+' ['+site.station_id+']';stationSelect.append(option);});
    function fillRows(id,rows){const body=document.getElementById(id);body.replaceChildren();rows.forEach(values=>{const tr=document.createElement('tr');values.forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});body.append(tr);});}
    function render(){const id=stationSelect.value,target=targetSelect.value,site=data.stations.find(s=>s.station_id===id);document.getElementById('current-title').textContent=(site.station_name||id)+' · '+data.targetNames[target];
      document.getElementById('test-image').src=data.charts[id][target].test;document.getElementById('validation-image').src=data.charts[id][target].validation;
      ['test','validation'].forEach(phase=>{const rows=Object.keys(data.modelNames).map(model=>{const m=data.metrics.find(r=>r.station_id===id&&r.target===target&&r.phase===phase&&r.model===model);return [data.modelNames[model]+(model===data.recommended?'（统一推荐）':''),number(m.mae),number(m.rmse),number(m.r2)];});fillRows(phase+'-metrics',rows);});
      const totals=data.totals.filter(r=>r.station_id===id&&r.target===target);fillRows('daily-totals',totals.map(r=>[r.period==='day'?'日':'周',r.date,number(r.actual),number(r.prediction),number(r.bias)]));
      document.getElementById('target-note').textContent=target==='net_outflow'?'净流出＝借出−归还，允许为负；当前曲线由同一方法的借还预测相减得到。':'次数预测保留小数，以表示预计需求；误差计算不提前取整。';
    }
    stationSelect.addEventListener('change',render);targetSelect.addEventListener('change',render);render();
    const dialog=document.getElementById('zoom');document.querySelectorAll('.figure').forEach(b=>b.addEventListener('click',()=>{dialog.querySelector('img').src=b.querySelector('img').src;dialog.showModal();}));dialog.querySelector('button').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
    </script></main></body></html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR/"clean_trips.csv")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    progress = lambda message: print(message, flush=True)
    progress("读取有效集，按已知端点构造事件…")
    data = read_clean_data(args.input)
    events, metadata = extract_events(data)
    candidates, selected = select_stations(events)
    progress(f"{int(candidates.eligible.sum())}个合格候选，已冻结Top10；构造10站完整小时网格…")
    panel = build_panel(events, selected.station_id.tolist())
    experiment = run_experiment(panel, selected, progress=progress)
    metadata.update({"eligible_stations": int(candidates.eligible.sum()),
                     "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                     "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "shared_code_sha256": hashlib.sha256((Path(__file__).parent/"demand_model.py").read_bytes()).hexdigest(),
                     "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                  "matplotlib": matplotlib.__version__, "scikit_learn": sklearn.__version__}})
    experiment.selection["provenance"] = metadata
    experiment.selection["feature_names"] = experiment.fits[("ridge", "outflow")].feature_names
    ids = selected.station_id.tolist()
    known_month = events.loc[events.station_id.isin(ids) & events.timestamp.ge(MONTH_START) & events.timestamp.lt(MONTH_END)]
    checks = []
    def check(label, observed, expected):
        checks.append({"phase":"input_selection", "check":label, "observed":str(observed), "expected":str(expected), "passed":observed==expected})
    check("top10_unique_ids", selected.station_id.nunique(), 10)
    check("all_selected_eligible", bool(selected.eligible.all()), True)
    check("panel_rows", len(panel), 10*744)
    check("unknown_first_3h_inflow", int(panel.inflow.isna().sum()), 30)
    check("outflow_event_conservation", float(panel.outflow.sum()), float(known_month.direction.eq("outflow").sum()))
    check("complete_inflow_event_conservation", float(panel.inflow.sum()), float((known_month.direction.eq("inflow") & known_month.timestamp.ge(SELECTION_START)).sum()))
    check("prediction_rows", len(experiment.predictions), 2*10*168*4*3)
    check("prediction_keys_unique", not bool(experiment.predictions.duplicated(["phase","station_id","timestamp","model","target"]).any()), True)
    check("per_station_metric_rows", len(experiment.metrics), 2*4*3*10)
    experiment.audit = pd.concat([pd.DataFrame(checks), experiment.audit], ignore_index=True)
    if not experiment.audit.passed.all(): raise ValueError("输入或筛选核验失败。")
    font = set_plot_style()
    out = args.output_dir
    for folder in (out/"figures", out/"tables", out/"sections"): folder.mkdir(parents=True, exist_ok=True)
    totals = daily_weekly_totals(experiment.predictions)
    exports = {"station_candidates.csv":candidates, "station_top10.csv":selected,
               "station_hourly_events.csv":panel.reset_index(), "station_weekly_predictions.csv":experiment.predictions,
               "station_model_metrics.csv":experiment.metrics, "station_overall_metrics.csv":experiment.overall,
               "station_parameter_candidates.csv":experiment.candidates, "station_method_scores.csv":experiment.method_scores,
               "station_daily_weekly_totals.csv":totals, "station_feature_importance.csv":importances(experiment),
               "station_prediction_validation.csv":experiment.audit}
    for name, frame in exports.items(): frame.to_csv(out/"tables"/name, index=False, encoding="utf-8-sig", float_format="%.10f", date_format="%Y-%m-%d %H:%M:%S")
    (out/"tables/station_model_selection.json").write_text(json.dumps(json_safe(experiment.selection), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    progress("生成按站点切换的预测图与报告…")
    charts, importance = plot_results(experiment, selected, out/"figures", progress)
    write_markdown(experiment, selected, candidates, metadata, charts, out/"sections/04_station_demand_prediction.md")
    write_html(experiment, selected, candidates, totals, metadata, charts, importance, out/"station_prediction.html")
    progress(json.dumps(json_safe({"top10":selected[["selection_rank","station_id","station_name","tidal_score"]].to_dict(orient="records"),
                                  "summary":summary(experiment), "checks_passed":len(experiment.audit), "font":font,
                                  "report":str(out/"station_prediction.html")}), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
