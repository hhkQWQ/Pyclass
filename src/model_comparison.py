"""Read saved city/station predictions; derive comparisons without fitting models."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

COMPARISON_MODELS = ("historical_mean", "last_week", "ridge", "random_forest")
COMPARISON_DISPLAY = ("historical_mean", "ridge", "random_forest")
COMPARISON_TARGETS = ("outflow", "inflow", "net_outflow")
COMPARISON_NAMES = {"historical_mean": "历史均值", "last_week": "上周同期", "ridge": "Ridge", "random_forest": "随机森林"}
COMPARISON_TARGET_NAMES = {"city_demand": "全城需求", "outflow": "站点借出", "inflow": "站点归还", "net_outflow": "站点净流出", "combined": "借还综合"}
COMPARISON_ORIGINS = {"validation": pd.Timestamp("2024-01-15"), "test": pd.Timestamp("2024-01-22")}


def comparison_json_safe(value):
    if isinstance(value, dict):
        return {str(k): comparison_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [comparison_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return comparison_json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def comparison_metrics(frame):
    actual = frame.actual.to_numpy(dtype=float)
    prediction = frame.prediction.to_numpy(dtype=float)
    error = prediction - actual
    peaks = frame.is_peak.to_numpy(dtype=bool)
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": 1.0 - float(np.sum(error ** 2)) / denominator if denominator > 0 else None,
            "peak_mae": float(np.mean(np.abs(error[peaks]))) if peaks.any() else None,
            "total_actual": float(actual.sum()), "total_predicted": float(prediction.sum()),
            "total_bias": float(prediction.sum() - actual.sum())}


def comparison_improvement(baseline, value):
    return 100.0 * (baseline - value) / baseline if baseline != 0 else None


def comparison_bias_pct(actual, bias, target):
    return 100.0 * bias / actual if target != "net_outflow" and actual != 0 else None


def comparison_peak_mask(history, timestamps, absolute=False):
    """Independently reconstruct the already frozen top-two-hour definition."""
    series = history.abs() if absolute else history
    means = series.groupby([series.index.dayofweek >= 5, series.index.hour]).mean()
    result = np.zeros(len(timestamps), dtype=bool)
    for weekend in (False, True):
        group = means.loc[weekend]
        chosen = sorted(group.index, key=lambda hour: (-float(group.loc[hour]), hour))[:2]
        result |= ((timestamps.dayofweek >= 5) == weekend) & np.isin(timestamps.hour, chosen)
    return result


def comparison_plot_style():
    global plt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    selected = None
    required = set(map(ord, "共享单车预测站点误差比较"))
    for name in ("Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", "SimHei"):
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=[name]), fallback_to_default=False)
            if required.issubset(font_manager.get_font(path).get_charmap()):
                selected = name
                break
        except (ValueError, RuntimeError, OSError):
            continue
    if selected is None:
        raise RuntimeError("没有找到可以完整显示中文的现有字体。")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [selected, "DejaVu Sans"],
                         "font.size": 11, "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#d9e3e7", "text.color": "#23394a",
                         "axes.labelcolor": "#526771", "xtick.color": "#526771", "ytick.color": "#526771",
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})
    return selected


def comparison_figures(relative, station, target_reports):
    font = comparison_plot_style()
    figures = target_reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frame = relative.loc[relative.phase.eq("test") & relative.model.isin(COMPARISON_DISPLAY)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
    values_all = frame.mae_improvement_pct.dropna().to_numpy()
    lower = min(-10, float(values_all.min()) - 12)
    upper = max(12, float(values_all.max()) + 12)
    palette = ["#92a9ad", "#315d8a", "#d78538"]
    for ax, target in zip(axes.flat, ("city_demand", *COMPARISON_TARGETS)):
        values = [float(frame.loc[frame.target.eq(target) & frame.model.eq(model), "mae_improvement_pct"].iloc[0]) for model in COMPARISON_DISPLAY]
        y = np.arange(3)
        ax.barh(y, values, color=palette, height=0.52)
        ax.axvline(0, color="#758e95", linewidth=1)
        ax.set_yticks(y, [COMPARISON_NAMES[m] for m in COMPARISON_DISPLAY])
        ax.invert_yaxis()
        ax.set_xlim(lower, upper)
        ax.set_title(COMPARISON_TARGET_NAMES[target], fontsize=13, fontweight="bold", loc="left", pad=12)
        ax.xaxis.grid(True, color="#e5ecef", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("MAE 改善率（%）")
        for row, value in enumerate(values):
            ax.text(value - 1.4 if value < 0 else value + 1.4, row, f"{value:+.1f}%" if value != 0 else "0.0% · 参照", ha="right" if value < 0 else "left", va="center", fontsize=11)
    fig.suptitle("图 17｜模型相对历史均值的误差变化", fontsize=17, fontweight="bold", x=.055, ha="left", y=.985)
    fig.text(.055, .925, "最终测试：2024 年 1 月 22–28 日；正值表示改善，负值表示退步。", fontsize=11, color="#60767e")
    fig.text(.055, .025, "各目标分别以自身的历史均值为参照；不据此对全城与站点预测层级排名。", fontsize=10, color="#60767e")
    fig.subplots_adjust(top=.84, bottom=.12, hspace=.55, wspace=.40, left=.12, right=.97)
    fig.savefig(figures / "fig17_model_comparison.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.25)
    plt.close(fig)

    combined = station.loc[station.phase.eq("test") & station.target.eq("combined") & station.model.isin(COMPARISON_DISPLAY)]
    ordered = combined.loc[combined.model.eq("historical_mean")].sort_values("selection_rank")
    ids = ordered.station_id.tolist()
    fig, ax = plt.subplots(figsize=(12, 8.4))
    y = np.arange(len(ids))
    bar_height = .22
    for offset, model, color in zip((-bar_height, 0, bar_height), COMPARISON_DISPLAY, ("#159a85", "#315d8a", "#d78538")):
        values = combined.loc[combined.model.eq(model)].set_index("station_id").reindex(ids).mae.to_numpy()
        ax.barh(y + offset, values, height=bar_height, color=color, label=COMPARISON_NAMES[model])
    ax.set_yticks(y, [f"{int(rank):02d} · {sid}" for sid, rank in zip(ids, ordered.selection_rank)])
    ax.invert_yaxis()
    ax.set_xlabel("借还综合 MAE（次/小时；越小越好）", labelpad=12)
    ax.set_ylabel("筛选排名 · 站点编号", labelpad=12)
    ax.xaxis.grid(True, color="#e5ecef", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.09), ncol=3, frameon=False)
    fig.suptitle("图 18｜10 个核心站点的预测误差差异", fontsize=17, fontweight="bold", x=.06, ha="left", y=.985)
    fig.text(.06, .93, "最终测试：2024 年 1 月 22–28 日；站点顺序沿用训练期冻结的筛选排名。", fontsize=11, color="#60767e")
    fig.text(.06, .025, "借还综合 MAE＝（借出 MAE＋归还 MAE）÷2；各站名称与分目标结果见配套表格。", fontsize=10, color="#60767e")
    fig.subplots_adjust(top=.81, bottom=.13, left=.15, right=.96)
    fig.savefig(figures / "fig18_station_error_comparison.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.25)
    plt.close(fig)
    return font


def build_analysis(source_reports: Path, target_reports: Path, *, render_figures: bool = True) -> dict:
    source_reports, target_reports = Path(source_reports), Path(target_reports)
    source_tables = source_reports / "tables"
    tables = target_reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    input_names = ("city_weekly_predictions.csv", "city_model_metrics.csv", "city_daily_weekly_totals.csv",
                   "city_hourly_demand.csv", "city_parameter_candidates.csv", "city_model_selection.json",
                   "city_feature_importance.csv", "station_weekly_predictions.csv", "station_model_metrics.csv",
                   "station_overall_metrics.csv", "station_daily_weekly_totals.csv", "station_hourly_events.csv",
                   "station_parameter_candidates.csv", "station_method_scores.csv", "station_top10.csv",
                   "station_model_selection.json", "station_feature_importance.csv")
    source_hashes = {name: hashlib.sha256((source_tables / name).read_bytes()).hexdigest() for name in input_names}
    frames = {name: pd.read_csv(source_tables / name, dtype={"station_id": str}) for name in input_names if name.endswith(".csv")}
    selection = {scope: json.loads((source_tables / f"{scope}_model_selection.json").read_text(encoding="utf-8")) for scope in ("city", "station")}
    checks = []

    def check(name, condition, observed=None, expected=True):
        record = {"check": name, "passed": bool(condition), "observed": comparison_json_safe(observed), "expected": comparison_json_safe(expected)}
        checks.append(record)
        if not record["passed"]:
            raise ValueError(f"比较报告核对失败：{name}；实际 {observed!r}，预期 {expected!r}")

    def close_records(name, calculated, original, keys, columns):
        left, right = calculated.set_index(keys).sort_index(), original.set_index(keys).sort_index()
        check(name + ":keys", left.index.equals(right.index), len(left), len(right))
        for column, source_column in columns.items():
            a, b = left[column].to_numpy(dtype=float), right[source_column].to_numpy(dtype=float)
            check(name + ":" + column, np.allclose(a, b, rtol=2e-9, atol=1e-8, equal_nan=True), float(np.nanmax(np.abs(a-b))) if np.isfinite(a-b).any() else None, 0)

    city = frames["city_weekly_predictions.csv"].copy()
    station_predictions = frames["station_weekly_predictions.csv"].copy()
    top10 = frames["station_top10.csv"].sort_values("selection_rank")
    ids = top10.station_id.tolist()
    names = top10.set_index("station_id").station_name.to_dict()
    ranks = top10.set_index("station_id").selection_rank.to_dict()
    check("frozen_station_ids", ids == selection["station"]["station_ids"], ids, selection["station"]["station_ids"])
    check("ten_unique_stations", len(ids) == len(set(ids)) == 10, len(ids), 10)
    for scope, data in (("city", city), ("station", station_predictions)):
        required = {"phase", "timestamp", "forecast_origin", "lag_source_time", "model", "actual", "prediction", "is_peak", "recommended_frozen"}
        check(scope + ":required_columns", required.issubset(data.columns), sorted(data.columns), sorted(required))
        for column in ("timestamp", "forecast_origin", "lag_source_time"):
            data[column] = pd.to_datetime(data[column], errors="raise")
        for column in ("is_peak", "recommended_frozen"):
            check(scope + ":boolean_" + column, data[column].isin([True, False]).all())
            data[column] = data[column].astype(bool)
        check(scope + ":model_set", set(data.model) == set(COMPARISON_MODELS), sorted(data.model.unique()), list(COMPARISON_MODELS))
        check(scope + ":phase_set", set(data.phase) == set(COMPARISON_ORIGINS), sorted(data.phase.unique()), list(COMPARISON_ORIGINS))
        check(scope + ":finite_values", bool(np.isfinite(data[["actual", "prediction"]].to_numpy()).all()))
        keys = ["phase", "model", "timestamp"] + (["station_id", "target"] if scope == "station" else [])
        check(scope + ":unique_prediction_keys", not data.duplicated(keys).any())
        expected_rows = 2 * 4 * 168 * (10 * 3 if scope == "station" else 1)
        check(scope + ":prediction_rows", len(data) == expected_rows, len(data), expected_rows)
        check(scope + ":lag_before_origin", bool((data.lag_source_time < data.forecast_origin).all()))
        check(scope + ":lag_is_168h", bool((data.timestamp-data.lag_source_time).eq(pd.Timedelta(hours=168)).all()))
        check(scope + ":frozen_recommended_flags", data.recommended_frozen.eq(data.model.eq(selection[scope]["recommended_model"])).all())
        for phase, origin in COMPARISON_ORIGINS.items():
            part = data.loc[data.phase.eq(phase)]
            check(scope + ":" + phase + ":origin", part.forecast_origin.eq(origin).all(), str(part.forecast_origin.iloc[0]), str(origin))
            check(scope + ":" + phase + ":168_hour_span", set(part.timestamp) == set(pd.date_range(origin, periods=168, freq="h")))
            group_keys = ["model"] + (["station_id", "target"] if scope == "station" else [])
            check(scope + ":" + phase + ":complete_groups", part.groupby(group_keys).size().eq(168).all())
        actual_keys = ["phase", "timestamp"] + (["station_id", "target"] if scope == "station" else [])
        check(scope + ":actual_same_across_models", data.groupby(actual_keys).actual.nunique().eq(1).all())
        check(scope + ":peaks_same_across_models", data.groupby(actual_keys).is_peak.nunique().eq(1).all())
        check(scope + ":no_weekly_updates", selection[scope]["within_week_actual_updates"] is False)
        check(scope + ":no_post_test_tuning", selection[scope]["post_test_tuning"] is False)
        check(scope + ":horizon_168", selection[scope]["forecast_horizon_hours"] == 168)
    check("station:target_set", set(station_predictions.target) == set(COMPARISON_TARGETS))
    check("station:prediction_station_ids", set(station_predictions.station_id) == set(ids))
    check("city:selection_origins", all(selection["city"][phase+"_origin"] == str(origin) for phase, origin in COMPARISON_ORIGINS.items()))
    check("station:screening_before_validation", pd.Timestamp(selection["station"]["selection_period"]["end_exclusive"]) == COMPARISON_ORIGINS["validation"])
    for column in ("actual", "prediction"):
        wide = station_predictions.pivot(index=["phase", "model", "station_id", "timestamp"], columns="target", values=column)
        check("station:net_"+column+"_is_out_minus_in", np.allclose(wide.net_outflow, wide.outflow-wide.inflow, atol=2e-10, rtol=2e-9))
    check("city:nonnegative_counts", (city.prediction >= 0).all())
    check("station:nonnegative_directions", (station_predictions.loc[station_predictions.target.ne("net_outflow"), "prediction"] >= 0).all())

    city_history = frames["city_hourly_demand.csv"].copy()
    city_history["timestamp"] = pd.to_datetime(city_history.timestamp)
    city_history = city_history.set_index("timestamp").ride_count
    station_history = frames["station_hourly_events.csv"].copy()
    station_history["timestamp"] = pd.to_datetime(station_history.timestamp)
    station_history = station_history.set_index(["station_id", "timestamp"])
    for phase, origin in COMPARISON_ORIGINS.items():
        group = city.loc[city.phase.eq(phase) & city.model.eq("historical_mean")].sort_values("timestamp")
        history = city_history.loc[city_history.index < origin]
        expected_peaks = comparison_peak_mask(history, pd.DatetimeIndex(group.timestamp))
        check("city:"+phase+":frozen_peak_definition", np.array_equal(expected_peaks, group.is_peak.to_numpy()))
        check("city:"+phase+":actual_matches_saved_hourly", np.array_equal(group.actual.to_numpy(), city_history.reindex(group.timestamp).to_numpy()))
        for sid in ids:
            h = station_history.xs(sid, level="station_id")
            for target in COMPARISON_TARGETS:
                group = station_predictions.loc[station_predictions.phase.eq(phase) & station_predictions.model.eq("historical_mean") & station_predictions.station_id.eq(sid) & station_predictions.target.eq(target)].sort_values("timestamp")
                expected_peaks = comparison_peak_mask(h.loc[h.index < origin, target], pd.DatetimeIndex(group.timestamp), target == "net_outflow")
                check("station:"+phase+":"+sid+":"+target+":frozen_peak_definition", np.array_equal(expected_peaks, group.is_peak.to_numpy()))
                check("station:"+phase+":"+sid+":"+target+":actual_matches_saved_hourly", np.array_equal(group.actual.to_numpy(), h[target].reindex(group.timestamp).to_numpy()))

    city_rows = []
    for (phase, model), group in city.groupby(["phase", "model"], sort=False):
        city_rows.append({"phase": phase, "scope": "city", "target": "city_demand", "model": model, **comparison_metrics(group),
                          "recommended_frozen": model == selection["city"]["recommended_model"], "mae_label": "MAE", "rmse_label": "RMSE"})
    city_metrics = pd.DataFrame(city_rows)
    columns = {name: name for name in ("mae", "rmse", "r2", "peak_mae", "total_actual", "total_predicted", "total_bias")}
    close_records("city:metrics_recomputed", city_metrics, frames["city_model_metrics.csv"], ["phase", "model"], columns)
    station_rows = []
    for (phase, model, target, sid), group in station_predictions.groupby(["phase", "model", "target", "station_id"], sort=False):
        station_rows.append({"phase": phase, "scope": "station", "target": target, "model": model, "station_id": sid,
                             "station_name": names[sid], "selection_rank": int(ranks[sid]), **comparison_metrics(group),
                             "recommended_frozen": model == selection["station"]["recommended_model"]})
    station_metrics = pd.DataFrame(station_rows)
    close_records("station:metrics_recomputed", station_metrics, frames["station_model_metrics.csv"], ["phase", "model", "target", "station_id"], columns)
    overall_rows = []
    for (phase, model, target), group in station_predictions.groupby(["phase", "model", "target"], sort=False):
        per_station = station_metrics.loc[station_metrics.phase.eq(phase) & station_metrics.model.eq(model) & station_metrics.target.eq(target)]
        metrics = comparison_metrics(group)
        metrics.update(mae=float(per_station.mae.mean()), peak_mae=float(per_station.peak_mae.mean()), r2=None)
        overall_rows.append({"phase": phase, "scope": "station", "target": target, "model": model, **metrics,
                             "recommended_frozen": model == selection["station"]["recommended_model"],
                             "mae_label": "各站 MAE 平均值", "rmse_label": "合并 RMSE"})
    overall = pd.DataFrame(overall_rows)
    mapped = dict(columns)
    mapped.pop("r2")
    mapped.update(mae="mean_station_mae", rmse="pooled_rmse", peak_mae="mean_station_peak_mae")
    close_records("station:overall_metrics_recomputed", overall, frames["station_overall_metrics.csv"], ["phase", "model", "target"], mapped)
    summary = pd.concat([city_metrics, overall], ignore_index=True)
    summary["bias_pct"] = [comparison_bias_pct(row.total_actual, row.total_bias, row.target) for row in summary.itertuples()]
    relative = summary.copy()
    baseline = summary.loc[summary.model.eq("historical_mean")].set_index(["phase", "scope", "target"]).mae
    relative["baseline_mae"] = [float(baseline.loc[(r.phase, r.scope, r.target)]) for r in relative.itertuples()]
    relative["mae_improvement_pct"] = [comparison_improvement(r.baseline_mae, r.mae) for r in relative.itertuples()]

    combined_rows = []
    for (phase, model, sid), group in station_metrics.loc[station_metrics.target.isin(["outflow", "inflow"])].groupby(["phase", "model", "station_id"], sort=False):
        combined_rows.append({"phase": phase, "scope": "station", "target": "combined", "model": model, "station_id": sid,
                              "station_name": names[sid], "selection_rank": int(ranks[sid]), "mae": float(group.mae.mean()),
                              "rmse": None, "r2": None, "peak_mae": float(group.peak_mae.mean()),
                              "total_actual": float(group.total_actual.sum()), "total_predicted": float(group.total_predicted.sum()),
                              "total_bias": float(group.total_bias.sum()), "recommended_frozen": model == selection["station"]["recommended_model"]})
    station = pd.concat([station_metrics, pd.DataFrame(combined_rows)], ignore_index=True)
    station["bias_pct"] = [comparison_bias_pct(row.total_actual, row.total_bias, row.target) for row in station.itertuples()]
    station_baselines = station.loc[station.model.eq("historical_mean")].set_index(["phase", "station_id", "target"]).mae
    station["mae_improvement_pct"] = [comparison_improvement(float(station_baselines.loc[(r.phase, r.station_id, r.target)]), r.mae) for r in station.itertuples()]

    daily_rows = []
    for scope, predictions in (("city", city.assign(target="city_demand")), ("station", station_predictions)):
        for (phase, model, target), group in predictions.groupby(["phase", "model", "target"], sort=False):
            for day, rows in group.groupby(group.timestamp.dt.strftime("%Y-%m-%d"), sort=True):
                actual, predicted = float(rows.actual.sum()), float(rows.prediction.sum())
                daily_rows.append({"phase": phase, "scope": scope, "target": target, "model": model, "period": "day", "date": day,
                                   "actual_total": actual, "predicted_total": predicted, "bias": predicted-actual,
                                   "bias_pct": comparison_bias_pct(actual, predicted-actual, target)})
            actual, predicted = float(group.actual.sum()), float(group.prediction.sum())
            daily_rows.append({"phase": phase, "scope": scope, "target": target, "model": model, "period": "week", "date": "week",
                               "actual_total": actual, "predicted_total": predicted, "bias": predicted-actual,
                               "bias_pct": comparison_bias_pct(actual, predicted-actual, target)})
    daily = pd.DataFrame(daily_rows)
    for scope in ("city", "station"):
        original = frames[f"{scope}_daily_weekly_totals.csv"].copy()
        if scope == "city":
            original["target"] = "city_demand"
        original["date"] = pd.to_datetime(original.date).dt.strftime("%Y-%m-%d")
        original.loc[original.period.eq("week"), "date"] = "week"
        keys = ["phase", "model", "target", "period", "date"]
        original = original.groupby(keys, as_index=False)[["actual", "prediction", "bias"]].sum()
        close_records(scope+":daily_weekly_recomputed", daily.loc[daily.scope.eq(scope)], original, keys, {"actual_total": "actual", "predicted_total": "prediction", "bias": "bias"})

    for scope in ("city", "station"):
        candidates = frames[f"{scope}_parameter_candidates.csv"]
        group_keys = ["model"] if scope == "city" else ["direction", "model"]
        selected_flag = "selected_within_family" if scope == "city" else "selected_within_family_direction"
        primary = "mae" if scope == "city" else "mean_station_mae"
        for key, group in candidates.groupby(group_keys, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            best = group.sort_values([primary, "rmse", "candidate_id"]).iloc[0]
            frozen = group.loc[group[selected_flag]]
            parameter_record = selection[scope]["selected_parameters"][key[-1]] if scope == "city" else selection[scope]["selected_parameters"][key[0]][key[-1]]
            check(scope+":selected_parameters:"+":".join(key), len(frozen) == 1 and json.loads(best.parameters) == parameter_record and json.loads(frozen.iloc[0].parameters) == parameter_record)
    city_val = city_metrics.loc[city_metrics.phase.eq("validation")].copy()
    city_val["order"] = city_val.model.map({m: i for i, m in enumerate(COMPARISON_MODELS)})
    check("city:recommendation_uses_validation", city_val.sort_values(["mae", "rmse", "order"]).iloc[0].model == selection["city"]["recommended_model"])
    scores = frames["station_method_scores.csv"]
    val = overall.loc[overall.phase.eq("validation") & overall.target.isin(["outflow", "inflow"])].groupby("model", as_index=False).agg(combined_mae=("mae", "mean"), combined_rmse=("rmse", "mean"))
    close_records("station:validation_recommendation_scores", val, scores, ["model"], {"combined_mae": "combined_mae", "combined_rmse": "combined_rmse"})
    eligible = val.loc[val.model.isin(COMPARISON_DISPLAY)].copy()
    eligible["order"] = eligible.model.map({m: i for i, m in enumerate(COMPARISON_DISPLAY)})
    check("station:recommendation_uses_borrow_return_only", eligible.sort_values(["combined_mae", "combined_rmse", "order"]).iloc[0].model == selection["station"]["recommended_model"])
    check("undefined_improvement_when_baseline_zero", comparison_improvement(0, 2) is None)
    check("improvement_sign_hand_check", comparison_improvement(10, 8) == 20 and comparison_improvement(10, 12) == -20)
    check("net_bias_percentage_omitted", summary.loc[summary.target.eq("net_outflow"), "bias_pct"].isna().all() and daily.loc[daily.target.eq("net_outflow"), "bias_pct"].isna().all())

    font = comparison_figures(relative, station, target_reports) if render_figures else None
    exported = {"model_comparison_summary.csv": summary, "model_comparison_relative_mae.csv": relative,
                "model_comparison_daily_bias.csv": daily, "model_comparison_station.csv": station}
    for name, frame in exported.items():
        frame.to_csv(tables / name, index=False, encoding="utf-8-sig", float_format="%.10f", na_rep="")
    check("source_tables_unchanged_after_analysis", all(hashlib.sha256((source_tables / name).read_bytes()).hexdigest() == digest for name, digest in source_hashes.items()))
    features = {scope: frames[f"{scope}_feature_importance.csv"].to_dict(orient="records") for scope in ("city", "station")}
    feature_groups = []
    for scope in ("city", "station"):
        frame = frames[f"{scope}_feature_importance.csv"].copy()
        if scope == "city":
            frame["direction"] = "city_demand"
        grouped = frame.groupby(["direction", "group_label"], sort=False).importance.sum().reset_index()
        feature_groups.extend({"scope": scope, **row} for row in grouped.to_dict(orient="records"))
    metadata = {"forecast_horizon_hours": 168, "source_hashes": source_hashes,
                "source_prediction_rows": {"city": len(city), "station": len(station_predictions)},
                "comparison_rows": {name: len(frame) for name, frame in exported.items()}, "display_models": list(COMPARISON_DISPLAY),
                "supplementary_model": "last_week", "font": font,
                "mae_improvement_formula": "100*(historical_mean_mae-model_mae)/historical_mean_mae; undefined at denominator 0",
                "peak_definition": "top two historical mean hours per workday/weekend, per station and target; absolute history for net outflow",
                "station_rmse_definition": "square root of mean squared errors pooled across station-hours; not average station RMSE",
                "combined_definition": "(outflow MAE + inflow MAE)/2; combined RMSE and R2 intentionally undefined"}
    payload = comparison_json_safe({"summary": summary.to_dict(orient="records"), "relative": relative.to_dict(orient="records"),
                                   "daily": daily.to_dict(orient="records"), "station": station.to_dict(orient="records"),
                                   "selection": selection, "checks": checks, "top10": top10.to_dict(orient="records"),
                                   "features": features, "feature_groups": feature_groups, "metadata": metadata})
    (tables / "model_comparison_checks.json").write_text(json.dumps({"checks": payload["checks"], "metadata": payload["metadata"]}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return payload


import argparse
import base64
from html import escape
import math


REPORT_MODELS = ('historical_mean', 'ridge', 'random_forest')
REPORT_MODEL_NAMES = {'historical_mean': '历史同期均值', 'last_week': '上周同期（补充）', 'ridge': 'Ridge 岭回归', 'random_forest': '随机森林'}
REPORT_TARGET_NAMES = {'city_demand': '全城需求', 'outflow': '站点借出', 'inflow': '站点归还', 'net_outflow': '站点净流出', 'combined': '借还综合'}


def report_number(value, digits=2, signed=False):
    if value is None or not math.isfinite(float(value)):
        return '—'
    return format(float(value), f'+,.{digits}f' if signed else f',.{digits}f')


def report_table(headers, rows, caption=None):
    cap = f'<caption>{escape(caption)}</caption>' if caption else ''
    return '<div class="table-scroll" tabindex="0"><table>' + cap + '<thead><tr>' + ''.join(f'<th scope="col">{escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(str(x))}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def report_metric_table(data, scope, phase):
    headers = ['对象', '方法', 'MAE', 'RMSE', '高峰 MAE']
    if scope == 'city':
        headers.append('R²')
    headers.append('MAE 改善率')
    rows = []
    for item in data['relative']:
        if item['scope'] != scope or item['phase'] != phase or item['model'] not in REPORT_MODELS:
            continue
        d = 2 if scope == 'city' else 3
        row = [REPORT_TARGET_NAMES[item['target']], REPORT_MODEL_NAMES[item['model']], report_number(item['mae'], d), report_number(item['rmse'], d), report_number(item['peak_mae'], d)]
        if scope == 'city':
            row.append(report_number(item['r2'], 3))
        row.append(report_number(item['mae_improvement_pct'], 1, True) + '%' if item['mae_improvement_pct'] is not None else '无定义')
        rows.append(row)
    return report_table(headers, rows)


def report_grouped_features(data):
    groups = {}
    for scope, records in data['features'].items():
        for r in records:
            obj = '全城随机森林' if scope == 'city' else ('借出随机森林' if r.get('direction') == 'outflow' else '归还随机森林')
            k = (obj, r['group_label'])
            groups[k] = groups.get(k, 0) + float(r['importance'])
    rows = []
    for obj in ['全城随机森林', '借出随机森林', '归还随机森林']:
        vals = sorted(((group, value) for (name, group), value in groups.items() if name == obj), key=lambda x: -x[1])
        rows.extend([obj, group, report_number(value * 100, 1) + '%'] for group, value in vals)
    return rows


def render_comparison_report(data, target_reports):
    target_reports = Path(target_reports)
    summary = data['summary']
    get = lambda scope, target, model='historical_mean', phase='test': next(r for r in summary if r['scope'] == scope and r['target'] == target and r['model'] == model and r['phase'] == phase)
    city = get('city', 'city_demand')
    outflow, inflow, net = [get('station', t) for t in ('outflow', 'inflow', 'net_outflow')]
    combined_mae = (outflow['mae'] + inflow['mae']) / 2
    feature_rows = report_grouped_features(data)
    weekly_rows = []
    for r in (city, outflow, inflow, net):
        weekly_rows.append([REPORT_TARGET_NAMES[r['target']], report_number(r['total_actual'], 0), report_number(r['total_predicted'], 1), report_number(r['total_bias'], 1, True), '不使用百分比' if r['target'] == 'net_outflow' else report_number(r['bias_pct'], 1, True) + '%'])
    weekly_table = report_table(['对象', '实际总量', '预测总量', '预测－实际', '总量偏差率'], weekly_rows)
    diagnostics = []
    for scope, target in [('city', 'city_demand'), ('station', 'outflow'), ('station', 'inflow'), ('station', 'net_outflow')]:
        candidates = [r for r in data['daily'] if r['phase'] == 'test' and r['scope'] == scope and r['target'] == target and r['model'] == 'historical_mean' and r['period'] == 'day']
        worst = max(candidates, key=lambda r: abs(r['bias']))
        diagnostics.append([REPORT_TARGET_NAMES[target], worst['date'], report_number(worst['actual_total'], 0), report_number(worst['predicted_total'], 1), report_number(worst['bias'], 1, True)])
    diag_table = report_table(['对象', '绝对总量偏差最大日', '实际次数', '预测次数', '偏差次数'], diagnostics)
    city_days = sorted([r for r in data['daily'] if r['phase'] == 'test' and r['scope'] == 'city' and r['model'] == 'historical_mean' and r['period'] == 'day'], key=lambda r: r['bias'])
    if len(city_days) == 7 and all(r['bias'] < 0 for r in city_days):
        two_days = sorted(city_days[:2], key=lambda r: r['date'])
        deficit_share = sum(-r['bias'] for r in two_days) / -city['total_bias'] * 100
        daily_insight = f"全城历史均值在测试周 7 天均低估；{two_days[0]['date'][5:].replace('-', ' 月 ')} 日与 {two_days[1]['date'][5:].replace('-', ' 月 ')} 日合计占整周总量缺口的 {deficit_share:.1f}%。这定位了偏差集中的日期，但不能确定波动原因。"
    else:
        daily_insight = '每日总量偏差有助于定位整周偏差的来源；对需求波动原因的解释仍需额外数据。'
    validation_out = get('station', 'outflow', phase='validation')
    validation_in = get('station', 'inflow', phase='validation')
    validation_combined = (validation_out['mae'] + validation_in['mae']) / 2
    station_combined = [r for r in data['station'] if r['phase'] == 'test' and r['target'] == 'combined' and r['model'] in REPORT_MODELS]
    station_ids = {r['station_id'] for r in station_combined}
    baseline_best = sum(next(r['mae'] for r in station_combined if r['station_id'] == sid and r['model'] == 'historical_mean') <= min(r['mae'] for r in station_combined if r['station_id'] == sid) + 1e-10 for sid in station_ids)
    station_insight = f'按借还综合 MAE 比较，本次 {baseline_best}/{len(station_ids)} 个站点的历史同期均值不差于两种学习模型。这一结论针对综合误差，单个目标仍需分别检查。'
    v_net_hist = get('station', 'net_outflow', phase='validation')['mae']
    v_net_rf = get('station', 'net_outflow', 'random_forest', 'validation')['mae']
    comparison_notes = []
    for scope, target in [('city', 'city_demand'), ('station', 'outflow'), ('station', 'inflow'), ('station', 'net_outflow')]:
        r0 = get(scope, target)
        comparison_notes.append(f"{REPORT_TARGET_NAMES[target]}：最终测试历史均值 MAE {report_number(r0['mae'], 2 if scope == 'city' else 3)}，高峰 MAE {report_number(r0['peak_mae'], 2 if scope == 'city' else 3)}。")
    phase_table = report_table(['阶段', '可用历史（2024 年）', '周初一次预测', '用途'], [
        ['验证', '1 月 1–14 日', '1 月 15–21 日，168 小时', '选择参数和推荐方法'],
        ['最终测试', '1 月 1–21 日', '1 月 22–28 日，168 小时', '检验已冻结方案'],
    ])
    metric_intro = report_table(['指标', '读法'], [
        ['MAE', '平均每小时相差多少次，越小越好。站点总体取 10 站 MAE 的平均。'],
        ['RMSE', '对大误差更敏感，越小越好。站点总体是合并所有站点小时误差后计算，并非逐站 RMSE 的平均。'],
        ['R²', '与目标序列均值参照比较，可能为负；不是准确率。站点保留逐站结果，不平均 R²。'],
        ['MAE 改善率', '（历史均值 MAE－模型 MAE）÷历史均值 MAE ×100%。正值更好、负值更差；分母为零时无定义。'],
    ])
    image_html = {}
    for key, filename, alt in [('models', 'fig17_model_comparison.png', '四类预测对象中，Ridge 和随机森林相对历史同期均值的最终测试 MAE 改善率'), ('stations', 'fig18_station_error_comparison.png', '冻结 Top 10 站点的最终测试借还综合 MAE 对比')]:
        encoded = base64.b64encode((target_reports / 'figures' / filename).read_bytes()).decode('ascii')
        image_html[key] = f'<a class="figure-link" href="data:image/png;base64,{encoded}" download="{filename}" title="保存完整图片"><img src="data:image/png;base64,{encoded}" alt="{alt}" width="1600" loading="eager"></a>'
    download_links = []
    for filename, label in [('model_comparison_summary.csv', '总体指标'), ('model_comparison_relative_mae.csv', 'MAE 改善率'), ('model_comparison_daily_bias.csv', '日周偏差'), ('model_comparison_station.csv', '逐站指标')]:
        encoded = base64.b64encode((target_reports / 'tables' / filename).read_bytes()).decode('ascii')
        download_links.append(f'<a class="download" download="{filename}" href="data:text/csv;charset=utf-8;base64,{encoded}">下载{label} CSV <span aria-hidden="true">↗</span></a>')
    js_data = json.dumps({'daily': data['daily'], 'station': data['station']}, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    html = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双层预测：模型对比与误差分析</title>
<style>
:root{--ink:#18343f;--muted:#536b73;--paper:#f4f5ef;--card:#fff;--teal:#166e6a;--line:#dce4dd;--soft:#edf3ee;--warm:#a65132}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;font-size:16px;line-height:1.75}main{max-width:1160px;margin:auto;padding:40px 28px 70px}header{padding:14px 0 30px;border-bottom:2px solid var(--ink)}.eyebrow{font-size:13px;letter-spacing:.12em;color:var(--teal);font-weight:700}h1{font-size:38px;line-height:1.35;margin:14px 0 16px;letter-spacing:-.02em}h2{font-size:25px;line-height:1.4;margin:0 0 20px}h3{font-size:18px;margin:0 0 8px}p{margin:10px 0 14px}.lead{max-width:850px;color:var(--muted);font-size:17px}nav{display:flex;gap:10px 24px;flex-wrap:wrap;margin-top:22px}a{color:var(--teal);text-underline-offset:4px}nav a{font-size:14px;text-decoration:none}section{padding-top:42px}.section-no{font-size:13px;color:var(--teal);font-weight:700;margin-bottom:6px}.methods,.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.method,.stat{padding:22px;background:var(--card);border:1px solid var(--line);border-radius:12px}.method p{font-size:14px;color:var(--muted);margin-bottom:0}.tag{display:inline-block;color:var(--teal);background:var(--soft);border-radius:20px;padding:2px 10px;font-size:12px;font-weight:700}.muted,.caption{color:var(--muted);font-size:14px}.stat .value{font-size:34px;line-height:1.4;font-weight:700;letter-spacing:-.02em;margin:8px 0 0}.stat .unit{font-size:12px;color:var(--muted)}.stat .label{font-size:14px;color:var(--muted)}.note{border-left:4px solid var(--teal);background:var(--soft);padding:16px 20px;margin:22px 0}.note.warn{border-left-color:var(--warm);background:#f7eee7}.note p:last-child{margin-bottom:0}.table-scroll{max-width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:#fff;margin:16px 0 20px}table{border-collapse:collapse;width:100%;font-size:14px;text-align:left;font-variant-numeric:tabular-nums}th,td{padding:11px 14px;border-bottom:1px solid var(--line);vertical-align:top}th{font-weight:700;background:var(--soft);white-space:nowrap}td{white-space:nowrap}td:first-child{white-space:normal;min-width:90px}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#f7faf8}.table-scroll:focus-visible{outline:2px solid var(--teal);outline-offset:3px}.figure{margin:22px 0 30px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px}.figure img{display:block;width:100%;height:auto}.figure-link{display:block}.figure figcaption{font-size:13px;color:var(--muted);padding:10px 10px 2px}.controls{display:flex;gap:18px;align-items:end;flex-wrap:wrap;margin:18px 0}.controls label{font-size:14px;display:grid;gap:5px}select{font:inherit;font-size:14px;background:white;color:var(--ink);padding:9px 35px 9px 12px;border:1px solid #aabdb5;border-radius:7px;max-width:100%}.control-status{font-size:14px;font-variant-numeric:tabular-nums;color:var(--muted)}details{border:1px solid var(--line);background:white;border-radius:10px;margin:20px 0;padding:0 18px}summary{padding:16px 0;cursor:pointer;font-size:15px;font-weight:700}details[open]{padding-bottom:10px}.downloads{display:flex;flex-wrap:wrap;gap:10px}.download{font-size:13px;text-decoration:none;background:#fff;border:1px solid var(--line);border-radius:7px;padding:9px 14px}.download:hover{border-color:var(--teal)}.conclusion{display:grid;grid-template-columns:1fr 1fr;gap:18px}.conclusion article{padding:20px;background:white;border:1px solid var(--line);border-radius:10px}.conclusion p{font-size:14px}.footer-links{border-top:1px solid var(--line);margin-top:36px;padding-top:22px;font-size:14px;display:flex;gap:24px;flex-wrap:wrap}button{font:inherit}a:focus-visible,summary:focus-visible,select:focus-visible{outline:3px solid #e2a957;outline-offset:4px}@media(max-width:720px){main{padding:24px 16px 40px}h1{font-size:29px}h2{font-size:22px}.methods,.stats,.conclusion{grid-template-columns:1fr}.method,.stat{padding:18px}.stat .value{font-size:30px}.figure{padding:6px}.figure figcaption{padding:9px 6px}th,td{padding:9px 11px}.lead{font-size:16px}section{padding-top:32px}}@media print{body{background:white}main{max-width:none;padding:0}nav,.downloads,.controls{display:none}details{break-inside:avoid}.figure{break-inside:avoid}a{text-decoration:none}.table-scroll{overflow:visible}h2{break-after:avoid}}
#methods td{white-space:normal}
</style></head><body><main>
<header><div class="eyebrow">陈宇涵 · 双层需求预测 · 第三步</div><h1>模型对比与误差分析</h1><p class="lead">把全城与核心站点的预测放回同一评价框架：比较方法的实际表现，检查误差集中在哪里，并明确这些结果可以支持的结论。</p><nav aria-label="报告目录"><a href="#methods">口径与指标</a><a href="#results">总体对比</a><a href="#errors">误差分析</a><a href="#stations">逐站差异</a><a href="#conclusions">结论与局限</a></nav></header>
<section id="methods"><div class="section-no">01 / 比较口径</div><h2>先理解方法，再看结果</h2><div class="methods"><article class="method"><span class="tag">简单对照</span><h3>历史同期均值</h3><p>用历史相同星期、相同小时的需求平均值预测。站点按各自历史计算。透明、易复核，但面对周总规模变化时可能反应不足。</p></article><article class="method"><span class="tag">线性学习</span><h3>Ridge 岭回归</h3><p>学习时间、上周同小时需求与预测量之间的线性关系，并约束系数大小。复杂度可控，但依赖特征表达，短历史仍会限制表现。</p></article><article class="method"><span class="tag">非线性学习</span><h3>随机森林</h3><p>让多棵受限决策树分别预测，再取平均。可以表达非线性关系和交互，但少量历史周仍难保证对需求突变的预测能力。</p></article></div><p class="caption">站点借出、归还分别使用 10 站共享模型；净流出由借出预测减归还预测得到，允许为负。全城计数与站点可归属事件的覆盖范围不同。</p>
@@METRIC_INTRO@@
<p class="caption">全城 MAE 的单位为次/小时；站点 MAE 的单位为次/站·小时。站点 RMSE 合并所有站点小时误差计算。两层原始误差分开解读，不能按数值大小横向判优。</p>
@@PHASE_TABLE@@
<p class="caption">每周一 00:00 一次预测随后 168 小时，周内不使用新真实需求更新输入。Top 10 只按前两周筛选并冻结。1 月 29–31 日不参与本次选型与评价。</p></section>
<section id="results"><div class="section-no">02 / 最终测试</div><h2>历史同期均值仍是本次推荐</h2><p>推荐依据验证周确定，最终测试只检验。下面三项数值均对应历史同期均值。</p><div class="stats"><article class="stat"><div class="label">全城 MAE</div><div class="value">@@CITY_MAE@@</div><div class="unit">次/小时</div></article><article class="stat"><div class="label">10 站借还综合 MAE</div><div class="value">@@COMBINED_MAE@@</div><div class="unit">次/站·小时；借出、归还的逐站平均 MAE 再取均值</div></article><article class="stat"><div class="label">10 站净流出 MAE</div><div class="value">@@NET_MAE@@</div><div class="unit">次/站·小时；单独评价，不参与统一选型</div></article></div>
<figure class="figure">@@FIG_MODELS@@<figcaption>图 17｜相对历史同期均值的最终测试 MAE 改善率。零线是基准，正值更好、负值更差；各面板分别比较，不构成跨层级的难度或精度排名。点击图片可保存原图。</figcaption></figure>
<h3>全城需求</h3>@@CITY_METRICS@@<h3>10 站总体表现</h3>@@STATION_METRICS@@
<p class="caption">站点表中的 MAE 和高峰 MAE 为逐站平均，RMSE 为合并误差计算。总体不汇总 R²，逐站 R² 见下方明细。MAE 改善率仅描述相对表现，不重新选择方法。</p>
<details id="validation-details"><summary>查看验证周结果与原有推荐依据</summary><p>全城按验证 MAE 推荐方法；站点在三种展示方法中，按借出、归还的逐站平均 MAE 再取均值统一推荐。历史均值的站点验证综合 MAE 为 <strong>@@VALIDATION_COMBINED@@</strong>。</p>@@VALIDATION_CITY@@@@VALIDATION_STATION@@<div class="note"><p>验证周随机森林的净流出 MAE 为 @@VALIDATION_NET_RF@@，低于历史均值的 @@VALIDATION_NET_HIST@@。净流出并非统一推荐的选择指标，不能据此替换借还综合推荐；借还误差也可能在相减时抵消。</p></div></details></section>
<section id="errors"><div class="section-no">03 / 误差分析</div><h2>排名较好，仍有明显的总量低估</h2><p>小时误差与整周规模需要分别判断。下表为最终测试周、历史同期均值的总量表现；站点总量是固定 10 站的加总。</p>@@WEEKLY_TABLE@@
<div class="note warn"><p>全城总量低估 <strong>@@CITY_UNDER@@%</strong>，站点借出和归还分别低估 <strong>@@OUT_UNDER@@%</strong>、<strong>@@IN_UNDER@@%</strong>。净流出实际总量为 @@NET_ACTUAL@@ 次、预测为 @@NET_PRED@@ 次，周差值接近并不代表每小时准确。</p></div><p>@@DAILY_INSIGHT@@</p>
<h3>高峰与逐日偏差</h3><p>高峰时段沿用每次预测前历史中确定的时段。上方指标表显示，高峰 MAE 与全部小时 MAE 应结合判断；不根据测试周重新挑选高峰。</p>@@DIAGNOSTICS@@
<p class="caption">上表按每日总量偏差的绝对值找出最大的一天，只描述总量偏差，不等于该日小时 MAE 最大。波动原因需要额外数据验证。</p>
<div class="controls"><label for="daily-target">查看对象<select id="daily-target"><option value="city_demand">全城需求</option><option value="outflow">10 站借出</option><option value="inflow">10 站归还</option><option value="net_outflow">10 站净流出</option></select></label><label for="daily-model">查看方法<select id="daily-model"><option value="historical_mean">历史同期均值</option><option value="ridge">Ridge 岭回归</option><option value="random_forest">随机森林</option></select></label><span class="control-status">最终测试周 · 1 月 22–28 日</span></div><div id="daily-table" aria-live="polite"></div>
<p class="caption">偏差＝预测－实际。净流出可能为负或接近零，只报告次数偏差；其余目标偏差率的分母是实际总量，分母为零时无定义。</p></section>
<section id="stations"><div class="section-no">04 / 核心站点</div><h2>统一推荐之外，还要检查逐站差异</h2><p>图 18 采用各站借出、归还 MAE 的平均值。站点顺序沿用训练期潮汐排名，不按测试表现重排。</p><figure class="figure">@@FIG_STATIONS@@<figcaption>图 18｜10 站最终测试借还综合 MAE。数值越低越好，站点名称及各目标明细见下表。点击图片可保存原图。</figcaption></figure><p>@@STATION_INSIGHT@@</p>
<div class="controls"><label for="station-target">逐站比较对象<select id="station-target"><option value="combined">借还综合 MAE</option><option value="outflow">借出 MAE</option><option value="inflow">归还 MAE</option><option value="net_outflow">净流出 MAE</option></select></label></div><div id="station-table" aria-live="polite"></div><p class="caption">综合 MAE＝（本站借出 MAE＋归还 MAE）÷2。单站某个目标表现更好，不改变验证时确定的统一推荐。</p>
<details id="station-details"><summary>查看逐站 RMSE、R² 与高峰 MAE</summary><div class="controls"><label for="station-detail-target">预测目标<select id="station-detail-target"><option value="outflow">借出</option><option value="inflow">归还</option><option value="net_outflow">净流出</option></select></label></div><div id="station-detail-table"></div><p class="caption">这里的 RMSE、R² 和高峰 MAE 均针对单个站点。R² 无定义时显示“—”，不填成零。</p></details>
<details id="feature-details"><summary>补充：随机森林如何使用特征</summary><p>下表汇总已有随机森林内部重要性，分别在三个模型内部合计为 100%。相关特征可能分摊权重；站点编号只是类别信息，不是连续数值。</p>@@FEATURE_TABLE@@<p class="caption">这些权重只解释随机森林，不能当作历史均值推荐方法的解释，也不代表因果关系或有效性证明。</p></details></section>
<section id="conclusions"><div class="section-no">05 / 结论</div><h2>本次支持什么结论</h2><div class="conclusion"><article><h3>方法选择</h3><p>历史同期均值在验证中被选中，在最终测试整体比较中也优于本次 Ridge 和随机森林。学习模型在当前设置下没有获得整体改善证据；这不代表它们在其他数据与设置下都无效。</p></article><article><h3>实际效果</h3><p>推荐方法仍明显低估测试周总量，且高峰误差更大。现有结果可用于展示历史规律和预测方法比较，尚不足以据此承诺未来周的稳定预测精度。</p></article><article><h3>证据范围</h3><p>数据只有一个月、一次验证周和一次最终测试周；10 站共享训练增加了站点样本，没有增加独立历史周数。结论限于本次历史窗口和冻结站点。</p></article><article><h3>业务含义</h3><p>预测对象是清洗后已观测订单与可归属站点事件。净流出是借还流量差；没有库存、容量、未满足需求和外部因素数据，不能将其直接解释为缺车数量或确定需求波动原因。</p></article></div>
<details><summary>后续改进的优先方向</summary><p>优先积累更多连续周数据，在多个预测起点检验稳定性；再考虑加入预测时已经可获得的天气、节假日、活动等信息，检验是否减少周规模与高峰偏差。新增方法应在新的验证安排中评价。</p></details>
<p class="caption">完整 CSV 保留四种方法；上周同期仅作为补充对照。页面集中展示历史同期均值、Ridge 与随机森林。</p><div class="downloads">@@DOWNLOADS@@</div><div class="footer-links"><a href="demand_prediction.html">查看第一步：全城预测 ↗</a><a href="station_prediction.html">查看第二步：核心站点预测 ↗</a></div></section>
</main><script id="report-data" type="application/json">@@JS_DATA@@</script><script>
const data=JSON.parse(document.getElementById('report-data').textContent);
const models=['historical_mean','ridge','random_forest'];
const names={historical_mean:'历史同期均值',ridge:'Ridge 岭回归',random_forest:'随机森林'};
const num=(v,d=2,s=false)=>v===null||!Number.isFinite(Number(v))?'—':(s&&v>0?'+':'')+Number(v).toLocaleString('zh-CN',{minimumFractionDigits:d,maximumFractionDigits:d});
const safe=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function table(headers,rows){return '<div class="table-scroll" tabindex="0"><table><thead><tr>'+headers.map(h=>'<th scope="col">'+safe(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+safe(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function daily(){const target=document.getElementById('daily-target').value,model=document.getElementById('daily-model').value;const rows=data.daily.filter(r=>r.phase==='test'&&r.target===target&&r.model===model&&r.period==='day');rows.sort((a,b)=>a.date.localeCompare(b.date));document.getElementById('daily-table').innerHTML=table(['日期','实际次数','预测次数','偏差次数','偏差率'],rows.map(r=>[r.date,num(r.actual_total,0),num(r.predicted_total,1),num(r.bias,1,true),target==='net_outflow'?'不使用百分比':r.bias_pct===null?'无定义':num(r.bias_pct,1,true)+'%']));}
function station(){const target=document.getElementById('station-target').value;const records=data.station.filter(r=>r.phase==='test'&&r.target===target&&models.includes(r.model));const unique=[...new Set(records.map(r=>r.station_id))];unique.sort((a,b)=>records.find(r=>r.station_id===a).selection_rank-records.find(r=>r.station_id===b).selection_rank);const rows=unique.map(id=>{const r=records.find(r=>r.station_id===id);return [String(r.selection_rank).padStart(2,'0')+' · '+r.station_name,...models.map(m=>num(records.find(x=>x.station_id===id&&x.model===m).mae,3))];});document.getElementById('station-table').innerHTML=table(['站点（训练期排名）','历史均值 MAE','Ridge MAE','随机森林 MAE'],rows);}
function detail(){const target=document.getElementById('station-detail-target').value;const records=data.station.filter(r=>r.phase==='test'&&r.target===target&&models.includes(r.model));records.sort((a,b)=>a.selection_rank-b.selection_rank||models.indexOf(a.model)-models.indexOf(b.model));document.getElementById('station-detail-table').innerHTML=table(['站点','方法','RMSE','R²','高峰 MAE'],records.map(r=>[r.station_name,names[r.model],num(r.rmse,3),num(r.r2,3),num(r.peak_mae,3)]));}
document.getElementById('daily-target').addEventListener('change',daily);document.getElementById('daily-model').addEventListener('change',daily);document.getElementById('station-target').addEventListener('change',station);document.getElementById('station-detail-target').addEventListener('change',detail);daily();station();detail();
</script></body></html>'''
    replacements = {
        'METRIC_INTRO': metric_intro, 'PHASE_TABLE': phase_table,
        'CITY_MAE': report_number(city['mae'], 2), 'COMBINED_MAE': report_number(combined_mae, 3), 'NET_MAE': report_number(net['mae'], 3),
        'FIG_MODELS': image_html['models'], 'FIG_STATIONS': image_html['stations'],
        'CITY_METRICS': report_metric_table(data, 'city', 'test'), 'STATION_METRICS': report_metric_table(data, 'station', 'test'),
        'VALIDATION_COMBINED': report_number(validation_combined, 3), 'VALIDATION_NET_RF': report_number(v_net_rf, 3), 'VALIDATION_NET_HIST': report_number(v_net_hist, 3),
        'VALIDATION_CITY': report_metric_table(data, 'city', 'validation'), 'VALIDATION_STATION': report_metric_table(data, 'station', 'validation'),
        'WEEKLY_TABLE': weekly_table, 'CITY_UNDER': report_number(-city['bias_pct'], 1), 'OUT_UNDER': report_number(-outflow['bias_pct'], 1), 'IN_UNDER': report_number(-inflow['bias_pct'], 1),
        'NET_ACTUAL': report_number(net['total_actual'], 0), 'NET_PRED': report_number(net['total_predicted'], 0), 'DIAGNOSTICS': diag_table,
        'DAILY_INSIGHT': escape(daily_insight),
        'STATION_INSIGHT': escape(station_insight),
        'FEATURE_TABLE': report_table(['模型', '信息类型', '内部重要性'], feature_rows), 'DOWNLOADS': ''.join(download_links), 'JS_DATA': js_data,
    }
    for key, value in replacements.items():
        html = html.replace('@@' + key + '@@', value)
    if '@@' in html:
        raise ValueError('报告模板存在未替换字段')
    (target_reports / 'model_comparison.html').write_text(html, encoding='utf-8')
    write_comparison_markdown(data, target_reports, city, outflow, inflow, net, diagnostics, feature_rows, daily_insight, station_insight)


def md_table(headers, rows):
    clean = lambda s: str(s).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(clean, headers)) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + ['| ' + ' | '.join(map(clean, row)) + ' |' for row in rows])


def write_comparison_markdown(data, target_reports, city, outflow, inflow, net, diagnostics, feature_rows, daily_insight, station_insight):
    combined = [r for r in data['station'] if r['phase'] == 'test' and r['target'] == 'combined' and r['model'] in REPORT_MODELS]
    combined_baseline = sorted([r for r in combined if r['model'] == 'historical_mean'], key=lambda r: r['selection_rank'])
    station_rows = [[r['selection_rank'], r['station_id'], r['station_name'], *[report_number(next(x['mae'] for x in combined if x['station_id'] == r['station_id'] and x['model'] == model), 3) for model in REPORT_MODELS]] for r in combined_baseline]
    sections = ['# 6.3 模型比较、误差分析与结论汇总\n',
        '本节汇总全城与核心站点预测，沿用各自的预测、参数、冻结 Top 10 和推荐记录，不在测试结果上重新选型。',
        '### 6.3.1 方法、时间与评价口径\n',
        '历史同期均值按相同星期和小时平均历史需求；Ridge 用系数约束控制线性模型复杂度；随机森林对多棵受限决策树的预测取平均。站点学习模型在 10 站间共享，借出与归还分别拟合，再相减得到净流出。方法的优点与适用理由见前两步说明，实际有效性以验证与测试为依据。',
        '2024 年 1 月 1–14 日预测 15–21 日作验证；1–21 日预测 22–28 日作最终测试。每周一 00:00 一次预测 168 小时，周内不更新真实需求。1 月 29–31 日不参与本轮评价，不生成跨月预测。',
        'MAE 为平均每小时绝对误差，RMSE 对大误差更敏感，R² 比较平方误差与目标序列均值参照，不是准确率。站点总体 MAE 是逐站 MAE 平均，RMSE 是所有站点小时误差合并计算；不把合并 RMSE 写成逐站 RMSE 平均，也不平均逐站 R²。全城与站点原始误差因规模、覆盖和目标不同而分开解读。',
        '**MAE 改善率＝（历史均值 MAE－模型 MAE）÷历史均值 MAE ×100%。** 正值改善、负值退步；分母为零时无定义。该比例是描述性比较，不重新选型。',
        '### 6.3.2 总体比较\n',
        '![图17：最终测试模型相对表现](../figures/fig17_model_comparison.png)\n',
        '图 17 各面板分别以本目标的历史均值为零基准，不用于跨层级判优。以下完整指标保留上周同期作为补充对照；HTML 只展示另外三种方法。站点总体 R² 留空，逐站值见 CSV。',
    ]
    for phase, label in [('test', '最终测试周'), ('validation', '验证周')]:
        rows = []
        for r in data['relative']:
            if r['phase'] != phase:
                continue
            rows.append([REPORT_TARGET_NAMES[r['target']], REPORT_MODEL_NAMES[r['model']], report_number(r['mae'], 3), report_number(r['rmse'], 3), report_number(r['peak_mae'], 3), report_number(r['r2'], 3), report_number(r['mae_improvement_pct'], 1, True) + '%' if r['mae_improvement_pct'] is not None else '无定义'])
        sections.extend(['### ' + label + '\n', md_table(['对象', '方法', 'MAE', 'RMSE', '高峰MAE', 'R²', 'MAE改善率'], rows)])
    sections.extend([
        '### 6.3.3 推荐依据与有效性\n',
        '两层验证均选中历史同期均值。全城按验证 MAE 选方法；站点在历史均值、Ridge、随机森林中，以借出与归还逐站平均 MAE 再取平均统一推荐。测试后的结果用于检验，不替换推荐。',
        f"最终测试历史均值全城 MAE 为 {report_number(city['mae'], 2)} 次/小时；站点借还综合 MAE 为 {report_number((outflow['mae'] + inflow['mae']) / 2, 4)} 次/站·小时，净流出 MAE 为 {report_number(net['mae'], 4)}。学习模型在本次整体比较中没有超过历史均值。",
        '验证周随机森林的净流出 MAE 虽较低，净流出并不是统一选型指标。借还预测相减可能抵消误差，因此它不能单独证明借还预测准确，也不应据此更改统一推荐。',
        '### 6.3.4 总量、高峰与逐日偏差\n',
        md_table(['最终测试对象', '实际总量', '预测总量', '预测－实际', '偏差率'], [[REPORT_TARGET_NAMES[r['target']], report_number(r['total_actual'], 0), report_number(r['total_predicted'], 1), report_number(r['total_bias'], 1, True), '不使用百分比' if r['target'] == 'net_outflow' else report_number(r['bias_pct'], 1, True) + '%'] for r in (city, outflow, inflow, net)]),
        f"历史均值测试周全城低估 {-city['bias_pct']:.1f}%，10 站借出低估 {-outflow['bias_pct']:.1f}%、归还低估 {-inflow['bias_pct']:.1f}%。推荐方法仍有明显规模偏差，不能仅因对照排名较好就称为高精度。",
        daily_insight,
        '高峰掩码直接使用原预测文件的 is_peak；原口径为预测前训练期工作日/周末每小时均值最高两个小时，净流出按绝对强度。逐站及总体高峰 MAE 沿用此定义，未根据目标周重新挑选。',
        md_table(['对象', '每日绝对总量偏差最大日期', '实际次数', '预测次数', '偏差次数'], diagnostics),
        '上表按每日加总后的偏差绝对值排序，仅描述每日总量问题，不等于小时误差最大的一天。完整日周数据见 model_comparison_daily_bias.csv；净流出可能为负或接近零，偏差率统一不定义。',
        '### 6.3.5 逐站差异与特征解释\n',
        '![图18：冻结Top10的借还综合MAE](../figures/fig18_station_error_comparison.png)\n',
        '图 18 中各站综合 MAE＝（借出 MAE＋归还 MAE）÷2，按训练期潮汐排名显示。单站单目标的最优方法可能不同，但不改变验证时的统一推荐。逐站 MAE、RMSE、R²、高峰 MAE、总量偏差，以及综合 MAE 见 model_comparison_station.csv。综合目标没有约定的 RMSE 或 R²，留空而不混算。',
        station_insight,
        md_table(['排名', '站点编号', '站点名称', '历史均值综合MAE', 'Ridge综合MAE', '随机森林综合MAE'], station_rows),
        md_table(['随机森林模型', '信息类型', '内部重要性'], feature_rows),
        '重要性为已有随机森林内部权重，相关或类别特征可能分摊权重。它不解释历史均值推荐方法，不代表因果，不构成有效性证明。',
        '### 6.3.6 结论与适用范围\n',
        '历史同期均值是本次验证选中的统一方法，最终测试的整体比较也支持优先采用这一简单对照。当前 Ridge 和随机森林设置没有带来整体改善证据；这不等于模型在其他数据和设置下均无效。',
        '本次只有一个月、一次验证周和一次测试周。共享站点模型增加站点行数，没有增加独立历史周数。结果限于冻结 Top 10 和可归属事件，预测的是清洗后的已观测订单而非全部潜在需求。净流出不能直接解释为库存变化、缺车数量或未满足需求。',
        '没有天气、节假日活动、库存、容量等解释变量，不能确定周规模波动的原因。后续宜先积累更多连续周，在多个预测起点评价稳定性，再检验预测时可获得的外部信息；新增方法应另用验证安排评价。',
        '### 6.3.7 运行与核验记录（不进入 HTML）\n',
        '独立入口：`python -m src.model_comparison`。可用 `--source-reports` 指向包含原有 tables 的报告目录，`--output-dir` 指定第三步输出报告目录。只读取已有产物，不导入或调用模型训练入口。',
        '新增四份 CSV、两张 PNG、HTML、本 MD，以及 model_comparison_checks.json。图像为 300 DPI，HTML 内嵌图片与 CSV，可单独离线使用；前两步详情跳转需相邻保留对应 HTML。运行同一输入会更新第三步同名产物。',
        '源文件指纹、逐项一致性检查与元信息保存在 tables/model_comparison_checks.json。工程检查记录另附交付说明。',
        '前两步详情：[全城说明](03_city_demand_prediction.md)、[站点说明](04_station_demand_prediction.md)。',
    ])
    dest = Path(target_reports) / 'sections'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / '05_model_comparison.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='只读取已有预测，生成双层需求预测综合比较报告。')
    default_reports = Path(__file__).resolve().parents[1] / 'reports'
    parser.add_argument('--source-reports', type=Path, default=default_reports)
    parser.add_argument('--output-dir', type=Path, default=default_reports)
    args = parser.parse_args()
    data = build_analysis(args.source_reports.resolve(), args.output_dir.resolve())
    render_comparison_report(data, args.output_dir.resolve())
    print(json.dumps({'html': str(args.output_dir.resolve() / 'model_comparison.html'), 'summary_rows': len(data['summary']), 'checks': len(data['checks']), 'checks_passed': all(c['passed'] for c in data['checks'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
