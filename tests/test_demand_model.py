"""检查预测契约：时间边界、手算基线、未来标签不影响选型/预测。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.demand_model import (
    Fitted, MODEL_ORDER, build_hourly, features, fit_model,
    run_experiment, scores, narrative, json_safe, aggregate_totals,
    write_markdown, write_html,
)


def synthetic_series():
    index = pd.date_range("2024-01-01", "2024-02-01", inclusive="left", freq="h")
    values = 20 + index.hour.to_numpy() * 3 + index.dayofweek.to_numpy() * 2 + np.arange(len(index)) // 168
    return pd.Series(values.astype(float), index=index, name="ride_count")


class WeeklyPredictionTests(unittest.TestCase):
    def test_hourly_counts_and_missing_dates(self):
        dates = pd.date_range("2024-01-01 08:05", periods=31, freq="D")
        data = pd.DataFrame({"ride_id": [str(i) for i in range(31)], "started_at": dates,
                             "hour": dates.hour, "weekday": dates.dayofweek,
                             "is_weekend": dates.dayofweek >= 5})
        hourly = build_hourly(data)
        self.assertEqual((len(hourly), int(hourly.sum()), int(hourly.loc['2024-01-01 08:00'])), (744, 31, 1))
        self.assertEqual(hourly.loc['2024-01-01 09:00'], 0)
        with self.assertRaisesRegex(ValueError, "整天没有记录"):
            build_hourly(data.iloc[1:])
        duplicate = pd.concat([data, data.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "重复"):
            build_hourly(duplicate)

    def test_week_lag_hand_calculation_and_full_horizon_boundary(self):
        y = synthetic_series()
        origin = pd.Timestamp("2024-01-15")
        index = pd.date_range(origin, periods=168, freq="h")
        history = y.loc[y.index < origin]
        x = features(history, index, origin)
        self.assertEqual(x.shape, (168, 16))
        self.assertEqual(x.iloc[0].lag_168h, y.loc['2024-01-08 00:00'])
        self.assertEqual(x.iloc[-1].lag_168h, y.loc['2024-01-14 23:00'])
        self.assertFalse(x.isna().any().any())
        with self.assertRaisesRegex(ValueError, "周初可知范围"):
            features(history, pd.date_range(origin, periods=169, freq="h"), origin)
        with self.assertRaisesRegex(ValueError, "预测起点之后"):
            features(y, index, origin)

    def test_both_baselines_against_hand_calculation(self):
        y = synthetic_series()
        origin = pd.Timestamp("2024-01-15")
        history = y.loc[y.index < origin]
        index = pd.DatetimeIndex([pd.Timestamp("2024-01-16 08:00")])
        _, average = fit_model("historical_mean", history, {}).predict(index, origin)
        _, seasonal = fit_model("last_week", history, {}).predict(index, origin)
        expected = (y.loc['2024-01-02 08:00'] + y.loc['2024-01-09 08:00']) / 2
        self.assertEqual(average[0], expected)
        self.assertEqual(seasonal[0], y.loc['2024-01-09 08:00'])

    def test_training_row_loss_and_scaler_are_training_only(self):
        y = synthetic_series()
        history = y.loc[y.index < "2024-01-15"]
        fitted = fit_model("ridge", history, {"alpha": 1.0})
        self.assertEqual(fitted.fit_rows, 168)
        expected_mean = features(history, history.index).dropna().mean().to_numpy()
        np.testing.assert_allclose(fitted.estimator.named_steps['standardscaler'].mean_, expected_mean)
        self.assertEqual(fit_model("ridge", y.loc[y.index < '2024-01-22'], {"alpha": 1.0}).fit_rows, 336)

    def test_missing_hour_in_history_is_rejected(self):
        y = synthetic_series().loc[:'2024-01-14 23:00'].drop(pd.Timestamp('2024-01-02 03:00'))
        with self.assertRaisesRegex(ValueError, "网格不连续"):
            fit_model("ridge", y, {"alpha": 1.0})

    def test_nonnegative_rule_preserves_raw_predictions(self):
        class NegativeEstimator:
            def predict(self, x):
                return np.full(len(x), -3.25)
        y = synthetic_series().loc[:'2024-01-14 23:00']
        fitted = Fitted("ridge", y, NegativeEstimator(), 168, [])
        raw, clipped = fitted.predict(pd.date_range('2024-01-15', periods=168, freq='h'), pd.Timestamp('2024-01-15'))
        self.assertTrue((raw == -3.25).all())
        self.assertTrue((clipped == 0).all())

    def test_metric_arithmetic_and_constant_target(self):
        metric = scores(np.array([1., 3.]), np.array([2., 1.]), np.array([True, False]))
        self.assertEqual(metric['mae'], 1.5)
        self.assertAlmostEqual(metric['rmse'], np.sqrt(2.5))
        self.assertEqual(metric['r2'], -1.5)
        self.assertEqual(metric['peak_mae'], 1.)
        self.assertEqual(metric['total_bias'], -1.)
        self.assertIsNone(scores(np.ones(2), np.zeros(2), np.ones(2, dtype=bool))['r2'])

    def test_future_test_labels_cannot_change_selection_or_predictions(self):
        grids = {"historical_mean": [{}], "last_week": [{}], "ridge": [{"alpha": 1.}],
                 "random_forest": [{"n_estimators": 5, "max_depth": 3, "min_samples_leaf": 8,
                                    "random_state": 42, "n_jobs": 1}]}
        y = synthetic_series()
        changed = y.copy()
        changed.loc[changed.index >= '2024-01-22'] += 100000
        original = run_experiment(y, grids)
        perturbed = run_experiment(changed, grids)
        self.assertEqual(original.selection, perturbed.selection)
        pd.testing.assert_frame_equal(original.candidates, perturbed.candidates)
        np.testing.assert_array_equal(original.predictions.prediction, perturbed.predictions.prediction)
        self.assertFalse(original.predictions.actual.equals(perturbed.predictions.actual))
        self.assertEqual(set(original.predictions.model), set(MODEL_ORDER))
        self.assertTrue(original.audit.passed.all())

    def test_perfect_baselines_and_undefined_r2_render_and_serialize(self):
        y = synthetic_series() * 0 + 12.
        grids = {"historical_mean": [{}], "last_week": [{}], "ridge": [{"alpha": 1.}],
                 "random_forest": [{"n_estimators": 2, "max_depth": 3, "min_samples_leaf": 8,
                                    "random_state": 42, "n_jobs": 1}]}
        experiment = run_experiment(y, grids)
        result = narrative(experiment)
        self.assertIsNone(result['improvements']['historical_mean'])
        self.assertIsNone(result['improvements']['last_week'])
        encoded = json.dumps(json_safe(experiment.metrics.to_dict(orient='records')), allow_nan=False)
        self.assertIn('null', encoded)
        metadata = {'input_rows': 744, 'input_sha256': 'input-test', 'code_sha256': 'code-test',
                    'versions': {'python': 'test', 'numpy': 'test', 'pandas': 'test', 'scikit_learn': 'test'}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            markdown = root / 'methods.md'
            figure = root / 'placeholder.png'
            figure.write_bytes(b'unit-test-placeholder')
            totals = aggregate_totals(experiment.predictions)
            write_markdown(experiment, totals, metadata, markdown)
            write_html(experiment, totals, metadata, [figure, figure], [markdown], root/'report.html')
            self.assertIn('无定义', markdown.read_text(encoding='utf-8'))
            self.assertIn('无定义', (root/'report.html').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
