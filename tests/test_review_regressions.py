"""本地代码审查发现的调度边界与空间数据接口回归测试。"""

from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.imbalance_dispatch import build_dispatch_plan, build_hourly_panel
from src import imbalance_dispatch
from src.spatial_analysis import attach_coordinates
from src.utils import PROJECT_ROOT
from src.zone_dispatch import (
    SEGMENTS, assign_nearest_zone, build_quota, build_zones, dockless_hotspots,
    load_core_predictions, zone_hotspot_balance,
)
from tests.test_spatial_dispatch import make_station_trips
from tests.test_zone_dispatch import make_quota_inputs


class StationDispatchRegressionTests(unittest.TestCase):
    def inputs(self, nets):
        ids = [f"S{i}" for i in range(len(nets))]
        peak = pd.DataFrame({
            "period": "早高峰", "station_id": ids,
            "station_name": ids, "net_flow": nets,
        })
        summary = pd.DataFrame({
            "station_id": ids, "station_name": ids,
            "lat": 38.90, "lng": -77.04,
        })
        return peak, summary

    def test_one_bike_is_matched(self):
        peak, summary = self.inputs([-1, 1])
        plan = build_dispatch_plan(peak, summary, days=1)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan.iloc[0].volume_per_day, 1)

    def test_empty_plan_retains_exportable_columns(self):
        peak, summary = self.inputs([2, 2])
        plan = build_dispatch_plan(peak, summary, days=1)
        self.assertTrue(plan.empty)
        self.assertIn("volume_per_day", plan.columns)
        self.assertIn("target_station", plan.columns)
        restored = pd.read_csv(StringIO(plan.to_csv(index=False)))
        self.assertEqual(restored.columns.tolist(), plan.columns.tolist())

    def test_published_volumes_do_not_exceed_unrounded_supply(self):
        peak, summary = self.inputs([-29, 15, 14])
        plan = build_dispatch_plan(peak, summary, days=10)
        self.assertAlmostEqual(plan.volume_per_day.sum(), 2.9)
        peak, summary = self.inputs([-45, 100])
        plan = build_dispatch_plan(peak, summary, days=23)
        self.assertLessEqual(plan.volume_per_day.sum(), 45 / 23 + 1e-12)
        self.assertAlmostEqual(plan.volume_per_day.sum(), 45 / 23)

    def test_invalid_day_denominator_is_rejected(self):
        peak, summary = self.inputs([-1, 1])
        for days in (0, -1):
            with self.subTest(days=days), self.assertRaises(ValueError):
                build_dispatch_plan(peak, summary, days=days)

    def test_dispatch_csv_round_trip_preserves_supply_limit(self):
        peak, summary = self.inputs([-45, 100])
        plan = build_dispatch_plan(peak, summary, days=23)
        temporary_root = PROJECT_ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            folder = Path(temporary)
            with patch.multiple(
                imbalance_dispatch, TABLE_DIR=folder,
                PEAK_TABLE_FILE=folder / "peak.csv", HOURLY_TABLE_FILE=folder / "hourly.csv",
                DISPATCH_TABLE_FILE=folder / "dispatch.csv",
            ):
                paths = imbalance_dispatch.write_tables(
                    peak.assign(grade_order=0), pd.DataFrame({"hour": [0]}), plan,
                )
            restored = pd.read_csv(paths["dispatch"], float_precision="round_trip")
            self.assertEqual(restored.volume_per_day.tolist(), plan.volume_per_day.tolist())
            self.assertLessEqual(restored.volume_per_day.sum(), 45 / 23 + 1e-12)


class SpatialInterfaceRegressionTests(unittest.TestCase):
    def test_station_coordinates_are_an_observed_pair(self):
        trips = make_station_trips([
            {"start_lat": 38.8, "start_lng": np.nan},
            {"start_lat": np.nan, "start_lng": -77.2},
            {"start_lat": 38.9, "start_lng": -77.1},
        ])
        row = attach_coordinates(trips).set_index("station_id").loc["S1"]
        self.assertEqual((row.lat, row.lng), (38.9, -77.1))

    def test_return_only_station_keeps_its_name(self):
        panel = build_hourly_panel(make_station_trips([{}]))
        names = panel.loc[panel.station_id.eq("S2"), "station_name"]
        self.assertTrue(names.eq("End").all())

    def test_core_station_ids_preserve_leading_zeros_and_literal_na(self):
        rows = [
            {"station_id": sid, "timestamp": "2024-01-22 08:00:00", "target": target,
             "actual": actual, "prediction": prediction}
            for sid in ("001", "NA")
            for target, actual, prediction in (("outflow", 3, 4), ("inflow", 1, 1), ("net_outflow", 2, 3))
        ]
        # 全数字文件会被推断为整数；含 NA 的文件还会把合法 ID 当作缺失值。
        for ids in (("001",), ("001", "NA")):
            raw = pd.DataFrame([row for row in rows if row["station_id"] in ids])
            result = load_core_predictions(StringIO(raw.to_csv(index=False)))
            self.assertEqual(set(result.station_id), set(ids))


class ZoneRegressionTests(unittest.TestCase):
    def setUp(self):
        self.stations = pd.DataFrame({
            "zone": ["西", "东"], "lat": [38.90, 38.90], "lng": [-77.10, -77.00],
        })

    def rides(self):
        return pd.DataFrame({
            "started_at": pd.to_datetime(["2024-01-08 08:00"]),
            "ended_at": pd.to_datetime(["2024-01-08 08:10"]),
            "start_lat": [38.90], "start_lng": [-77.10],
            "end_lat": [np.nan], "end_lng": [np.nan],
        })

    def test_unknown_endpoint_is_not_assigned_to_a_zone(self):
        rides = self.rides()
        assigned = assign_nearest_zone(rides, self.stations, "end_lat", "end_lng")
        self.assertTrue(assigned.isna().all())
        balance = zone_hotspot_balance(rides, self.stations, 1)
        self.assertEqual(balance.departures.sum(), 1)
        self.assertEqual(balance.arrivals.sum(), 0)

    def test_hotspots_skip_only_the_unlocated_endpoint(self):
        for missing in (np.nan, pd.NA):
            rides = self.rides()
            rides["end_lat"] = pd.Series([missing], dtype="Float64")
            rides["end_lng"] = pd.Series([missing], dtype="Float64")
            hotspots = dockless_hotspots(rides, self.stations, 1)
            self.assertEqual(hotspots.departures.sum(), 1)
            self.assertEqual(hotspots.arrivals.sum(), 0)
            self.assertEqual(hotspots.zone.tolist(), ["西"])

    def test_quota_includes_hotspot_and_prediction_only_segments(self):
        flows, correction, hotspots, zones = make_quota_inputs()
        flows = flows.loc[flows.zone.ne("甲")]
        quota = build_quota(flows, correction, hotspots, zones)
        self.assertEqual(len(quota), len(zones) * len(SEGMENTS))
        self.assertFalse(quota.duplicated(["zone", "segment"]).any())
        row = quota.set_index(["zone", "segment"]).loc[("甲", "早高峰")]
        self.assertEqual(row.observed_net, 0)
        self.assertEqual(row.delta_net, 10)
        self.assertEqual(row.hotspot_net, 10)
        self.assertEqual(row.gap, 0)

    def test_zone_labels_remain_unique_when_bearings_collide(self):
        summary = pd.DataFrame({
            "station_id": ["A", "B"], "total_trips": [10, 10],
            "lat": [38.900, 38.901], "lng": [-77.040, -77.040],
        })
        stations, zones = build_zones(summary, k=2)
        self.assertEqual(stations.zone.nunique(), 2)
        self.assertTrue(zones.zone.is_unique)
        self.assertEqual(set(stations.zone), set(zones.zone))


if __name__ == "__main__":
    unittest.main()
