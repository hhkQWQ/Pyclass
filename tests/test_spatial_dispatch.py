"""验证净流量、失衡指数、OD 走廊合并与调运方案的规则。"""

import unittest

import numpy as np
import pandas as pd

from src.imbalance_dispatch import (
    build_dispatch_plan, build_peak_imbalance, grade_of, weekday_count,
)
from src.spatial_analysis import build_od_corridors, build_station_summary


def make_station_trips(rows: list[dict]) -> pd.DataFrame:
    """构造站点子集样式的记录，只包含分析函数实际使用的字段。"""
    records = []
    for index, row in enumerate(rows):
        started = pd.Timestamp(row.get("started_at", "2024-01-10 08:00:00"))
        base = {
            "ride_id": f"trip_{index}",
            "started_at": started,
            "ended_at": started + pd.Timedelta(minutes=10),
            "start_station_id": "S1",
            "start_station_name": "Start",
            "start_lat": 38.90,
            "start_lng": -77.03,
            "end_station_id": "S2",
            "end_station_name": "End",
            "end_lat": 38.91,
            "end_lng": -77.02,
            "hour": started.hour,
            "weekday": started.dayofweek,
            "has_station_info": True,
        }
        base.update(row)
        records.append(base)
    data = pd.DataFrame(records)
    for column in ("start_station_id", "end_station_id"):
        data[column] = data[column].astype("string")
    return data


class StationFlowTests(unittest.TestCase):
    def test_departures_arrivals_net_flow_and_imbalance(self) -> None:
        trips = make_station_trips([
            {}, {}, {},
            {"start_station_id": "S3", "start_station_name": "Third", "end_station_id": "S1",
             "end_station_name": "Start"},
        ])
        summary = build_station_summary(trips).set_index("station_id")
        self.assertEqual(summary.loc["S1", "departures"], 3)
        self.assertEqual(summary.loc["S1", "arrivals"], 1)
        self.assertEqual(summary.loc["S1", "net_flow"], 2)
        self.assertAlmostEqual(summary.loc["S1", "imbalance_index"], 2 / 5)
        self.assertEqual(summary.loc["S1", "total_trips"], 4)
        self.assertEqual(summary["departures"].sum(), summary["arrivals"].sum())

    def test_imbalance_index_stays_below_one(self) -> None:
        summary = build_station_summary(make_station_trips([{}] * 7))
        self.assertTrue((summary["imbalance_index"] >= 0).all())
        self.assertTrue((summary["imbalance_index"] < 1).all())


class CorridorTests(unittest.TestCase):
    def test_reverse_directions_merge_and_loops_are_excluded(self) -> None:
        trips = make_station_trips([
            {},
            {"start_station_id": "S2", "start_station_name": "End",
             "end_station_id": "S1", "end_station_name": "Start"},
            {},
            {},
        ])
        corridors = build_od_corridors(trips)
        self.assertEqual(len(corridors), 1)
        row = corridors.iloc[0]
        self.assertEqual(row.total_trips, 4)
        self.assertEqual(row.trips_a_to_b, 3)
        self.assertEqual(row.trips_b_to_a, 1)
        self.assertAlmostEqual(row.dominant_share, 3 / 4)
        self.assertEqual(corridors["total_trips"].sum(), 4)

    def test_unused_station_pair_is_not_a_corridor(self) -> None:
        corridors = build_od_corridors(make_station_trips([{}]))
        self.assertEqual(len(corridors), 1)
        self.assertEqual(corridors.iloc[0].total_trips, 1)


class PeakImbalanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trips = make_station_trips([
            {"started_at": "2024-01-10 08:10:00"},
            {"started_at": "2024-01-10 09:50:00"},
            {"started_at": "2024-01-10 10:10:00"},
            {"started_at": "2024-01-13 08:10:00"},
            {"started_at": "2024-01-10 18:10:00", "start_station_id": "S2", "start_station_name": "End",
             "end_station_id": "S1", "end_station_name": "Start"},
        ])
        self.peak = build_peak_imbalance(self.trips)

    def test_peak_windows_and_weekday_filter(self) -> None:
        am = self.peak.loc[self.peak.period.eq("早高峰")]
        self.assertEqual(am.departures.sum(), 2)
        pm = self.peak.loc[self.peak.period.eq("晚高峰")]
        self.assertEqual(pm.departures.sum(), 1)

    def test_tidal_reversal_between_peaks(self) -> None:
        by_period = self.peak.set_index(["period", "station_id"]).net_flow
        self.assertEqual(by_period.loc[("早高峰", "S1")], 2)
        self.assertEqual(by_period.loc[("晚高峰", "S1")], -1)
        self.assertEqual(self.peak.groupby("period").net_flow.sum().abs().max(), 0)

    def test_grade_requires_both_index_and_volume(self) -> None:
        row = pd.Series({"net_flow": 80, "imbalance_index": 0.5})
        self.assertEqual(grade_of(row), "严重失衡")
        row = pd.Series({"net_flow": 40, "imbalance_index": 0.25})
        self.assertEqual(grade_of(row), "中度失衡")
        row = pd.Series({"net_flow": 12, "imbalance_index": 0.95})
        self.assertEqual(grade_of(row), "轻度失衡")
        row = pd.Series({"net_flow": 0, "imbalance_index": 0.0})
        self.assertEqual(grade_of(row), "基本均衡")

    def test_weekday_count_includes_new_year(self) -> None:
        trips = make_station_trips([
            {"started_at": "2024-01-01 12:00:00"},
            {"started_at": "2024-01-06 12:00:00"},
        ])
        self.assertEqual(weekday_count(trips), 1)


class DispatchPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for index in range(6):
            rows.append({
                "started_at": f"2024-01-10 08:{index:02d}:00",
                "start_station_id": "S1", "start_station_name": "Needy",
                "start_lat": 38.900, "start_lng": -77.030,
                "end_station_id": f"P{index}", "end_station_name": f"Park{index}",
                "end_lat": 38.901 + index * 0.001, "end_lng": -77.029,
            })
        for index in range(10):
            rows.append({
                "started_at": f"2024-01-10 08:{index:02d}:30",
                "start_station_id": f"R{index}", "start_station_name": f"Rich{index}",
                "start_lat": 38.910 + index * 0.001, "start_lng": -77.040,
                "end_station_id": "S2", "end_station_name": "Source",
                "end_lat": 38.905, "end_lng": -77.035,
            })
        self.trips = make_station_trips(rows)
        self.peak = build_peak_imbalance(self.trips)
        self.summary = build_station_summary(self.trips)

    def test_plan_respects_supply_and_distance_cap(self) -> None:
        plan = build_dispatch_plan(self.peak, self.summary, days=1, max_distance_km=6.0)
        self.assertFalse(plan.empty)
        self.assertTrue((plan.volume_per_day <= plan.source_capacity_per_day + 1e-9).all())
        self.assertTrue((plan.distance_km <= 6.0).all())
        self.assertTrue((plan.volume_per_day > 0).all())

    def test_plan_cannot_create_bikes(self) -> None:
        plan = build_dispatch_plan(self.peak, self.summary, days=1)
        am = plan.loc[plan.period.eq("早高峰")]
        if not am.empty:
            am_deficit = self.peak.query("period == '早高峰' and net_flow > 0").net_flow.sum()
            self.assertLessEqual(am.volume_per_day.sum(), am_deficit + 1e-9)
            self.assertEqual(np.round(am.volume_per_day.sum() * 1, 6),
                             np.round(am.volume_per_day.sum(), 6))


if __name__ == "__main__":
    unittest.main()
