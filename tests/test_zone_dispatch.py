"""验证分区、时段映射、配额公式与跨区路线的基本约束。"""

import unittest

import numpy as np
import pandas as pd

from src.zone_dispatch import (
    SEGMENTS, attach_coverage, build_quota, build_routes, dockless_hotspots,
    dockless_trips, segment_of, zone_hotspot_balance, zone_name,
)


class SegmentTests(unittest.TestCase):
    def test_hours_map_to_four_segments(self) -> None:
        hours = pd.Series([0, 6, 7, 9, 10, 16, 17, 19, 20, 23])
        result = segment_of(hours).tolist()
        self.assertEqual(result, ["夜间", "夜间", "早高峰", "早高峰", "日间平峰",
                                  "日间平峰", "晚高峰", "晚高峰", "夜间", "夜间"])

    def test_zone_name_uses_bearing_and_distance(self) -> None:
        center_lat, center_lng = 38.90, -77.04
        self.assertEqual(zone_name(38.90, -77.04, center_lat, center_lng), "中心商务区")
        self.assertEqual(zone_name(38.95, -77.04, center_lat, center_lng), "北部")
        self.assertEqual(zone_name(38.90, -76.98, center_lat, center_lng), "东部")
        self.assertEqual(zone_name(38.85, -77.04, center_lat, center_lng), "南部")
        self.assertIn("外围", zone_name(39.10, -77.20, center_lat, center_lng))


def make_quota_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flows = pd.DataFrame({
        "zone": ["甲", "乙", "丙"],
        "segment": ["早高峰"] * 3,
        "departures": [80.0, 120.0, 50.0],
        "arrivals": [20.0, 30.0, 40.0],
        "observed_net": [60.0, 90.0, 10.0],
    })
    correction = pd.DataFrame({
        "zone": ["甲"], "segment": ["早高峰"], "delta_net": [10.0], "core_stations": [1],
    })
    hotspots = pd.DataFrame({
        "zone": ["甲", "乙", "丙"], "segment": ["早高峰"] * 3,
        "departures": [5.0, 20.0, 2.0], "arrivals": [15.0, 5.0, 2.0],
        "hotspot_net": [10.0, -15.0, 0.0],
    })
    zones = pd.DataFrame({
        "zone": ["甲", "乙", "丙"], "lat": [38.90, 38.95, 38.85],
        "lng": [-77.04, -77.04, -77.04], "stations": [10, 20, 5], "trips_share": [0.5, 0.4, 0.1],
    })
    return flows, correction, hotspots, zones


class QuotaTests(unittest.TestCase):
    def test_quota_applies_safety_stock_and_hotspot_supply(self) -> None:
        flows, correction, hotspots, zones = make_quota_inputs()
        quota = build_quota(flows, correction, hotspots, zones).set_index("zone")
        # 甲：观测 60 + 修正 10 − 热点 10 = 60，配额 ceil(60×1.15)=69
        self.assertAlmostEqual(quota.loc["甲", "gap"], 60.0)
        self.assertAlmostEqual(quota.loc["甲", "quota"], 69.0)
        self.assertEqual(quota.loc["甲", "available"], 0.0)
        # 乙：观测 90 − (−15) = 105，配额 ceil(105×1.15)=121
        self.assertAlmostEqual(quota.loc["乙", "gap"], 105.0)
        self.assertAlmostEqual(quota.loc["乙", "quota"], 121.0)
        # 丙：观测 10，热点 0 → 配额 12
        self.assertAlmostEqual(quota.loc["丙", "quota"], 12.0)

    def test_negative_gap_becomes_available(self) -> None:
        flows = pd.DataFrame({
            "zone": ["甲"], "segment": ["早高峰"], "departures": [10.0],
            "arrivals": [50.0], "observed_net": [-40.0],
        })
        correction = pd.DataFrame({"zone": ["甲"], "segment": ["早高峰"], "delta_net": [0.0],
                                   "core_stations": [0]})
        hotspots = pd.DataFrame({"zone": ["甲"], "segment": ["早高峰"], "departures": [0.0],
                                 "arrivals": [0.0], "hotspot_net": [0.0]})
        zones = pd.DataFrame({"zone": ["甲"], "lat": [38.9], "lng": [-77.0],
                              "stations": [1], "trips_share": [1.0]})
        quota = build_quota(flows, correction, hotspots, zones).iloc[0]
        self.assertEqual(quota["quota"], 0.0)
        self.assertAlmostEqual(quota["available"], 40.0)


class RouteTests(unittest.TestCase):
    def test_routes_respect_supply_and_demand(self) -> None:
        flows, correction, hotspots, zones = make_quota_inputs()
        quota = build_quota(flows, correction, hotspots, zones)
        quota["available"] = [0.0, 0.0, 50.0]
        routes = build_routes(quota)
        self.assertFalse(routes.empty)
        self.assertTrue((routes["volume_per_day"] > 0).all())
        self.assertTrue(set(routes["target_zone"]).issubset({"甲", "乙"}))
        self.assertLessEqual(routes["volume_per_day"].sum(), 50.0 + 1e-9)
        self.assertTrue((routes["volume_per_day"] <= routes["target_quota"] + 1e-9).all())

    def test_coverage_is_computed_per_target(self) -> None:
        quota = pd.DataFrame({
            "segment": ["早高峰", "早高峰"], "zone": ["甲", "乙"],
            "quota": [100.0, 40.0], "available": [0.0, 0.0], "predicted_net": [100.0, 40.0],
            "observed_net": [100.0, 40.0], "delta_net": [0.0, 0.0], "hotspot_net": [0.0, 0.0],
            "gap": [100.0, 40.0], "core_stations": [0, 0], "stations": [1, 1],
            "trips_share": [0.5, 0.5], "lat": [38.9, 38.95], "lng": [-77.0, -77.0],
        })
        routes = pd.DataFrame({
            "segment": ["早高峰"], "source_zone": ["乙"], "target_zone": ["甲"],
            "volume_per_day": [40.0], "distance_km": [1.2], "target_quota": [100.0],
            "share_of_target": [0.4],
        })
        result = attach_coverage(quota, routes).set_index("zone")
        self.assertAlmostEqual(result.loc["甲", "delivered"], 40.0)
        self.assertAlmostEqual(result.loc["甲", "uncovered"], 60.0)
        self.assertAlmostEqual(result.loc["甲", "coverage"], 0.4)
        self.assertEqual(result.loc["乙", "delivered"], 0.0)
        self.assertIn("缺口", result.loc["甲", "remark"])


class DispatchBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stations = pd.DataFrame({
            "zone": ["西", "东"], "lat": [38.90, 38.90], "lng": [-77.10, -77.00],
        })

    def rides(self, starts, ends) -> pd.DataFrame:
        return pd.DataFrame({
            "started_at": pd.to_datetime(starts), "ended_at": pd.to_datetime(ends),
            "start_lat": 38.90, "start_lng": -77.10,
            "end_lat": 38.90, "end_lng": -77.00,
            "is_dockless_ebike": True,
        })

    def test_cross_segment_arrival_uses_return_time(self) -> None:
        data = self.rides(["2024-01-08 09:50"], ["2024-01-08 10:10"])
        balance = zone_hotspot_balance(data, self.stations, 1).set_index(["zone", "segment"])
        self.assertEqual(balance.loc[("西", "早高峰"), "departures"], 1)
        self.assertEqual(balance.loc[("东", "日间平峰"), "arrivals"], 1)
        self.assertNotIn(("东", "早高峰"), balance.index)

    def test_weekend_boundary_filters_each_endpoint_independently(self) -> None:
        data = self.rides(["2024-01-05 23:50", "2024-01-07 23:50"],
                          ["2024-01-06 00:10", "2024-01-08 00:10"])
        cohort = dockless_trips(data)
        self.assertEqual(len(cohort), 2)  # 周日出发、周一归还不能提前删掉。
        balance = zone_hotspot_balance(cohort, self.stations, 1)
        self.assertEqual(balance.departures.sum(), 1)
        self.assertEqual(balance.arrivals.sum(), 1)
        hotspots = dockless_hotspots(cohort, self.stations, 1)
        self.assertEqual(hotspots.departures.sum(), 1)
        self.assertEqual(hotspots.arrivals.sum(), 1)

    def test_non_electric_missing_station_is_not_dockless(self) -> None:
        data = self.rides(["2024-01-08 09:50"], ["2024-01-08 10:10"])
        data["is_dockless_ebike"] = False
        self.assertTrue(dockless_trips(data).empty)

    def test_supply_rounding_never_creates_an_extra_vehicle(self) -> None:
        flows, correction, hotspots, zones = make_quota_inputs()
        flows.loc[0, "observed_net"] = -1.1
        correction["delta_net"] = 0
        hotspots["hotspot_net"] = 0
        quota = build_quota(flows, correction, hotspots, zones)
        source = quota.loc[quota.zone.eq("甲")].iloc[0]
        self.assertEqual(source.available, 1)
        self.assertLessEqual(source.available, -source.gap)

    def test_one_vehicle_can_satisfy_one_vehicle_demand(self) -> None:
        quota = self.stations.assign(segment="早高峰", quota=[0., 1.], available=[1., 0.])
        routes = build_routes(quota)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes.iloc[0].volume_per_day, 1)
        self.assertEqual(routes.iloc[0].share_of_target, 1)

    def test_empty_routes_have_schema_and_zero_coverage(self) -> None:
        quota = self.stations.assign(segment="早高峰", quota=2., available=0.)
        routes = build_routes(quota)
        self.assertTrue(routes.empty)
        result = attach_coverage(quota, routes)
        self.assertTrue(result.delivered.eq(0).all())
        self.assertTrue(result.uncovered.eq(2).all())
        self.assertTrue(result.coverage.eq(0).all())

    def test_total_supply_and_target_caps_hold_across_multiple_routes(self) -> None:
        quota = pd.DataFrame({
            "zone": ["甲", "乙", "丙", "丁"], "segment": ["早高峰"] * 4,
            "quota": [0., 0., 5., 5.], "available": [4., 3., 0., 0.],
            "lat": [38.90, 38.91, 38.92, 38.93], "lng": [-77.04] * 4,
        })
        routes = build_routes(quota)
        sources = routes.groupby("source_zone").volume_per_day.sum()
        targets = routes.groupby("target_zone").volume_per_day.sum()
        self.assertEqual(routes.volume_per_day.sum(), 7)
        self.assertTrue(sources.le(quota.set_index("zone").available.reindex(sources.index)).all())
        self.assertTrue(targets.le(quota.set_index("zone").quota.reindex(targets.index)).all())


if __name__ == "__main__":
    unittest.main()
