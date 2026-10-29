"""验证分区、时段映射、配额公式与跨区路线的基本约束。"""

import unittest

import numpy as np
import pandas as pd

from src.zone_dispatch import (
    SEGMENTS, attach_coverage, build_quota, build_routes, segment_of, zone_name,
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


if __name__ == "__main__":
    unittest.main()
