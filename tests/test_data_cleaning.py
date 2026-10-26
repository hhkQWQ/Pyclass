"""验证实际导出、地理距离、未知环线与 CSV 类型恢复。"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data_audit import load_rules
from src.data_cleaning import clean_data, export_datasets, haversine_km, read_clean_data
from src.utils import PROJECT_ROOT
from tests.test_data_audit import make_data


class DataCleaningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = load_rules()

    def test_duration_filter_is_applied_to_actual_export(self) -> None:
        result = clean_data(make_data(
            {"ended_at": "2024-01-10 12:00:59"},
            {"ended_at": "2024-01-10 12:01:00"},
            {"ended_at": "2024-01-10 15:00:00"},
            {"ended_at": "2024-01-10 15:00:01"},
        ), self.rules)
        self.assertEqual(result.clean_trips.ride_id.tolist(), ["trip_1", "trip_2"])
        self.assertEqual(result.clean_trips.duration_min.tolist(), [1.0, 180.0])

    def test_haversine_known_distances_and_invalid_coordinates(self) -> None:
        coordinates = pd.DataFrame({
            "start_lat": [0, 0, 0, 91, 0], "start_lng": [0, 0, 179, 0, 0],
            "end_lat": [0, 0, 0, 0, None], "end_lng": [0, 90, -179, 0, 0],
        })
        distances = haversine_km(coordinates)
        self.assertEqual(distances.iloc[0], 0)
        self.assertAlmostEqual(distances.iloc[1], 10007.543398, places=5)
        self.assertAlmostEqual(distances.iloc[2], 222.389853, places=5)
        self.assertTrue(distances.iloc[3:].isna().all())

    def test_unknown_loop_is_not_false_and_classic_is_not_dockless(self) -> None:
        result = clean_data(make_data(
            {"end_station_id": "001", "end_station_name": "Start"},
            {"rideable_type": "electric_bike", "end_station_id": "", "end_station_name": ""},
            {"end_station_id": "", "end_station_name": ""}, {},
        ), self.rules)
        data = result.clean_trips
        self.assertTrue(data.loc[0, "is_loop_trip"])
        self.assertTrue(pd.isna(data.loc[1, "is_loop_trip"]))
        self.assertTrue(data.loc[1, "is_dockless_ebike"])
        self.assertFalse(data.loc[2, "is_dockless_ebike"])
        self.assertFalse(data.loc[3, "is_loop_trip"])
        self.assertEqual(len(result.station_trips), 2)

    def test_missing_and_bad_coordinates_keep_time_records_without_fake_distances(self) -> None:
        result = clean_data(make_data({"end_lat": ""}, {"end_lng": "oops"}, {"start_lat": "95"}), self.rules)
        self.assertEqual(len(result.clean_trips), 3)
        self.assertTrue(result.clean_trips.dist_km.isna().all())
        self.assertTrue(pd.isna(result.clean_trips.loc[0, "end_lat"]))
        self.assertTrue(pd.isna(result.clean_trips.loc[1, "end_lng"]))
        self.assertTrue(pd.isna(result.clean_trips.loc[2, "start_lat"]))

    def test_names_and_cross_month_time_features_are_exported(self) -> None:
        result = clean_data(make_data(
            {"start_station_name": "Variant"}, {"start_station_name": "Canonical"},
            {"start_station_name": "Canonical"},
            {"started_at": "2024-01-31 23:50:00", "ended_at": "2024-02-01 00:05:00", "start_station_name": "Canonical"},
            {"started_at": "2024-01-06 12:00:00", "ended_at": "2024-01-06 12:10:00", "start_station_name": "Canonical"},
        ), self.rules)
        data = result.clean_trips
        self.assertEqual(data.start_station_name.unique().tolist(), ["Canonical"])
        self.assertEqual(result.renamed_endpoints, 1)
        self.assertEqual(data.loc[3, "duration_min"], 15)
        self.assertEqual(data.loc[3, "hour"], 23)
        self.assertEqual(data.loc[3, "weekday"], 2)
        self.assertFalse(data.loc[3, "is_weekend"])
        self.assertTrue(data.loc[4, "is_weekend"])

    def test_csv_round_trip_preserves_ids_null_booleans_and_subset(self) -> None:
        result = clean_data(make_data(
            {}, {"rideable_type": "electric_bike", "end_station_id": "", "end_station_name": "", "end_lat": ""},
        ), self.rules)
        temporary_root = PROJECT_ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            outputs = export_datasets(result, Path(temporary))
            restored = read_clean_data(outputs["clean_trips"])
            station = read_clean_data(outputs["station_trips"])
            self.assertTrue(outputs["clean_trips"].read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertEqual(restored.loc[0, "start_station_id"], "001")
            self.assertFalse(restored.loc[0, "is_loop_trip"])
            self.assertTrue(pd.isna(restored.loc[1, "is_loop_trip"]))
            self.assertFalse(any(column.startswith("Unnamed") for column in restored.columns))
            pd.testing.assert_frame_equal(result.clean_trips, restored, check_exact=False, rtol=1e-12, atol=1e-12)
            pd.testing.assert_frame_equal(restored[restored.has_station_info].reset_index(drop=True), station)


if __name__ == "__main__":
    unittest.main()
