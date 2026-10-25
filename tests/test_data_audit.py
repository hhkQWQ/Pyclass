"""验证会影响数据保留结果的边界情况。"""

from dataclasses import replace
import unittest

import pandas as pd

from src.data_audit import RAW_COLUMNS, audit_data, load_rules


def make_data(*changes: dict[str, str]) -> pd.DataFrame:
    rows = []
    for index, updates in enumerate(changes):
        row = {
            "ride_id": f"trip_{index}", "rideable_type": "classic_bike",
            "started_at": "2024-01-10 12:00:00", "ended_at": "2024-01-10 12:10:00",
            "start_station_name": "Start", "start_station_id": "001",
            "end_station_name": "End", "end_station_id": "002",
            "start_lat": "38.9", "start_lng": "-77.0",
            "end_lat": "38.91", "end_lng": "-77.01", "member_casual": "member",
        }
        row.update(updates)
        rows.append(row)
    return pd.DataFrame(rows, columns=RAW_COLUMNS, dtype="string")


class DataAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = load_rules()

    def test_duration_boundaries_are_inclusive(self) -> None:
        data = make_data(*[
            {"ended_at": f"2024-01-10 {time}"}
            for time in ["12:00:59", "12:01:00", "15:00:00", "15:00:01", "12:00:00", "11:59:00"]
        ])
        result = audit_data(data, self.rules)
        self.assertEqual(result.valid_mask.tolist(), [False, True, True, False, False, False])
        self.assertEqual(int(result.flags.nonpositive_duration.sum()), 2)

    def test_cross_month_end_is_kept_but_outside_start_is_excluded(self) -> None:
        data = make_data(
            {"started_at": "2024-01-31 23:50:00", "ended_at": "2024-02-01 00:05:00"},
            {"started_at": "2024-02-01 00:00:00", "ended_at": "2024-02-01 00:10:00"},
            {"ended_at": "not-a-date"},
        )
        result = audit_data(data, self.rules)
        self.assertEqual(result.valid_mask.tolist(), [True, False, False])
        self.assertEqual(result.summary["valid_cross_month_end_rows"], 1)

    def test_missing_station_and_bad_coordinates_do_not_remove_time_records(self) -> None:
        data = make_data(
            {"end_station_id": "", "end_station_name": ""},
            {"end_lat": ""}, {"end_lng": "not-a-number"}, {"start_lng": "181"},
        )
        result = audit_data(data, self.rules)
        self.assertEqual(result.valid_mask.tolist(), [True] * 4)
        self.assertEqual(result.station_mask.tolist(), [False, True, True, True])
        self.assertTrue(result.flags.loc[0, "classic_missing_station"])
        self.assertTrue(result.flags.loc[1, "missing_coordinate"])
        self.assertTrue(result.flags.loc[2, "nonnumeric_coordinate"])
        self.assertTrue(result.flags.loc[3, "invalid_coordinate_range"])

    def test_duplicate_ids_keep_first_occurrence(self) -> None:
        result = audit_data(make_data({"ride_id": "same"}, {"ride_id": "same"}), self.rules)
        self.assertEqual(result.valid_mask.tolist(), [True, False])
        self.assertEqual(result.summary["excluded_rows"], 1)

    def test_overlapping_flags_are_not_counted_twice_in_funnel(self) -> None:
        data = make_data({"ended_at": "2024-01-11 13:00:00", "end_lat": ""})
        result = audit_data(data, self.rules)
        self.assertTrue(result.flags.loc[0, "long_duration"])
        self.assertTrue(result.flags.loc[0, "overdue_duration"])
        self.assertTrue(result.flags.loc[0, "missing_coordinate"])
        self.assertEqual(int(result.funnel.query("scope == 'full'").removed_rows.sum()), 1)

    def test_station_names_use_id_and_observed_frequency_without_mutating_input(self) -> None:
        data = make_data(
            {"start_station_name": "Beta"}, {"start_station_name": "  Alpha  "},
            {"start_station_name": "Alpha"},
            {"start_station_id": "003", "start_station_name": "Alpha"},
        )
        original = data.copy(deep=True)
        result = audit_data(data, self.rules)
        canonical = result.station_mapping.drop_duplicates("station_id").set_index("station_id")
        self.assertEqual(canonical.loc["001", "canonical_name"], "Alpha")
        self.assertIn("003", canonical.index)
        self.assertEqual(result.summary["station_name_repair_endpoints"], 1)
        self.assertEqual(result.summary["station_names_with_multiple_ids"], 1)
        pd.testing.assert_frame_equal(data, original)

    def test_station_name_tie_break_is_deterministic(self) -> None:
        data = make_data({"start_station_name": "Beta"}, {"start_station_name": "Alpha"})
        forward = audit_data(data, self.rules).station_mapping
        backward = audit_data(data.iloc[::-1], self.rules).station_mapping
        pd.testing.assert_frame_equal(forward, backward)
        self.assertEqual(forward.loc[forward.station_id.eq("001"), "canonical_name"].iloc[0], "Alpha")

    def test_invalid_rule_boundaries_fail_early(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.rules, duration_min_inclusive=0)
        with self.assertRaises(ValueError):
            replace(self.rules, analysis_end_exclusive=self.rules.analysis_start)

    def test_reference_counts_do_not_control_filtering(self) -> None:
        rules = replace(self.rules, reference_counts={"clean_trips": 0, "station_trips": 0})
        result = audit_data(make_data({}, {}), rules)
        self.assertEqual(result.summary["clean_trips"], 2)
        self.assertEqual(result.summary["station_trips"], 2)


if __name__ == "__main__":
    unittest.main()
