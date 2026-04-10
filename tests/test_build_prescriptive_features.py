import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "features"))
from build_prescriptive_features import extract_lap_features


# ---------------------------------------------------------------------------
# BUG-06 regression tests — rename key typos in extract_lap_features()
#
# Old code had:
#   "Compund": "dominant_compound"   (missing 'o')
#   "String":  "stint_count"         ('String' instead of 'Stint')
#
# Both silently no-op'd: the output kept the aggregated pandas names
# "Compound" and "Stint" (the groupby-agg keys) instead of the expected
# "dominant_compound" and "stint_count".
#
# Fix: corrected keys to "Compound" and "Stint".
# ---------------------------------------------------------------------------

def make_laps_df():
    """Minimal laps DataFrame with the columns extract_lap_features needs."""
    return pd.DataFrame({
        "Year":       [2024, 2024, 2024, 2024, 2024, 2024],
        "GrandPrix":  ["Bahrain"] * 6,
        "Driver":     ["VER", "VER", "VER", "HAM", "HAM", "HAM"],
        "LapTime":    [90.1, 91.2, 90.5, 92.0, 91.8, 92.3],
        "SpeedI1":    [200.0, 201.0, 202.0, 198.0, 199.0, 197.0],
        "SpeedFL":    [310.0, 312.0, 311.0, 308.0, 309.0, 307.0],
        "IsAccurate": [1, 1, 0, 1, 0, 1],
        "Compound":   ["SOFT", "SOFT", "MEDIUM", "HARD", "HARD", "SOFT"],
        "Stint":      [1, 2, 3, 1, 2, 3],
        "LapNumber":  [1, 2, 3, 1, 2, 3],
    })


class TestExtractLapFeatures:
    def test_dominant_compound_column_exists(self):
        """Output must contain 'dominant_compound', not the raw aggregation key 'Compound'."""
        df = make_laps_df()
        result = extract_lap_features(df)
        assert "dominant_compound" in result.columns, (
            "'dominant_compound' missing — 'Compound' rename key was likely still a typo (BUG-06)"
        )

    def test_stint_count_column_exists(self):
        """Output must contain 'stint_count', not the raw aggregation key 'Stint'."""
        df = make_laps_df()
        result = extract_lap_features(df)
        assert "stint_count" in result.columns, (
            "'stint_count' missing — 'Stint' rename key was likely still a typo (BUG-06)"
        )

    def test_raw_compound_key_not_in_output(self):
        """'Compound' (the agg key) must not appear as a column name after rename."""
        df = make_laps_df()
        result = extract_lap_features(df)
        assert "Compound" not in result.columns, (
            "'Compound' still present — rename to 'dominant_compound' did not apply"
        )

    def test_raw_stint_key_not_in_output(self):
        """'Stint' (the agg key) must not appear as a column name after rename."""
        df = make_laps_df()
        result = extract_lap_features(df)
        assert "Stint" not in result.columns, (
            "'Stint' still present — rename to 'stint_count' did not apply"
        )

    def test_dominant_compound_has_correct_values(self):
        """dominant_compound should be the mode Compound per driver."""
        df = make_laps_df()
        result = extract_lap_features(df)
        ver_row = result[result["Driver"] == "VER"].iloc[0]
        # VER: SOFT x2, MEDIUM x1 → mode = SOFT
        assert ver_row["dominant_compound"] == "SOFT"

    def test_stint_count_has_correct_values(self):
        """stint_count should be max Stint per driver."""
        df = make_laps_df()
        result = extract_lap_features(df)
        ver_row = result[result["Driver"] == "VER"].iloc[0]
        ham_row = result[result["Driver"] == "HAM"].iloc[0]
        assert ver_row["stint_count"] == 3
        assert ham_row["stint_count"] == 3

    def test_all_expected_columns_present(self):
        """Smoke test: all renamed output columns must be present."""
        expected = {
            "Year", "GrandPrix", "Driver",
            "avg_lap_time", "avg_speed", "max_speed",
            "data_accuracy", "dominant_compound", "stint_count", "laps_completed",
        }
        result = extract_lap_features(make_laps_df())
        missing = expected - set(result.columns)
        assert not missing, f"Missing columns in output: {missing}"
