import sys
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "utils"))
from data_utils import load_features_pre_race


# ---------------------------------------------------------------------------
# load_features_pre_race — BUG-03 regression tests
#
# Only one combined CSV exists per (start_year, end_year) pair, named
# features_pre_race_{start_year}-{end_year}.csv.
# The old code looped over range(2018, 2026) regardless of arguments,
# trying to load files that don't exist.
# ---------------------------------------------------------------------------

def _write_feature_csv(directory: Path, start: int, end: int) -> Path:
    """Write a minimal features CSV and return its path."""
    path = directory / f"features_pre_race_{start}-{end}.csv"
    path.write_text("Driver,Position\nVER,1\nHAM,2\n")
    return path


class TestLoadFeaturesPreRace:
    def test_returns_dict_with_correct_key(self, tmp_path):
        """Return value must be a dict keyed by '{start_year}-{end_year}'."""
        _write_feature_csv(tmp_path, 2018, 2025)
        result = load_features_pre_race(tmp_path, start_year=2018, end_year=2025)
        assert list(result.keys()) == ["2018-2025"]

    def test_returns_single_entry(self, tmp_path):
        """Dict must contain exactly one entry — not one per year in range."""
        _write_feature_csv(tmp_path, 2018, 2025)
        result = load_features_pre_race(tmp_path, start_year=2018, end_year=2025)
        assert len(result) == 1

    def test_dataframe_contents_are_correct(self, tmp_path):
        """The loaded DataFrame must match the file contents."""
        _write_feature_csv(tmp_path, 2018, 2025)
        result = load_features_pre_race(tmp_path, start_year=2018, end_year=2025)
        df = result["2018-2025"]
        assert list(df.columns) == ["Driver", "Position"]
        assert len(df) == 2

    def test_respects_start_year_param(self, tmp_path):
        """start_year must be used in the filename — not hardcoded to 2018."""
        _write_feature_csv(tmp_path, 2020, 2025)
        result = load_features_pre_race(tmp_path, start_year=2020, end_year=2025)
        assert "2020-2025" in result

    def test_respects_end_year_param(self, tmp_path):
        """end_year must be used in the filename — not hardcoded to 2025."""
        _write_feature_csv(tmp_path, 2018, 2024)
        result = load_features_pre_race(tmp_path, start_year=2018, end_year=2024)
        assert "2018-2024" in result

    def test_old_bug_would_ignore_start_year(self, tmp_path):
        """
        Regression for BUG-03.

        The old loop always started at 2018, so calling with start_year=2020
        would try to load features_pre_race_2018-2025.csv (which doesn't exist
        for a 2020-2025 run) and ignore features_pre_race_2020-2025.csv.
        The fix must load exactly the file matching the given start_year.
        """
        _write_feature_csv(tmp_path, 2020, 2025)
        # Must NOT raise — the correct file exists
        result = load_features_pre_race(tmp_path, start_year=2020, end_year=2025)
        assert "2020-2025" in result

        # The 2018-2025 file does NOT exist — calling with start_year=2018 must raise
        with pytest.raises(FileNotFoundError):
            load_features_pre_race(tmp_path, start_year=2018, end_year=2025)
