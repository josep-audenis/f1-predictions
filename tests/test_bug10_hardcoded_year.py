"""
BUG-10 regression tests — hardcoded year 2025 in predict_next_race.py.

Old behaviour:
  RACE_CALENDAR: only 4 races, hardcoded to 2025 dates
  load_features_for_race(year="2025"): path always ended with "-2025.csv"
  predict_next_race() called the local next_race() / quali_done() helpers

Fix:
  - RACE_CALENDAR, next_race(), quali_done() removed
  - load_features_for_race(year) derives the correct path: features_pre_race_{year}-{year}.csv
  - predict_next_race() delegates to calendar_utils.get_next_race() which uses
    FastF1 and derives the year from datetime.now()
"""
import sys
import json
import pytest
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "predictions"))
import predict_next_race as pnr


# ---------------------------------------------------------------------------
# Removed globals / functions
# ---------------------------------------------------------------------------

class TestRemovedHardcodedCalendar:
    def test_race_calendar_removed(self):
        """RACE_CALENDAR must not exist — it was hardcoded to 2025 only."""
        assert not hasattr(pnr, "RACE_CALENDAR"), (
            "RACE_CALENDAR still present — hardcoded 2025 calendar must be removed (BUG-10)"
        )

    def test_next_race_helper_removed(self):
        """Local next_race() must be gone — replaced by calendar_utils.get_next_race()."""
        assert not hasattr(pnr, "next_race"), (
            "next_race() still present — must be replaced by calendar_utils.get_next_race() (BUG-10)"
        )

    def test_quali_done_helper_removed(self):
        """Local quali_done() must be gone — replaced by has_quali from calendar_utils."""
        assert not hasattr(pnr, "quali_done"), (
            "quali_done() still present — must be replaced by has_quali from calendar_utils (BUG-10)"
        )

    def test_get_next_race_imported(self):
        """get_next_race from calendar_utils must be imported."""
        import utils.calendar_utils as cu
        assert pnr.get_next_race is cu.get_next_race, (
            "get_next_race not imported from calendar_utils (BUG-10)"
        )


# ---------------------------------------------------------------------------
# load_features_for_race uses the year parameter, not hardcoded 2025
# ---------------------------------------------------------------------------

class TestLoadFeaturesForRace:
    def test_path_uses_year_parameter(self, tmp_path):
        """load_features_for_race(2026) must read features_pre_race_2026-2026.csv."""
        csv_path = tmp_path / "features_pre_race_2026-2026.csv"
        csv_path.write_text("Driver,Position\nVER,1\n")

        with patch.object(pnr, "DATA_DIR", tmp_path):
            df = pnr.load_features_for_race(2026)
        assert len(df) == 1
        assert df.iloc[0]["Driver"] == "VER"

    def test_does_not_hardcode_2025(self, tmp_path):
        """
        load_features_for_race(2027) must NOT try to open features_pre_race_2027-2025.csv.
        Only features_pre_race_2027-2027.csv must be opened.
        """
        wrong_path = tmp_path / "features_pre_race_2027-2025.csv"
        wrong_path.write_text("Driver,Position\nHAM,1\n")
        # Correct path deliberately absent → must raise

        with patch.object(pnr, "DATA_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                pnr.load_features_for_race(2027)

    def test_path_matches_year_year_pattern(self, tmp_path):
        """Filename pattern must be features_pre_race_{year}-{year}.csv for any year."""
        for year in [2025, 2026, 2030]:
            csv_path = tmp_path / f"features_pre_race_{year}-{year}.csv"
            csv_path.write_text("Driver,Position\nLEC,2\n")
            with patch.object(pnr, "DATA_DIR", tmp_path):
                df = pnr.load_features_for_race(year)
            assert len(df) == 1
            csv_path.unlink()


# ---------------------------------------------------------------------------
# predict_next_race() uses get_next_race() and the returned year
# ---------------------------------------------------------------------------

def _make_event(year=2026, gp="Monaco", has_quali=False):
    return {
        "year": year,
        "grand_prix": gp,
        "has_quali": has_quali,
        "race_datetime": datetime(year, 5, 25, 13, 0, tzinfo=timezone.utc),
        "quali_datetime": datetime(year, 5, 24, 12, 0, tzinfo=timezone.utc),
        "round": 8,
        "country": "Monaco",
        "event_format": "conventional",
        "raw": {},
    }


def _make_features_df(gp="Monaco", n=5):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Driver":                  [f"D{i}" for i in range(n)],
        "TeamName":                ["TeamA"] * n,
        "GrandPrix":               [gp] * n,
        "Position":                list(range(1, n + 1)),
        "grid_position":           list(range(1, n + 1)),
        "is_top10_start":          [1] * n,
        "grid_vs_team_avg":        rng.uniform(-1, 1, size=n),
        "driver_avg_quali_last5":  rng.uniform(1, 10, size=n),
        "team_avg_quali_last5":    rng.uniform(1, 10, size=n),
        "driver_avg_finish_last5": rng.uniform(1, 10, size=n),
        "driver_points_last5":     rng.uniform(0, 25, size=n),
        "team_avg_finish_last5":   rng.uniform(1, 10, size=n),
        "team_points_last5":       rng.uniform(0, 25, size=n),
    })


def _make_dummy_model(tmp_path):
    model = LogisticRegression()
    model.fit(np.random.randn(20, 10), np.random.randint(0, 2, 20))
    return model


def _stub_prepare_features(df, use_quali_features=True, fill_mean=None):
    """Stand-in for prepare_features that returns a correctly-shaped array."""
    n = len(df)
    return np.zeros((n, 3))


class TestPredictNextRaceUsesCalendarUtils:
    def test_raises_when_get_next_race_returns_none(self):
        """If no upcoming race exists, predict_next_race must raise ValueError."""
        with patch.object(pnr, "get_next_race", return_value=None):
            with pytest.raises(ValueError, match="No upcoming race"):
                pnr.predict_next_race()

    def test_uses_year_from_get_next_race(self):
        """predict_next_race must load features for the year returned by get_next_race."""
        event = _make_event(year=2026, gp="Monaco")
        features_df = _make_features_df(gp="Monaco")

        dummy_model = MagicMock()
        dummy_model.predict_proba.return_value = np.array([[0.6, 0.4]] * len(features_df))

        with (
            patch.object(pnr, "get_next_race", return_value=event),
            patch.object(pnr, "load_features_for_race", return_value=features_df) as mock_load,
            patch.object(pnr, "load_best_model", return_value=dummy_model),
            patch.object(pnr, "prepare_features", side_effect=_stub_prepare_features),
        ):
            pnr.predict_next_race()

        mock_load.assert_called_once_with(2026)

    def test_uses_has_quali_from_event(self):
        """predict_next_race must pass event['has_quali'] to load_best_model."""
        event = _make_event(year=2026, gp="Monaco", has_quali=True)
        features_df = _make_features_df(gp="Monaco")

        dummy_model = MagicMock()
        dummy_model.predict_proba.return_value = np.array([[0.6, 0.4]] * len(features_df))

        with (
            patch.object(pnr, "get_next_race", return_value=event),
            patch.object(pnr, "load_features_for_race", return_value=features_df),
            patch.object(pnr, "load_best_model", return_value=dummy_model) as mock_lbm,
            patch.object(pnr, "prepare_features", side_effect=_stub_prepare_features),
        ):
            pnr.predict_next_race()

        mock_lbm.assert_called_once_with(True)
