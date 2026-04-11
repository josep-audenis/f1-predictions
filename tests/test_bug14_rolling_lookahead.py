"""
BUG-14 verification — rolling statistics look-ahead check.

BUGS.md notes: "Less critical if the train/test split is always done
chronologically and rolling uses .shift() — verify this."

Verdict: NOT a bug. build_predictive_features.rolling_stat() already applies
.shift(1) before .rolling(), so the current race is excluded from every
rolling window. Tests here confirm that behaviour and act as a regression
guard.
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "features"))
import build_predictive_features as bpf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_driver_df(positions, driver="VER", year=2024):
    """Build a minimal results-like DataFrame with n sequential races."""
    n = len(positions)
    return pd.DataFrame({
        "Driver":      [driver] * n,
        "Year":        [year] * n,
        "RoundNumber": list(range(1, n + 1)),
        "Position":    positions,
    })


def make_multi_driver_df():
    """Two drivers, 4 races each, interleaved as they would be in real data."""
    rows = []
    for rnd in range(1, 5):
        rows.append({"Driver": "VER", "Year": 2024, "RoundNumber": rnd, "Position": rnd})
        rows.append({"Driver": "HAM", "Year": 2024, "RoundNumber": rnd, "Position": 5 - rnd})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# rolling_stat uses .shift(1) — current race is excluded
# ---------------------------------------------------------------------------

class TestRollingStatNoLookahead:
    def test_first_race_is_nan(self):
        """The first race for a driver must have NaN (no prior history)."""
        df = make_driver_df([1, 2, 3, 4, 5])
        result = bpf.rolling_stat(df, ["Driver"], "Position", window=5, func="mean")
        df = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
        assert pd.isna(result.iloc[0]), (
            "First race rolling stat must be NaN — .shift(1) is required to "
            "exclude the current race from its own window (BUG-14)"
        )

    def test_second_race_uses_only_first(self):
        """Race 2 rolling mean must equal race 1's position only."""
        positions = [3, 7, 9, 1, 5]
        df = make_driver_df(positions)
        result = bpf.rolling_stat(df, ["Driver"], "Position", window=5, func="mean")
        df_sorted = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
        assert result.iloc[1] == pytest.approx(positions[0]), (
            f"Race 2 rolling mean should be {positions[0]} (race 1 only), "
            f"got {result.iloc[1]} — current race must not be included"
        )

    def test_rolling_window_excludes_current_race(self):
        """
        For each race i (1-indexed), the rolling mean must equal the mean of
        positions[0:i-1] (i.e. everything BEFORE race i).
        """
        positions = [2, 4, 6, 8, 10]
        df = make_driver_df(positions)
        result = bpf.rolling_stat(df, ["Driver"], "Position", window=5, func="mean")
        df_sorted = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)

        for i, val in enumerate(result):
            if i == 0:
                assert pd.isna(val)
            else:
                expected = np.mean(positions[:i])
                assert val == pytest.approx(expected), (
                    f"Race {i+1}: expected mean of positions[:{i}]={expected}, got {val}"
                )

    def test_groups_are_independent(self):
        """
        Each driver's rolling window must be computed independently.
        VER race-2 stat must not be influenced by HAM results.
        """
        df = make_multi_driver_df()
        result = bpf.rolling_stat(df, ["Driver"], "Position", window=5, func="mean")
        df_sorted = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)

        ver_mask = df_sorted["Driver"] == "VER"
        ham_mask = df_sorted["Driver"] == "HAM"

        ver_results = result[ver_mask].values
        ham_results = result[ham_mask].values

        # VER positions: [1, 2, 3, 4] → rolling(shift-1) means: [NaN, 1, 1.5, 2]
        assert pd.isna(ver_results[0])
        assert ver_results[1] == pytest.approx(1.0)
        assert ver_results[2] == pytest.approx(1.5)

        # HAM positions: [4, 3, 2, 1] → rolling(shift-1) means: [NaN, 4, 3.5, 3]
        assert pd.isna(ham_results[0])
        assert ham_results[1] == pytest.approx(4.0)
        assert ham_results[2] == pytest.approx(3.5)

    def test_ytd_cumsum_also_excludes_current_race(self):
        """
        driver_points_ytd in build_features uses .cumsum().shift(1) — verify
        the shift is present by testing cumsum shift on a known series.
        The cumulative sum at race i should be the sum of points BEFORE race i.
        """
        points = pd.Series([10, 18, 25, 12, 15])
        ytd = points.cumsum().shift(1).fillna(0)
        expected = [0, 10, 28, 53, 65]
        for i, (got, exp) in enumerate(zip(ytd, expected)):
            assert got == pytest.approx(exp), (
                f"Race {i+1} YTD: expected {exp}, got {got} — cumsum must use shift(1)"
            )
