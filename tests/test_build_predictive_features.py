import sys
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "features"))
from build_predictive_features import rolling_stat, build_round_number_map


# ---------------------------------------------------------------------------
# rolling_stat
# ---------------------------------------------------------------------------

def make_results(gp_names, round_numbers, positions, driver="VER"):
    return pd.DataFrame({
        "Year": [2024] * len(gp_names),
        "GrandPrix": gp_names,
        "RoundNumber": round_numbers,
        "Driver": [driver] * len(gp_names),
        "Position": positions,
    })


class TestRollingStat:
    def test_uses_round_number_order_not_alphabetical(self):
        """
        Core regression test for bug #3.

        'Abu Dhabi' sorts before 'Australian' and 'Bahrain' alphabetically,
        but Round 24 is AFTER Rounds 1 and 2 chronologically.

        Old code sorted by GrandPrix string → Abu Dhabi appeared first → wrong
        rolling window.  New code sorts by RoundNumber → correct order.
        """
        # Chronological order: Australian (R1, pos=3) → Bahrain (R2, pos=2) → Abu Dhabi (R24, pos=1)
        # Alphabetical order:  Abu Dhabi              → Australian            → Bahrain
        df = make_results(
            gp_names=    ["Abu Dhabi", "Australian", "Bahrain"],
            round_numbers=[24,          1,            2],
            positions=   [1,            3,            2],
        )

        result = rolling_stat(df, ["Driver"], "Position", window=5, func="mean")
        # Assign back by index so pandas aligns correctly (result keeps original df index)
        df["rolling"] = result

        # Australian (first race): no prior data → NaN
        aus_row = df[df["GrandPrix"] == "Australian"].iloc[0]
        assert pd.isna(aus_row["rolling"]), \
            "Australian GP is round 1; there should be no prior data"

        # Bahrain (round 2): only prior race is Australian (pos=3) → mean=3.0
        bah_row = df[df["GrandPrix"] == "Bahrain"].iloc[0]
        assert bah_row["rolling"] == pytest.approx(3.0), \
            f"Bahrain rolling mean should be 3.0 (only Australian before it), got {bah_row['rolling']}"

        # Abu Dhabi (round 24): prior races are Australian(3) and Bahrain(2) → mean=2.5
        abu_row = df[df["GrandPrix"] == "Abu Dhabi"].iloc[0]
        assert abu_row["rolling"] == pytest.approx(2.5), \
            f"Abu Dhabi rolling mean should be 2.5, got {abu_row['rolling']}"

    def test_alphabetical_order_would_give_wrong_result(self):
        """
        Confirm that sorting by GrandPrix string (the old approach) produces
        a different — incorrect — result for the same data.
        """
        df = make_results(
            gp_names=    ["Abu Dhabi", "Australian", "Bahrain"],
            round_numbers=[24,          1,            2],
            positions=   [1,            3,            2],
        )

        # Manually replicate what the OLD code did (sort by string, assign back by index)
        df_alpha = df.copy()
        df_alpha_sorted = df_alpha.sort_values(["Year", "GrandPrix"])
        result_alpha = (
            df_alpha_sorted.groupby(["Driver"], group_keys=False)["Position"]
            .apply(lambda x: x.shift(1).rolling(5, min_periods=1).agg("mean"))
        )
        df_alpha["rolling"] = result_alpha

        # Under alpha sort: Abu Dhabi is "first" → gets NaN (wrong, it's race 24)
        abu_row = df_alpha[df_alpha["GrandPrix"] == "Abu Dhabi"].iloc[0]
        assert pd.isna(abu_row["rolling"]), \
            "Confirming alpha sort wrongly treats Abu Dhabi as the first race"

        # Under alpha sort: Australian gets prior = Abu Dhabi(1) → 1.0 (wrong, should be NaN)
        aus_row = df_alpha[df_alpha["GrandPrix"] == "Australian"].iloc[0]
        assert aus_row["rolling"] == pytest.approx(1.0), \
            "Confirming alpha sort wrongly assigns a prior value to the opening race"

    def test_window_respects_chronological_boundary(self):
        """Rolling window of 3 should only look back 3 races chronologically."""
        # Round order: Australia(R1,pos=10), Bahrain(R2,pos=2), Japan(R4,pos=4), Spain(R6,pos=6), Monaco(R8,pos=8)
        # shift(1) series:                   NaN,               10,              2,               4,              6
        # rolling(3):                         NaN,               10.0,            6.0,             5.33,           4.0
        # Monaco (last): window over [Spain(6), Japan(4), Bahrain(2)] shifted values → mean([4,2]) ...
        # More precisely: at Monaco, shift(1) value=6; rolling looks back 3 → [2, 4, 6] → mean=4.0
        gp_names =     ["Monaco", "Bahrain", "Australia", "Japan", "Spain"]
        round_numbers = [8,        2,         1,           4,       6]
        positions =     [8,        2,         10,          4,       6]

        df = make_results(gp_names, round_numbers, positions)
        result = rolling_stat(df, ["Driver"], "Position", window=3, func="mean")
        df["rolling"] = result

        # Monaco is round 8 (last): prior 3 races are Bahrain(2), Japan(4), Spain(6) → mean=4.0
        monaco_row = df[df["GrandPrix"] == "Monaco"].iloc[0]
        assert monaco_row["rolling"] == pytest.approx(4.0), \
            f"Expected 4.0 for Monaco rolling(3), got {monaco_row['rolling']}"

    def test_multiple_drivers_are_independent(self):
        """Each driver's rolling stat should only use their own race history."""
        df = pd.DataFrame({
            "Year":        [2024, 2024, 2024, 2024],
            "GrandPrix":   ["Bahrain", "Australia", "Bahrain", "Australia"],
            "RoundNumber": [1,         2,            1,         2],
            "Driver":      ["VER",     "VER",        "HAM",     "HAM"],
            "Position":    [1,         3,             5,         7],
        })
        result = rolling_stat(df, ["Driver"], "Position", window=5, func="mean")
        df["rolling"] = result

        ver_aus = df[(df["Driver"] == "VER") & (df["GrandPrix"] == "Australia")].iloc[0]
        ham_aus = df[(df["Driver"] == "HAM") & (df["GrandPrix"] == "Australia")].iloc[0]

        assert ver_aus["rolling"] == pytest.approx(1.0)
        assert ham_aus["rolling"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# build_round_number_map
# ---------------------------------------------------------------------------

def make_mock_schedule(events):
    """Build a minimal FastF1-style schedule DataFrame."""
    rows = [{"RoundNumber": r, "EventName": name} for r, name in events]
    return pd.DataFrame(rows)


class TestBuildRoundNumberMap:
    def test_returns_expected_columns(self):
        mock_schedule = make_mock_schedule([(1, "Bahrain Grand Prix"), (2, "Saudi Arabian Grand Prix")])
        with patch("fastf1.get_event_schedule", return_value=mock_schedule):
            result = build_round_number_map([2024])
        assert set(result.columns) == {"Year", "GrandPrix", "RoundNumber"}

    def test_strips_grand_prix_suffix(self):
        mock_schedule = make_mock_schedule([(1, "Bahrain Grand Prix")])
        with patch("fastf1.get_event_schedule", return_value=mock_schedule):
            result = build_round_number_map([2024])
        assert result.iloc[0]["GrandPrix"] == "Bahrain"

    def test_round_numbers_are_integers(self):
        mock_schedule = make_mock_schedule([(1, "Bahrain Grand Prix"), (2, "Saudi Arabian Grand Prix")])
        with patch("fastf1.get_event_schedule", return_value=mock_schedule):
            result = build_round_number_map([2024])
        assert result["RoundNumber"].dtype == int or all(isinstance(v, int) for v in result["RoundNumber"])

    def test_multiple_years(self):
        mock_schedule = make_mock_schedule([(1, "Bahrain Grand Prix")])
        with patch("fastf1.get_event_schedule", return_value=mock_schedule):
            result = build_round_number_map([2023, 2024])
        assert set(result["Year"]) == {2023, 2024}
        assert len(result) == 2

    def test_non_grand_prix_event_names_preserved(self):
        """Events whose name doesn't contain 'Grand Prix' should be left as-is."""
        mock_schedule = make_mock_schedule([(1, "70th Anniversary")])
        with patch("fastf1.get_event_schedule", return_value=mock_schedule):
            result = build_round_number_map([2020])
        assert result.iloc[0]["GrandPrix"] == "70th Anniversary"
