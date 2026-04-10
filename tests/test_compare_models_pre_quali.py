import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "predictions"))
import compare_models_top1_pre_quali as top1
import compare_models_top3_pre_quali as top3


# ---------------------------------------------------------------------------
# BUG-05 regression tests — column name typos in pre-quali preprocess()
#
# Old code dropped:
#   "driver_vag_quali_last5"   (typo: "vag" instead of "avg")
#   "team_avg_quali_last_5"    (extra underscore before "5")
# Because errors="ignore", the drop silently did nothing, leaking qualifying
# features into the "pre-quali" model.
#
# Fix: use the correct names "driver_avg_quali_last5" and "team_avg_quali_last5".
# ---------------------------------------------------------------------------

QUALI_LEAK_COLS = ["driver_avg_quali_last5", "team_avg_quali_last5"]


def make_minimal_df():
    """Return a minimal DataFrame that includes the qualifying columns that must be dropped."""
    n = 20
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Driver":                  ["VER", "HAM"] * (n // 2),
        "TeamName":                ["RBR", "MER"] * (n // 2),
        "GrandPrix":               ["Bahrain", "Australia"] * (n // 2),
        "Position":                rng.integers(1, 20, size=n),
        "grid_position":           rng.integers(1, 20, size=n),
        "is_top10_start":          rng.integers(0, 2, size=n),
        "grid_vs_team_avg":        rng.uniform(-5, 5, size=n),
        # The two columns that BUG-05 failed to drop:
        "driver_avg_quali_last5":  rng.uniform(1, 20, size=n),
        "team_avg_quali_last5":    rng.uniform(1, 20, size=n),
        # Additional feature columns
        "driver_avg_finish_last5": rng.uniform(1, 20, size=n),
        "driver_points_last5":     rng.uniform(0, 25, size=n),
        "team_avg_finish_last5":   rng.uniform(1, 20, size=n),
        "team_points_last5":       rng.uniform(0, 25, size=n),
    })


class TestTop1PreQualiDropsQualiCols:
    def test_driver_avg_quali_last5_not_in_features(self):
        """driver_avg_quali_last5 must be excluded from X (qualifying data leak)."""
        df = make_minimal_df()
        X, _ = top1.preprocess(df)
        # X is a numpy array; we need to check the column count reflects the drops.
        # Verify by checking the column count of the source df minus dropped cols.
        expected_cols = df.drop(
            columns=["Position", "grid_position", "is_top10_start",
                     "grid_vs_team_avg", "driver_avg_quali_last5", "team_avg_quali_last5"],
            errors="ignore",
        ).select_dtypes(include="number").columns
        # The scaled array width must equal the number of remaining numeric columns
        # after LabelEncoder has replaced string columns.
        assert X.shape[1] == len(df.drop(
            columns=["Position", "grid_position", "is_top10_start",
                     "grid_vs_team_avg", "driver_avg_quali_last5", "team_avg_quali_last5"],
            errors="ignore",
        ).columns)

    def test_typo_columns_would_not_drop(self):
        """
        Regression: the old typo names must NOT match any real column in the DataFrame.
        If they did, the bug would not have existed in the first place.
        """
        df = make_minimal_df()
        assert "driver_vag_quali_last5" not in df.columns, \
            "typo column name should not exist — this is the whole point of the bug"
        assert "team_avg_quali_last_5" not in df.columns, \
            "typo column name should not exist — this is the whole point of the bug"

    def test_correct_names_do_exist_in_dataframe(self):
        """The real column names must be present before preprocess() drops them."""
        df = make_minimal_df()
        for col in QUALI_LEAK_COLS:
            assert col in df.columns, f"Expected real column '{col}' to be present in input df"

    def test_output_width_matches_expected_drop(self):
        """
        Concrete column-count check: output must be narrower than input by exactly
        the number of columns that are dropped (Position, grid_position,
        is_top10_start, grid_vs_team_avg, driver_avg_quali_last5, team_avg_quali_last5).
        """
        df = make_minimal_df()
        cols_to_drop = {"Position", "grid_position", "is_top10_start",
                        "grid_vs_team_avg", "driver_avg_quali_last5", "team_avg_quali_last5"}
        expected_width = len(df.columns) - len(cols_to_drop)
        X, _ = top1.preprocess(df)
        assert X.shape[1] == expected_width, (
            f"Expected {expected_width} features after drop, got {X.shape[1]}. "
            "Qualifying columns may not have been dropped (BUG-05)."
        )


class TestTop3PreQualiDropsQualiCols:
    def test_output_width_matches_expected_drop(self):
        """Same column-count regression for the top3 variant."""
        df = make_minimal_df()
        cols_to_drop = {"Position", "grid_position", "is_top10_start",
                        "grid_vs_team_avg", "driver_avg_quali_last5", "team_avg_quali_last5"}
        expected_width = len(df.columns) - len(cols_to_drop)
        X, _ = top3.preprocess(df)
        assert X.shape[1] == expected_width, (
            f"Expected {expected_width} features after drop, got {X.shape[1]}. "
            "Qualifying columns may not have been dropped (BUG-05)."
        )

    def test_y_is_top3_binary(self):
        """top3 preprocess must produce a binary label where 1 means Position <= 3."""
        df = make_minimal_df()
        _, y = top3.preprocess(df)
        expected_y = (df["Position"] <= 3).astype(int)
        assert list(y) == list(expected_y)

    def test_y_top1_is_winner_binary(self):
        """top1 preprocess must produce a binary label where 1 means Position == 1."""
        df = make_minimal_df()
        _, y = top1.preprocess(df)
        expected_y = (df["Position"] == 1).astype(int)
        assert list(y) == list(expected_y)
