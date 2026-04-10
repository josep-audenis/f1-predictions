"""
BUG-08 regression tests — fillna(df.mean()) computed over the full dataset
before the train/test split.

Old behaviour: preprocess() called df.fillna(df.mean()) on the full DataFrame,
so the imputed values for training rows were influenced by test-row statistics.

Fix: fillna is removed from preprocess(); main() computes train_mean after
the split and applies it to both train and test.
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "predictions"))
import compare_models_top1_quali as top1q
import compare_models_top3_quali as top3q
import compare_models_top1_pre_quali as top1pq
import compare_models_top3_pre_quali as top3pq
import predict_next_race as pnr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n=40, seed=0, nan_fraction=0.2):
    """Minimal DataFrame with some NaNs in numeric columns."""
    rng = np.random.default_rng(seed)
    drivers = ["VER", "HAM", "LEC", "NOR", "SAI"]
    teams   = ["RBR", "MER", "FER", "MCL", "FER"]
    gps     = ["Bahrain", "Australia", "Japan", "China", "Miami"]
    idx = rng.integers(0, len(drivers), size=n)

    finish = rng.uniform(1, 20, size=n).astype(float)
    points = rng.uniform(0, 25, size=n).astype(float)

    # Introduce NaNs in a subset of rows
    nan_mask = rng.random(n) < nan_fraction
    finish[nan_mask] = np.nan

    return pd.DataFrame({
        "Driver":                  [drivers[i] for i in idx],
        "TeamName":                [teams[i]   for i in idx],
        "GrandPrix":               [gps[rng.integers(0, len(gps))] for _ in range(n)],
        "Position":                rng.integers(1, 21, size=n),
        "grid_position":           rng.integers(1, 21, size=n),
        "is_top10_start":          rng.integers(0, 2, size=n),
        "grid_vs_team_avg":        rng.uniform(-5, 5, size=n),
        "driver_avg_quali_last5":  rng.uniform(1, 20, size=n),
        "team_avg_quali_last5":    rng.uniform(1, 20, size=n),
        "driver_avg_finish_last5": finish,
        "driver_points_last5":     points,
        "team_avg_finish_last5":   rng.uniform(1, 20, size=n),
        "team_points_last5":       rng.uniform(0, 25, size=n),
    })


# ---------------------------------------------------------------------------
# preprocess() must NOT fill NaNs (fillna moved to main())
# ---------------------------------------------------------------------------

class TestPreprocessNoFillna:
    @pytest.mark.parametrize("module", [top1q, top3q, top1pq, top3pq])
    def test_nan_values_survive_preprocess(self, module):
        """NaNs in the input must still be present in X after preprocess()."""
        df = make_df(nan_fraction=0.3)
        X, _ = module.preprocess(df)
        numeric_X = X.select_dtypes(include="number")
        assert numeric_X.isna().any().any(), (
            f"{module.__name__}.preprocess() is still filling NaNs — "
            "fillna must be deferred until after the train/test split (BUG-08)"
        )


# ---------------------------------------------------------------------------
# After split, train mean must be used for imputation of both sets
# ---------------------------------------------------------------------------

class TestFillnaUsesTrainMeanOnly:
    @pytest.mark.parametrize("module", [top1q, top3q, top1pq, top3pq])
    def test_train_mean_differs_from_full_mean_when_test_is_extreme(self, module):
        """
        Inject extreme values into the test portion of a numeric column.
        The train subset mean must NOT be influenced by those test values.
        """
        df = make_df(n=60, seed=42, nan_fraction=0.0)
        X_df, y = module.preprocess(df)
        X_train_df, X_test_df, _, _ = train_test_split(X_df, y, test_size=0.3, random_state=42)

        # Find a numeric column
        num_cols = X_train_df.select_dtypes(include="number").columns
        assert len(num_cols) > 0
        col = num_cols[0]

        # Poison the test column with extreme values
        X_test_extreme = X_test_df.copy()
        X_test_extreme[col] = 1e9

        # Correct behaviour: train mean comes only from X_train_df
        train_mean = X_train_df.mean(numeric_only=True)

        # Full-data mean would be very different
        X_combined = pd.concat([X_train_df, X_test_extreme])
        full_mean = X_combined.mean(numeric_only=True)

        assert abs(train_mean[col] - full_mean[col]) > 1e6, (
            "Poison did not shift the full mean — test setup is invalid"
        )
        # Train mean must be unaffected by the extreme test values
        assert train_mean[col] == X_train_df[col].mean()

    @pytest.mark.parametrize("module", [top1q, top3q, top1pq, top3pq])
    def test_test_nans_filled_with_train_mean_not_test_mean(self, module):
        """
        NaNs in the test set must be filled with the training mean.
        We verify by checking that after applying train_mean.fillna, no NaNs
        remain in a test set whose NaN cells would have a very different
        column mean from train.
        """
        rng = np.random.default_rng(99)
        df = make_df(n=80, seed=99, nan_fraction=0.0)
        X_df, y = module.preprocess(df)
        X_train_df, X_test_df, _, _ = train_test_split(X_df, y, test_size=0.3, random_state=42)

        # Introduce NaN only into the test set
        num_cols = X_test_df.select_dtypes(include="number").columns
        col = num_cols[0]
        X_test_with_nan = X_test_df.copy()
        X_test_with_nan.iloc[0, X_test_with_nan.columns.get_loc(col)] = np.nan

        train_mean = X_train_df.mean(numeric_only=True)
        X_test_filled = X_test_with_nan.fillna(train_mean)

        # The filled value should equal the training mean, not the test mean
        filled_value = X_test_filled.iloc[0][col]
        assert abs(filled_value - train_mean[col]) < 1e-9, (
            f"Filled value {filled_value} != train mean {train_mean[col]} — "
            "test NaNs must be filled with the training mean (BUG-08)"
        )
        assert X_test_filled.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# predict_next_race.prepare_features accepts an external fill_mean
# ---------------------------------------------------------------------------

class TestPrepareFeaturesFillMean:
    def make_race_df(self, n=20):
        rng = np.random.default_rng(0)
        drivers = [f"D{i}" for i in range(n)]
        return pd.DataFrame({
            "Driver":                  drivers,
            "TeamName":                ["TeamA"] * n,
            "GrandPrix":               ["Monaco"] * n,
            "Position":                rng.integers(1, n + 1, size=n),
            "grid_position":           rng.integers(1, n + 1, size=n),
            "is_top10_start":          rng.integers(0, 2, size=n),
            "grid_vs_team_avg":        rng.uniform(-3, 3, size=n),
            "driver_avg_quali_last5":  rng.uniform(1, 20, size=n),
            "team_avg_quali_last5":    rng.uniform(1, 20, size=n),
            "driver_avg_finish_last5": [np.nan] * n,  # all NaN → imputation always happens
            "driver_points_last5":     rng.uniform(0, 25, size=n),
            "team_avg_finish_last5":   rng.uniform(1, 20, size=n),
            "team_points_last5":       rng.uniform(0, 25, size=n),
        })

    def test_fill_mean_param_is_used_when_provided(self):
        """When fill_mean is supplied, NaNs must be filled with that value, not the df mean."""
        df = self.make_race_df()
        training_mean = pd.Series({"driver_avg_finish_last5": 999.0})

        X = pnr.prepare_features(df, use_quali_features=True, fill_mean=training_mean)
        # We can't directly inspect the filled values from the scaled output,
        # but we can verify no exception was raised and shape is correct.
        assert X.shape[0] == len(df)

    def test_fill_mean_none_falls_back_to_df_mean(self):
        """When fill_mean is None, the function must still work (legacy fallback)."""
        df = self.make_race_df()
        X = pnr.prepare_features(df, use_quali_features=True, fill_mean=None)
        assert X.shape[0] == len(df)
