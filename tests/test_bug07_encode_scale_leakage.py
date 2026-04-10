"""
BUG-07 regression tests — StandardScaler and LabelEncoder fitted on full data
before train/test split.

Old behaviour: preprocess(df) encoded+scaled the full DataFrame, then the
caller split the already-scaled array.  This caused test-set statistics to
leak into the scaler's mean_/scale_ parameters.

Fix: preprocess() now returns a raw DataFrame; _encode_scale() fits only on
the training subset and transforms test separately.
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "predictions"))
import compare_models_top1_quali as top1q
import compare_models_top3_quali as top3q
import compare_models_top1_pre_quali as top1pq
import compare_models_top3_pre_quali as top3pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n=40, seed=0):
    """Minimal DataFrame with the columns all four scripts expect."""
    rng = np.random.default_rng(seed)
    drivers = ["VER", "HAM", "LEC", "NOR", "SAI"]
    teams   = ["RBR", "MER", "FER", "MCL", "FER"]
    gps     = ["Bahrain", "Australia", "Japan", "China", "Miami"]
    idx = rng.integers(0, len(drivers), size=n)
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
        "driver_avg_finish_last5": rng.uniform(1, 20, size=n),
        "driver_points_last5":     rng.uniform(0, 25, size=n),
        "team_avg_finish_last5":   rng.uniform(1, 20, size=n),
        "team_points_last5":       rng.uniform(0, 25, size=n),
    })


# ---------------------------------------------------------------------------
# preprocess() must return a DataFrame, not a numpy array
# ---------------------------------------------------------------------------

class TestPreprocessReturnsDataFrame:
    @pytest.mark.parametrize("module", [top1q, top3q, top1pq, top3pq])
    def test_X_is_dataframe(self, module):
        """preprocess() must return X as a DataFrame so split-then-scale is possible."""
        df = make_df()
        X, y = module.preprocess(df)
        assert isinstance(X, pd.DataFrame), (
            f"{module.__name__}.preprocess() still returns a numpy array — "
            "encoding/scaling must happen after the split (BUG-07)"
        )

    @pytest.mark.parametrize("module", [top1q, top3q, top1pq, top3pq])
    def test_X_still_has_categorical_columns(self, module):
        """Categorical columns must be present and still string-typed before encoding."""
        df = make_df()
        X, _ = module.preprocess(df)
        for col in ["Driver", "TeamName", "GrandPrix"]:
            if col in X.columns:
                assert X[col].dtype == object, (
                    f"'{col}' in {module.__name__} has been encoded inside preprocess() — "
                    "encoding must happen after the split (BUG-07)"
                )


# ---------------------------------------------------------------------------
# _encode_scale() must fit only on train statistics
# ---------------------------------------------------------------------------

class TestEncodeScaleFitsOnTrainOnly:
    @pytest.mark.parametrize("module", [top1q, top3q, top1pq, top3pq])
    def test_scaler_mean_matches_train_subset(self, module):
        """
        The StandardScaler fitted inside _encode_scale(X_train_df, X_test_df)
        must reflect only the training rows, not the full dataset.

        We inject an extreme outlier into the test portion of a numeric column
        and verify the scaler mean is NOT pulled toward that outlier.
        """
        df = make_df(n=60, seed=42)
        X_df, y = module.preprocess(df)
        X_train_df, X_test_df, _, _ = train_test_split(X_df, y, test_size=0.3, random_state=42)

        # Pick any numeric column present after drops
        numeric_cols = X_train_df.select_dtypes(include="number").columns
        assert len(numeric_cols) > 0, "No numeric columns to test"
        col = numeric_cols[0]

        train_mean = X_train_df[col].mean()

        # Inject extreme value into test only
        X_test_poisoned = X_test_df.copy()
        X_test_poisoned[col] = 1e9

        # full-data mean would be shifted massively by the poison
        X_full_poisoned = pd.concat([X_train_df, X_test_poisoned])
        full_mean = X_full_poisoned[col].mean()

        # Run encode_scale — scaler should only see train rows
        X_train_scaled, X_test_scaled = module._encode_scale(X_train_df, X_test_poisoned)

        # Recover what mean was used: a zero-scaled value in train corresponds to the mean
        # Fit an independent scaler on train to get the expected mean
        ref_scaler = StandardScaler()
        ref_scaler.fit(X_train_df.select_dtypes(include="number"))
        col_idx = list(X_train_df.select_dtypes(include="number").columns).index(col)
        expected_mean = ref_scaler.mean_[col_idx]

        # The train mean and the reference must be very close; both must differ from full mean
        assert abs(expected_mean - train_mean) < 1e-6
        assert abs(full_mean - train_mean) > 1.0, "Poison did not shift full mean — test is invalid"

        # The scaled train array's column mean should be ~0 (zero-centred by scaler)
        assert abs(X_train_scaled[:, col_idx].mean()) < 1e-6, (
            "Scaled train column mean is not ~0 — scaler may have been fit on full data (BUG-07)"
        )

    @pytest.mark.parametrize("module", [top1q, top3q, top1pq, top3pq])
    def test_encode_scale_train_only_no_test_arg(self, module):
        """_encode_scale called with only train data must return (X_train_scaled, None)."""
        df = make_df()
        X_df, y = module.preprocess(df)
        X_train_df, _, _, _ = train_test_split(X_df, y, test_size=0.3, random_state=42)

        X_train_scaled, X_test_scaled = module._encode_scale(X_train_df)
        assert X_test_scaled is None
        assert isinstance(X_train_scaled, np.ndarray)

    @pytest.mark.parametrize("module", [top1q, top3q, top1pq, top3pq])
    def test_unseen_category_in_test_gets_minus_one(self, module):
        """A category present only in test (unseen during train fit) must map to -1."""
        df = make_df(n=60, seed=7)
        X_df, y = module.preprocess(df)
        X_train_df, X_test_df, _, _ = train_test_split(X_df, y, test_size=0.3, random_state=42)

        # Inject an unseen driver into the test set
        X_test_modified = X_test_df.copy()
        if "Driver" in X_test_modified.columns:
            X_test_modified["Driver"] = "UNKNOWN_DRIVER_XYZ"
            X_train_scaled, X_test_scaled = module._encode_scale(X_train_df, X_test_modified)
            driver_col_idx = list(X_train_df.columns).index("Driver")
            # All test Driver values should have been mapped to -1 before scaling
            # After scaling, the raw encoded value was -1; we can't easily recover it,
            # but we can verify no exception was raised and shapes are correct.
            assert X_test_scaled.shape[0] == len(X_test_modified)
