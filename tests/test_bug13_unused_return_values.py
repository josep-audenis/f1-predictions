"""
BUG-13 regression tests — prepare_features() returns driver_names/team_names
that are never used at the call site.

Old behaviour:
  return X_scaled, df["Driver"], df["TeamName"]
  ...
  X, driver_names, team_names = prepare_features(df_race, ...)
  # driver_names and team_names assigned and never referenced again

Fix: prepare_features() returns only X_scaled; call site updated accordingly.
Note: after LabelEncoder is applied inside prepare_features, df["Driver"] and
df["TeamName"] are integer-encoded anyway — not the original string names —
making the return values doubly misleading.
"""
import sys
import ast
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "predictions"))
import predict_next_race as pnr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_race_df(n=10):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Driver":                  [f"D{i}" for i in range(n)],
        "TeamName":                ["TeamA", "TeamB"] * (n // 2),
        "GrandPrix":               ["Monaco"] * n,
        "Position":                list(range(1, n + 1)),
        "grid_position":           list(range(1, n + 1)),
        "is_top10_start":          [1] * n,
        "grid_vs_team_avg":        rng.uniform(-1, 1, n),
        "driver_avg_quali_last5":  rng.uniform(1, 10, n),
        "team_avg_quali_last5":    rng.uniform(1, 10, n),
        "driver_avg_finish_last5": rng.uniform(1, 10, n),
        "driver_points_last5":     rng.uniform(0, 25, n),
        "team_avg_finish_last5":   rng.uniform(1, 10, n),
        "team_points_last5":       rng.uniform(0, 25, n),
    })


# ---------------------------------------------------------------------------
# prepare_features() must return only X_scaled
# ---------------------------------------------------------------------------

class TestPrepareFeaturesSingleReturn:
    def test_returns_ndarray_not_tuple(self):
        """prepare_features must return a numpy array, not a tuple."""
        df = make_race_df()
        result = pnr.prepare_features(df)
        assert isinstance(result, np.ndarray), (
            f"prepare_features returned {type(result).__name__} — "
            "expected ndarray; driver_names/team_names must be removed (BUG-13)"
        )

    def test_return_value_is_2d_array(self):
        """The returned array must be 2D: (n_samples, n_features)."""
        df = make_race_df(n=10)
        X = pnr.prepare_features(df)
        assert X.ndim == 2
        assert X.shape[0] == 10

    def test_no_driver_column_in_output(self):
        """
        Old code returned df["Driver"] which, after LabelEncoder, contains
        integers not driver names — a misleading API.  The return must not
        include a driver column at all.
        """
        df = make_race_df()
        result = pnr.prepare_features(df)
        # Result should be a plain ndarray, not something with column names
        assert not hasattr(result, "columns"), (
            "prepare_features returned a DataFrame with columns — "
            "should return a plain ndarray (BUG-13)"
        )

    def test_source_has_no_driver_names_return(self):
        """
        AST check: the return statement in prepare_features must not include
        references to 'Driver' or 'TeamName' columns.
        """
        source = Path(pnr.__file__).read_text()
        tree = ast.parse(source)

        func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "prepare_features"
        )

        # Collect all string constants in return statements
        return_strings = []
        for node in ast.walk(func):
            if isinstance(node, ast.Return):
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        return_strings.append(child.value)

        assert "Driver" not in return_strings, (
            "prepare_features return statement still references 'Driver' (BUG-13)"
        )
        assert "TeamName" not in return_strings, (
            "prepare_features return statement still references 'TeamName' (BUG-13)"
        )


# ---------------------------------------------------------------------------
# Call site in predict_next_race must not unpack unused names
# ---------------------------------------------------------------------------

class TestCallSiteClean:
    def test_predict_next_race_source_has_no_driver_names(self):
        """
        driver_names and team_names must not appear as assigned variables
        in predict_next_race().
        """
        source = Path(pnr.__file__).read_text()
        tree = ast.parse(source)

        func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "predict_next_race"
        )

        assigned_names = {
            node.id
            for node in ast.walk(func)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }

        assert "driver_names" not in assigned_names, (
            "'driver_names' still assigned in predict_next_race() (BUG-13)"
        )
        assert "team_names" not in assigned_names, (
            "'team_names' still assigned in predict_next_race() (BUG-13)"
        )
