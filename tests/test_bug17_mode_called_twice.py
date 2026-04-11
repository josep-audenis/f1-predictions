"""
BUG-17 regression tests — .mode() called twice in extract_lap_features.

Old behaviour:
  lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
  .mode() is computed twice — once for the len() guard and once for .iloc[0].

Fix: walrus operator stores the result once:
  lambda x: (m := x.mode()).iloc[0] if len(m) > 0 else None
"""
import sys
import ast
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, call

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "features"))
import build_prescriptive_features as bpf


# ---------------------------------------------------------------------------
# AST check: mode() must be called at most once in the Compound lambda
# ---------------------------------------------------------------------------

class TestModeCalledOnce:
    def test_source_has_no_double_mode_call(self):
        """
        The Compound lambda in extract_lap_features must not call .mode() twice.
        After the fix the walrus operator caches the result, so there is only
        one .mode() call in that expression.
        """
        source = Path(bpf.__file__).read_text()
        tree = ast.parse(source)

        func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "extract_lap_features"
        )

        # Collect all attribute-access calls named "mode" within the function
        mode_calls = [
            node for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "mode"
        ]

        assert len(mode_calls) <= 1, (
            f"extract_lap_features contains {len(mode_calls)} .mode() calls — "
            "must be at most 1 after the BUG-17 fix (use walrus operator or variable)"
        )


# ---------------------------------------------------------------------------
# Behavioural tests: extract_lap_features returns correct dominant_compound
# ---------------------------------------------------------------------------

def make_laps(compound_seq, driver="VER", year=2024, gp="Monaco"):
    n = len(compound_seq)
    return pd.DataFrame({
        "Year":       [year] * n,
        "GrandPrix":  [gp] * n,
        "Driver":     [driver] * n,
        "LapTime":    [90.0] * n,
        "SpeedI1":    [200.0] * n,
        "SpeedFL":    [210.0] * n,
        "IsAccurate": [1.0] * n,
        "Compound":   compound_seq,
        "Stint":      list(range(1, n + 1)),
        "LapNumber":  list(range(1, n + 1)),
    })


class TestExtractLapFeaturesCompound:
    def test_dominant_compound_majority(self):
        """dominant_compound must be the most frequent tyre."""
        laps = make_laps(["SOFT"] * 6 + ["HARD"] * 4)
        features = bpf.extract_lap_features(laps)
        assert features.iloc[0]["dominant_compound"] == "SOFT"

    def test_dominant_compound_single_value(self):
        """When only one compound is used it must be returned."""
        laps = make_laps(["MEDIUM"] * 5)
        features = bpf.extract_lap_features(laps)
        assert features.iloc[0]["dominant_compound"] == "MEDIUM"

    def test_dominant_compound_all_nan_returns_none(self):
        """When all compound values are NaN, dominant_compound must be None."""
        laps = make_laps([np.nan] * 5)
        features = bpf.extract_lap_features(laps)
        val = features.iloc[0]["dominant_compound"]
        assert val is None or (isinstance(val, float) and np.isnan(val)), (
            f"Expected None/NaN for all-NaN compound, got {val!r}"
        )
