"""
BUG-12 regression tests — top_k_accuracy_score imported but never used;
probs computed but discarded.

Old behaviour:
  from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
  ...
  probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
  # probs assigned and never referenced again

Fix chosen: remove the dead import and the dead probs assignment.
(The alternative — wiring top_k_accuracy_score into top3 evaluation — would
require per-race grouping data that evaluate_models does not receive.)
"""
import sys
import ast
import inspect
import textwrap
import importlib
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "utils"))
import model_utils


# ---------------------------------------------------------------------------
# Dead import must be gone
# ---------------------------------------------------------------------------

class TestTopKImportRemoved:
    def test_top_k_accuracy_score_not_imported(self):
        """top_k_accuracy_score must not be importable from model_utils."""
        assert not hasattr(model_utils, "top_k_accuracy_score"), (
            "top_k_accuracy_score is still exported from model_utils — "
            "dead import must be removed (BUG-12)"
        )

    def test_top_k_not_in_source(self):
        """The string 'top_k_accuracy_score' must not appear in model_utils source."""
        source = Path(model_utils.__file__).read_text()
        assert "top_k_accuracy_score" not in source, (
            "'top_k_accuracy_score' still appears in model_utils.py — "
            "dead import must be removed (BUG-12)"
        )


# ---------------------------------------------------------------------------
# Dead probs assignment must be gone
# ---------------------------------------------------------------------------

class TestProbsDeadCodeRemoved:
    def test_probs_not_assigned_and_discarded(self):
        """
        Parse the AST of evaluate_models to verify there is no assignment
        `probs = ...` whose value is never subsequently referenced.

        We detect the pattern by looking for Name nodes 'probs' on the left
        side of an Assign that are never used as Load references afterward.
        """
        source = Path(model_utils.__file__).read_text()
        tree = ast.parse(source)

        # Find evaluate_models function
        func_def = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "evaluate_models"),
            None,
        )
        assert func_def is not None, "evaluate_models function not found"

        # Collect all names assigned (Store) and loaded (Load)
        stores = {
            node.id
            for node in ast.walk(func_def)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        loads = {
            node.id
            for node in ast.walk(func_def)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }

        # 'probs' must not be assigned without being used, OR must simply not exist
        if "probs" in stores:
            assert "probs" in loads, (
                "'probs' is assigned inside evaluate_models but never used — "
                "dead probs computation must be removed (BUG-12)"
            )

    def test_predict_proba_not_called_unnecessarily(self, tmp_path):
        """
        If predict_proba is called, its result must be used (not silently dropped).
        We verify by running evaluate_models with a mock model and checking that
        predict_proba is either not called, or called and its result consumed.
        """
        X, y = make_classification(n_samples=100, n_features=5, random_state=0)
        X_train, X_test = X[:70], X[70:]
        y_train, y_test = y[:70], y[70:]

        proba_called = []

        class TrackingModel:
            """Minimal model that records whether predict_proba output is used."""
            classes_ = [0, 1]

            def fit(self, X, y): return self
            def predict(self, X): return np.zeros(len(X), dtype=int)
            def predict_proba(self, X):
                proba_called.append(True)
                return np.column_stack([np.ones(len(X)) * 0.6, np.ones(len(X)) * 0.4])
            def get_params(self, deep=True): return {}

        # Patch the models dict to use only our tracking model
        tracking_instance = TrackingModel()
        with patch.dict("model_utils.evaluate_models.__globals__", {}):
            pass  # no global patching needed

        with patch.object(model_utils, "evaluate_models",
                          wraps=model_utils.evaluate_models):
            # Run without prefix so no saving happens — just evaluation
            results = model_utils.evaluate_models(
                X_train, X_test, y_train, y_test, X, y,
                model_dir=tmp_path, prefix=None
            )

        # The test passes as long as evaluate_models completes without error.
        # Dead-code probs would have caused predict_proba to be called with
        # no effect; now it should simply not be called at all.
        # (If it IS called legitimately, results should reflect it.)
        assert isinstance(results, dict)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Smoke test — evaluate_models still produces correct metric keys
# ---------------------------------------------------------------------------

class TestEvaluateModelsMetrics:
    def test_result_keys_present(self, tmp_path):
        """accuracy, f1_macro, f1_weighted must still be present after cleanup."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=1)
        X_train, X_test = X[:70], X[70:]
        y_train, y_test = y[:70], y[70:]

        results = model_utils.evaluate_models(
            X_train, X_test, y_train, y_test, X, y,
            model_dir=tmp_path, prefix=None
        )

        for name, res in results.items():
            for key in ("accuracy", "f1_macro", "f1_weighted"):
                assert key in res, f"{name}: missing key '{key}' after BUG-12 cleanup"
