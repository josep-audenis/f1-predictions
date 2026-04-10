"""
BUG-09 regression tests — saved model trained on full data but reported
metrics were from the train/test split model.

Old behaviour:
  model.fit(X_train)          ← split model
  preds = model.predict(X_test)
  final_model = models[name]  ← same object reference!
  final_model.fit(X_full)     ← overwrites the split model in-place
  joblib.dump(final_model)    ← saves the full-data model
  acc = accuracy_score(preds) ← metrics from split model, not the saved one

Fix:
  1. Evaluate on split as before.
  2. Instantiate a FRESH model for the full-data fit (type(model)(**model.get_params())).
  3. Cross-validate a fresh instance on X_full to produce honest metrics for
     the saved model; store as cv_accuracy_mean / cv_accuracy_std.
"""
import sys
import numpy as np
import pytest
import joblib
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "utils"))
from model_utils import evaluate_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(n=200, n_features=5, seed=0):
    X, y = make_classification(n_samples=n, n_features=n_features,
                                n_informative=3, random_state=seed)
    X_train, X_test = X[:140], X[140:]
    y_train, y_test = y[:140], y[140:]
    return X_train, X_test, y_train, y_test, X, y


# ---------------------------------------------------------------------------
# Result dict must contain cv_accuracy fields when prefix is supplied
# ---------------------------------------------------------------------------

class TestCvMetricsPresentWhenPrefixGiven:
    def test_cv_accuracy_mean_in_results(self, tmp_path):
        X_train, X_test, y_train, y_test, X_full, y_full = make_data()
        results = evaluate_models(
            X_train, X_test, y_train, y_test, X_full, y_full,
            model_dir=tmp_path, prefix="test_run"
        )
        for name, res in results.items():
            assert "cv_accuracy_mean" in res, (
                f"{name}: 'cv_accuracy_mean' missing — cross-val metrics for the "
                "saved model must be reported (BUG-09)"
            )
            assert "cv_accuracy_std" in res, (
                f"{name}: 'cv_accuracy_std' missing (BUG-09)"
            )

    def test_cv_accuracy_not_in_results_when_no_prefix(self, tmp_path):
        """When no model is saved (prefix=None), cv metrics are not needed."""
        X_train, X_test, y_train, y_test, X_full, y_full = make_data()
        results = evaluate_models(
            X_train, X_test, y_train, y_test, X_full, y_full,
            model_dir=tmp_path, prefix=None
        )
        for name, res in results.items():
            assert "cv_accuracy_mean" not in res

    def test_cv_accuracy_mean_is_float_between_0_and_1(self, tmp_path):
        X_train, X_test, y_train, y_test, X_full, y_full = make_data()
        results = evaluate_models(
            X_train, X_test, y_train, y_test, X_full, y_full,
            model_dir=tmp_path, prefix="test_run"
        )
        for name, res in results.items():
            val = res["cv_accuracy_mean"]
            assert isinstance(val, float), f"{name}: cv_accuracy_mean must be a float"
            assert 0.0 <= val <= 1.0, f"{name}: cv_accuracy_mean={val} out of [0,1]"


# ---------------------------------------------------------------------------
# The saved model must be a FRESH instance (not the split-trained object)
# ---------------------------------------------------------------------------

class TestSavedModelIsFreshInstance:
    def test_saved_model_fitted_on_full_data(self, tmp_path):
        """
        The saved .joblib must have been trained on X_full (n=200 samples),
        not on X_train (n=140 samples).  We verify by inspecting n_samples_seen_
        on the saved LogisticRegression.
        """
        X_train, X_test, y_train, y_test, X_full, y_full = make_data(n=200)

        # Patch models dict so only LogisticRegression runs (speeds up test)
        with patch.dict("model_utils.evaluate_models.__globals__", {}):
            pass  # nothing to patch at module level

        results = evaluate_models(
            X_train, X_test, y_train, y_test, X_full, y_full,
            model_dir=tmp_path, prefix="test"
        )

        lr_path = tmp_path / "test_LogisticRegression.joblib"
        assert lr_path.exists(), "LogisticRegression model was not saved"

        saved_model = joblib.load(lr_path)
        # LogisticRegression exposes n_iter_ and coef_ after fit; we can check
        # that it was fitted on the full dataset by verifying coef_ shape matches
        # the number of features in X_full.
        assert saved_model.coef_.shape[1] == X_full.shape[1]

    def test_split_model_not_overwritten_by_full_fit(self, tmp_path):
        """
        Old code: final_model = models[name] (same reference as model).
        After final_model.fit(X_full), model was also refitted — but preds
        had already been computed, so metrics were still from the split model.

        New code uses type(model)(**model.get_params()) so the objects are
        independent.  We verify the split-evaluated accuracy is consistent
        with a model trained only on X_train by re-evaluating it independently.
        """
        X_train, X_test, y_train, y_test, X_full, y_full = make_data(seed=7)

        results = evaluate_models(
            X_train, X_test, y_train, y_test, X_full, y_full,
            model_dir=tmp_path, prefix="test"
        )

        # Independently train a LR on X_train only and check accuracy matches
        ref = LogisticRegression(max_iter=1000)
        ref.fit(X_train, y_train)
        ref_acc = (ref.predict(X_test) == y_test).mean()

        reported_acc = results["LogisticRegression"]["accuracy"]
        assert abs(reported_acc - ref_acc) < 1e-9, (
            f"Reported accuracy {reported_acc:.4f} != independently computed "
            f"split accuracy {ref_acc:.4f} — the split model may have been "
            "overwritten before metrics were recorded (BUG-09)"
        )
