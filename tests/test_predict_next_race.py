import sys
import json
import pytest
import joblib
from pathlib import Path
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "predictions"))
import predict_next_race as pnr


# ---------------------------------------------------------------------------
# load_best_model — BUG-04 regression tests
#
# model_utils.py saves: {model_dir}/{tag}_{task_prefix}_{ModelName}.joblib
# Old code loaded:       {DATA_DIR}/{ModelName}_model.pkl
# Three bugs: wrong directory, wrong filename pattern, wrong extension.
#
# Path note: in load_best_model, model_dir = BASE_DIR / ".." / "models" / ...
# So we set BASE_DIR = tmp_path / "src" so BASE_DIR / ".." == tmp_path,
# and models live at tmp_path / "models" / ...
# ---------------------------------------------------------------------------

def _write_perf_json(path: Path, tag: str, model_name: str, accuracy: float = 0.9):
    path.write_text(json.dumps({tag: {model_name: {"accuracy": accuracy}}}))


def _write_dummy_model(path: Path):
    """Persist a minimal sklearn model so joblib.load works."""
    model = LogisticRegression()
    model.fit([[0], [1]], [0, 1])
    joblib.dump(model, path)


def _setup(tmp_path):
    """Return (data_dir, fake_base_dir) with the expected directory structure."""
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    fake_base = tmp_path / "src"   # BASE_DIR / ".." resolves to tmp_path
    fake_base.mkdir(parents=True)
    return data_dir, fake_base


class TestLoadBestModel:
    def test_quali_loads_from_correct_directory(self, tmp_path):
        """quali=True must load from models/top1/quali/, not data/processed/."""
        data_dir, fake_base = _setup(tmp_path)
        model_dir = tmp_path / "models" / "top1" / "quali"
        model_dir.mkdir(parents=True)

        _write_perf_json(data_dir / "model_performance_top1_quali.json", "2018-2025", "RandomForest")
        _write_dummy_model(model_dir / "2018-2025_top1_quali_RandomForest.joblib")

        with patch.object(pnr, "DATA_DIR", data_dir), patch.object(pnr, "BASE_DIR", fake_base):
            model = pnr.load_best_model(quali=True)
        assert model is not None

    def test_pre_quali_loads_from_correct_directory(self, tmp_path):
        """quali=False must load from models/top1/pre-quali/."""
        data_dir, fake_base = _setup(tmp_path)
        model_dir = tmp_path / "models" / "top1" / "pre-quali"
        model_dir.mkdir(parents=True)

        _write_perf_json(data_dir / "model_performance_top1_pre_quali.json", "2018-2025", "XGBoost")
        _write_dummy_model(model_dir / "2018-2025_top1_pre-quali_XGBoost.joblib")

        with patch.object(pnr, "DATA_DIR", data_dir), patch.object(pnr, "BASE_DIR", fake_base):
            model = pnr.load_best_model(quali=False)
        assert model is not None

    def test_uses_joblib_extension_not_pkl(self, tmp_path):
        """Model file must have .joblib extension — .pkl must not be loaded."""
        data_dir, fake_base = _setup(tmp_path)
        model_dir = tmp_path / "models" / "top1" / "quali"
        model_dir.mkdir(parents=True)

        _write_perf_json(data_dir / "model_performance_top1_quali.json", "2018-2025", "RandomForest")
        # Write only a .pkl at the correct location — must NOT be found
        _write_dummy_model(model_dir / "2018-2025_top1_quali_RandomForest.pkl")

        with patch.object(pnr, "DATA_DIR", data_dir), patch.object(pnr, "BASE_DIR", fake_base):
            with pytest.raises(FileNotFoundError):
                pnr.load_best_model(quali=True)

    def test_picks_model_with_highest_accuracy(self, tmp_path):
        """Best model is selected by highest accuracy value."""
        data_dir, fake_base = _setup(tmp_path)
        model_dir = tmp_path / "models" / "top1" / "quali"
        model_dir.mkdir(parents=True)

        perf = {"2018-2025": {
            "RandomForest": {"accuracy": 0.75},
            "XGBoost":      {"accuracy": 0.92},
            "SVC":          {"accuracy": 0.80},
        }}
        (data_dir / "model_performance_top1_quali.json").write_text(json.dumps(perf))
        # Only write the expected winner — if the wrong model is picked, FileNotFoundError
        _write_dummy_model(model_dir / "2018-2025_top1_quali_XGBoost.joblib")

        with patch.object(pnr, "DATA_DIR", data_dir), patch.object(pnr, "BASE_DIR", fake_base):
            model = pnr.load_best_model(quali=True)
        assert model is not None

    def test_old_bug_pkl_pattern_would_fail(self, tmp_path):
        """
        Regression for BUG-04.

        The old code built the path as DATA_DIR / '{name}_model.pkl'.
        Place a file at that old (wrong) location only — the fix must not find it.
        """
        data_dir, fake_base = _setup(tmp_path)
        (tmp_path / "models" / "top1" / "quali").mkdir(parents=True)

        _write_perf_json(data_dir / "model_performance_top1_quali.json", "2018-2025", "RandomForest")
        # Place model at the OLD wrong location only
        _write_dummy_model(data_dir / "RandomForest_model.pkl")

        with patch.object(pnr, "DATA_DIR", data_dir), patch.object(pnr, "BASE_DIR", fake_base):
            with pytest.raises(FileNotFoundError):
                pnr.load_best_model(quali=True)
