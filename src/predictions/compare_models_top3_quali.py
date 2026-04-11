import json
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

from utils.data_utils import load_features_pre_race
from utils.model_utils import evaluate_models

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / ".." / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / ".." / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / ".." / "models" / "top3" / "quali"

def preprocess(df):
    """Return raw feature DataFrame and target; no imputation, encoding, or scaling applied."""
    df = df.copy()
    df = df.dropna(axis=1, how='all')

    # y_col = "ClassifiedPosition" if "ClassifiedPosition" in df.columns else "Position"
    X = df.drop(columns=["Position"], errors="ignore")
    y = df["Position"].apply(lambda x: 1 if x <= 3 else 0)
    return X, y


def _encode_scale(X_train_df, X_test_df=None):
    """Fit LabelEncoder and StandardScaler on train; transform train and optionally test.

    Returns (X_train_scaled, X_test_scaled, scaler, encoders) where encoders includes
    a "__columns__" key with the ordered column list used for alignment at inference time.
    """
    cat_cols = ["TeamName", "GrandPrix", "Driver"]
    X_train = X_train_df.copy()
    X_test = X_test_df.copy() if X_test_df is not None else None
    encoders = {}

    for col in cat_cols:
        if col not in X_train.columns:
            continue
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))
        X_train[col] = le.transform(X_train[col].astype(str))
        encoders[col] = le
        if X_test is not None and col in X_test.columns:
            known = set(le.classes_)
            X_test[col] = [le.transform([v])[0] if v in known else -1
                           for v in X_test[col].astype(str)]

    encoders["__columns__"] = list(X_train.columns)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler, encoders


def _fit_full_artifacts(X_df):
    """Fit LabelEncoder and StandardScaler on full data for inference-time saving.

    Returns (X_scaled, scaler, encoders) where encoders includes "__columns__".
    """
    cat_cols = ["TeamName", "GrandPrix", "Driver"]
    X = X_df.copy()
    encoders = {}

    for col in cat_cols:
        if col not in X.columns:
            continue
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        X[col] = le.transform(X[col].astype(str))
        encoders[col] = le

    encoders["__columns__"] = list(X.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, encoders


def main():
    all_data = load_features_pre_race(DATA_DIR)
    logger.info("Found %d datasets: %s", len(all_data), list(all_data.keys()))

    final_results = {}

    for tag, df in all_data.items():
        logger.info("Evaluating dataset: %s", tag)
        X_df, y = preprocess(df)
        X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

        # Impute with train mean only to avoid leaking test statistics
        train_mean = X_train_df.mean(numeric_only=True)
        X_train_df = X_train_df.fillna(train_mean)
        X_test_df = X_test_df.fillna(train_mean)

        X_train, X_test, _, _ = _encode_scale(X_train_df, X_test_df)
        full_mean = X_df.mean(numeric_only=True)
        X_full, full_scaler, full_encoders = _fit_full_artifacts(X_df.fillna(full_mean))

        prefix = f"{tag}_top3_quali"
        joblib.dump(full_scaler, MODEL_DIR / f"{prefix}_scaler.joblib")
        joblib.dump(full_encoders, MODEL_DIR / f"{prefix}_encoders.joblib")
        full_mean.to_json(MODEL_DIR / f"{prefix}_train_mean.json")

        final_results[tag] = evaluate_models(X_train, X_test, y_train, y_test, X_full, y, MODEL_DIR, prefix=prefix)

    output_file = OUTPUT_DIR / "model_performance_top3_quali.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=4)

    logger.info("Results saved to %s", output_file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    main()
