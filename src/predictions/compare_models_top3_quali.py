import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

from utils.data_utils import load_features_pre_race
from utils.model_utils import evaluate_models

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
    """Fit LabelEncoder and StandardScaler on train; transform train and optionally test."""
    cat_cols = ["TeamName", "GrandPrix", "Driver"]
    X_train = X_train_df.copy()
    X_test = X_test_df.copy() if X_test_df is not None else None

    for col in cat_cols:
        if col not in X_train.columns:
            continue
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))
        X_train[col] = le.transform(X_train[col].astype(str))
        if X_test is not None and col in X_test.columns:
            known = set(le.classes_)
            X_test[col] = [le.transform([v])[0] if v in known else -1
                           for v in X_test[col].astype(str)]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled


def main():
    all_data = load_features_pre_race(DATA_DIR)
    print(f"Found {len(all_data)} datasets: {list(all_data.keys())}")

    final_results = {}

    for tag, df in all_data.items():
        print(f"\nEvaluating dataset: {tag}")
        X_df, y = preprocess(df)
        X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

        # Impute with train mean only to avoid leaking test statistics
        train_mean = X_train_df.mean(numeric_only=True)
        X_train_df = X_train_df.fillna(train_mean)
        X_test_df = X_test_df.fillna(train_mean)

        X_train, X_test = _encode_scale(X_train_df, X_test_df)
        X_full, _ = _encode_scale(X_df.fillna(X_df.mean(numeric_only=True)))  # fit on all data for the saved model

        final_results[tag] = evaluate_models(X_train, X_test, y_train, y_test, X_full, y, MODEL_DIR, prefix=f"{tag}_top3_quali")

    with open(OUTPUT_DIR / "model_performance_top3_quali.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\nResults saved to model_performance.json")

if __name__ == "__main__":
    main()
