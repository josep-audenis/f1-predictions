import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_utils import load_features_pre_race
from utils.model_utils import evaluate_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = Path(__file__).resolve().parent.parent / ".." / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / ".." / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / ".." / "models" / "top1" / "pre-quali"

def preprocess(df):
    df = df.copy()
    df = df.dropna(axis=1, how='all')
    df = df.fillna(df.mean(numeric_only=True))
    
    for col in ["TeamName", "GrandPrix", "Driver"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # y_col = "ClassifiedPosition" if "ClassifiedPosition" in df.columns else "Position"
    X = df.drop(columns=["Position", "grid_position", "is_top10_start", "grid_vs_team_avg", "driver_vag_quali_last5", "team_avg_quali_last_5"], errors="ignore")
    y = df["Position"].apply(lambda x: 1 if x == 1 else 0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def main():
    all_data = load_features_pre_race(DATA_DIR)
    print(f"Found {len(all_data)} datasets: {list(all_data.keys())}")

    final_results = {}

    for tag, df in all_data.items():
        print(f"\nEvaluating dataset: {tag}")
        X, y = preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        final_results[tag] = evaluate_models(X_train, X_test, y_train, y_test, X, y, MODEL_DIR, prefix=f"{tag}_top1_pre-quali")

    with open(OUTPUT_DIR / "model_performance_top1_pre_quali.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\nResults saved to model_performance.json")

if __name__ == "__main__":
    main()
