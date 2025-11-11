import pandas as pd
import numpy as np
import os
import glob
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = Path(__file__).resolve().parent.parent / ".." / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / ".." / "data" / "processed"

def load_datasets():
    dataframes = {}
    for year in ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
        df = pd.read_csv(DATA_DIR / f"features_pre_race_{year}-2025.csv")
        dataframes[year] = df
    return dataframes

def preprocess(df):
    df = df.copy()
    df = df.dropna(axis=1, how='all')
    df = df.fillna(df.mean(numeric_only=True))
    
    for col in ["TeamName", "GrandPrix", "Driver"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # y_col = "ClassifiedPosition" if "ClassifiedPosition" in df.columns else "Position"
    X = df.drop(columns=["Position", "grid_position", "is_top10_start", "grid_vs_team_avg", "driver_vag_quali_last5", "team_avg_quali_last_5"], errors="ignore")
    y = df["Position"].apply(lambda x: 1 if x <= 3 else 0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(X)
    return X_scaled, y

def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
        "SVC": SVC(probability=True),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        f1_weighted = f1_score(y_test, preds, average="weighted")
        # top3 = top_k_accuracy_score(y_test, probs, k=3) if probs is not None else None

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            # "top3_accuracy": top3
        }
        print(f"{name} done: acc={acc:.3f}, f1_macro={f1_macro:.3f}") # , top3={top3:.3f if top3 else 0.0}
    return results

def main():
    all_data = load_datasets()
    print(f"Found {len(all_data)} datasets: {list(all_data.keys())}")

    if "2025" not in all_data:
        raise ValueError("No dataset named 'features_pre_race_2025.csv' found.")
    X_2025, y_2025 = preprocess(all_data["2025"])

    final_results = {}

    for tag, df in all_data.items():
        print(f"\nEvaluating dataset: {tag}")
        if tag == "2025":
            X, y = preprocess(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            final_results[tag] = evaluate_models(X_train, X_test, y_train, y_test)
        else:
            X_train, y_train = preprocess(df)
            X_test, y_test = X_2025, y_2025
            final_results[tag] = evaluate_models(X_train, X_test, y_train, y_test)

    with open(OUTPUT_DIR / "model_performance_top3_pre_quali.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\nResults saved to model_performance.json")

if __name__ == "__main__":
    main()
