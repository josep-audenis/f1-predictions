import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

from utils.calendar_utils import get_next_race


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / ".." / "data" / "processed"
RAW_DIR = BASE_DIR / ".." / "data" / "raw"


def load_features_for_race(year):
    return pd.read_csv(DATA_DIR / f"features_pre_race_{year}-{year}.csv")


def prepare_features(df, use_quali_features=True, fill_mean=None):
    """Prepare features for inference.

    fill_mean: optional Series of column means from the training set.  When
    provided it is used for NaN imputation instead of the race-day mean,
    preventing the imputed value from depending on other drivers in the same
    race.  If None, falls back to the mean of the supplied DataFrame (legacy
    behaviour; ideally the training mean should be saved alongside the model).
    """
    df = df.copy()
    impute_mean = fill_mean if fill_mean is not None else df.mean(numeric_only=True)
    df = df.fillna(impute_mean)

    cat_cols = ["TeamName", "GrandPrix", "Driver"]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.drop(columns=["Position"], errors="ignore")

    if not use_quali_features:
        df = df.drop(
            columns=[
                "grid_position",
                "is_top10_start",
                "grid_vs_team_avg",
                "driver_avg_quali_last5",
                "team_avg_quali_last5",
            ],
            errors="ignore"
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    return X_scaled


def load_best_model(quali):
    if quali:
        perf_path = DATA_DIR / "model_performance_top1_quali.json"
        model_dir = BASE_DIR / ".." / "models" / "top1" / "quali"
        task_prefix = "top1_quali"
    else:
        perf_path = DATA_DIR / "model_performance_top1_pre_quali.json"
        model_dir = BASE_DIR / ".." / "models" / "top1" / "pre-quali"
        task_prefix = "top1_pre-quali"

    with open(perf_path, "r") as f:
        perf = json.load(f)

    best_year = max(perf.keys())
    best_model_name = max(perf[best_year], key=lambda m: perf[best_year][m]["accuracy"])

    model_path = model_dir / f"{best_year}_{task_prefix}_{best_model_name}.joblib"
    return joblib.load(model_path)


def predict_next_race():
    event = get_next_race()
    if event is None:
        raise ValueError("No upcoming race found for the current season.")

    year = event["year"]
    gp = event["grand_prix"]
    is_quali_done = event["has_quali"]

    print(f"➡ Next race: {gp} ({event['race_datetime']})")
    print(f"➡ Qualifying already done? {is_quali_done}")

    df = load_features_for_race(year)

    df_race = df[df["GrandPrix"] == gp]
    if df_race.empty:
        raise ValueError(f"No features for GP {gp}")

    X = prepare_features(df_race, use_quali_features=is_quali_done)

    model = load_best_model(is_quali_done)
    probs = model.predict_proba(X)[:, 1]

    df_race["win_prob"] = probs
    df_race_sorted = df_race.sort_values("win_prob", ascending=False)

    top3 = df_race_sorted.head(3)[["Driver", "TeamName", "win_prob"]]
    winner = df_race_sorted.head(1)[["Driver", "TeamName", "win_prob"]]

    print("\n🏆 Predicted Winner:")
    print(winner)

    print("\n🥇🥈🥉 Predicted Podium:")
    print(top3)

    return top3, winner


if __name__ == "__main__":
    predict_next_race()
