import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / ".." / "data" / "processed"
RAW_DIR = BASE_DIR / ".." / "data" / "raw"

RACE_CALENDAR = {
    "Bahrain GP": {"race": "2025-03-15", "quali": "2025-03-14"},
    "Saudi Arabia GP": {"race": "2025-03-28", "quali": "2025-03-27"},
    "Australian GP": {"race": "2025-04-12", "quali": "2025-04-11"},
    "Japanese GP": {"race": "2025-04-19", "quali": "2025-04-18"},
    # Add full season later
}


def next_race():    # TODO: get from fastf1 api
    today = datetime.today().date()
    future_races = []

    for gp, dates in RACE_CALENDAR.items():
        race_day = datetime.fromisoformat(dates["race"]).date()
        if race_day >= today:
            future_races.append((gp, race_day, dates["quali"]))

    future_races.sort(key=lambda x: x[1])
    return future_races[0]


def quali_done(quali_date_str):
    today = datetime.today().date()
    quali_date = datetime.fromisoformat(quali_date_str).date()
    return today >= quali_date


def load_features_for_race(year="2025"):
    return pd.read_csv(DATA_DIR / f"features_pre_race_{year}-2025.csv")


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

    return X_scaled, df["Driver"], df["TeamName"]


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
    gp, race_day, quali_day = next_race()
    print(f"➡ Next race: {gp} ({race_day})")

    is_quali_done = quali_done(quali_day)
    print(f"➡ Qualifying already done? {is_quali_done}")

    df = load_features_for_race("2025")

    df_race = df[df["GrandPrix"] == gp]
    if df_race.empty:
        raise ValueError(f"No features for GP {gp}")

    X, driver_names, team_names = prepare_features(df_race, use_quali_features=is_quali_done)

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
