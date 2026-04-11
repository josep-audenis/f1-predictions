import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

from utils.calendar_utils import get_next_race
from predictions.accuracy_tracker import log_prediction

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / ".." / "data" / "processed"
RAW_DIR = BASE_DIR / ".." / "data" / "raw"


def load_features_for_race(year):
    return pd.read_csv(DATA_DIR / f"features_pre_race_{year}-{year}.csv")


def prepare_features(df, use_quali_features=True, fill_mean=None, scaler=None, encoders=None):
    """Prepare features for inference using pre-fitted preprocessing artifacts.

    fill_mean: Series of column means from the training set used for NaN imputation.
               Falls back to the inference DataFrame mean when None (not recommended).
    scaler: fitted StandardScaler from training. Falls back to re-fitting when None.
    encoders: dict of {col: fitted LabelEncoder} plus "__columns__" key for column order,
              as saved by the compare_models_*.py scripts. Falls back to re-fitting when None.
    """
    df = df.copy()
    impute_mean = fill_mean if fill_mean is not None else df.mean(numeric_only=True)
    df = df.fillna(impute_mean)

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

    cat_cols = ["TeamName", "GrandPrix", "Driver"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        if encoders and col in encoders:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = [le.transform([v])[0] if v in known else -1
                       for v in df[col].astype(str)]
        else:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    if encoders and "__columns__" in encoders:
        cols = [c for c in encoders["__columns__"] if c in df.columns]
        df = df[cols]

    if scaler is not None:
        X_scaled = scaler.transform(df)
    else:
        X_scaled = StandardScaler().fit_transform(df)

    return X_scaled


def load_best_model(quali):
    """Load the best model and its preprocessing artifacts.

    Returns (model, scaler, encoders, train_mean) where scaler and encoders are the
    objects fitted on the full training dataset during compare_models_*.py, and
    train_mean is a pandas Series used for NaN imputation at inference time.
    """
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

    prefix = f"{best_year}_{task_prefix}"
    model = joblib.load(model_dir / f"{prefix}_{best_model_name}.joblib")
    scaler = joblib.load(model_dir / f"{prefix}_scaler.joblib")
    encoders = joblib.load(model_dir / f"{prefix}_encoders.joblib")
    train_mean = pd.read_json(model_dir / f"{prefix}_train_mean.json", typ="series")

    return model, scaler, encoders, train_mean


def predict_next_race():
    event = get_next_race()
    if event is None:
        raise ValueError("No upcoming race found for the current season.")

    year = event["year"]
    gp = event["grand_prix"]
    is_quali_done = event["has_quali"]

    logger.info("Next race: %s (%s)", gp, event["race_datetime"])
    logger.info("Qualifying already done? %s", is_quali_done)

    df = load_features_for_race(year)

    df_race = df[df["GrandPrix"] == gp]
    if df_race.empty:
        raise ValueError(f"No features for GP {gp}")

    model, scaler, encoders, train_mean = load_best_model(is_quali_done)
    X = prepare_features(df_race, use_quali_features=is_quali_done,
                         fill_mean=train_mean, scaler=scaler, encoders=encoders)
    probs = model.predict_proba(X)[:, 1]

    df_race = df_race.copy()
    df_race["win_prob"] = probs
    df_race_sorted = df_race.sort_values("win_prob", ascending=False)

    top3 = df_race_sorted.head(3)[["Driver", "TeamName", "win_prob"]]
    winner = df_race_sorted.head(1)[["Driver", "TeamName", "win_prob"]]

    logger.info("Predicted Winner:\n%s", winner.to_string(index=False))
    logger.info("Predicted Podium:\n%s", top3.to_string(index=False))

    log_prediction(
        race_year=year,
        grand_prix=gp,
        predicted_winner=winner["Driver"].iloc[0],
        predicted_podium=top3["Driver"].tolist(),
        model_name=model.__class__.__name__,
    )

    return top3, winner


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    predict_next_race()
