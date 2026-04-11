"""
Prediction accuracy tracker.

Maintains data/processed/prediction_log.csv with one row per race prediction.
After a race is run and results are collected, call fill_actuals() to back-fill
the actual winner / podium and compute correctness flags.
"""
import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_PATH = BASE_DIR / "data" / "processed" / "prediction_log.csv"
RESULTS_DIR = BASE_DIR / "data" / "processed"

_COLUMNS = [
    "race_year",
    "grand_prix",
    "predicted_winner",
    "predicted_podium",
    "actual_winner",
    "actual_podium",
    "winner_correct",
    "podium_correct",
    "model_name",
    "prediction_timestamp",
]


def _load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=_COLUMNS)


def _save_log(df: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


def log_prediction(
    race_year: int,
    grand_prix: str,
    predicted_winner: str,
    predicted_podium: list[str],
    model_name: str,
) -> None:
    """Append a prediction row. Actual results are filled in later by fill_actuals()."""
    df = _load_log()

    # Overwrite if a prediction for this race already exists
    mask = (df["race_year"] == race_year) & (df["grand_prix"] == grand_prix)
    df = df[~mask]

    new_row = {
        "race_year": race_year,
        "grand_prix": grand_prix,
        "predicted_winner": predicted_winner,
        "predicted_podium": json.dumps(predicted_podium),
        "actual_winner": None,
        "actual_podium": None,
        "winner_correct": None,
        "podium_correct": None,
        "model_name": model_name,
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_log(df)


def fill_actuals(results_dir: Path = RESULTS_DIR) -> int:
    """Back-fill actual results for any logged predictions that lack them.

    Reads results_{year}.csv files and updates winner_correct / podium_correct.
    Returns the number of rows updated.
    """
    df = _load_log()
    if df.empty:
        return 0

    pending = df[df["actual_winner"].isna()]
    if pending.empty:
        return 0

    # Ensure mixed-type assignment doesn't hit pandas dtype inference warnings
    for col in ["actual_winner", "actual_podium", "winner_correct", "podium_correct"]:
        df[col] = df[col].astype(object)

    updated = 0
    for idx, row in pending.iterrows():
        year = int(row["race_year"])
        gp = row["grand_prix"]
        results_path = results_dir / f"results_{year}.csv"
        if not results_path.exists():
            continue

        results = pd.read_csv(results_path)
        # GrandPrix column stores the full EventName; match flexibly
        race = results[results["GrandPrix"].str.contains(gp, na=False, regex=False)]
        if race.empty:
            continue

        race = race.copy()
        race["Position"] = pd.to_numeric(race["Position"], errors="coerce")
        race = race.dropna(subset=["Position"]).sort_values("Position")

        top3 = race[race["Position"] <= 3]["FullName"].tolist()
        actual_winner = top3[0] if top3 else None
        if actual_winner is None:
            continue

        predicted_winner = row["predicted_winner"]
        predicted_podium = json.loads(row["predicted_podium"])

        df.at[idx, "actual_winner"] = actual_winner
        df.at[idx, "actual_podium"] = json.dumps(top3)
        df.at[idx, "winner_correct"] = predicted_winner == actual_winner
        df.at[idx, "podium_correct"] = set(predicted_podium) == set(top3)
        updated += 1

    if updated:
        _save_log(df)

    return updated


def summary(n: int | None = None) -> pd.DataFrame:
    """Return a DataFrame summarising prediction accuracy.

    n: if given, restrict to the last n completed races (rows where actual_winner is set).
    """
    df = _load_log()
    completed = df[df["actual_winner"].notna()].copy()
    completed["winner_correct"] = completed["winner_correct"].astype(bool)
    completed["podium_correct"] = completed["podium_correct"].astype(bool)
    completed = completed.sort_values(["race_year", "prediction_timestamp"])

    if n is not None:
        completed = completed.tail(n)

    return completed[
        [
            "race_year",
            "grand_prix",
            "predicted_winner",
            "actual_winner",
            "winner_correct",
            "predicted_podium",
            "actual_podium",
            "podium_correct",
            "model_name",
            "prediction_timestamp",
        ]
    ].reset_index(drop=True)
