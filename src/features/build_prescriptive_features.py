import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

def load_data(year: int):

    laps_path = DATA_DIR / f"laps_{year}.csv"
    results_path = DATA_DIR / f"results_{year}.csv"

    laps = pd.read_csv(laps_path)
    results = pd.read_csv(results_path)
    
    return laps, results

def process_features(laps: pd.DataFrame) -> pd.DataFrame:

    laps["LapTime"] = pd.to_timedelta(laps["LapTime"], errors="coerce").dt.total_seconds()
    laps["SpeedI1"] = pd.to_numeric(laps["SpeedI1"], errors="coerce")
    laps["SpeedFL"] = pd.to_numeric(laps["SpeedFL"], errors="coerce")
    laps = laps.dropna(subset=['LapTime'])
    
    return laps

def extract_lap_features(laps: pd.DataFrame) -> pd.DataFrame:
    features = (laps.groupby(["Year", "GrandPrix", "Driver"]).agg({
        "LapTime": "mean",
        "SpeedI1": "mean",
        "SpeedFL": "max",
        "IsAccurate": "mean",
        "Compound": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None,
        "Stint": "max",
        "LapNumber": "count"
    }).rename(columns={
        "LapTime": "avg_lap_time",
        "SpeedI1": "avg_speed",
        "SpeedFL": "max_speed",
        "IsAccurate": "data_accuracy",
        "Compund": "dominant_compound",
        "String": "stint_count",
        "LapNumber": "laps_completed"
    }).reset_index())

    return features


def merge_race_results(lap_features: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merged = lap_features.merge(results.rename(columns={"Abbreviation": "Driver"})[["Year", "GrandPrix", "Driver", "Position", "TeamName"]],
                            on=["Year", "GrandPrix", "Driver"],
                            how="left")
    merged["is_podium"] = merged["Position"].apply(lambda x: 1 if x <= 3 else 0)
    
    return merged

def build_driver_features(year: int):

    laps, results = load_data(year)
    laps = process_features(laps)
    features = extract_lap_features(laps)
    merged = merge_race_results(features, results)

    return  merged

def build_full_dataset(start_year: int, end_year: int):
    all_years = [build_driver_features(y) for y in range(start_year, end_year + 1)]
    df = pd.concat(all_years, ignore_index=True)
    df.to_csv(DATA_DIR / "driver_features.csv", index=False)
    print(f"Saved combined dataset with {len(df)} rows.")

if __name__ == "__main__":
    build_full_dataset(2018, 2025)