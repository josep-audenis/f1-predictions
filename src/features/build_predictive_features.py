import os
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed")

def load_all_csvs(folder: str, start: int = 2018, end: int = 2025) -> pd.DataFrame:
    base_path = Path(__file__).resolve().parent
    folder_path = base_path / folder

    all_dfs = []
    for file in sorted(folder_path.glob("*.csv")):
        if any(year in file.name for year in str(range(start, end+1))):
            print(f"Loading {file.name} ...")
            df = pd.read_csv(file)
            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No CSVs found in {folder_path}")
    
    return pd.concat(all_dfs, ignore_index=True)


def rolling_stat(df: pd.DataFrame, group_cols, target_col, window, func) -> pd.DataFrame:
    df = df.sort_values(["Year", "GrandPrix"])
    result = (
        df.groupby(group_cols, group_keys=False)[target_col]
        .apply(lambda x: x.shift(1).rolling(window, min_periods=1).agg(func))
    )
    return result

def build_features(start: int = 2018, end: int = 2025, window: int = 5):

    results = load_all_csvs("../../data/raw/results", start=start, end=end)
    laps = load_all_csvs("../../data/raw/laps", start=start, end=end)

    results["Year"] = results["Year"].astype(int)
    laps["Year"] = laps["Year"].astype(int)

    results = results.dropna(subset=["DriverId", "GrandPrix"])
    results = results.sort_values(["Year", "GrandPrix", "DriverId"])

    results.rename(columns={"FullName": "Driver"}, inplace=True)

    results["driver_avg_finish_last5"] = rolling_stat(results, ["Driver"], "Position", window, "mean")
    results["driver_std_finish_last5"] = rolling_stat(results, ["Driver"], "Position", window, "std")
    results["driver_avg_quali_last5"] = rolling_stat(results, ["Driver"], "GridPosition", window, "mean")
    results["driver_points_last5"] = rolling_stat(results, ["Driver"], "Points", window, "mean")

    results["DNF"] = results["Status"].apply(lambda s: 1 if str(s).lower() not in ["finished", "classified"] else 0)
    results["driver_dnf_rate_last5"] = rolling_stat(results, ["Driver"], "DNF", window, "mean")

    results["team_avg_finish_last5"] = rolling_stat(results, ["TeamName"], "Position", window, "mean")
    results["team_avg_quali_last5"] = rolling_stat(results, ["TeamName"], "GridPosition", window, "mean")
    results["team_points_last5"] = rolling_stat(results, ["TeamName"], "Points", window, "mean")
    results["team_dnf_rate_last5"] = rolling_stat(results, ["TeamName"], "DNF", window, "mean")

    results["driver_points_ytd"] = (results.groupby(["Year", "Driver"])["Points"].cumsum().shift(1).fillna(0))
    results["team_points_ytd"] = (results.groupby(["Year", "TeamName"])["Points"].cumsum().shift(1).fillna(0))

    results["grid_position"] = results["GridPosition"]
    results["is_top10_start"] = (results["GridPosition"] <= 10).astype(int)

    team_grid = (results.groupby(["Year", "GrandPrix", "TeamName"])["GridPosition"].transform("mean"))
    results["team_grid_avg"] = team_grid
    results["grid_vs_team_avg"] = results["GridPosition"] - team_grid

    feature_cols = [
        "Year",
        "GrandPrix",
        "Driver",
        "TeamName",
        "grid_position",
        "is_top10_start",
        "grid_vs_team_avg",
        "driver_avg_finish_last5",
        "driver_std_finish_last5",
        "driver_avg_quali_last5",
        "driver_points_last5",
        "driver_dnf_rate_last5",
        "driver_points_ytd",
        "team_avg_finish_last5",
        "team_avg_quali_last5",
        "team_points_last5",
        "team_dnf_rate_last5",
        "team_points_ytd",
        #"track_type",  # TBI
        #"track_length_km", # TBI
        #"track_speed_class", # TBI
        "Position",  
    ]

    df_final = results[feature_cols]
    df_final.to_csv(os.path.join(OUTPUT_PATH, f"features_pre_race_{start}-{end}.csv"), index=False)
    print(f"Saved features to {OUTPUT_PATH}.")

    return df_final



if __name__ == "__main__":
    
    start = 2018 
    end = 2025
    assert start > 2017, "[WARNING] FastF1API only has detailed data from 2018"
    


    for i in range(start, end + 1):
        build_features(start=i, end=end)