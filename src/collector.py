import fastf1
import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

fastf1.Cache.enable_cache(str(RAW_DIR))


def get_race_data(year: int, grand_prix: str):
    try:
        session = fastf1.get_session(year, grand_prix, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        laps = session.laps
        results = session.results

        laps["Year"] = year
        laps["GrandPrix"] = grand_prix
        results["Year"] = year
        results["GrandPrix"] = grand_prix

    except Exception as e:
        print(f"ERROR: Failed to load {grand_prix} {year}: {e}")
        return None, None
    
    return laps, results


def collect_season_data(year: int, save_dir: Path = PROCESSED_DIR):    
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        calendar = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"ERROR: Failed to load {year} schedule: {e}")
        return

    laps_all = []
    results_all = []

    print(f"Collecting data for {year} season...")
    for gp in calendar["EventName"]:
        print(f" - {gp}")
        laps, results = get_race_data(year, gp)
        if laps is not None:
            laps_all.append(laps)
        if results is not None:
            results_all.append(results)

    if laps_all:
        pd.concat(laps_all).to_csv(os.path.join(save_dir, f"laps_{year}.csv"), index=False)
    if results_all:
        pd.concat(results_all).to_csv(os.path.join(save_dir, f"results_{year}.csv"), index=False)

    print(f"Saved data for {year} season")


def collect_multi_year(start: int = 2021, end: int = 2024):
    for year in range(start, end + 1):
        collect_season_data(year)


if __name__ == "__main__":
    print("Exctracting data")
    collect_multi_year(2015, 2025)