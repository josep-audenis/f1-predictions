import logging
import os
import pandas as pd
from pathlib import Path
import fastf1

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"

fastf1.Cache.enable_cache(str(CACHE_DIR))

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
        logger.error("Failed to load %s %s: %s", grand_prix, year, e)
        return None, None

    return laps, results


def collect_season_data(year: int, save_dir: Path = PROCESSED_DIR, force: bool = False):
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        calendar = fastf1.get_event_schedule(year)
    except Exception as e:
        logger.error("Failed to load %s schedule: %s", year, e)
        return

    results_path = save_dir / f"results_{year}.csv"
    laps_path = save_dir / f"laps_{year}.csv"

    existing_gps: set = set()
    if not force and results_path.exists():
        existing_gps = set(pd.read_csv(results_path)["GrandPrix"].unique())

    laps_all = []
    results_all = []

    logger.info("Collecting data for %s season...", year)
    for gp in calendar["EventName"]:
        if gp in existing_gps:
            logger.info(" - %s (cached, skipping)", gp)
            continue
        logger.info(" - %s", gp)
        laps, results = get_race_data(year, gp)
        if laps is not None:
            laps_all.append(laps)
        if results is not None:
            results_all.append(results)

    if laps_all:
        new_laps = pd.concat(laps_all)
        if not force and laps_path.exists():
            new_laps = pd.concat([pd.read_csv(laps_path), new_laps], ignore_index=True)
        new_laps.to_csv(laps_path, index=False)

    if results_all:
        new_results = pd.concat(results_all)
        if not force and results_path.exists():
            new_results = pd.concat([pd.read_csv(results_path), new_results], ignore_index=True)
        new_results.to_csv(results_path, index=False)

    logger.info("Saved data for %s season", year)
