import pandas as pd

def load_features_pre_race(data_dir, start_year: int = 2018, end_year: int = 2025):
    df = pd.read_csv(data_dir / f"features_pre_race_{start_year}-{end_year}.csv")
    return {f"{start_year}-{end_year}": df}