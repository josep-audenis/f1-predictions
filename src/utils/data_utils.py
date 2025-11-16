import pandas as pd

def load_features_pre_race(data_dir, start_year: int = 2018, end_year: int = 2025):
    dataframes = {}
    for year in range(2018, 2025+1):
        df = pd.read_csv(data_dir / f"features_pre_race_{year}-{end_year}.csv")
        dataframes[f"{year}-{end_year}"] = df
    return dataframes