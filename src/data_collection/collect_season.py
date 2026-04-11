from datetime import datetime
from data_collection.collection_utils import collect_season_data

def collect_multi_year(start: int = 2018, end: int = None):
    if end is None:
        end = datetime.now().year
    for year in range(start, end + 1):
        collect_season_data(year)

def collect_year(year: int = None):
    if year is None:
        year = datetime.now().year
    collect_season_data(year)

if __name__ == "__main__":
    collect_multi_year(2018)
