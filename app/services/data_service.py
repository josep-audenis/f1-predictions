import pandas as pd
from datetime import datetime, timezone
from typing import Tuple, Dict, Union
from pathlib import Path
import logging

from src.data_collection.collect_season import collect_year
from src.features.build_predictive_features import build_features

from app.core.config import settings
from app.utils.calendar_utils import get_next_race

logger = logging.getLogger(__name__)

class DataService:
    def __init__(self):
        self.data_dir = settings.data_dir
        
    def get_next_race(self) -> Tuple[str, datetime, datetime]:
        next_race = get_next_race()
        if not next_race:
            raise ValueError("No upcoming races found")
        
        return (
            next_race["grand_prix"],
            next_race["race_datetime"],
            next_race["quali_datetime"]
        )
    
    def is_quali_done(self, quali_date: Union[str, datetime]) -> bool:
        
        now = datetime.now(timezone.utc)
        
        if isinstance(quali_date, str):
            quali_datetime = datetime.fromisoformat(quali_date)
        else:
            quali_datetime = quali_date
        
        if quali_datetime.tzinfo is None:
            quali_datetime = quali_datetime.replace(tzinfo=timezone.utc)
        else:
            quali_datetime = quali_datetime.astimezone(timezone.utc)
        
        try:
            result = now >= quali_datetime
            return result
        except Exception as e:
            logger.error(f"DEBUG is_quali_done - Comparison failed: {e}")
            logger.error(f"DEBUG is_quali_done - now type: {type(now)}, tzinfo: {now.tzinfo}")
            logger.error(f"DEBUG is_quali_done - quali_datetime type: {type(quali_datetime)}, tzinfo: {quali_datetime.tzinfo}")
            raise
    
    
    def load_features_for_race(self, year: str = None) -> pd.DataFrame:
        current_year = str(datetime.now().year)
        if year is None:
            year = current_year
        features_file = settings.data_dir / "processed" / f"features_pre_race_{year}-{current_year}.csv"
        
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
            
        return pd.read_csv(features_file)
    
    
    async def update_data(self) -> Dict:
        try:
            current_year = datetime.now().year
            collect_year(current_year)
            
            build_features(start=current_year, end=current_year)
            
            return {"status": "success", "message": "Data updated successfully"}
            
        except Exception as e:
            logger.error(f"Data update error: {str(e)}")
            return {"status": "error", "message": str(e)}