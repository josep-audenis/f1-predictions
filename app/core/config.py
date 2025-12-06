from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    api_prefix: str = "/api/v1"
    project_name: str = "F1 Predictions API"
    version: str = "1.0.0"
    
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    
    model_performance_top1_quali: Path = data_dir / "processed" / "model_performance_top1_quali.json"
    model_performance_top1_pre_quali: Path = data_dir / "processed" / "model_performance_top1_pre_quali.json"
    
    features_pattern: str = "features_pre_race_{year}-2025.csv"
    
    class Config:
        env_file = ".env"

settings = Settings()