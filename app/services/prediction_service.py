import pandas as pd
import numpy as np
import json
import joblib
from typing import Dict, List, Tuple
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.core.config import settings
from app.services.data_service import DataService

logger = logging.getLogger(__name__)


def normalize_gp_name(name: str) -> str:
    name = name.replace("Grand Prix", "").strip()
    name = name.replace("GP", "").strip()
    return name


class PredictionService:
    def __init__(self):
        self.data_service = DataService()
        self.models_cache = {}
        
    async def get_next_race_prediction(self) -> Dict:
        try:
            gp, race_day, quali_day = self.data_service.get_next_race()

            is_quali_done = self.data_service.is_quali_done(quali_day.isoformat())
        
            df = self.data_service.load_features_for_race("2025")
            
            if 'GrandPrix' in df.columns:
                df['GrandPrix'] = df['GrandPrix'].apply(normalize_gp_name)

            df_race = df[df["GrandPrix"] == gp]
            
            if df_race.empty:
                available_gps = df['GrandPrix'].unique().tolist()
                raise ValueError(f"No features found for '{gp}'. Available: {available_gps}")

            X, driver_names, team_names = self._prepare_features(df_race, use_quali_features=is_quali_done)

            model = self._load_best_model(is_quali_done)
            
            probs = model.predict_proba(X)

            if len(probs.shape) == 2 and probs.shape[1] == 2:
                win_probs = probs[:, 1]
            else:
                logger.error(f"DEBUG: Unexpected probs shape: {probs.shape}")
                win_probs = probs
            
            df_race = df_race.copy()
            df_race["win_prob"] = win_probs
            
            df_race_sorted = df_race.sort_values("win_prob", ascending=False)
            
            required_cols = ["Driver", "TeamName", "win_prob"]
            for col in required_cols:
                if col not in df_race_sorted.columns:
                    raise ValueError(f"Required column '{col}' not found in df_race_sorted. Available: {df_race_sorted.columns.tolist()}")
            
            podium = df_race_sorted.head(3)[required_cols].to_dict('records')
            winner_rows = df_race_sorted.head(1)[required_cols].to_dict('records')
            
            if not winner_rows:
                raise ValueError("No winner found in predictions")
            
            winner = winner_rows[0]
            logger.info(f"DEBUG: winner: {winner}")
            
            return {
                "grand_prix": gp,
                "race_date": race_day.isoformat(),
                "quali_date": quali_day.isoformat(),
                "quali_completed": is_quali_done,
                "predicted_winner": winner,
                "predicted_podium": podium,
                "all_predictions": df_race_sorted[required_cols].to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _prepare_features(self, df: pd.DataFrame, use_quali_features: bool = True) -> Tuple[np.ndarray, pd.Series, pd.Series]:
        df = df.copy()
        
        driver_names = df["Driver"].copy()
        team_names = df["TeamName"].copy()
        
        df = df.fillna(df.mean(numeric_only=True))

        cat_cols = ["TeamName", "GrandPrix", "Driver"]
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        df = df.drop(columns=["Position"], errors="ignore")

        if not use_quali_features:
            df = df.drop(
                columns=[
                    "grid_position",
                    "is_top10_start",
                    "grid_vs_team_avg",
                    "driver_avg_quali_last5",
                    "team_avg_quali_last5",
                ],
                errors="ignore"
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        return X_scaled, driver_names, team_names
    
    def _load_best_model(self, quali_done: bool) -> any:
        model_type = "quali" if quali_done else "pre-quali"
        
        model_name = "GradientBoosting"
        model_file = f"{model_name}_model.pkl"
        model_full_path = settings.models_dir / "top1" / model_type / model_file
        
        if not model_full_path.exists():
            model_dir = settings.models_dir / "top1" / model_type
            available = []
            if model_dir.exists():
                available = [f.name for f in model_dir.glob("*.pkl")]
            
            raise FileNotFoundError(
                f"Model file not found: {model_full_path}\n"
                f"Available models in {model_dir}: {available}\n"
                f"Make sure you ran: python train_production_models.py"
            )
        
        return joblib.load(model_full_path)
    
    async def get_race_predictions(self, grand_prix: str, year: str = "2025") -> Dict:
        try:
            df = self.data_service.load_features_for_race(year)
            
            if 'GrandPrix' in df.columns:
                df['GrandPrix'] = df['GrandPrix'].apply(normalize_gp_name)
            
            grand_prix_normalized = normalize_gp_name(grand_prix)
            df_race = df[df["GrandPrix"] == grand_prix_normalized]
            
            if df_race.empty:
                available_gps = df['GrandPrix'].unique().tolist()
                raise ValueError(f"No data found for '{grand_prix}' (normalized: '{grand_prix_normalized}'). Available: {available_gps}")
                
            # TODO: Implement proper quali status check based on race date
            is_quali_done = True
            
            X, driver_names, team_names = self._prepare_features(df_race, use_quali_features=is_quali_done)
            model = self._load_best_model(is_quali_done)
            
            probs = model.predict_proba(X)[:, 1]
            df_race = df_race.copy()
            df_race["win_prob"] = probs
            df_race_sorted = df_race.sort_values("win_prob", ascending=False)
            
            return {
                "grand_prix": grand_prix,
                "year": year,
                "predictions": df_race_sorted[["Driver", "TeamName", "win_prob"]].to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Race prediction error: {str(e)}")
            raise