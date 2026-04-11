from fastapi import APIRouter, HTTPException
from typing import Dict, List
from datetime import datetime

from app.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()

@router.get("/next-race")
async def predict_next_race():
    try:
        result = await prediction_service.get_next_race_prediction()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/race/{year}/{grand_prix}")
async def predict_specific_race(grand_prix: str, year: str = None):
    if year is None:
        year = str(datetime.now().year)
    try:
        result = await prediction_service.get_race_predictions(grand_prix, year)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))