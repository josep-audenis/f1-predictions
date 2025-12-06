from fastapi import APIRouter, HTTPException
from typing import List, Dict
from datetime import datetime, timezone

from app.utils.calendar_utils import load_season_calendar, get_next_race
from app.services.data_service import DataService

router = APIRouter()
data_service = DataService()

@router.get("/calendar/{year}")
async def get_race_calendar(year: int):
    try:
        calendar = load_season_calendar(year)
        return {
            "year": year,
            "total_races": len(calendar),
            "calendar": calendar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/next")
async def get_next_race_info():
    try:
        next_race = get_next_race()
        if not next_race:
            raise HTTPException(status_code=404, detail="No upcoming races found")
        
        result = dict(next_race)
        if result.get("race_datetime"):
            result["race_datetime"] = result["race_datetime"].isoformat()
        if result.get("quali_datetime"):
            result["quali_datetime"] = result["quali_datetime"].isoformat()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{year}/{grand_prix}")
async def get_race_details(year: int, grand_prix: str):
    try:
        calendar = load_season_calendar(year)
        race = next((r for r in calendar if r["GrandPrix"].lower() == grand_prix.lower()), None)
        
        if not race:
            raise HTTPException(
                status_code=404, 
                detail=f"Race '{grand_prix}' not found in {year} calendar"
            )
        
        return race
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_next_race_status():
    try:
        next_race = get_next_race()
        if not next_race:
            return {"status": "no_upcoming_races"}
        
        is_quali_done = data_service.is_quali_done(next_race["quali_datetime"])
        
        return {
            "grand_prix": next_race["grand_prix"],
            "race_date": next_race["race_datetime"].isoformat(),
            "quali_date": next_race["quali_datetime"].isoformat(),
            "quali_completed": is_quali_done,
            "event_format": next_race["event_format"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
