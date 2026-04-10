import fastf1
from datetime import datetime, timezone, timedelta
import pandas as pd


def normalize_gp_name(name: str) -> str:
    name = name.replace("Grand Prix", "").strip()
    name = name.replace("GP", "").strip()
    return name


def load_season_calendar(year: int):
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    events = []
    for _, row in schedule.iterrows():
        event_name = row['EventName']
        events.append({
            "RoundNumber": int(row['RoundNumber']),
            "EventName": event_name,
            "GrandPrix": normalize_gp_name(event_name),
            "Country": row['Country'],
            "EventFormat": row['EventFormat'],

            "Session1": row["Session1DateUtc"],
            "Session2": row["Session2DateUtc"],
            "Session3": row["Session3DateUtc"],
            "Quali": row["Session4DateUtc"],
            "Race": row["Session5DateUtc"],
        })
    return events


def get_next_race(today=None):
    if today is None:
        today = datetime.now(timezone.utc)
    elif today.tzinfo is None:
        today = today.replace(tzinfo=timezone.utc)
    else:
        today = today.astimezone(timezone.utc)
    
    year = today.year

    events = load_season_calendar(year)

    future_events = []
    for e in events:
        race_time = e["Race"]
        
        if isinstance(race_time, pd.Timestamp):
            race_time = race_time.to_pydatetime()
        
        if race_time.tzinfo is None:
            race_time = race_time.replace(tzinfo=timezone.utc)
        else:
            race_time = race_time.astimezone(timezone.utc)
        
        if race_time > today:
            future_events.append(e)

    if not future_events:
        return None

    next_event = min(future_events, key=lambda e: e["Race"])

    quali_time = next_event["Quali"]
    race_time = next_event["Race"]
    
    if isinstance(quali_time, pd.Timestamp):
        quali_time = quali_time.to_pydatetime()
    if isinstance(race_time, pd.Timestamp):
        race_time = race_time.to_pydatetime()
    
    if quali_time.tzinfo is None:
        quali_time = quali_time.replace(tzinfo=timezone.utc)
    else:
        quali_time = quali_time.astimezone(timezone.utc)
    
    if race_time.tzinfo is None:
        race_time = race_time.replace(tzinfo=timezone.utc)
    else:
        race_time = race_time.astimezone(timezone.utc)
    
    has_quali = today >= quali_time

    return {
        "year": year,
        "round": next_event["RoundNumber"],
        "grand_prix": next_event["GrandPrix"],
        "country": next_event["Country"],
        "race_datetime": race_time,
        "quali_datetime": quali_time,
        "has_quali": has_quali,
        "event_format": next_event["EventFormat"],
        "raw": next_event
    }