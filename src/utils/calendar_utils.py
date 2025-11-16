import fastf1
from datetime import datetime, timezone, timedelta



def load_season_calendar(year: int):
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    events = []
    for _, row in schedule.iterrows():
        events.append({
            "RoundNumber": int(row['RoundNumber']),
            "EventName": row['EventName'],
            "GrandPrix": row['EventName'].replace("Grand Prix", "").strip(),
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
    today = today or datetime.now(timezone.utc)
    year = today.year

    events = load_season_calendar(year)

    future_events = [e for e in events if e["Race"] > today]

    if not future_events:
        return None

    next_event = min(future_events, key=lambda e: e["Race"])

    quali_time = next_event["Quali"]
    has_quali = today >= quali_time

    return {
        "year": year,
        "round": next_event["RoundNumber"],
        "grand_prix": next_event["GrandPrix"],
        "country": next_event["Country"],
        "race_datetime": next_event["Race"],
        "quali_datetime": next_event["Quali"],
        "has_quali": has_quali,
        "event_format": next_event["EventFormat"],
        "raw": next_event
    }
