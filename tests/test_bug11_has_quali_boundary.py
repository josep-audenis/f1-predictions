"""
BUG-11 regression tests — has_quali boundary condition.

Old behaviour: `has_quali = today >= quali_time`
This returns True even when today IS the qualifying session start time,
meaning qualifying is being treated as done before it has happened.

Fix: `has_quali = today > quali_time`
Qualifying is only considered done if the current time is strictly after
the scheduled qualifying start.
"""
import sys
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "utils"))
from calendar_utils import get_next_race


# ---------------------------------------------------------------------------
# Helper: build a fake calendar with one event
# ---------------------------------------------------------------------------

def make_event(race_dt, quali_dt):
    return {
        "RoundNumber": 1,
        "EventName": "Monaco Grand Prix",
        "GrandPrix": "Monaco",
        "Country": "Monaco",
        "EventFormat": "conventional",
        "Session1": race_dt - timedelta(days=4),
        "Session2": race_dt - timedelta(days=3),
        "Session3": race_dt - timedelta(days=2),
        "Quali": quali_dt,
        "Race": race_dt,
    }


def patch_calendar(events):
    return patch(
        "calendar_utils.load_season_calendar",
        return_value=events,
    )


# All times in UTC; race is always 2 days after quali so the event is "upcoming"
RACE_DT  = datetime(2026, 5, 25, 13, 0, tzinfo=timezone.utc)
QUALI_DT = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


class TestHasQualiBoundary:
    def test_has_quali_false_before_quali(self):
        """An hour before qualifying start → has_quali must be False."""
        now = QUALI_DT - timedelta(hours=1)
        event = make_event(RACE_DT, QUALI_DT)

        with patch_calendar([event]):
            result = get_next_race(today=now)

        assert result["has_quali"] is False, (
            "has_quali should be False when now is before the qualifying session"
        )

    def test_has_quali_false_at_exact_quali_start(self):
        """
        Regression for BUG-11.

        When now == quali_time exactly, qualifying has just started and cannot
        be considered done.  Old code (>=) would return True here.
        """
        now = QUALI_DT  # exactly at qualifying start
        event = make_event(RACE_DT, QUALI_DT)

        with patch_calendar([event]):
            result = get_next_race(today=now)

        assert result["has_quali"] is False, (
            "has_quali must be False at the exact qualifying start time — "
            "old code using >= would incorrectly return True here (BUG-11)"
        )

    def test_has_quali_true_one_second_after_quali(self):
        """One second after qualifying start → has_quali must be True."""
        now = QUALI_DT + timedelta(seconds=1)
        event = make_event(RACE_DT, QUALI_DT)

        with patch_calendar([event]):
            result = get_next_race(today=now)

        assert result["has_quali"] is True, (
            "has_quali should be True when now is after qualifying start"
        )

    def test_has_quali_true_well_after_quali(self):
        """Several hours after qualifying → has_quali must be True."""
        now = QUALI_DT + timedelta(hours=3)
        event = make_event(RACE_DT, QUALI_DT)

        with patch_calendar([event]):
            result = get_next_race(today=now)

        assert result["has_quali"] is True

    def test_has_quali_false_one_second_before_quali(self):
        """One second before qualifying → has_quali must be False."""
        now = QUALI_DT - timedelta(seconds=1)
        event = make_event(RACE_DT, QUALI_DT)

        with patch_calendar([event]):
            result = get_next_race(today=now)

        assert result["has_quali"] is False

    def test_returns_none_when_no_future_race(self):
        """get_next_race returns None when all races are in the past."""
        past_race  = datetime(2020, 1, 1, 13, 0, tzinfo=timezone.utc)
        past_quali = datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc)
        event = make_event(past_race, past_quali)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)

        with patch_calendar([event]):
            result = get_next_race(today=now)

        assert result is None
