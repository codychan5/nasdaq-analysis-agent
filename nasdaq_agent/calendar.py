from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

EASTERN = ZoneInfo("America/New_York")
CALENDAR_NAME = "NASDAQ"
LOOKBACK_DAYS_FOR_ONE_SESSION = 14
REGULAR_CLOSE_HOUR = 16
# Calendar days to look back to be sure of covering `count` trading sessions: at most ~3 calendar
# days per session (a weekend between two sessions), plus a fixed pad so a run of holidays never
# truncates the window. Deliberately generous; the caller slices the exact `count` sessions out.
CALENDAR_DAYS_PER_SESSION = 3
SESSION_LOOKBACK_PAD_DAYS = 10
# A run for AGENT_SESSION_DATE is pinned to this New York time on that day: half an hour after the regular close, the
# scheduler's default trigger (AGENT_SCHEDULE_CRON "30 16 * * 1-5"), so it sees the news a scheduled run would have.
SESSION_DATE_CLOCK_TIME = time(16, 30)


class CalendarError(RuntimeError):
    """The trading calendar could not resolve a session."""


@dataclass(frozen=True)
class TradingSession:
    date: date
    label: str
    early_close: bool


@lru_cache(maxsize=1)
def _calendar():
    """Cached NASDAQ calendar to avoid redundant construction and deprecation warnings."""
    return mcal.get_calendar(CALENDAR_NAME)


def _schedule(start: date, end: date) -> pd.DataFrame:
    return _calendar().schedule(start_date=start.isoformat(), end_date=end.isoformat())


def _to_eastern(now: datetime) -> datetime:
    if now.tzinfo is None:
        raise CalendarError("now must be timezone-aware")
    return now.astimezone(EASTERN)


def exchange_time(moment: datetime) -> datetime:
    """moment in the exchange's time zone, US Eastern, for showing a timestamp beside the session date (an Eastern date).
    Unlike _to_eastern, a naive moment is read as UTC, the providers' convention, instead of being rejected: this only
    formats timestamps for display, and a display helper must never fail a report."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(EASTERN)


def exchange_timestamp(moment: datetime | None) -> str | None:
    """moment as ISO 8601 in exchange time, to the minute (2026-09-16T21:30-04:00): how the model and the judge read a
    headline's publication time beside the session date. None stays None."""
    return exchange_time(moment).isoformat(timespec="minutes") if moment is not None else None


def resolve_last_completed_session(now: datetime) -> TradingSession:
    """The most recent NASDAQ session whose close is at or before `now`."""
    now_et = _to_eastern(now)
    sched = _schedule(now_et.date() - timedelta(days=LOOKBACK_DAYS_FOR_ONE_SESSION), now_et.date())
    completed = sched[sched["market_close"] <= pd.Timestamp(now_et)]
    if completed.empty:
        raise CalendarError("no completed session in the lookback window")
    row = completed.iloc[-1]
    close_et = row["market_close"].tz_convert(EASTERN)
    return TradingSession(date=row.name.date(), label="official close", early_close=close_et.hour < REGULAR_CLOSE_HOUR)


def clock_for_session_date(day: date, now: datetime) -> datetime:
    """The pinned clock for a run that reports on `day`: SESSION_DATE_CLOCK_TIME, New York time, that day. Raises
    CalendarError when `day` is not a NASDAQ trading day, or when its session has not closed by `now`."""
    try:
        previous_session_dates(day, 1)
    except CalendarError:
        raise CalendarError(f"{day} is not a NASDAQ trading day") from None
    if resolve_last_completed_session(now).date < day:
        raise CalendarError(f"the {day} session has not closed yet; leave AGENT_SESSION_DATE empty to report the last "
                            "completed session")
    return datetime.combine(day, SESSION_DATE_CLOCK_TIME, tzinfo=EASTERN)


def is_market_closed(now: datetime) -> bool:
    """True outside regular trading hours, on weekends and on holidays."""
    now_et = _to_eastern(now)
    sched = _schedule(now_et.date(), now_et.date())
    if sched.empty:
        return True
    row = sched.iloc[0]
    ts = pd.Timestamp(now_et)
    return not (row["market_open"] <= ts < row["market_close"])


def previous_session_dates(session: date, count: int) -> list[date]:
    """The `count` sessions ending with `session`, ascending. Raises if `session` is not a session."""
    sched = _schedule(session - timedelta(days=count * CALENDAR_DAYS_PER_SESSION + SESSION_LOOKBACK_PAD_DAYS), session)
    dates = [ts.date() for ts in sched.index]
    if not dates or dates[-1] != session:
        raise CalendarError(f"{session} is not a NASDAQ session")
    if len(dates) < count:
        raise CalendarError(f"only {len(dates)} sessions available, need {count}")
    return dates[-count:]
