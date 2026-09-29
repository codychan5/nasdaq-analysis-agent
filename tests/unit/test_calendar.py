from datetime import date, datetime
from zoneinfo import ZoneInfo
import pytest

ET = ZoneInfo("America/New_York")

def et(y, m, d, hh, mm, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=ET)

@pytest.mark.parametrize("now, expected", [
    (et(2026, 9, 23, 8, 0), date(2026, 9, 22)),      # before open on a trading day
    (et(2026, 9, 23, 9, 29), date(2026, 9, 22)),     # one minute before open
    (et(2026, 9, 23, 12, 0), date(2026, 9, 22)),     # during the session
    (et(2026, 9, 23, 16, 0, 1), date(2026, 9, 23)),  # one second after close
    (et(2026, 9, 26, 12, 0), date(2026, 9, 25)),     # Saturday
    (et(2026, 11, 26, 12, 0), date(2026, 11, 25)),   # Thanksgiving
])
def test_resolve_last_completed_session(now, expected):
    from nasdaq_agent.calendar import resolve_last_completed_session
    s = resolve_last_completed_session(now)
    assert s.date == expected
    assert s.label == "official close"

def test_early_close_flag():
    from nasdaq_agent.calendar import resolve_last_completed_session
    s = resolve_last_completed_session(et(2026, 11, 27, 13, 0, 1))
    assert s.date == date(2026, 11, 27) and s.early_close is True

def test_is_market_closed():
    from nasdaq_agent.calendar import is_market_closed
    assert is_market_closed(et(2026, 9, 23, 10, 0)) is False
    assert is_market_closed(et(2026, 9, 23, 16, 0, 1)) is True
    assert is_market_closed(et(2026, 9, 26, 10, 0)) is True

def test_previous_session_dates():
    from nasdaq_agent.calendar import previous_session_dates
    got = previous_session_dates(date(2026, 9, 24), 6)
    assert got == [date(2026, 9, 17), date(2026, 9, 18), date(2026, 9, 21),
                   date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]

def test_utc_input_is_converted():
    from nasdaq_agent.calendar import resolve_last_completed_session
    now_utc = datetime(2026, 9, 23, 20, 0, 1, tzinfo=ZoneInfo("UTC"))  # 16:00:01 ET in EDT
    assert resolve_last_completed_session(now_utc).date == date(2026, 9, 23)

def test_naive_datetime_rejected():
    from nasdaq_agent.calendar import resolve_last_completed_session, CalendarError
    with pytest.raises(CalendarError):
        resolve_last_completed_session(datetime(2026, 9, 23, 12, 0))

def test_previous_session_dates_rejects_non_session():
    from nasdaq_agent.calendar import previous_session_dates, CalendarError
    with pytest.raises(CalendarError):
        previous_session_dates(date(2026, 9, 26), 3)  # Saturday

def test_previous_session_dates_rejects_too_many(monkeypatch):
    from nasdaq_agent import calendar as cal
    from nasdaq_agent.calendar import previous_session_dates, CalendarError
    three_sessions = cal._calendar().schedule(start_date="2026-09-21", end_date="2026-09-23")
    monkeypatch.setattr(cal, "_schedule", lambda start, end: three_sessions)
    with pytest.raises(CalendarError):
        previous_session_dates(date(2026, 9, 23), 5)


def test_exchange_time_shows_a_moment_in_us_eastern_time():
    # Headline times sit beside the session date, which is an Eastern date: 01:30 UTC on the 17th is still the evening
    # of the 16th in New York, after that session's close.
    from datetime import timezone
    from nasdaq_agent.calendar import exchange_time
    assert exchange_time(datetime(2026, 9, 17, 1, 30, tzinfo=timezone.utc)) == et(2026, 9, 16, 21, 30)


def test_exchange_time_reads_a_naive_timestamp_as_utc_rather_than_failing():
    # A display helper must never fail a report; provider timestamps are UTC by convention.
    from nasdaq_agent.calendar import exchange_time
    assert exchange_time(datetime(2026, 9, 16, 12, 0)) == et(2026, 9, 16, 8, 0)


EASTERN_TIME = ZoneInfo("America/New_York")
AFTER_SEPTEMBER_28_CLOSE = datetime(2026, 9, 28, 20, 30, tzinfo=ZoneInfo("UTC"))  # 16:30 New York


def test_a_session_date_pins_the_clock_to_half_past_four_new_york_time():
    from nasdaq_agent.calendar import clock_for_session_date
    pinned = clock_for_session_date(date(2026, 7, 8), AFTER_SEPTEMBER_28_CLOSE)
    assert pinned == datetime(2026, 7, 8, 16, 30, tzinfo=EASTERN_TIME)
    assert pinned.utcoffset().total_seconds() == -4 * 3600  # daylight time in July


@pytest.mark.parametrize("day", [date(2026, 7, 3), date(2026, 7, 4)])  # the Independence Day holiday, a Saturday
def test_a_session_date_that_is_not_a_trading_day_is_refused(day):
    from nasdaq_agent.calendar import CalendarError, clock_for_session_date
    with pytest.raises(CalendarError, match="not a NASDAQ trading day"):
        clock_for_session_date(day, AFTER_SEPTEMBER_28_CLOSE)


def test_a_session_date_whose_session_has_not_closed_is_refused():
    from nasdaq_agent.calendar import CalendarError, clock_for_session_date
    during_the_session = datetime(2026, 9, 28, 15, 0, tzinfo=EASTERN_TIME)
    with pytest.raises(CalendarError, match="not closed"):
        clock_for_session_date(date(2026, 9, 28), during_the_session)
    with pytest.raises(CalendarError, match="not closed"):
        clock_for_session_date(date(2026, 9, 30), AFTER_SEPTEMBER_28_CLOSE)
    assert clock_for_session_date(date(2026, 9, 28), AFTER_SEPTEMBER_28_CLOSE).date() == date(2026, 9, 28)
