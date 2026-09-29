from datetime import date, datetime, timezone
from nasdaq_agent.sources.errors import SourceError, SourceNoData
from nasdaq_agent.sources.models import Bar, BarSeries, Candidate, CorporateAction, Headline

SESSIONS = [date(2026, 9, 17), date(2026, 9, 18), date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]
CLOSES = [10.00, 10.20, 9.95, 10.60, 10.70, 11.10]
BENCH = [500.0, 501.0, 499.0, 502.0, 503.0, 504.0]

def series(symbol, closes, dates=SESSIONS, source="fake"):
    return BarSeries(symbol=symbol, source=source, bars=[
        Bar(date=d, open=c, high=c, low=c, close=c, adj_close=c, volume=1000) for d, c in zip(dates, closes)])

class FakeGainerSource:
    def __init__(self, name, candidates=None, error=None, requires_market_closed=False):
        self.name, self._cands, self._error, self.requires_market_closed = name, candidates or [], error, requires_market_closed
        self.calls = 0
    def top_candidates(self, session_date, prev_session_date, limit):
        self.calls += 1
        if self._error: raise SourceError(self._error)
        return self._cands[:limit]

class FakeHistorySource:
    """error fails every lookup; errors maps a symbol to the exception its lookups raise. A symbol with no data gets
    SourceNoData, as the real adapters answer when they have no bars."""
    def __init__(self, name="fakehist", data=None, splits=None, error=None, errors=None, raw_closes=False):
        self.name, self._data, self._splits, self._error = name, data or {}, splits or [], error
        self._errors = errors or {}
        self.closes_are_raw = raw_closes  # True for a source whose closes are the traded prices, as Massive's are
    def bars(self, symbol, start, end):
        if self._error: raise SourceError(self._error)
        if symbol in self._errors: raise self._errors[symbol]
        if symbol not in self._data: raise SourceNoData(f"{self.name}: no bars for {symbol} between {start} and {end}")
        s = self._data[symbol]
        return BarSeries(symbol=symbol, source=self.name, bars=[b for b in s.bars if start <= b.date <= end])
    def corporate_actions(self, symbol, start, end):
        return [a for a in self._splits if start <= a.date <= end]

class FakeNewsSource:
    def __init__(self, name="fakenews", headlines=None, error=None):
        self.name, self._hs, self._error = name, headlines or [], error
    def headlines(self, symbol, since, until):
        if self._error: raise SourceError(self._error)
        return list(self._hs)

def acme_candidate(source="massive", pct=3.738):
    return Candidate(symbol="ACME", name="Acme Corp", prev_close=10.70, close=11.10, pct_change=pct, volume=1000, source=source)

def clock_after_close():
    # The market closes at 16:00 America/New_York. 18:00 UTC is only
    # 14:00 New York (EDT, UTC-4 in September) -- still inside the trading session -- so a
    # clock meant to represent "after the close" must be timezone-converted, not just any UTC
    # afternoon. 22:00 UTC is 18:00 New York, two hours past the 16:00 close.
    return datetime(2026, 9, 24, 22, 0, tzinfo=timezone.utc).astimezone(timezone.utc)
