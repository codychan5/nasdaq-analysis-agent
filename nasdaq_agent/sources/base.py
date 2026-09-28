from datetime import date, datetime
from typing import Protocol
from .models import BarSeries, Candidate, CorporateAction, Headline


class GainerSource(Protocol):
    name: str
    requires_market_closed: bool
    def top_candidates(self, session_date: date, prev_session_date: date, limit: int) -> list[Candidate]: ...


class HistorySource(Protocol):
    name: str
    def bars(self, symbol: str, start: date, end: date) -> BarSeries: ...
    def corporate_actions(self, symbol: str, start: date, end: date) -> list[CorporateAction]: ...


class NewsSource(Protocol):
    name: str
    # The window runs from `since` to `until`, the run's clock. A source that can filter by date asks for exactly that
    # window; get_news drops anything later in any case.
    def headlines(self, symbol: str, since: datetime, until: datetime) -> list[Headline]: ...
