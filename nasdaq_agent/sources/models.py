from datetime import date, datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict


class Bar(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float


class BarSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    bars: list[Bar]
    source: str

    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    def adj_closes(self) -> list[float]:
        return [b.adj_close for b in self.bars]

    def dates(self) -> list[date]:
        return [b.date for b in self.bars]


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    name: str | None = None
    prev_close: float | None = None
    close: float
    pct_change: float
    volume: float | None = None
    market_cap: float | None = None
    source: str


# The longest article summary kept from a news provider: enough for the key facts of a story, bounded so twenty
# headlines cannot swell the prompt.
SUMMARY_MAX_CHARS = 1000


def clean_summary(text: str | None) -> str | None:
    """A provider's article summary with its whitespace collapsed and its length bounded; None when there is none."""
    collapsed = " ".join((text or "").split())
    return collapsed[:SUMMARY_MAX_CHARS] or None


class Headline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    title: str
    provider: str
    published: datetime | None = None
    url: str | None = None
    provider_sentiment: str | None = None
    # The news source (API) the headline came from, tagged by get_news; provider is its publisher.
    source: str | None = None
    # The provider's own summary of the article (Yahoo's summary, Massive's description): untrusted quoted text, the
    # material the agent summarises each story from.
    summary: str | None = None


def _aware(moment: datetime) -> datetime:
    """A naive timestamp is read as UTC, the providers' convention, so aware and naive ones compare safely."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def published_by(headline: Headline, moment: datetime) -> bool:
    """Whether the headline existed at `moment`. An undated headline is kept: nothing says it is later."""
    return headline.published is None or _aware(headline.published) <= _aware(moment)


def recency_key(published: datetime | None) -> tuple[int, float]:
    """Sort key for most recent first, undated last."""
    return (1, 0.0) if published is None else (0, -_aware(published).timestamp())


def most_recent_first(headlines: list[Headline]) -> list[Headline]:
    """The headlines most recent first, those without a publication time last. Ties keep their given order, since the
    sort is stable."""
    return sorted(headlines, key=lambda h: recency_key(h.published))


def first_published(headlines: list[Headline]) -> datetime | None:
    """When a story first appeared: the earliest publication time among the headlines that report it, or None when
    none of them is dated."""
    dated = [_aware(h.published) for h in headlines if h.published is not None]
    return min(dated) if dated else None


class CorporateAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: date
    kind: Literal["split", "dividend"]
    ratio: float
