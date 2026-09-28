import logging
from datetime import date, datetime, timezone
from typing import Any

from ..universe import Universe
from .errors import SourceError, SourceNoData
from .http import HttpClient
from .models import Bar, BarSeries, Candidate, CorporateAction, Headline, clean_summary

MASSIVE_BASE_URL = "https://api.massive.com"
MASSIVE_CALLS_PER_MINUTE = 5

log = logging.getLogger(__name__)


def _grouped_closes(client: HttpClient, api_key: str, day: date) -> tuple[dict[str, dict[str, Any]], int]:
    """Return {ticker: row} for well-formed rows, plus a count of rows dropped for missing "T" or "c"."""
    data = client.get_json(f"{MASSIVE_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}",
                           params={"adjusted": "false", "apiKey": api_key})
    rows = data.get("results") or []
    if not rows:
        raise SourceError(f"massive: no grouped results for {day}")
    closes: dict[str, dict[str, Any]] = {}
    dropped = 0
    for r in rows:
        if "T" in r and "c" in r:
            closes[r["T"]] = r
        else:
            dropped += 1
    return closes, dropped


class MassiveGainerSource:
    name = "massive"
    requires_market_closed = False

    def __init__(self, client: HttpClient, api_key: str, universe: Universe):
        self._client, self._api_key, self._universe = client, api_key, universe

    def top_candidates(self, session_date: date, prev_session_date: date, limit: int) -> list[Candidate]:
        today, today_dropped = _grouped_closes(self._client, self._api_key, session_date)
        prev, prev_dropped = _grouped_closes(self._client, self._api_key, prev_session_date)
        out: list[Candidate] = []
        skipped = today_dropped + prev_dropped
        for symbol, row in today.items():
            try:
                canonical = self._universe.to_canonical(symbol, "massive")
                prev_row = prev.get(symbol)
                if prev_row is None or not self._universe.is_common_stock(canonical):
                    continue
                prev_close, close = float(prev_row["c"]), float(row["c"])
                if prev_close <= 0:
                    continue
                out.append(Candidate(symbol=canonical, name=self._universe.record(canonical).name,
                                     prev_close=prev_close, close=close, pct_change=(close / prev_close - 1) * 100,
                                     volume=float(row.get("v", 0)), source=self.name))
            except (KeyError, ValueError, TypeError):
                skipped += 1
        if skipped:
            log.warning("%s: skipped %d malformed gainer row(s)", self.name, skipped)
        out.sort(key=lambda c: c.pct_change, reverse=True)
        return out[:limit]


NEWS_LIMIT = 10
# How the news endpoint's published_utc filters read a moment: RFC 3339 in UTC.
PUBLISHED_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
MS_PER_SECOND = 1000


class MassiveHistorySource:
    name = "massive"

    def __init__(self, client: HttpClient, api_key: str, universe: Universe):
        self._client, self._api_key, self._universe = client, api_key, universe

    def _aggs(self, symbol: str, start: date, end: date, adjusted: bool) -> dict[date, dict[str, Any]]:
        url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        data = self._client.get_json(url, params={"adjusted": str(adjusted).lower(), "sort": "asc", "apiKey": self._api_key})
        return {datetime.fromtimestamp(r["t"] / MS_PER_SECOND, tz=timezone.utc).date(): r for r in data.get("results") or []}

    def bars(self, symbol: str, start: date, end: date) -> BarSeries:
        provider_symbol = self._universe.to_provider(symbol, "massive")
        raw, adj = self._aggs(provider_symbol, start, end, False), self._aggs(provider_symbol, start, end, True)
        if not raw:
            raise SourceNoData(f"massive: no bars for {symbol} between {start} and {end}")
        bars = [Bar(date=d, open=r["o"], high=r["h"], low=r["l"], close=r["c"],
                    adj_close=adj.get(d, r)["c"], volume=r.get("v", 0)) for d, r in sorted(raw.items())]
        return BarSeries(symbol=symbol, bars=bars, source=self.name)

    def corporate_actions(self, symbol: str, start: date, end: date) -> list[CorporateAction]:
        data = self._client.get_json(f"{MASSIVE_BASE_URL}/v3/reference/splits",
                                     params={"ticker": self._universe.to_provider(symbol, "massive"),
                                             "execution_date.gte": start.isoformat(),
                                             "execution_date.lte": end.isoformat(), "apiKey": self._api_key})
        return [CorporateAction(date=date.fromisoformat(r["execution_date"]), kind="split",
                                ratio=float(r["split_to"]) / float(r["split_from"])) for r in data.get("results") or []]


class MassiveNewsSource:
    name = "massive"

    def __init__(self, client: HttpClient, api_key: str):
        self._client, self._api_key = client, api_key

    def headlines(self, symbol: str, since: datetime, until: datetime) -> list[Headline]:
        # Both ends of the window go in the request. The endpoint fills its NEWS_LIMIT slots newest first, so without
        # the end a run pinned to a past session would get only stories written after it.
        data = self._client.get_json(f"{MASSIVE_BASE_URL}/v2/reference/news",
                                     params={"ticker": symbol, "published_utc.gte": since.date().isoformat(),
                                             "published_utc.lte": until.astimezone(timezone.utc).strftime(PUBLISHED_UTC_FORMAT),
                                             "order": "desc", "limit": NEWS_LIMIT, "apiKey": self._api_key})
        out: list[Headline] = []
        for i, r in enumerate(data.get("results") or [], start=1):
            sentiment = next((ins.get("sentiment") for ins in r.get("insights", []) if ins.get("ticker") == symbol), None)
            out.append(Headline(id=i, title=r["title"], provider=(r.get("publisher") or {}).get("name", "unknown"),
                                published=datetime.fromisoformat(r["published_utc"].replace("Z", "+00:00")),
                                url=r.get("article_url"), provider_sentiment=sentiment,
                                summary=clean_summary(r.get("description"))))
        return out
