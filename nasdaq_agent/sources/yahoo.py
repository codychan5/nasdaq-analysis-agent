import logging
from datetime import date, datetime
from typing import Any, Callable

from ..universe import SessionUniverse, Universe
from .errors import SourceError, SourceNoData
from .models import Bar, BarSeries, Candidate, CorporateAction, Headline, clean_summary, rank_candidates

NASDAQ_EXCHANGE_CODES = ["NMS", "NGM", "NCM"]
SCREENER_PAGE_SIZE = 100

log = logging.getLogger(__name__)


def _default_screen(query, sortField, sortAsc, size):
    import yfinance as yf
    return yf.screen(query, sortField=sortField, sortAsc=sortAsc, size=size)


def _nasdaq_query():
    from yfinance import EquityQuery
    return EquityQuery("is-in", ["exchange", *NASDAQ_EXCHANGE_CODES])


class YahooScreenerGainerSource:
    name = "yahoo"
    requires_market_closed = True  # a live screener shows the current session while the market is open

    def __init__(self, universe: SessionUniverse, screen_fn: Callable = _default_screen):
        self._universe, self._screen = universe, screen_fn

    def top_candidates(self, session_date: date, prev_session_date: date, limit: int) -> list[Candidate]:
        query = _nasdaq_query() if self._screen is _default_screen else None
        data = self._screen(query, sortField="percentchange", sortAsc=False, size=SCREENER_PAGE_SIZE)
        quotes = (data or {}).get("quotes") or []
        if not quotes:
            raise SourceError("yahoo: screener returned no quotes")
        listing = self._universe.for_session(session_date)
        out: list[Candidate] = []
        skipped = 0
        for q in quotes:
            try:
                symbol = listing.to_canonical(q["symbol"], "yahoo")
                out.append(Candidate(symbol=symbol, name=q.get("shortName"), prev_close=q.get("regularMarketPreviousClose"),
                                     close=float(q["regularMarketPrice"]), pct_change=float(q["regularMarketChangePercent"]),
                                     volume=q.get("regularMarketVolume"), market_cap=q.get("marketCap"), source=self.name,
                                     excluded=listing.exclusion(symbol)))
            except (KeyError, ValueError, TypeError):
                skipped += 1
        if skipped:
            log.warning("%s: skipped %d malformed gainer row(s)", self.name, skipped)
        return rank_candidates(out, limit)


NEWS_LIMIT = 10


def _default_ticker(symbol: str):
    import yfinance as yf
    return yf.Ticker(symbol)


class YfinanceHistorySource:
    name = "yfinance"

    def __init__(self, universe: Universe, ticker_factory: Callable = _default_ticker):
        self._universe, self._ticker = universe, ticker_factory
        # One download per window for the life of this source, which is one run: find_top_gainer and the price check
        # ask for a window's bars and then its splits, and each call used to download the same history again. A failed
        # download raises before anything is kept.
        self._frames: dict[tuple[str, date, date], Any] = {}

    def _frame(self, symbol: str, start: date, end: date):
        provider_symbol = self._universe.to_provider(symbol, "yahoo")
        key = (provider_symbol, start, end)
        if key not in self._frames:
            # yfinance treats `end` as exclusive, so ask for one day more.
            from datetime import timedelta
            self._frames[key] = self._ticker(provider_symbol).history(
                start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), auto_adjust=False, actions=True)
        return self._frames[key]

    def bars(self, symbol: str, start: date, end: date) -> BarSeries:
        df = self._frame(symbol, start, end)
        if df is None or df.empty:
            raise SourceNoData(f"yfinance: no bars for {symbol} between {start} and {end}")
        adj_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        bars = [Bar(date=ts.date(), open=float(row["Open"]), high=float(row["High"]), low=float(row["Low"]),
                    close=float(row["Close"]), adj_close=float(row[adj_col]), volume=float(row["Volume"]))
                for ts, row in df.sort_index().iterrows()]
        return BarSeries(symbol=symbol, bars=bars, source=self.name)

    def corporate_actions(self, symbol: str, start: date, end: date) -> list[CorporateAction]:
        df = self._frame(symbol, start, end)
        if df is None or df.empty or "Stock Splits" not in df.columns:
            return []
        return [CorporateAction(date=ts.date(), kind="split", ratio=float(v))
                for ts, v in df["Stock Splits"].items() if float(v) != 0.0]


class YfinanceNewsSource:
    name = "yfinance"

    def __init__(self, ticker_factory: Callable = _default_ticker):
        self._ticker = ticker_factory

    def headlines(self, symbol: str, since: datetime, until: datetime) -> list[Headline]:
        # yfinance cannot filter by date: it returns the latest NEWS_LIMIT items, and get_news drops those after `until`.
        items = self._ticker(Universe.to_provider(symbol, "yahoo")).get_news(count=NEWS_LIMIT) or []
        out: list[Headline] = []
        for item in items:
            c = item.get("content") or {}
            published = datetime.fromisoformat(c["pubDate"].replace("Z", "+00:00")) if c.get("pubDate") else None
            if published is not None and published < since:
                continue
            out.append(Headline(id=len(out) + 1, title=c.get("title", "").strip(),
                                provider=((c.get("provider") or {}).get("displayName")) or "yahoo",
                                published=published, url=((c.get("canonicalUrl") or {}).get("url")),
                                summary=clean_summary(c.get("summary"))))
        return out
