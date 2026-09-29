# tests/unit/test_history_news_sources.py
from datetime import date, datetime, timezone
import httpx, pandas as pd, pytest, respx
from tests.unit.test_universe import SAMPLE

@pytest.fixture
def universe():
    from nasdaq_agent.universe import Universe
    return Universe.from_text(SAMPLE)

class FakeTicker:
    def __init__(self, symbol): self.symbol = symbol
    def history(self, start, end, auto_adjust, actions=True):
        idx = pd.to_datetime(["2026-09-23", "2026-09-24"])
        return pd.DataFrame({"Open": [10.5, 10.75], "High": [10.8, 11.2], "Low": [10.5, 10.7],
                             "Close": [10.7, 11.1], "Adj Close": [10.7, 11.1], "Volume": [140000, 160000],
                             "Dividends": [0.0, 0.0], "Stock Splits": [0.0, 0.1]}, index=idx)
    def get_news(self, count=10, tab="news"):
        return [{"content": {"title": "Acme files for FDA review", "pubDate": "2026-09-24T13:00:00Z",
                             "summary": "  Acme asked regulators to review its lead drug.  ",
                             "provider": {"displayName": "Reuters"}, "canonicalUrl": {"url": "https://r.example/1"}}}]

def test_yfinance_history_and_actions(universe):
    from nasdaq_agent.sources.yahoo import YfinanceHistorySource
    src = YfinanceHistorySource(universe, ticker_factory=FakeTicker)
    s = src.bars("BRK.B", date(2026, 9, 23), date(2026, 9, 24))
    assert s.symbol == "BRK.B" and s.closes() == [10.7, 11.1] and s.dates()[-1] == date(2026, 9, 24)
    acts = src.corporate_actions("BRK.B", date(2026, 9, 23), date(2026, 9, 24))
    assert acts and acts[0].kind == "split" and acts[0].date == date(2026, 9, 24)

def test_yfinance_news_nested_format():
    from nasdaq_agent.sources.yahoo import YfinanceNewsSource
    src = YfinanceNewsSource(ticker_factory=FakeTicker)
    hs = src.headlines("BRK.B", since=datetime(2026, 9, 20, tzinfo=timezone.utc), until=datetime(2026, 9, 25, tzinfo=timezone.utc))
    assert hs[0].id == 1 and hs[0].title == "Acme files for FDA review" and hs[0].provider == "Reuters"
    assert hs[0].summary == "Acme asked regulators to review its lead drug."

@respx.mock
def test_massive_history_merges_adjusted_and_raw(universe):
    from nasdaq_agent.sources.http import HttpClient
    from nasdaq_agent.sources.massive import MassiveHistorySource, MASSIVE_BASE_URL
    base = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/AAPL/range/1/day/2026-09-23/2026-09-24"
    t1, t2 = 1790121600000, 1790208000000  # 2026-09-23 and 2026-09-24 00:00 UTC in ms
    def handler(request):
        adjusted = request.url.params.get("adjusted") == "true"
        c = [10.5, 11.0] if adjusted else [10.7, 11.1]
        return httpx.Response(200, json={"results": [
            {"t": t1, "o": 1, "h": 1, "l": 1, "c": c[0], "v": 10}, {"t": t2, "o": 1, "h": 1, "l": 1, "c": c[1], "v": 11}]})
    respx.get(base).mock(side_effect=handler)
    respx.get(f"{MASSIVE_BASE_URL}/v3/reference/splits").mock(return_value=httpx.Response(200, json={"results": []}))
    src = MassiveHistorySource(HttpClient(1, 1), api_key="k", universe=universe)
    s = src.bars("AAPL", date(2026, 9, 23), date(2026, 9, 24))
    assert s.closes() == [10.7, 11.1] and s.adj_closes() == [10.5, 11.0]
    assert s.dates() == [date(2026, 9, 23), date(2026, 9, 24)]
    assert src.corporate_actions("AAPL", date(2026, 9, 23), date(2026, 9, 24)) == []

class EmptyTicker(FakeTicker):
    def history(self, start, end, auto_adjust, actions=True):
        return pd.DataFrame()

def test_yfinance_history_without_bars_says_it_has_no_data(universe):
    """"No bars" answers a question about the data, unlike a failure to answer, so it has its own error class."""
    from nasdaq_agent.sources.errors import SourceNoData
    from nasdaq_agent.sources.yahoo import YfinanceHistorySource
    with pytest.raises(SourceNoData):
        YfinanceHistorySource(universe, ticker_factory=EmptyTicker).bars("AAPL", date(2026, 9, 23), date(2026, 9, 24))

@respx.mock
def test_massive_history_without_bars_says_it_has_no_data(universe):
    from nasdaq_agent.sources.errors import SourceNoData
    from nasdaq_agent.sources.http import HttpClient
    from nasdaq_agent.sources.massive import MassiveHistorySource, MASSIVE_BASE_URL
    respx.get(url__startswith=f"{MASSIVE_BASE_URL}/v2/aggs/ticker/AAPL/").mock(
        return_value=httpx.Response(200, json={"results": []}))
    src = MassiveHistorySource(HttpClient(1, 1), api_key="k", universe=universe)
    with pytest.raises(SourceNoData):
        src.bars("AAPL", date(2026, 9, 23), date(2026, 9, 24))

@respx.mock
def test_massive_news_keeps_insights():
    from nasdaq_agent.sources.http import HttpClient
    from nasdaq_agent.sources.massive import MassiveNewsSource, MASSIVE_BASE_URL
    respx.get(f"{MASSIVE_BASE_URL}/v2/reference/news").mock(return_value=httpx.Response(200, json={"results": [
        {"title": "Acme raises guidance", "publisher": {"name": "Business Wire"}, "published_utc": "2026-09-24T12:00:00Z",
         "article_url": "https://bw.example/2", "description": "Acme lifted its full-year revenue outlook.",
         "insights": [{"ticker": "AAPL", "sentiment": "positive"}]}]}))
    hs = MassiveNewsSource(HttpClient(1, 1), api_key="k").headlines("AAPL", since=datetime(2026, 9, 20, tzinfo=timezone.utc),
                                                                  until=datetime(2026, 9, 25, tzinfo=timezone.utc))
    assert hs[0].provider_sentiment == "positive" and hs[0].id == 1
    assert hs[0].summary == "Acme lifted its full-year revenue outlook."

@respx.mock
def test_massive_news_asks_only_for_stories_published_by_the_end_of_the_window():
    # Massive answers with the newest stories first, ten at most. Asked with a start date alone, a run pinned to a past
    # session got stories written since then in every slot, and get_news dropped them all as later than its clock.
    from nasdaq_agent.sources.http import HttpClient
    from nasdaq_agent.sources.massive import MassiveNewsSource, MASSIVE_BASE_URL
    route = respx.get(f"{MASSIVE_BASE_URL}/v2/reference/news").mock(return_value=httpx.Response(200, json={"results": []}))
    MassiveNewsSource(HttpClient(1, 1), api_key="k").headlines(
        "WETO", since=datetime(2026, 8, 11, tzinfo=timezone.utc), until=datetime.fromisoformat("2026-08-15T04:30:00+08:00"))
    params = route.calls.last.request.url.params
    assert params["published_utc.gte"] == "2026-08-11"
    assert params["published_utc.lte"] == "2026-08-14T20:30:00Z"  # the same moment in UTC, which Massive reads


def test_a_long_article_summary_is_cut_to_a_bounded_length_and_a_blank_one_is_none():
    from nasdaq_agent.sources.models import SUMMARY_MAX_CHARS, clean_summary
    assert len(clean_summary("x" * (SUMMARY_MAX_CHARS + 50))) == SUMMARY_MAX_CHARS
    assert clean_summary("   ") is None and clean_summary(None) is None
    assert clean_summary("Line one.\n\nLine   two.") == "Line one. Line two."

def test_yfinance_downloads_each_window_once_for_its_bars_and_splits(universe):
    """find_top_gainer and the price check ask for a window's bars and then its splits: one download serves both, and a
    different window still gets its own."""
    from nasdaq_agent.sources.yahoo import YfinanceHistorySource
    downloads = []

    class CountingTicker(FakeTicker):
        def history(self, *args, **kwargs):
            downloads.append((self.symbol, kwargs.get("start"), kwargs.get("end")))
            return super().history(*args, **kwargs)

    src = YfinanceHistorySource(universe, ticker_factory=CountingTicker)
    src.bars("BRK.B", date(2026, 9, 23), date(2026, 9, 24))
    src.corporate_actions("BRK.B", date(2026, 9, 23), date(2026, 9, 24))
    assert len(downloads) == 1
    src.corporate_actions("BRK.B", date(2026, 9, 16), date(2026, 9, 24))
    assert len(downloads) == 2
