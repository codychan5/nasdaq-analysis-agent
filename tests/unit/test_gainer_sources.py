# tests/unit/test_gainer_sources.py
import logging
from datetime import date
import httpx, pytest, respx
from tests.unit.test_universe import SAMPLE

SESSION, PREV = date(2026, 9, 24), date(2026, 9, 23)


def counted(candidates):
    """The candidates that count, in the order the source ranked them."""
    return [c for c in candidates if c.excluded is None]

@pytest.fixture
def universe():
    from nasdaq_agent.universe import Universe
    return Universe.from_text(SAMPLE)

@pytest.fixture
def client():
    from nasdaq_agent.sources.http import HttpClient
    return HttpClient(connect_timeout=1, read_timeout=1)

@respx.mock
def test_massive_grouped_daily_computes_pct_and_filters(universe, client, caplog):
    from nasdaq_agent.sources.massive import MassiveGainerSource, MASSIVE_BASE_URL
    respx.get(f"{MASSIVE_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/2026-09-24").mock(
        return_value=httpx.Response(200, json={"results": [
            {"T": "AAPL", "c": 11.10, "v": 1000}, {"T": "TSLA", "c": 20.0, "v": 500},
            {"T": "QQQ", "c": 999.0, "v": 5}, {"T": "ASTLW", "c": 0.02, "v": 5},
            {"T": "BABA", "c": None, "v": 1}, {"T": "BRK.B", "v": 1}]}))
    respx.get(f"{MASSIVE_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/2026-09-23").mock(
        return_value=httpx.Response(200, json={"results": [
            {"T": "AAPL", "c": 10.0, "v": 900}, {"T": "TSLA", "c": 25.0, "v": 400},
            {"T": "QQQ", "c": 100.0, "v": 5}, {"T": "ASTLW", "c": 0.001, "v": 5},
            {"T": "BABA", "c": 10.0, "v": 1}, {"T": "BRK.B", "c": 10.0, "v": 1}]}))
    src = MassiveGainerSource(client, api_key="k", universe=universe)
    with caplog.at_level(logging.WARNING, logger="nasdaq_agent.sources.massive"):
        got = src.top_candidates(SESSION, PREV, limit=5)
    # The warrant and the ETF ranked above AAPL are returned too, marked with the reason they do not count.
    assert [c.symbol for c in counted(got)] == ["AAPL", "TSLA"]
    assert {c.symbol: c.excluded for c in got if c.excluded} == {"ASTLW": "excluded as warrant", "QQQ": "excluded as etf"}
    aapl = counted(got)[0]
    assert abs(aapl.pct_change - 11.0) < 0.01 and aapl.prev_close == 10.0 and aapl.source == "massive"
    assert src.requires_market_closed is False
    # BRK.B (today, missing "c") is dropped inside _grouped_closes; BABA (today, c=None) fails the
    # per-row parse. Both must still be counted in the single skipped-row warning.
    massive_warnings = [r for r in caplog.records if r.name == "nasdaq_agent.sources.massive"]
    assert len(massive_warnings) == 1
    assert "skipped 2 malformed gainer row" in massive_warnings[0].getMessage()

@respx.mock
def test_alphavantage_parses_and_filters(universe, client, tmp_path):
    from nasdaq_agent.sources.alphavantage import AlphaVantageGainerSource
    from nasdaq_agent.sources.http import DailyQuota
    respx.get("https://www.alphavantage.co/query").mock(return_value=httpx.Response(200, json={
        "last_updated": "2026-09-24 16:15:58 US/Eastern",
        "top_gainers": [
            {"ticker": "SLND+", "price": "1.20", "change_amount": "1.10", "change_percentage": "1100.0%", "volume": "5"},
            {"ticker": "AAPL", "price": "11.10", "change_amount": "1.10", "change_percentage": "11.0%", "volume": "1000"},
            {"ticker": "TSLA", "price": "20.0", "change_amount": "1.0", "change_percentage": "N/A", "volume": "500"},
        ]}))
    quota = DailyQuota("alphavantage", limit=25, state_path=tmp_path / "q.json")
    src = AlphaVantageGainerSource(client, api_key="k", universe=universe, quota=quota)
    got = src.top_candidates(SESSION, PREV, limit=20)
    assert [c.symbol for c in counted(got)] == ["AAPL"] and abs(counted(got)[0].prev_close - 10.0) < 1e-9
    assert [(c.symbol, c.excluded) for c in got if c.excluded] == [("SLND+", "not on NASDAQ's list of listed securities")]

def test_yahoo_screener_wraps_yfinance(universe):
    from nasdaq_agent.sources.yahoo import YahooScreenerGainerSource
    def fake_screen(query, sortField, sortAsc, size):
        return {"quotes": [
            {"symbol": "BRK-B", "shortName": "Berkshire", "regularMarketPrice": 11.1,
             "regularMarketPreviousClose": 10.0, "regularMarketChangePercent": 11.0,
             "regularMarketVolume": 100, "marketCap": 5e9},
            {"symbol": "ASTLW", "regularMarketPrice": 0.02, "regularMarketPreviousClose": 0.01,
             "regularMarketChangePercent": 100.0, "regularMarketVolume": 5},
            {"symbol": "AAPL", "regularMarketPreviousClose": 10.0, "regularMarketChangePercent": 5.0,
             "regularMarketVolume": 10}]}
    src = YahooScreenerGainerSource(universe, screen_fn=fake_screen)
    got = src.top_candidates(SESSION, PREV, limit=10)
    assert [c.symbol for c in counted(got)] == ["BRK.B"] and src.requires_market_closed is True
    assert [(c.symbol, c.excluded) for c in got if c.excluded] == [("ASTLW", "excluded as warrant")]

@respx.mock
def test_nasdaqcom_parses_download(universe, client):
    from nasdaq_agent.sources.nasdaqcom import NasdaqComGainerSource, NASDAQCOM_URL
    route = respx.get(NASDAQCOM_URL).mock(return_value=httpx.Response(200, json={"data": {"rows": [
        {"symbol": "AAPL", "name": "Apple", "lastsale": "$11.10", "netchange": "1.10", "pctchange": "11.0%",
         "volume": "1,000", "marketCap": "1,000,000"},
        {"symbol": "ZTEST", "name": "test", "lastsale": "$1.00", "netchange": "0.90", "pctchange": "900%",
         "volume": "1", "marketCap": "1"},
        {"symbol": "TSLA", "name": "Tesla", "lastsale": "$20.00", "netchange": "0.00", "pctchange": "UNCH",
         "volume": "1", "marketCap": "1"}]}}))
    src = NasdaqComGainerSource(client, universe)
    got = src.top_candidates(SESSION, PREV, limit=10)
    assert [c.symbol for c in counted(got)] == ["AAPL"] and abs(counted(got)[0].prev_close - 10.0) < 1e-9
    assert [(c.symbol, c.excluded) for c in got if c.excluded] == [("ZTEST", "excluded as test issue")]
    assert "Mozilla" in route.calls[0].request.headers["User-Agent"] and src.requires_market_closed is True

def test_registry_order_and_key_gating(universe, tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("MASSIVE_API_KEY", "m")
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    from nasdaq_agent.config import Settings
    from nasdaq_agent.sources.registry import build_gainer_sources, massive_rate_limiter
    names = [s.name for s in build_gainer_sources(Settings(_env_file=None), universe, tmp_path,
                                                  massive_limiter=massive_rate_limiter())]
    assert names == ["massive", "yahoo", "nasdaqcom"]


def _candidate(symbol, pct, excluded=None):
    from nasdaq_agent.sources.models import Candidate
    return Candidate(symbol=symbol, close=1.0, pct_change=pct, source="massive", excluded=excluded)


def test_ranking_stops_after_the_limit_of_stocks_that_count_and_keeps_those_excluded_above_it():
    from nasdaq_agent.sources.models import rank_candidates
    ranked = rank_candidates([_candidate("C", 30.0), _candidate("W", 90.0, "excluded as warrant"), _candidate("A", 50.0),
                              _candidate("E", 10.0, "excluded as etf"), _candidate("B", 40.0), _candidate("D", 20.0)],
                             limit=2)
    assert [(c.symbol, c.excluded) for c in ranked] == [("W", "excluded as warrant"), ("A", None), ("B", None)]


def test_ranking_keeps_at_most_the_cap_of_excluded_stocks():
    from nasdaq_agent.sources.models import rank_candidates
    many = [_candidate(f"X{i}", 100.0 - i, "excluded as warrant") for i in range(10)] + [_candidate("A", 1.0)]
    ranked = rank_candidates(many, limit=1, max_excluded=3)
    assert [c.symbol for c in ranked] == ["X0", "X1", "X2", "A"]


NVVE_DAY, NVVE_PREV = date(2026, 7, 8), date(2026, 7, 7)


def _grouped(route_day, rows):
    from nasdaq_agent.sources.massive import MASSIVE_BASE_URL
    respx.get(f"{MASSIVE_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{route_day}").mock(
        return_value=httpx.Response(200, json={"results": rows}))


def _nvve_bars(session, prev):
    """2026-07-08 as it happened: NVVE +63.58%, TVRD +61.29%, and the warrant NXGLW +114.66% above both."""
    _grouped(session.isoformat(), [{"T": "NVVE", "c": 8.49, "v": 1}, {"T": "TVRD", "c": 5.0, "v": 1},
                                   {"T": "NXGLW", "c": 0.3006, "v": 1}])
    _grouped(prev.isoformat(), [{"T": "NVVE", "c": 5.19, "v": 1}, {"T": "TVRD", "c": 3.1, "v": 1},
                                {"T": "NXGLW", "c": 0.14, "v": 1}])


@respx.mock
def test_massive_counts_a_stock_listed_that_day_though_delisted_since(client, tmp_path):
    """The 2026-07-08 report named TVRD because today's file no longer lists NVVE. Filtered with the list for that
    day, NVVE is the top stock that counts."""
    from nasdaq_agent.sources.massive import MassiveGainerSource
    from nasdaq_agent.universe import Universe, UniverseByDate
    from tests.unit.test_session_universe import TODAY, FetchLog
    _nvve_bars(NVVE_DAY, NVVE_PREV)
    src = MassiveGainerSource(client, api_key="k", universe=UniverseByDate(Universe.from_text(TODAY), FetchLog(), tmp_path))
    got = src.top_candidates(NVVE_DAY, NVVE_PREV, limit=5)
    assert [c.symbol for c in counted(got)] == ["NVVE", "TVRD"]
    assert counted(got)[0].name == "Nuvve Holding Corp. Common Stock"
    assert [(c.symbol, c.excluded) for c in got if c.excluded] == [("NXGLW", "excluded as warrant")]


@respx.mock
def test_massive_filters_a_session_on_the_files_day_with_the_file(client, tmp_path):
    from nasdaq_agent.sources.massive import MassiveGainerSource
    from nasdaq_agent.universe import Universe, UniverseByDate
    from tests.unit.test_session_universe import TODAY, FetchLog
    day, prev = date(2026, 9, 28), date(2026, 9, 25)
    _nvve_bars(day, prev)
    fetch = FetchLog()
    src = MassiveGainerSource(client, api_key="k", universe=UniverseByDate(Universe.from_text(TODAY), fetch, tmp_path))
    got = src.top_candidates(day, prev, limit=5)
    assert [c.symbol for c in counted(got)] == ["TVRD"] and fetch.days == []
    assert ("NVVE", "not on NASDAQ's list of listed securities") in [(c.symbol, c.excluded) for c in got]
