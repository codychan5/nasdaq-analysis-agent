# tests/unit/test_session_universe.py
"""Which stocks count for a session. NASDAQ's symbol file lists only the securities listed on the day it was created,
so a run for an earlier session asks Massive which securities were listed on NASDAQ that day. A stock delisted since,
like NVVE after 2026-07-08, then still counts on the days it was listed."""
import json
from datetime import date

import httpx
import pytest
import respx

HEADER = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"
# Today's file, created on 2026-09-28. NVVE is gone; IPOX was listed after 2026-07-08.
TODAY = "\n".join([
    HEADER,
    "TVRD|Tvardi Therapeutics, Inc. - Common Stock|S|N|N|100|N|N",
    "PFBC|Preferred Bank - Common Stock|Q|N|N|100|N|N",
    "DHC|Diversified Healthcare Trust - Common Shares of Beneficial Interest|S|N|N|100|N|N",
    ("ACGLN|Arch Capital Group Ltd. - Depositary Shares, each representing 1/1,000th interest in a share of 4.55% "
     "Non-Cumulative Preferred Shares, Series G|Q|N|N|100|N|N"),
    "NXGLW|NexGel, Inc - Warrant|S|N|N|100|N|N",
    "IPOX|Ipox Holdings Inc. - Common Stock|S|N|N|100|N|N",
    "File Creation Time: 0928202609:46|||||||",
    ""])
DAY = date(2026, 7, 8)
FILE_DAY = date(2026, 9, 28)


def listed_then():
    """Massive's NASDAQ list for 2026-07-08, with Massive's own security types: coarser than the symbol file's names,
    so the depositary preferred ACGLN is "CS" and the trust DHC is "FUND"."""
    from nasdaq_agent.universe import ListedSecurity
    return [ListedSecurity("TVRD", "Tvardi Therapeutics, Inc. Common Stock", "CS"),
            ListedSecurity("PFBC", "Preferred Bank Common Stock", "CS"),
            ListedSecurity("DHC", "Diversified Healthcare Trust Common Shares of Beneficial Interest", "FUND"),
            ListedSecurity("ACGLN", "Arch Capital Group Ltd. Depositary Shares 4.55% Preferred Series G", "CS"),
            ListedSecurity("NXGLW", "NexGel, Inc Warrant", "WARRANT"),
            ListedSecurity("NVVE", "Nuvve Holding Corp. Common Stock", "CS"),
            ListedSecurity("OLDW", "Old Acquisition Corp. Warrants", "WARRANT"),
            ListedSecurity("OLDP", "Old Bank Corp. Depositary Shares Series A Preferred Stock", "CS"),
            ListedSecurity("OLDA", "Old Pharma Ltd. American Depositary Shares", "ADRC"),
            ListedSecurity("OLDF", "Old Income Fund Shares of Beneficial Interest", "FUND")]


@pytest.fixture
def today():
    from nasdaq_agent.universe import Universe
    return Universe.from_text(TODAY)


def test_the_symbol_file_says_which_day_it_lists(today):
    from nasdaq_agent.universe import Universe
    assert today.listed_on == FILE_DAY
    assert Universe.from_text(TODAY.replace("File Creation Time: 0928202609:46|||||||", "")).listed_on is None


def test_a_fixed_list_answers_for_every_session(today):
    assert today.for_session(DAY) is today


def test_exclusion_names_the_reason_a_stock_does_not_count(today):
    assert today.exclusion("TVRD") is None
    assert "warrant" in today.exclusion("NXGLW")
    assert "not" in today.exclusion("NVVE")


def test_the_days_list_keeps_stocks_delisted_since_and_drops_those_listed_later(today):
    from nasdaq_agent.universe import Universe
    day = Universe.for_day(today, listed_then(), DAY)
    assert day.listed_on == DAY and day.source == "massive"
    assert day.is_common_stock("NVVE") and day.is_common_stock("OLDA")
    assert not day.is_common_stock("IPOX") and "2026-07-08" in day.exclusion("IPOX")


def test_the_days_list_classifies_a_stock_still_listed_by_todays_file(today):
    """Massive's labels are coarser than the file's security names, so a stock NASDAQ still lists keeps the file's
    classification: the bank named Preferred and the trust count, the depositary preferred does not."""
    from nasdaq_agent.universe import Universe
    day = Universe.for_day(today, listed_then(), DAY)
    assert day.is_common_stock("PFBC") and day.is_common_stock("DHC") and day.is_common_stock("TVRD")
    assert "preferred" in day.exclusion("ACGLN") and "warrant" in day.exclusion("NXGLW")
    assert day.record("TVRD").name == "Tvardi Therapeutics, Inc. - Common Stock"


def test_the_days_list_classifies_a_stock_gone_since_by_its_type_and_name(today):
    from nasdaq_agent.universe import Universe
    day = Universe.for_day(today, listed_then(), DAY)
    assert "warrant" in day.exclusion("OLDW")
    assert "preferred" in day.exclusion("OLDP")
    assert day.exclusion("OLDF") is not None
    assert day.record("NVVE").name == "Nuvve Holding Corp. Common Stock"


class FetchLog:
    """A stand-in for Massive's NASDAQ list: answers from listed_then() and counts the days asked for."""
    def __init__(self, listing=None, error=None):
        self.days, self._listing, self._error = [], listing, error

    def __call__(self, day):
        self.days.append(day)
        if self._error is not None:
            raise self._error
        return listed_then() if self._listing is None else self._listing


def test_a_session_on_or_after_the_files_day_uses_the_file(today, tmp_path):
    from nasdaq_agent.universe import UniverseByDate
    fetch = FetchLog()
    by_date = UniverseByDate(today, fetch, tmp_path)
    assert by_date.for_session(FILE_DAY) is today and by_date.for_session(date(2026, 9, 29)) is today
    assert fetch.days == []


def test_an_earlier_session_uses_that_days_list_fetched_once(today, tmp_path):
    from nasdaq_agent.universe import UniverseByDate
    fetch = FetchLog()
    by_date = UniverseByDate(today, fetch, tmp_path)
    first = by_date.for_session(DAY)
    assert first.is_common_stock("NVVE") and not first.is_common_stock("IPOX")
    assert by_date.for_session(DAY) is first
    assert fetch.days == [DAY]


def test_a_days_list_is_saved_and_reused_by_later_runs(today, tmp_path):
    from nasdaq_agent.universe import UniverseByDate
    UniverseByDate(today, FetchLog(), tmp_path).for_session(DAY)
    fetch = FetchLog(error=AssertionError("a saved day must not be fetched again"))
    again = UniverseByDate(today, fetch, tmp_path).for_session(DAY)
    assert again.is_common_stock("NVVE") and "preferred" in again.exclusion("ACGLN")
    assert fetch.days == []


def test_an_unreadable_saved_list_is_fetched_again(today, tmp_path):
    from nasdaq_agent.universe import UniverseByDate
    UniverseByDate(today, FetchLog(), tmp_path).for_session(DAY)
    for saved in tmp_path.iterdir():
        saved.write_text("{not json")
    fetch = FetchLog()
    assert UniverseByDate(today, fetch, tmp_path).for_session(DAY).is_common_stock("NVVE")
    assert fetch.days == [DAY]


def test_record_and_replay_keep_no_saved_lists(today, tmp_path):
    from nasdaq_agent.universe import UniverseByDate
    UniverseByDate(today, FetchLog(), None).for_session(DAY)
    assert list(tmp_path.iterdir()) == []


def test_an_earlier_session_without_a_massive_key_is_refused(today, tmp_path):
    from nasdaq_agent.universe import UniverseByDate, UniverseError
    with pytest.raises(UniverseError, match="MASSIVE_API_KEY"):
        UniverseByDate(today, None, tmp_path).for_session(DAY)


def test_a_failed_fetch_is_reported_once_and_not_retried(today, tmp_path):
    from nasdaq_agent.sources.errors import SourceError
    from nasdaq_agent.universe import UniverseByDate, UniverseError
    fetch = FetchLog(error=SourceError("massive: HTTP 503"))
    by_date = UniverseByDate(today, fetch, tmp_path)
    for _ in range(2):
        with pytest.raises(UniverseError, match="2026-07-08"):
            by_date.for_session(DAY)
    assert fetch.days == [DAY] and list(tmp_path.iterdir()) == []


def test_an_empty_days_list_is_refused_and_never_saved(today, tmp_path):
    from nasdaq_agent.universe import UniverseByDate, UniverseError
    with pytest.raises(UniverseError):
        UniverseByDate(today, FetchLog(listing=[]), tmp_path).for_session(DAY)
    assert list(tmp_path.iterdir()) == []


def test_a_file_without_a_creation_day_is_used_for_every_session(tmp_path):
    from nasdaq_agent.universe import Universe, UniverseByDate
    undated = Universe.from_text(TODAY.replace("File Creation Time: 0928202609:46|||||||", ""))
    fetch = FetchLog()
    assert UniverseByDate(undated, fetch, tmp_path).for_session(DAY) is undated and fetch.days == []


# Massive's list of NASDAQ securities for a day, paged by ticker so every page carries the same filters.

LISTING_URL = "https://api.massive.com/v3/reference/tickers"


def _row(ticker, kind="CS", exchange="XNAS"):
    return {"ticker": ticker, "name": f"{ticker} Inc. Common Stock", "market": "stocks", "locale": "us",
            "primary_exchange": exchange, "type": kind, "active": True, "currency_name": "usd",
            "cik": "0000000001", "composite_figi": "BBG000000001", "share_class_figi": "BBG000000002",
            "last_updated_utc": "2026-07-08T00:00:00Z"}


def _listing_source(page_size=2, max_pages=5):
    from nasdaq_agent.sources.http import HttpClient
    from nasdaq_agent.sources.massive import MassiveListingSource
    return MassiveListingSource(HttpClient(connect_timeout=1, read_timeout=1), api_key="k",
                                page_size=page_size, max_pages=max_pages)


@respx.mock
def test_the_listing_pages_by_ticker_with_the_same_filters_on_every_page():
    pages = {None: [_row("AAA"), _row("BBB")], "BBB": [_row("CCC"), _row("DDD", "WARRANT")], "DDD": [_row("EEE")]}
    seen = []

    def answer(request):
        params = dict(request.url.params)
        seen.append(params)
        return httpx.Response(200, json={"results": pages[params.get("ticker.gt")], "status": "OK"})

    respx.get(LISTING_URL).mock(side_effect=answer)
    listed = _listing_source().nasdaq_listing(DAY)
    assert [(s.symbol, s.kind) for s in listed] == [("AAA", "CS"), ("BBB", "CS"), ("CCC", "CS"), ("DDD", "WARRANT"),
                                                    ("EEE", "CS")]
    assert [p.get("ticker.gt") for p in seen] == [None, "BBB", "DDD"]
    for params in seen:
        assert (params["exchange"], params["date"], params["active"], params["sort"], params["order"], params["limit"]) \
            == ("XNAS", "2026-07-08", "true", "ticker", "asc", "2")


@respx.mock
def test_the_listing_drops_rows_from_other_exchanges_and_malformed_rows(caplog):
    respx.get(LISTING_URL).mock(return_value=httpx.Response(200, json={"results": [
        _row("AAA"), _row("OTCX", exchange="OTC Link"), {"ticker": "BAD"}], "status": "OK"}))
    assert [s.symbol for s in _listing_source(page_size=5).nasdaq_listing(DAY)] == ["AAA"]


@respx.mock
def test_a_listing_that_does_not_end_within_the_page_cap_is_refused():
    from nasdaq_agent.sources.errors import SourceError
    counter = iter(range(100))

    def endless(request):
        n = next(counter)
        return httpx.Response(200, json={"results": [_row(f"A{n:03d}0"), _row(f"A{n:03d}1")], "status": "OK"})

    respx.get(LISTING_URL).mock(side_effect=endless)
    with pytest.raises(SourceError, match="3 pages"):
        _listing_source(page_size=2, max_pages=3).nasdaq_listing(DAY)


@respx.mock
def test_an_empty_listing_is_refused():
    from nasdaq_agent.sources.errors import SourceError
    respx.get(LISTING_URL).mock(return_value=httpx.Response(200, json={"results": [], "status": "OK"}))
    with pytest.raises(SourceError):
        _listing_source().nasdaq_listing(DAY)


def test_the_saved_list_holds_only_the_listing_facts(today, tmp_path):
    """What is saved is Massive's answer, not the classification, so a later change to today's file or the rules
    applies to a saved day too."""
    from nasdaq_agent.universe import UniverseByDate
    UniverseByDate(today, FetchLog(), tmp_path).for_session(DAY)
    (saved,) = list(tmp_path.iterdir())
    data = json.loads(saved.read_text())
    assert data["listed_on"] == "2026-07-08"
    assert {"symbol": "NVVE", "name": "Nuvve Holding Corp. Common Stock", "kind": "CS"} in data["securities"]


def test_a_failed_fetch_never_quotes_an_unexpected_errors_text(today, tmp_path):
    """The message reaches the model and the failure email. A source error is ours and names no key; any other error's
    text may quote the whole request, key included, so only its type is kept."""
    from nasdaq_agent.universe import UniverseByDate, UniverseError
    leaky = RuntimeError("GET https://api.massive.com/v3/reference/tickers?apiKey=not-a-real-key")
    with pytest.raises(UniverseError) as caught:
        UniverseByDate(today, FetchLog(error=leaky), tmp_path).for_session(DAY)
    assert "RuntimeError" in str(caught.value) and "not-a-real-key" not in str(caught.value)
