# tests/unit/test_sec_news_source.py
"""The SEC EDGAR news source, against payloads shaped like the live responses of efts.sec.gov (company search and
full-text search) and data.sec.gov (a company's filing list), recorded on 2026-09-28 for WETO."""
import logging
from datetime import datetime, timezone

import httpx
import pytest
import respx

SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
WETO_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0001941158.json"
SINCE = datetime(2026, 8, 11, tzinfo=timezone.utc)
UNTIL = datetime(2026, 8, 14, 20, 30, tzinfo=timezone.utc)  # 16:30 New York on the session day, 14 August
FILING_COLUMNS = ("accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "act", "form", "fileNumber",
                  "filmNumber", "items", "core_type", "size", "isXBRL", "isInlineXBRL", "isXBRLNumeric",
                  "primaryDocument", "primaryDocDescription")


def _search_payload(hits):
    return {"took": 5, "timed_out": False, "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
            "hits": {"total": {"value": len(hits), "relation": "eq"}, "max_score": 10.0, "hits": hits},
            "query": {"query": {"bool": {"should": []}}}}


def _entity(cik, name, tickers):
    label = f"{name} ({tickers})" if tickers else name
    return {"_index": "edgar_entity_20260918_040210", "_id": cik, "_score": 10.0,
            "_source": {"entity": label, "entity_words": label, "tickers": tickers, "rank": 20839360}}


def _filing(accession, accepted, form="6-K", items=""):
    return {"accessionNumber": accession, "filingDate": accepted[:10], "reportDate": accepted[:10],
            "acceptanceDateTime": accepted, "act": "34", "form": form, "fileNumber": "001-42536",
            "filmNumber": "261267995", "items": items, "core_type": form, "size": 21456, "isXBRL": 0,
            "isInlineXBRL": 0, "isXBRLNumeric": 0, "primaryDocument": "ea0301733-6k_wetour.htm",
            "primaryDocDescription": "REPORT OF FOREIGN PRIVATE ISSUER"}


def _submissions(filings, files=()):
    return {"cik": "1941158", "entityType": "operating", "sic": "4100", "sicDescription": "Transportation Services",
            "ownerOrg": "", "insiderTransactionForOwnerExists": 0, "insiderTransactionForIssuerExists": 0,
            "name": "Wetour Robotics Ltd", "tickers": ["WETO"], "exchanges": ["Nasdaq"], "ein": "000000000",
            "lei": None, "description": "", "website": "", "investorWebsite": "",
            "category": "Non-accelerated filer<br>Emerging growth company", "fiscalYearEnd": "1231",
            "stateOfIncorporation": "E9", "stateOfIncorporationDescription": "Cayman Islands",
            "addresses": {"mailing": {"city": "Austin", "stateOrCountry": "TX"},
                          "business": {"city": "Austin", "stateOrCountry": "TX"}},
            "phone": "000-000-0000", "flags": "", "formerNames": [{"name": "Webus International Ltd"}],
            "filings": {"recent": {column: [f[column] for f in filings] for column in FILING_COLUMNS},
                        "files": list(files)}}


def _document(accession, filename, file_type, description, sequence, form="6-K"):
    return {"_index": "edgar_file", "_id": f"{accession}:{filename}", "_score": 25.0,
            "_source": {"ciks": ["0001941158"], "period_ending": "2026-08-12", "file_num": ["001-42536"],
                        "display_names": ["Wetour Robotics Ltd  (WETO)  (CIK 0001941158)"], "xsl": None,
                        "sequence": sequence, "root_forms": [form], "file_date": "2026-08-12", "biz_states": ["TX"],
                        "sics": ["4100"], "form": form, "adsh": accession, "film_num": ["261267995"],
                        "biz_locations": ["Austin, TX"], "file_type": file_type, "file_description": description,
                        "inc_states": ["E9"], "items": []}}


# The real 6-K from 12 August and its documents: the cover page, a securities purchase agreement, and XBRL data.
PLACEMENT = _filing("0001213900-26-088400", "2026-08-12T20:30:33.000Z")
PLACEMENT_DOCUMENTS = _search_payload([
    _document("0001213900-26-088400", "ea0301733-6k_wetour.htm", "6-K", "REPORT OF FOREIGN PRIVATE ISSUER", 1),
    _document("0001213900-26-088400", "ea030173301ex10-1.htm", "EX-10.1", "FORM OF SECURITIES PURCHASE AGREEMENT", 2),
    _document("0001213900-26-088400", "weto-20260812.xsd", "EX-101.SCH", "XBRL TAXONOMY EXTENSION SCHEMA", 3)])


def _source():
    from nasdaq_agent.sources.http import HttpClient
    from nasdaq_agent.sources.sec import SecFilingsNewsSource
    return SecFilingsNewsSource(HttpClient(1, 1))


def _mock_weto(filings, documents=None, files=()):
    respx.get(SEARCH_URL, params={"keysTyped": "WETO"}).mock(
        return_value=httpx.Response(200, json=_search_payload([_entity("1941158", "Wetour Robotics Ltd", "WETO")])))
    respx.get(WETO_SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=_submissions(filings, files)))
    return respx.get(SEARCH_URL, params={"ciks": "0001941158"}).mock(
        return_value=httpx.Response(200, json=documents or _search_payload([])))


@respx.mock
def test_keeps_current_reports_accepted_inside_the_window_newest_first():
    # Acceptance times are UTC. A 6-K accepted at 16:30 New York on the session day is inside a window that ends then;
    # one accepted a second later is not. A prospectus supplement (424B5) is not a current report.
    # Listed out of order on purpose: the source sorts, rather than trusting the list's order.
    filings = [PLACEMENT,
               _filing("0001213900-26-089003", "2026-08-14T20:30:01.000Z"),
               _filing("0001213900-26-087400", "2026-08-11T00:00:00.000Z"),
               _filing("0001213900-26-089002", "2026-08-14T20:30:00.000Z"),
               _filing("0001213900-26-089001", "2026-08-13T12:00:00.000Z", form="424B5"),
               _filing("0001213900-26-087326", "2026-08-10T21:00:12.000Z")]
    _mock_weto(filings)
    hs = _source().headlines("WETO", since=SINCE, until=UNTIL)
    assert [h.published for h in hs] == [datetime(2026, 8, 14, 20, 30, tzinfo=timezone.utc),
                                         datetime(2026, 8, 12, 20, 30, 33, tzinfo=timezone.utc),
                                         datetime(2026, 8, 11, 0, 0, tzinfo=timezone.utc)]
    assert [h.id for h in hs] == [1, 2, 3]


@respx.mock
def test_a_6k_headline_names_the_filer_and_its_documents_and_links_the_filing():
    documents_route = _mock_weto([PLACEMENT], documents=PLACEMENT_DOCUMENTS)
    [h] = _source().headlines("WETO", since=SINCE, until=UNTIL)
    assert h.title == "Wetour Robotics Ltd filed a Form 6-K with the SEC: FORM OF SECURITIES PURCHASE AGREEMENT"
    assert h.provider == "Wetour Robotics Ltd"  # the filer publishes it; the email adds "via SEC EDGAR"
    assert h.url == "https://www.sec.gov/Archives/edgar/data/1941158/000121390026088400/0001213900-26-088400-index.htm"
    assert "Documents filed: FORM OF SECURITIES PURCHASE AGREEMENT (EX-10.1)." in h.summary
    assert "XBRL" not in h.summary
    # A live run read "not its text" as "the filing disclosed no terms"; the note must say its contents are unknown.
    assert "The filing's text was not read, so what it says, including any terms, is unknown here." in h.summary
    params = documents_route.calls.last.request.url.params
    assert (params["q"], params["startdt"], params["enddt"]) == ('"Wetour"', "2026-08-12", "2026-08-12")


@respx.mock
def test_an_8k_title_names_its_items_except_the_exhibits_item():
    # Item 9.01 only lists a filing's exhibits, so it adds nothing to a title. An item code SEC may add later shows as is.
    _mock_weto([_filing("0001213900-26-088500", "2026-08-13T20:05:00.000Z", form="8-K", items="1.01,3.02,4.99,9.01")])
    [h] = _source().headlines("WETO", since=SINCE, until=UNTIL)
    assert h.title == ("Wetour Robotics Ltd filed a Form 8-K with the SEC: Entry into a Material Definitive Agreement; "
                       "Unregistered Sales of Equity Securities; Item 4.99")
    assert "Items reported: Entry into a Material Definitive Agreement;" in h.summary


@respx.mock
def test_the_company_is_found_by_an_exact_ticker_match():
    # The company search matches prefixes: WETH comes back for WETO, and here it is listed first.
    respx.get(SEARCH_URL, params={"keysTyped": "WETO"}).mock(return_value=httpx.Response(200, json=_search_payload([
        _entity("1826660", "Wetouch Technology Inc.", "WETH"), _entity("1941158", "Wetour Robotics Ltd", "WETO")])))
    respx.get(WETO_SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=_submissions([PLACEMENT])))
    respx.get(SEARCH_URL, params={"ciks": "0001941158"}).mock(return_value=httpx.Response(200, json=_search_payload([])))
    [h] = _source().headlines("WETO", since=SINCE, until=UNTIL)
    assert h.provider == "Wetour Robotics Ltd"


@respx.mock
def test_a_company_with_several_tickers_is_found_by_any_of_them():
    respx.get(SEARCH_URL, params={"keysTyped": "GOOGL"}).mock(return_value=httpx.Response(200, json=_search_payload([
        _entity("1652044", "Alphabet Inc.", "GOOG, GOOGN, GOOGM, GOOGL"), _entity("1288776", "GOOGLE INC.", None)])))
    route = respx.get("https://data.sec.gov/submissions/CIK0001652044.json").mock(
        return_value=httpx.Response(200, json=_submissions([])))
    assert _source().headlines("GOOGL", since=SINCE, until=UNTIL) == [] and route.called


@respx.mock
def test_a_ticker_no_company_lists_gives_no_headlines_and_reads_no_filings():
    respx.get(SEARCH_URL, params={"keysTyped": "WETO"}).mock(
        return_value=httpx.Response(200, json=_search_payload([_entity("1826660", "Wetouch Technology Inc.", "WETH")])))
    filings_route = respx.get(url__startswith="https://data.sec.gov/submissions/")
    assert _source().headlines("WETO", since=SINCE, until=UNTIL) == []
    assert not filings_route.called


@respx.mock
def test_filings_are_kept_when_their_document_names_cannot_be_fetched(caplog):
    # The document names only enrich a headline; the filing list is the source of truth.
    documents_route = _mock_weto([PLACEMENT])
    documents_route.mock(return_value=httpx.Response(404))
    with caplog.at_level(logging.WARNING):
        [h] = _source().headlines("WETO", since=SINCE, until=UNTIL)
    assert h.title == "Wetour Robotics Ltd filed a Form 6-K with the SEC"
    assert "Documents filed" not in h.summary and "document names" in caplog.text


@respx.mock
def test_a_malformed_filing_row_is_skipped_and_the_rest_kept(caplog):
    bad_time = _filing("0001213900-26-088600", "2026-08-13T12:00:00.000Z") | {"acceptanceDateTime": "not a time"}
    bad_accession = _filing("0001213900-26-088601", "2026-08-13T13:00:00.000Z") | {"accessionNumber": "../../etc"}
    _mock_weto([bad_accession, bad_time, PLACEMENT])
    with caplog.at_level(logging.WARNING):
        hs = _source().headlines("WETO", since=SINCE, until=UNTIL)
    assert [h.url.rsplit("/", 1)[-1] for h in hs] == ["0001213900-26-088400-index.htm"]
    assert "skipped 2 malformed filing row(s)" in caplog.text


@pytest.mark.parametrize("payload", [
    {"name": "Wetour Robotics Ltd"},  # no filings list at all
    _submissions([PLACEMENT]) | {"filings": {"recent": {**_submissions([PLACEMENT])["filings"]["recent"],
                                                        "form": ["6-K", "6-K"]}, "files": []}},  # columns disagree
])
@respx.mock
def test_an_unreadable_filing_list_is_a_source_error(payload):
    from nasdaq_agent.sources.errors import SourceError
    respx.get(SEARCH_URL, params={"keysTyped": "WETO"}).mock(
        return_value=httpx.Response(200, json=_search_payload([_entity("1941158", "Wetour Robotics Ltd", "WETO")])))
    respx.get(WETO_SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=payload))
    with pytest.raises(SourceError):
        _source().headlines("WETO", since=SINCE, until=UNTIL)


OLDER_PAGE = {"name": "CIK0001941158-submissions-001.json", "filingCount": 1000, "filingFrom": "2019-01-02",
              "filingTo": "2026-08-01"}


@pytest.mark.parametrize("oldest_recent", [
    PLACEMENT,                                                    # the oldest recent filing is after the window starts
    _filing("0001213900-26-087400", "2026-08-11T09:00:00.000Z"),  # the same day: that day's earlier filings may be older
])
@respx.mock
def test_a_window_reaching_past_the_recent_filing_list_is_a_source_error(oldest_recent):
    # SEC lists a company's recent filings in one page and older ones in further pages, which are not read. A window
    # starting on or before the oldest recent filing's date could miss filings, so it fails rather than look empty.
    from nasdaq_agent.sources.errors import SourceError
    _mock_weto([PLACEMENT, oldest_recent] if oldest_recent is not PLACEMENT else [PLACEMENT], files=[OLDER_PAGE])
    with pytest.raises(SourceError):
        _source().headlines("WETO", since=SINCE, until=UNTIL)


@respx.mock
def test_older_filing_pages_do_not_matter_when_the_window_is_inside_the_recent_list():
    _mock_weto([PLACEMENT, _filing("0001213900-26-087326", "2026-08-10T21:00:12.000Z")], files=[OLDER_PAGE])
    assert len(_source().headlines("WETO", since=SINCE, until=UNTIL)) == 1


def test_a_replay_missing_the_document_request_fails_instead_of_dropping_the_documents():
    # A failed document lookup is survivable live, but in replay a missing recording means the cassette no longer
    # matches the code, which must fail loudly like every other replay miss.
    import json
    from nasdaq_agent.sources.errors import CassetteMiss
    from nasdaq_agent.sources.http import HttpClient, request_key
    from nasdaq_agent.sources.sec import SecFilingsNewsSource

    class ReplayCassette:
        mode = "replay"
        def __init__(self, entries): self._entries = entries
        def lookup(self, key): return self._entries.get(key)
        def store(self, key, status, text): raise AssertionError("replay never records")
        def store_failure(self, key, error): raise AssertionError("replay never records")

    company_search = _search_payload([_entity("1941158", "Wetour Robotics Ltd", "WETO")])
    cassette = ReplayCassette({request_key("GET", SEARCH_URL, {"keysTyped": "WETO"}): (200, json.dumps(company_search)),
                               request_key("GET", WETO_SUBMISSIONS_URL, None): (200, json.dumps(_submissions([PLACEMENT])))})
    with pytest.raises(CassetteMiss):
        SecFilingsNewsSource(HttpClient(1, 1, cassette=cassette)).headlines("WETO", since=SINCE, until=UNTIL)


@respx.mock
def test_the_registry_adds_the_sec_source_only_with_a_user_agent_and_declares_it(monkeypatch):
    from pydantic import SecretStr
    from nasdaq_agent.config import Settings
    from nasdaq_agent.sources.registry import build_news_sources, massive_rate_limiter
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_SEC_USER_AGENT", raising=False)
    settings = Settings(_env_file=None)
    limiter = massive_rate_limiter()
    assert [s.name for s in build_news_sources(settings, massive_limiter=limiter)] == ["yfinance"]
    sources = build_news_sources(settings.model_copy(update={"sec_user_agent": SecretStr("Jane Doe jane@example.com")}),
                                 massive_limiter=limiter)
    assert [s.name for s in sources] == ["yfinance", "sec"]
    route = respx.get(SEARCH_URL, params={"keysTyped": "WETO"}).mock(return_value=httpx.Response(200, json=_search_payload([])))
    sources[-1].headlines("WETO", since=SINCE, until=UNTIL)
    assert route.calls.last.request.headers["User-Agent"] == "Jane Doe jane@example.com"
