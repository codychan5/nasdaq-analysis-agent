"""SEC EDGAR filings as a news source. A small company's biggest moves often come with a filing and little or no press
coverage, and its current reports, on Forms 8-K and 6-K, are the nearest thing EDGAR has to news.

Companies and filings are found through efts.sec.gov (company search and full-text search) and data.sec.gov (a
company's filing list). A headline carries a filing's details (form, items, document names, acceptance time) and, when
the filing attaches a press release as an Exhibit 99, that release's headline and opening words, read from www.sec.gov.
www.sec.gov refuses requests that do not declare a contact (HTTP 403 to a generic User-Agent), so a refused or
unreadable exhibit leaves the headline with the filing's details only. Its link points at the filing's index page.
"""
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from typing import Any

from .errors import CassetteMiss, SourceError
from .http import HttpClient
from .models import SUMMARY_MAX_CHARS, Headline, clean_summary

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_digits}/{accession}-index.htm"
# SEC's fair-access policy allows at most ten requests a second.
SEC_CALLS_PER_SECOND = 10
CIK_DIGITS = 10
ACCESSION_PATTERN = re.compile(r"\d{10}-\d{2}-\d{6}")

# The forms kept, and what each one is, for the headline's summary.
CURRENT_REPORT_FORMS = {
    "8-K": "a current report on a major company event",
    "8-K/A": "an amendment to a current report on a major company event",
    "6-K": "a foreign private issuer's report of information it has made public, such as a press release",
    "6-K/A": "an amendment to a foreign private issuer's report of information it has made public",
}
# SEC's names for the 8-K items (Form 8-K, General Instructions B).
EIGHT_K_ITEMS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement of a "
            "Registrant",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation or an Obligation under an "
            "Off-Balance Sheet Arrangement",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or Standard; Transfer of Listing",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements or a Related Audit Report or Completed Interim "
            "Review",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; "
            "Compensatory Arrangements of Certain Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
    "5.04": "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
    "5.05": "Amendment to Registrant's Code of Ethics, or Waiver of a Provision of the Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "6.01": "ABS Informational and Computational Material",
    "6.02": "Change of Servicer or Trustee",
    "6.03": "Change in Credit Enhancement or Other External Support",
    "6.04": "Failure to Make a Required Distribution",
    "6.05": "Securities Act Updating Disclosure",
    "6.06": "Static Pool",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}
# Item 9.01 only lists a filing's exhibits, so it says nothing about the event itself.
EXHIBITS_ONLY_ITEM = "9.01"
# Document types with no text a reader would call news: XBRL data files and images.
NON_NARRATIVE_DOCUMENT_TYPES = ("EX-101", "EX-104", "GRAPHIC", "ZIP", "XML")
# Said plainly, because a live run read "not its text" as "the filing disclosed no terms".
METADATA_ONLY_NOTE = "The filing's text was not read, so what it says, including any terms, is unknown here."
FILING_COLUMNS = ("accessionNumber", "filingDate", "acceptanceDateTime", "form", "items")

EDGAR_DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_digits}/{name}"
# The exhibit type a current report attaches its press release to: EX-99.1, sometimes EX-99.2 and on.
PRESS_RELEASE_EXHIBIT_TYPE = "EX-99"
# A document's name comes from a search result and goes into a URL, so only a plain EDGAR file name is accepted.
DOCUMENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\.(?:htm|html|txt)")
# How much of an exhibit reaches the model: its headline and opening lines, which is where a release says what happened.
EXHIBIT_OPENING_MAX_CHARS = 600
# A release headline can run past 300 characters; the headline's title keeps its first words.
TITLE_HEADLINE_MAX_CHARS = 150
# Said as plainly as METADATA_ONLY_NOTE: the opening was read and nothing after it.
EXHIBIT_READ_NOTE = "Only the opening of {exhibit} was read, so anything later in the filing is unknown here."
# EDGAR's header fields ahead of a document's HTML, and elements whose text a reader never sees.
EDGAR_HEADER_ELEMENTS = frozenset({"type", "sequence", "filename", "description"})
HIDDEN_ELEMENTS = frozenset({"head", "title", "script", "style"})
BLOCK_ELEMENTS = frozenset({"p", "div", "br", "li", "ul", "ol", "table", "tr", "td", "th", "center", "body",
                            "h1", "h2", "h3", "h4", "h5", "h6"})
# Lines above a release's headline that are labels, not news: the exhibit number and the release markers.
RELEASE_LABEL_LINE = re.compile(r"(exhibit\s+\d+(\.\d+)?|for immediate release|(press|news) release)[:.]?",
                                re.IGNORECASE)
ELLIPSIS = "…"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Filing:
    accession: str
    accepted: datetime
    filing_date: date
    form: str
    items: tuple[str, ...]


@dataclass(frozen=True)
class _Document:
    description: str
    file_type: str
    name: str


class _TextBlocks(HTMLParser):
    """The text a reader sees in an EDGAR document, as one string per block (paragraph, list item, table cell)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._current: list[str] = []
        self._hidden: str | None = None
        # EDGAR's header fields have no end tags: each one's value runs to the next tag.
        self._in_header_field = False

    def handle_starttag(self, tag, attrs):
        self._in_header_field = tag in EDGAR_HEADER_ELEMENTS
        if tag in HIDDEN_ELEMENTS and self._hidden is None:
            self._hidden = tag
        if tag in BLOCK_ELEMENTS or self._in_header_field:
            self._end_block()

    def handle_endtag(self, tag):
        if tag == self._hidden:
            self._hidden = None
        if tag in BLOCK_ELEMENTS:
            self._end_block()

    def handle_data(self, data):
        if self._hidden is None and not self._in_header_field:
            self._current.append(data)

    def close(self):
        super().close()
        self._end_block()

    def _end_block(self):
        text = " ".join("".join(self._current).split())
        self._current = []
        if text and not (not self.blocks and RELEASE_LABEL_LINE.fullmatch(text)):
            self.blocks.append(text)


def _text_blocks(html: str) -> list[str]:
    """An EDGAR document's visible text in blocks, without the labels above its headline."""
    parser = _TextBlocks()
    parser.feed(html)
    parser.close()
    return parser.blocks


def _shorten(text: str, limit: int) -> str:
    """The text, or its first words up to `limit` characters with an ellipsis when it is longer."""
    if len(text) <= limit:
        return text
    if limit <= len(ELLIPSIS):
        return ""
    return text[:limit - len(ELLIPSIS)].rsplit(" ", 1)[0].rstrip(" ,;:-") + ELLIPSIS


def _sec_ticker(symbol: str) -> str:
    """SEC writes a share class with a hyphen (BRK-B) where the NASDAQ symbol file uses a dot (BRK.B)."""
    return symbol.strip().upper().replace(".", "-")


def _search_hits(data: Any, what: str) -> list[dict]:
    hits = data.get("hits") if isinstance(data, dict) else None
    hits = hits.get("hits") if isinstance(hits, dict) else None
    if not isinstance(hits, list):
        raise SourceError(f"sec: unexpected {what} payload")
    return [hit for hit in hits if isinstance(hit, dict)]


def _hit_source(hit: dict) -> dict:
    source = hit.get("_source")
    return source if isinstance(source, dict) else {}


def _parse_filing(row: dict[str, Any]) -> _Filing:
    """One row of the filing list. Raises ValueError or TypeError on a malformed row."""
    accession = str(row["accessionNumber"])
    if not ACCESSION_PATTERN.fullmatch(accession):
        raise ValueError("malformed accession number")
    # EDGAR states acceptance times in UTC, with a "Z": a filing accepted after 17:30 New York time gets the next
    # business day's filing date, and these times fit that only when read as UTC.
    accepted = datetime.fromisoformat(str(row["acceptanceDateTime"]).replace("Z", "+00:00"))
    if accepted.tzinfo is None:
        raise ValueError("acceptance time without a time zone")
    items = tuple(code.strip() for code in str(row["items"] or "").split(",") if code.strip())
    return _Filing(accession=accession, accepted=accepted, filing_date=date.fromisoformat(str(row["filingDate"])),
                   form=str(row["form"]).strip(), items=items)


def _search_word(company: str) -> str | None:
    """The full-text query that finds a filing's documents: the first word of the company's name, which appears in
    nearly every document it files. The CIK and the dates already narrow the search to the filings themselves."""
    words = company.split()
    word = re.sub(r"[^A-Za-z0-9-]", "", words[0]) if words else ""
    return word or None


class SecFilingsNewsSource:
    name = "sec"

    def __init__(self, client: HttpClient):
        self._client = client

    def headlines(self, symbol: str, since: datetime, until: datetime) -> list[Headline]:
        """The company's current reports accepted from `since` to `until`, both included, newest first."""
        cik = self._cik(symbol)
        if cik is None:
            log.info("%s: no SEC company lists the ticker %s", self.name, symbol)
            return []
        company, filings = self._filings(cik, since, until)
        documents = self._documents(cik, company, filings)
        headlines = []
        for number, filing in enumerate(filings, start=1):
            filed = documents.get(filing.accession, [])
            headlines.append(self._headline(number, cik, company, filing, filed,
                                            self._exhibit_opening(cik, filing, filed)))
        return headlines

    def _cik(self, symbol: str) -> str | None:
        """The company's CIK, zero-padded, from EDGAR's company search. The search matches prefixes, so a hit counts
        only when one of its tickers is exactly the one asked for."""
        wanted = _sec_ticker(symbol)
        data = self._client.get_json(EDGAR_SEARCH_URL, params={"keysTyped": wanted})
        for hit in _search_hits(data, "company search"):
            tickers = {ticker.strip().upper() for ticker in str(_hit_source(hit).get("tickers") or "").split(",")}
            cik = str(hit.get("_id", ""))
            if wanted in tickers and cik.isdigit():
                return cik.zfill(CIK_DIGITS)
        return None

    def _filings(self, cik: str, since: datetime, until: datetime) -> tuple[str, list[_Filing]]:
        data = self._client.get_json(EDGAR_SUBMISSIONS_URL.format(cik=cik))
        try:
            company = str(data["name"]).strip()
            recent = data["filings"]["recent"]
            columns = {column: list(recent[column]) for column in FILING_COLUMNS}
            older_pages = list(data["filings"].get("files") or [])
        except (KeyError, TypeError, AttributeError) as e:
            raise SourceError(f"sec: unexpected filing list for CIK {cik}") from e
        if not company or len({len(values) for values in columns.values()}) > 1:
            raise SourceError(f"sec: unexpected filing list for CIK {cik}")
        parsed: list[_Filing] = []
        skipped = 0
        for values in zip(*columns.values()):
            try:
                parsed.append(_parse_filing(dict(zip(FILING_COLUMNS, values))))
            except (ValueError, TypeError):
                skipped += 1
        if skipped:
            log.warning("%s: skipped %d malformed filing row(s) for CIK %s", self.name, skipped, cik)
        # The list holds the recent filings and names further pages for older ones, which are not read. A window that
        # starts on or before the oldest recent filing's date could miss filings, so it fails instead of looking empty.
        if older_pages and parsed and since.date() <= min(f.filing_date for f in parsed):
            raise SourceError(f"sec: the window starts before the {len(parsed)} recent filings SEC lists for CIK {cik}; "
                              "older filing pages are not read")
        kept = [f for f in parsed if f.form in CURRENT_REPORT_FORMS and since <= f.accepted <= until]
        return company, sorted(kept, key=lambda f: f.accepted, reverse=True)

    def _documents(self, cik: str, company: str, filings: list[_Filing]) -> dict[str, list[_Document]]:
        """Each filing's documents other than its cover page, in filing order, from EDGAR's full-text search. They
        only enrich the headlines: a failed lookup keeps the filings without them."""
        word = _search_word(company)
        if not filings or word is None:
            return {}
        dates = [f.filing_date for f in filings]
        params = {"q": f'"{word}"', "ciks": cik, "dateRange": "custom",
                  "startdt": min(dates).isoformat(), "enddt": max(dates).isoformat()}
        try:
            hits = _search_hits(self._client.get_json(EDGAR_SEARCH_URL, params=params), "full-text search")
        except CassetteMiss:
            raise  # in replay a missing recording means the cassette no longer matches the code: fail loudly
        except SourceError as e:
            log.warning("%s: document names for CIK %s unavailable (%s); keeping the filings without them",
                        self.name, cik, type(e).__name__)
            return {}
        wanted = {f.accession for f in filings}
        found: dict[str, list[tuple[int, _Document]]] = {}
        for hit in hits:
            document = _hit_source(hit)
            accession, file_type = document.get("adsh"), str(document.get("file_type") or "").strip()
            if (accession not in wanted or not file_type or file_type in CURRENT_REPORT_FORMS
                    or file_type.startswith(NON_NARRATIVE_DOCUMENT_TYPES)):
                continue
            description = " ".join(str(document.get("file_description") or "").split()) or file_type
            sequence = document.get("sequence") if isinstance(document.get("sequence"), int) else 0
            # A hit's id is "<accession>:<file name>". The name is checked before any use in a URL.
            name = str(hit.get("_id") or "").partition(":")[2]
            found.setdefault(accession, []).append((sequence, _Document(description, file_type, name)))
        return {accession: [doc for _, doc in sorted(docs, key=lambda pair: pair[0])]
                for accession, docs in found.items()}

    def _exhibit_opening(self, cik: str, filing: _Filing, documents: list[_Document]) -> tuple[str, list[str]] | None:
        """The filing's first press-release exhibit, named as a reader would say it ("Exhibit 99.1"), and its text in
        blocks, headline first; None when it has none or it cannot be read. Like the document names, it only enriches
        the headline, so a failure keeps the filing's details alone."""
        exhibit = next((d for d in documents if d.file_type.startswith(PRESS_RELEASE_EXHIBIT_TYPE)
                        and DOCUMENT_NAME_PATTERN.fullmatch(d.name)), None)
        if exhibit is None:
            return None
        label = exhibit.file_type.replace("EX-", "Exhibit ", 1)
        url = EDGAR_DOCUMENT_URL.format(cik=int(cik), accession_digits=filing.accession.replace("-", ""),
                                        name=exhibit.name)
        try:
            blocks = _text_blocks(self._client.get_text(url))
        except CassetteMiss:
            raise  # in replay a missing recording means the cassette no longer matches the code: fail loudly
        except SourceError as e:
            log.warning("%s: %s of filing %s unavailable (%s); keeping the filing's details only",
                        self.name, label, filing.accession, type(e).__name__)
            return None
        if not blocks:
            log.warning("%s: %s of filing %s has no readable text; keeping the filing's details only",
                        self.name, label, filing.accession)
            return None
        return label, blocks

    def _headline(self, number: int, cik: str, company: str, filing: _Filing, documents: list[_Document],
                  opening: tuple[str, list[str]] | None) -> Headline:
        items = [EIGHT_K_ITEMS.get(code, f"Item {code}") for code in filing.items if code != EXHIBITS_ONLY_ITEM]
        title = f"{company} filed a Form {filing.form} with the SEC"
        if opening is not None:
            title += ": " + _shorten(opening[1][0], TITLE_HEADLINE_MAX_CHARS)
        elif items:
            title += ": " + "; ".join(items)
        elif documents:
            title += ": " + documents[0].description
        summary = [f"Form {filing.form} is {CURRENT_REPORT_FORMS[filing.form]}."]
        if items:
            summary.append("Items reported: " + "; ".join(items) + ".")
        documents_line = ("Documents filed: " + "; ".join(f"{doc.description} ({doc.file_type})"
                                                          for doc in documents) + ".") if documents else None
        if opening is None:
            summary += [line for line in (documents_line, METADATA_ONLY_NOTE) if line]
        else:
            # The note and the opening come before the document list and the opening gets the room the summary's
            # bound leaves, so when a filing lists many documents, the list is what the bound cuts.
            exhibit, blocks = opening
            summary.append(EXHIBIT_READ_NOTE.format(exhibit=exhibit))
            lead_in = f'{exhibit} opens: "'
            room = SUMMARY_MAX_CHARS - len(" ".join(summary)) - len(lead_in) - len('"') - 1
            summary.append(lead_in + _shorten(" ".join(blocks), min(EXHIBIT_OPENING_MAX_CHARS, room)) + '"')
            if documents_line:
                summary.append(documents_line)
        url = EDGAR_FILING_INDEX_URL.format(cik=int(cik), accession_digits=filing.accession.replace("-", ""),
                                            accession=filing.accession)
        # The filer publishes the filing; the email names the source beside it ("... via SEC EDGAR").
        return Headline(id=number, title=title, provider=company, published=filing.accepted, url=url,
                        summary=clean_summary(" ".join(summary)))
