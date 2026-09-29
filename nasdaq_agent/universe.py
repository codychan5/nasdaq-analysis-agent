import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Protocol

from .sources.errors import SourceError

log = logging.getLogger(__name__)

SYMBOL_FILE_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
EXCLUDED_NAME_PATTERN = re.compile(
    r"\b(warrants?|rights?|units?|preferred|pfd|notes?|debentures?)\b", re.IGNORECASE
)
SECONDS_PER_HOUR = 3600
EXCLUSION_REASONS = ("etf", "test issue", "warrant", "right", "unit", "preferred", "note", "debenture")
# The symbol file's last line says when NASDAQ made it, as "File Creation Time: 0928202609:46" (month, day, year, then
# the time). The day is the one whose listings the file holds.
FILE_CREATION_PREFIX = "File Creation Time:"
FILE_CREATION_FORMAT = "%m%d%Y%H:%M"
SOURCE_NASDAQ_FILE = "nasdaqtrader"
SOURCE_MASSIVE = "massive"
# Massive's security types for the two kinds that count, common stock and ADR common. Massive labels more coarsely than
# the file's security names (a depositary preferred can be "CS", a trust's common shares "FUND"), so a type decides only
# for a security today's file no longer lists.
COMMON_STOCK_KINDS = frozenset({"CS", "ADRC"})
MASSIVE_KIND_REASONS = {"ETF": "etf", "WARRANT": "warrant", "RIGHT": "right", "UNIT": "unit", "PFD": "preferred"}
DAY_LIST_FILENAME = "nasdaq-listed-{day}.json"


class UniverseError(ValueError):
    """Raised when universe data is invalid or fetch fails."""
    pass


@dataclass(frozen=True)
class ListedSecurity:
    """One security listed on NASDAQ on a given day, as Massive's reference data has it: the ticker, the name and
    Massive's security type (CS, ADRC, ETF, WARRANT, RIGHT, UNIT, PFD, FUND and so on)."""
    symbol: str
    name: str
    kind: str


@dataclass(frozen=True)
class SymbolRecord:
    symbol: str
    name: str
    market_category: str
    is_test: bool
    is_etf: bool
    exclusion_reason: str | None


def _classify(name: str, is_test: bool, is_etf: bool) -> str | None:
    """Classify security as excluded or None. Checks ETF and test issue first, then security type suffix."""
    if is_test:
        return "test issue"
    if is_etf:
        return "etf"
    # Split on the LAST " - " to isolate security type suffix; if no separator, check whole name
    parts = name.rsplit(" - ", 1)
    security_type = parts[-1] if len(parts) > 1 else name
    match = EXCLUDED_NAME_PATTERN.search(security_type)
    if match:
        word = match.group(1).lower().rstrip("s")
        result = {"pfd": "preferred", "note": "note", "debenture": "debenture"}.get(word, word)
        assert result in EXCLUSION_REASONS, f"_classify returned '{result}' not in EXCLUSION_REASONS"
        return result
    return None


def _creation_day(lines: list[str]) -> date | None:
    """The day the symbol file lists, from its "File Creation Time" line; None when the line is missing or unreadable."""
    for line in lines:
        if line.startswith(FILE_CREATION_PREFIX):
            stamp = line[len(FILE_CREATION_PREFIX):].split("|")[0].strip()
            try:
                return datetime.strptime(stamp, FILE_CREATION_FORMAT).date()
            except ValueError:
                log.warning("the symbol file's creation time %r is unreadable", stamp)
                return None
    return None


def _classify_listed(security: ListedSecurity) -> str | None:
    """The exclusion reason for a security today's file no longer lists: Massive's type first, then the same words in
    its name that exclude a security in the file (a depositary preferred Massive calls common stock, for instance)."""
    if security.kind not in COMMON_STOCK_KINDS:
        return MASSIVE_KIND_REASONS.get(security.kind, f"{security.kind.lower()} (Massive's security type)")
    return _classify(security.name, is_test=False, is_etf=False)


class Universe:
    """Which securities count for a session. `listed_on` is the day the list describes (for the symbol file, the day
    NASDAQ made it; None when the file does not say) and `source` where it came from: NASDAQ's file, or Massive's
    records for a past day."""

    def __init__(self, records: dict[str, SymbolRecord], listed_on: date | None = None, source: str = SOURCE_NASDAQ_FILE):
        self._records = records
        self.listed_on = listed_on
        self.source = source

    @classmethod
    def from_text(cls, text: str) -> "Universe":
        """Parse pipe-delimited NASDAQ symbol file. Raises UniverseError if empty or missing columns."""
        created_on = _creation_day(text.splitlines())
        lines = [ln for ln in text.splitlines() if ln and not ln.startswith(FILE_CREATION_PREFIX)]
        if not lines:
            raise UniverseError("Empty or malformed universe file")
        header = lines[0].split("|")
        idx = {name: i for i, name in enumerate(header)}
        required_columns = ["Symbol", "Security Name", "Market Category", "Test Issue", "ETF"]
        for col in required_columns:
            if col not in idx:
                raise UniverseError(f"Missing required column: {col}")
        records: dict[str, SymbolRecord] = {}
        for ln in lines[1:]:
            cols = ln.split("|")
            if len(cols) < len(header):
                continue
            symbol = cols[idx["Symbol"]].strip()
            name = cols[idx["Security Name"]].strip()
            is_test = cols[idx["Test Issue"]].strip() == "Y"
            is_etf = cols[idx["ETF"]].strip() == "Y"
            records[symbol] = SymbolRecord(
                symbol=symbol, name=name, market_category=cols[idx["Market Category"]].strip(),
                is_test=is_test, is_etf=is_etf, exclusion_reason=_classify(name, is_test, is_etf),
            )
        return cls(records, listed_on=created_on)

    @classmethod
    def for_day(cls, official: "Universe", listed: list[ListedSecurity], day: date) -> "Universe":
        """The securities listed on NASDAQ on `day`, from Massive's list for that day. A security today's file still
        lists keeps the file's record, name and classification; one gone since is classified by Massive's type and its
        name. A security in today's file but not in the day's list was not listed that day, so it does not count."""
        records: dict[str, SymbolRecord] = {}
        for security in listed:
            still_listed = official.record(security.symbol)
            records[security.symbol] = still_listed if still_listed is not None else SymbolRecord(
                symbol=security.symbol, name=security.name, market_category="", is_test=False,
                is_etf=security.kind == "ETF", exclusion_reason=_classify_listed(security))
        return cls(records, listed_on=day, source=SOURCE_MASSIVE)

    def for_session(self, session_date: date) -> "Universe":
        """A fixed list answers the same for every session; UniverseByDate chooses between lists by session."""
        return self

    def record(self, symbol: str) -> SymbolRecord | None:
        return self._records.get(symbol)

    def is_common_stock(self, symbol: str) -> bool:
        rec = self._records.get(symbol)
        return rec is not None and rec.exclusion_reason is None

    def exclusion(self, symbol: str) -> str | None:
        """Why the symbol does not count, in plain words; None when it counts."""
        rec = self._records.get(symbol)
        if rec is None:
            return (f"not listed on NASDAQ on {self.listed_on}" if self.source == SOURCE_MASSIVE
                    else "not on NASDAQ's list of listed securities")
        return f"excluded as {rec.exclusion_reason}" if rec.exclusion_reason is not None else None

    def common_stock_count(self) -> int:
        return sum(1 for rec in self._records.values() if rec.exclusion_reason is None)

    @staticmethod
    def to_provider(symbol: str, provider: str) -> str:
        return symbol.replace(".", "-") if provider == "yahoo" else symbol

    @staticmethod
    def to_canonical(symbol: str, provider: str) -> str:
        return symbol.replace("-", ".") if provider == "yahoo" else symbol

    def __len__(self) -> int:
        return len(self._records)


def _usable_universe(text: str, origin: str) -> Universe:
    """The universe in a symbol file, or UniverseError when the file cannot be parsed or holds no records."""
    universe = Universe.from_text(text)
    if len(universe) == 0:
        raise UniverseError(f"{origin} symbol file contains zero records")
    return universe


def load_universe(fetch_text: Callable[[str], str], cache_path: Path | None, max_age_hours: int = 24) -> Universe:
    """Use the cached symbol file when fresh; otherwise fetch and cache it. Raises UniverseError on fetch failure or zero records.

    Only a file that parses into at least one record is cached, and a fresh cached file is checked the same way and
    fetched again when it fails. Otherwise one bad response, such as a captive-portal page, would block every run until
    the cache aged out.

    cache_path=None fetches every time and neither reads nor writes a local cache: record and replay take the symbol
    file through the cassette, so a clean-machine replay never depends on, or leaves behind, a local copy.
    """
    if cache_path is not None and cache_path.exists() and time.time() - cache_path.stat().st_mtime < max_age_hours * SECONDS_PER_HOUR:
        try:
            return _usable_universe(cache_path.read_text(), "Cached")
        except UniverseError as e:
            log.warning("cached symbol file %s is unusable (%s); fetching it again", cache_path, e)
    try:
        text = fetch_text(SYMBOL_FILE_URL)
    except Exception as e:
        raise UniverseError(f"Failed to fetch symbol file from {SYMBOL_FILE_URL}: {e}") from e
    universe = _usable_universe(text, "Fetched")  # raises before anything is cached
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
    return universe


class SessionUniverse(Protocol):
    """Anything that says which securities count for a session: a Universe, which answers the same for every session,
    or a UniverseByDate."""

    def for_session(self, session_date: date) -> Universe: ...


class UniverseByDate:
    """NASDAQ's symbol file lists only the securities listed on the day it was made, so a stock delisted since is
    missing from it. A session on or after that day uses the file. An earlier session uses the securities Massive's
    reference data lists on NASDAQ that day (Universe.for_day), fetched once per run through `fetch_day` and saved as
    one file per day under `cache_dir` (a past day's list never changes). cache_dir=None keeps no files: record and
    replay take every response through the cassette. A day that cannot be listed raises UniverseError, and the same
    error again for the rest of the run, so a failure is never retried call after call."""

    def __init__(self, official: Universe, fetch_day: Callable[[date], list[ListedSecurity]] | None,
                 cache_dir: Path | None):
        self.official = official
        self._fetch_day, self._cache_dir = fetch_day, cache_dir
        self._days: dict[date, Universe | UniverseError] = {}
        if official.listed_on is None:
            log.warning("the symbol file does not say which day it lists; it is used for every session")

    def for_session(self, session_date: date) -> Universe:
        if self.official.listed_on is None or session_date >= self.official.listed_on:
            return self.official
        known = self._days.get(session_date)
        if isinstance(known, UniverseError):
            raise known
        if known is not None:
            return known
        try:
            listed = self._saved(session_date)
            if listed is None:
                listed = self._fetch(session_date)
                self._save(session_date, listed)
        except UniverseError as e:
            self._days[session_date] = e
            raise
        day = Universe.for_day(self.official, listed, session_date)
        self._days[session_date] = day
        return day

    def _fetch(self, day: date) -> list[ListedSecurity]:
        if self._fetch_day is None:
            raise UniverseError(f"a run for {day} needs MASSIVE_API_KEY: NASDAQ's symbol file lists only the securities "
                                f"listed on {self.official.listed_on}, and Massive's records say which were listed on {day}")
        try:
            listed = self._fetch_day(day)
        except SourceError as e:
            # Our own source errors name the URL without its query, so the key that travels in it is never in them.
            raise UniverseError(f"could not get the list of NASDAQ securities for {day}: {e}") from e
        except Exception as e:
            # Any other error's text may quote the full request, key included, and this message reaches the model and
            # the failure email: the type alone.
            raise UniverseError(f"could not get the list of NASDAQ securities for {day}: {type(e).__name__}") from e
        if not listed:
            raise UniverseError(f"Massive listed no NASDAQ securities for {day}")
        return listed

    def _path(self, day: date) -> Path | None:
        return self._cache_dir / DAY_LIST_FILENAME.format(day=day.isoformat()) if self._cache_dir is not None else None

    def _saved(self, day: date) -> list[ListedSecurity] | None:
        """A saved list for the day, or None when there is none or it cannot be used (then it is fetched again)."""
        path = self._path(day)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if data["listed_on"] != day.isoformat():
                raise ValueError(f"it lists {data['listed_on']}")
            listed = [ListedSecurity(symbol=str(s["symbol"]), name=str(s["name"]), kind=str(s["kind"]))
                      for s in data["securities"]]
            if not listed:
                raise ValueError("it lists no securities")
            return listed
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("saved NASDAQ list %s is unusable (%s); fetching it again", path, type(e).__name__)
            return None

    def _save(self, day: date, listed: list[ListedSecurity]) -> None:
        """Written to a temporary file and renamed into place, so a crash never leaves half a list to be read later."""
        path = self._path(day)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"listed_on": day.isoformat(), "source": SOURCE_MASSIVE,
                              "securities": [asdict(s) for s in listed]})
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as f:
                f.write(payload)
            os.replace(temporary, path)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            raise
