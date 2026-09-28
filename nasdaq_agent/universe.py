import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SYMBOL_FILE_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
EXCLUDED_NAME_PATTERN = re.compile(
    r"\b(warrants?|rights?|units?|preferred|pfd|notes?|debentures?)\b", re.IGNORECASE
)
SECONDS_PER_HOUR = 3600
EXCLUSION_REASONS = ("etf", "test issue", "warrant", "right", "unit", "preferred", "note", "debenture")


class UniverseError(ValueError):
    """Raised when universe data is invalid or fetch fails."""
    pass


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


class Universe:
    def __init__(self, records: dict[str, SymbolRecord]):
        self._records = records

    @classmethod
    def from_text(cls, text: str) -> "Universe":
        """Parse pipe-delimited NASDAQ symbol file. Raises UniverseError if empty or missing columns."""
        lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
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
        return cls(records)

    def record(self, symbol: str) -> SymbolRecord | None:
        return self._records.get(symbol)

    def is_common_stock(self, symbol: str) -> bool:
        rec = self._records.get(symbol)
        return rec is not None and rec.exclusion_reason is None

    @staticmethod
    def to_provider(symbol: str, provider: str) -> str:
        return symbol.replace(".", "-") if provider == "yahoo" else symbol

    @staticmethod
    def to_canonical(symbol: str, provider: str) -> str:
        return symbol.replace("-", ".") if provider == "yahoo" else symbol

    def __len__(self) -> int:
        return len(self._records)


def load_universe(fetch_text: Callable[[str], str], cache_path: Path | None, max_age_hours: int = 24) -> Universe:
    """Use the cached symbol file when fresh; otherwise fetch and cache it. Raises UniverseError on fetch failure or zero records.

    cache_path=None fetches every time and neither reads nor writes a local cache: record and replay take the symbol
    file through the cassette, so a clean-machine replay never depends on, or leaves behind, a local copy (Task 22).
    """
    if cache_path is not None and cache_path.exists() and time.time() - cache_path.stat().st_mtime < max_age_hours * SECONDS_PER_HOUR:
        return Universe.from_text(cache_path.read_text())
    try:
        text = fetch_text(SYMBOL_FILE_URL)
    except Exception as e:
        raise UniverseError(f"Failed to fetch symbol file from {SYMBOL_FILE_URL}: {e}") from e
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
    universe = Universe.from_text(text)
    if len(universe) == 0:
        raise UniverseError("Fetched symbol file contains zero records")
    return universe
