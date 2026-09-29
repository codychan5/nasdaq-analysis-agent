import logging
from datetime import date

from ..universe import SessionUniverse
from .errors import SourceError
from .http import HttpClient
from .models import Candidate, rank_candidates

NASDAQCOM_URL = "https://api.nasdaq.com/api/screener/stocks"
NASDAQCOM_PARAMS = {"tableonly": "true", "limit": "25", "offset": "0", "exchange": "nasdaq", "download": "true"}
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

log = logging.getLogger(__name__)


def _num(text: str) -> float:
    return float(str(text).replace("$", "").replace(",", "").replace("%", "").strip() or "0")


class NasdaqComGainerSource:
    name = "nasdaqcom"
    requires_market_closed = True

    def __init__(self, client: HttpClient, universe: SessionUniverse):
        self._client, self._universe = client, universe

    def top_candidates(self, session_date: date, prev_session_date: date, limit: int) -> list[Candidate]:
        data = self._client.get_json(NASDAQCOM_URL, params=NASDAQCOM_PARAMS, headers=BROWSER_HEADERS)
        rows = ((data or {}).get("data") or {}).get("rows") or []
        if not rows:
            raise SourceError("nasdaqcom: screener returned no rows")
        listing = self._universe.for_session(session_date)
        out: list[Candidate] = []
        skipped = 0
        for r in rows:
            try:
                symbol = r["symbol"].strip()
                last, net = _num(r["lastsale"]), _num(r["netchange"])
                out.append(Candidate(symbol=symbol, name=r.get("name"), prev_close=last - net, close=last,
                                     pct_change=_num(r["pctchange"]), volume=_num(r.get("volume", "0")),
                                     market_cap=_num(r.get("marketCap", "0")) or None, source=self.name,
                                     excluded=listing.exclusion(symbol)))
            except (KeyError, ValueError, TypeError):
                skipped += 1
        if skipped:
            log.warning("%s: skipped %d malformed gainer row(s)", self.name, skipped)
        return rank_candidates(out, limit)
