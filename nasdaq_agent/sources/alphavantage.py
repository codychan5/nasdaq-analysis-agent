import logging
from datetime import date

from ..universe import Universe
from .errors import SourceError
from .http import DailyQuota, HttpClient
from .models import Candidate

ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
ALPHAVANTAGE_DAILY_LIMIT = 25

log = logging.getLogger(__name__)


def _pct(text: str) -> float:
    return float(text.strip().rstrip("%"))


class AlphaVantageGainerSource:
    name = "alphavantage"
    requires_market_closed = False  # free tier updates after the close, so it always reports the last session

    def __init__(self, client: HttpClient, api_key: str, universe: Universe, quota: DailyQuota):
        self._client, self._api_key, self._universe, self._quota = client, api_key, universe, quota

    def top_candidates(self, session_date: date, prev_session_date: date, limit: int) -> list[Candidate]:
        # Tasks 22+23 fix round 1, K4: in replay no request is made, so no quota is spent -- quota.json is neither
        # read nor written, and a cassette recorded with Alpha Vantage replays any number of times a day.
        if not self._client.replaying:
            self._quota.consume()
        data = self._client.get_json(ALPHAVANTAGE_URL, params={"function": "TOP_GAINERS_LOSERS", "apikey": self._api_key})
        rows = data.get("top_gainers")
        if not rows:
            raise SourceError(f"alphavantage: unexpected payload keys {sorted(data)[:5]}")
        out: list[Candidate] = []
        skipped = 0
        for r in rows:
            try:
                symbol = self._universe.to_canonical(r["ticker"], "alphavantage")
                if not self._universe.is_common_stock(symbol):
                    continue
                price, change = float(r["price"]), float(r["change_amount"])
                out.append(Candidate(symbol=symbol, name=self._universe.record(symbol).name, prev_close=price - change,
                                     close=price, pct_change=_pct(r["change_percentage"]),
                                     volume=float(r.get("volume", 0)), source=self.name))
            except (KeyError, ValueError, TypeError):
                skipped += 1
        if skipped:
            log.warning("%s: skipped %d malformed gainer row(s)", self.name, skipped)
        out.sort(key=lambda c: c.pct_change, reverse=True)
        return out[:limit]
