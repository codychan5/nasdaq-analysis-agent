"""Record and replay for the sources that do not use our HTTP client: the yfinance-backed adapters."""
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ..sources.errors import CassetteMiss
from ..sources.models import BarSeries, Candidate, CorporateAction, Headline
from .failures import failure_entry, is_failure_entry, raise_recorded_failure
from .files import REPLAY, read_json_entry, require_cassette_mode, write_json_atomic

SOURCES_DIR = "sources"
# Each method's result model, and whether it returns one object or a list of them. A stored entry of the other shape
# is refused.
RECORDED_METHODS: dict[str, tuple[type[BaseModel], type]] = {
    "top_candidates": (Candidate, list), "bars": (BarSeries, dict),
    "corporate_actions": (CorporateAction, list), "headlines": (Headline, list)}
# The "yahoo" gainer source calls yf.screen and the "yfinance" history and news sources call yf.Ticker, so their
# traffic never reaches the HTTP cassette; every other adapter goes through HttpClient.
YFINANCE_BACKED_SOURCE_NAMES = frozenset({"yahoo", "yfinance"})


class RecordedSource:
    """Wraps a source whose calls are not HTTP through our client (yfinance) so replay can serve them."""

    def __init__(self, inner, root: Path, mode: str):
        self._inner, self._root, self._mode = inner, Path(root) / SOURCES_DIR, require_cassette_mode(mode)
        self._root.mkdir(parents=True, exist_ok=True)
        self.name = inner.name
        self.requires_market_closed = getattr(inner, "requires_market_closed", False)

    def _key(self, method: str, args: tuple) -> Path:
        raw = json.dumps({"src": self.name, "m": method, "a": [str(a) for a in args]}, sort_keys=True)
        return self._root / (hashlib.sha256(raw.encode()).hexdigest() + ".json")

    def _call(self, method: str, *args: Any):
        path, (model, shape) = self._key(method, args), RECORDED_METHODS[method]
        if path.exists():
            data = read_json_entry(path)
            if is_failure_entry(data):
                raise_recorded_failure(data, path)
            return _rebuild(data, model, shape, path)
        if self._mode == REPLAY:
            raise CassetteMiss(f"no recording for {self.name}.{method}{args}; run `nasdaq-agent record` again")
        try:
            result = getattr(self._inner, method)(*args)
        except Exception as e:
            # The failure is recorded too, so replay shows the model the same failure. yfinance adapters hold no
            # credentials, and the seal's secret scan would refuse the recording if a message carried one.
            write_json_atomic(path, failure_entry(e))
            raise
        payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else [r.model_dump(mode="json") for r in result]
        write_json_atomic(path, payload)
        return result

    def top_candidates(self, session_date, prev_session_date, limit):
        return self._call("top_candidates", session_date, prev_session_date, limit)

    def bars(self, symbol, start, end):
        return self._call("bars", symbol, start, end)

    def corporate_actions(self, symbol, start, end):
        return self._call("corporate_actions", symbol, start, end)

    def headlines(self, symbol, since, until):
        return self._call("headlines", symbol, since, until)


def _rebuild(data: Any, model: type[BaseModel], shape: type, path: Path):
    try:
        if shape is dict and isinstance(data, dict):
            return model.model_validate(data)
        if shape is list and isinstance(data, list):
            return [model.model_validate(item) for item in data]
    except ValidationError:
        pass
    raise CassetteMiss(f"cassette entry {path} is malformed; run `nasdaq-agent record` again")


def wrap_sources(sources: list, root: Path, mode: str) -> list:
    return [RecordedSource(s, root, mode) if s.name in YFINANCE_BACKED_SOURCE_NAMES else s for s in sources]
