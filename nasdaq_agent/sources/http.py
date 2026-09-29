import email.utils
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import httpx
from tenacity import RetryCallState, retry, retry_if_exception_type, wait_random_exponential

from .errors import CassetteMiss, SourceError, SourceQuotaExceeded, SourceUnavailable

log = logging.getLogger(__name__)

RETRY_ATTEMPTS = 3
# Transport failures a second try can outlast: timeouts, network errors (connect, read, write, close) and a server that
# hangs up mid-response. Other request errors, such as an unsupported protocol, would fail the same way every time.
TRANSIENT_TRANSPORT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
RETRY_WAIT_MIN_SECONDS = 0.5
RETRY_WAIT_MAX_SECONDS = 8.0
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
# A 429 means the provider's rate-limit window is full, and quick retries cannot outlast a window that is often a
# minute long. So a rate-limited request waits as long as the server's Retry-After says, or else backs off
# exponentially from a longer first wait, with more attempts but one budget for all its waits.
RATE_LIMITED_STATUS = 429
RATE_LIMIT_ATTEMPTS = 6
RATE_LIMIT_FIRST_WAIT_SECONDS = 2.0
RATE_LIMIT_MAX_WAIT_SECONDS = 60.0  # the longest single wait a server's Retry-After may impose
RATE_LIMIT_WAIT_BUDGET_SECONDS = 90.0  # the most one request may spend waiting, so one source cannot eat the run
RATE_LIMIT_JITTER = 0.2  # each rate-limit wait is scaled by a random factor between 0.8 and 1.2
DELAY_SECONDS_PATTERN = re.compile(r"[0-9]+")  # Retry-After as a whole number of seconds
DEFAULT_USER_AGENT = "nasdaq-agent/0.1 (proof of concept)"


class TransientHttpError(Exception):
    pass


class RateLimitedHttpError(TransientHttpError):
    """HTTP 429: the provider's rate-limit window is full. retry_after_seconds is the server's own wait, when usable."""

    def __init__(self, retry_after_seconds: float | None):
        super().__init__(f"HTTP {RATE_LIMITED_STATUS}")
        self.retry_after_seconds = retry_after_seconds


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    """Seconds to wait from a Retry-After header (RFC 9110, section 10.2.3), which is either delay-seconds or an
    HTTP-date. None when the header is absent or unusable -- neither plain ASCII digits nor a dated GMT time, or not in
    the future -- so the caller backs off on its own schedule instead. str.isdigit() is not enough for the digits: it
    accepts characters such as "²" that float() rejects."""
    if value is None:
        return None
    text = value.strip()
    if DELAY_SECONDS_PATTERN.fullmatch(text):
        seconds = float(text)
    else:
        try:
            moment = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:  # an HTTP-date is always in GMT; one without a zone is malformed
            return None
        seconds = (moment - now).total_seconds()
    return seconds if seconds > 0 else None


def _retry_wait_seconds(retry_state: RetryCallState) -> float:
    """The wait before the next attempt: quick jittered waits after a network error or a 5xx; after a 429, the
    server's Retry-After, or else an exponential backoff from RATE_LIMIT_FIRST_WAIT_SECONDS."""
    error = retry_state.outcome.exception()
    if not isinstance(error, RateLimitedHttpError):
        return wait_random_exponential(min=RETRY_WAIT_MIN_SECONDS, max=RETRY_WAIT_MAX_SECONDS)(retry_state)
    if error.retry_after_seconds is not None:
        return min(error.retry_after_seconds, RATE_LIMIT_MAX_WAIT_SECONDS)
    backoff = RATE_LIMIT_FIRST_WAIT_SECONDS * 2 ** (retry_state.attempt_number - 1)
    return backoff * random.uniform(1 - RATE_LIMIT_JITTER, 1 + RATE_LIMIT_JITTER)


def _retry_should_stop(retry_state: RetryCallState) -> bool:
    """The latest failure decides. A 429 may use RATE_LIMIT_ATTEMPTS attempts while its waits, the next one included,
    stay within RATE_LIMIT_WAIT_BUDGET_SECONDS; anything else gets RETRY_ATTEMPTS. tenacity computes the next wait
    before it asks whether to stop, and idle_for sums the waits so far."""
    error = retry_state.outcome.exception()
    if isinstance(error, RateLimitedHttpError):
        over_budget = retry_state.idle_for + retry_state.upcoming_sleep > RATE_LIMIT_WAIT_BUDGET_SECONDS
        return over_budget or retry_state.attempt_number >= RATE_LIMIT_ATTEMPTS
    return retry_state.attempt_number >= RETRY_ATTEMPTS


class Cassette(Protocol):
    mode: str  # "off", "record" or "replay"
    def lookup(self, key: str) -> tuple[int, str] | None: ...
    def store(self, key: str, status: int, text: str) -> None: ...
    # A request that failed after its retries while recording, so replay fails it the same
    # way; lookup raises that recorded failure again.
    def store_failure(self, key: str, error: Exception) -> None: ...


# Query parameters that carry a credential (Massive sends apiKey, Alpha Vantage apikey), compared case-insensitively.
# They are left out of the request hash so a cassette recorded with keys replays with none -- and so the
# cassette's file names never depend on a secret.
CREDENTIAL_PARAMS = frozenset({"apikey", "api_key"})


def request_key(method: str, url: str, params: dict[str, Any] | None) -> str:
    kept = {name: value for name, value in (params or {}).items() if name.lower() not in CREDENTIAL_PARAMS}
    canonical = json.dumps({"m": method, "u": url, "p": sorted(kept.items())}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class RateLimiter:
    """Sliding window: at most max_calls in any trailing per_seconds window. It keeps a timestamp per
    recent call, drops those older than the window, and when the window is full blocks until the oldest
    one ages out. (Not a token bucket, which refills at a steady rate; this bounds calls per window.)"""

    def __init__(self, max_calls: int, per_seconds: float):
        self.max_calls, self.per_seconds = max_calls, per_seconds
        self._stamps: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._stamps = [s for s in self._stamps if now - s < self.per_seconds]
            if len(self._stamps) >= self.max_calls:
                time.sleep(self.per_seconds - (now - self._stamps[0]))
                now = time.monotonic()
                self._stamps = [s for s in self._stamps if now - s < self.per_seconds]
            self._stamps.append(time.monotonic())


_QUOTA_LOCK = threading.Lock()


class DailyQuota:
    """Persisted per-day call counter for providers with small daily limits.

    Thread-safe within one process via a module-level lock; separate processes sharing the same state_path are not coordinated.
    """

    def __init__(self, name: str, limit: int, state_path: Path):
        self.name, self.limit, self.state_path = name, limit, state_path

    def _load(self) -> dict[str, Any]:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            if data.get("date") == date.today().isoformat():
                return data
        return {"date": date.today().isoformat(), "counts": {}}

    def _store_atomic(self, data: dict[str, Any]) -> None:
        """Write via a same-directory temp file plus rename, so a reader never observes a partial write."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data))
        os.replace(tmp_path, self.state_path)

    def consume(self) -> int:
        with _QUOTA_LOCK:
            data = self._load()
            used = data["counts"].get(self.name, 0)
            if used >= self.limit:
                raise SourceQuotaExceeded(f"{self.name}: daily quota of {self.limit} used")
            data["counts"][self.name] = used + 1
            self._store_atomic(data)
            return used + 1


class HttpClient:
    def __init__(self, connect_timeout: float, read_timeout: float, rate_limiter: RateLimiter | None = None,
                 cassette: Cassette | None = None, user_agent: str = DEFAULT_USER_AGENT,
                 retry_sleep: Callable[[float], None] = time.sleep):
        self._client = httpx.Client(timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
                                    headers={"User-Agent": user_agent}, follow_redirects=True)
        self._rate_limiter = rate_limiter
        self._cassette = cassette
        # How the client waits between retries. Tests pass a recorder, so they check each wait without sleeping.
        self._retry_sleep = retry_sleep

    @property
    def replaying(self) -> bool:
        """True when responses come from a replay cassette: no request is made, so no quota is spent."""
        return self._cassette is not None and self._cassette.mode == "replay"

    @property
    def _recording(self) -> bool:
        return self._cassette is not None and self._cassette.mode == "record"

    def _fetch(self, url: str, params: dict[str, Any] | None, headers: dict[str, str] | None) -> tuple[int, str]:
        key = request_key("GET", url, params)
        if self.replaying:
            hit = self._cassette.lookup(key)  # raises a recorded failure again
            if hit is None:
                raise CassetteMiss(f"no recording for GET {url}")
            return hit
        try:
            status, text = self._fetch_live(url, params, headers)
        except SourceError as e:
            # Only our own failures are recorded. SourceUnavailable carries no request details (see
            # _fetch_live), whereas a raw httpx error's text can name the full URL, credentials included.
            if self._recording:
                self._cassette.store_failure(key, e)
            raise
        if self._recording:
            self._cassette.store(key, status, text)
        return status, text

    def _fetch_live(self, url, params, headers) -> tuple[int, str]:
        host = httpx.URL(url).host

        def log_wait(retry_state: RetryCallState) -> None:
            # The host only, never the URL or its query: Massive and Alpha Vantage carry their API key in the query.
            error = retry_state.outcome.exception()
            log.warning("GET %s failed (%s: %s); retrying in %.1f s", host, type(error).__name__, error,
                        retry_state.upcoming_sleep)

        @retry(stop=_retry_should_stop, wait=_retry_wait_seconds, retry=retry_if_exception_type(TransientHttpError),
               sleep=self._retry_sleep, before_sleep=log_wait, reraise=True)
        def attempt() -> tuple[int, str]:
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            error_message: str | None = None
            try:
                resp = self._client.get(url, params=params, headers=headers)
            except TRANSIENT_TRANSPORT_ERRORS as e:
                raise TransientHttpError(str(e)) from e
            except httpx.RequestError as e:
                # Not worth retrying, such as an unsupported protocol, but still our own SourceError, so record mode
                # stores it and replay repeats it. The type alone: a raw httpx error's text can name the full URL.
                error_message = f"GET {url}: {type(e).__name__}"
            if error_message is not None:
                raise SourceError(error_message)  # outside the except block, so no raw httpx error is chained
            if resp.status_code in RETRYABLE_STATUSES:
                if resp.status_code == RATE_LIMITED_STATUS:
                    raise RateLimitedHttpError(parse_retry_after(resp.headers.get("Retry-After"),
                                                                 datetime.now(timezone.utc)))
                raise TransientHttpError(f"HTTP {resp.status_code}")
            return resp.status_code, resp.text

        error_message: str | None = None
        try:
            return attempt()
        except TransientHttpError as e:
            error_message = f"GET {url}: {type(e).__name__}: {e}"
        raise SourceUnavailable(error_message)

    def get_text(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> str:
        status, text = self._fetch(url, params, headers)
        if status >= 400:
            raise SourceError(f"GET {url} returned HTTP {status}")
        return text

    def get_json(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        text = self.get_text(url, params, headers)
        error_message: str | None = None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            error_message = f"GET {url} returned a non-JSON body: {text[:80]!r}"
        raise SourceError(error_message)
