from pathlib import Path

from ..config import Settings
from ..universe import Universe
from .alphavantage import ALPHAVANTAGE_DAILY_LIMIT, AlphaVantageGainerSource
from .base import GainerSource, HistorySource, NewsSource
from .http import DEFAULT_USER_AGENT, Cassette, DailyQuota, HttpClient, RateLimiter
from .massive import MASSIVE_CALLS_PER_MINUTE, MassiveGainerSource, MassiveHistorySource, MassiveNewsSource
from .nasdaqcom import NasdaqComGainerSource
from .sec import SEC_CALLS_PER_SECOND, SecFilingsNewsSource
from .yahoo import YahooScreenerGainerSource, YfinanceHistorySource, YfinanceNewsSource

SECONDS_PER_MINUTE = 60
ONE_SECOND = 1.0


def make_http_client(settings: Settings, cassette: Cassette | None = None,
                     rate_limiter: RateLimiter | None = None, user_agent: str = DEFAULT_USER_AGENT) -> HttpClient:
    return HttpClient(connect_timeout=settings.http_connect_timeout, read_timeout=settings.http_read_timeout,
                      rate_limiter=rate_limiter, cassette=cassette, user_agent=user_agent)


def massive_rate_limiter() -> RateLimiter:
    return RateLimiter(max_calls=MASSIVE_CALLS_PER_MINUTE, per_seconds=SECONDS_PER_MINUTE)


def sec_rate_limiter() -> RateLimiter:
    return RateLimiter(max_calls=SEC_CALLS_PER_SECOND, per_seconds=ONE_SECOND)


# Massive counts calls per API key, so every Massive source of a run takes the same limiter, passed in by the caller.
# The argument is required and keyword-only, so no caller can quietly give one source a limiter of its own.


def build_gainer_sources(settings: Settings, universe: Universe, state_dir: Path, cassette: Cassette | None = None,
                         *, massive_limiter: RateLimiter) -> list[GainerSource]:
    sources: list[GainerSource] = []
    if settings.massive_api_key:
        sources.append(MassiveGainerSource(make_http_client(settings, cassette, massive_limiter),
                                           settings.massive_api_key.get_secret_value(), universe))
    if settings.alphavantage_api_key:
        quota = DailyQuota("alphavantage", ALPHAVANTAGE_DAILY_LIMIT, state_dir / "quota.json")
        sources.append(AlphaVantageGainerSource(make_http_client(settings, cassette),
                                                settings.alphavantage_api_key.get_secret_value(), universe, quota))
    sources.append(YahooScreenerGainerSource(universe))
    sources.append(NasdaqComGainerSource(make_http_client(settings, cassette), universe))
    return sources


def build_history_sources(settings: Settings, universe: Universe, cassette: Cassette | None = None,
                          *, massive_limiter: RateLimiter) -> list[HistorySource]:
    sources: list[HistorySource] = [YfinanceHistorySource(universe)]
    if settings.massive_api_key:
        sources.append(MassiveHistorySource(make_http_client(settings, cassette, massive_limiter),
                                            settings.massive_api_key.get_secret_value(), universe))
    return sources


def build_news_sources(settings: Settings, cassette: Cassette | None = None,
                       *, massive_limiter: RateLimiter) -> list[NewsSource]:
    sources: list[NewsSource] = []
    if settings.massive_api_key:
        sources.append(MassiveNewsSource(make_http_client(settings, cassette, massive_limiter),
                                         settings.massive_api_key.get_secret_value()))
    sources.append(YfinanceNewsSource())
    if settings.sec_user_agent:
        # SEC asks every automated client to declare itself with a contact, so this source runs only with one set.
        sources.append(SecFilingsNewsSource(make_http_client(settings, cassette, sec_rate_limiter(),
                                                             user_agent=settings.sec_user_agent.get_secret_value())))
    return sources
