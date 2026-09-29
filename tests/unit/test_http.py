import json, logging, time
import httpx, pytest, respx

def make_client(**kw):
    from nasdaq_agent.sources.http import HttpClient
    return HttpClient(connect_timeout=1, read_timeout=1, **kw)

def recorded_waits():
    """A stand-in for time.sleep between retries: it records each wait and returns at once, so no test sleeps."""
    waits = []
    return waits, waits.append

@respx.mock
def test_a_429_without_retry_after_backs_off_from_two_seconds():
    route = respx.get("https://api.example.com/x").mock(side_effect=[
        httpx.Response(429), httpx.Response(429), httpx.Response(200, json={"ok": True})])
    waits, sleep = recorded_waits()
    assert make_client(retry_sleep=sleep).get_json("https://api.example.com/x") == {"ok": True}
    assert route.call_count == 3
    assert len(waits) == 2
    assert 1.6 <= waits[0] <= 2.4 and 3.2 <= waits[1] <= 4.8  # 2 s, then 4 s, each within 20% jitter

@respx.mock
def test_a_429_that_never_clears_gives_up_after_six_attempts():
    route = respx.get("https://api.example.com/x").mock(return_value=httpx.Response(429))
    from nasdaq_agent.sources.errors import SourceUnavailable
    waits, sleep = recorded_waits()
    with pytest.raises(SourceUnavailable) as exc_info:
        make_client(retry_sleep=sleep).get_json("https://api.example.com/x", params={"apiKey": "SECRET"})
    assert route.call_count == 6
    assert len(waits) == 5 and sum(waits) <= 90  # 2 + 4 + 8 + 16 + 32 s, each within 20% jitter
    assert "SECRET" not in str(exc_info.value)

@respx.mock
def test_waiting_stops_before_it_would_pass_ninety_seconds_in_total():
    route = respx.get("https://api.example.com/x").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "50"}))
    from nasdaq_agent.sources.errors import SourceUnavailable
    waits, sleep = recorded_waits()
    with pytest.raises(SourceUnavailable):
        make_client(retry_sleep=sleep).get_json("https://api.example.com/x")
    assert waits == [50.0] and route.call_count == 2  # a second 50 s wait would make 100 s

@respx.mock
def test_a_retry_wait_is_logged_with_the_host_but_not_the_query_string(caplog):
    respx.get("https://api.example.com/x").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(200, json={"ok": True})])
    _, sleep = recorded_waits()
    with caplog.at_level(logging.WARNING, logger="nasdaq_agent.sources.http"):
        make_client(retry_sleep=sleep).get_json("https://api.example.com/x", params={"apiKey": "SECRET"})
    assert "api.example.com" in caplog.text and "HTTP 429" in caplog.text
    assert "SECRET" not in caplog.text

@respx.mock
def test_a_429_waits_as_long_as_retry_after_says():
    respx.get("https://api.example.com/x").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(200, json={"ok": True})])
    waits, sleep = recorded_waits()
    assert make_client(retry_sleep=sleep).get_json("https://api.example.com/x") == {"ok": True}
    assert waits == [3.0]

@respx.mock
def test_a_retry_after_longer_than_a_minute_is_capped_at_a_minute():
    respx.get("https://api.example.com/x").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "3600"}), httpx.Response(200, json={"ok": True})])
    waits, sleep = recorded_waits()
    assert make_client(retry_sleep=sleep).get_json("https://api.example.com/x") == {"ok": True}
    assert waits == [60.0]

# Raw header bytes, as a server sends them: httpx decodes the byte 0xB2 as "²", which str.isdigit() accepts.
@pytest.mark.parametrize("header", [b"soon", b"", b"0", b"-5", b"1.5", b"\xb2", b"Mon, 01 Jan 2001 00:00:00 GMT"])
@respx.mock
def test_an_unusable_retry_after_falls_back_to_the_backoff(header):
    respx.get("https://api.example.com/x").mock(side_effect=[
        httpx.Response(429, headers=[(b"Retry-After", header)]), httpx.Response(200, json={"ok": True})])
    waits, sleep = recorded_waits()
    assert make_client(retry_sleep=sleep).get_json("https://api.example.com/x") == {"ok": True}
    assert len(waits) == 1 and 1.6 <= waits[0] <= 2.4

def test_a_retry_after_date_means_the_seconds_until_then():
    from datetime import datetime, timezone
    from nasdaq_agent.sources.http import parse_retry_after
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("Mon, 28 Sep 2026 12:00:05 GMT", now) == 5.0

def test_a_retry_after_date_without_a_time_zone_is_unusable():
    from datetime import datetime, timezone
    from nasdaq_agent.sources.http import parse_retry_after
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("Mon, 28 Sep 2026 12:00:05", now) is None

@respx.mock
def test_server_errors_keep_three_quick_attempts():
    route = respx.get("https://api.example.com/x").mock(return_value=httpx.Response(503))
    from nasdaq_agent.sources.errors import SourceUnavailable
    waits, sleep = recorded_waits()
    with pytest.raises(SourceUnavailable):
        make_client(retry_sleep=sleep).get_json("https://api.example.com/x")
    assert route.call_count == 3
    assert len(waits) == 2 and all(wait <= 2.0 for wait in waits)  # 0.5 to 1 s, then 0.5 to 2 s

@respx.mock
def test_source_unavailable_does_not_leak_request_details():
    respx.get("https://api.example.com/x").mock(side_effect=httpx.ConnectError("boom"))
    from nasdaq_agent.sources.errors import SourceUnavailable
    _, sleep = recorded_waits()
    with pytest.raises(SourceUnavailable) as exc_info:
        make_client(retry_sleep=sleep).get_json("https://api.example.com/x", params={"apiKey": "SECRET"})
    exc = exc_info.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert "SECRET" not in str(exc)

@respx.mock
def test_404_is_not_retried():
    route = respx.get("https://api.example.com/x").mock(return_value=httpx.Response(404))
    from nasdaq_agent.sources.errors import SourceError
    with pytest.raises(SourceError):
        make_client().get_json("https://api.example.com/x")
    assert route.call_count == 1

@respx.mock
def test_get_json_rejects_non_json_body():
    respx.get("https://api.example.com/x").mock(
        return_value=httpx.Response(200, text="<html>not json</html>"))
    from nasdaq_agent.sources.errors import SourceError
    with pytest.raises(SourceError):
        make_client().get_json("https://api.example.com/x")

def test_rate_limiter_spaces_calls():
    from nasdaq_agent.sources.http import RateLimiter
    rl = RateLimiter(max_calls=2, per_seconds=0.2)
    t0 = time.monotonic()
    for _ in range(3):
        rl.acquire()
    assert time.monotonic() - t0 >= 0.2

def test_daily_quota_persists(tmp_path):
    from nasdaq_agent.sources.http import DailyQuota
    from nasdaq_agent.sources.errors import SourceQuotaExceeded
    q = DailyQuota("av", limit=2, state_path=tmp_path / "quota.json")
    q.consume(); q.consume()
    q2 = DailyQuota("av", limit=2, state_path=tmp_path / "quota.json")
    with pytest.raises(SourceQuotaExceeded):
        q2.consume()

def test_daily_quota_thread_safe(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from nasdaq_agent.sources.http import DailyQuota
    from nasdaq_agent.sources.errors import SourceQuotaExceeded
    q = DailyQuota("av", limit=5, state_path=tmp_path / "quota.json")

    def attempt(_):
        try:
            q.consume()
            return True
        except SourceQuotaExceeded:
            return False

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(attempt, range(10)))

    assert results.count(True) == 5
    assert results.count(False) == 5
    persisted = json.loads((tmp_path / "quota.json").read_text())
    assert persisted["counts"]["av"] == 5

class MemoryCassette:
    def __init__(self, mode): self.mode, self.data = mode, {}
    def lookup(self, key): return self.data.get(key)
    def store(self, key, status, text): self.data[key] = (status, text)

@respx.mock
def test_cassette_record_then_replay():
    respx.get("https://api.example.com/x").mock(return_value=httpx.Response(200, json={"v": 1}))
    rec = MemoryCassette("record")
    assert make_client(cassette=rec).get_json("https://api.example.com/x") == {"v": 1}
    rep = MemoryCassette("replay"); rep.data = rec.data
    with respx.mock(assert_all_called=False) as m:
        m.get("https://api.example.com/x").mock(return_value=httpx.Response(500))
        assert make_client(cassette=rep).get_json("https://api.example.com/x") == {"v": 1}

def test_cassette_replay_miss_raises():
    from nasdaq_agent.sources.errors import CassetteMiss
    with pytest.raises(CassetteMiss):
        make_client(cassette=MemoryCassette("replay")).get_json("https://api.example.com/missing")

@respx.mock
def test_a_dropped_connection_is_retried_like_a_timeout():
    route = respx.get("https://api.example.com/x").mock(side_effect=[
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.Response(200, json={"ok": True})])
    _, sleep = recorded_waits()
    assert make_client(retry_sleep=sleep).get_json("https://api.example.com/x") == {"ok": True}
    assert route.call_count == 2

@respx.mock
def test_a_read_error_that_persists_ends_as_source_unavailable():
    from nasdaq_agent.sources.errors import SourceUnavailable
    route = respx.get("https://api.example.com/x").mock(side_effect=httpx.ReadError("connection reset by peer"))
    _, sleep = recorded_waits()
    with pytest.raises(SourceUnavailable) as exc_info:
        make_client(retry_sleep=sleep).get_json("https://api.example.com/x", params={"apiKey": "SECRET"})
    assert route.call_count == 3 and "SECRET" not in str(exc_info.value)

@respx.mock
def test_a_request_error_retrying_cannot_fix_fails_at_once_as_a_source_error():
    """Still our own SourceError, so record mode stores it and replay repeats it, with no raw httpx error attached."""
    from nasdaq_agent.sources.errors import SourceError, SourceUnavailable
    route = respx.get("https://api.example.com/x").mock(side_effect=httpx.UnsupportedProtocol("unsupported protocol"))
    _, sleep = recorded_waits()
    with pytest.raises(SourceError) as exc_info:
        make_client(retry_sleep=sleep).get_json("https://api.example.com/x", params={"apiKey": "SECRET"})
    exc = exc_info.value
    assert route.call_count == 1 and not isinstance(exc, SourceUnavailable)
    assert exc.__cause__ is None and exc.__context__ is None and "SECRET" not in str(exc)
