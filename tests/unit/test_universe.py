import pytest

SAMPLE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
BRK.B|Berkshire Hathaway Inc. - Class B Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ASTLW|Astra Space, Inc. - Warrant|S|N|N|100|N|N
ACMEU|Acme Acquisition Corp. - Unit|G|N|N|100|N|N
ACMER|Acme Acquisition Corp. - Right|G|N|N|100|N|N
BANKP|Bank Corp - Depositary Shares Series A Preferred Stock|S|N|N|100|N|N
TSLA|Tesla, Inc. - Common Stock|Q|N|N|100|N|N
ZTEST|NASDAQ TEST STOCK|G|Y|N|100|N|N
BABA|Alibaba Group Holding Limited - American Depositary Shares|Q|N|N|100|N|N
PACA|Preferred Apartment Communities, Inc. - Class A Common Stock|Q|N|N|100|N|N
XBND|Xyz Corp - 6.5% Notes due 2031|G|N|N|100|N|N
File Creation Time: 0925202621:31|||||||
"""

def test_parse_and_classify():
    from nasdaq_agent.universe import Universe
    u = Universe.from_text(SAMPLE)
    assert u.is_common_stock("AAPL") and u.is_common_stock("BRK.B") and u.is_common_stock("BABA")
    assert u.is_common_stock("PACA"), "PACA (Preferred Apartment...) should be common stock"
    for sym in ["QQQ", "ASTLW", "ACMEU", "ACMER", "BANKP", "ZTEST"]:
        assert not u.is_common_stock(sym), sym
    assert u.record("QQQ").exclusion_reason == "etf"
    assert u.record("ZTEST").exclusion_reason == "test issue"
    assert u.record("ASTLW").exclusion_reason == "warrant"
    assert u.record("ACMEU").exclusion_reason == "unit"
    assert u.record("ACMER").exclusion_reason == "right"
    assert u.record("BANKP").exclusion_reason == "preferred"
    assert u.record("XBND").exclusion_reason == "note"
    assert u.record("SLND+") is None and not u.is_common_stock("SLND+")

def test_symbol_round_trip_across_providers():
    from nasdaq_agent.universe import Universe
    u = Universe.from_text(SAMPLE)
    assert u.to_provider("BRK.B", "yahoo") == "BRK-B"
    assert u.to_canonical("BRK-B", "yahoo") == "BRK.B"
    assert u.to_provider("BRK.B", "massive") == "BRK.B"
    assert u.to_canonical("BRK.B", "alphavantage") == "BRK.B"
    assert u.to_canonical("AAPL", "yahoo") == "AAPL"

def test_load_universe_uses_cache(tmp_path):
    from nasdaq_agent.universe import load_universe
    calls = []
    def fetch(url):
        calls.append(url)
        return SAMPLE
    cache = tmp_path / "nasdaqlisted.txt"
    u1 = load_universe(fetch, cache)
    u2 = load_universe(fetch, cache)
    assert len(calls) == 1 and u2.is_common_stock("TSLA")

def test_load_universe_refetches_when_stale(tmp_path):
    import os
    from nasdaq_agent.universe import load_universe
    calls = []
    def fetch(url):
        calls.append(url)
        return SAMPLE
    cache = tmp_path / "nasdaqlisted.txt"
    cache.write_text(SAMPLE)
    # Set mtime to 48 hours in the past
    old_time = os.stat(cache).st_mtime - (48 * 3600)
    os.utime(cache, (old_time, old_time))
    u = load_universe(fetch, cache, max_age_hours=24)
    assert len(calls) == 1 and u.is_common_stock("TSLA")

def test_empty_universe_raises():
    from nasdaq_agent.universe import Universe, UniverseError
    with pytest.raises(UniverseError, match="Empty or malformed"):
        Universe.from_text("")

def test_missing_column_raises():
    from nasdaq_agent.universe import Universe, UniverseError
    bad_header = "Symbol|Security Name|Market Category|Test Issue"
    with pytest.raises(UniverseError, match="Missing required column: ETF"):
        Universe.from_text(bad_header)

def test_fetch_exception_wrapped():
    from nasdaq_agent.universe import load_universe, UniverseError
    def fetch_error(url):
        raise RuntimeError("Network error")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cache = __import__("pathlib").Path(tmp) / "cache.txt"
        with pytest.raises(UniverseError, match="Failed to fetch"):
            load_universe(fetch_error, cache)
