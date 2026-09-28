def test_package_imports():
    import nasdaq_agent
    assert nasdaq_agent.__version__ == "0.1.0"

def test_canonical_fixture(canonical_closes, canonical_bench_closes):
    assert canonical_closes == [10.00, 10.20, 9.95, 10.60, 10.70, 11.10]
    assert canonical_bench_closes == [500.0, 501.0, 499.0, 502.0, 503.0, 504.0]
