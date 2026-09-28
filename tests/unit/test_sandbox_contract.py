import json, pytest

GOOD = {"daily_changes_pct": [2.0, -2.451, 6.533, 0.943, 3.738], "avg_daily_change_pct": 2.153,
        "cumulative_return_pct": 11.0, "volatility_annualized_pct": 52.87, "max_drawdown_pct": -2.451,
        "relative_vs_spy_pct": 10.2, "trend": "uptrend"}

def test_parse_takes_last_sentinel_line():
    from nasdaq_agent.sandbox.contract import parse_sentinel_output, SENTINEL
    out = "noise\n" + SENTINEL + json.dumps({**GOOD, "trend": "mixed"}) + "\n" + SENTINEL + json.dumps(GOOD) + "\n"
    assert parse_sentinel_output(out).trend == "uptrend"

def test_parse_rejects_missing_sentinel_and_bad_schema():
    from nasdaq_agent.sandbox.contract import parse_sentinel_output, ContractError, SENTINEL
    with pytest.raises(ContractError):
        parse_sentinel_output("no result here")
    with pytest.raises(ContractError):
        parse_sentinel_output(SENTINEL + json.dumps({**GOOD, "extra": 1}))
    with pytest.raises(ContractError):
        parse_sentinel_output(SENTINEL + json.dumps({**GOOD, "avg_daily_change_pct": "NaN"}))

def test_tail_caps_length():
    from nasdaq_agent.sandbox.contract import tail, TAIL_CHARS
    assert len(tail("x" * (TAIL_CHARS * 3))) == TAIL_CHARS

def test_contract_error_omits_pydantic_url():
    # B7: the ContractError text reaches the model, so it must not carry pydantic's version-stamped
    # documentation URL (a dependency upgrade would otherwise change the prompt and miss the cache).
    from nasdaq_agent.sandbox.contract import parse_sentinel_output, ContractError, SENTINEL
    with pytest.raises(ContractError) as exc:
        parse_sentinel_output(SENTINEL + json.dumps({**GOOD, "extra": 1}))
    message = str(exc.value)
    assert "pydantic.dev" not in message and "https://" not in message
    assert "extra" in message  # the offending field is still named
