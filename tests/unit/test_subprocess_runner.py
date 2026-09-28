import pytest

GOOD_CODE = """
closes = df["adj_close"]
daily = closes.pct_change().dropna() * 100
result = {
    "daily_changes_pct": daily.round(6).tolist(),
    "avg_daily_change_pct": float(daily.mean()),
    "cumulative_return_pct": float((closes.iloc[-1] / closes.iloc[0] - 1) * 100),
    "volatility_annualized_pct": float(daily.std(ddof=1) * 252 ** 0.5),
    "max_drawdown_pct": float(((closes / closes.cummax()) - 1).min() * 100),
    "relative_vs_spy_pct": float(((closes.iloc[-1] / closes.iloc[0]) - (bench["adj_close"].iloc[-1] / bench["adj_close"].iloc[0])) * 100),
    "trend": "uptrend",
}
"""

@pytest.fixture
def sandbox_dir(tmp_run_dir, canonical_rows):
    from nasdaq_agent.sandbox.runner import prepare_sandbox_dir
    from nasdaq_agent.metrics import METRIC_NAMES, DEFINITIONS_TEXT
    cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    ticker_csv = ",".join(cols) + "\n" + "\n".join(",".join(r[c] for c in cols) for r in canonical_rows) + "\n"
    bench_csv = "date,open,high,low,close,adj_close,volume\n" + "\n".join(
        f'{r["date"]},{r["bench_close"]},{r["bench_close"]},{r["bench_close"]},{r["bench_close"]},{r["bench_close"]},1' for r in canonical_rows) + "\n"
    meta = {"symbol": "ACME", "session_date": "2026-09-24", "required_keys": list(METRIC_NAMES),
            "definitions": DEFINITIONS_TEXT, "close_column": "adj_close"}
    return prepare_sandbox_dir(tmp_run_dir, ticker_csv, bench_csv, meta)

def test_happy_path(sandbox_dir):
    from nasdaq_agent.sandbox.runner import SubprocessRunner
    r = SubprocessRunner().run(GOOD_CODE, sandbox_dir)
    assert r.exit_code == 0 and r.result is not None and r.backend == "subprocess"
    assert abs(r.result.cumulative_return_pct - 11.0) < 0.01

def test_traceback_is_returned(sandbox_dir):
    from nasdaq_agent.sandbox.runner import SubprocessRunner
    r = SubprocessRunner().run('closes = df["adj_close_typo"]\nresult = {}', sandbox_dir)
    assert r.exit_code != 0 and "KeyError" in r.stderr_tail and r.result is None

def test_gate_rejection_does_not_execute(sandbox_dir):
    from nasdaq_agent.sandbox.runner import SubprocessRunner, GATE_EXIT_CODE
    r = SubprocessRunner().run("import os\nresult = {}", sandbox_dir)
    assert r.exit_code == GATE_EXIT_CODE and "import of 'os'" in r.error

def test_timeout_kills_process(sandbox_dir):
    from nasdaq_agent.sandbox.runner import SubprocessRunner, SandboxLimits, TIMEOUT_EXIT_CODE
    r = SubprocessRunner(limits=SandboxLimits(wall_clock_s=1, cpu_s=5)).run("while True:\n    pass", sandbox_dir)
    assert r.timed_out and r.exit_code == TIMEOUT_EXIT_CODE

def test_url_read_rejected_by_gate(sandbox_dir):
    from nasdaq_agent.sandbox.runner import SubprocessRunner, GATE_EXIT_CODE
    code = "import pandas as pd\nresult = {}\npd.read_csv('https://example.com/x.csv')"
    r = SubprocessRunner().run(code, sandbox_dir)
    assert r.exit_code == GATE_EXIT_CODE and "url" in r.error

def test_missing_result_key_is_contract_violation(sandbox_dir):
    from nasdaq_agent.sandbox.runner import SubprocessRunner
    r = SubprocessRunner().run("result = {'trend': 'uptrend'}", sandbox_dir)
    assert r.exit_code != 0 and "missing keys" in r.stderr_tail

def test_bootstrap_blocks_sockets_at_runtime(sandbox_dir):
    import subprocess, sys
    from nasdaq_agent.sandbox.bootstrap import write_sandbox_files
    code = "import socket\nsocket.create_connection(('example.com', 80), timeout=2)\nresult = {}"
    bootstrap_path, code_path = write_sandbox_files(sandbox_dir, code)
    proc = subprocess.run([sys.executable, "-I", str(bootstrap_path), str(code_path)], cwd=sandbox_dir,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0 and "network access is disabled" in proc.stderr

def test_setup_failure_returns_runresult(sandbox_dir, monkeypatch):
    from nasdaq_agent.sandbox import runner as runner_module
    from nasdaq_agent.sandbox.runner import SubprocessRunner, SETUP_EXIT_CODE

    def _raise_oserror(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(runner_module.subprocess, "Popen", _raise_oserror)
    r = SubprocessRunner().run("result = {}", sandbox_dir)
    assert r.exit_code == SETUP_EXIT_CODE and "sandbox setup failed" in r.error


def test_bootstrap_blocks_reading_outside_sandbox_at_runtime(sandbox_dir, tmp_path):
    # B6: even code that slips past the static gate cannot read a file outside the sandbox. Run the
    # bootstrap directly (as the socket test does) to bypass the gate and prove the runtime open() guard.
    import subprocess, sys
    from nasdaq_agent.sandbox.bootstrap import write_sandbox_files
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-VALUE")
    code = f"data = open({str(secret)!r}).read()\nresult = {{'leak': data}}"
    bootstrap_path, code_path = write_sandbox_files(sandbox_dir, code)
    proc = subprocess.run([sys.executable, "-I", str(bootstrap_path), str(code_path)], cwd=sandbox_dir,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
    assert "TOP-SECRET-VALUE" not in proc.stdout  # the secret never left the sandbox
    assert "PermissionError" in proc.stderr

def test_bootstrap_blocks_writing_files_at_runtime(sandbox_dir):
    # B6: every write mode is refused, even for a path inside the sandbox directory.
    import subprocess, sys
    from nasdaq_agent.sandbox.bootstrap import write_sandbox_files
    target = sandbox_dir / "should_not_appear.txt"
    code = "open('should_not_appear.txt', 'w').write('nope')\nresult = {}"
    bootstrap_path, code_path = write_sandbox_files(sandbox_dir, code)
    proc = subprocess.run([sys.executable, "-I", str(bootstrap_path), str(code_path)], cwd=sandbox_dir,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0 and not target.exists()
    assert "writing files is disabled" in proc.stderr

def test_gate_bypass_write_is_stopped_at_runtime(sandbox_dir, tmp_path):
    # B6 end to end through the runner: apply('to_csv', ...) is a string-dispatch the static gate does
    # not catch, but the runtime open() guard refuses the write, so no file is created outside the sandbox.
    from nasdaq_agent.sandbox.runner import SubprocessRunner
    from nasdaq_agent.sandbox.gate import check_code
    outside = tmp_path / "written_by_apply.csv"
    code = f"df.apply('to_csv', path_or_buf={str(outside)!r})\nresult = {{}}"
    assert check_code(code).ok  # the gate lets the string dispatch through...
    r = SubprocessRunner().run(code, sandbox_dir)
    assert r.exit_code != 0 and not outside.exists()  # ...the runtime guard stops the write

def test_bootstrap_rebinds_numpy_datasource_opener(sandbox_dir, tmp_path):
    # Final residual 1b: numpy's DataSource captured the built-in open at import (before the bootstrap
    # wrapped it). The bootstrap rebinds its default opener to the guard. Run the bootstrap directly (gate
    # bypassed) and drive the rebound opener for both a read outside the sandbox and a write -- both fail closed.
    import subprocess, sys
    from nasdaq_agent.sandbox.bootstrap import write_sandbox_files
    secret = tmp_path / "fake.env"
    secret.write_text("MASSIVE_API_KEY=top-secret-value\n")
    victim = tmp_path / "victim.txt"
    victim.write_text("original")
    code = (
        "import numpy.lib._datasource as ds\n"
        f"try:\n    print('READ', repr(ds.open({str(secret)!r}).read()))\n"
        "except Exception as e:\n    print('READ_BLOCKED', type(e).__name__)\n"
        f"try:\n    ds.open({str(victim)!r}, 'w').write('overwritten')\n    print('WROTE')\n"
        "except Exception as e:\n    print('WRITE_BLOCKED', type(e).__name__)\n"
        "result = {}\n"
    )
    bootstrap_path, code_path = write_sandbox_files(sandbox_dir, code)
    proc = subprocess.run([sys.executable, "-I", str(bootstrap_path), str(code_path)], cwd=sandbox_dir,
                          capture_output=True, text=True, timeout=60)
    assert "READ_BLOCKED PermissionError" in proc.stdout, (proc.stdout, proc.stderr)
    assert "WRITE_BLOCKED PermissionError" in proc.stdout, (proc.stdout, proc.stderr)
    assert "top-secret-value" not in proc.stdout  # the secret never left the sandbox
    assert victim.read_text() == "original"  # the write never happened

def test_relative_sandbox_dir_is_resolved_by_the_runner(sandbox_dir, monkeypatch):
    """Tasks 22+23 fix round 1, K9: the child runs inside the sandbox directory, so the runner resolves it itself (as
    DockerRunner does) -- a relative path to the bootstrap would otherwise be resolved against the sandbox again."""
    from pathlib import Path
    from nasdaq_agent.sandbox.runner import SubprocessRunner
    monkeypatch.chdir(sandbox_dir.parent)
    r = SubprocessRunner().run(GOOD_CODE, Path(sandbox_dir.name))
    assert r.exit_code == 0 and r.result is not None, r.stderr_tail
