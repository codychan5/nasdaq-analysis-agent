import json, logging


def test_run_dir_layout_and_json(tmp_path):
    from nasdaq_agent.artifacts import RunDir, sha256_text, new_run_id
    rd = RunDir(tmp_path, "run-1")
    rd.write_json("a.json", {"x": 1}); rd.append_jsonl("tool_log.jsonl", {"t": 1}); rd.append_jsonl("tool_log.jsonl", {"t": 2})
    assert rd.read_json("a.json") == {"x": 1}
    assert (rd.path / "tool_log.jsonl").read_text().count("\n") == 2
    assert rd.sandbox_dir == rd.path / "sandbox" and rd.outbox_dir == rd.path / "outbox"
    assert len(sha256_text("abc")) == 16 and new_run_id() != new_run_id()


def test_logging_redacts_secrets(tmp_path, capsys):
    from nasdaq_agent.artifacts import RunDir, configure_logging
    rd = RunDir(tmp_path, "run-2")
    log = configure_logging(rd, secrets=["hunter2", "sk-live-123"])
    log.info("connecting with password=hunter2 and key sk-live-123", extra={"password": "hunter2"})
    line = (rd.path / "log.jsonl").read_text().strip().splitlines()[-1]
    assert "hunter2" not in line and "sk-live-123" not in line and "***" in line
    assert "hunter2" not in capsys.readouterr().out


def test_logging_formats_percent_style_args_before_scrubbing(tmp_path):
    # Controller correction 1: SecretRedactingFilter must format the record (substitute
    # %s-style args via record.getMessage()) before scrubbing, then clear args. Scrubbing
    # the raw format string and clearing args afterwards would turn every "tool %s: %s"
    # line into that literal, unsubstituted text instead of the redacted real message.
    from nasdaq_agent.artifacts import RunDir, configure_logging
    rd = RunDir(tmp_path, "run-3")
    log = configure_logging(rd, secrets=["hunter2"])
    log.info("tool %s: %s", "find_top_gainer", "key=hunter2")
    line = (rd.path / "log.jsonl").read_text().strip().splitlines()[-1]
    assert "tool find_top_gainer: key=***" in line


def test_configure_logging_closes_previous_handlers(tmp_path):
    # Controller correction 2: configure_logging must close and remove existing handlers
    # before adding new ones, so repeated calls in the same process (e.g. successive runs)
    # do not leak open log file descriptors.
    from nasdaq_agent.artifacts import RunDir, configure_logging
    rd_a = RunDir(tmp_path, "run-4a")
    log_a = configure_logging(rd_a, secrets=[])
    first_handlers = list(log_a.handlers)
    assert len(first_handlers) == 2

    rd_b = RunDir(tmp_path, "run-4b")
    log_b = configure_logging(rd_b, secrets=[])
    assert log_b is log_a
    assert len(log_b.handlers) == 2
    assert not (set(first_handlers) & set(log_b.handlers))
    for h in first_handlers:
        if isinstance(h, logging.FileHandler):
            assert h.stream is None


def test_logging_scrubs_exception_text(tmp_path, capsys):
    # Review finding: _JsonFormatter.format never serialised exception information, so
    # log.exception(...) (used by the tool-crash handler in a later task) wrote a line with
    # no exception type, message, or traceback. The fix must also scrub that text before it
    # reaches any handler, so a secret embedded in an exception message cannot leak either.
    from nasdaq_agent.artifacts import RunDir, configure_logging
    rd = RunDir(tmp_path, "run-6")
    log = configure_logging(rd, secrets=["hunter2"])
    try:
        raise ValueError("bad key hunter2")
    except ValueError:
        log.exception("provider call failed")
    line = (rd.path / "log.jsonl").read_text().strip().splitlines()[-1]
    payload = json.loads(line)
    assert "ValueError" in payload["exc"] and "***" in payload["exc"]
    assert "hunter2" not in payload["exc"]
    assert "hunter2" not in line
    assert "hunter2" not in capsys.readouterr().out
