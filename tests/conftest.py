import csv
import logging
from pathlib import Path
import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def _restore_nasdaq_agent_logger():
    """configure_logging(...) (nasdaq_agent/artifacts.py) mutates the
    process-global "nasdaq_agent" logger -- handlers, filters, level, propagate -- and nothing
    puts it back, so whichever test last called it (directly, or via graph._prepare) leaves that
    state for every test that runs after it in the same process. pytest's own caplog handler
    re-attaches itself to any logger that already has propagate=False *before* a given test starts
    (see _pytest.logging.catching_logs), but by its own admission "misses loggers that become
    non-propagating after __enter__" -- so a test that flips propagate mid-test while also using
    caplog on a "nasdaq_agent.*" child logger is still exposed. Snapshotting before each test and
    restoring after (closing whatever handlers were added, so the FileHandler configure_logging
    attaches can't leak an open file descriptor past the test that created it) removes the shared
    mutable state entirely rather than relying on that partial workaround.
    """
    logger = logging.getLogger("nasdaq_agent")
    original_handlers = list(logger.handlers)
    original_filters = list(logger.filters)
    original_level = logger.level
    original_propagate = logger.propagate
    yield
    for handler in logger.handlers:
        if handler not in original_handlers:
            handler.close()
    logger.handlers = original_handlers
    logger.filters = original_filters
    logger.level = original_level
    logger.propagate = original_propagate


@pytest.fixture
def canonical_rows():
    with open(FIXTURES / "canonical_bars.csv", newline="") as f:
        return list(csv.DictReader(f))

@pytest.fixture
def canonical_closes(canonical_rows):
    return [float(r["adj_close"]) for r in canonical_rows]

@pytest.fixture
def canonical_bench_closes(canonical_rows):
    return [float(r["bench_close"]) for r in canonical_rows]

@pytest.fixture
def canonical_dates(canonical_rows):
    return [r["date"] for r in canonical_rows]

@pytest.fixture
def tmp_run_dir(tmp_path):
    d = tmp_path / "runs" / "run-test"
    d.mkdir(parents=True)
    return d
