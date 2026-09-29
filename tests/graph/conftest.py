from datetime import datetime, timezone
import pytest
from tests import fakes
from tests.unit.test_tools_agent import FakeRunner, success
from tests.unit.test_grounding import headlines as good_headlines

UNIVERSE_TEXT = ("Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
                 "ACME|Acme Corp - Common Stock|Q|N|N|100|N|N\nSPY|SPDR S&P 500|G|N|N|100|Y|N\n")
NOW = datetime(2026, 9, 24, 22, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_ARTIFACTS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AGENT_LLM_CALL_DELAY_SECONDS", "0")
    from nasdaq_agent.config import Settings
    return Settings(_env_file=None)


def make_deps_factory(runner_results=None, gainer_sources=None, news_source=None, news_sources=None, judge=None):
    from nasdaq_agent.agent.tools.common import Deps
    from nasdaq_agent.email.transports import FileTransport
    from nasdaq_agent.report.schemas import JudgeVerdict
    from nasdaq_agent.universe import Universe
    hist = fakes.FakeHistorySource(data={"ACME": fakes.series("ACME", fakes.CLOSES), "SPY": fakes.series("SPY", fakes.BENCH)})

    def factory(settings, ctx, run_dir, clock, cassette=None):
        # news_sources (plural) lets a test supply an ordered chain, e.g. a named source that fails
        # followed by one that succeeds; news_source (singular) is the common one-source shortcut.
        return Deps(settings=settings, universe=Universe.from_text(UNIVERSE_TEXT),
                    gainer_sources=gainer_sources or [fakes.FakeGainerSource("massive", [fakes.acme_candidate()])],
                    history_sources=[hist], news_sources=news_sources or [news_source or fakes.FakeNewsSource(headlines=good_headlines())],
                    runner=FakeRunner(runner_results or [success()]), transport=FileTransport(run_dir.outbox_dir),
                    # Every judge lambda takes the extra_facts and session_facts keywords, matching
                    # compose_report's real call site (compose.py calls deps.judge(verified, headlines,
                    # narrative, extra_facts=extra_facts, session_facts=session_facts)).
                    judge=judge or (lambda v, h, n, extra_facts=None, session_facts=None: JudgeVerdict(faithful=True)), clock=clock, run_dir=run_dir)
    return factory
