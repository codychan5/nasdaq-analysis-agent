import json
from pathlib import Path
from langchain_core.messages import AIMessage
from tests import fakes
from tests.fake_model import scripted, tool_call as tc
from tests.graph.conftest import NOW, make_deps_factory
from tests.unit.test_tools_agent import failure, success
from tests.unit.test_grounding import good_narrative, headlines as good_headlines
from nasdaq_agent.report.schemas import JudgeVerdict, JudgeIssue

SENTIMENT = {"score": 0.6, "label": "positive", "rationale": "guidance raised",
             "per_headline": [{"headline_id": 1, "score": 0.4}, {"headline_id": 2, "score": 0.8}, {"headline_id": 3, "score": 0.5}]}
CODE = "result = {}"


def happy_script(extra_before_send=()):
    return [tc("resolve_session", {}, "c1"), tc("find_top_gainer", {"source": "auto"}, "c2"),
            tc("get_price_history", {"source": "auto"}, "c3"), tc("run_python", {"code": CODE}, "c4"),
            tc("verify_analysis", {}, "c5"), tc("get_news", {"source": "auto"}, "c6"),
            tc("record_sentiment", SENTIMENT, "c7"), tc("compose_report", good_narrative().model_dump(), "c8"),
            *extra_before_send, tc("send_email", {}, "c9"), AIMessage(content="Sent the report.")]


def run(settings, script, **factory_kw):
    from nasdaq_agent.agent.graph import run_once
    model = scripted(script)
    outcome = run_once(settings, now=NOW, model=model, deps_factory=make_deps_factory(**factory_kw))
    return outcome, model


def load_ctx(outcome):
    from nasdaq_agent.agent.context import RunContext
    return RunContext.load(Path(outcome.artifacts_path))


def emails(outcome):
    return list((Path(outcome.artifacts_path) / "outbox").glob("*.eml"))


def test_happy_path(settings):
    outcome, _ = run(settings, happy_script())
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 0 and ctx.progress.sent and len(emails(outcome)) == 1
    assert (Path(outcome.artifacts_path) / "closing_summary.txt").read_text() == "Sent the report."
    assert len((Path(outcome.artifacts_path) / "tool_log.jsonl").read_text().splitlines()) == 9


def test_model_picks_fallback_source(settings):
    script = [tc("resolve_session", {}, "c1"), tc("find_top_gainer", {"source": "massive"}, "c2a"),
              tc("find_top_gainer", {"source": "alphavantage"}, "c2b")] + happy_script()[2:]
    dead = fakes.FakeGainerSource("massive", error="timeout")
    live = fakes.FakeGainerSource("alphavantage", [fakes.acme_candidate("alphavantage")])
    outcome, _ = run(settings, script, gainer_sources=[dead, live])
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 0 and ctx.gainer.source == "alphavantage"
    assert [a.source for a in ctx.gainer_sources_tried] == ["massive", "alphavantage"]


def test_code_repair_after_traceback(settings):
    script = happy_script()[:3] + [tc("run_python", {"code": "bad"}, "c4a")] + happy_script()[3:]
    outcome, _ = run(settings, script, runner_results=[failure(), success()])
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 0 and [a.ok for a in ctx.analysis.attempts] == [False, True]


def test_verifier_rejection_then_pass(settings):
    script = happy_script()[:4] + [tc("verify_analysis", {}, "c5a"), tc("run_python", {"code": CODE}, "c4b")] + happy_script()[4:]
    outcome, _ = run(settings, script, runner_results=[success(volatility_annualized_pct=47.30), success()])
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 0 and ctx.analysis.rejections == 1 and ctx.progress.verified


def test_six_crashes_then_give_up(settings):
    script = happy_script()[:3] + [tc("run_python", {"code": "bad"}, f"c4{i}") for i in range(6)]
    script += [tc("give_up", {"reason": "cannot compute the metrics"}, "c99"), AIMessage(content="Gave up.")]
    outcome, _ = run(settings, script, runner_results=[failure()] * 6)
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 1 and ctx.progress.analysis_terminal and ctx.progress.gave_up
    body = emails(outcome)[0].read_bytes().decode()
    assert "cannot compute the metrics" in body and "failed" in body


def test_news_down_degrades(settings):
    script = happy_script()
    # Compose rejects a non-empty news paragraph when no headlines were fetched ("no headlines
    # were fetched; leave news_paragraph empty"), so the script must pass an empty
    # news_paragraph alongside the empty citations list.
    script[7] = tc("compose_report", {**good_narrative().model_dump(), "news_paragraph": "", "citations": [], "headline_notes": []}, "c8")
    script.pop(6)  # no sentiment without news
    outcome, _ = run(settings, script, news_source=fakes.FakeNewsSource(error="503"))
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 2 and ctx.progress.sent and any("news unavailable" in d for d in ctx.degradations)


def test_judge_outage_degrades_to_exit_2(settings):
    # A judge outage is a decoration -- the grounded report is still sent, and the run exits 2
    # with a degradation note (not exit 0, and not a fatal exit 1).
    def judge_down(v, h, n, extra_facts=None, session_facts=None):
        raise RuntimeError("provider down")
    outcome, _ = run(settings, happy_script(), judge=judge_down)
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 2 and ctx.progress.sent and len(emails(outcome)) == 1
    assert "judge unavailable: prose checked by the deterministic layer only" in ctx.degradations


def test_chart_error_degrades_to_exit_2_and_sends_without_a_chart(settings, monkeypatch):
    # A chart error must never crash compose. The report renders and sends without the inline
    # chart, and the run exits 2 with a "chart unavailable" degradation.
    def boom(*a, **k):
        raise RuntimeError("matplotlib down")
    monkeypatch.setattr("nasdaq_agent.agent.tools.compose.render_chart", boom)
    outcome, _ = run(settings, happy_script())
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 2 and ctx.progress.sent and len(emails(outcome)) == 1
    assert ctx.report.chart_path is None and "chart unavailable" in ctx.degradations


def test_headlines_without_sentiment_degrade_to_exit_2(settings):
    # Headlines were fetched but no sentiment was recorded, so the run notes "sentiment unavailable" and exits 2.
    script = happy_script()
    script.pop(6)  # drop record_sentiment; get_news still fetches headlines
    outcome, _ = run(settings, script)
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 2 and ctx.progress.sent
    assert "sentiment unavailable" in ctx.degradations


def test_failed_then_successful_news_source_leaves_no_degradation(settings):
    # A named news source fails, the auto chain then succeeds on the next one.
    # No stale "news unavailable from every source" must survive, so the run exits 0 with news present.
    news_sources = [fakes.FakeNewsSource("massive", error="HTTP 500"),
                    fakes.FakeNewsSource("yfinance", headlines=good_headlines())]
    outcome, _ = run(settings, happy_script(), news_sources=news_sources)
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 0 and ctx.progress.sent
    assert [a.ok for a in ctx.news_sources_tried] == [False, True]
    assert not any("news unavailable" in d for d in ctx.degradations)
    assert "news was not fetched" not in ctx.degradations


def test_judge_rejects_once(settings):
    verdicts = iter([JudgeVerdict(faithful=False, issues=[JudgeIssue(sentence="s", problem="overstates")]), JudgeVerdict(faithful=True)])
    script = happy_script(extra_before_send=[])
    script.insert(7, tc("compose_report", good_narrative().model_dump(), "c8a"))
    # The judge is called as judge(verified, headlines, narrative, extra_facts=...,
    # session_facts=...), so every judge lambda -- including this one -- must accept those keywords.
    outcome, _ = run(settings, script, judge=lambda v, h, n, extra_facts=None, session_facts=None: next(verdicts))
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 0 and ctx.compose_rewrites == 1 and ctx.report.judge_faithful is True


def test_send_before_verify_is_refused_and_not_offered(settings):
    script = happy_script()[:3] + [tc("send_email", {}, "early")] + happy_script()[3:]
    outcome, model = run(settings, script)
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 0 and len(emails(outcome)) == 1
    log = [json.loads(l) for l in (Path(outcome.artifacts_path) / "tool_log.jsonl").read_text().splitlines()]
    assert any(e["tool"] == "send_email" and "precondition" in e["outcome"] for e in log)
    assert "send_email" not in model.bound_tool_names[3]  # the fourth model turn was not offered send_email


def test_extra_numbers_passed_to_compose_never_reach_the_report(settings):
    bogus = {**good_narrative().model_dump(), "metrics_table": {"cumulative_return_pct": 99.0}}
    script = happy_script()
    script.insert(7, tc("compose_report", bogus, "c8x"))
    outcome, _ = run(settings, script)
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 0 and ctx.compose_rewrites == 0 and "99.0" not in Path(ctx.report.html_path).read_text()


def test_resume_after_send_does_not_resend(settings):
    from nasdaq_agent.agent.graph import resume_run
    outcome, _ = run(settings, happy_script())
    resumed = resume_run(settings, outcome.run_id, model=scripted([AIMessage(content="noop")]), deps_factory=make_deps_factory())
    assert resumed.exit_code == 0 and len(emails(outcome)) == 1


def test_model_ends_without_sending(settings):
    # The finish guard reminds a text-only reply MAX_NUDGES times before letting the loop end.
    from nasdaq_agent.agent.middleware import MAX_NUDGES
    script = happy_script()[:3] + [AIMessage(content=f"I am done ({i}).") for i in range(MAX_NUDGES + 1)]
    outcome, _ = run(settings, script)
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 1 and not ctx.progress.sent and len(emails(outcome)) == 1
    assert "did not produce a report" in emails(outcome)[0].read_bytes().decode()


def test_a_text_only_reply_mid_run_is_reminded_and_the_run_finishes(settings):
    # A live run: after get_news the model wrote "Let me try an alternative news source" instead of calling a tool,
    # and the run ended unsent. The guard reminds it, and the model carries on to compose and send.
    script = happy_script()
    script.insert(6, AIMessage(content="Let me try an alternative news source."))
    outcome, _ = run(settings, script)
    ctx = load_ctx(outcome)
    assert outcome.exit_code == 0 and ctx.progress.sent and len(emails(outcome)) == 1
