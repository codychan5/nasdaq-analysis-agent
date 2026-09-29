import json
import pytest
from tests import fakes
from tests.unit.test_tools_data import deps, ctx, universe  # noqa: F401
from nasdaq_agent.sandbox.contract import RunResult

GOOD = {"daily_changes_pct": [2.0, -2.45098, 6.532663, 0.943396, 3.738318], "avg_daily_change_pct": 2.152679,
        "cumulative_return_pct": 11.0, "volatility_annualized_pct": 52.874, "max_drawdown_pct": -2.45098,
        "relative_vs_spy_pct": 10.2, "trend": "uptrend"}

class FakeRunner:
    name = "fake"
    def __init__(self, results): self.results = list(results); self.codes = []
    def run(self, code, sandbox_dir):
        self.codes.append(code)
        return self.results.pop(0)

def success(**over):
    from nasdaq_agent.metrics import AnalysisResult
    return RunResult(exit_code=0, backend="fake", wall_time_s=0.1, result=AnalysisResult(**{**GOOD, **over}))

def failure():
    return RunResult(exit_code=1, backend="fake", wall_time_s=0.1, stderr_tail="KeyError: 'adj_close_typo'", error="code exited with an error")

@pytest.fixture
def ready(ctx, deps):
    from nasdaq_agent.agent.tools.session import make_resolve_session
    from nasdaq_agent.agent.tools.gainer import make_find_top_gainer
    from nasdaq_agent.agent.tools.history import make_get_price_history
    from nasdaq_agent.agent.tools.news import make_get_news
    from nasdaq_agent.email.transports import FileTransport
    from tests.unit.test_grounding import headlines
    deps.news_sources = [fakes.FakeNewsSource(headlines=headlines())]
    deps.transport = FileTransport(deps.run_dir.outbox_dir)
    deps.judge = lambda v, h, n, extra_facts=None, session_facts=None: __import__("nasdaq_agent.report.schemas", fromlist=["JudgeVerdict"]).JudgeVerdict(faithful=True)
    for t in (make_resolve_session, make_find_top_gainer, make_get_price_history, make_get_news):
        t(ctx, deps).invoke({} if t is make_resolve_session else {"source": "auto"})
    return ctx

def test_run_python_records_attempts_and_result(ready, deps):
    from nasdaq_agent.agent.tools.code import make_run_python
    deps.runner = FakeRunner([failure(), success()])
    tool = make_run_python(ready, deps)
    first = tool.invoke({"code": "closes = df['adj_close_typo']"})
    assert first.startswith("ERROR") and "KeyError" in first and "remaining attempts: 5" in first
    second = json.loads(tool.invoke({"code": "result = {}"}))
    assert second["attempt"] == 2 and ready.progress.successful_run and (deps.run_dir.attempts_dir / "attempt_1.py").exists()

def test_run_python_cap_makes_analysis_terminal(ready, deps):
    from nasdaq_agent.agent.tools.code import make_run_python
    deps.runner = FakeRunner([failure()] * 7)
    tool = make_run_python(ready, deps)
    for _ in range(6):
        tool.invoke({"code": "x"})
    assert ready.progress.analysis_terminal and tool.invoke({"code": "x"}).startswith("ERROR: analysis is terminal")

def test_verify_pass_and_reject(ready, deps):
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.agent.tools.verify import make_verify_analysis
    deps.runner = FakeRunner([success(volatility_annualized_pct=47.30), success()])
    run, verify = make_run_python(ready, deps), make_verify_analysis(ready, deps)
    run.invoke({"code": "a"})
    critique = verify.invoke({})
    assert critique.startswith("ERROR") and "volatility_annualized_pct" in critique and "SAMPLE standard deviation" in critique
    assert ready.analysis.rejections == 1 and not ready.progress.verified
    run.invoke({"code": "b"})
    out = json.loads(verify.invoke({}))
    assert out["verified"] and ready.progress.verified and abs(ready.analysis.verified_result.volatility_annualized_pct - 52.874) < 0.01

def test_verify_third_rejection_is_terminal(ready, deps):
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.agent.tools.verify import make_verify_analysis
    deps.runner = FakeRunner([success(cumulative_return_pct=12.0)] * 3)
    run, verify = make_run_python(ready, deps), make_verify_analysis(ready, deps)
    for _ in range(3):
        run.invoke({"code": "a"}); msg = verify.invoke({})
    assert ready.progress.analysis_terminal and "terminal" in msg

def test_verify_returns_stored_result_without_recomputing_once_verified(ready, deps):
    # A finished stage must close. Once verified, verify_analysis must hand back the stored
    # result every time, never recompute and never rewrite verification.json -- proven here by
    # deleting verification.json and checking it stays gone.
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.agent.tools.verify import make_verify_analysis
    deps.runner = FakeRunner([success()])
    make_run_python(ready, deps).invoke({"code": "a"})
    first = json.loads(make_verify_analysis(ready, deps).invoke({}))
    assert first["verified"]
    verification_path = deps.run_dir.path / "verification.json"
    assert verification_path.exists()
    verification_path.unlink()
    second = json.loads(make_verify_analysis(ready, deps).invoke({}))
    assert second["verified"] and second["metrics"] == first["metrics"]
    assert not verification_path.exists()

def test_verify_refuses_repeat_call_without_new_attempt(ready, deps):
    # Calling verify_analysis twice with no new run_python call in between must refuse the
    # second call and must NOT spend another rejection.
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.agent.tools.verify import make_verify_analysis
    deps.runner = FakeRunner([success(cumulative_return_pct=12.0)])
    run, verify = make_run_python(ready, deps), make_verify_analysis(ready, deps)
    run.invoke({"code": "a"})
    first = verify.invoke({})
    assert first.startswith("ERROR") and ready.analysis.rejections == 1
    second = verify.invoke({})
    assert second.startswith("ERROR") and "already judged" in second
    assert ready.analysis.rejections == 1

def test_verify_terminal_when_attempts_reach_run_cap(ready, deps):
    # Reaching the run cap must end the analysis even when the rejection count alone would not
    # (here: 2 rejections, cap also lowered to 2 attempts).
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.agent.tools.verify import make_verify_analysis
    deps.settings = deps.settings.model_copy(update={"max_code_runs": 2})
    deps.runner = FakeRunner([success(cumulative_return_pct=12.0), success(cumulative_return_pct=13.0)])
    run, verify = make_run_python(ready, deps), make_verify_analysis(ready, deps)
    run.invoke({"code": "a"})
    msg1 = verify.invoke({})
    assert msg1.startswith("ERROR") and not ready.progress.analysis_terminal
    run.invoke({"code": "b"})
    msg2 = verify.invoke({})
    assert ready.progress.analysis_terminal and "give_up" in msg2 and "cap" in msg2
    assert ready.analysis.rejections == 2  # terminal came from the cap, not from rejection count

def test_run_python_refuses_once_verified(ready, deps):
    # A finished stage must close. Once verify_analysis has passed, run_python must refuse
    # rather than spend more attempts on numbers the report will never use (compose_report only
    # ever cites the verified attempt).
    from nasdaq_agent.agent.tools.code import make_run_python
    _verified(ready, deps)
    assert len(ready.analysis.attempts) == 1
    msg = make_run_python(ready, deps).invoke({"code": "z"})
    assert msg.startswith("ERROR") and "precondition" in msg
    assert len(ready.analysis.attempts) == 1

def test_record_sentiment_validates_and_cross_checks(ready, deps):
    from nasdaq_agent.agent.tools.sentiment import make_record_sentiment
    tool = make_record_sentiment(ready, deps)
    bad = tool.invoke({"score": 2.0, "label": "positive", "rationale": "x", "per_headline": []})
    assert bad.startswith("ERROR")
    out = json.loads(tool.invoke({"score": 0.6, "label": "positive", "rationale": "guidance raised",
                                  "per_headline": [{"headline_id": 1, "score": 0.5}, {"headline_id": 2, "score": 0.8}]}))
    assert out["recorded"] and ready.progress.sentiment_recorded and ready.sentiment.label == "positive"

def test_record_sentiment_refuses_when_no_headlines_were_fetched(ready, deps):
    # A live run found no headlines and the model recorded a sentiment anyway, scored from nothing. With no headlines
    # there is nothing to assess, so the tool refuses, records nothing, and nothing reaches the report.
    from nasdaq_agent.agent.tools.sentiment import make_record_sentiment
    ready.news = ready.news.model_copy(update={"headlines": []})
    out = make_record_sentiment(ready, deps).invoke({"score": 0.0, "label": "neutral", "rationale": "no news",
                                                      "per_headline": []})
    assert out.startswith("ERROR") and "no headlines" in out
    assert ready.sentiment is None and not ready.progress.sentiment_recorded
    log = [json.loads(line) for line in (deps.run_dir.path / "tool_log.jsonl").read_text().splitlines()]
    assert log[-1]["tool"] == "record_sentiment" and log[-1]["outcome"].startswith("ERROR")

def test_record_sentiment_rejects_label_inconsistent_with_score(ready, deps):
    # Positive requires score > 0, negative requires score < 0, neutral requires abs(score) <= 0.25.
    from nasdaq_agent.agent.tools.sentiment import make_record_sentiment
    tool = make_record_sentiment(ready, deps)
    msg = tool.invoke({"score": 0.6, "label": "negative", "rationale": "x", "per_headline": []})
    assert msg.startswith("ERROR") and "inconsistent" in msg
    assert not ready.progress.sentiment_recorded
    msg2 = tool.invoke({"score": -0.6, "label": "positive", "rationale": "x", "per_headline": []})
    assert msg2.startswith("ERROR") and "inconsistent" in msg2
    msg3 = tool.invoke({"score": 0.4, "label": "neutral", "rationale": "x", "per_headline": []})
    assert msg3.startswith("ERROR") and "inconsistent" in msg3

def test_record_sentiment_rejects_duplicate_headline_ids(ready, deps):
    # Duplicate headline ids in per_headline are rejected, and the rejection is recorded in the tool log.
    from nasdaq_agent.agent.tools.sentiment import make_record_sentiment
    tool = make_record_sentiment(ready, deps)
    msg = tool.invoke({"score": 0.5, "label": "positive", "rationale": "x",
                       "per_headline": [{"headline_id": 1, "score": 0.5}, {"headline_id": 1, "score": 0.6}]})
    assert msg.startswith("ERROR") and "duplicate" in msg
    assert not ready.progress.sentiment_recorded
    entries = [json.loads(line) for line in (deps.run_dir.path / "tool_log.jsonl").read_text().splitlines()]
    assert any(e["tool"] == "record_sentiment" and "duplicate" in e["outcome"] for e in entries)

def _verified(ready, deps):
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.agent.tools.verify import make_verify_analysis
    deps.runner = FakeRunner([success()])
    make_run_python(ready, deps).invoke({"code": "a"}); make_verify_analysis(ready, deps).invoke({})

def _good_args():
    from tests.unit.test_grounding import good_narrative
    return good_narrative().model_dump()

def test_compose_passes_and_renders(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    out = json.loads(make_compose_report(ready, deps).invoke(_good_args()))
    assert out["composed"] and ready.progress.composed and not ready.report.template_prose
    assert (deps.run_dir.path / "chart.png").exists() and (deps.run_dir.path / "report.html").exists()
    assert "11.00%" in (deps.run_dir.path / "report.html").read_text()

def test_compose_report_labels_sentiment_as_model_assessed(ready, deps):
    # Both templates must label the sentiment as model-assessed, not imply it is a fact about
    # the world -- checked end to end through the real render path.
    from nasdaq_agent.agent.tools.sentiment import make_record_sentiment
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    make_record_sentiment(ready, deps).invoke({"score": 0.6, "label": "positive", "rationale": "guidance raised", "per_headline": []})
    make_compose_report(ready, deps).invoke(_good_args())
    html = (deps.run_dir.path / "report.html").read_text()
    text = (deps.run_dir.path / "report.txt").read_text()
    assert "model-assessed sentiment: positive, 0.60" in html
    assert "model-assessed sentiment: positive, 0.60" in text

def test_compose_uses_verified_attempt_not_latest_attempt(ready, deps):
    # The report must cite the attempt verify_analysis actually verified, not attempts[-1] --
    # which can be a later, failed attempt whose hash has nothing to do with the numbers in the
    # email.
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.agent.tools.verify import make_verify_analysis
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.artifacts import sha256_text
    deps.runner = FakeRunner([success(), failure()])
    run, verify = make_run_python(ready, deps), make_verify_analysis(ready, deps)
    run.invoke({"code": "good"})
    run.invoke({"code": "bad"})
    out = json.loads(verify.invoke({}))
    assert out["verified"] and ready.analysis.verified_attempt == 1
    make_compose_report(ready, deps).invoke(_good_args())
    html = (deps.run_dir.path / "report.html").read_text()
    assert sha256_text("good") in html
    assert sha256_text("bad") not in html

def test_compose_rejects_then_falls_back_to_template(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    bad = {**_good_args(), "declared_metrics": [{"name": "cumulative_return_pct", "value": 12.0}]}
    tool = make_compose_report(ready, deps)
    first = tool.invoke(bad)
    assert first.startswith("ERROR") and "verified value is 11.0" in first and ready.compose_rewrites == 1
    tool.invoke(bad)
    out = json.loads(tool.invoke(bad))
    assert out["composed"] and out["template_prose"] and ready.report.template_prose

def test_compose_judge_rejection_and_outage(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.report.schemas import JudgeVerdict, JudgeIssue
    _verified(ready, deps)
    deps.judge = lambda v, h, n, extra_facts=None, session_facts=None: JudgeVerdict(faithful=False, issues=[JudgeIssue(sentence="s", problem="overstates headline 1")])
    msg = make_compose_report(ready, deps).invoke(_good_args())
    assert msg.startswith("ERROR") and "overstates headline 1" in msg
    def broken(v, h, n, extra_facts=None, session_facts=None): raise RuntimeError("judge down")
    deps.judge = broken
    out = json.loads(make_compose_report(ready, deps).invoke(_good_args()))
    # A judge outage is a decoration, so it degrades the run (exit 2 via finalize), recorded in
    # ctx.degradations (not ctx.notes), with the deterministic-layer wording.
    assert out["composed"]
    assert "judge unavailable: prose checked by the deterministic layer only" in ready.degradations
    assert not any("judge unavailable" in n for n in ready.notes)

def test_compose_rewrites_when_the_judge_says_unfaithful_but_names_no_issue(ready, deps):
    # A live run: the judge returned faithful=False with an empty issue list, and because a rewrite was requested only
    # for listed issues, the prose ("only two of five days were positive", when three were) went out as if checked.
    # An unfaithful verdict must always cost a rewrite, with a general critique when no sentence is named.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.report.schemas import JudgeVerdict
    _verified(ready, deps)
    deps.judge = lambda v, h, n, extra_facts=None, session_facts=None: JudgeVerdict(faithful=False, issues=[])
    msg = make_compose_report(ready, deps).invoke(_good_args())
    assert msg.startswith("ERROR") and "judge" in msg and "not faithful" in msg
    assert ready.compose_rewrites == 1 and not ready.progress.composed


def test_compose_falls_back_to_template_prose_when_the_judge_keeps_saying_unfaithful(ready, deps):
    from nasdaq_agent.agent.tools.compose import MAX_REWRITES, make_compose_report
    from nasdaq_agent.report.schemas import JudgeVerdict
    _verified(ready, deps)
    deps.judge = lambda v, h, n, extra_facts=None, session_facts=None: JudgeVerdict(faithful=False, issues=[])
    tool = make_compose_report(ready, deps)
    for _ in range(MAX_REWRITES):
        assert tool.invoke(_good_args()).startswith("ERROR")
    out = json.loads(tool.invoke(_good_args()))
    assert out["composed"] and out["template_prose"] and ready.progress.composed


def test_compose_tool_tells_the_model_the_paragraph_limit(ready, deps):
    # A live run's first draft was rejected for length because the tool never stated the limit.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    assert "800 characters" in make_compose_report(ready, deps).description


def test_compose_tool_tells_the_model_the_headline_note_rules(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.report.schemas import HEADLINE_NOTE_MAX_CHARS
    description = " ".join(make_compose_report(ready, deps).description.split())  # line wrapping is not content
    assert f"at most {HEADLINE_NOTE_MAX_CHARS} characters" in description
    assert "one note per story" in description and "most recent first" in description
    assert "60 to 100 words" in description and "plain English" in description


def test_compose_tool_and_system_prompt_tell_the_model_that_numbers_may_be_rounded(ready, deps):
    # A live run's model spent most of a reply's thinking on whether "must appear in declared_metrics with its verified
    # value" allowed 309.65 for 309.645, and was cut off before any tool call. The grounding check accepts a number
    # rounded to the decimals it shows, and the model is now told so.
    from nasdaq_agent.agent.orchestrator import render_system_prompt
    from nasdaq_agent.agent.tools.compose import make_compose_report
    description = " ".join(make_compose_report(ready, deps).description.split())
    prompt = " ".join(render_system_prompt().split())
    assert "rounded to the decimals" in description and "rounded to the decimals" in prompt


def test_compose_asks_for_a_rewrite_when_a_headline_has_no_note(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from tests.unit.test_grounding import good_notes
    _verified(ready, deps)
    msg = make_compose_report(ready, deps).invoke({**_good_args(), "headline_notes": [n.model_dump() for n in good_notes()[:2]]})
    assert msg.startswith("ERROR") and "headline 3 has no headline note" in msg and not ready.progress.composed


def test_judge_prompt_requires_a_named_sentence_for_an_unfaithful_verdict():
    from nasdaq_agent.report.judge import JUDGE_PROMPT
    assert "at least one issue" in JUDGE_PROMPT


def test_compose_calls_judge_with_extra_facts_by_keyword(ready, deps):
    # A recording fake, not just a faithful/unfaithful stub, so a signature mismatch in
    # deps.judge (e.g. positional-only, no extra_facts) shows up as a missed call here instead of
    # being silently swallowed as "judge unavailable".
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.report.schemas import JudgeVerdict
    _verified(ready, deps)
    calls = []
    def recording_judge(v, h, n, extra_facts=None, session_facts=None):
        calls.append({"verified": v, "headlines": h, "narrative": n, "extra_facts": extra_facts, "session_facts": session_facts})
        return JudgeVerdict(faithful=True)
    deps.judge = recording_judge
    out = json.loads(make_compose_report(ready, deps).invoke(_good_args()))
    assert out["composed"]
    # A faithful judge that was actually called adds no judge degradation or note.
    assert not any("judge unavailable" in d for d in ready.degradations)
    assert not any("judge unavailable" in n for n in ready.notes)
    assert len(calls) == 1
    assert calls[0]["extra_facts"] == {"session_pct_change": ready.gainer.pct_change, "prev_close": ready.gainer.prev_close,
                                       "close": ready.gainer.close}
    # The judge also needs the session itself: without the symbol, date and top-gainer status, it flagged sentences
    # that mention them as unsupported additions, and every live run paid a rewrite for it.
    assert calls[0]["session_facts"] == {"symbol": ready.gainer.symbol, "company": ready.gainer.company,
                                         "session_date": ready.session.date, "session_label": ready.session.label,
                                         "role": "top NASDAQ gainer of the session"}

def test_compose_allows_session_and_price_declarations(ready, deps):
    # extra_facts={"session_pct_change", "prev_close", "close"} makes those three names
    # declarable alongside the metric names, so the narrative may quote the session's own
    # close-to-close move and prices, not only the five-day verified metrics.
    from tests.unit.test_grounding import good_notes
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.report.schemas import DeclaredMetric, Narrative
    _verified(ready, deps)
    narrative = Narrative(
        trend_paragraph=(f"Acme closed at {ready.gainer.close:.2f} after rising {ready.gainer.pct_change:.2f}% on the "
                         "session, and returned 11.00% over the five-day window."),
        news_paragraph="",
        declared_metrics=[DeclaredMetric(name="close", value=ready.gainer.close),
                          DeclaredMetric(name="session_pct_change", value=ready.gainer.pct_change),
                          DeclaredMetric(name="cumulative_return_pct", value=11.0)],
        citations=[], headline_notes=good_notes())
    out = json.loads(make_compose_report(ready, deps).invoke(narrative.model_dump()))
    assert out["composed"] and not out["template_prose"]

def test_compose_refuses_once_sent(ready, deps):
    # A finished stage must close. Once send_email has succeeded, compose_report must refuse
    # rather than silently redo work nobody will read.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.agent.tools.send import make_send_email
    _verified(ready, deps)
    make_compose_report(ready, deps).invoke(_good_args())
    make_send_email(ready, deps).invoke({})
    assert ready.progress.sent
    msg = make_compose_report(ready, deps).invoke(_good_args())
    assert msg.startswith("ERROR") and "precondition" in msg

def test_compose_flags_uncited_news_when_headlines_empty(ctx, deps):
    # With no headlines to cite, a non-empty news_paragraph is itself a finding --
    # check_grounding's own citation loop is skipped entirely when there are no headlines, so
    # nothing else would ever catch uncited text in this situation.
    from nasdaq_agent.agent.tools.session import make_resolve_session
    from nasdaq_agent.agent.tools.gainer import make_find_top_gainer
    from nasdaq_agent.agent.tools.history import make_get_price_history
    from nasdaq_agent.agent.tools.news import make_get_news
    from nasdaq_agent.agent.tools.compose import make_compose_report
    deps.news_sources = [fakes.FakeNewsSource(headlines=[])]
    make_resolve_session(ctx, deps).invoke({})
    make_find_top_gainer(ctx, deps).invoke({"source": "auto"})
    make_get_price_history(ctx, deps).invoke({"source": "auto"})
    make_get_news(ctx, deps).invoke({"source": "auto"})
    _verified(ctx, deps)
    args = {**_good_args(), "news_paragraph": "Something happened with no citation.", "citations": [], "headline_notes": []}
    msg = make_compose_report(ctx, deps).invoke(args)
    assert msg.startswith("ERROR") and "no headlines were fetched" in msg

def test_compose_adds_news_not_fetched_degradation_once(ctx, deps):
    # When get_news was never called at all, add the degradation once -- before rendering -- not
    # once per compose_report call (a rewrite loop must not pile up duplicate copies of the same
    # note).
    from nasdaq_agent.agent.tools.session import make_resolve_session
    from nasdaq_agent.agent.tools.gainer import make_find_top_gainer
    from nasdaq_agent.agent.tools.history import make_get_price_history
    from nasdaq_agent.agent.tools.compose import make_compose_report
    make_resolve_session(ctx, deps).invoke({})
    make_find_top_gainer(ctx, deps).invoke({"source": "auto"})
    make_get_price_history(ctx, deps).invoke({"source": "auto"})
    _verified(ctx, deps)
    tool = make_compose_report(ctx, deps)
    bad = {**_good_args(), "declared_metrics": []}
    first = tool.invoke(bad)
    assert first.startswith("ERROR")
    good = {**_good_args(), "news_paragraph": "", "citations": [], "headline_notes": []}
    out = json.loads(tool.invoke(good))
    assert out["composed"]
    assert ctx.degradations.count("news was not fetched") == 1

def test_compose_defers_news_not_fetched_degradation_to_render_time(ctx, deps):
    # The degradation is decided inside _finish, at render time -- a rejected call that never
    # renders must not add it prematurely.
    from nasdaq_agent.agent.tools.session import make_resolve_session
    from nasdaq_agent.agent.tools.gainer import make_find_top_gainer
    from nasdaq_agent.agent.tools.history import make_get_price_history
    from nasdaq_agent.agent.tools.compose import make_compose_report
    make_resolve_session(ctx, deps).invoke({})
    make_find_top_gainer(ctx, deps).invoke({"source": "auto"})
    make_get_price_history(ctx, deps).invoke({"source": "auto"})
    _verified(ctx, deps)
    bad = {**_good_args(), "declared_metrics": []}
    rejected = make_compose_report(ctx, deps).invoke(bad)
    assert rejected.startswith("ERROR")
    assert "news was not fetched" not in ctx.degradations

def test_compose_no_news_degradation_after_failed_then_successful_source(ctx, deps):
    # get_news's auto chain fails one named source and succeeds on the next. _finish derives the
    # news degradation from the final state, so no stale "news unavailable from every source"
    # survives once headlines are present.
    from nasdaq_agent.agent.tools.session import make_resolve_session
    from nasdaq_agent.agent.tools.gainer import make_find_top_gainer
    from nasdaq_agent.agent.tools.history import make_get_price_history
    from nasdaq_agent.agent.tools.news import make_get_news
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from tests.unit.test_grounding import headlines
    deps.news_sources = [fakes.FakeNewsSource("massive", error="HTTP 500"), fakes.FakeNewsSource("yfinance", headlines=headlines())]
    make_resolve_session(ctx, deps).invoke({})
    make_find_top_gainer(ctx, deps).invoke({"source": "auto"})
    make_get_price_history(ctx, deps).invoke({"source": "auto"})
    make_get_news(ctx, deps).invoke({"source": "auto"})
    assert ctx.news is not None and [a.ok for a in ctx.news_sources_tried] == [False, True]
    _verified(ctx, deps)
    out = json.loads(make_compose_report(ctx, deps).invoke(_good_args()))
    assert out["composed"]
    assert not any("news unavailable" in d for d in ctx.degradations)
    assert "news was not fetched" not in ctx.degradations

def test_compose_names_a_news_source_that_failed_while_another_answered(ctx, deps):
    # News is gathered from every source, so a source that fails leaves the report with fewer sources than usual. A
    # warning says which, by the sources' names, without degrading the run: the news itself is present.
    from nasdaq_agent.agent.tools.session import make_resolve_session
    from nasdaq_agent.agent.tools.gainer import make_find_top_gainer
    from nasdaq_agent.agent.tools.history import make_get_price_history
    from nasdaq_agent.agent.tools.news import make_get_news
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from tests.unit.test_grounding import headlines
    deps.news_sources = [fakes.FakeNewsSource("massive", error="HTTP 500"), fakes.FakeNewsSource("yfinance", headlines=headlines())]
    make_resolve_session(ctx, deps).invoke({})
    make_find_top_gainer(ctx, deps).invoke({"source": "auto"})
    make_get_price_history(ctx, deps).invoke({"source": "auto"})
    make_get_news(ctx, deps).invoke({"source": "auto"})
    _verified(ctx, deps)
    out = json.loads(make_compose_report(ctx, deps).invoke(_good_args()))
    assert out["composed"]
    text = (deps.run_dir.path / "report.txt").read_text()
    assert "  - Warning: News from Massive was unavailable this run, so the headlines come from Yahoo Finance only.\n" in text
    assert not any("news" in d for d in ctx.degradations)

def test_two_failed_news_sources_are_named_as_a_plain_list(ctx, deps):
    from nasdaq_agent.agent.tools.session import make_resolve_session
    from nasdaq_agent.agent.tools.gainer import make_find_top_gainer
    from nasdaq_agent.agent.tools.history import make_get_price_history
    from nasdaq_agent.agent.tools.news import make_get_news
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from tests.unit.test_grounding import headlines
    deps.news_sources = [fakes.FakeNewsSource("massive", error="HTTP 500"), fakes.FakeNewsSource("sec", error="HTTP 503"),
                         fakes.FakeNewsSource("yfinance", headlines=headlines())]
    make_resolve_session(ctx, deps).invoke({})
    make_find_top_gainer(ctx, deps).invoke({"source": "auto"})
    make_get_price_history(ctx, deps).invoke({"source": "auto"})
    make_get_news(ctx, deps).invoke({"source": "auto"})
    _verified(ctx, deps)
    make_compose_report(ctx, deps).invoke(_good_args())
    text = (deps.run_dir.path / "report.txt").read_text()
    assert ("  - Warning: News from Massive and SEC EDGAR was unavailable this run, so the headlines come from "
            "Yahoo Finance only.\n") in text

def test_compose_sentiment_unavailable_when_headlines_present_without_sentiment(ready, deps):
    # Headlines were fetched (the `ready` fixture) but no sentiment was recorded, so the report
    # degrades with a "sentiment unavailable" note.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    out = json.loads(make_compose_report(ready, deps).invoke(_good_args()))
    assert out["composed"] and "sentiment unavailable" in ready.degradations

def test_compose_no_sentiment_degradation_without_headlines(ctx, deps):
    # A missing sentiment is only a decoration when there were headlines to assess.
    from nasdaq_agent.agent.tools.session import make_resolve_session
    from nasdaq_agent.agent.tools.gainer import make_find_top_gainer
    from nasdaq_agent.agent.tools.history import make_get_price_history
    from nasdaq_agent.agent.tools.compose import make_compose_report
    make_resolve_session(ctx, deps).invoke({})
    make_find_top_gainer(ctx, deps).invoke({"source": "auto"})
    make_get_price_history(ctx, deps).invoke({"source": "auto"})
    _verified(ctx, deps)
    good = {**_good_args(), "news_paragraph": "", "citations": [], "headline_notes": []}
    out = json.loads(make_compose_report(ctx, deps).invoke(good))
    assert out["composed"] and "sentiment unavailable" not in ctx.degradations

def test_compose_chart_error_degrades_without_crashing(ready, deps, monkeypatch, caplog):
    # A chart error must never crash compose. The report renders and the email builds with no
    # inline image, chart_path is None, and "chart unavailable" is recorded so finalize exits 2.
    from nasdaq_agent.agent.tools import compose as compose_mod
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.agent.tools.send import make_send_email
    _verified(ready, deps)
    def boom(*a, **k):
        raise RuntimeError("matplotlib exploded")
    monkeypatch.setattr(compose_mod, "render_chart", boom)
    with caplog.at_level("WARNING"):
        out = json.loads(make_compose_report(ready, deps).invoke(_good_args()))
    assert out["composed"] and ready.report.chart_path is None
    assert "chart unavailable" in ready.degradations
    html = (deps.run_dir.path / "report.html").read_text()
    assert "cid:chart@" not in html  # no inline image reference in the HTML body
    sent = json.loads(make_send_email(ready, deps).invoke({}))  # the email still builds and sends
    assert sent["sent"]

def test_compose_refuses_once_composed(ready, deps):
    # A finished stage must close. Once composed, compose_report refuses (before any rewrite is
    # counted), in the same style as the other tools' precondition guards.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    tool = make_compose_report(ready, deps)
    first = json.loads(tool.invoke(_good_args()))
    assert first["composed"] and ready.progress.composed
    rewrites_after_success = ready.compose_rewrites
    second = tool.invoke(_good_args())
    assert second.startswith("ERROR") and "already been composed" in second
    assert ready.compose_rewrites == rewrites_after_success  # the refusal did not count a rewrite

def test_compose_rewrite_count_wording_is_grammatical(ready, deps):
    # "2 rewrites left" then "1 rewrite left" -- never "1 rewrites left".
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    bad = {**_good_args(), "declared_metrics": [{"name": "cumulative_return_pct", "value": 12.0}]}
    tool = make_compose_report(ready, deps)
    first = tool.invoke(bad)
    assert first.startswith("ERROR") and "2 rewrites left" in first
    second = tool.invoke(bad)
    assert second.startswith("ERROR") and "1 rewrite left" in second and "1 rewrites left" not in second

def test_compose_exempts_session_year_from_grounding(ready, deps):
    # compose_report passes exempt_years={session_year, session_year - 1} from
    # ctx.session.date, so the model may write about the session's own year (and the year
    # before it) without declaring it as a number.
    from tests.unit.test_grounding import good_notes
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.report.schemas import Narrative, DeclaredMetric
    _verified(ready, deps)
    assert ready.session.date == "2026-09-24"  # session_year 2026, exempt_years {2026, 2025}
    narrative = Narrative(trend_paragraph="Acme extended a rally that began earlier in 2026, returning 11.00% over five days.",
                          news_paragraph="", declared_metrics=[DeclaredMetric(name="cumulative_return_pct", value=11.0)], citations=[],
                          headline_notes=good_notes())
    out = json.loads(make_compose_report(ready, deps).invoke(narrative.model_dump()))
    assert out["composed"]

def test_compose_does_not_exempt_an_unrelated_year(ready, deps):
    from tests.unit.test_grounding import good_notes
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.report.schemas import Narrative
    _verified(ready, deps)
    narrative = Narrative(trend_paragraph="Acme has traded publicly since 1999.", news_paragraph="", declared_metrics=[], citations=[],
                          headline_notes=good_notes())
    msg = make_compose_report(ready, deps).invoke(narrative.model_dump())
    assert msg.startswith("ERROR")

def test_compose_allows_previous_year_reference(ready, deps):
    # exempt_years={session_year, session_year - 1} must cover the previous year too, not only
    # the session's own year -- session date is 2026-09-24, so "began in 2025" must be
    # composed, not rejected.
    from tests.unit.test_grounding import good_notes
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.report.schemas import Narrative, DeclaredMetric
    _verified(ready, deps)
    assert ready.session.date == "2026-09-24"
    narrative = Narrative(trend_paragraph="Acme's rally began in 2025 and returned 11.00% over five days.",
                          news_paragraph="", declared_metrics=[DeclaredMetric(name="cumulative_return_pct", value=11.0)], citations=[],
                          headline_notes=good_notes())
    out = json.loads(make_compose_report(ready, deps).invoke(narrative.model_dump()))
    assert out["composed"]

def test_send_is_idempotent_and_recipient_is_config(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.agent.tools.send import make_send_email
    from nasdaq_agent.email.idempotency import SendMarker
    import email as email_lib
    _verified(ready, deps); make_compose_report(ready, deps).invoke(_good_args())
    tool = make_send_email(ready, deps)
    out = json.loads(tool.invoke({}))
    assert out["sent"] and ready.progress.sent and SendMarker(deps.run_dir.path / "sent.json").state == "sent"
    eml = next(deps.run_dir.outbox_dir.glob("*.eml"))
    assert email_lib.message_from_bytes(eml.read_bytes())["To"] == "r@example.com"
    assert "already_sent" in tool.invoke({})

def test_send_refuses_when_marker_is_sending(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.agent.tools.send import make_send_email
    from nasdaq_agent.email.idempotency import SendMarker
    _verified(ready, deps); make_compose_report(ready, deps).invoke(_good_args())
    SendMarker(deps.run_dir.path / "sent.json").begin()
    msg = make_send_email(ready, deps).invoke({})
    assert msg.startswith("ERROR") and "unknown" in msg and not list(deps.run_dir.outbox_dir.glob("*.eml"))

def test_send_writes_sent_eml_copy_after_success(ready, deps):
    # A fixed-name copy of exactly what was delivered lives in the run directory itself,
    # independent of the transport (SmtpTransport writes nothing locally; FileTransport's own
    # copy is under a timestamped name in the outbox, not this well-known path).
    import email as email_lib
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.agent.tools.send import make_send_email
    _verified(ready, deps); make_compose_report(ready, deps).invoke(_good_args())
    out = json.loads(make_send_email(ready, deps).invoke({}))
    assert out["sent"]
    sent_eml = deps.run_dir.path / "sent.eml"
    assert sent_eml.exists()
    assert email_lib.message_from_bytes(sent_eml.read_bytes())["To"] == "r@example.com"

def test_send_still_marks_sent_when_sent_eml_write_fails(ready, deps, caplog):
    # The sent.eml copy is best-effort. ctx.email / ctx.progress.sent are updated and saved
    # BEFORE attempting it, so a failure writing that convenience copy must never leave a
    # message that really was sent looking unsent -- only a logged warning.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.agent.tools.send import make_send_email
    from nasdaq_agent.email.idempotency import SendMarker
    _verified(ready, deps); make_compose_report(ready, deps).invoke(_good_args())
    (deps.run_dir.path / "sent.eml").mkdir()  # write_bytes onto a directory raises IsADirectoryError
    with caplog.at_level("WARNING"):
        out = json.loads(make_send_email(ready, deps).invoke({}))
    assert out["sent"] and ready.progress.sent
    assert SendMarker(deps.run_dir.path / "sent.json").state == "sent"
    assert any("sent.eml" in r.message for r in caplog.records)

def test_send_catches_up_progress_sent_from_marker(ready, deps):
    # A resumed run can find the marker already "sent" from an earlier process, before this
    # in-memory ctx was ever marked -- the already-sent branch must catch progress.sent up
    # rather than leaving it False for a message that really was delivered.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.agent.tools.send import make_send_email
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult
    _verified(ready, deps); make_compose_report(ready, deps).invoke(_good_args())
    marker = SendMarker(deps.run_dir.path / "sent.json")
    marker.begin()
    marker.complete(SendResult(transport="file", message_id="<abc@x>", location=None))
    assert not ready.progress.sent  # in-memory ctx was never told about this earlier send
    out = json.loads(make_send_email(ready, deps).invoke({}))
    assert out["already_sent"] and ready.progress.sent

def test_send_refuses_without_transport_before_marking_sending(ready, deps):
    # Checking deps.transport is not None must happen BEFORE marker.begin(), so a missing
    # transport can never leave a "sending" marker behind for a message that was never actually
    # attempted (SendMarker has no way back from "sending" except a send that completes or fails
    # through the transport itself).
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.agent.tools.send import make_send_email
    from nasdaq_agent.email.idempotency import SendMarker
    _verified(ready, deps); make_compose_report(ready, deps).invoke(_good_args())
    deps.transport = None
    msg = make_send_email(ready, deps).invoke({})
    assert msg.startswith("ERROR")
    assert SendMarker(deps.run_dir.path / "sent.json").state == "none"
    assert not (deps.run_dir.path / "sent.eml").exists()

def test_send_email_precondition_is_recorded(ready, deps):
    # A precondition refusal must be recorded via record_call before it is returned, as every
    # other tool does. send_email is called with nothing composed.
    from nasdaq_agent.agent.tools.send import make_send_email
    msg = make_send_email(ready, deps).invoke({})
    assert msg.startswith("ERROR") and "precondition" in msg
    entries = [json.loads(line) for line in (deps.run_dir.path / "tool_log.jsonl").read_text().splitlines()]
    assert any(e["tool"] == "send_email" for e in entries)

def test_every_call_is_recorded_exactly_once_whatever_the_outcome(ready, deps):
    # Every return path in these tools -- refusals and terminal guards included -- must go
    # through record_call exactly once, so the audit trail never silently drops a call or
    # double-records one. Drive a mixed sequence (a precondition refusal, a non-precondition
    # refusal, a success, and give_up) and check the log grew by exactly one line per call, in
    # order, naming the right tool each time.
    from nasdaq_agent.agent.tools.give_up import make_give_up
    from nasdaq_agent.agent.tools.send import make_send_email
    from nasdaq_agent.agent.tools.sentiment import make_record_sentiment

    log_path = deps.run_dir.path / "tool_log.jsonl"
    before = len(log_path.read_text().splitlines())

    send_result = make_send_email(ready, deps).invoke({})
    assert send_result.startswith("ERROR") and "precondition" in send_result

    sentiment_tool = make_record_sentiment(ready, deps)
    bad_sentiment = sentiment_tool.invoke({"score": 2.0, "label": "positive", "rationale": "x", "per_headline": []})
    assert bad_sentiment.startswith("ERROR")

    good_sentiment = sentiment_tool.invoke({"score": 0.5, "label": "positive", "rationale": "ok", "per_headline": []})
    assert json.loads(good_sentiment)["recorded"]

    gave_up = make_give_up(ready, deps).invoke({"reason": "done"})
    assert json.loads(gave_up)["stopped"]

    lines = log_path.read_text().splitlines()
    assert len(lines) == before + 4
    new_entries = [json.loads(line) for line in lines[before:]]
    assert [e["tool"] for e in new_entries] == ["send_email", "record_sentiment", "record_sentiment", "give_up"]

def test_give_up(ready, deps):
    from nasdaq_agent.agent.tools.give_up import make_give_up
    make_give_up(ready, deps).invoke({"reason": "no data"})
    assert ready.progress.gave_up and ready.give_up_reason == "no data"

def test_build_tools_order(ready, deps):
    from nasdaq_agent.agent.tools import build_tools
    names = [t.name for t in build_tools(ready, deps)]
    assert names == ["resolve_session", "find_top_gainer", "get_price_history", "run_python", "verify_analysis",
                     "get_news", "record_sentiment", "compose_report", "send_email", "give_up"]



@pytest.mark.parametrize("backend, words", [
    ("docker", "ran in an isolated sandbox (Docker)"),
    ("subprocess", "ran in a less isolated fallback sandbox (subprocess)"),
    ("fake", "ran in the fake sandbox"),  # a backend without a description keeps its own name
])
def test_the_footer_names_the_sandbox_in_plain_words(ready, deps, backend, words):
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.agent.tools.verify import make_verify_analysis
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.metrics import AnalysisResult
    deps.runner = FakeRunner([RunResult(exit_code=0, backend=backend, wall_time_s=0.1, result=AnalysisResult(**GOOD))])
    make_run_python(ready, deps).invoke({"code": "a"}); make_verify_analysis(ready, deps).invoke({})
    make_compose_report(ready, deps).invoke(_good_args())
    text = (deps.run_dir.path / "report.txt").read_text()
    footer = text.split("\nHow this report was made\n", 1)[1]
    assert f"  The analysis code was written by AI and {words}; separate code re-checked every calculated figure.\n" in footer
    assert "reviewed code" not in text

def test_the_footer_names_the_sources_by_their_display_names(ready, deps):
    # The gainer came from the fake "massive" source, which has a display name; the fake history source has none, so
    # it keeps its own.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    make_compose_report(ready, deps).invoke(_good_args())
    footer = (deps.run_dir.path / "report.txt").read_text().split("\nHow this report was made\n", 1)[1]
    assert "  Top gainer from Massive; prices from fakehist.\n" in footer

def _above_the_metrics(text):
    return text.split("\nFive-day metrics\n", 1)[0]

def _footer(text):
    return text.split("\nHow this report was made\n", 1)[1]

def test_a_price_disagreement_shows_in_the_report_and_degrades_the_run(ready, deps):
    from nasdaq_agent.agent.context import PriceCheck
    from nasdaq_agent.agent.tools.compose import make_compose_report
    detail = ("Price check: Massive disagrees with Yahoo Finance on 1 of 6 closes by more than 0.5% "
              "(2026-09-24: 11.10 vs 11.20).")
    ready.history.price_check = PriceCheck(status="disagree", source="massive", detail=detail)
    _verified(ready, deps)
    make_compose_report(ready, deps).invoke(_good_args())
    text = (deps.run_dir.path / "report.txt").read_text()
    # The disagreement is a problem, so it sits under the status line, worded by the check that found it.
    assert f"  - {detail}\n" in _above_the_metrics(text)
    assert detail not in _footer(text)
    assert ready.degradations.count("closes disagree with a second price source") == 1

def test_an_unchecked_price_shows_in_the_report_without_degrading_the_run(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)  # the fixture configures one history source, so the check could not be made
    make_compose_report(ready, deps).invoke(_good_args())
    text = (deps.run_dir.path / "report.txt").read_text()
    assert "  Prices were not cross-checked: no second price source is configured.\n" in _footer(text)
    assert not any("price" in note for note in ready.degradations)

def test_an_agreeing_price_check_is_background_in_the_footer(ready, deps):
    from nasdaq_agent.agent.context import PriceCheck
    from nasdaq_agent.agent.tools.compose import make_compose_report
    detail = "Closes cross-checked against Massive: all 6 agree within 0.5%."
    ready.history.price_check = PriceCheck(status="agree", source="massive", detail=detail)
    _verified(ready, deps)
    make_compose_report(ready, deps).invoke(_good_args())
    text = (deps.run_dir.path / "report.txt").read_text()
    assert f"  {detail}\n" in _footer(text) and detail not in _above_the_metrics(text)

def test_compose_keeps_the_headlines_of_a_news_source_asked_by_name(ctx, deps):
    """get_news for one named source leaves the others untried, so the news step stays open. The report must still use
    that source's headlines, and must not say that every news source failed."""
    from nasdaq_agent.agent.tools.session import make_resolve_session
    from nasdaq_agent.agent.tools.gainer import make_find_top_gainer
    from nasdaq_agent.agent.tools.history import make_get_price_history
    from nasdaq_agent.agent.tools.news import make_get_news
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.email.transports import FileTransport
    from nasdaq_agent.report.schemas import JudgeVerdict
    from tests.unit.test_grounding import headlines
    deps.news_sources = [fakes.FakeNewsSource("massive", headlines=headlines()), fakes.FakeNewsSource("yfinance")]
    deps.transport = FileTransport(deps.run_dir.outbox_dir)
    deps.judge = lambda v, h, n, extra_facts=None, session_facts=None: JudgeVerdict(faithful=True)
    make_resolve_session(ctx, deps).invoke({})
    for tool in (make_find_top_gainer, make_get_price_history):
        tool(ctx, deps).invoke({"source": "auto"})
    make_get_news(ctx, deps).invoke({"source": "massive"})  # yfinance is left untried
    _verified(ctx, deps)
    out = make_compose_report(ctx, deps).invoke(_good_args())
    assert not out.startswith("ERROR"), out
    assert "Acme files for FDA review of lead drug candidate" in (deps.run_dir.path / "report.txt").read_text()
    assert not any("news" in note for note in ctx.degradations)

def test_a_news_source_asked_by_name_that_had_nothing_is_not_reported_as_failed(ctx, deps):
    from nasdaq_agent.agent.tools.session import make_resolve_session
    from nasdaq_agent.agent.tools.gainer import make_find_top_gainer
    from nasdaq_agent.agent.tools.history import make_get_price_history
    from nasdaq_agent.agent.tools.news import make_get_news
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.report.schemas import JudgeVerdict
    deps.news_sources = [fakes.FakeNewsSource("massive"), fakes.FakeNewsSource("yfinance")]
    deps.judge = lambda v, h, n, extra_facts=None, session_facts=None: JudgeVerdict(faithful=True)
    make_resolve_session(ctx, deps).invoke({})
    for tool in (make_find_top_gainer, make_get_price_history):
        tool(ctx, deps).invoke({"source": "auto"})
    make_get_news(ctx, deps).invoke({"source": "massive"})  # answers with no headlines; yfinance is left untried
    _verified(ctx, deps)
    args = {**_good_args(), "news_paragraph": "", "citations": [], "headline_notes": []}
    out = make_compose_report(ctx, deps).invoke(args)
    assert not out.startswith("ERROR"), out
    assert "news unavailable from every source" not in ctx.degradations

def _record_sentiment(ctx, deps):
    from nasdaq_agent.agent.tools.sentiment import make_record_sentiment
    make_record_sentiment(ctx, deps).invoke({"score": 0.6, "label": "positive", "rationale": "guidance raised", "per_headline": []})

STATUS_ALL_GOOD = "Report status: everything worked, and every calculated figure was re-checked by separate code."
STATUS_ATTENTION = "Report status: some things need your attention."
TEMPLATE_PROSE_PROBLEM = ("The written summary is a fixed template: the AI's draft did not pass the fact check within "
                          "the allowed rewrites.")

def test_a_clean_run_says_everything_worked(ready, deps):
    # The status line follows the exit code: no degradation (exit 0) is the only way to "everything worked".
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    _record_sentiment(ready, deps)
    make_compose_report(ready, deps).invoke(_good_args())
    text = (deps.run_dir.path / "report.txt").read_text()
    assert ready.degradations == []
    assert f"\n{STATUS_ALL_GOOD}\n\nFive-day metrics\n" in text

def test_a_degraded_run_lists_its_problem_in_plain_words_under_the_status(ready, deps):
    # The fixture has headlines but no sentiment, so the run degrades (exit 2). The fixed note stays the same in
    # summary.json; the email words it for a reader.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    make_compose_report(ready, deps).invoke(_good_args())
    text = (deps.run_dir.path / "report.txt").read_text()
    assert ready.degradations == ["sentiment unavailable"]
    assert f"\n{STATUS_ATTENTION}\n  - No sentiment score: the headlines were not rated.\n\nFive-day metrics\n" in text
    assert "sentiment unavailable" not in text

def test_a_template_narrative_is_noted_before_rendering_so_the_status_says_so(ready, deps):
    # The template-narrative degradation used to be recorded after the email was rendered, so a status line built from
    # the degradations would have said "everything worked" on a run that exits 2.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    _record_sentiment(ready, deps)
    bad = {**_good_args(), "declared_metrics": [{"name": "cumulative_return_pct", "value": 12.0}]}
    tool = make_compose_report(ready, deps)
    tool.invoke(bad); tool.invoke(bad)
    assert json.loads(tool.invoke(bad))["template_prose"]
    text = (deps.run_dir.path / "report.txt").read_text()
    assert ready.degradations == ["narrative is template-generated after grounding failures"]
    assert f"\n{STATUS_ATTENTION}\n  - {TEMPLATE_PROSE_PROBLEM}\n" in text

def test_a_failed_render_leaves_no_stale_template_note_for_the_next_attempt(ready, deps, monkeypatch):
    # A render or file write that fails leaves the report uncomposed, so compose_report is offered again. When the next
    # attempt's own prose passes, the note from the failed template attempt must not stay behind and exit the run 2.
    import nasdaq_agent.agent.tools.compose as compose
    _verified(ready, deps)
    _record_sentiment(ready, deps)
    bad = {**_good_args(), "declared_metrics": [{"name": "cumulative_return_pct", "value": 12.0}]}
    tool = compose.make_compose_report(ready, deps)
    tool.invoke(bad); tool.invoke(bad)
    real_render = compose.render_report
    def disk_full(rc):
        raise OSError("disk full")
    monkeypatch.setattr(compose, "render_report", disk_full)
    with pytest.raises(OSError):
        tool.invoke(bad)  # the rewrites are used up, so this attempt falls back to the template, then fails to render
    monkeypatch.setattr(compose, "render_report", real_render)
    out = json.loads(tool.invoke(_good_args()))
    assert out["composed"] and not out["template_prose"]
    assert ready.degradations == []
    assert f"\n{STATUS_ALL_GOOD}\n" in (deps.run_dir.path / "report.txt").read_text()

def _disk_full(rc):
    raise OSError("disk full")

def test_a_failed_render_leaves_no_stale_sentiment_note_once_sentiment_is_recorded(ready, deps, monkeypatch):
    # The fixture has headlines and no sentiment, so the first attempt notes "sentiment unavailable" and then fails to
    # render. The model records sentiment before composing again, so the sent report must not say it is missing.
    import nasdaq_agent.agent.tools.compose as compose
    _verified(ready, deps)
    tool = compose.make_compose_report(ready, deps)
    real_render = compose.render_report
    monkeypatch.setattr(compose, "render_report", _disk_full)
    with pytest.raises(OSError):
        tool.invoke(_good_args())
    monkeypatch.setattr(compose, "render_report", real_render)
    _record_sentiment(ready, deps)
    assert json.loads(tool.invoke(_good_args()))["composed"]
    assert ready.degradations == []
    assert f"\n{STATUS_ALL_GOOD}\n" in (deps.run_dir.path / "report.txt").read_text()

def test_a_failed_render_leaves_no_stale_chart_note_once_the_chart_draws(ready, deps, monkeypatch):
    # The first attempt cannot draw the chart and then fails to render. The next attempt draws it, so the sent report
    # has a chart and must not say it is missing.
    import nasdaq_agent.agent.tools.compose as compose
    _verified(ready, deps)
    _record_sentiment(ready, deps)
    tool = compose.make_compose_report(ready, deps)
    real_chart, real_render = compose.render_chart, compose.render_report
    def no_backend(*args, **kwargs):
        raise RuntimeError("no matplotlib backend")
    monkeypatch.setattr(compose, "render_chart", no_backend)
    monkeypatch.setattr(compose, "render_report", _disk_full)
    with pytest.raises(OSError):
        tool.invoke(_good_args())
    assert ready.degradations == ["chart unavailable"]  # the failed attempt did note the missing chart
    monkeypatch.setattr(compose, "render_chart", real_chart)
    monkeypatch.setattr(compose, "render_report", real_render)
    assert json.loads(tool.invoke(_good_args()))["composed"]
    assert ready.degradations == [] and ready.report.chart_path is not None
    assert f"\n{STATUS_ALL_GOOD}\n" in (deps.run_dir.path / "report.txt").read_text()

@pytest.mark.parametrize("note", [  # the fixed notes summary.json records, as the README lists them
    "news was not fetched", "news unavailable from every source", "sentiment unavailable", "chart unavailable",
    "judge unavailable: prose checked by the deterministic layer only",
    "narrative is template-generated after grounding failures", "closes disagree with a second price source",
])
def test_every_fixed_degradation_note_has_plain_words_for_the_email(note):
    from nasdaq_agent.agent.tools.compose import PROBLEM_TEXTS
    words = PROBLEM_TEXTS[note]
    assert words != note and words[0].isupper() and words.endswith(".") and "_" not in words

def test_the_template_problem_names_no_rewrite_count():
    # The fixed-template written summary takes over once the allowed rewrites are used up, so its note must carry no
    # count that can go stale when MAX_REWRITES changes (it once said "twice").
    from nasdaq_agent.agent.tools.compose import PROBLEM_TEXTS
    words = PROBLEM_TEXTS["narrative is template-generated after grounding failures"]
    assert "twice" not in words and not any(ch.isdigit() for ch in words)

def test_a_degradation_note_without_plain_words_is_shown_as_it_is(ready, deps):
    # Never dropped: a note the email has no words for still reaches the reader, and still means attention.
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    _record_sentiment(ready, deps)
    ready.degradations.append("quota exhausted on a new source")
    make_compose_report(ready, deps).invoke(_good_args())
    text = (deps.run_dir.path / "report.txt").read_text()
    assert f"\n{STATUS_ATTENTION}\n  - quota exhausted on a new source\n" in text

@pytest.mark.parametrize("prev_close, warned", [(4.99, True), (5.00, False)])
def test_a_prior_close_under_five_dollars_is_a_warning_not_a_problem(ready, deps, prev_close, warned):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    _record_sentiment(ready, deps)
    ready.gainer.prev_close = prev_close
    make_compose_report(ready, deps).invoke(_good_args())
    text = (deps.run_dir.path / "report.txt").read_text()
    warning = ("  - Warning: The prior close was under $5. Stocks this cheap are often thinly traded, so treat the move "
               "with caution.\n")
    assert (warning in _above_the_metrics(text)) is warned
    assert ready.degradations == [] and f"\n{STATUS_ALL_GOOD}\n" in text

def test_an_early_close_is_a_warning_not_a_problem(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    _record_sentiment(ready, deps)
    ready.session.early_close = True
    make_compose_report(ready, deps).invoke(_good_args())
    text = (deps.run_dir.path / "report.txt").read_text()
    assert "  - Warning: The session was an early close, so the day's trading was shorter than usual.\n" in _above_the_metrics(text)
    assert ready.degradations == []
