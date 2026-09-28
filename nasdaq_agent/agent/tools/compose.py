from datetime import date
from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

from ...report.chart import render_chart
from ...report.grounding import check_grounding
from ...report.render import ReportContext, render_report, source_label
from ...report.schemas import Citation, DeclaredMetric, HeadlineNote, Narrative
from ...sources.errors import CassetteMiss
from ..context import AttemptInfo, ReportInfo, RunContext
from .common import Deps, err, log, ok, precondition, record_call

MAX_REWRITES = 2
JUDGE_ERROR_CHARS = 300
CHART_FILENAME = "chart.png"
# Decoration degradations (spec section 10: each degrades the run to exit 2, never fails it). News and
# sentiment are derived from the final state at render time (a named news source that failed and was then
# retried successfully must leave no stale note). The chart note is decided again on each attempt, where the
# chart is drawn, and the judge note is recorded where it is caught.
NEWS_NOT_FETCHED_NOTE = "news was not fetched"
NEWS_UNAVAILABLE_NOTE = "news unavailable from every source"
SENTIMENT_UNAVAILABLE_NOTE = "sentiment unavailable"
CHART_UNAVAILABLE_NOTE = "chart unavailable"
JUDGE_UNAVAILABLE_NOTE = "judge unavailable: prose checked by the deterministic layer only"
# Not a decoration but a warning about the data: when a second price source disagrees, the owner chose to send the
# report with the closes named under its status line and exit 2, rather than fail the run. Derived at render time too.
PRICE_DISAGREEMENT_NOTE = "closes disagree with a second price source"
# Recorded before the email is rendered, so its status line and the exit code agree.
TEMPLATE_PROSE_NOTE = "narrative is template-generated after grounding failures"
# What each degradation says in the email, in plain words. The notes themselves stay fixed, because summary.json records
# them and the README lists them; only the email words them for a reader. A note without an entry is shown as it is,
# never dropped. B3: the template sentence carries no rewrite count, which would go stale when MAX_REWRITES changes.
PROBLEM_TEXTS = {
    NEWS_NOT_FETCHED_NOTE: "No news: the news step did not run, so there are no headlines.",
    NEWS_UNAVAILABLE_NOTE: "No news: every news source failed, so there are no headlines.",
    SENTIMENT_UNAVAILABLE_NOTE: "No sentiment score: the headlines were not rated.",
    CHART_UNAVAILABLE_NOTE: "No chart: it could not be drawn.",
    JUDGE_UNAVAILABLE_NOTE: ("The AI reviewer that double-checks the written summary was unavailable, so only the "
                             "automatic fact check ran."),
    TEMPLATE_PROSE_NOTE: ("The written summary is a fixed template: the AI's draft did not pass the fact check within "
                          "the allowed rewrites."),
    # Used only if the price check has no disagreement of its own to show; its detail names each date and both closes.
    PRICE_DISAGREEMENT_NOTE: "Closing prices disagree with a second price source.",
}
# A prior close below this is a warning, not a filter: the report keeps the literal top gainer, and stocks this cheap
# are often thinly traded, so a big move can come from a few trades.
THIN_STOCK_PRICE_THRESHOLD = 5.0
# How the footer names the sandbox that ran the verified analysis; any other backend keeps its own name.
SANDBOX_PHRASES = {"docker": "an isolated sandbox (Docker)", "subprocess": "a less isolated fallback sandbox (subprocess)"}
# The rewrite critique when the judge rules the prose unfaithful but names no sentence. A live run showed that verdict;
# with no listed issue there was no finding, so no rewrite, and unchecked prose went out.
# How the judge is told what the stock is to this session. Without the session facts, it flagged sentences naming the
# stock, the date or its top-gainer status as unsupported additions, and every live run paid a rewrite for it.
GAINER_ROLE = "top NASDAQ gainer of the session"
JUDGE_UNNAMED_ISSUE = ("judge: the paragraphs are not faithful to the verified data or headlines, but no sentence "
                       "was named; recheck every claim, count and direction against the verified metrics and cited "
                       "headlines")


def _validation_summary(error: ValidationError) -> str:
    """Field and problem for each invalid part of the narrative. Built from errors(include_url=False): this text
    reaches the model, and pydantic's documentation URL names its version, so an upgrade would change the next prompt
    (Tasks 22+23 fix round 1, Important 2)."""
    return "; ".join(f"{'.'.join(str(part) for part in detail['loc']) or 'narrative'}: {detail['msg']}"
                     for detail in error.errors(include_url=False))


def _verified_attempt_info(ctx: RunContext) -> AttemptInfo | None:
    """Review fix (Important): the report must cite the attempt verify_analysis actually
    verified, not whichever run_python call happened to run last (which may have failed, or
    succeeded but never been re-verified)."""
    if ctx.analysis.verified_attempt is None:
        return None
    return next((a for a in ctx.analysis.attempts if a.number == ctx.analysis.verified_attempt), None)


def _plain_list(names: list[str]) -> str:
    """Names joined for a sentence: "A", "A and B", "A, B and C"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _problems(ctx: RunContext) -> list[str]:
    """One plain sentence per degradation, in the order they were recorded. A price disagreement is worded by the check
    that found it, which names each date and both closes."""
    check = ctx.history.price_check if ctx.history else None
    problems = []
    for note in ctx.degradations:
        if note == PRICE_DISAGREEMENT_NOTE and check is not None and check.status == "disagree":
            problems.append(check.detail)
        else:
            problems.append(PROBLEM_TEXTS.get(note, note))
    return problems


def _warnings(ctx: RunContext) -> list[str]:
    """Worth knowing, but nothing went wrong with the run, so none of these changes the exit code."""
    warnings = []
    if ctx.gainer.prev_close < THIN_STOCK_PRICE_THRESHOLD:
        warnings.append(f"The prior close was under ${THIN_STOCK_PRICE_THRESHOLD:.0f}. Stocks this cheap are often "
                        "thinly traded, so treat the move with caution.")
    if ctx.session.early_close:
        warnings.append("The session was an early close, so the day's trading was shorter than usual.")
    # News is gathered from every source, so one that failed while another answered leaves fewer sources than usual.
    # Only a warning: the news is present, so the run is not degraded.
    failed = [source_label(a.source) for a in ctx.news_sources_tried if not a.ok]
    answered = [source_label(a.source) for a in ctx.news_sources_tried if a.ok]
    if failed and answered:
        warnings.append(f"News from {_plain_list(failed)} was unavailable this run, so the headlines come from "
                        f"{_plain_list(answered)} only.")
    return warnings


def _background(ctx: RunContext) -> list[str]:
    """How the report was made, for the footer: where the data came from, which prices the figures use, and how the
    analysis ran and was checked. Plain words first, with the technical name in brackets."""
    attempt = _verified_attempt_info(ctx)
    backend = attempt.backend if attempt else "unknown"
    sandbox = SANDBOX_PHRASES.get(backend, f"the {backend} sandbox")
    background = [f"Top gainer from {source_label(ctx.gainer.source)}; prices from {source_label(ctx.history.source)}.",
                  ("The day's gain uses closing prices as traded; the five-day figures use prices adjusted for events "
                   "such as splits, so the two can differ slightly."),
                  (f"The analysis code was written by AI and ran in {sandbox}; separate code re-checked every "
                   "calculated figure.")]
    check = ctx.history.price_check
    if check is not None and check.status != "disagree":  # agreement, or why the check could not be made
        background.append(check.detail)
    return background


def _derive_news_and_sentiment_degradations(ctx: RunContext) -> None:
    """Decide the news and sentiment degradations from the final state, at render time (spec
    section 10). Called once per composed report, from _finish. The confirmed review bug this
    fixes: a named news source that failed recorded 'news unavailable from every source' eagerly
    and kept it after a later source succeeded, so the run exited 2 with news present. The sentiment
    note is re-derived too: an attempt that fails to render leaves compose_report offered, and the
    model may record sentiment before the next attempt."""
    # Remove any stale news or sentiment note first, then re-derive the correct ones (if any).
    for stale in (NEWS_NOT_FETCHED_NOTE, NEWS_UNAVAILABLE_NOTE, SENTIMENT_UNAVAILABLE_NOTE):
        while stale in ctx.degradations:
            ctx.degradations.remove(stale)
    news_succeeded = ctx.news is not None
    headlines_present = news_succeeded and bool(ctx.news.headlines)
    if headlines_present:
        pass  # a source returned headlines: no news degradation
    elif not news_succeeded and ctx.news_sources_tried:
        ctx.degradations.append(NEWS_UNAVAILABLE_NOTE)  # tried every source, none succeeded
    elif not ctx.news_sources_tried:
        ctx.degradations.append(NEWS_NOT_FETCHED_NOTE)  # get_news was never called
    # else: a source succeeded but returned no headlines -- not a degradation.
    # Sentiment is a decoration only when there were headlines to assess.
    if headlines_present and ctx.sentiment is None:
        ctx.degradations.append(SENTIMENT_UNAVAILABLE_NOTE)


def _derive_price_check_degradation(ctx: RunContext) -> None:
    """A disagreement between the price sources degrades the run to exit 2. A check that could not be made does not:
    the run is as good as one without the check, and the footer says why."""
    check = ctx.history.price_check if ctx.history else None
    if check is not None and check.status == "disagree" and PRICE_DISAGREEMENT_NOTE not in ctx.degradations:
        ctx.degradations.append(PRICE_DISAGREEMENT_NOTE)


def _derive_template_prose_degradation(ctx: RunContext, template_prose: bool) -> None:
    """Decide the template-narrative degradation before the email is rendered, so its status line and the exit code
    agree. A render or file write that fails leaves the report uncomposed and compose_report offered again, so any note
    from that attempt is removed first, and this attempt adds it only if it also uses the template."""
    while TEMPLATE_PROSE_NOTE in ctx.degradations:
        ctx.degradations.remove(TEMPLATE_PROSE_NOTE)
    if template_prose:
        ctx.degradations.append(TEMPLATE_PROSE_NOTE)


def _finish(ctx: RunContext, deps: Deps, narrative: Narrative | None, template_prose: bool, findings: list[str], judge_faithful: bool | None) -> str:
    _derive_news_and_sentiment_degradations(ctx)
    _derive_price_check_degradation(ctx)
    # A chart error is a decoration, never fatal (spec section 10). The report renders without the
    # chart in both HTML and text, the email is built with no inline image, and "chart unavailable"
    # is recorded so finalize exits 2. Rendered before ReportContext so the problems carry the note.
    # Each attempt decides the note again: one that failed to render may have left it after a chart
    # that this attempt draws.
    while CHART_UNAVAILABLE_NOTE in ctx.degradations:
        ctx.degradations.remove(CHART_UNAVAILABLE_NOTE)
    chart_path: Path | None
    chart_cid: str | None
    try:
        chart_path = render_chart(ctx.history.expected_dates, ctx.history.ticker.adj_closes(),
                                  deps.run_dir.path / CHART_FILENAME)
        chart_cid = f"chart@{ctx.run_id}"
    except Exception:
        # Intentionally broad: any charting failure (matplotlib backend, bad data, disk) degrades
        # rather than crashing compose. Logged with traceback so a real fault is diagnosable.
        log.warning("chart unavailable", exc_info=True)
        ctx.degradations.append(CHART_UNAVAILABLE_NOTE)
        chart_path = None
        chart_cid = None
    _derive_template_prose_degradation(ctx, template_prose)
    attempt = _verified_attempt_info(ctx)
    rc = ReportContext(run_id=ctx.run_id, session_date=ctx.session.date, session_label=ctx.session.label,
                       symbol=ctx.gainer.symbol, company=ctx.gainer.company, pct_change=ctx.gainer.pct_change,
                       prev_close=ctx.gainer.prev_close, close=ctx.gainer.close, verified=ctx.analysis.verified_result,
                       benchmark_symbol=deps.settings.benchmark_symbol, narrative=narrative, template_prose=template_prose,
                       headlines=ctx.news.headlines if ctx.news else [],
                       sentiment_label=ctx.sentiment.label if ctx.sentiment else None,
                       sentiment_score=ctx.sentiment.score if ctx.sentiment else None, problems=_problems(ctx),
                       warnings=_warnings(ctx), background=_background(ctx),
                       code_hash=attempt.code_hash if attempt else None, chart_cid=chart_cid)
    subject, text, html = render_report(rc)
    text_path = deps.run_dir.write_text("report.txt", text)
    html_path = deps.run_dir.write_text("report.html", html)
    ctx.report = ReportInfo(subject=subject, text_path=str(text_path), html_path=str(html_path),
                            chart_path=str(chart_path) if chart_path else None,
                            template_prose=template_prose, rewrites=ctx.compose_rewrites, grounding_findings=findings,
                            judge_faithful=judge_faithful)
    ctx.progress.composed = True
    return ok({"composed": True, "subject": subject, "template_prose": template_prose, "next": "call send_email"})


def make_compose_report(ctx: RunContext, deps: Deps):
    @tool("compose_report")
    def compose_report(trend_paragraph: str, news_paragraph: str, declared_metrics: list[DeclaredMetric],
                       citations: list[Citation], headline_notes: list[HeadlineNote]) -> str:
        """Write the report narrative for a reader, in plain English: say "annualised volatility", never
        "volatility_annualized_pct"; metric names belong only in declared_metrics. trend_paragraph and
        news_paragraph are at most 800 characters each. trend_paragraph interprets the verified five-day metrics;
        news_paragraph gives the overall picture of the news in a few sentences, each citing headline ids like [1].
        headline_notes has one note per story: a story is every headline that reports the same information,
        whichever sources it came from, so list all their ids in one note, the one whose title the email should show
        first, and put every headline in exactly one note. Each note's summary tells the reader what the story
        reports, using its headlines' titles and summaries, what it means for the company, and how it may relate to
        the stock's price move: context, not a proven cause, weighing when it was published against the session
        (news after the session can report the move but cannot have caused it). Write 60 to 100 words for a story
        about the company, and one or two sentences for a general market or sector item; at most 800 characters.
        List the notes most recent first, by when each story first appeared, stories without a date last; the email
        shows one entry per story in that order, tagged with every source. Leave news_paragraph and headline_notes
        empty if there are no headlines. Every number you write must appear in declared_metrics with its verified
        value (a verified metric name, or session_pct_change, prev_close or close for the session's own move and
        closing prices), except that a note, or a news sentence, may quote a figure its own headlines state. Clock
        times and dates are fine. The numbers in the email itself come from the verified state, never from this
        prose. Returns a critique to rewrite, or confirms the report is composed."""
        # Amendment: every return path below funnels through _record, so tool_log.jsonl gets
        # exactly one entry per call whatever the outcome, including the invalid-narrative
        # (ValidationError) refusal that used to return early without recording.
        def _record(outcome: str) -> str:
            record_call(ctx, deps, "compose_report", {"declared": len(declared_metrics), "citations": len(citations),
                                                      "notes": len(headline_notes)}, outcome)
            return outcome

        blocked = precondition(ctx, "verified", "compose_report", "verify_analysis must pass first")
        if blocked:
            return _record(blocked)
        # Review fix (Important, B2): a finished stage must close, in the other tools' style. Once
        # the report has been composed, composing again could only replace a finished report (and,
        # after send, one nobody will ever see) -- refuse rather than silently redo finished work.
        # composed is set only on success (in _finish), so a rewrite loop, which never sets it, is
        # unaffected. This also covers the already-sent case, since send requires a composed report.
        if ctx.progress.composed:
            return _record(err("precondition for compose_report not met: the report has already been composed"))
        try:
            narrative = Narrative(trend_paragraph=trend_paragraph, news_paragraph=news_paragraph,
                                  declared_metrics=declared_metrics, citations=citations, headline_notes=headline_notes)
        except ValidationError as e:
            return _record(err(f"invalid narrative: {_validation_summary(e)}"))
        headlines = ctx.news.headlines if ctx.news else []
        # Controller correction b: these three names -- not produced by verify_analysis, since
        # they come from find_top_gainer -- become declarable alongside the metric names, so
        # the narrative may quote the session's own close-to-close move and prices.
        extra_facts = {"session_pct_change": ctx.gainer.pct_change, "prev_close": ctx.gainer.prev_close, "close": ctx.gainer.close}
        # Review fix (Critical): a year-shaped number is exempt from the grounding check only
        # for years the model could legitimately be writing about -- the session's own year and
        # the one before it (e.g. "since 2025"). Any other 19xx/20xx-shaped number must match a
        # declared value like any other number.
        session_year = date.fromisoformat(ctx.session.date).year
        grounding = check_grounding(narrative, ctx.analysis.verified_result, headlines, extra_facts=extra_facts,
                                    exempt_years={session_year, session_year - 1})
        findings = list(grounding.findings)
        # Review fix (Important): with no headlines to cite, the news paragraph must be empty;
        # anything else is uncited prose that check_grounding never sees, because its own
        # citation loop is skipped entirely when there are no headlines to check ids against.
        if not headlines and news_paragraph.strip():
            findings.append("no headlines were fetched; leave news_paragraph empty")
        judge_faithful: bool | None = None
        if not findings and deps.judge is not None:
            session_facts = {"symbol": ctx.gainer.symbol, "company": ctx.gainer.company, "session_date": ctx.session.date,
                             "session_label": ctx.session.label, "role": GAINER_ROLE}
            try:
                verdict = deps.judge(ctx.analysis.verified_result, headlines, narrative, extra_facts=extra_facts,
                                     session_facts=session_facts)
                judge_faithful = verdict.faithful
                if not verdict.faithful:
                    issues = [f"judge: \"{i.sentence}\": {i.problem}" for i in verdict.issues]
                    findings.extend(issues or [JUDGE_UNNAMED_ISSUE])
            except CassetteMiss as e:
                # Fix round 1, K3: in replay the verdict comes from the cassette. A miss means the recording cannot
                # serve this run -- not an outage to decorate around -- so it propagates and the run fails loudly
                # (a failure notice and exit 1), and the run's errors name the judge.
                ctx.errors.append(f"judge: {type(e).__name__}: {e}"[:JUDGE_ERROR_CHARS])
                raise
            except Exception:
                # Controller correction d / B1: intentionally broad. The judge is a decoration on
                # top of the deterministic grounding check above, not a gate, so an LLM-judge
                # outage (network, provider, parsing, ...) must never fail an otherwise-grounded
                # run. It is a degradation, not a plain note (spec section 10: exit 2), recording
                # that the prose was still checked by the deterministic layer. Logged as a warning
                # (with traceback) so a real outage is diagnosable, and so a signature mismatch in
                # deps.judge itself is never mistaken for one. judge_failures is kept for the
                # recording seal, which must not seal a cassette whose judge verdict is missing.
                log.warning("judge unavailable", exc_info=True)
                if JUDGE_UNAVAILABLE_NOTE not in ctx.degradations:
                    ctx.degradations.append(JUDGE_UNAVAILABLE_NOTE)
                ctx.judge_failures += 1
        if findings:
            ctx.compose_rewrites += 1
            if ctx.compose_rewrites > MAX_REWRITES:
                outcome = _finish(ctx, deps, None, True, findings, judge_faithful)
            else:
                remaining = MAX_REWRITES - ctx.compose_rewrites + 1
                plural = "rewrite" if remaining == 1 else "rewrites"
                outcome = err("grounding failed: " + " | ".join(findings)
                              + f". Rewrite and call compose_report again ({remaining} {plural} left).")
        else:
            outcome = _finish(ctx, deps, narrative, False, [], judge_faithful)
        return _record(outcome)
    return compose_report
