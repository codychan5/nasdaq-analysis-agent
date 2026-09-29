from langchain_core.tools import tool

from ...metrics import DEFINITIONS_TEXT, METRIC_NAMES, PCT_TOLERANCE, compute_all
from ..context import AttemptInfo, RunContext, VerificationRow
from .common import Deps, err, ok, precondition, record_call, round_floats

MAX_REJECTIONS = 2


def _definition_line(metric: str) -> str:
    for line in DEFINITIONS_TEXT.splitlines():
        if line.strip().startswith(f"- {metric}"):
            return line.strip()
    return ""


def _compare(name: str, model_value, verifier_value) -> VerificationRow:
    if name == "trend":
        okay = str(model_value) == str(verifier_value)
    elif name == "daily_changes_pct":
        okay = len(model_value) == len(verifier_value) and all(abs(a - b) <= PCT_TOLERANCE for a, b in zip(model_value, verifier_value))
    else:
        okay = abs(float(model_value) - float(verifier_value)) <= PCT_TOLERANCE
    return VerificationRow(metric=name, model_value=str(model_value), verifier_value=str(verifier_value), ok=okay)


def _last_successful_attempt(ctx: RunContext) -> AttemptInfo | None:
    return next((a for a in reversed(ctx.analysis.attempts) if a.ok), None)


def make_verify_analysis(ctx: RunContext, deps: Deps):
    @tool("verify_analysis")
    def verify_analysis() -> str:
        """Recompute every metric with reviewed code from the same bars and compare with your latest successful
        run. Passes when all metrics match within tolerance; otherwise returns a critique naming each mismatch."""
        # Every return path below funnels through _record, so tool_log.jsonl gets
        # exactly one entry per call whatever the outcome, including the analysis-terminal guard.
        def _record(outcome: str) -> str:
            record_call(ctx, deps, "verify_analysis", {}, outcome)
            return outcome

        blocked = precondition(ctx, "successful_run", "verify_analysis", "run_python must succeed first")
        if blocked:
            return _record(blocked)
        if ctx.progress.analysis_terminal:
            return _record(err("analysis is terminal after repeated rejections; call give_up with a reason"))
        # A finished stage must close. Once verified, hand back the
        # stored result every time -- never recompute, never rewrite verification.json -- so a
        # repeat call (e.g. right before compose_report) can't drift from the number the report
        # will actually use or spend any more of the rejection/attempt budget.
        if ctx.progress.verified:
            return _record(ok({"verified": True, "metrics": ctx.analysis.verified_result.model_dump(),
                               "next": "get_news if not done, then compose_report"}))
        last_successful = _last_successful_attempt(ctx)
        # This exact successful attempt was already compared against
        # the verifier and rejected. Calling verify_analysis again with no new run_python call
        # in between would just re-judge identical numbers -- refuse without spending another
        # rejection, and tell the model what it actually needs to do next.
        if last_successful is not None and ctx.analysis.last_judged_attempt == last_successful.number:
            return _record(err(f"attempt {last_successful.number} was already judged and rejected; call "
                               "run_python again to produce a new result before calling verify_analysis"))
        model = ctx.analysis.latest_result
        expected = compute_all(ctx.history.ticker.adj_closes(), ctx.history.benchmark.adj_closes())
        rows = [_compare(name, getattr(model, name), getattr(expected, name)) for name in METRIC_NAMES]
        ctx.analysis.verification = rows
        ctx.analysis.last_judged_attempt = last_successful.number
        deps.run_dir.write_json("verification.json", [r.model_dump() for r in rows])
        if all(r.ok for r in rows):
            ctx.analysis.verified_result = expected
            ctx.analysis.verified_attempt = last_successful.number
            ctx.progress.verified = True
            outcome = ok({"verified": True, "metrics": expected.model_dump(), "next": "get_news if not done, then compose_report"})
        else:
            ctx.analysis.rejections += 1
            bad = [r.metric for r in rows if not r.ok]
            # The model sees both values rounded: the model's comes from the sandbox's numpy, whose builds can differ in
            # the last digit between machines, and a full-precision value would change the next prompt and break
            # replay. The verification rows keep full precision.
            lines = [f"{metric}: model {round_floats(getattr(model, metric))} vs verifier "
                     f"{round_floats(getattr(expected, metric))}; {_definition_line(metric)}" for metric in bad]
            # Reaching the run cap must end the analysis even when the
            # rejection count alone has not exceeded MAX_REJECTIONS -- otherwise the model is
            # told to "run again" with no attempts left to spend, and only discovers the cap on
            # the next run_python call instead of here, where the decision actually belongs.
            at_cap = len(ctx.analysis.attempts) >= deps.settings.max_code_runs
            if ctx.analysis.rejections > MAX_REJECTIONS or at_cap:
                ctx.progress.analysis_terminal = True
                reason = (f"the code-run cap of {deps.settings.max_code_runs} attempts has been reached" if at_cap
                          else f"verification rejected {ctx.analysis.rejections} times")
                outcome = err(f"{reason}; analysis is terminal; call give_up. " + " | ".join(lines))
            else:
                outcome = err("verification failed: " + " | ".join(lines) + ". Fix the code and call run_python again.")
        return _record(outcome)
    return verify_analysis
