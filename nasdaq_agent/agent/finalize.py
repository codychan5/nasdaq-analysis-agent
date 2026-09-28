import html
import logging

from ..artifacts import RunDir, redact
from ..config import Settings
from ..email.idempotency import OUTCOME_UNKNOWN_STATES, SendMarker
from ..email.message import build_message
from ..email.transports import FileTransport, SendError, Transport
from .context import EmailInfo, RunContext
from .tools.common import Deps
from .tools.send import DEFAULT_SENDER, SENT_MARKER

log = logging.getLogger("nasdaq_agent.finalize")
EXIT_CLEAN, EXIT_DEGRADED, EXIT_FAILED = 0, 2, 1
FAILURE_MARKER = "failure_sent.json"
MAX_ERROR_CHARS = 300


def reconcile_delivery(ctx: RunContext, run_dir: RunDir) -> bool:
    """Fix round 1, item 1: send_email (tools/send.py) calls marker.complete(result) -- the report
    SendMarker on disk becomes "sent" -- before ctx.progress.sent = True is persisted. A crash in
    that exact window leaves sent.json at "sent" and context.json at sent=False; without this,
    finalize (or a resumed run) would wrongly conclude no report was ever produced. Catches ctx up
    to what actually happened. Returns True when it changed anything, so callers know whether ctx
    needs saving.

    Coordinator amendment: ctx.email must be filled from the marker whenever the report marker
    says "sent" and ctx.email is still None, even when ctx.progress.sent is already True.
    send_email's own already-sent branch (tools/send.py) restores progress.sent on a resumed call
    but leaves ctx.email as None, and skipping reconciliation entirely just because progress.sent
    was already True would leave the run's record never naming the transport/message_id sent.json
    already holds. The reconciliation note is added only when progress.sent itself actually
    changes from False to True -- filling in ctx.email on its own is not that event.

    Final fix wave A5: never raises. The marker is read once, and a missing or unreadable ("unknown") marker reads as
    {}, so an unknown outcome is never reconciled as sent; of a "sent" marker's fields only strings reach ctx.email.
    """
    contents = SendMarker(run_dir.path / SENT_MARKER).read()
    if contents.get("state") != "sent":
        return False
    changed = False
    if not ctx.progress.sent:
        ctx.progress.sent = True
        # A note, not a degradation: the delivered report itself is complete: only the
        # bookkeeping around it was interrupted.
        ctx.notes.append("delivery reconciled from the send marker after an interrupted run")
        changed = True
    if ctx.email is None:
        ctx.email = EmailInfo(status="sent", transport=_string_or_none(contents.get("transport")),
                              message_id=_string_or_none(contents.get("message_id")),
                              location=_string_or_none(contents.get("location")))
        changed = True
    return changed


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def build_failure_notice(ctx: RunContext, run_dir: RunDir, secrets: list[str]) -> tuple[str, str]:
    """The failure notice's subject and body, redacted.

    Final fix wave A6: the reason and each error are redacted BEFORE they are cut to MAX_ERROR_CHARS. Cutting first
    could keep the front of a secret that straddles the cut, which no later redaction can recognise. The whole body
    is then redacted again (fix round 1, item 4: defence in depth, since ctx.errors can carry text this module does not
    control the source of).
    """
    def clipped(text: str) -> str:
        return redact(text, secrets)[:MAX_ERROR_CHARS]

    reason = ctx.give_up_reason or (ctx.errors[-1] if ctx.errors else "the agent ended without sending a report")
    completed = [name for name, done in ctx.progress.model_dump().items() if done and name not in ("gave_up", "analysis_terminal")]
    # Controller correction d (round 0): a report SendMarker left in "sending" means the send
    # attempt's outcome -- did the transport actually accept it? -- was never recorded, e.g. the
    # process crashed mid-send. Narrower, different claim than "no report was produced", so it
    # gets its own subject/opening line. Final fix wave A5: a marker that exists but cannot be read
    # ("unknown") gets the same wording, for the same reason.
    unknown_delivery_outcome = SendMarker(run_dir.path / SENT_MARKER).state in OUTCOME_UNKNOWN_STATES
    if unknown_delivery_outcome:
        subject = f"[NASDAQ agent] run {ctx.run_id} delivery outcome unknown"
        opening = (f"Run {ctx.run_id}: the report may or may not have been delivered. "
                   "A previous send attempt began but its outcome was never recorded.")
    else:
        subject = f"[NASDAQ agent] run {ctx.run_id} failed"
        opening = f"Run {ctx.run_id} did not produce a report."
    body = "\n".join([
        opening,
        f"Session: {ctx.session.date if ctx.session else 'unresolved'}",
        f"Last tool called: {ctx.last_tool or 'none'}",
        f"Reason: {clipped(reason)}",
        f"Completed steps: {', '.join(completed) or 'none'}",
        f"Errors: {'; '.join(clipped(e) for e in ctx.errors[-5:]) or 'none'}",
        f"Artefacts: {ctx.artifacts_path}",
    ])
    return subject, redact(body, secrets)


def _send_failure_notice(ctx: RunContext, settings: Settings, run_dir: RunDir, transport: Transport | None,
                         secrets: list[str]) -> None:
    marker = SendMarker(run_dir.path / FAILURE_MARKER)
    marker_state = marker.state
    if marker_state == "sent" or marker_state in OUTCOME_UNKNOWN_STATES:
        return
    # Controller correction d (round 0): build the subject/body/message BEFORE calling
    # marker.begin() below -- begin() has no way back to "none", only complete()/fail() move it
    # forward from "sending" -- so composing the message after it would leave the failure marker
    # stuck in "sending" forever (with no message ever attempted) if composing ever raised.
    # build_failure_notice redacts the body (fix round 1, item 4; final fix wave A6).
    subject, body = build_failure_notice(ctx, run_dir, secrets)
    # Controller correction a (round 0): the HTML part must escape the (already redacted) body.
    msg = build_message(settings.smtp_from or DEFAULT_SENDER, settings.email_to, subject, body,
                        "<pre>" + html.escape(body) + "</pre>", None, None)
    if not marker.begin():
        return
    try:
        marker.complete(transport.send(msg))
    except SendError as e:
        marker.fail(str(e))
        try:
            FileTransport(run_dir.outbox_dir).send(msg)
        except SendError as fallback_error:
            # Fix round 1, item 3: the outbox fallback is itself guarded -- it can fail too (e.g.
            # the outbox path itself is unwritable) -- and must not raise out of here.
            message = redact(f"failure notice outbox fallback failed: {type(fallback_error).__name__}: {fallback_error}",
                             secrets)[:MAX_ERROR_CHARS]
            log.error(message, exc_info=True)
            ctx.errors.append(message)
        log.error("failure notice could not be sent: %s", e)


def finalize_with(ctx: RunContext, settings: Settings, run_dir: RunDir, transport: Transport | None) -> int:
    """Fix round 1, item 2: the core of finalize, needing only settings/run_dir/transport rather
    than a full Deps, so run_once/resume_run can call it for a setup failure or an already-
    resolved resume, neither of which ever builds a Deps.
    """
    reconcile_delivery(ctx, run_dir)
    secrets = settings.secret_values()
    if ctx.progress.sent:
        code = EXIT_DEGRADED if ctx.degradations else EXIT_CLEAN
    else:
        code = EXIT_FAILED
        try:
            _send_failure_notice(ctx, settings, run_dir, transport, secrets)
        except Exception as e:
            # Fix round 1, item 3: finalize must never raise. A failure marker left at "sending"
            # by this is the documented outcome-unknown state (_send_failure_notice sends nothing
            # again on the next call) -- leave it as is.
            message = redact(f"failure notice error: {type(e).__name__}: {e}", secrets)[:MAX_ERROR_CHARS]
            log.error(message, exc_info=True)
            ctx.errors.append(message)
    write_summary(ctx, run_dir, code, secrets)
    return code


def write_summary(ctx: RunContext, run_dir: RunDir, code: int, secrets: list[str]) -> None:
    """summary.json and context.json for an exit code. Never raises. Task 22: also rewritten when a record run that
    finalized with 0 or 2 then fails to seal its cassette and exits 1, so the artefacts agree with the exit code."""
    try:
        run_dir.write_json("summary.json", {
            "run_id": ctx.run_id, "exit_code": code, "sent": ctx.progress.sent,
            "gainer": ctx.gainer.symbol if ctx.gainer else None,
            "degradations": ctx.degradations,
            "errors": [redact(e, secrets) for e in ctx.errors],
            "tool_calls": ctx.tool_calls,
            "give_up_reason": redact(ctx.give_up_reason, secrets) if ctx.give_up_reason else None,
        })
    except Exception as e:
        log.error("failed to write summary.json: %s: %s", type(e).__name__, e, exc_info=True)
    try:
        ctx.save()
    except Exception as e:
        log.error("failed to save context: %s: %s", type(e).__name__, e, exc_info=True)


def finalize(ctx: RunContext, deps: Deps) -> int:
    return finalize_with(ctx, deps.settings, deps.run_dir, deps.transport)
