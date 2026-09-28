from pathlib import Path

from langchain_core.tools import tool

from ...email.idempotency import OUTCOME_UNKNOWN_STATES, SendMarker
from ...email.message import build_message
from ...email.transports import FileTransport, SendError
from ..context import EmailInfo, RunContext
from .common import Deps, err, log, ok, precondition, record_call

SENT_MARKER = "sent.json"
SENT_EML_FILENAME = "sent.eml"
DEFAULT_SENDER = "nasdaq-agent@localhost"


def make_send_email(ctx: RunContext, deps: Deps):
    @tool("send_email")
    def send_email() -> str:
        """Send the composed report to the configured recipient. Idempotent: a second call does nothing."""
        # Amendment: every return path below funnels through _record, so tool_log.jsonl gets
        # exactly one entry per call whatever the outcome, including the already-sent,
        # unknown-outcome (marker "sending" or unreadable) and begin-failed refusals that used
        # to return early without recording.
        def _record(outcome: str) -> str:
            record_call(ctx, deps, "send_email", {}, outcome)
            return outcome

        blocked = precondition(ctx, "composed", "send_email", "call compose_report first")
        if blocked:
            return _record(blocked)
        marker = SendMarker(deps.run_dir.path / SENT_MARKER)
        marker_state = marker.state  # read once: every branch below decides on the same reading
        if marker_state == "sent":
            # Review fix (folded): a resumed run can find the marker already "sent" from an
            # earlier process, before this in-memory ctx was ever marked -- catch it up rather
            # than leaving progress.sent False for a message that really was delivered.
            ctx.progress.sent = True
            # Task 22 correction b: no message id in a model-visible result -- it differs on every run, so replay
            # would miss the cache on the next prompt. It stays in sent.json.
            return _record(ok({"sent": True, "already_sent": True}))
        if marker_state in OUTCOME_UNKNOWN_STATES:
            # Final fix wave A5: a marker that exists but cannot be read ("unknown") is refused exactly like
            # "sending" -- either way a send attempt began and whether it was delivered is not known.
            return _record(err("a previous send attempt is in an unknown state; not sending again. Call give_up with this reason."))
        sender = deps.settings.smtp_from or DEFAULT_SENDER
        chart = Path(ctx.report.chart_path).read_bytes() if ctx.report.chart_path else None
        msg = build_message(sender, deps.settings.email_to, ctx.report.subject, Path(ctx.report.text_path).read_text(),
                            Path(ctx.report.html_path).read_text(), chart, f"chart@{ctx.run_id}")
        # Review fix (Folded Minor): build the message and check the transport BEFORE marking
        # "sending", so a missing transport can never leave the marker stuck in-flight for a
        # message that was never actually attempted -- SendMarker.state has no way back from
        # "sending" except a send that completes or fails through the transport itself.
        if deps.transport is None:
            return _record(err("no email transport is configured; call give_up with this reason"))
        if not marker.begin():
            return _record(err("send marker already exists; not sending again"))
        try:
            result = deps.transport.send(msg)
        except SendError as e:
            marker.fail(str(e))
            fallback = FileTransport(deps.run_dir.outbox_dir).send(msg)
            ctx.email = EmailInfo(status="failed", transport=deps.transport.name, message_id=None, location=fallback.location)
            ctx.errors.append(f"send failed: {e}")
            # Task 22 correction b: the outbox path names this run's directory; it stays in ctx.email.
            return _record(err(f"transport {deps.transport.name} failed ({e}); message written to the outbox. Call give_up."))
        marker.complete(result)
        # Review fix (folded): the send itself is what matters. Record it as delivered and save
        # durably BEFORE attempting the best-effort sent.eml copy below, so a failure writing
        # that copy can never leave a message that really was sent looking unsent.
        ctx.email = EmailInfo(status="sent", transport=result.transport, message_id=result.message_id, location=result.location)
        ctx.progress.sent = True
        ctx.save()
        # Review fix (Important, best-effort): keep a fixed-name copy of exactly what was
        # delivered, in the run directory itself, independent of whichever transport sent it
        # (SmtpTransport writes nothing locally; FileTransport's own copy lives under a
        # timestamped name in the outbox, not at this well-known path). A failure writing this
        # convenience copy must never undo a send that already succeeded -- log and move on.
        try:
            (deps.run_dir.path / SENT_EML_FILENAME).write_bytes(bytes(msg))
        except OSError as e:
            log.warning("failed to write sent.eml copy: %s", e, exc_info=True)
        # Task 22 correction b: message id, location and transport stay in ctx.email and sent.json. The model-visible
        # result carries none of them: the id and path differ on every run, and replay always delivers to the file
        # outbox (correction j) whatever transport the recording used, so any of them would change the next prompt
        # and replay would miss the cache.
        return _record(ok({"sent": True}))
    return send_email
