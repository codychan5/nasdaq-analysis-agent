import json
from tests import fakes
from tests.unit.test_tools_data import deps, ctx, universe  # noqa: F401


def _with_file_transport(deps):
    from nasdaq_agent.email.transports import FileTransport
    deps.transport = FileTransport(deps.run_dir.outbox_dir)
    return deps


def test_clean_run_exits_zero(ctx, deps):
    from nasdaq_agent.agent.finalize import finalize
    ctx.progress.sent = True
    assert finalize(ctx, _with_file_transport(deps)) == 0
    assert json.loads((deps.run_dir.path / "summary.json").read_text())["exit_code"] == 0


def test_degraded_run_exits_two(ctx, deps):
    from nasdaq_agent.agent.finalize import finalize
    ctx.progress.sent = True; ctx.degradations.append("news unavailable from every source")
    assert finalize(ctx, _with_file_transport(deps)) == 2


def test_unsent_run_sends_failure_notice_and_exits_one(ctx, deps):
    from nasdaq_agent.agent.finalize import finalize
    from nasdaq_agent.email.idempotency import SendMarker
    ctx.progress.session_resolved = True; ctx.last_tool = "find_top_gainer"; ctx.errors.append("massive: timeout")
    ctx.give_up_reason = "all gainer sources failed"
    assert finalize(ctx, _with_file_transport(deps)) == 1
    eml = next(deps.run_dir.outbox_dir.glob("*.eml")).read_bytes().decode()
    assert "failed" in eml and "all gainer sources failed" in eml and "find_top_gainer" in eml
    assert SendMarker(deps.run_dir.path / "failure_sent.json").state == "sent"
    assert finalize(ctx, deps) == 1 and len(list(deps.run_dir.outbox_dir.glob("*.eml"))) == 1  # idempotent


def test_unsent_run_with_sending_marker_reports_unknown_delivery_outcome(ctx, deps):
    """A report SendMarker left in "sending" means the send attempt's outcome was never recorded (the process could have
    crashed on either side of the actual delivery). finalize must not claim "did not produce a report" in that case --
    it must say the delivery outcome is unknown -- but still exits 1, since it cannot confirm the report went out.
    """
    from nasdaq_agent.agent.finalize import finalize
    from nasdaq_agent.agent.tools.send import SENT_MARKER
    from nasdaq_agent.email.idempotency import SendMarker
    deps = _with_file_transport(deps)
    SendMarker(deps.run_dir.path / SENT_MARKER).begin()
    assert finalize(ctx, deps) == 1
    eml = next(deps.run_dir.outbox_dir.glob("*.eml")).read_bytes().decode()
    assert "delivery outcome unknown" in eml
    assert "may or may not have been delivered" in eml
    assert "did not produce a report" not in eml


def test_finalize_reconciles_sent_marker_before_branching(ctx, deps):
    """send_email marks the report SendMarker "sent" before persisting ctx.progress.sent = True. A crash in that window
    must not make finalize send a failure notice about a report that was, in fact, delivered -- finalize must reconcile
    first, flip progress.sent, populate ctx.email from the marker, and exit clean, sending nothing.
    """
    from nasdaq_agent.agent.finalize import finalize
    from nasdaq_agent.agent.tools.send import SENT_MARKER
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult
    deps = _with_file_transport(deps)
    marker = SendMarker(deps.run_dir.path / SENT_MARKER)
    marker.begin()
    marker.complete(SendResult(transport="file", message_id="<x@y>", location="somewhere"))
    assert ctx.progress.sent is False  # context.json is the stale side of the crash window

    assert finalize(ctx, deps) == 0

    assert ctx.progress.sent is True
    assert "delivery reconciled from the send marker after an interrupted run" in ctx.notes
    assert ctx.email is not None and ctx.email.message_id == "<x@y>"
    assert list(deps.run_dir.outbox_dir.glob("*.eml")) == []


def test_finalize_fills_ctx_email_from_marker_even_when_already_marked_sent(ctx, deps):
    """send_email's own already-sent branch (tools/send.py) restores ctx.progress.sent on a resumed call but leaves
    ctx.email as None. reconcile_delivery must still fill ctx.email from the marker in that case -- progress.sent
    already being True must not skip reconciliation entirely -- but must NOT add the reconciliation note, since
    progress.sent did not itself change.
    """
    from nasdaq_agent.agent.finalize import finalize
    from nasdaq_agent.agent.tools.send import SENT_MARKER
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult
    deps = _with_file_transport(deps)
    marker = SendMarker(deps.run_dir.path / SENT_MARKER)
    marker.begin()
    marker.complete(SendResult(transport="file", message_id="<z@w>", location="somewhere-else"))
    ctx.progress.sent = True  # as send_email's already-sent branch would leave it
    assert ctx.email is None

    assert finalize(ctx, deps) == 0

    assert ctx.email is not None
    assert ctx.email.transport == "file" and ctx.email.message_id == "<z@w>" and ctx.email.location == "somewhere-else"
    assert "delivery reconciled from the send marker after an interrupted run" not in ctx.notes
    assert list(deps.run_dir.outbox_dir.glob("*.eml")) == []


def test_finalize_survives_transport_raising_unexpected_exception(ctx, deps):
    """finalize_with must never raise, even when the transport itself blows up with something that isn't SendError. It
    logs, records a redacted and truncated error, still writes summary.json, and returns the computed exit code."""
    from nasdaq_agent.agent.finalize import finalize

    class ExplodingTransport:
        name = "exploding"

        def send(self, msg):
            raise RuntimeError("boom")

    deps.transport = ExplodingTransport()

    assert finalize(ctx, deps) == 1
    assert json.loads((deps.run_dir.path / "summary.json").read_text())["exit_code"] == 1
    assert any("failure notice error: RuntimeError" in e for e in ctx.errors)


def _notice_parts(deps) -> tuple[str, str, str]:
    """The failure notice as raw text, plain part and HTML part. The parts are decoded, because a body line longer
    than 78 characters is sent quoted-printable, whose soft line breaks can split a string in the raw bytes."""
    import email
    from email import policy
    (eml_path,) = deps.run_dir.outbox_dir.glob("*.eml")
    raw = eml_path.read_bytes()
    msg = email.message_from_bytes(raw, policy=policy.default)
    return (raw.decode(), msg.get_body(preferencelist=("plain",)).get_content(),
            msg.get_body(preferencelist=("html",)).get_content())


def test_unreadable_report_marker_reports_unknown_delivery_outcome(ctx, deps):
    """A sent.json that exists but cannot be read -- here empty, as a crash between begin()'s exclusive create and its
    write would leave it -- means a send attempt began and its outcome is unknown. finalize words the notice exactly as
    for "sending", never "did not produce a report", does not reconcile the marker as sent, and does not raise."""
    from nasdaq_agent.agent.finalize import finalize
    from nasdaq_agent.agent.tools.send import SENT_MARKER
    deps = _with_file_transport(deps)
    (deps.run_dir.path / SENT_MARKER).write_text("")

    assert finalize(ctx, deps) == 1

    assert ctx.progress.sent is False and ctx.email is None
    raw, plain, _ = _notice_parts(deps)
    assert "delivery outcome unknown" in raw
    assert "may or may not have been delivered" in plain
    assert "did not produce a report" not in plain
    assert json.loads((deps.run_dir.path / "summary.json").read_text())["exit_code"] == 1


def test_failure_notice_redacts_each_piece_before_truncating_it(ctx, deps):
    """Each piece is redacted before it is cut to MAX_ERROR_CHARS. Cutting first would keep the first five characters of
    a secret that starts at character 296 -- a fragment the whole-body redaction can no longer recognise -- in both the
    reason and the error list."""
    from pydantic import SecretStr

    from nasdaq_agent.agent.finalize import MAX_ERROR_CHARS, finalize
    secret = "sk-test-straddle-0123456789abcdef"
    kept_by_a_cut_first = 5
    padding = MAX_ERROR_CHARS - kept_by_a_cut_first  # 295 characters, then the secret
    deps = _with_file_transport(deps)
    deps.settings = deps.settings.model_copy(update={"google_api_key": SecretStr(secret)})
    ctx.errors.append("e" * padding + secret)
    ctx.give_up_reason = "r" * padding + secret

    assert finalize(ctx, deps) == 1

    fragment = secret[:kept_by_a_cut_first]
    _, plain, html = _notice_parts(deps)
    for text in (plain, html):
        assert secret not in text and fragment not in text
    assert plain.count("***") >= 2  # redacted, not merely cut away: the reason and the error both keep the marker


class _RefusingTransport:
    """An SMTP transport whose every send fails, as when the relay is down."""
    name = "smtp"

    def send(self, msg):
        from nasdaq_agent.email.transports import SendError
        raise SendError("smtp send failed: ConnectionRefusedError")


def test_a_failure_notice_smtp_cannot_send_is_kept_in_the_outbox(ctx, deps):
    from nasdaq_agent.agent.finalize import finalize
    from nasdaq_agent.email.idempotency import SendMarker
    ctx.give_up_reason = "all gainer sources failed"
    deps.transport = _RefusingTransport()
    assert finalize(ctx, deps) == 1
    _, plain, _ = _notice_parts(deps)
    assert "all gainer sources failed" in plain
    assert SendMarker(deps.run_dir.path / "failure_sent.json").state == "failed"


def test_finalize_survives_when_the_outbox_fallback_fails_too(ctx, deps):
    from nasdaq_agent.agent.finalize import finalize
    deps.transport = _RefusingTransport()
    deps.run_dir.outbox_dir.parent.mkdir(parents=True, exist_ok=True)
    deps.run_dir.outbox_dir.write_text("a file where the outbox directory should be")
    assert finalize(ctx, deps) == 1
    assert any("failure notice outbox fallback failed" in e for e in ctx.errors)
    assert json.loads((deps.run_dir.path / "summary.json").read_text())["exit_code"] == 1


def test_finalize_returns_the_exit_code_when_it_cannot_write_its_records(ctx, deps, monkeypatch):
    from nasdaq_agent.agent.finalize import finalize

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    ctx.progress.sent = True
    monkeypatch.setattr(type(deps.run_dir), "write_json", refuse)
    monkeypatch.setattr(type(ctx), "save", refuse)
    assert finalize(ctx, _with_file_transport(deps)) == 0
