import email, logging, smtplib, ssl, threading, pytest

def test_message_has_text_html_and_exactly_one_inline_chart():
    # The chart appears once, inline in the HTML part. A second attached copy made mail apps show it twice.
    from nasdaq_agent.email.message import build_message
    msg = build_message("bot@example.com", "r@example.com", "Subj", "plain", "<p>html</p>", b"\x89PNGfake", "chart@run")
    parts = {p.get_content_type() for p in msg.walk()}
    assert {"text/plain", "text/html", "image/png"} <= parts
    png_parts = [p for p in msg.walk() if p.get_content_type() == "image/png"]
    assert len(png_parts) == 1
    (chart_part,) = png_parts
    assert chart_part["Content-ID"] == "<chart@run>"
    assert chart_part.get_content_disposition() == "inline"
    assert not any(p.get_content_disposition() == "attachment" for p in msg.walk())
    assert msg["To"] == "r@example.com" and msg["Subject"] == "Subj"

def test_file_transport_writes_eml(tmp_path):
    from nasdaq_agent.email.message import build_message
    from nasdaq_agent.email.transports import FileTransport
    msg = build_message("bot@example.com", "r@example.com", "Subj", "plain", "<p>html</p>", None, None)
    res = FileTransport(tmp_path / "outbox").send(msg)
    files = list((tmp_path / "outbox").glob("*.eml"))
    assert len(files) == 1 and res.transport == "file" and res.location == str(files[0])
    assert email.message_from_bytes(files[0].read_bytes())["Subject"] == "Subj"

def test_file_transport_raises_send_error_not_oserror_when_outbox_path_is_a_file(tmp_path):
    """The transport contract is that every delivery failure surfaces as SendError -- an existing regular file at the
    outbox path makes mkdir(exist_ok=True) raise FileExistsError (an OSError), which must come out as SendError, not the
    raw OSError."""
    from nasdaq_agent.email.message import build_message
    from nasdaq_agent.email.transports import FileTransport, SendError
    outbox_path = tmp_path / "outbox"
    outbox_path.write_text("not a directory")
    msg = build_message("bot@example.com", "r@example.com", "Subj", "plain", "<p>html</p>", None, None)
    with pytest.raises(SendError):
        FileTransport(outbox_path).send(msg)

def test_smtp_transport_uses_starttls_and_login(monkeypatch):
    from nasdaq_agent.email import transports as t
    from nasdaq_agent.email.message import build_message
    calls = []
    class FakeSMTP:
        def __init__(self, host, port, timeout): calls.append(("connect", host, port))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self, context=None): calls.append(("starttls",))
        def login(self, u, p): calls.append(("login", u, p))
        def send_message(self, msg): calls.append(("send", msg["Message-ID"]))
    monkeypatch.setattr(t.smtplib, "SMTP", FakeSMTP)
    msg = build_message("bot@example.com", "r@example.com", "S", "p", "<p>h</p>", None, None)
    res = t.SmtpTransport("smtp.example.com", 587, "u", "pw").send(msg)
    assert calls[0] == ("connect", "smtp.example.com", 587) and ("starttls",) in calls and ("login", "u", "pw") in calls
    assert res.message_id == msg["Message-ID"]

def test_smtp_transport_starttls_false_skips_starttls(monkeypatch):
    from nasdaq_agent.email import transports as t
    from nasdaq_agent.email.message import build_message
    calls = []
    class FakeSMTP:
        def __init__(self, host, port, timeout): calls.append(("connect", host, port))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self, context=None): calls.append(("starttls",))
        def login(self, u, p): calls.append(("login", u, p))
        def send_message(self, msg): calls.append(("send", msg["Message-ID"]))
    monkeypatch.setattr(t.smtplib, "SMTP", FakeSMTP)
    msg = build_message("bot@example.com", "r@example.com", "S", "p", "<p>h</p>", None, None)
    res = t.SmtpTransport("smtp.example.com", 587, "u", "pw", starttls=False).send(msg)
    assert ("starttls",) not in calls
    assert ("login", "u", "pw") in calls
    assert res.message_id == msg["Message-ID"]

def scripted_smtp(monkeypatch, plans):
    """Replace smtplib.SMTP with a fake whose n-th connection follows plans[n]: None succeeds, and a (step, error)
    pair raises the error at that step: "connect", "starttls", "login", "send" or "quit". Returns the event log."""
    from nasdaq_agent.email import transports as t
    events, remaining = [], list(plans)
    class FakeSMTP:
        def __init__(self, host, port, timeout):
            self.plan = remaining.pop(0)
            events.append("connect")
            self._fail_at("connect")
        def _fail_at(self, step):
            if self.plan is not None and self.plan[0] == step:
                raise self.plan[1]
        def __enter__(self): return self
        def __exit__(self, *a):
            self._fail_at("quit")
            return False
        def starttls(self, context=None): self._fail_at("starttls")
        def login(self, u, p): self._fail_at("login")
        def send_message(self, msg):
            self._fail_at("send")
            events.append("accepted")
    monkeypatch.setattr(t.smtplib, "SMTP", FakeSMTP)
    return events

def smtp_send(retry_sleep, password="pw"):
    from nasdaq_agent.email import transports as t
    from nasdaq_agent.email.message import build_message
    msg = build_message("bot@example.com", "r@example.com", "S", "p", "<p>h</p>", None, None)
    return msg, t.SmtpTransport("smtp.example.com", 587, "u", password, retry_sleep=retry_sleep).send(msg)

def test_smtp_retries_a_connection_that_failed_before_anything_was_sent(monkeypatch):
    events = scripted_smtp(monkeypatch, [("connect", ConnectionRefusedError()), None])
    waits = []
    msg, res = smtp_send(waits.append)
    assert events == ["connect", "connect", "accepted"] and waits == [10.0]
    assert res.transport == "smtp" and res.message_id == msg["Message-ID"]

def test_smtp_gives_up_after_three_attempts(monkeypatch):
    from nasdaq_agent.email.transports import SendError
    events = scripted_smtp(monkeypatch, [("connect", TimeoutError())] * 3)
    waits = []
    with pytest.raises(SendError):
        smtp_send(waits.append)
    assert events == ["connect"] * 3 and waits == [10.0, 30.0]

@pytest.mark.parametrize("step, error", [
    ("login", smtplib.SMTPAuthenticationError(535, b"5.7.8 credentials rejected")),
    ("starttls", ssl.SSLCertVerificationError(1, "certificate verify failed")),
    ("connect", smtplib.SMTPConnectError(554, b"no SMTP service here")),
])
def test_smtp_does_not_retry_a_failure_another_attempt_cannot_fix(monkeypatch, step, error):
    from nasdaq_agent.email.transports import SendError
    events = scripted_smtp(monkeypatch, [(step, error), None])
    waits = []
    with pytest.raises(SendError):
        smtp_send(waits.append)
    assert events == ["connect"] and waits == []

@pytest.mark.parametrize("error", [
    smtplib.SMTPDataError(451, b"4.3.0 try again later"),
    smtplib.SMTPRecipientsRefused({"r@example.com": (450, b"4.2.1 mailbox busy")}),
])
def test_smtp_retries_a_temporary_refusal_of_the_message(monkeypatch, error):
    events = scripted_smtp(monkeypatch, [("send", error), None])
    waits = []
    smtp_send(waits.append)
    assert events == ["connect", "connect", "accepted"] and waits == [10.0]

def test_smtp_logs_each_retry_without_the_credentials(monkeypatch, caplog):
    scripted_smtp(monkeypatch, [("send", smtplib.SMTPDataError(451, b"4.3.0 try again later")), None])
    with caplog.at_level(logging.WARNING, logger="nasdaq_agent.email.transports"):
        smtp_send(lambda seconds: None, password="s3cret-pw")
    assert "smtp.example.com:587" in caplog.text and "SMTPDataError 451" in caplog.text and "10 s" in caplog.text
    assert "s3cret-pw" not in caplog.text

def test_smtp_counts_an_accepted_message_as_sent_even_if_closing_the_session_fails(monkeypatch):
    """Once the server accepts the message, a failed QUIT cannot undo it, and another attempt would deliver it twice."""
    events = scripted_smtp(monkeypatch, [("quit", smtplib.SMTPResponseException(421, b"4.4.2 closing")), None])
    waits = []
    msg, res = smtp_send(waits.append)
    assert events == ["connect", "accepted"] and waits == []
    assert res.message_id == msg["Message-ID"]

def test_smtp_never_retries_a_connection_lost_while_sending(monkeypatch):
    """The server may already hold the message, so another attempt could deliver it twice."""
    from nasdaq_agent.email.transports import SendError
    events = scripted_smtp(monkeypatch, [("send", smtplib.SMTPServerDisconnected("lost")), None])
    waits = []
    with pytest.raises(SendError):
        smtp_send(waits.append)
    assert events == ["connect"] and waits == []

def test_marker_begin_is_atomic_under_concurrency(tmp_path):
    from nasdaq_agent.email.idempotency import SendMarker
    path = tmp_path / "sent.json"
    wins = []
    def worker():
        if SendMarker(path).begin():
            wins.append(1)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(wins) == 1 and SendMarker(path).state == "sending"

def test_marker_states_and_resume_semantics(tmp_path):
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult
    m = SendMarker(tmp_path / "sent.json")
    assert m.state == "none" and m.begin() is True
    m.complete(SendResult(transport="file", message_id="<x@y>", location=None))
    assert m.state == "sent" and m.read()["message_id"] == "<x@y>"
    crashed = SendMarker(tmp_path / "sent2.json"); crashed.begin()
    assert crashed.state == "sending" and crashed.begin() is False  # a resume must not send again

def test_select_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.config import Settings
    from nasdaq_agent.email.transports import select_transport
    assert select_transport(Settings(_env_file=None), tmp_path).name == "file"

def test_select_transport_smtp_with_starttls_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("AGENT_SMTP_USERNAME", "u")
    monkeypatch.setenv("AGENT_SMTP_PASSWORD", "pw")
    monkeypatch.setenv("AGENT_SMTP_FROM", "bot@example.com")
    monkeypatch.setenv("AGENT_SMTP_STARTTLS", "false")
    from nasdaq_agent.config import Settings
    from nasdaq_agent.email.transports import select_transport
    transport = select_transport(Settings(_env_file=None), tmp_path)
    assert transport.name == "smtp"
    assert transport.starttls is False
