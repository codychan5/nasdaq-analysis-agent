import logging
import smtplib
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Protocol

from ..config import EmailTransport, Settings

log = logging.getLogger(__name__)

SMTP_TIMEOUT_S = 20
SMTP_ATTEMPTS = 3
SMTP_RETRY_WAITS_SECONDS = (10.0, 30.0)  # before the second and the third attempt
SMTP_TEMPORARY_CODES = range(400, 500)  # RFC 5321, section 4.2.1: a 4yz reply is a transient failure


class SendError(RuntimeError):
    pass


class _RetryableSmtpFailure(Exception):
    """An attempt the server kept nothing from: it failed before the message was handed over, or the server refused
    it with a temporary 4xx reply. Trying again cannot deliver the message twice."""

    def __init__(self, cause: Exception):
        super().__init__(type(cause).__name__)
        self.cause = cause


def _temporary_reply(error: Exception) -> bool:
    """The server answered with a 4xx code: it refused this attempt and kept nothing. A refusal of every recipient
    carries one code per recipient, and counts only when all of them are temporary."""
    if isinstance(error, smtplib.SMTPRecipientsRefused):
        codes = [code for code, _ in error.recipients.values()]
        return bool(codes) and all(code in SMTP_TEMPORARY_CODES for code in codes)
    return isinstance(error, smtplib.SMTPResponseException) and error.smtp_code in SMTP_TEMPORARY_CODES


def _retry_cannot_fix(error: Exception) -> bool:
    """A permanent refusal (5xx, such as rejected credentials), a TLS failure, or a feature the server lacks."""
    permanent = isinstance(error, (smtplib.SMTPResponseException, smtplib.SMTPNotSupportedError, ssl.SSLError))
    return permanent and not _temporary_reply(error)


def _failure_label(error: Exception) -> str:
    """The failure's class, with the server's reply code when it gave one. Never the reply text or credentials."""
    code = getattr(error, "smtp_code", None)
    return f"{type(error).__name__} {code}" if code is not None else type(error).__name__


@dataclass(frozen=True)
class SendResult:
    transport: str
    message_id: str
    location: str | None


class Transport(Protocol):
    name: str
    def send(self, msg: EmailMessage) -> SendResult: ...


class SmtpTransport:
    name = "smtp"

    def __init__(self, host: str, port: int, username: str, password: str, timeout: int = SMTP_TIMEOUT_S,
                 starttls: bool = True, retry_sleep: Callable[[float], None] = time.sleep):
        self.host, self.port, self.username, self.password, self.timeout, self.starttls = (
            host, port, username, password, timeout, starttls
        )
        # How the transport waits between attempts. Tests pass a recorder, so they check each wait without sleeping.
        self._retry_sleep = retry_sleep

    def send(self, msg: EmailMessage) -> SendResult:
        for attempt in range(1, SMTP_ATTEMPTS + 1):
            try:
                self._send_once(msg)
            except _RetryableSmtpFailure as failure:
                if attempt == SMTP_ATTEMPTS:
                    raise SendError(f"smtp send failed: {type(failure.cause).__name__}") from failure.cause
                wait = SMTP_RETRY_WAITS_SECONDS[attempt - 1]
                log.warning("SMTP send to %s:%d failed (%s) before the server took the message; retrying in %.0f s "
                            "(attempt %d of %d)", self.host, self.port, _failure_label(failure.cause), wait,
                            attempt, SMTP_ATTEMPTS)
                self._retry_sleep(wait)
                continue
            return SendResult(transport=self.name, message_id=msg["Message-ID"], location=None)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _send_once(self, msg: EmailMessage) -> None:
        sending = accepted = False
        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as smtp:
                if self.starttls:
                    smtp.starttls(context=ssl.create_default_context())
                smtp.login(self.username, self.password)
                sending = True
                smtp.send_message(msg)
                accepted = True
        except (smtplib.SMTPException, OSError) as e:
            if accepted:
                # Only closing the session failed (smtplib's QUIT on leaving the with block). The server has the
                # message, so this counts as sent, and another attempt would deliver it twice.
                return
            # Safe to try again only when the server holds nothing: it said "not now" with a 4xx code, or the
            # attempt failed before the message was handed over. A connection lost while sending is not retried,
            # because the server may already have accepted the message.
            if _temporary_reply(e) or (not sending and not _retry_cannot_fix(e)):
                raise _RetryableSmtpFailure(e) from e
            raise SendError(f"smtp send failed: {type(e).__name__}") from e


class FileTransport:
    name = "file"

    def __init__(self, outbox_dir: Path):
        self.outbox_dir = outbox_dir

    def send(self, msg: EmailMessage) -> SendResult:
        try:
            self.outbox_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = self.outbox_dir / f"{stamp}-{msg['Message-ID'].strip('<>').split('@')[0]}.eml"
            path.write_bytes(bytes(msg))
        except OSError as e:
            # Fix round 1, item 3: transport contract is that every delivery failure surfaces as
            # SendError, not a raw OSError -- e.g. outbox_dir already exists as a regular file.
            raise SendError(f"file outbox write failed: {type(e).__name__}") from e
        return SendResult(transport=self.name, message_id=msg["Message-ID"], location=str(path))


def select_transport(settings: Settings, outbox_dir: Path) -> Transport:
    if settings.resolved_email_transport == EmailTransport.smtp:
        return SmtpTransport(settings.smtp_host, settings.smtp_port, settings.smtp_username,
                             settings.smtp_password.get_secret_value(), starttls=settings.smtp_starttls)
    return FileTransport(outbox_dir)
