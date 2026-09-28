"""Final fix wave A5: send markers are written atomically, and a marker that exists but cannot be read is "unknown" --
treated exactly like "sending" (a send attempt began and its outcome is not known), never as "none" and never as an
exception. The finalize wording for "unknown" is tested in test_finalize.py."""
import errno
import json
import logging
import os
import stat

import pytest

from tests.unit.test_tools_data import ctx, deps, universe  # noqa: F401

# Each is something a crash or a stray edit could leave at the marker path. None is a JSON object with a string
# "state" that the marker itself writes.
UNREADABLE_MARKER_CONTENTS = {
    "empty": "",
    "cut short mid-write": '{"state": "sen',
    "not an object": '["sent"]',
    "a bare string": '"sent"',
    "no state": '{"transport": "file"}',
    "non-string state": '{"state": 1}',
    "unhashable state": '{"state": ["sent"]}',
    "a state the marker never writes": '{"state": "delivered"}',
}


def _write_marker(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.mark.parametrize("contents", list(UNREADABLE_MARKER_CONTENTS.values()), ids=list(UNREADABLE_MARKER_CONTENTS))
def test_an_unreadable_marker_reads_as_unknown_without_raising(tmp_path, contents):
    from nasdaq_agent.email.idempotency import SendMarker
    marker_path = tmp_path / "sent.json"
    _write_marker(marker_path, contents)
    marker = SendMarker(marker_path)
    assert marker.state == "unknown"
    assert marker.read() == {}  # nothing from an unreadable marker is trusted


def test_marker_states_none_sending_sent_failed(tmp_path):
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult
    sent = SendMarker(tmp_path / "sent.json")
    assert sent.state == "none" and sent.read() == {}
    assert sent.begin() is True and sent.state == "sending"
    sent.complete(SendResult(transport="file", message_id="<m@x>", location="outbox/m.eml"))
    assert sent.state == "sent" and sent.read()["message_id"] == "<m@x>" and "started_at" in sent.read()
    failed = SendMarker(tmp_path / "failed.json")
    failed.begin()
    failed.fail("smtp send failed: SMTPServerDisconnected")
    assert failed.state == "failed" and failed.read()["error"] == "smtp send failed: SMTPServerDisconnected"


def test_outcome_unknown_states_are_sending_and_unknown():
    from nasdaq_agent.email.idempotency import OUTCOME_UNKNOWN_STATES
    assert OUTCOME_UNKNOWN_STATES == frozenset({"sending", "unknown"})


def test_update_leaves_no_temp_file_behind(tmp_path):
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult
    marker = SendMarker(tmp_path / "sent.json")
    marker.begin()
    marker.complete(SendResult(transport="file", message_id="<m@x>", location=None))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sent.json"]
    failed = SendMarker(tmp_path / "failure_sent.json")
    failed.begin()
    failed.fail("boom")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["failure_sent.json", "sent.json"]


def test_update_that_cannot_replace_the_marker_keeps_the_old_one_and_removes_its_temp_file(tmp_path, monkeypatch):
    """os.replace is the commit point: if it fails, the marker still holds its previous, complete content and the
    temp file is gone."""
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult
    marker = SendMarker(tmp_path / "sent.json")
    marker.begin()

    def replace_fails(src, dst):
        raise OSError("simulated: no space left on device")

    monkeypatch.setattr(os, "replace", replace_fails)
    with pytest.raises(OSError, match="no space left"):
        marker.complete(SendResult(transport="file", message_id="<m@x>", location=None))
    monkeypatch.undo()
    assert marker.state == "sending"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sent.json"]


def test_update_fsyncs_a_temp_file_in_the_same_directory_before_replacing(tmp_path, monkeypatch):
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult
    marker_path = tmp_path / "sent.json"
    marker = SendMarker(marker_path)
    marker.begin()
    events: list[tuple] = []
    real_fsync, real_replace = os.fsync, os.replace

    def recording_fsync(fd):
        events.append(("fsync",))
        return real_fsync(fd)

    def recording_replace(src, dst):
        events.append(("replace", os.fspath(src), os.fspath(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    marker.complete(SendResult(transport="file", message_id="<m@x>", location=None))

    kinds = [event[0] for event in events]
    assert "fsync" in kinds and kinds.count("replace") == 1
    assert kinds.index("fsync") < kinds.index("replace")  # the content is durable before it becomes the marker
    _, src, dst = events[kinds.index("replace")]
    assert dst == os.fspath(marker_path) and os.path.dirname(src) == os.fspath(tmp_path)
    assert json.loads(marker_path.read_text())["state"] == "sent"


def test_begin_stays_exclusive_and_fsyncs_its_content(tmp_path, monkeypatch):
    from nasdaq_agent.email.idempotency import SendMarker
    fsynced_files: list[int] = []  # regular files only: begin() also fsyncs the directory, tested separately
    real_fsync = os.fsync

    def recording_fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            fsynced_files.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    marker = SendMarker(tmp_path / "sent.json")
    assert marker.begin() is True
    assert fsynced_files, "begin() must fsync the marker it creates"
    assert marker.begin() is False and SendMarker(tmp_path / "sent.json").begin() is False
    assert marker.state == "sending"


def test_begin_and_update_fsync_the_markers_directory(tmp_path, monkeypatch):
    """Follow-up: fsync on a file makes its content durable, but the entry naming it lives in the directory. A power
    loss right after begin() must not lose the marker, or a resume could send twice, so begin() fsyncs the marker's
    directory after the file, and an update fsyncs it after os.replace. Each fsynced descriptor is identified with
    fstat while it is still open."""
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult
    directory = os.stat(tmp_path)
    marker_directory = (directory.st_dev, directory.st_ino)
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def recording_fsync(fd):
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode):
            events.append("fsync file")
        else:
            events.append("fsync marker directory" if (st.st_dev, st.st_ino) == marker_directory else "fsync other directory")
        return real_fsync(fd)

    def recording_replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    marker = SendMarker(tmp_path / "sent.json")

    assert marker.begin() is True
    assert events == ["fsync file", "fsync marker directory"]
    events.clear()
    marker.complete(SendResult(transport="file", message_id="<m@x>", location=None))
    assert events == ["fsync file", "replace", "fsync marker directory"]
    events.clear()
    marker.fail("a later failure, only to exercise fail()")
    assert events == ["fsync file", "replace", "fsync marker directory"]


@pytest.mark.parametrize("refused", ["open", "fsync"])
def test_a_directory_that_cannot_be_fsynced_never_fails_the_marker_write(tmp_path, monkeypatch, caplog, refused):
    """Follow-up: the directory fsync is best-effort. Some filesystems reject fsync on a directory, or opening one; the
    failure is logged at debug level and the write stands, since the file itself is already fsynced."""
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult
    real_open, real_fsync = os.open, os.fsync

    def open_refusing_the_directory(path, flags, *args, **kwargs):
        if refused == "open" and os.fspath(path) == os.fspath(tmp_path):
            raise PermissionError(errno.EACCES, "Permission denied", os.fspath(path))
        return real_open(path, flags, *args, **kwargs)

    def fsync_refusing_directories(fd):
        if refused == "fsync" and stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "Invalid argument")
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", open_refusing_the_directory)
    monkeypatch.setattr(os, "fsync", fsync_refusing_directories)
    marker = SendMarker(tmp_path / "sent.json")
    with caplog.at_level(logging.DEBUG, logger="nasdaq_agent.email"):
        assert marker.begin() is True
        marker.complete(SendResult(transport="file", message_id="<m@x>", location=None))

    assert marker.state == "sent" and marker.read()["message_id"] == "<m@x>"
    debug = [r for r in caplog.records if r.name == "nasdaq_agent.email" and r.levelno == logging.DEBUG]
    assert len(debug) == 2  # one for begin(), one for the update
    assert all(os.fspath(tmp_path) in r.getMessage() for r in debug)


def test_begin_refuses_when_an_unreadable_marker_already_exists(tmp_path):
    """An empty marker is what a crash between begin()'s exclusive create and its write leaves: a send may have
    started, so a new attempt must not."""
    from nasdaq_agent.email.idempotency import SendMarker
    _write_marker(tmp_path / "sent.json", "")
    assert SendMarker(tmp_path / "sent.json").begin() is False


class _MustNotSend:
    name = "must-not-send"

    def send(self, msg):
        raise AssertionError("send_email must not reach the transport when the marker's outcome is unknown")


def test_send_email_refuses_an_unknown_marker_exactly_as_a_sending_one(ctx, deps):
    from nasdaq_agent.agent.tools.send import SENT_MARKER, make_send_email
    from nasdaq_agent.email.idempotency import SendMarker
    ctx.progress.composed = True  # the marker is checked before anything about the composed report is read
    deps.transport = _MustNotSend()
    marker_path = deps.run_dir.path / SENT_MARKER
    send_email = make_send_email(ctx, deps)

    _write_marker(marker_path, "")
    unknown = send_email.invoke({})
    assert marker_path.read_text() == ""  # left exactly as found
    marker_path.unlink()
    SendMarker(marker_path).begin()
    sending = send_email.invoke({})

    assert unknown == sending
    assert unknown.startswith("ERROR") and "unknown state" in unknown and "give_up" in unknown
    assert not ctx.progress.sent and not list(deps.run_dir.outbox_dir.glob("*.eml"))


def test_reconcile_delivery_never_reconciles_from_unknown_and_never_raises(ctx, deps):
    from nasdaq_agent.agent.finalize import reconcile_delivery
    from nasdaq_agent.agent.tools.send import SENT_MARKER
    for contents in UNREADABLE_MARKER_CONTENTS.values():
        _write_marker(deps.run_dir.path / SENT_MARKER, contents)
        assert reconcile_delivery(ctx, deps.run_dir) is False
        assert ctx.progress.sent is False and ctx.email is None and ctx.notes == []


def test_reconcile_delivery_takes_only_string_fields_from_a_sent_marker(ctx, deps):
    """The marker is a file on disk: a "sent" marker's other fields are copied into ctx.email only when they are
    strings, so a stray value can never make reconciliation raise."""
    from nasdaq_agent.agent.finalize import reconcile_delivery
    from nasdaq_agent.agent.tools.send import SENT_MARKER
    _write_marker(deps.run_dir.path / SENT_MARKER,
                  json.dumps({"state": "sent", "transport": 7, "message_id": "<m@x>", "location": ["x"]}))
    assert reconcile_delivery(ctx, deps.run_dir) is True
    assert ctx.progress.sent is True
    assert ctx.email.status == "sent" and ctx.email.transport is None and ctx.email.message_id == "<m@x>"
    assert ctx.email.location is None
