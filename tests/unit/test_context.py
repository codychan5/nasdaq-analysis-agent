def test_context_round_trip(tmp_path):
    from nasdaq_agent.agent.context import RunContext
    ctx = RunContext.new("run-3", "hash", str(tmp_path))
    ctx.progress.session_resolved = True
    ctx.notes.append("hello")
    ctx.save()
    loaded = RunContext.load(tmp_path)
    assert loaded.progress.session_resolved and loaded.notes == ["hello"] and loaded.run_id == "run-3"


def test_context_rejects_unknown_fields():
    import pytest
    from nasdaq_agent.agent.context import RunContext
    with pytest.raises(Exception):
        RunContext(run_id="x", started_at="now", settings_hash="h", artifacts_path=".", bogus=1)


def test_new_contexts_do_not_share_mutable_state():
    # Controller correction 3: every mutable default (Progress, AnalysisInfo, and every
    # list field) must come from Field(default_factory=...), so two RunContext.new()
    # instances never share the same underlying Progress/AnalysisInfo/list object.
    from nasdaq_agent.agent.context import RunContext
    a = RunContext.new("run-a", "hash", "unused-a")
    b = RunContext.new("run-b", "hash", "unused-b")

    a.progress.sent = True
    a.analysis.rejections = 1
    a.notes.append("hello")

    assert b.progress.sent is False
    assert b.analysis.rejections == 0
    assert b.notes == []
