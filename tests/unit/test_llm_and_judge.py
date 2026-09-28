def test_build_chat_model_uses_prefixed_string(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_LLM_MODEL", "google_genai:gemini-3.5-flash-lite")
    captured = {}
    def fake_init(model, **kw):
        captured["model"], captured["kw"] = model, kw
        class M:
            def with_fallbacks(self, others): captured["fallbacks"] = others; return self
        return M()
    from nasdaq_agent.agent import llm
    monkeypatch.setattr(llm, "init_chat_model", fake_init)
    from nasdaq_agent.config import Settings
    llm.build_chat_model(Settings(_env_file=None), "orchestrator")
    assert captured["model"] == "google_genai:gemini-3.5-flash-lite" and captured["kw"]["temperature"] == 0

def test_build_chat_model_rejects_unprefixed(monkeypatch):
    import pytest
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_LLM_MODEL", "gemini-3.5-flash-lite")
    from nasdaq_agent.agent import llm
    from nasdaq_agent.config import Settings
    with pytest.raises(ValueError):
        llm.build_chat_model(Settings(_env_file=None), "orchestrator")

def test_build_chat_model_sends_temperature_only_when_supported(monkeypatch):
    # Controller correction 4: current Anthropic models reject sampling parameters with an
    # HTTP 400, so a model string starting with "anthropic:" must get no temperature kwarg;
    # every other provider still gets temperature=0. The same rule applies to the fallback
    # model, so a fallback that is itself an "anthropic:" model also gets no temperature.
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    captured = {}
    def fake_init(model, **kw):
        captured[model] = kw
        class M:
            def with_fallbacks(self, others):
                captured[f"{model}:fallbacks"] = others
                return self
        return M()
    from nasdaq_agent.agent import llm
    monkeypatch.setattr(llm, "init_chat_model", fake_init)
    from nasdaq_agent.config import Settings

    monkeypatch.setenv("AGENT_LLM_MODEL", "anthropic:claude-opus-5")
    monkeypatch.delenv("AGENT_LLM_FALLBACK_MODEL", raising=False)
    llm.build_chat_model(Settings(_env_file=None), "orchestrator")

    monkeypatch.setenv("AGENT_LLM_MODEL", "google_genai:gemini-3.5-flash-lite")
    monkeypatch.setenv("AGENT_LLM_FALLBACK_MODEL", "anthropic:claude-haiku-5")
    llm.build_chat_model(Settings(_env_file=None), "orchestrator")

    assert "temperature" not in captured["anthropic:claude-opus-5"]
    assert captured["google_genai:gemini-3.5-flash-lite"]["temperature"] == 0
    assert "temperature" not in captured["anthropic:claude-haiku-5"]

def test_default_model_runs_on_openrouter_through_the_openai_client(monkeypatch):
    # The default model is a paid model on OpenRouter, reachable from Hong Kong where Gemini and Claude are
    # region-restricted. Everything after "openrouter:" is OpenRouter's own model id, sent through LangChain's OpenAI
    # client at OpenRouter's endpoint with the key from Settings.
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("AGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("AGENT_LLM_FALLBACK_MODEL", raising=False)
    captured = {}
    def fake_init(model, **kw):
        captured["model"], captured["kw"] = model, kw
        return object()
    from nasdaq_agent.agent import llm
    monkeypatch.setattr(llm, "init_chat_model", fake_init)
    from nasdaq_agent.config import Settings
    llm.build_chat_model(Settings(_env_file=None), "orchestrator")
    assert captured["model"] == "z-ai/glm-5.3"
    assert captured["kw"]["model_provider"] == "openai"
    assert captured["kw"]["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["kw"]["api_key"].get_secret_value() == "or-test-key"
    assert captured["kw"]["temperature"] == 0

def test_openrouter_fallback_model_is_built_the_same_way(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("AGENT_LLM_MODEL", "openrouter:qwen/qwen3.8-27b:free")
    monkeypatch.setenv("AGENT_LLM_FALLBACK_MODEL", "openrouter:nvidia/nemotron-3-super-120b-a12b:free")
    calls = []
    def fake_init(model, **kw):
        calls.append((model, kw))
        class M:
            def with_fallbacks(self, others): return self
        return M()
    from nasdaq_agent.agent import llm
    monkeypatch.setattr(llm, "init_chat_model", fake_init)
    from nasdaq_agent.config import Settings
    llm.build_chat_model(Settings(_env_file=None), "orchestrator")
    assert [model for model, _ in calls] == ["qwen/qwen3.8-27b:free", "nvidia/nemotron-3-super-120b-a12b:free"]
    assert all(kw["base_url"] == "https://openrouter.ai/api/v1" and kw["model_provider"] == "openai" for _, kw in calls)

def test_openrouter_builds_a_real_client_offline(monkeypatch):
    # No network: building the client only configures it. Proves the colon in a ":free" model id survives
    # LangChain's provider parsing and the endpoint and key land on the real object.
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("AGENT_LLM_MODEL", "openrouter:qwen/qwen3.8-27b:free")
    monkeypatch.delenv("AGENT_LLM_FALLBACK_MODEL", raising=False)
    from nasdaq_agent.agent import llm
    from nasdaq_agent.config import Settings
    model = llm.build_chat_model(Settings(_env_file=None), "orchestrator")
    assert model.model_name == "qwen/qwen3.8-27b:free"
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.openai_api_key.get_secret_value() == "or-test-key"

def test_judge_prompt_and_parsing():
    from nasdaq_agent.report.judge import make_judge, JUDGE_PROMPT
    from nasdaq_agent.report.schemas import JudgeVerdict, JudgeIssue
    from tests.unit.test_grounding import verified, headlines, good_narrative
    seen = {}
    class FakeStructured:
        def invoke(self, messages):
            seen["messages"] = messages
            return JudgeVerdict(faithful=False, issues=[JudgeIssue(sentence="x", problem="y")])
    class FakeModel:
        def with_structured_output(self, schema): assert schema is JudgeVerdict; return FakeStructured()
    verdict = make_judge(FakeModel())(verified(), headlines(), good_narrative())
    assert verdict.faithful is False and verdict.issues[0].problem == "y"
    text = str(seen["messages"])
    assert "not instructions" in JUDGE_PROMPT and "Acme files for FDA review" in text and "11.0" in text

def test_judge_includes_extra_facts_when_present():
    # Review fix (Important): the judge previously never saw extra_facts, so it could flag a
    # session-move or closing-price sentence that grounding had already allowed as a lie.
    from nasdaq_agent.report.judge import make_judge
    from nasdaq_agent.report.schemas import JudgeVerdict
    from tests.unit.test_grounding import verified, headlines, good_narrative
    seen = {}
    class FakeStructured:
        def invoke(self, messages):
            seen["messages"] = messages
            return JudgeVerdict(faithful=True)
    class FakeModel:
        def with_structured_output(self, schema): return FakeStructured()
    make_judge(FakeModel())(verified(), headlines(), good_narrative(), extra_facts={"session_pct_change": 3.738, "close": 11.10})
    text = str(seen["messages"])
    assert "session_pct_change" in text and "3.738" in text

def test_judge_omits_extra_facts_key_when_absent():
    from nasdaq_agent.report.judge import make_judge
    from nasdaq_agent.report.schemas import JudgeVerdict
    from tests.unit.test_grounding import verified, headlines, good_narrative
    seen = {}
    class FakeStructured:
        def invoke(self, messages):
            seen["messages"] = messages
            return JudgeVerdict(faithful=True)
    class FakeModel:
        def with_structured_output(self, schema): return FakeStructured()
    make_judge(FakeModel())(verified(), headlines(), good_narrative())
    assert "extra_facts" not in str(seen["messages"])


def _judge_messages(**judge_kwargs):
    """The messages the judge sends for the shared good narrative, captured by a fake structured model."""
    from nasdaq_agent.report.judge import make_judge
    from nasdaq_agent.report.schemas import JudgeVerdict
    from tests.unit.test_grounding import verified, headlines, good_narrative
    seen = {}
    class FakeStructured:
        def invoke(self, messages):
            seen["messages"] = messages
            return JudgeVerdict(faithful=True)
    class FakeModel:
        def with_structured_output(self, schema): return FakeStructured()
    make_judge(FakeModel())(verified(), headlines(), good_narrative(), **judge_kwargs)
    return str(seen["messages"])


def test_judge_checks_the_headline_notes_against_publication_times_in_eastern_time():
    from tests.unit.test_grounding import good_notes
    text = _judge_messages()
    assert "HEADLINE NOTES" in text and good_notes()[1].summary in text and "[2] " in text
    # The fixture headlines were published at midnight UTC on the 24th: 20:00 on the 23rd in New York.
    assert "2026-09-23T20:00-04:00" in text


def test_judge_sees_the_session_facts_when_given():
    text = _judge_messages(session_facts={"symbol": "ACME", "company": "Acme Corp", "session_date": "2026-09-24",
                                          "session_label": "official close", "role": "top NASDAQ gainer of the session"})
    assert "top NASDAQ gainer of the session" in text and "2026-09-24" in text and "Acme Corp" in text


def test_judge_omits_the_session_key_when_no_session_facts_are_given():
    assert '"session"' not in _judge_messages()


def test_judge_sees_each_headline_source_and_story_notes_with_all_their_ids():
    from nasdaq_agent.report.judge import make_judge
    from nasdaq_agent.report.schemas import HeadlineNote, JudgeVerdict
    from tests.unit.test_grounding import verified, multi_source_headlines, good_narrative
    seen = {}
    class FakeStructured:
        def invoke(self, messages):
            seen["messages"] = messages
            return JudgeVerdict(faithful=True)
    class FakeModel:
        def with_structured_output(self, schema): return FakeStructured()
    notes = [HeadlineNote(headline_ids=[2, 4], summary="Clearance news."), HeadlineNote(headline_ids=[1, 3], summary="Offering news.")]
    make_judge(FakeModel())(verified(), multi_source_headlines(), good_narrative().model_copy(update={"headline_notes": notes}))
    text = str(seen["messages"])
    assert "[1, 3] Offering news." in text and '"source": "yfinance"' in text


def test_judge_sees_each_article_summary_and_checks_story_summaries_against_it():
    from nasdaq_agent.report.judge import JUDGE_PROMPT, make_judge
    from nasdaq_agent.report.schemas import JudgeVerdict
    from tests.unit.test_grounding import verified, summarised_headlines, good_narrative
    seen = {}
    class FakeStructured:
        def invoke(self, messages):
            seen["messages"] = messages
            return JudgeVerdict(faithful=True)
    class FakeModel:
        def with_structured_output(self, schema): return FakeStructured()
    make_judge(FakeModel())(verified(), summarised_headlines(), good_narrative())
    assert "involving up to 48% of Tarbagatay Munay" in str(seen["messages"])
    assert "titles and summaries" in " ".join(JUDGE_PROMPT.split())


def test_judge_prompt_checks_the_consolidation_both_ways():
    from nasdaq_agent.report.judge import JUDGE_PROMPT
    prompt = " ".join(JUDGE_PROMPT.split())
    assert "groups headlines reporting different information" in prompt
    assert "report the same information but sit in separate notes" in prompt


def test_judge_prompt_allows_hedged_relevance_but_not_invented_causes_or_wrong_timing():
    from nasdaq_agent.report.judge import JUDGE_PROMPT
    prompt = " ".join(JUDGE_PROMPT.split())
    assert "headline note" in prompt and "hedged reasoning" in prompt
    assert "states a cause as fact" in prompt and "timing" in prompt


def test_every_provider_gets_the_model_call_timeout_and_retries(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("AGENT_LLM_MAX_RETRIES", "1")
    monkeypatch.delenv("AGENT_LLM_FALLBACK_MODEL", raising=False)
    captured = {}
    def fake_init(model, **kw):
        captured[model] = kw
        return object()
    from nasdaq_agent.agent import llm
    monkeypatch.setattr(llm, "init_chat_model", fake_init)
    from nasdaq_agent.config import Settings
    for model_string in ("openrouter:qwen/qwen3.8-27b:free", "google_genai:gemini-3.5-flash-lite", "anthropic:claude-opus-5"):
        monkeypatch.setenv("AGENT_LLM_MODEL", model_string)
        llm.build_chat_model(Settings(_env_file=None), "orchestrator")
    assert len(captured) == 3
    assert all(kw["timeout"] == 45 and kw["max_retries"] == 1 for kw in captured.values())
    assert all(kw["max_tokens"] == 8192 for kw in captured.values())


def test_real_clients_accept_the_timeout_and_retries(monkeypatch):
    # No network: construction only. Each provider's client must take the same two arguments.
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    monkeypatch.delenv("AGENT_LLM_FALLBACK_MODEL", raising=False)
    from nasdaq_agent.agent import llm
    from nasdaq_agent.config import Settings
    monkeypatch.setenv("AGENT_LLM_MODEL", "openrouter:qwen/qwen3.8-27b:free")
    openrouter = llm.build_chat_model(Settings(_env_file=None), "orchestrator")
    assert openrouter.request_timeout == 90 and openrouter.max_retries == 2
    monkeypatch.setenv("AGENT_LLM_MODEL", "google_genai:gemini-3.5-flash-lite")
    google = llm.build_chat_model(Settings(_env_file=None), "orchestrator")
    assert google.timeout == 90 and google.max_retries == 2
    monkeypatch.setenv("AGENT_LLM_MODEL", "anthropic:claude-opus-5")
    anthropic = llm.build_chat_model(Settings(_env_file=None), "orchestrator")
    assert anthropic.default_request_timeout == 90 and anthropic.max_retries == 2
    assert openrouter.max_tokens == 8192 and google.max_output_tokens == 8192 and anthropic.max_tokens == 8192


def _fake_openai_server(slow_seconds):
    """A local stand-in for OpenRouter: "slow/model" answers only after slow_seconds; any other model replies at once.
    Returns the server and an event set once the slow model has finished producing its (late) answer."""
    import json
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    slow_finished = threading.Event()

    def completion(content):
        return json.dumps({"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                           "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                        "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            slow = body["model"] == "slow/model"
            if slow:
                time.sleep(slow_seconds)
            payload = completion("late answer from the slow model" if slow else "answer from the fallback")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client already gave up on this request and closed the connection
            finally:
                if slow:
                    slow_finished.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, slow_finished


def _point_openrouter_at(monkeypatch, server, model, fallback, timeout="0.5", retries="0"):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("AGENT_LLM_MODEL", model)
    if fallback:
        monkeypatch.setenv("AGENT_LLM_FALLBACK_MODEL", fallback)
    else:
        monkeypatch.delenv("AGENT_LLM_FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", timeout)
    monkeypatch.setenv("AGENT_LLM_MAX_RETRIES", retries)
    from nasdaq_agent.agent import llm
    monkeypatch.setattr(llm, "OPENROUTER_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    return llm


def test_a_timed_out_call_is_abandoned_and_its_late_answer_is_never_used(monkeypatch):
    # The question this answers: what happens if a model call times out and its answer arrives later? The client
    # closes the connection at the timeout and moves on to the fallback model, so the late answer is never delivered;
    # the agent only ever sees a reply it received in time.
    import time
    server, slow_finished = _fake_openai_server(slow_seconds=2.0)
    try:
        llm = _point_openrouter_at(monkeypatch, server, "openrouter:slow/model", "openrouter:fast/model")
        from nasdaq_agent.config import Settings
        started = time.monotonic()
        reply = llm.build_chat_model(Settings(_env_file=None), "orchestrator").invoke("hello")
        waited = time.monotonic() - started
        assert reply.content == "answer from the fallback"
        assert waited < 2.0  # it did not wait for the slow model
        assert slow_finished.wait(5)  # the slow model did finish its answer later ...
        assert reply.content == "answer from the fallback"  # ... and that late answer never replaced this one
    finally:
        server.shutdown()


def test_a_timed_out_call_without_a_fallback_fails_fast(monkeypatch):
    # With no fallback the timeout surfaces as an error within the bound; the agent node turns it into a failure notice.
    import time
    import pytest
    from openai import APITimeoutError
    server, _ = _fake_openai_server(slow_seconds=3.0)
    try:
        llm = _point_openrouter_at(monkeypatch, server, "openrouter:slow/model", None, timeout="0.5", retries="1")
        from nasdaq_agent.config import Settings
        started = time.monotonic()
        with pytest.raises(APITimeoutError):
            llm.build_chat_model(Settings(_env_file=None), "orchestrator").invoke("hello")
        assert time.monotonic() - started < 3.0  # two attempts of 0.5 s plus the client's short retry backoff
    finally:
        server.shutdown()


def _in_body_error_server(failures, code):
    """A stand-in for OpenRouter that answers HTTP 200 with an error object in the body for the first `failures`
    requests, as OpenRouter does when an upstream provider is overloaded, then a normal completion."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    calls = {"count": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls["count"] += 1
            if calls["count"] <= failures:
                body = {"error": {"message": "Upstream error: Service temporarily overloaded", "code": code,
                                  "metadata": {"error_type": "provider_overloaded"}}}
            else:
                body = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "recovered"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls


def test_an_overload_reported_inside_a_normal_reply_is_retried(monkeypatch):
    # Live runs: OpenRouter reported "provider_overloaded" (503) inside an HTTP 200 reply, which the client's own
    # retries never saw. Transient codes in the body are now treated like the HTTP status they name, so they retry.
    server, calls = _in_body_error_server(failures=2, code=503)
    try:
        llm = _point_openrouter_at(monkeypatch, server, "openrouter:some/model", None, timeout="5", retries="2")
        from nasdaq_agent.config import Settings
        reply = llm.build_chat_model(Settings(_env_file=None), "orchestrator").invoke("hello")
        assert reply.content == "recovered" and calls["count"] == 3
    finally:
        server.shutdown()


def test_a_non_transient_error_inside_a_reply_is_not_retried(monkeypatch):
    import pytest
    server, calls = _in_body_error_server(failures=5, code=400)
    try:
        llm = _point_openrouter_at(monkeypatch, server, "openrouter:some/model", None, timeout="5", retries="2")
        from nasdaq_agent.config import Settings
        with pytest.raises(Exception):
            llm.build_chat_model(Settings(_env_file=None), "orchestrator").invoke("hello")
        assert calls["count"] == 1
    finally:
        server.shutdown()
