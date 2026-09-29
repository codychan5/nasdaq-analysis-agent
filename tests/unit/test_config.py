import pytest
from pydantic import ValidationError

def test_defaults_and_prefix(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    from nasdaq_agent.config import Settings, EmailTransport
    s = Settings(_env_file=None)
    assert s.llm_model == "openrouter:z-ai/glm-5.3"
    assert s.max_model_calls == 25 and s.max_tool_calls == 25 and s.max_code_runs == 6
    assert s.resolved_email_transport == EmailTransport.file
    assert s.resolved_judge_model == s.llm_model

def test_unknown_key_in_env_file_rejected(tmp_path):
    env = tmp_path / ".env"
    env.write_text("AGENT_EMAIL_TO=reports@example.com\nAGENT_BOGUS=1\n")
    from nasdaq_agent.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=env)

def test_secret_not_in_repr(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv("AGENT_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("AGENT_SMTP_USERNAME", "u")
    monkeypatch.setenv("AGENT_SMTP_PASSWORD", "hunter2")
    monkeypatch.setenv("AGENT_SMTP_FROM", "bot@example.com")
    from nasdaq_agent.config import Settings, EmailTransport
    s = Settings(_env_file=None)
    assert "hunter2" not in repr(s) and "hunter2" not in str(s.model_dump())
    assert s.resolved_email_transport == EmailTransport.smtp

def test_smtp_host_requires_credentials(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv("AGENT_SMTP_HOST", "smtp.example.com")
    from nasdaq_agent.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_provider_keys_read_without_prefix(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-123")
    from nasdaq_agent.config import Settings
    s = Settings(_env_file=None)
    assert s.google_api_key.get_secret_value() == "g-123"

def test_blank_smtp_credentials_are_rejected(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv("AGENT_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("AGENT_SMTP_USERNAME", "")
    monkeypatch.setenv("AGENT_SMTP_PASSWORD", "")
    monkeypatch.setenv("AGENT_SMTP_FROM", "")
    from nasdaq_agent.config import Settings, EmailTransport
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    # Also test that empty SMTP_HOST resolves to file transport
    monkeypatch.delenv("AGENT_SMTP_HOST")
    monkeypatch.delenv("AGENT_SMTP_USERNAME")
    monkeypatch.delenv("AGENT_SMTP_PASSWORD")
    monkeypatch.delenv("AGENT_SMTP_FROM")
    monkeypatch.setenv("AGENT_SMTP_HOST", "")
    s = Settings(_env_file=None)
    assert s.resolved_email_transport == EmailTransport.file

def test_invalid_email_to_rejected(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "not-an-email")
    from nasdaq_agent.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_email_to_header_injection_rejected(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com\nBcc: x@y.com")
    from nasdaq_agent.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_invalid_smtp_from_rejected_when_smtp_configured(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv("AGENT_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("AGENT_SMTP_USERNAME", "u")
    monkeypatch.setenv("AGENT_SMTP_PASSWORD", "pw")
    monkeypatch.setenv("AGENT_SMTP_FROM", "not-an-email")
    from nasdaq_agent.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_validation_error_hides_sensitive_input(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "not-an-email")
    monkeypatch.setenv("AGENT_SMTP_PASSWORD", "hunter2")
    from nasdaq_agent.config import Settings
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)
    message = str(exc_info.value)
    assert "not-an-email" not in message
    assert "hunter2" not in message

def test_secret_field_names_matches_current_secretstr_fields():
    """The secret field names are introspected, not a second hand-written list -- so a future
    SecretStr field is picked up automatically instead of silently missing redaction."""
    from nasdaq_agent.config import Settings
    assert Settings.secret_field_names() == frozenset({
        "smtp_password", "google_api_key", "anthropic_api_key",
        "massive_api_key", "alphavantage_api_key", "langsmith_api_key", "openrouter_api_key", "sec_user_agent",
    })

def test_secret_values_returns_only_set_secrets(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv("AGENT_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("AGENT_SMTP_USERNAME", "u")
    monkeypatch.setenv("AGENT_SMTP_PASSWORD", "hunter2")
    monkeypatch.setenv("AGENT_SMTP_FROM", "bot@example.com")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-123")
    from nasdaq_agent.config import Settings
    s = Settings(_env_file=None)
    assert set(s.secret_values()) == {"hunter2", "g-123"}

def test_run_deadline_defaults_to_600_seconds_and_reads_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    from nasdaq_agent.config import Settings
    assert Settings(_env_file=None).run_deadline_seconds == 600
    monkeypatch.setenv("AGENT_RUN_DEADLINE_SECONDS", "30")
    assert Settings(_env_file=None).run_deadline_seconds == 30

@pytest.mark.parametrize("value", ["0", "-5", "1.5", "ten"])
def test_run_deadline_must_be_a_positive_integer(monkeypatch, value):
    """The deadline middleware reads this, so a zero, negative or fractional deadline is refused
    when settings load rather than ending every run before its first model call."""
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv("AGENT_RUN_DEADLINE_SECONDS", value)
    from nasdaq_agent.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_quality_preset_setting_is_gone(tmp_path):
    """The opt-in quality-preset extension was never built, so its setting is removed rather than
    left inert. Settings rejects unknown keys, so an .env that still sets it now fails to load (a breaking change for
    an .env copied from an older .env.example)."""
    from nasdaq_agent import config
    assert "quality_preset" not in config.Settings.model_fields and not hasattr(config, "QualityPreset")
    env = tmp_path / ".env"
    env.write_text("AGENT_EMAIL_TO=reports@example.com\nAGENT_QUALITY_PRESET=off\n")
    with pytest.raises(ValidationError):
        config.Settings(_env_file=env)


def test_model_call_timeout_and_retries_have_defaults_and_are_validated(monkeypatch):
    # A hung model call must not stall a run: each call times out on our side and retries a bounded number of times.
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.config import Settings
    s = Settings(_env_file=None)
    assert s.llm_timeout_seconds == 90 and s.llm_max_retries == 2
    for name, bad in (("AGENT_LLM_TIMEOUT_SECONDS", "0"), ("AGENT_LLM_TIMEOUT_SECONDS", "-5"), ("AGENT_LLM_MAX_RETRIES", "-1")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)
        monkeypatch.delenv(name)


def test_model_output_cap_has_a_default_and_is_validated(monkeypatch):
    # OpenRouter reserves credit for the largest reply a model may give; an uncapped call failed a live run with
    # "requires more credits, or fewer max_tokens". Reasoning models think within the cap: GLM-5.3's compose_report
    # replies reached 7,920 of 8,192 tokens, and one run failed after three replies were cut off at 8,192.
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.config import Settings
    assert Settings(_env_file=None).llm_max_output_tokens == 16384
    monkeypatch.setenv("AGENT_LLM_MAX_OUTPUT_TOKENS", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_news_lookback_days_defaults_to_three_and_is_bounded(monkeypatch):
    # How far back before the session get_news looks for headlines: the original three days by default, a setting so a
    # thinly covered gainer can be given longer, bounded so a typo cannot ask for years of news.
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.config import Settings
    assert Settings(_env_file=None).news_lookback_days == 3
    for bad in ("0", "91"):
        monkeypatch.setenv("AGENT_NEWS_LOOKBACK_DAYS", bad)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)
    monkeypatch.setenv("AGENT_NEWS_LOOKBACK_DAYS", "30")
    assert Settings(_env_file=None).news_lookback_days == 30


def test_the_schedule_defaults_to_1630_new_york_time_on_weekdays(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.config import Settings
    settings = Settings(_env_file=None)
    assert settings.schedule_cron == "30 16 * * 1-5" and settings.schedule_timezone == "America/New_York"


def test_the_schedule_is_configurable_from_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SCHEDULE_CRON", "0 18 * * mon-fri")
    monkeypatch.setenv("AGENT_SCHEDULE_TIMEZONE", "Asia/Hong_Kong")
    from nasdaq_agent.config import Settings
    settings = Settings(_env_file=None)
    assert settings.schedule_cron == "0 18 * * mon-fri" and settings.schedule_timezone == "Asia/Hong_Kong"


@pytest.mark.parametrize("name, value", [("AGENT_SCHEDULE_CRON", "30 16 * *"), ("AGENT_SCHEDULE_CRON", "0 0 30 2 *"),
                                         ("AGENT_SCHEDULE_CRON", "0 9 1 * 1"), ("AGENT_SCHEDULE_TIMEZONE", "Mars/Olympus")])
def test_an_invalid_schedule_fails_at_startup_not_at_the_first_trigger(monkeypatch, name, value):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv(name, value)
    from nasdaq_agent.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


SEC_CONTACT = " jane@example.com"


def test_sec_user_agent_is_off_by_default_and_a_blank_value_leaves_it_off(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    from nasdaq_agent.config import Settings
    assert Settings(_env_file=None).sec_user_agent is None
    monkeypatch.setenv("AGENT_SEC_USER_AGENT", "   ")
    assert Settings(_env_file=None).sec_user_agent is None


@pytest.mark.parametrize("value", [
    "Jane Doe",                                  # no contact: SEC's fair-access policy asks for an email address
    "Jane Doe jane@example.com\r\nX-Extra: 1",   # a line break would start a second header
    "Jane Doé jane@example.com",                 # HTTP headers carry ASCII only
    "J" * (201 - len(SEC_CONTACT)) + SEC_CONTACT,  # one character over the limit
])
def test_sec_user_agent_must_be_one_header_safe_line_that_names_a_contact(monkeypatch, value):
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv("AGENT_SEC_USER_AGENT", value)
    from nasdaq_agent.config import Settings
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_sec_user_agent_up_to_the_limit_is_kept_trimmed_and_treated_as_sensitive(monkeypatch):
    # It carries a person's email address, so it is held like a key: hidden from reprs and redacted from logs.
    longest = "J" * (200 - len(SEC_CONTACT)) + SEC_CONTACT
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv("AGENT_SEC_USER_AGENT", f"  {longest}  ")
    from nasdaq_agent.config import Settings
    s = Settings(_env_file=None)
    assert s.sec_user_agent.get_secret_value() == longest
    assert "jane@example.com" not in repr(s) and longest in s.secret_values()

@pytest.mark.parametrize("field", ["AGENT_EMAIL_TO", "AGENT_SMTP_FROM"])
def test_an_email_address_with_a_trailing_newline_is_rejected(monkeypatch, field):
    """A newline makes the address header invalid, so every send, the failure notice included, would fail."""
    from nasdaq_agent.config import Settings
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv(field, "ops@example.com\n")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

@pytest.mark.parametrize("value", ["4", "10", "0", "-1"])
def test_the_lookback_must_match_the_five_return_metrics(monkeypatch, value):
    """The metrics and their verifier are defined over exactly five daily returns, so any other window fails every run."""
    from nasdaq_agent.config import Settings
    monkeypatch.setenv("AGENT_EMAIL_TO", "reports@example.com")
    monkeypatch.setenv("AGENT_LOOKBACK_SESSIONS", value)
    with pytest.raises(ValidationError, match="lookback_sessions must be 5"):
        Settings(_env_file=None)


def test_session_date_is_empty_by_default_and_blank_means_empty(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.config import Settings
    assert Settings(_env_file=None).session_date is None
    monkeypatch.setenv("AGENT_SESSION_DATE", "  ")
    assert Settings(_env_file=None).session_date is None


def test_session_date_reads_a_calendar_date(monkeypatch):
    from datetime import date
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SESSION_DATE", "2026-07-08")
    from nasdaq_agent.config import Settings
    assert Settings(_env_file=None).session_date == date(2026, 7, 8)


@pytest.mark.parametrize("bad", ["07/08/2026", "2026-7-8", "1751932800", "2026-02-30", "yesterday"])
def test_session_date_accepts_only_a_real_yyyy_mm_dd_date(monkeypatch, bad):
    """A number would otherwise be read as a Unix timestamp, and a day-first or month-first date is ambiguous."""
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SESSION_DATE", bad)
    from nasdaq_agent.config import Settings
    with pytest.raises(ValidationError, match="session_date"):
        Settings(_env_file=None)
