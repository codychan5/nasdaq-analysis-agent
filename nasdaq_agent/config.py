import re
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, get_args
from zoneinfo import ZoneInfo

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .cron import cron_trigger, next_fire_time

# A single plain address: exactly one "@", no whitespace or newlines (this also blocks
# header injection such as a trailing "\nBcc: ..."), and at least one "." in the domain part.
SINGLE_EMAIL_ADDRESS_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# SEC's fair-access policy asks automated clients to declare themselves with a name and a contact email, such as
# "Jane Doe jane@example.com". The value goes out as one HTTP header, so it must be a single line of printable ASCII.
CONTACT_EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PRINTABLE_ASCII_LINE_PATTERN = re.compile(r"[\x20-\x7e]+")
SEC_USER_AGENT_MAX_CHARS = 200


class Mode(StrEnum):
    live = "live"
    replay = "replay"
    record = "record"


class SandboxBackend(StrEnum):
    auto = "auto"
    docker = "docker"
    subprocess = "subprocess"


class EmailTransport(StrEnum):
    auto = "auto"
    smtp = "smtp"
    file = "file"


class Settings(BaseSettings):
    """All runtime configuration. Unknown keys are rejected. Secrets never print."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_", env_file=".env", extra="forbid", hide_input_in_errors=True
    )

    # GLM-5.3 on OpenRouter: Gemini and Claude are region-restricted in Hong Kong, where this was built, and free
    # OpenRouter models proved too congested to finish a run. It is pay-as-you-go and served by many providers; the
    # settings template adds DeepSeek V4.1 Flash, from another lab on other hosts, as the fallback.
    llm_model: str = "openrouter:z-ai/glm-5.3"
    llm_judge_model: str | None = None
    llm_fallback_model: str | None = None
    llm_call_delay_seconds: float = 4.0
    # Each model request times out on our side after this many seconds and is retried at most llm_max_retries times;
    # a timed-out request is abandoned, so its answer can never arrive late. With the defaults, a model and its
    # fallback together wait at most 2 x 90 x (1 + 2) = 540 s, inside the 600 s run deadline.
    llm_timeout_seconds: PositiveFloat = 90.0
    llm_max_retries: NonNegativeInt = 2
    # Largest reply a model call may produce. OpenRouter reserves credit for it up front, and an uncapped call failed a
    # live run with "requires more credits, or fewer max_tokens". Reasoning models think within it, so it leaves room.
    llm_max_output_tokens: PositiveInt = 8192
    max_model_calls: int = 25
    max_tool_calls: int = 25
    max_code_runs: int = 6
    mode: Mode = Mode.live
    sandbox_backend: SandboxBackend = SandboxBackend.auto
    email_transport: EmailTransport = EmailTransport.auto
    email_to: str
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    smtp_from: str | None = None
    smtp_starttls: bool = True
    lookback_sessions: int = 5
    # How many days before the session get_news looks for headlines. Three by default, the original fixed window, so
    # the news stays about the move; a setting so a thinly covered gainer can be given a longer window, bounded so a
    # typo cannot ask a news API for years of history.
    news_lookback_days: int = Field(default=3, ge=1, le=90)
    benchmark_symbol: str = "SPY"
    artifacts_dir: Path = Path("./runs")
    # Record writes, and replay reads, HTTP and model responses here (Task 22). Committed, so never a secret in it.
    cassette_dir: Path = Path("./fixtures/cassettes")
    # Read by RunDeadlineMiddleware, which ends the agent loop once an invocation has run this long; a resume starts
    # a fresh deadline, and the call caps bound the whole run (final fix wave A2).
    run_deadline_seconds: PositiveInt = 600
    http_connect_timeout: float = 5.0
    http_read_timeout: float = 10.0
    # When `nasdaq-agent schedule` runs the agent: a standard five-field crontab expression, read in schedule_timezone.
    # By default 16:30 New York time, Monday to Friday: half an hour after the close, and never at weekends, when the
    # market is shut. The scheduler also skips a trigger whose session it already reported, such as a market holiday.
    schedule_cron: str = "30 16 * * 1-5"
    schedule_timezone: str = "America/New_York"
    # Turns on the SEC EDGAR news source: the User-Agent its requests declare, a name and a contact email. It holds a
    # person's address, so it is kept like a key: hidden from reprs, redacted from logs, sent only to SEC.
    sec_user_agent: SecretStr | None = None

    # Provider and tracing keys keep their native names, so no prefix.
    google_api_key: SecretStr | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    massive_api_key: SecretStr | None = Field(default=None, validation_alias="MASSIVE_API_KEY")
    alphavantage_api_key: SecretStr | None = Field(default=None, validation_alias="ALPHAVANTAGE_API_KEY")
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str | None = Field(default=None, validation_alias="LANGSMITH_PROJECT")

    @field_validator("smtp_host", "smtp_username", "smtp_password", "smtp_from", mode="before")
    @classmethod
    def _blank_smtp_to_none(cls, v):
        """Convert empty or whitespace-only strings to None for SMTP fields."""
        if isinstance(v, (str, SecretStr)):
            val_str = v.get_secret_value() if isinstance(v, SecretStr) else v
            if isinstance(val_str, str) and not val_str.strip():
                return None
        return v

    @field_validator("sec_user_agent", mode="before")
    @classmethod
    def _trim_sec_user_agent_blank_to_none(cls, v):
        """Surrounding whitespace is dropped, and a blank value leaves the SEC source off."""
        text = v.get_secret_value() if isinstance(v, SecretStr) else v
        if isinstance(text, str):
            return text.strip() or None
        return v

    @field_validator("sec_user_agent")
    @classmethod
    def _sec_user_agent_must_name_a_contact(cls, v: SecretStr | None) -> SecretStr | None:
        """One line of printable ASCII, bounded, with a contact email in it. The messages never quote the value."""
        if v is None:
            return v
        text = v.get_secret_value()
        if len(text) > SEC_USER_AGENT_MAX_CHARS or not PRINTABLE_ASCII_LINE_PATTERN.fullmatch(text):
            raise ValueError(f"sec_user_agent must be one line of printable ASCII, at most {SEC_USER_AGENT_MAX_CHARS} "
                             "characters")
        if not CONTACT_EMAIL_PATTERN.search(text):
            raise ValueError("sec_user_agent must include a contact email, as SEC asks: 'Your Name you@example.com'")
        return v

    @field_validator("email_to")
    @classmethod
    def _email_to_must_be_single_address(cls, v: str) -> str:
        """Reject anything but a single plain address; also blocks header injection via newlines."""
        if not SINGLE_EMAIL_ADDRESS_PATTERN.match(v):
            raise ValueError("email_to must be a single plain email address")
        return v

    @field_validator("smtp_from")
    @classmethod
    def _smtp_from_must_be_single_address(cls, v: str | None) -> str | None:
        """Runs after `_blank_smtp_to_none`; a blank value is already None here and skips the check."""
        if v is not None and not SINGLE_EMAIL_ADDRESS_PATTERN.match(v):
            raise ValueError("smtp_from must be a single plain email address")
        return v

    @field_validator("schedule_timezone")
    @classmethod
    def _schedule_timezone_must_exist(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ValueError, KeyError):  # ZoneInfoNotFoundError is a KeyError
            raise ValueError("schedule_timezone must be an IANA time zone name, such as America/New_York") from None
        return v

    @field_validator("schedule_cron")
    @classmethod
    def _schedule_cron_must_fire(cls, v: str) -> str:
        """Fail at startup rather than at the first trigger: the expression must read the way the scheduler reads it,
        and it must fire at least once. The time zone cannot change either answer, so UTC stands in for it here."""
        next_fire_time(cron_trigger(v, "UTC"), datetime.now(timezone.utc))
        return v

    @model_validator(mode="after")
    def _smtp_complete(self) -> "Settings":
        wants_smtp = self.email_transport == EmailTransport.smtp or (
            self.email_transport == EmailTransport.auto and self.smtp_host
        )
        if wants_smtp:
            missing = [n for n in ("smtp_host", "smtp_username", "smtp_password", "smtp_from") if getattr(self, n) is None]
            if missing:
                raise ValueError(f"SMTP transport selected but settings missing: {', '.join(missing)}")
        return self

    @property
    def resolved_email_transport(self) -> EmailTransport:
        if self.email_transport != EmailTransport.auto:
            return self.email_transport
        return EmailTransport.smtp if self.smtp_host else EmailTransport.file

    @property
    def resolved_judge_model(self) -> str:
        return self.llm_judge_model or self.llm_model

    @classmethod
    def secret_field_names(cls) -> frozenset[str]:
        """Fix round 1, item 4: every field whose annotation includes SecretStr (direct, or via
        `SecretStr | None`), found by introspecting the model instead of hand-listing field names
        in a second place -- a future SecretStr field is picked up automatically."""
        def _mentions_secret_str(annotation: Any) -> bool:
            if annotation is SecretStr:
                return True
            return any(_mentions_secret_str(arg) for arg in get_args(annotation))
        return frozenset(name for name, field in cls.model_fields.items() if _mentions_secret_str(field.annotation))

    def secret_values(self) -> list[str]:
        """Every actually-set secret's plain value, for redaction. Never logged or printed itself."""
        return [value.get_secret_value() for name in self.secret_field_names()
                if (value := getattr(self, name)) is not None]


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from the environment and an optional .env file."""
    return Settings(_env_file=env_file) if env_file is not None else Settings()
