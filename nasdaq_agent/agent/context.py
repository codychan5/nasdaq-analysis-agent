from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..config import Mode
from ..metrics import AnalysisResult
from ..sources.models import BarSeries, Headline

CONTEXT_FILENAME = "context.json"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Progress(Strict):
    session_resolved: bool = False
    gainer_chosen: bool = False
    history_ready: bool = False
    successful_run: bool = False
    verified: bool = False
    news_fetched: bool = False
    sentiment_recorded: bool = False
    composed: bool = False
    sent: bool = False
    gave_up: bool = False
    analysis_terminal: bool = False


class SessionInfo(Strict):
    date: str
    label: str
    early_close: bool
    market_closed_at_resolution: bool


class SourceAttempt(Strict):
    source: str
    ok: bool
    detail: str


class SkippedCandidate(Strict):
    symbol: str
    source: str
    reason: str


class ExcludedCandidate(Strict):
    """A stock a gainer source ranked above the ones it checked, which did not count for the session: not NASDAQ
    common stock or an ADR that day. Kept so a run shows what the filter removed ahead of the pick."""
    symbol: str
    source: str
    pct_change: float
    reason: str


class ListingInfo(Strict):
    """Which list decided what counts: NASDAQ's symbol file ("nasdaqtrader") or, for a session before the file's day,
    Massive's records for that session ("massive"). listed_on is the day the list describes."""
    source: str
    listed_on: str | None
    common_stocks: int


class GainerInfo(Strict):
    symbol: str
    company: str | None
    prev_close: float
    close: float
    pct_change: float
    source: str
    reconciled_pct: float


class PriceCheck(Strict):
    """The stock's closes compared with a second history source. detail is the sentence the report shows. restated:
    every close differs by one factor, the mark of a split after the sessions that one source restated its history
    for; traded_prev_close and traded_close are then the raw source's closes for the session and the one before."""
    status: Literal["agree", "disagree", "not_checked", "restated"]
    source: str | None = None
    detail: str
    traded_prev_close: float | None = None
    traded_close: float | None = None


class HistoryInfo(Strict):
    ticker: BarSeries
    benchmark: BarSeries
    source: str
    expected_dates: list[str]
    price_check: PriceCheck | None = None  # None in a context saved before the check existed


class AttemptInfo(Strict):
    number: int
    code_hash: str
    backend: str
    ok: bool
    exit_code: int
    wall_time_s: float
    error: str | None


class VerificationRow(Strict):
    metric: str
    model_value: str
    verifier_value: str
    ok: bool


class AnalysisInfo(Strict):
    # Mutable defaults use Field(default_factory=...) so
    # separate RunContext instances never share the same list object.
    attempts: list[AttemptInfo] = Field(default_factory=list)
    latest_result: AnalysisResult | None = None
    verified_result: AnalysisResult | None = None
    verification: list[VerificationRow] = Field(default_factory=list)
    rejections: int = 0
    # The attempt number whose result verify_analysis actually
    # verified, so compose_report can cite that attempt's code hash/backend instead of
    # whatever run_python call happened to be most recent (which may be a later, failed, or
    # still-unverified attempt).
    verified_attempt: int | None = None
    # The attempt number verify_analysis last compared against the
    # verifier, whether it passed or was rejected. Calling verify_analysis again with no new
    # successful run_python call in between re-judges the same result and must refuse instead
    # of spending another rejection.
    last_judged_attempt: int | None = None


class NewsInfo(Strict):
    headlines: list[Headline]
    # The sources whose headlines are included, in the order they were asked.
    sources: list[str]


class SentimentInfo(Strict):
    score: float
    label: str
    rationale: str
    per_headline: dict[int, float]
    disagreements_with_provider: int


class ReportInfo(Strict):
    subject: str
    text_path: str
    html_path: str
    chart_path: str | None
    template_prose: bool
    rewrites: int
    grounding_findings: list[str]
    judge_faithful: bool | None


class EmailInfo(Strict):
    status: str
    transport: str | None
    message_id: str | None
    location: str | None


class RunContext(Strict):
    run_id: str
    started_at: str
    settings_hash: str
    artifacts_path: str
    # The mode the run started in; resume refuses any other. A context written
    # before this field existed loads as live, the only mode those runs could have been resumed in.
    mode: Mode = Mode.live
    # Every mutable default (Progress, AnalysisInfo, and every
    # list field below) uses Field(default_factory=...) instead of a shared literal
    # instance/list, so two RunContext.new() calls never alias each other's state.
    progress: Progress = Field(default_factory=Progress)
    session: SessionInfo | None = None
    gainer: GainerInfo | None = None
    skipped_candidates: list[SkippedCandidate] = Field(default_factory=list)
    excluded_candidates: list[ExcludedCandidate] = Field(default_factory=list)
    listing: ListingInfo | None = None
    gainer_sources_tried: list[SourceAttempt] = Field(default_factory=list)
    history: HistoryInfo | None = None
    history_sources_tried: list[SourceAttempt] = Field(default_factory=list)
    analysis: AnalysisInfo = Field(default_factory=AnalysisInfo)
    news: NewsInfo | None = None
    news_sources_tried: list[SourceAttempt] = Field(default_factory=list)
    # Headlines gathered from the sources asked so far, tagged and numbered across sources. The news step closes into
    # `news` once every source has been tried.
    news_gathered: list[Headline] = Field(default_factory=list)
    sentiment: SentimentInfo | None = None
    report: ReportInfo | None = None
    email: EmailInfo | None = None
    notes: list[str] = Field(default_factory=list)
    degradations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    give_up_reason: str | None = None
    last_tool: str | None = None
    tool_calls: int = 0
    compose_rewrites: int = 0
    # Judge calls that failed ("judge unavailable"). A record run with any is not sealed: its
    # verdict is not in the cassette, so a replay would fail at the judge.
    judge_failures: int = 0

    @classmethod
    def new(cls, run_id: str, settings_hash: str, artifacts_path: str, mode: Mode = Mode.live) -> "RunContext":
        return cls(run_id=run_id, started_at=datetime.now(timezone.utc).isoformat(),
                   settings_hash=settings_hash, artifacts_path=artifacts_path, mode=mode)

    def save(self) -> None:
        Path(self.artifacts_path).mkdir(parents=True, exist_ok=True)
        (Path(self.artifacts_path) / CONTEXT_FILENAME).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "RunContext":
        return cls.model_validate_json((Path(path) / CONTEXT_FILENAME).read_text())
