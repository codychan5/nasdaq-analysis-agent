from pydantic import BaseModel, ConfigDict, Field, field_validator

# A story summary is asked for in 60 to 100 words for company news; this ceiling (about 130 words) leaves room for
# that without letting one story swamp the section.
HEADLINE_NOTE_MAX_CHARS = 800


class DeclaredMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="A verified metric name, or daily_changes_pct[i] with i from 0 to 4")
    value: float


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentence_index: int = Field(ge=0, description="0-based index of the sentence in news_paragraph")
    headline_ids: list[int] = Field(min_length=1)


class HeadlineNote(BaseModel):
    """The agent's summary of one story: what it reports, what it means for the company, and how it may bear on the
    stock's price move. A story is every headline that reports the same information, from whichever sources; the email
    shows it once, tagged with each of them, with this summary beneath."""
    model_config = ConfigDict(extra="forbid")
    headline_ids: list[int] = Field(min_length=1, description="ids of every headline that reports this story; the "
                                                              "email shows the first one's title")
    summary: str = Field(max_length=HEADLINE_NOTE_MAX_CHARS,
                         description="What the story reports, what it means for the company, and how it may relate to "
                                     "the stock's price move, as context and not a proven cause")

    @field_validator("summary")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must not be blank")
        return value


class Narrative(BaseModel):
    """What the model returns from compose_report. Numbers in prose must be declared metrics."""
    model_config = ConfigDict(extra="forbid")
    trend_paragraph: str = Field(max_length=800)
    news_paragraph: str = Field(max_length=800)
    declared_metrics: list[DeclaredMetric]
    citations: list[Citation]
    # The email lists the stories in this order, each with its note; grounding checks the order is most recent first.
    headline_notes: list[HeadlineNote] = Field(default_factory=list,
                                               description="One note per story, most recent first by when each story "
                                                           "first appeared, undated last")


class JudgeIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentence: str
    problem: str


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    faithful: bool
    issues: list[JudgeIssue] = []


class GroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    findings: list[str] = []
