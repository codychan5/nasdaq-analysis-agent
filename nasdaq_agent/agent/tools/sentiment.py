from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from ..context import RunContext, SentimentInfo
from .common import Deps, err, ok, precondition, record_call

PROVIDER_SIGN = {"positive": 1, "negative": -1, "neutral": 0}
# A label must agree with its own score, not just be any of the three literals. Neutral gets a tolerance band rather
# than requiring exactly 0.0, since the model is scoring subjective tone, not reporting a computed value.
NEUTRAL_SCORE_BOUND = 0.25
LABEL_CONSISTENT_WITH_SCORE = {
    "positive": lambda score: score > 0,
    "negative": lambda score: score < 0,
    "neutral": lambda score: abs(score) <= NEUTRAL_SCORE_BOUND,
}


class HeadlineScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    headline_id: int
    score: float = Field(ge=-1.0, le=1.0)


def make_record_sentiment(ctx: RunContext, deps: Deps):
    @tool("record_sentiment")
    def record_sentiment(score: float, label: Literal["negative", "neutral", "positive"], rationale: str,
                         per_headline: list[HeadlineScore]) -> str:
        """Record your sentiment assessment of the fetched headlines: an overall score in [-1, 1], a label,
        a one-sentence rationale, and a score per headline id. Judge only from the headline text. label must
        agree with score: positive needs score > 0, negative needs score < 0, neutral needs abs(score) <= 0.25."""
        # Every return path below funnels through _record, so tool_log.jsonl gets exactly one entry per call whatever
        # the outcome, including the refusals that return early, such as a score out of range or an unknown headline id.
        def _record(outcome: str) -> str:
            record_call(ctx, deps, "record_sentiment", {"score": score, "label": label}, outcome)
            return outcome

        blocked = precondition(ctx, "news_fetched", "record_sentiment", "call get_news first")
        if blocked:
            return _record(blocked)
        # With no headlines there is nothing to assess: a live run showed a model inventing a sentiment from nothing,
        # which would have reached the email. Refuse; the report then carries no sentiment and no degradation.
        if ctx.news is None or not ctx.news.headlines:
            return _record(err("no headlines were fetched, so there is nothing to score; skip record_sentiment and "
                               "continue with compose_report"))
        if not -1.0 <= score <= 1.0:
            return _record(err("score must be between -1 and 1"))
        # Reject a label that contradicts its own score (e.g. "positive" with score -0.4) instead of trusting the
        # model's label and score to agree.
        if not LABEL_CONSISTENT_WITH_SCORE[label](score):
            return _record(err(f"label '{label}' is inconsistent with score {score}: positive requires score > 0, "
                               f"negative requires score < 0, neutral requires abs(score) <= {NEUTRAL_SCORE_BOUND}"))
        known = {h.id: h for h in ctx.news.headlines}
        unknown = [p.headline_id for p in per_headline if p.headline_id not in known]
        if unknown:
            return _record(err(f"unknown headline ids {unknown}; valid ids: {sorted(known)}"))
        ids = [p.headline_id for p in per_headline]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            return _record(err(f"duplicate headline ids in per_headline: {duplicates}"))
        disagreements = 0
        for p in per_headline:
            provider = known[p.headline_id].provider_sentiment
            if provider in PROVIDER_SIGN and PROVIDER_SIGN[provider] != 0 and (p.score > 0) != (PROVIDER_SIGN[provider] > 0):
                disagreements += 1
        ctx.sentiment = SentimentInfo(score=score, label=label, rationale=rationale,
                                      per_headline={p.headline_id: p.score for p in per_headline},
                                      disagreements_with_provider=disagreements)
        ctx.progress.sentiment_recorded = True
        outcome = ok({"recorded": True, "disagreements_with_provider_labels": disagreements})
        return _record(outcome)
    return record_sentiment
