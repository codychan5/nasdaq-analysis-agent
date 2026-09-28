from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, ConfigDict

from ..calendar import exchange_time
from ..metrics import AnalysisResult
from ..sources.models import Headline, first_published, most_recent_first
from .schemas import Narrative

TEMPLATE_DIR = Path(__file__).parent / "templates"
# The status line under the headline. It follows the exit code: attention is needed exactly when the run has a problem
# (a degradation, exit 2). A sent report always has re-checked figures, because sending requires a verified analysis.
STATUS_ALL_GOOD = "everything worked, and every calculated figure was re-checked by separate code."
STATUS_NEEDS_ATTENTION = "some things need your attention."
# B4: schemes a headline may be rendered as a clickable link. Anything else (javascript:, data:, ...) is
# rendered as plain text so an untrusted headline URL cannot smuggle an active-content link into the email.
LINKABLE_URL_SCHEMES = ("http", "https")
# A headline's publication time in the email: exchange time, to read beside the session date.
HEADLINE_TIME_FORMAT = "%Y-%m-%d %H:%M ET"
# How each data source is named in the email, in its news source tags and its footer; an unlisted source shows its
# own name.
SOURCE_LABELS = {"massive": "Massive", "yfinance": "Yahoo Finance", "sec": "SEC EDGAR", "alphavantage": "Alpha Vantage",
                 "yahoo": "the Yahoo Finance screener", "nasdaqcom": "the Nasdaq.com screener"}
# The labels of the line under a news entry's title. A linked title already leads to the story, so the line then names
# only the other outlets that reported it; under a title with no link it names every source.
ALSO_REPORTED_LABEL = "Also reported by"
SOURCE_LABEL = "Source"
SOURCES_LABEL = "Sources"


class ReportContext(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    run_id: str
    session_date: str
    session_label: str
    symbol: str
    company: str | None
    pct_change: float
    prev_close: float
    close: float
    verified: AnalysisResult
    benchmark_symbol: str
    narrative: Narrative | None
    template_prose: bool
    headlines: list[Headline]
    sentiment_label: str | None
    sentiment_score: float | None
    # The report's notes, in plain words. problems: one per degradation, listed under the status line, which then says
    # something needs attention. warnings: worth knowing, listed there too, but nothing went wrong with the run.
    # background: how the report was made, in the footer.
    problems: list[str]
    warnings: list[str]
    background: list[str]
    code_hash: str | None
    chart_cid: str | None


# The News section's text when there are no headlines. It stays true whether none were found, the sources failed or
# news was never fetched; the notes under the status line say which.
NO_NEWS_SENTENCE = "No news was available for {symbol}."


class SourceLine(BaseModel):
    """The line under a news entry's title: its label and the headlines it tags, in the story's order."""
    model_config = ConfigDict(extra="forbid")
    label: str
    headlines: list[Headline]


class NewsItem(BaseModel):
    """One story as the email lists it: every headline that reports it, the first one shown, and the agent's summary
    of the story, if any."""
    model_config = ConfigDict(extra="forbid")
    headlines: list[Headline]
    note: str | None = None

    @property
    def lead(self) -> Headline:
        return self.headlines[0]

    @property
    def label(self) -> str:
        return "[" + ", ".join(str(h.id) for h in self.headlines) + "]"

    @property
    def first_published(self) -> datetime | None:
        return first_published(self.headlines)

    @property
    def link(self) -> str | None:
        """The link on the entry's title: the first headline's, when it is safe to show."""
        return headline_link(self.lead.url)

    @property
    def source_line(self) -> SourceLine | None:
        """The line under the title, or None when it would name no outlet. Under a linked title it names the other
        outlets; under a title with no link, every source. Each link appears once in an entry."""
        if self.link is None:
            tagged = headlines_with_new_links(self.headlines, already_shown=set())
            label = SOURCES_LABEL if len(tagged) > 1 else SOURCE_LABEL
        else:
            tagged = headlines_with_new_links(self.headlines[1:], already_shown={self.link})
            label = ALSO_REPORTED_LABEL
        return SourceLine(label=label, headlines=tagged) if tagged else None


def headlines_with_new_links(headlines: list[Headline], already_shown: set[str]) -> list[Headline]:
    """The headlines, in order, without any whose link is already shown or belongs to an earlier headline: two sources
    can return the same article. A headline with no safe link is kept, since its source tag still names its outlet."""
    shown = set(already_shown)
    kept: list[Headline] = []
    for headline in headlines:
        link = headline_link(headline.url)
        if link is not None:
            if link in shown:
                continue
            shown.add(link)
        kept.append(headline)
    return kept


def news_items(ctx: ReportContext) -> list[NewsItem]:
    """The stories in the order the email lists them. With the agent's prose they follow its headline notes, which
    compose_report has checked put every headline in exactly one story, most recent first, and each carries its note.
    Template prose has no notes, so each headline is its own entry, most recent first. A headline the notes miss is
    still listed, after the rest, and an id that is not there is dropped: rendering never loses a headline or fails
    the report."""
    notes = ctx.narrative.headline_notes if ctx.narrative is not None and not ctx.template_prose else []
    by_id = {h.id: h for h in ctx.headlines}
    items: list[NewsItem] = []
    placed: set[int] = set()
    for note in notes:
        story = [by_id[hid] for hid in dict.fromkeys(note.headline_ids) if hid in by_id and hid not in placed]
        if not story:
            continue
        items.append(NewsItem(headlines=story, note=note.summary))
        placed.update(h.id for h in story)
    items.extend(NewsItem(headlines=[h]) for h in most_recent_first([h for h in ctx.headlines if h.id not in placed]))
    return items


def signed_pct(value: float) -> str:
    """A percentage with its sign, as the metrics table shows daily changes: +2.00%."""
    return f"{value:+.2f}%"


def headline_time(moment: datetime) -> str:
    return exchange_time(moment).strftime(HEADLINE_TIME_FORMAT)


def source_label(name: str) -> str:
    """A data source as the email names it, e.g. "massive" is "Massive"; an unlisted source keeps its own name."""
    return SOURCE_LABELS.get(name, name)


def source_tag(headline: Headline) -> str:
    """Who published the headline and which news source it came through, e.g. "GlobeNewswire via Massive"."""
    if not headline.source:
        return headline.provider
    return f"{headline.provider} via {source_label(headline.source)}"


def source_tag_text(headline: Headline) -> str:
    """The plain-text source tag, followed by the headline's link when it is safe to show."""
    link = headline_link(headline.url)
    return f"{source_tag(headline)} {link}" if link else source_tag(headline)


def template_paragraphs(ctx: ReportContext) -> tuple[str, str]:
    v = ctx.verified
    trend = (f"{ctx.symbol} closed the session {ctx.pct_change:+.2f}%. Over the last five sessions it returned "
             f"{v.cumulative_return_pct:.2f}% with an average daily change of {v.avg_daily_change_pct:.2f}%, "
             f"a maximum drawdown of {v.max_drawdown_pct:.2f}% and a trend classified as {v.trend.value}.")
    news = (f"{len(ctx.headlines)} headlines were retrieved; see the list below."
            if ctx.headlines else NO_NEWS_SENTENCE.format(symbol=ctx.symbol))
    return trend, news


def headline_link(url: str | None) -> str | None:
    """B4: the URL to render a headline as a link, or None to render the title as plain text. Only http
    and https URLs are linkable; urlsplit lower-cases the scheme, so case does not matter.

    Final residual 2: headline URLs are unvalidated provider data. urlsplit raises ValueError on some
    malformed inputs (e.g. an unclosed or non-address IPv6 literal like "http://[::1"); a raised
    exception here would escape render_report and end the run unsent. Any URL that fails to parse is
    treated as non-linkable and rendered as plain text."""
    if not url:
        return None
    try:
        scheme = urlsplit(url).scheme.lower()
    except ValueError:
        return None
    return url if scheme in LINKABLE_URL_SCHEMES else None


def _env() -> Environment:
    # Controller correction 14a: select_autoescape(["html"]) checks the template name against
    # ".html" only, which "report.html.j2" never matches, so escaping was silently off for an
    # email body that embeds untrusted headline text and model prose. Match both suffixes so
    # the html template is autoescaped while the plain-text template (report.txt.j2) is not.
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(enabled_extensions=("html", "html.j2")),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters["headline_link"] = headline_link
    env.filters["headline_time"] = headline_time
    env.filters["signed_pct"] = signed_pct
    env.filters["source_tag"] = source_tag
    env.filters["source_tag_text"] = source_tag_text
    return env


def render_report(ctx: ReportContext) -> tuple[str, str, str]:
    env = _env()
    trend_p, news_p = (ctx.narrative.trend_paragraph, ctx.narrative.news_paragraph) if ctx.narrative and not ctx.template_prose else template_paragraphs(ctx)
    if not ctx.headlines:
        # With no headlines the model writes no news paragraph (compose enforces it), and a blank section reads like
        # a rendering fault, so the section says there is no news in one plain sentence.
        news_p = NO_NEWS_SENTENCE.format(symbol=ctx.symbol)
    status = STATUS_NEEDS_ATTENTION if ctx.problems else STATUS_ALL_GOOD
    data = dict(ctx=ctx, v=ctx.verified, trend_paragraph=trend_p, news_paragraph=news_p, news_items=news_items(ctx),
                status=status)
    subject = f"[NASDAQ top gainer] {ctx.symbol} {ctx.pct_change:+.2f}% on {ctx.session_date}"
    return subject, env.get_template("report.txt.j2").render(**data), env.get_template("report.html.j2").render(**data)
