from datetime import datetime, timezone
import pytest

def verified():
    from nasdaq_agent.metrics import AnalysisResult
    return AnalysisResult(daily_changes_pct=[2.0, -2.451, 6.533, 0.943, 3.738], avg_daily_change_pct=2.153,
                          cumulative_return_pct=11.0, volatility_annualized_pct=52.87, max_drawdown_pct=-2.451,
                          relative_vs_spy_pct=10.2, trend="uptrend")

def headlines():
    from nasdaq_agent.sources.models import Headline
    t = datetime(2026, 9, 24, tzinfo=timezone.utc)
    return [Headline(id=1, title="Acme files for FDA review of lead drug candidate", provider="Reuters", published=t),
            Headline(id=2, title="Acme raises full-year revenue guidance", provider="Business Wire", published=t),
            Headline(id=3, title="Analyst upgrades Acme to Buy", provider="MarketBeat", published=t)]

def good_notes():
    """One relevance note per fixture headline. The three share a publication time, so any order is most recent first."""
    from nasdaq_agent.report.schemas import HeadlineNote
    return [HeadlineNote(headline_ids=[1], summary="A filing for FDA review is an early regulatory step; it may have drawn buyers, though the headline does not tie it to the move."),
            HeadlineNote(headline_ids=[2], summary="Raised revenue guidance is the kind of news that can lift a stock, so it may help explain the gain."),
            HeadlineNote(headline_ids=[3], summary="An analyst upgrade can add buying interest; the headline does not say how much it moved the stock.")]

def good_narrative():
    from nasdaq_agent.report.schemas import Narrative, DeclaredMetric, Citation
    return Narrative(
        trend_paragraph="Acme closed the session up 3.7% and finished the five-day window 11.0% higher, with four of five sessions positive. The average daily move was 2.2%, and the only pullback was a 2.5% dip mid-window.",
        news_paragraph="The company filed for FDA review of its lead drug candidate. [1] It also raised full-year revenue guidance. [2] One analyst upgraded the stock to Buy. [3]",
        declared_metrics=[DeclaredMetric(name="daily_changes_pct[4]", value=3.738), DeclaredMetric(name="cumulative_return_pct", value=11.0),
                          DeclaredMetric(name="avg_daily_change_pct", value=2.153), DeclaredMetric(name="max_drawdown_pct", value=-2.451)],
        citations=[Citation(sentence_index=0, headline_ids=[1]), Citation(sentence_index=1, headline_ids=[2]), Citation(sentence_index=2, headline_ids=[3])],
        headline_notes=good_notes())

def test_good_narrative_passes():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(good_narrative(), verified(), headlines())
    assert res.ok, res.findings

def test_bad_narrative_reports_every_finding():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import Narrative, DeclaredMetric, Citation
    bad = Narrative(
        trend_paragraph="Acme gained 12% over the week and its daily volatility averaged 52.9%.",
        news_paragraph="The company won FDA approval for its lead drug. [1] It also announced a share buyback.",
        declared_metrics=[DeclaredMetric(name="cumulative_return_pct", value=12.0), DeclaredMetric(name="avg_daily_change_pct", value=52.88)],
        citations=[Citation(sentence_index=0, headline_ids=[1])])
    res = check_grounding(bad, verified(), headlines())
    joined = "\n".join(res.findings)
    assert not res.ok
    assert "cumulative_return_pct = 12.0 but verified value is 11.0" in joined
    assert "avg_daily_change_pct = 52.88 but verified value is 2.153" in joined and "volatility_annualized_pct" in joined
    assert "52.9%" in joined
    assert "sentence 2" in joined and "no citation" in joined

def test_unknown_metric_name_and_bad_citation_id():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import Narrative, DeclaredMetric, Citation
    n = Narrative(trend_paragraph="Flat.", news_paragraph="Something happened. [9]",
                  declared_metrics=[DeclaredMetric(name="sharpe", value=1.0)], citations=[Citation(sentence_index=0, headline_ids=[9])])
    res = check_grounding(n, verified(), headlines())
    joined = "\n".join(res.findings)
    assert "unknown metric 'sharpe'" in joined and "headline 9 does not exist" in joined

def test_no_headlines_means_no_citation_checks():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import Narrative
    n = Narrative(trend_paragraph="Flat.", news_paragraph="No headlines were available for this stock.", declared_metrics=[], citations=[])
    assert check_grounding(n, verified(), []).ok

def test_exempt_tokens():
    # Years are no longer exempt by default -- the caller must name which years are in play for this run.
    from nasdaq_agent.report.grounding import numeric_tokens
    assert numeric_tokens("On 2026-09-24 the stock rose 3.7% over five sessions [1] in 2026", exempt_years={2026}) == ["3.7"]


def test_good_narrative_news_paragraph_splits_into_three_cited_sentences():
    # split_sentences must keep a trailing citation marker attached to the sentence it cites,
    # not let it start the next sentence.
    from nasdaq_agent.report.grounding import split_sentences
    sentences = split_sentences(good_narrative().news_paragraph)
    assert len(sentences) == 3
    for i, sentence in enumerate(sentences, start=1):
        assert sentence.endswith(f"[{i}]"), sentences


def test_extra_facts_allow_session_and_price_declarations():
    # extra_facts keys become declarable names alongside metric names, so the model can quote the
    # session's percentage change and closes.
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import Narrative, DeclaredMetric
    n = Narrative(
        trend_paragraph="Acme closed at 11.10 after rising 3.74% on the session.",
        news_paragraph="",
        declared_metrics=[DeclaredMetric(name="close", value=11.10), DeclaredMetric(name="session_pct_change", value=3.738)],
        citations=[], headline_notes=good_notes())
    extra_facts = {"close": 11.10, "prev_close": 10.70, "session_pct_change": 3.738}

    with_facts = check_grounding(n, verified(), headlines(), extra_facts=extra_facts)
    assert with_facts.ok, with_facts.findings

    without_facts = check_grounding(n, verified(), headlines())
    assert not without_facts.ok


@pytest.mark.parametrize("written, grounded", [("288%", True), ("287.98%", True), ("287.9808%", True), ("287.9%", False)])
def test_a_prose_number_matches_its_verified_value_rounded_to_the_decimals_it_shows(written, grounded):
    """compose_report's description promises the model this rule with these numbers: rounding is accepted, and cutting
    digits off is not."""
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import DeclaredMetric, Narrative
    value = 287.98076878990645
    narrative = Narrative(trend_paragraph=f"The stock rose {written} across the window.", news_paragraph="",
                          declared_metrics=[DeclaredMetric(name="cumulative_return_pct", value=value)], citations=[])
    result = check_grounding(narrative, verified().model_copy(update={"cumulative_return_pct": value}), [])
    assert result.ok is grounded, result.findings


# Years and number words: a year-shaped token is exempt only when the caller names it via
# exempt_years, and only when nothing around it says it's really a quantity ($, a unit). Every
# complete number is tokenized whole -- never stripped to a substring first -- so a number a year
# merely happens to prefix (e.g. "1950.2%") can never lose its non-year remainder.

def _n(trend_paragraph):
    from nasdaq_agent.report.schemas import Narrative
    return Narrative(trend_paragraph=trend_paragraph, news_paragraph="", declared_metrics=[], citations=[])


def test_bare_year_in_exempt_set_with_no_unit_or_currency_is_exempt():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("Acme rose in 2026 on strong demand."), verified(), [], exempt_years={2026})
    assert res.ok, res.findings


def test_bare_year_not_in_exempt_set_fails():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("Acme has traded publicly since 1950."), verified(), [], exempt_years={2026, 2025})
    assert not res.ok


def test_year_preceded_by_currency_is_not_exempt_even_if_named():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("The stock is trading at $2026 a share."), verified(), [], exempt_years={2026})
    assert not res.ok


@pytest.mark.parametrize("suffix", ["bps", "points", "bp", "pct", "percent", "pp"])
def test_year_followed_by_a_unit_is_not_exempt_even_if_named(suffix):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(f"The spread widened by 2026 {suffix} this week."), verified(), [], exempt_years={2026})
    assert not res.ok, (suffix, res.findings)


def test_decimal_number_is_never_treated_as_a_year():
    # "closed at 2015.40" must pass when close=2015.40 is declared: the decimal point rules out
    # year treatment entirely (a year is a BARE four-digit token), so this is an ordinary
    # number that only passes because it matches a declared value, not because it looks like
    # "20xx".
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import DeclaredMetric
    n = _n("Acme closed at 2015.40 on the session.").model_copy(update={"declared_metrics": [DeclaredMetric(name="close", value=2015.40)]})
    res = check_grounding(n, verified(), [], extra_facts={"close": 2015.40})
    assert res.ok, res.findings


# Each of these leaks used to pass silently. exempt_years={2026, 2025} stands in for a realistic
# compose_report call (the session year and the one before it) and must not rescue any of
# these -- none of the numbers below are 2025 or 2026.
NUMBER_LEAKS_THAT_MUST_NOW_FAIL = [
    "Acme is up 1950.2% since listing.",          # decimal number a year is a substring of
    "Five percentage points were added.",          # compound unit word
    "Acme rose 1950 per cent this year.",          # two-word unit
    "Acme reported a 1950-percent gain.",          # hyphenated number-unit
    "The spread widened by 2019 basis points.",    # compound unit word
    "The spread widened by 2019 bp.",              # abbreviated unit
    "Acme rose 1950 pct on the news.",             # abbreviated unit
    "The stock trades at $ 2015 a share.",         # currency prefix, spaced
    "The stock trades at 2015 dollars a share.",   # trailing currency-word unit
    "The stock trades at \u20ac2015 a share.",     # euro prefix, no space
    "Acme rose five per cent today.",              # number word + two-word unit
    "Acme rose one-percent today.",                # number word, hyphenated unit
    "Acme rose one dollar today.",                 # number word + currency-word unit
    "The spread widened by three bps.",            # number word + abbreviated unit
    "Acme rose half a percent today.",             # number word, "a" between word and unit
    "Investors added two trillion dollars in value.",  # magnitude word
]


@pytest.mark.parametrize("phrase", NUMBER_LEAKS_THAT_MUST_NOW_FAIL)
def test_number_leaks_are_closed(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert not res.ok, phrase


def test_magnitude_word_alone_with_no_unit_is_still_always_a_finding():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("Analysts expect a million new users."), verified(), [], exempt_years={2026, 2025})
    assert not res.ok
    assert any('quantity "million" must be a declared number written in digits' in f for f in res.findings)


# Plain counts that must keep passing: zero..ninety, dozen/half/quarter are fine whenever no
# unit follows and no currency precedes them.
SAFE_NUMBER_WORD_PHRASES_THAT_MUST_PASS = [
    "Acme patched a zero-day flaw in its software.",
    "Acme adopted the Zero Trust platform this quarter.",
    "Twenty-First Century Fox was mentioned in the filing.",
    "One analyst upgraded the stock to Buy.",
    "Acme's rally spans a five-day window.",
    "Investors focused on the first half of the year.",
    "Acme rose on four of five sessions this week.",
    "The memo covers the key points of the strategy.",
]


@pytest.mark.parametrize("phrase", SAFE_NUMBER_WORD_PHRASES_THAT_MUST_PASS)
def test_safe_number_word_phrases_still_pass(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert res.ok, (phrase, res.findings)


def test_good_narrative_number_words_used_as_counts_still_pass():
    # "four of five sessions" and "One analyst" in the shared good narrative must keep passing.
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(good_narrative(), verified(), headlines())
    assert res.ok, res.findings


# Hyphen spellings: gaps and false positives.

# A "-" directly after a letter or digit is a hyphen joining words, not a minus sign, so it must
# not be consumed into the number that follows it. An earlier version treated it as a minus sign,
# so "mid-2026" read as the signed number "-2026".
HYPHEN_BEFORE_YEAR_PHRASES_THAT_MUST_PASS = [
    "Acme expects approval in mid-2026.",
    "Guidance reflects pre-2025 levels.",
    "Acme is planning a late-2026 launch.",
    "Guidance covers 2025-2026.",
]


@pytest.mark.parametrize("phrase", HYPHEN_BEFORE_YEAR_PHRASES_THAT_MUST_PASS)
def test_hyphen_before_a_bare_year_is_not_a_minus_sign(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert res.ok, (phrase, res.findings)


def test_minus_after_whitespace_bracket_or_start_of_text_keeps_its_sign():
    from nasdaq_agent.report.grounding import numeric_tokens
    assert numeric_tokens("Shares fell -2.3% today.") == ["-2.3"]
    assert numeric_tokens("(-2.3%) was the move.") == ["-2.3"]
    assert numeric_tokens("-2.3% was the move.") == ["-2.3"]


def test_unicode_minus_sign_is_always_a_minus():
    # U+2212 is never typeset as a hyphen, so it keeps its sign regardless of what precedes it,
    # and is normalised to ASCII "-" in the bare comparison form so float() can parse it.
    from nasdaq_agent.report.grounding import check_grounding, numeric_tokens
    from nasdaq_agent.report.schemas import DeclaredMetric
    assert numeric_tokens("The stock fell \u22122.3% today.") == ["-2.3"]
    n = _n("The stock fell \u22122.45% today.").model_copy(update={"declared_metrics": [DeclaredMetric(name="max_drawdown_pct", value=-2.451)]})
    res = check_grounding(n, verified(), [])
    assert res.ok, res.findings


# The "a "-gap before a unit applies only after a number word, never after a numeric token --
# otherwise "2025 a dollar" would misread the year as attaching to "dollar" through the article
# and lose its exemption, as an earlier version did.
@pytest.mark.parametrize("phrase", [
    "In 2025 a dollar of revenue cost more.",
    "In 2026 a point of contention emerged.",
])
def test_a_gap_before_unit_applies_only_after_number_words_not_numeric_tokens(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert res.ok, (phrase, res.findings)


# U+2010/U+2011 normalise to "-", and a hyphen (or whitespace) joins the words of a
# multi-word unit, and joins "a" to a number word and its unit on either side.
HYPHEN_SPELLED_UNIT_PHRASES_THAT_MUST_FAIL = [
    "Acme announced a five-basis-point cut.",
    "Acme announced a twenty-five-basis-point cut.",
    "Margins improved five per-cent.",
    "Margins improved half-a-percent.",
    "Acme announced a half-a-percentage-point cut.",
    "Margins improved one\u2011percent.",
    "Margins improved five\u2011point.",
]


@pytest.mark.parametrize("phrase", HYPHEN_SPELLED_UNIT_PHRASES_THAT_MUST_FAIL)
def test_hyphen_spelled_units_are_still_findings(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert not res.ok, phrase


EXEMPT_YEAR_WITH_HYPHEN_SPELLED_UNIT_PHRASES_THAT_MUST_FAIL = [
    "Acme announced a 2026-basis-point cut.",
    "Acme announced 2025 basis-point tightening.",
    "Acme announced 2026 per-cent growth.",
    "Acme announced 2026\u2011percent growth.",
    "Acme announced 2025\u2011bps tightening.",
]


@pytest.mark.parametrize("phrase", EXEMPT_YEAR_WITH_HYPHEN_SPELLED_UNIT_PHRASES_THAT_MUST_FAIL)
def test_exempt_year_with_hyphen_spelled_unit_still_fails(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert not res.ok, phrase


# Proper-noun labels that merely contain digits are removed before tokenizing, the way
# dates and citations are, so they are never mistaken for a quantity to declare.
PROPER_NOUN_PHRASES_THAT_MUST_PASS = [
    "Acme outperformed the S&P 500 this week.",
    "Acme joined the Nasdaq-100 index.",
    "Acme joined the Nasdaq 100 index.",
    "Acme joined the Russell 2000 index.",
    "Acme began a Phase 3 trial.",
    "Acme began a Phase 2b/3 study.",
    "Acme began a Phase III trial.",
]


@pytest.mark.parametrize("phrase", PROPER_NOUN_PHRASES_THAT_MUST_PASS)
def test_proper_noun_labels_are_removed_before_tokenizing(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert res.ok, (phrase, res.findings)


def test_other_numbers_near_a_proper_noun_label_are_still_checked():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("The S&P 500 rose 500 points today."), verified(), [], exempt_years={2026, 2025})
    assert not res.ok


# Pin the unit and currency rules using years that ARE in the exempt set -- the number-leak tests
# above all use years outside the set, so a mutation that gutted the unit/currency rule entirely
# would still pass every one of them.
RULED_UNIT_SPELLINGS = ["%", "percent", "percentage", "percentage points", "per cent", "pct",
                       "pp", "bp", "bps", "basis points", "points", "dollars", "cents", "usd"]


@pytest.mark.parametrize("unit", RULED_UNIT_SPELLINGS)
def test_exempt_year_followed_by_each_ruled_unit_still_fails(unit):
    from nasdaq_agent.report.grounding import check_grounding
    sep = "" if unit == "%" else " "
    res = check_grounding(_n(f"The estimate uses 2026{sep}{unit} as a baseline."), verified(), [], exempt_years={2026, 2025})
    assert not res.ok, unit


def test_exempt_year_followed_by_percent_with_or_without_space_fails():
    from nasdaq_agent.report.grounding import check_grounding
    for phrase in ("The estimate uses 2026% as a baseline.", "The estimate uses 2026 % as a baseline."):
        res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
        assert not res.ok, phrase


@pytest.mark.parametrize("currency", ["$", "\u20ac", "\u00a3"])
@pytest.mark.parametrize("spacing", ["", " "])
def test_exempt_year_preceded_by_each_currency_fails(currency, spacing):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(f"Shares traded at {currency}{spacing}2026 apiece."), verified(), [], exempt_years={2026, 2025})
    assert not res.ok, (currency, spacing)


def test_five_percentage_points_exact_finding_text():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("Margins improved five percentage points."), verified(), [], exempt_years={2026, 2025})
    assert not res.ok
    assert any(f == 'quantity "five percentage points" must be a declared number written in digits' for f in res.findings), res.findings


# Label boundaries, the "a" gap, Unicode normalisation, the Unicode-aware hyphen
# rule and common financial labels. Non-ASCII characters are written as escapes so that each
# one is visible in review.

# A label is removed only when it stands alone. A label whose number carries a unit, or runs on
# through a decimal point or a thousands comma, is a quantity, so the label stays and the whole
# number is checked. Accepted trade-off: "the S&P 500 points to a recovery" is rejected too,
# because "points" is a unit.
LABEL_NUMBER_CARRYING_A_UNIT_MUST_FAIL = [
    ("Acme logged an S&P 500% gain.", "500%"),
    ("Acme logged an S&P 500 points gain.", "500"),
    ("Acme rode an S&P 500-point swing.", "500"),
    ("Spreads widened Russell 2000 bps.", "2000"),
    ("Acme logged a Nasdaq-100% gain.", "100%"),
    ("Acme logged a Nasdaq 100 percent gain.", "100"),
    ("Acme reported a Phase 3% response rate.", "3%"),
    ("Acme reported a Phase 3 percent response rate.", "3"),
    ("Acme reported a Phase 2/3% response rate.", "3%"),  # "Phase 2" stands alone; "3%" is checked
    ("Acme reported a Phase 3-percent response rate.", "3"),
]


@pytest.mark.parametrize("phrase, token", LABEL_NUMBER_CARRYING_A_UNIT_MUST_FAIL)
def test_label_number_carrying_a_unit_is_checked(phrase, token):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert f'prose token "{token}" is not a declared, verified value' in res.findings, (phrase, res.findings)


@pytest.mark.parametrize("phrase, token", [
    ("Acme outran the S&P 500.11% move.", "500.11%"),
    ("Acme reported a Phase 3.11% response rate.", "3.11%"),
])
def test_label_number_continued_by_a_decimal_is_checked_whole(phrase, token):
    # With 11.0 declared, removing the label used to leave a ".11%" remainder that passed as 11.
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import DeclaredMetric
    n = _n(phrase).model_copy(update={"declared_metrics": [DeclaredMetric(name="cumulative_return_pct", value=11.0)]})
    res = check_grounding(n, verified(), [], exempt_years={2026, 2025})
    assert f'prose token "{token}" is not a declared, verified value' in res.findings, (phrase, res.findings)


def test_label_number_continued_by_a_thousands_comma_is_checked_whole():
    # With a declared value that rounds to 0, removing the label used to leave a ",000" remainder
    # that passed as 0.
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import DeclaredMetric
    n = _n("Acme matched the S&P 500,000 in volume.").model_copy(
        update={"declared_metrics": [DeclaredMetric(name="session_pct_change", value=0.3)]})
    res = check_grounding(n, verified(), [], extra_facts={"session_pct_change": 0.3}, exempt_years={2026, 2025})
    assert 'prose token "500,000" is not a declared, verified value' in res.findings, res.findings


def test_label_is_never_cut_out_of_the_decimal_part_of_a_number():
    # Module invariant: no substring is ever removed from inside a number. Cutting "8-K" out of
    # "1.8-K" would leave a "1" that matches the declared 0.943 at 0 decimals.
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import DeclaredMetric
    n = _n("Acme's ratio reached 1.8-K.").model_copy(
        update={"declared_metrics": [DeclaredMetric(name="daily_changes_pct[3]", value=0.943)]})
    res = check_grounding(n, verified(), [], exempt_years={2026, 2025})
    assert 'prose token "1.8" is not a declared, verified value' in res.findings, res.findings


# The "a" gap before a unit needs a separator after the "a", so "app" is never read as
# "a" plus the unit "pp".
@pytest.mark.parametrize("phrase", [
    "Acme's one app strategy drew praise.",
    "Acme launched a one-app bundle.",
])
def test_word_beginning_with_a_is_not_the_a_gap_before_a_unit(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert res.ok, (phrase, res.findings)


@pytest.mark.parametrize("phrase, quantity", [
    ("Margins improved half a percent.", "half a percent"),
    ("Margins improved half-a-percent.", "half-a-percent"),
    ("Acme announced a half-a-point cut.", "half-a-point"),
])
def test_a_gap_with_a_separator_still_attaches_the_unit(phrase, quantity):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert f'quantity "{quantity}" must be a declared number written in digits' in res.findings, (phrase, res.findings)


# NFKC normalisation first (full-width digits and symbols become ASCII), then every dash
# look-alike becomes "-", then invisible characters are deleted, all before tokenizing.
UNICODE_SPELLING_PHRASES_THAT_MUST_FAIL = [
    "Margins improved one\u2013percent.",                                # U+2013 en dash
    "Acme announced 2026\u2013percent growth.",                          # U+2013 en dash
    "Acme announced a five basis\u2014point cut.",                       # U+2014 em dash
    "The estimate uses 2026\uff05 as a baseline.",                       # U+FF05 full-width percent
    "The estimate uses \uff12\uff10\uff12\uff16\uff05 as a baseline.",   # full-width digits and percent
    "Shares traded at \uff042026 apiece.",                               # U+FF04 full-width dollar
]


@pytest.mark.parametrize("phrase", UNICODE_SPELLING_PHRASES_THAT_MUST_FAIL)
def test_unicode_spellings_of_units_and_currency_are_normalised(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert not res.ok, phrase


# U+2010 hyphen, U+2011 non-breaking hyphen, U+2012 figure dash, U+2013 en dash, U+2014 em dash,
# U+FE63 small hyphen-minus, U+FF0D full-width hyphen-minus.
DASH_LOOKALIKES = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\ufe63", "\uff0d"]


@pytest.mark.parametrize("dash", DASH_LOOKALIKES)
def test_every_dash_lookalike_joins_a_year_to_its_unit(dash):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(f"Acme announced 2026{dash}percent growth."), verified(), [], exempt_years={2026, 2025})
    assert not res.ok, f"U+{ord(dash):04X}"


@pytest.mark.parametrize("dash", DASH_LOOKALIKES)
def test_every_dash_lookalike_between_years_is_a_range_hyphen(dash):
    # Includes the pass case "Guidance covers 2025\u20132026." (en dash range).
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(f"Guidance covers 2025{dash}2026."), verified(), [], exempt_years={2026, 2025})
    assert res.ok, (f"U+{ord(dash):04X}", res.findings)


# U+00AD soft hyphen, U+200B zero-width space, U+200C zero-width non-joiner, U+200D zero-width
# joiner, U+2060 word joiner, U+FEFF zero-width no-break space.
INVISIBLE_CHARACTERS = ["\u00ad", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]


@pytest.mark.parametrize("invisible", INVISIBLE_CHARACTERS)
def test_invisible_character_inside_a_number_is_deleted(invisible):
    # Without the deletion, "20" and "26%" would be checked as two separate numbers.
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(f"The estimate uses 20{invisible}26% as a baseline."), verified(), [], exempt_years={2026, 2025})
    assert 'prose token "2026%" is not a declared, verified value' in res.findings, (f"U+{ord(invisible):04X}", res.findings)


def test_unicode_minus_sign_is_not_a_dash_lookalike():
    # U+2212 keeps its own handling: always a sign, even straight after a letter, so a signed
    # year is never exempt.
    from nasdaq_agent.report.grounding import numeric_tokens
    assert numeric_tokens("Acme targets mid\u22122026.", exempt_years={2026}) == ["-2026"]


# A hyphen after any letter or digit, ASCII or not, is a hyphen, not a minus sign.
@pytest.mark.parametrize("phrase, exempt_years", [
    ("Nestl\u00e9-2026 guidance was reaffirmed.", {2026}),         # precomposed e-acute
    ("Soci\u00e9t\u00e9-2025 results were restated.", {2025}),
    ("Nestle\u0301-2026 guidance was reaffirmed.", {2026}),        # e + combining acute, composed by NFKC
])
def test_hyphen_after_a_non_ascii_letter_is_a_hyphen(phrase, exempt_years):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years=exempt_years)
    assert res.ok, (phrase, res.findings)


# Common financial labels are removed like the index-name and trial-phase labels above, with the
# same boundaries and lookaheads, so each is removed only when it stands alone.
FINANCIAL_LABEL_PHRASES_THAT_MUST_PASS = [
    "Q3 revenue beat estimates.",
    "The company filed an 8-K.",
    "Its 10-K showed higher margins.",
    "Acme priced an S-1 filing.",
    "Acme joined the Russell 1000.",
    "Acme outperformed the S&P500.",
    "Acme reported a Phase-3 readout.",
    "Acme began a Phase 1b/2a study.",
    # The remaining label alternatives, so that every one is exercised.
    "Q1, Q2 and Q4 sales all grew.",
    "H1 sales grew and H2 guidance rose.",
    "Its 10-Q and 20-F filings were cited.",
    "A 6-K arrived before the S-3 and S-4 filings.",
    "Directors filed a Form 4 on Monday.",
    "Acme joined the Russell 3000.",
]


@pytest.mark.parametrize("phrase", FINANCIAL_LABEL_PHRASES_THAT_MUST_PASS)
def test_financial_labels_are_removed_before_tokenizing(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert res.ok, (phrase, res.findings)


def test_quarter_label_keeps_the_year_that_follows_it():
    from nasdaq_agent.report.grounding import numeric_tokens
    assert numeric_tokens("Q3 2026 revenue beat estimates.") == ["2026"]
    assert numeric_tokens("Q3 2026 revenue beat estimates.", exempt_years={2026}) == []


def test_other_numbers_next_to_a_financial_label_are_still_checked():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("Q3 rose 5%."), verified(), [], exempt_years={2026, 2025})
    assert res.findings == ['prose token "5%" is not a declared, verified value']


@pytest.mark.parametrize("phrase, token", [
    ("Acme guided to Q3.5% growth.", "3.5%"),
    ("Acme cited an 8-K% figure.", "8"),
])
def test_financial_label_continued_by_a_number_or_carrying_a_unit_is_checked(phrase, token):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert f'prose token "{token}" is not a declared, verified value' in res.findings, (phrase, res.findings)


# Spaced dash asides, a case-sensitive "Form 4", multi-word units with no separator,
# "_" in place of a removed label, guarded ISO-date removal, and citations removed before labels.

# A figure dash, en dash or em dash with whitespace on both sides is an aside, not a hyphen,
# so it never joins what comes before it to a unit after it.
SPACED_DASH_ASIDE_PHRASES_THAT_MUST_PASS = [
    "Acme expects approval in 2026 \u2014 points of contention remain.",
    "Momentum in the second half \u2014 points to more upside.",
    "Acme outperformed the S&P 500 \u2014 points worth noting.",
    "Guidance for 2026 \u2013 dollar sales aside \u2013 was raised.",
]


@pytest.mark.parametrize("phrase", SPACED_DASH_ASIDE_PHRASES_THAT_MUST_PASS)
def test_spaced_dash_aside_does_not_join_a_unit(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert res.ok, (phrase, res.findings)


@pytest.mark.parametrize("dash", ["\u2012", "\u2013", "\u2014"])
def test_each_spaced_aside_dash_is_an_aside(dash):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(f"Acme expects approval in 2026 {dash} points of contention remain."), verified(), [],
                          exempt_years={2026, 2025})
    assert res.ok, (f"U+{ord(dash):04X}", res.findings)


# Only those three dashes, and only with whitespace on both sides, are asides. Anything else is
# still a hyphen that attaches the unit to the year.
@pytest.mark.parametrize("phrase", [
    "Acme rose 2026 \u2014percent this year.",   # whitespace before the dash only
    "Acme rose 2026\u2014 percent this year.",   # whitespace after the dash only
    "Acme rose 2026 - percent this year.",         # ASCII hyphen with whitespace on both sides
    "Acme rose 2026 \u2011 percent this year.",  # non-breaking hyphen with whitespace on both sides
])
def test_other_spaced_or_one_sided_dashes_still_attach_the_unit(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert 'prose token "2026" is not a declared, verified value' in res.findings, (phrase, res.findings)


# "Form" is case-sensitive in the "Form 4" label, so the verb "form" never hides a number.
def test_lowercase_form_followed_by_a_number_is_not_a_label():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import DeclaredMetric
    n = _n("Together they form 4 of the top 10 holdings.").model_copy(
        update={"declared_metrics": [DeclaredMetric(name="prev_close", value=10.0)]})
    res = check_grounding(n, verified(), [], extra_facts={"prev_close": 10.0}, exempt_years={2026, 2025})
    assert res.findings == ['prose token "4" is not a declared, verified value']


def test_capitalised_form_4_is_still_a_label():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("Acme's CFO made a Form 4 filing."), verified(), [], exempt_years={2026, 2025})
    assert res.ok, res.findings


# The words of a multi-word unit may touch, so deleting an invisible character between them
# can no longer glue them into a word that the unit pattern misses.
@pytest.mark.parametrize("phrase, finding", [
    ("Margins improved one percentage\u200bpoint.",
     'quantity "one percentagepoint" must be a declared number written in digits'),
    ("Acme announced 2026 percentage\u00adpoints of growth.", 'prose token "2026" is not a declared, verified value'),
    ("Acme announced 2026 basispoints of tightening.", 'prose token "2026" is not a declared, verified value'),
])
def test_multi_word_unit_with_no_separator_is_still_a_unit(phrase, finding):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert finding in res.findings, (phrase, res.findings)


# A removed label becomes "_", a word character, so a hyphen after it is still a hyphen (as after
# a letter or digit) rather than a minus sign on the number that follows.
@pytest.mark.parametrize("phrase", [
    "Acme raised its Q3-2026 guidance.",
    "Acme's H1-2026 results beat estimates.",
])
def test_hyphen_after_a_removed_label_is_still_a_hyphen(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026})
    assert res.ok, (phrase, res.findings)


def test_number_after_a_hyphenated_trial_phase_is_checked_unsigned():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("Acme began a Phase 2-3 trial."), verified(), [], exempt_years={2026})
    assert res.findings == ['prose token "3" is not a declared, verified value']


# An ISO date is removed only when it stands alone. It must not start right after a digit
# and a decimal point or comma, must not be continued by a decimal point or comma and a digit, and
# must not be followed by a percent sign. Only "%" is checked after a date, so "the 2026-09-24
# dollar volume" still has its date removed.
@pytest.mark.parametrize("phrase, token", [
    ("Acme rose 2026-10-15% on the day.", "15%"),
    ("Acme rose 2026-10-15.5% on the day.", "15.5%"),
    ("Acme rose 2026\u201310\u201315% on the day.", "15%"),
])
def test_date_shaped_run_carrying_a_percent_or_a_decimal_is_checked(phrase, token):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert f'prose token "{token}" is not a declared, verified value' in res.findings, (phrase, res.findings)


def test_date_is_never_cut_out_of_a_thousands_group():
    # Removing "0002-09-24" from "11,0002-09-24" would leave "11,", which matches the declared 11.0.
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import DeclaredMetric
    n = _n("Acme rose 11,0002-09-24").model_copy(
        update={"declared_metrics": [DeclaredMetric(name="cumulative_return_pct", value=11.0)]})
    res = check_grounding(n, verified(), [], exempt_years={2026, 2025})
    assert 'prose token "11,000" is not a declared, verified value' in res.findings, res.findings


def test_date_is_never_cut_out_of_the_decimal_part_of_a_number():
    # Removing "2026-10-15" from "1.2026-10-15" would leave "1.", which matches the declared 0.943
    # at 0 decimals.
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import DeclaredMetric
    n = _n("Acme's ratio reached 1.2026-10-15").model_copy(
        update={"declared_metrics": [DeclaredMetric(name="daily_changes_pct[3]", value=0.943)]})
    res = check_grounding(n, verified(), [], exempt_years={2026, 2025})
    assert 'prose token "1.2026" is not a declared, verified value' in res.findings, res.findings


def test_date_followed_by_a_unit_word_is_still_removed():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("Traders watched the 2026-09-24 dollar volume."), verified(), [], exempt_years={2026, 2025})
    assert res.ok, res.findings


# Citation markers are removed before labels, so a citation between a label and a unit can
# no longer hide the unit from the label's unit lookahead.
def test_citation_between_a_label_and_a_unit_does_not_hide_the_unit():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("Acme logged an S&P 500[1]% gain."), verified(), [], exempt_years={2026, 2025})
    assert 'prose token "500" is not a declared, verified value' in res.findings, res.findings


def test_citation_after_a_label_still_lets_the_label_be_removed():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("Acme outperformed the S&P 500 [1]."), verified(), [], exempt_years={2026, 2025})
    assert res.ok, res.findings


# Headline notes: one note per headline on how it may relate to the price move. The email lists the headlines in the
# order of the agent's notes, so grounding checks that the notes cover every headline once and run most recent first,
# and checks their numbers like the paragraphs'.

def dated_headlines():
    """Three headlines published on different days, so the only most-recent-first order is 2, 3, 1."""
    from nasdaq_agent.sources.models import Headline
    return [Headline(id=1, title="Acme prices share offering", provider="GlobeNewswire",
                     published=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)),
            Headline(id=2, title="Acme wins FDA clearance", provider="Business Wire",
                     published=datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)),
            Headline(id=3, title="Acme presents at investor conference", provider="Reuters",
                     published=datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc))]


def _with_notes(*ids, text="The kind of news that can move a small stock."):
    from nasdaq_agent.report.schemas import HeadlineNote, Narrative
    return Narrative(trend_paragraph="Flat.", news_paragraph="", declared_metrics=[], citations=[],
                     headline_notes=[HeadlineNote(headline_ids=[i], summary=text) for i in ids])


def test_headline_notes_listed_most_recent_first_pass():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_with_notes(2, 3, 1), verified(), dated_headlines())
    assert res.ok, res.findings


def test_headline_notes_out_of_order_are_rejected_with_an_order_that_works():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_with_notes(1, 2, 3), verified(), dated_headlines())
    joined = "\n".join(res.findings)
    assert not res.ok and "most recent first" in joined and "[2], [3], [1]" in joined


def test_a_headline_without_a_publication_time_goes_last():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.sources.models import Headline
    headlines = dated_headlines() + [Headline(id=4, title="Acme named in sector roundup", provider="Yahoo")]
    assert check_grounding(_with_notes(2, 3, 1, 4), verified(), headlines).ok
    assert not check_grounding(_with_notes(4, 2, 3, 1), verified(), headlines).ok


def test_every_headline_needs_exactly_one_note():
    from nasdaq_agent.report.grounding import check_grounding
    missing = "\n".join(check_grounding(_with_notes(2, 3), verified(), dated_headlines()).findings)
    assert "headline 1 has no headline note" in missing
    duplicate = "\n".join(check_grounding(_with_notes(2, 2, 3, 1), verified(), dated_headlines()).findings)
    assert "headline 2 is listed more than once" in duplicate
    unknown = "\n".join(check_grounding(_with_notes(2, 3, 1, 9), verified(), dated_headlines()).findings)
    assert "headline note is for headline 9, which does not exist" in unknown


def test_headline_notes_must_be_empty_without_headlines():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_with_notes(1), verified(), [])
    assert not res.ok and "leave headline_notes empty" in "\n".join(res.findings)


def test_numbers_in_headline_notes_must_be_declared():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import DeclaredMetric
    undeclared = _with_notes(2, 3, 1, text="It came before an 11.0% five-day gain.")
    joined = "\n".join(check_grounding(undeclared, verified(), dated_headlines()).findings)
    assert 'headline note for [2]: prose token "11.0%" is not a declared, verified value' in joined
    declared = undeclared.model_copy(update={"declared_metrics": [DeclaredMetric(name="cumulative_return_pct", value=11.0)]})
    res = check_grounding(declared, verified(), dated_headlines())
    assert res.ok, res.findings


def test_number_words_in_headline_notes_are_flagged():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_with_notes(2, 3, 1, text="The deal could be worth a million dollars."), verified(), dated_headlines())
    joined = "\n".join(res.findings)
    assert not res.ok and "headline note for [2]" in joined and "million" in joined


def test_a_headline_note_may_only_cite_headlines_that_exist():
    from nasdaq_agent.report.grounding import check_grounding
    ok = check_grounding(_with_notes(2, 3, 1, text="It follows the offering [1]."), verified(), dated_headlines())
    assert ok.ok, ok.findings
    res = check_grounding(_with_notes(2, 3, 1, text="It follows the offering [7]."), verified(), dated_headlines())
    assert "headline note for [2] cites headline 7, which does not exist" in "\n".join(res.findings)


def test_a_blank_headline_note_is_invalid():
    from pydantic import ValidationError
    from nasdaq_agent.report.schemas import HeadlineNote
    with pytest.raises(ValidationError):
        HeadlineNote(headline_ids=[1], summary="   ")


def test_a_headline_note_has_a_length_limit():
    from pydantic import ValidationError
    from nasdaq_agent.report.schemas import HEADLINE_NOTE_MAX_CHARS, HeadlineNote
    HeadlineNote(headline_ids=[1], summary="x" * HEADLINE_NOTE_MAX_CHARS)
    with pytest.raises(ValidationError):
        HeadlineNote(headline_ids=[1], summary="x" * (HEADLINE_NOTE_MAX_CHARS + 1))


# Consolidation: headlines gathered from several sources can report the same story. The agent lists all their ids in
# one note, and the email shows the story once, tagged with every source.

def multi_source_headlines():
    """Headline 1 (Massive) and 3 (yfinance) report the same offering under different titles; 2 is a different story;
    4 is a word-for-word copy of 2's title from another source."""
    from nasdaq_agent.sources.models import Headline
    t = lambda d, h: datetime(2026, 9, d, h, 0, tzinfo=timezone.utc)
    return [Headline(id=1, title="Acme prices $5 million registered direct offering", provider="GlobeNewswire",
                     source="massive", published=t(23, 13), url="https://example.com/offering"),
            Headline(id=2, title="Acme wins FDA clearance", provider="Business Wire", source="massive", published=t(24, 12)),
            Headline(id=3, title="Acme Corp announces pricing of registered direct offering", provider="Yahoo Finance",
                     source="yfinance", published=t(23, 20)),
            Headline(id=4, title="ACME wins FDA clearance!", provider="Zacks", source="yfinance", published=t(24, 15))]


def _stories(*groups, text="The kind of news that can move a small stock."):
    from nasdaq_agent.report.schemas import HeadlineNote, Narrative
    return Narrative(trend_paragraph="Flat.", news_paragraph="", declared_metrics=[], citations=[],
                     headline_notes=[HeadlineNote(headline_ids=list(g), summary=text) for g in groups])


def test_one_note_may_consolidate_a_story_reported_by_several_sources():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_stories((2, 4), (1, 3)), verified(), multi_source_headlines())
    assert res.ok, res.findings


def test_stories_are_ordered_by_when_they_first_appeared():
    # Story [2, 4] first appeared at 12:00 on the 24th, story [1, 3] at 13:00 on the 23rd, so [2, 4] comes first even
    # though headline 3 is later than headline 1.
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_stories((1, 3), (2, 4)), verified(), multi_source_headlines())
    assert "in this order: [2, 4], [1, 3]" in "\n".join(res.findings)


def test_headlines_with_the_same_title_must_share_a_note():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_stories((4,), (2,), (1, 3)), verified(), multi_source_headlines())
    assert "headlines 2 and 4 have the same title" in "\n".join(res.findings)


def test_headlines_with_the_same_link_must_share_a_note():
    from nasdaq_agent.report.grounding import check_grounding
    headlines = multi_source_headlines()
    headlines[2] = headlines[2].model_copy(update={"url": "https://example.com/offering"})
    res = check_grounding(_stories((2, 4), (3,), (1,)), verified(), headlines)
    assert "headlines 1 and 3 have the same link" in "\n".join(res.findings)


def test_a_headline_listed_twice_in_one_note_is_rejected():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_stories((2, 4, 2), (1, 3)), verified(), multi_source_headlines())
    assert "headline 2 is listed more than once" in "\n".join(res.findings)


def test_a_story_note_names_all_its_headlines_in_findings():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_stories((2, 4), (1, 3), text="Priced at a 12% discount."), verified(), multi_source_headlines())
    assert 'headline note for [2, 4]: prose token "12%"' in "\n".join(res.findings)


# Sentence splitting. A live run's model cited mid-sentence ("a sector update [1] and a company announcement ... [2].");
# the old splitter broke the sentence after "[1]", every later citation index shifted by one, and a correctly cited
# paragraph failed grounding three times and fell back to template prose.

LIVE_RUN_PARAGRAPH = ("Two stories touched the stock around the session: a late-afternoon sector update [1] and a company "
                      "announcement of a non-binding letter of intent for a stake in Kazakhstan's Sarybulak oil field [2]. "
                      "The sector note was timed at the closing bell, so it can only reflect the session's move, and it "
                      "describes energy names falling rather than rallying [1]. The company's letter of intent, published "
                      "before the close, is the item that may relate to the price move, as context and not a proven "
                      "cause [2].")


def test_a_citation_marker_inside_a_sentence_does_not_end_it():
    from nasdaq_agent.report.grounding import split_sentences
    sentences = split_sentences(LIVE_RUN_PARAGRAPH)
    assert len(sentences) == 3 and sentences[0].endswith("oil field [2].")


def test_a_marker_after_the_full_stop_stays_with_its_sentence():
    from nasdaq_agent.report.grounding import split_sentences
    assert split_sentences("It filed. [1] It raised guidance. [2][3] Done!") == ["It filed. [1]", "It raised guidance. [2][3]", "Done!"]


def test_decimals_and_a_final_sentence_without_punctuation_are_handled():
    from nasdaq_agent.report.grounding import split_sentences
    assert split_sentences("It closed at 2.29 after 0.415. Volume rose [1]") == ["It closed at 2.29 after 0.415.", "Volume rose [1]"]


def test_the_live_run_paragraph_passes_with_the_model_s_own_citations():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import Citation, HeadlineNote, Narrative
    from nasdaq_agent.sources.models import Headline
    headlines = [Headline(id=1, title="Sector Update: Energy Stocks Fall Late Afternoon", provider="MT Newswires",
                          published=datetime(2026, 9, 16, 20, 0, tzinfo=timezone.utc)),
                 Headline(id=2, title="Delixy Holdings Signs Non-Binding LOI", provider="InvestorsHub",
                          published=datetime(2026, 9, 16, 14, 32, tzinfo=timezone.utc))]
    narrative = Narrative(trend_paragraph="Flat.", news_paragraph=LIVE_RUN_PARAGRAPH, declared_metrics=[],
                          citations=[Citation(sentence_index=0, headline_ids=[1, 2]), Citation(sentence_index=1, headline_ids=[1]),
                                     Citation(sentence_index=2, headline_ids=[2])],
                          headline_notes=[HeadlineNote(headline_ids=[1], summary="Timed at the close."),
                                          HeadlineNote(headline_ids=[2], summary="Company-specific news.")])
    res = check_grounding(narrative, verified(), headlines)
    assert res.ok, res.findings


def test_a_sentence_with_an_inline_marker_counts_as_cited_even_if_the_indices_disagree():
    # "Inc." ends a fragment for the splitter though not for the writer, so the writer's indices run one short. Each
    # fragment that carries a marker, or an index entry, is still cited.
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import Citation, Narrative
    narrative = Narrative(trend_paragraph="Flat.", news_paragraph="Acme Corp Inc. filed for review [1]. It raised guidance [2].",
                          declared_metrics=[], citations=[Citation(sentence_index=0, headline_ids=[1]),
                                                          Citation(sentence_index=1, headline_ids=[2])],
                          headline_notes=good_notes())
    res = check_grounding(narrative, verified(), headlines())
    assert res.ok, res.findings


def test_an_inline_marker_for_a_headline_that_does_not_exist_is_a_finding():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import Narrative
    narrative = Narrative(trend_paragraph="Flat.", news_paragraph="It filed for review [7].", declared_metrics=[],
                          citations=[], headline_notes=good_notes())
    joined = "\n".join(check_grounding(narrative, verified(), headlines()).findings)
    assert "news sentence 1 cites headline 7" in joined and "does not exist" in joined


def test_an_uncited_sentence_is_still_a_finding():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import Citation, Narrative
    narrative = Narrative(trend_paragraph="Flat.", news_paragraph="It filed for review [1]. Analysts were upbeat.",
                          declared_metrics=[], citations=[Citation(sentence_index=0, headline_ids=[1])],
                          headline_notes=good_notes())
    joined = "\n".join(check_grounding(narrative, verified(), headlines()).findings)
    assert 'news sentence 2 ("Analysts were upbeat.") has no citation' in joined


# Plain English. A live email read "volatility_annualized_pct was 3215.827%": the model wrote the metric names it had
# declared. Prose is for readers, so an underscore anywhere in it is a finding that names the words to use.

def test_a_metric_name_in_the_prose_is_a_finding_that_names_the_plain_words():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import DeclaredMetric
    narrative = _n("The volatility_annualized_pct was 52.87%.").model_copy(
        update={"declared_metrics": [DeclaredMetric(name="volatility_annualized_pct", value=52.87)]})
    joined = "\n".join(check_grounding(narrative, verified(), []).findings)
    assert '"volatility_annualized_pct" is not plain English; write "annualised volatility"' in joined


def test_an_indexed_metric_name_and_any_other_underscore_are_findings():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n("The daily_changes_pct[4] move and some_word here."), verified(), [])
    joined = "\n".join(res.findings)
    assert '"daily_changes_pct[4]" is not plain English; write "the daily changes"' in joined
    assert '"some_word" is not plain English; use plain words' in joined


def test_every_declarable_name_with_an_underscore_has_plain_words():
    from nasdaq_agent.metrics import METRIC_NAMES
    from nasdaq_agent.report.grounding import PLAIN_WORDS
    declarable = [*METRIC_NAMES, "session_pct_change", "prev_close", "close"]
    assert [name for name in declarable if "_" in name and name not in PLAIN_WORDS] == []


def test_plain_english_is_checked_in_headline_notes_too():
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_with_notes(2, 3, 1, text="It lifted the prev_close sharply."), verified(), dated_headlines())
    assert 'headline note for [2]: "prev_close" is not plain English' in "\n".join(res.findings)


# Figures from the story itself. A story summary may quote a number its own headlines state ("a 48% stake", "$1.5
# million"); any other number must still be a declared metric.

def summarised_headlines():
    from nasdaq_agent.sources.models import Headline
    return [Headline(id=1, title="Delixy Holdings Signs Non-Binding LOI for Up to 48% Stake in Kazakhstan's Sarybulak Oil Field",
                     provider="InvestorsHub", published=datetime(2026, 9, 16, 14, 32, tzinfo=timezone.utc),
                     summary="Delixy signed a non-binding letter of intent involving up to 48% of Tarbagatay Munay."),
            Headline(id=2, title="DataMEDS AI Completes $1.5 Million Helomics Acquisition", provider="InvestorsHub",
                     published=datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc))]


def test_a_story_may_quote_the_figures_its_own_headlines_state():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import HeadlineNote, Narrative
    narrative = Narrative(trend_paragraph="Flat.", news_paragraph="", declared_metrics=[], citations=[],
                          headline_notes=[HeadlineNote(headline_ids=[1], summary="Delixy agreed to pursue up to 48% of the operator."),
                                          HeadlineNote(headline_ids=[2], summary="It completed a $1.5 million acquisition.")])
    res = check_grounding(narrative, verified(), summarised_headlines())
    assert res.ok, res.findings


def test_a_story_may_not_quote_figures_from_another_story_or_nowhere():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import HeadlineNote, Narrative
    narrative = Narrative(trend_paragraph="Flat.", news_paragraph="", declared_metrics=[], citations=[],
                          headline_notes=[HeadlineNote(headline_ids=[1], summary="Worth $1.5 million, for 60% of the field."),
                                          HeadlineNote(headline_ids=[2], summary="A 48% stake followed.")])
    joined = "\n".join(check_grounding(narrative, verified(), summarised_headlines()).findings)
    assert 'headline note for [1]: prose token "1.5"' in joined and 'headline note for [1]: prose token "60%"' in joined
    assert 'headline note for [2]: prose token "48%"' in joined


def test_a_news_sentence_may_quote_figures_from_the_headlines_it_cites():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import Citation, HeadlineNote, Narrative
    notes = [HeadlineNote(headline_ids=[1], summary="Oil deal."), HeadlineNote(headline_ids=[2], summary="Acquisition.")]
    cited = Narrative(trend_paragraph="Flat.", news_paragraph="Delixy may take up to 48% of the operator [1].",
                      declared_metrics=[], citations=[Citation(sentence_index=0, headline_ids=[1])], headline_notes=notes)
    assert check_grounding(cited, verified(), summarised_headlines()).ok
    uncited = cited.model_copy(update={"news_paragraph": "A 48% stake was mentioned [2].",
                                       "citations": [Citation(sentence_index=0, headline_ids=[2])]})
    assert 'prose token "48%"' in "\n".join(check_grounding(uncited, verified(), summarised_headlines()).findings)


def test_the_trend_paragraph_may_not_quote_headline_figures():
    from nasdaq_agent.report.grounding import check_grounding
    from nasdaq_agent.report.schemas import HeadlineNote, Narrative
    narrative = Narrative(trend_paragraph="A 48% stake lifted it.", news_paragraph="", declared_metrics=[], citations=[],
                          headline_notes=[HeadlineNote(headline_ids=[1], summary="Oil deal."),
                                          HeadlineNote(headline_ids=[2], summary="Acquisition.")])
    assert 'prose token "48%"' in "\n".join(check_grounding(narrative, verified(), summarised_headlines()).findings)


# Times and dates. Stories weigh when news appeared against the session, so a clock time ("10:32") or a month-and-day
# date ("September 16") is not a quantity; a percentage beside a month name still is.

@pytest.mark.parametrize("phrase", ["It was published at 10:32 ET, before the close.",
                                    "It came out on September 16, the session day.",
                                    "It came out on Sept. 16 at 4:00 p.m.",
                                    "It came out on 16 September."])
def test_clock_times_and_month_day_dates_are_not_quantities(phrase):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert res.ok, (phrase, res.findings)


@pytest.mark.parametrize("phrase, token", [("In May 5% of float traded.", "5%"), ("It rose 16:00% overnight.", "16"),
                                           ("By 12:30 it was up 12%.", "12%")])
def test_a_quantity_beside_a_time_or_month_is_still_checked(phrase, token):
    from nasdaq_agent.report.grounding import check_grounding
    res = check_grounding(_n(phrase), verified(), [], exempt_years={2026, 2025})
    assert f'prose token "{token}' in "\n".join(res.findings), (phrase, res.findings)
