"""Deterministic grounding check for the model-authored narrative.

The model writes two prose paragraphs for the email (trend + news) and one note per headline
on how it may relate to the price move. This module never trusts that prose: every number it
contains must match one of the model's own ``declared_metrics`` (each of which must in turn
match a verified, deterministically computed value), every sentence in the news paragraph must
cite a real headline id, and the notes must put every headline in exactly one story, most
recent first, because the email lists one entry per story in the order of the notes.

Headline notes summarise their stories, so a note, or a news sentence, may also quote a figure
its own headlines state ("a 48% stake", "$1.5 million"); the trend paragraph may not. Clock
times and month-and-day dates are not quantities (TIME_PATTERN, MONTH_DAY_PATTERN), and an
underscore anywhere in the prose is a finding: it means a metric name leaked into text meant for
readers (PLAIN_WORDS).

Review fix (Critical), full redesign of year and number-word handling. The previous approach
stripped year-shaped substrings out of the text with a regex .sub() before tokenizing numbers,
which corrupted numbers a year merely happened to be a substring of (e.g. "1950.2%" had "1950"
cut out, leaving the stray, unchecked token "2"), and only recognised a handful of unit words,
so many spellings of the same magnitude (per cent, pct, bp, basis points, ...), a currency
prefix ($2015, 2015 dollars), and most number words (six, twenty, hundred, trillion, ...) all
went unchecked. The design here never removes a substring from inside a number: every complete
number is tokenized first (NUMBER_PATTERN, unchanged in shape), and each token is then
classified in place from its own shape and immediate context. A year is exempt only when the
caller explicitly says which years are in play (`exempt_years`) -- there is no default "any
19xx/20xx looks like a year" behaviour any more.

Fix round 3: a hyphen directly after a letter or digit is a word-joiner, not a minus sign
(NUMBER_PATTERN); the "a " gap before a unit only ever applies after a number word, never after
a numeric token (UNIT_AFTER_NUMBER_PATTERN vs UNIT_AFTER_WORD_PATTERN); U+2010/U+2011 hyphen
look-alikes are normalised to "-" and multi-word units accept a hyphen or whitespace between
their words (UNIT_SOURCE); and proper-noun labels that are not quantities (index names, trial
phases) are removed before tokenizing, the same way dates and citations are (PROPER_NOUN_PATTERN).

Fix round 4: a label is removed only when it stands alone -- never when a decimal point or a
thousands comma continues its number, when a unit follows it, or when it starts right after a
digit and a decimal point -- and the label list gains quarter and half labels, SEC form names,
more index names and more trial-phase spellings (PROPER_NOUN_PATTERN); the "a" gap before a unit
needs a separator after the "a" (UNIT_AFTER_WORD_PATTERN); the text is NFKC-normalised, every
dash look-alike becomes "-" and invisible characters are deleted before tokenizing (_preprocess);
and a hyphen after any Unicode letter or digit is a hyphen, not a minus sign (NUMBER_PATTERN).

Fix round 5: a figure, en or em dash with whitespace on both sides is an aside and becomes ";"
(SPACED_DASH_ASIDE_PATTERN); "Form" in the "Form 4" label is case-sensitive (LABEL_SOURCE); the
words of a multi-word unit may touch (UNIT_SOURCE); a removed label becomes "_", so a hyphen after
it is still a hyphen; ISO-date removal has stand-alone guards like the label pattern
(DATE_PATTERN); and citation markers are removed before labels (_preprocess).
"""
import re
import unicodedata
from decimal import Decimal
from typing import NamedTuple

from ..metrics import METRIC_NAMES, PCT_TOLERANCE, RETURNS_PER_WINDOW, AnalysisResult
from ..sources.models import Headline, first_published, recency_key
from .schemas import GroundingResult, HeadlineNote, Narrative

# Fix round 5, item 6: an ISO date is removed only when it stands alone, like a label (see
# PROPER_NOUN_PATTERN below). It must not start right after a digit and a decimal point or comma,
# where its digits would continue a number: "1.2026-10-15", or "11,0002-09-24", where ",000" is
# the thousands group of "11,000". Unlike a label, a date starts with four digits, so the comma
# guard is needed here. It must not be continued by a decimal point or comma and a digit
# ("2026-10-15.5%"), and must not be followed by a percent sign ("2026-10-15%"). Only "%" is
# checked after a date, not every unit, so "the 2026-09-24 dollar volume" still has its date
# removed.
DATE_PATTERN = re.compile(r"(?<!\d\.)(?<!\d,)\b\d{4}-\d{2}-\d{2}\b(?![.,]\d)(?!\s*%)")
CITATION_PATTERN = re.compile(r"\[\d+\]")
# Fix round 5, item 1: a figure dash (U+2012), en dash (U+2013) or em dash (U+2014) with
# whitespace on both sides sets off an aside ("approval in 2026 — points of contention remain"); it
# does not join the words around it. Before the hyphen map below, it becomes ";", which is neither
# whitespace nor a hyphen, so it can never join a number, a number word or a label to a unit after
# it. A dash without whitespace on both sides, and every other hyphen look-alike, stays a hyphen.
SPACED_DASH_ASIDE_PATTERN = re.compile(r"(?<=\s)[\u2012\u2013\u2014](?=\s)")
# Fix round 3, item 3, extended in fix round 4, item 3: every dash that reads as a hyphen is
# treated exactly like the ASCII hyphen, in unit spellings and in years alike: U+2010 HYPHEN,
# U+2011 NON-BREAKING HYPHEN, U+2012 FIGURE DASH, U+2013 EN DASH, U+2014 EM DASH, U+FE63 SMALL
# HYPHEN-MINUS and U+FF0D FULLWIDTH HYPHEN-MINUS. This runs after NFKC normalisation, which also
# folds a few presentation forms (for example U+FE58 SMALL EM DASH) into characters listed here.
# U+2212 (MINUS SIGN) is deliberately NOT included: unlike a hyphen, it is never a word-joiner,
# so it keeps its own always-a-sign handling in NUMBER_PATTERN below.
HYPHEN_LOOKALIKE_PATTERN = re.compile("[\u2010\u2011\u2012\u2013\u2014\ufe63\uff0d]")
# Fix round 4, item 3: invisible characters are deleted before tokenizing, so that one inside a
# number ("20<U+200B>26%") or between a number and its unit cannot split it into pieces that are
# checked separately, or detach the unit so it is not checked at all: U+00AD SOFT HYPHEN, U+200B
# ZERO WIDTH SPACE, U+200C ZERO WIDTH NON-JOINER, U+200D ZERO WIDTH JOINER, U+2060 WORD JOINER
# and U+FEFF ZERO WIDTH NO-BREAK SPACE. NFKC leaves all six unchanged and never produces any of
# them, so deleting them after normalising misses nothing.
INVISIBLE_CHARACTER_PATTERN = re.compile("[\u00ad\u200b\u200c\u200d\u2060\ufeff]")
# Fix round 3, item 1 (regression from round 2): a "-" immediately after a letter or digit is a
# hyphen joining words together ("mid-2026", "2025-2026"), never a minus sign, so it must not be
# consumed as part of the number that follows it -- otherwise "mid-2026" tokenizes as "-2026"
# instead of the bare year "2026". A "-" anywhere else (after whitespace, an opening bracket, or
# at the very start of the text) still keeps its sign ("fell -2.3%"). U+2212, the actual minus
# sign character, is unconditionally a sign regardless of what precedes it -- a hyphen is never
# typeset as U+2212, so there is no ambiguity to resolve for it. Fix round 4, item 4: "letter or
# digit" means any Unicode word character (\w), not only ASCII, so "Nestlé-2026" and
# "Société-2025" end in a hyphen, not in a signed year.
NUMBER_PATTERN = re.compile(r"(?:\u2212|(?<!\w)-)?\d+(?:,\d{3})*(?:\.\d+)?")
# The end of a sentence: sentence punctuation, then any citation markers that trail it ("filed. [1][2]"), then
# whitespace or the end of the text. A marker after the punctuation stays with the sentence it closes (correction 13a).
# A marker inside a sentence ends nothing: a live run's "a sector update [1] and a company announcement [2]." was cut
# after "[1]" by the previous rule, which also split after every "]", so every later citation index was off by one.
SENTENCE_END_PATTERN = re.compile(r"[.!?](?:\s*\[\d+\])*(?=\s|$)")

CURRENCY_SYMBOLS = {"$", "€", "£"}  # $, €, £
# One case-insensitive unit source shared by numeric tokens and number words. "%" is matched
# bare (a trailing \b after a symbol can fail at end-of-string, since neither side is a word
# character there); every word alternative gets its own \b so e.g. "percent" can never match as
# a prefix of an unrelated longer word. Fix round 3, item 3: a multi-word unit's own words may be
# joined by whitespace OR a hyphen ("basis-point", "per-cent", "percentage-point"). Fix round 5,
# item 4: they may also touch ("basispoints"), which is what deleting an invisible character
# between them produces ("percentage<U+200B>point"), so their internal separator is [\s-]*, zero
# or more. Order lists compound phrases before the single word they contain (matters only for
# readability -- regex backtracking finds any matching alternative regardless of order).
UNIT_SOURCE = (
    r"%|percentage[\s-]*points?\b|percentage\b|per[\s-]*cent\b|percent\b|pct\b|"
    r"basis[\s-]*points?\b|bps\b|bp\b|pp\b|points?\b|dollars?\b|cents?\b|usd\b"
)
# Fix round 3, item 4, extended in fix round 4, item 5: labels that merely contain digits are
# not quantities and are removed before tokenizing, case-insensitively, the same way dates and
# citations are: index names, clinical-trial phases, quarter and half labels, and SEC form names.
# Roman numerals are tried longest-first (IV before III before II before I) so e.g. "Phase III"
# is never left with a dangling, unremoved "I" or "II". Fix round 5, item 3: "Form" is matched
# case-sensitively ((?-i:...) inside the case-insensitive pattern), so the verb in "together they
# form 4 of the top 10 holdings" is never read as the SEC form name.
LABEL_SOURCE = (
    r"S&P\s*500|Nasdaq[\s-]+100|Russell\s+[123]000|"
    r"Phase[\s-]+(?:[1-4][ab]?(?:/\d[ab]?)?|IV|III|II|I)|"
    r"Q[1-4]|H[12]|"
    r"10-[KQ]|8-K|20-F|6-K|S-[134]|(?-i:Form)\s+4"
)
# Fix round 4, item 1: a label is removed only when it stands alone, so removing it can never
# swallow a number (the module's rule: never remove a substring from inside a number).
#   - (?<!\d\.): it does not start right after a digit and a decimal point, where a label that
#     begins with digits would be the decimal part of a number ("8-K" in "1.8-K"). A comma needs
#     no guard here: a thousands group is always three digits, and no label starts with three.
#   - \b(?![.,]\d): its number is not continued by a decimal point or a thousands comma
#     ("S&P 500.11%", "S&P 500,000"); removing the label would leave a remainder (".11%" reads
#     as 11) that could match a declared value.
#   - (?![\s-]*(?:UNIT_SOURCE)): no unit follows it, since "S&P 500%" and "Phase 3-percent" are
#     quantities, not labels. Accepted trade-off: "the S&P 500 points to a recovery" is rejected.
# A label that is not removed stays in the text, so its number is tokenized whole and checked.
PROPER_NOUN_PATTERN = re.compile(
    rf"(?<!\d\.)\b(?:{LABEL_SOURCE})\b(?![.,]\d)(?![\s-]*(?:{UNIT_SOURCE}))",
    re.IGNORECASE,
)
# A unit "attaches" to a preceding numeric token across optional whitespace or an optional
# hyphen only -- "one-percent", "a 1950-percent gain" attach through the number word / hyphen,
# never through an inserted "a ".
UNIT_AFTER_NUMBER_PATTERN = re.compile(rf"^[\s-]*(?:{UNIT_SOURCE})", re.IGNORECASE)
# Times and dates in words. Headline notes weigh when news appeared against the session, so a clock time ("10:32",
# "4:00 p.m.") or a month-and-day date ("September 16", "Sept. 16", "16 September") is removed before tokenizing, like
# an ISO date. Guards as for labels: a time or day continued by a digit, or followed by a unit ("16:00%", "May 5%"),
# stays in the text and is checked. Month names are matched with their capital, so the verb "may" is never a month.
TIME_PATTERN = re.compile(r"(?<![\d.:,])\b(?:[01]?\d|2[0-3]):[0-5]\d\b(?![.:,]?\d)(?!\s*%)")
MONTH_SOURCE = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
                r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
MONTH_DAY_PATTERN = re.compile(
    rf"\b{MONTH_SOURCE}\.?\s+[0-3]?\d(?:st|nd|rd|th)?\b(?![.,]\d)(?![\s-]*(?i:{UNIT_SOURCE}))"
    rf"|(?<![\d.,])\b[0-3]?\d(?:st|nd|rd|th)?\s+{MONTH_SOURCE}\b"
)
# Fix round 3, item 2 (regression from round 2): the optional "a " gap applies ONLY after a
# number word ("half a percent", "half-a-percent"), never after a numeric token -- otherwise
# "In 2025 a dollar of revenue cost more." would misread "2025 a dollar" as the year 2025
# attaching to the unit "dollar" through the article, and lose its year exemption. Fix round 3,
# item 3: "a" may itself be joined by whitespace or a hyphen on either side ("half-a-point").
# Fix round 4, item 2: at least one separator must follow the "a" ([\s-]+, not [\s-]*), so a word
# that merely begins with "a" is never split into "a" plus a unit ("one app" is not "one a pp").
UNIT_AFTER_WORD_PATTERN = re.compile(rf"^[\s-]*(?:a[\s-]+)?(?:{UNIT_SOURCE})", re.IGNORECASE)
BARE_YEAR_PATTERN = re.compile(r"^\d{4}$")
WORD_PATTERN = re.compile(r"[A-Za-z]+")

# The full number-word vocabulary (item 3 of the round-2 ruling). one..ninety and dozen/half/
# quarter are allowed as plain counts ("four of five sessions", "the first half", "a five-day
# window") and are a finding only when they're used as a quantity instead -- immediately
# followed by a unit, or immediately preceded by a currency symbol. The magnitude words are
# never plain counts in practice and are always a finding, unit or not ("two trillion dollars",
# "a million users").
ALLOWED_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "dozen", "half", "quarter",
}
MAGNITUDE_WORDS = {"hundred", "thousand", "million", "billion", "trillion"}
ALL_NUMBER_WORDS = ALLOWED_NUMBER_WORDS | MAGNITUDE_WORDS


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for end in SENTENCE_END_PATTERN.finditer(text):
        sentences.append(text[start:end.end()].strip())
        start = end.end()
    sentences.append(text[start:].strip())
    return [s for s in sentences if s]


def _preprocess(text: str) -> str:
    """Normalise the text, then remove citation markers, labels and ISO dates, in that order,
    before any number/word scan. None of the removed spans is ever a number or unit the model
    needs to declare.

    Normalising comes first so every later pattern sees one spelling of each character: NFKC
    folds compatibility forms (full-width digits, and the full-width percent and dollar signs)
    into their ASCII equivalents, a spaced figure, en or em dash (an aside) becomes ";", every
    other dash look-alike becomes "-", and invisible characters are deleted.

    Fix round 5, item 7: citation markers are removed before labels, so one between a label and
    a unit ("S&P 500[1]%") cannot hide the unit from the label's unit lookahead.

    Fix round 5, item 5: a removed label becomes "_", not a space. "_" is a word character, so a
    hyphen straight after a label is still read as a hyphen ("Q3-2026" keeps the bare year 2026),
    and no number, number-word, unit, currency or date pattern ever matches "_"."""
    text = unicodedata.normalize("NFKC", text)
    text = SPACED_DASH_ASIDE_PATTERN.sub(";", text)
    text = HYPHEN_LOOKALIKE_PATTERN.sub("-", text)
    text = INVISIBLE_CHARACTER_PATTERN.sub("", text)
    text = CITATION_PATTERN.sub(" ", text)
    text = PROPER_NOUN_PATTERN.sub("_", text)
    text = DATE_PATTERN.sub(" ", text)
    text = TIME_PATTERN.sub(" ", text)
    return MONTH_DAY_PATTERN.sub(" ", text)


def _is_year_exempt(token: str, before: str, after: str, exempt_years: set[int]) -> bool:
    """A token is a year, not a magnitude to verify, only when it is a bare four-digit run (no
    comma, no decimal -- "2015.40" or "2,015" can never be a year), it is one of the years the
    caller actually named, and nothing around it says it is really a quantity: not immediately
    preceded by a currency symbol ("$2026"), and not immediately followed by a unit ("2026 bps",
    "2026-percent"). The "a "-gap allowance never applies here -- that's for number words only
    (UNIT_AFTER_WORD_PATTERN, checked separately in _number_word_findings)."""
    if not BARE_YEAR_PATTERN.match(token):
        return False
    if int(token) not in exempt_years:
        return False
    if UNIT_AFTER_NUMBER_PATTERN.match(after):
        return False
    if before.rstrip()[-1:] in CURRENCY_SYMBOLS:
        return False
    return True


def _scan_numbers(text: str, exempt_years: set[int]):
    """Yield (bare, raw, is_year_exempt) for every numeric token in text, in order. `bare` has
    thousands commas stripped, ready to compare against a declared value; `raw` keeps an
    immediately-following '%' so a finding can quote the token as written."""
    cleaned = _preprocess(text)
    for m in NUMBER_PATTERN.finditer(cleaned):
        token = m.group(0)
        before, after = cleaned[:m.start()], cleaned[m.end():]
        exempt = _is_year_exempt(token, before, after, exempt_years)
        raw = token + ("%" if after.startswith("%") else "")
        # float() (used by _matches() below) does not understand U+2212 as a sign, only ASCII
        # "-", so the bare comparison form normalises it; `raw` keeps U+2212 so a finding still
        # quotes the sign the model actually typed. (`raw` comes from the text as normalised by
        # _preprocess, so a full-width "２０２６％" is quoted as "2026%".)
        bare = token.replace(",", "").replace("\u2212", "-")
        yield bare, raw, exempt


def numeric_tokens(text: str, exempt_years: set[int] | None = None) -> list[str]:
    """Every numeric token in text that is NOT exempted as a year, bare form, in order."""
    exempt_years = exempt_years or set()
    return [bare for bare, _raw, exempt in _scan_numbers(text, exempt_years) if not exempt]


def _number_word_findings(text: str, allowed_words: frozenset[str] = frozenset()) -> list[str]:
    """Spelled-out numbers never appear in numeric_tokens() (they're not digits), so without
    this check the model could write "twelve percent" or "one thousand" and no declared-value
    comparison would ever catch it. See the module docstring and ALLOWED_NUMBER_WORDS /
    MAGNITUDE_WORDS above for the exact rule."""
    findings = []
    cleaned = _preprocess(text)
    for m in WORD_PATTERN.finditer(cleaned):
        word = m.group(0).lower()
        if word not in ALL_NUMBER_WORDS or word in allowed_words:
            continue
        before, after = cleaned[:m.start()], cleaned[m.end():]
        if word in MAGNITUDE_WORDS:
            findings.append(f'quantity "{word}" must be a declared number written in digits')
            continue
        unit_match = UNIT_AFTER_WORD_PATTERN.match(after)
        if unit_match:
            span = cleaned[m.start():m.end() + unit_match.end()]
            findings.append(f'quantity "{span}" must be a declared number written in digits')
        elif before.rstrip()[-1:] in CURRENCY_SYMBOLS:
            start = len(before.rstrip()) - 1
            findings.append(f'quantity "{cleaned[start:m.end()]}" must be a declared number written in digits')
    return findings


def _verified_lookup(verified: AnalysisResult) -> dict[str, float]:
    table = {name: getattr(verified, name) for name in METRIC_NAMES if name not in ("daily_changes_pct", "trend")}
    for i in range(RETURNS_PER_WINDOW):
        table[f"daily_changes_pct[{i}]"] = verified.daily_changes_pct[i]
    return table


def _matches(token: str, value: float) -> bool:
    decimals = abs(Decimal(token).as_tuple().exponent) if "." in token else 0
    return abs(abs(float(token)) - abs(round(value, decimals))) < 10 ** (-decimals) / 2 + 1e-9


TITLE_NOISE_PATTERN = re.compile(r"[^0-9a-z]+")


def _note_label(note: HeadlineNote) -> str:
    return "[" + ", ".join(str(hid) for hid in note.headline_ids) + "]"


def _normalized_title(title: str) -> str:
    """A title with case, punctuation and spacing removed, so two copies of one wire story compare equal."""
    return " ".join(TITLE_NOISE_PATTERN.sub(" ", title.lower()).split())


def _same_story_marker(a: Headline, b: Headline) -> str | None:
    """"link" or "title" when two headlines are plainly the same story, else None. Subtler duplicates are the agent's
    call, checked by the judge."""
    if a.url and b.url and a.url.strip() == b.url.strip():
        return "link"
    title = _normalized_title(a.title)
    if title and title == _normalized_title(b.title):
        return "title"
    return None


# The plain words for each declarable name, used when prose quotes a name instead of saying what it means.
PLAIN_WORDS = {
    "daily_changes_pct": "the daily changes",
    "avg_daily_change_pct": "the average daily change",
    "cumulative_return_pct": "the cumulative return",
    "volatility_annualized_pct": "annualised volatility",
    "max_drawdown_pct": "the maximum drawdown",
    "relative_vs_spy_pct": "performance relative to the benchmark",
    "session_pct_change": "the session's move",
    "prev_close": "the previous close",
}
IDENTIFIER_PATTERN = re.compile(r"[\w\[\]]*_[\w\[\]]*")


class _Section(NamedTuple):
    """A piece of model prose: the prefix its findings carry, its text, and the headlines whose own figures it may
    quote (the headlines a news sentence cites, or a story's headlines)."""
    prefix: str
    text: str
    sources: list[Headline]


def _identifier_findings(text: str) -> list[str]:
    """Prose is for readers: an underscore means a metric or variable name leaked in ("volatility_annualized_pct"), so
    each one is a finding naming the words to use instead."""
    findings = []
    for token in dict.fromkeys(m.group(0) for m in IDENTIFIER_PATTERN.finditer(text)):
        plain = PLAIN_WORDS.get(token.split("[")[0])
        hint = f'write "{plain}"' if plain else "use plain words"
        findings.append(f'"{token}" is not plain English; {hint} (metric names belong only in declared_metrics)')
    return findings


def _source_figures(sources: list[Headline]) -> tuple[set[Decimal], frozenset[str]]:
    """The numbers and number words the sources' own titles and summaries state, which prose about them may quote."""
    texts = [f"{h.title} {h.summary or ''}" for h in sources]
    figures = {abs(Decimal(bare)) for text in texts for bare, _raw, _exempt in _scan_numbers(text, set())}
    words = frozenset(w.lower() for text in texts for w in WORD_PATTERN.findall(text) if w.lower() in ALL_NUMBER_WORDS)
    return figures, words


def _prose_sections(narrative: Narrative, headlines: list[Headline]) -> list[_Section]:
    """Every piece of model prose, split so each knows its sources. The trend paragraph has none. Each news sentence
    has the headlines it cites, by a marker or a citations entry; its findings stay unprefixed. Each headline note has
    its own headlines, and its findings name them, so the model knows which note to fix."""
    by_id = {h.id: h for h in headlines}
    by_sentence = {c.sentence_index: c.headline_ids for c in narrative.citations}
    sections = [_Section("", narrative.trend_paragraph, [])]
    for idx, sentence in enumerate(split_sentences(narrative.news_paragraph)):
        cited = [int(marker[1:-1]) for marker in CITATION_PATTERN.findall(sentence)] + list(by_sentence.get(idx, []))
        sections.append(_Section("", sentence, [by_id[i] for i in dict.fromkeys(cited) if i in by_id]))
    for note in narrative.headline_notes:
        sections.append(_Section(f"headline note for {_note_label(note)}: ", note.summary,
                                 [by_id[i] for i in dict.fromkeys(note.headline_ids) if i in by_id]))
    return sections


def _headline_note_findings(narrative: Narrative, headlines: list[Headline]) -> list[str]:
    """The email lists one entry per story, in the order of the agent's notes. A story is every headline reporting the
    same information, from whichever sources, so every headline belongs to exactly one note; headlines with the same
    link or title must share one; and the notes run most recent first by when each story first appeared, stories
    without a date last. A note may cite other headlines, like [1], only if they exist."""
    notes = narrative.headline_notes
    if not headlines:
        return ["no headlines were fetched; leave headline_notes empty"] if notes else []
    by_id = {h.id: h for h in headlines}
    findings: list[str] = []
    story_of: dict[int, int] = {}
    stories: list[tuple[HeadlineNote, list[Headline]]] = []
    for index, note in enumerate(notes):
        story: list[Headline] = []
        for hid in note.headline_ids:
            if hid not in by_id:
                findings.append(f"a headline note is for headline {hid}, which does not exist")
            elif hid in story_of:
                findings.append(f"headline {hid} is listed more than once in headline_notes")
            else:
                story_of[hid] = index
                story.append(by_id[hid])
        for marker in CITATION_PATTERN.findall(note.summary):
            cited = int(marker[1:-1])
            if cited not in by_id:
                findings.append(f"headline note for {_note_label(note)} cites headline {cited}, which does not exist")
        if story:
            stories.append((note, story))
    for h in headlines:
        if h.id not in story_of:
            findings.append(f"headline {h.id} has no headline note; list every headline in exactly one note")
    listed = [h for h in headlines if h.id in story_of]
    for i, a in enumerate(listed):
        for b in listed[i + 1:]:
            marker = _same_story_marker(a, b)
            if marker and story_of[a.id] != story_of[b.id]:
                findings.append(f"headlines {a.id} and {b.id} have the same {marker}, so they report the same story; "
                                "list them in one headline note")
    expected = sorted(stories, key=lambda story: recency_key(first_published(story[1])))
    if [note for note, _ in expected] != [note for note, _ in stories]:
        order = ", ".join(_note_label(note) for note, _ in expected)
        findings.append("headline_notes must run most recent first, by when each story first appeared, stories without "
                        f"a date last; in this order: {order}")
    return findings


def check_grounding(narrative: Narrative, verified: AnalysisResult, headlines: list[Headline],
                     extra_facts: dict[str, float] | None = None, exempt_years: set[int] | None = None) -> GroundingResult:
    findings: list[str] = []
    exempt_years = exempt_years or set()
    sections = _prose_sections(narrative, headlines)
    source_figures = [_source_figures(section.sources) for section in sections]
    for section, (_figures, words) in zip(sections, source_figures):
        findings.extend(section.prefix + finding for finding in _identifier_findings(section.text))
        findings.extend(section.prefix + finding for finding in _number_word_findings(section.text, words))

    table = _verified_lookup(verified)
    if extra_facts:
        table.update(extra_facts)
    declared_values: list[float] = []

    for d in narrative.declared_metrics:
        if d.name not in table:
            findings.append(f"unknown metric '{d.name}'; valid names: {sorted(table)}")
            continue
        expected = table[d.name]
        if abs(d.value - expected) > PCT_TOLERANCE:
            # A small epsilon guards against float representation error exactly at the
            # tolerance boundary (e.g. 52.87 vs 52.88), matching the padding _matches() uses.
            other = next((n for n, v in table.items() if abs(v - d.value) <= PCT_TOLERANCE + 1e-9), None)
            hint = f" ({d.value} is {other})" if other else ""
            findings.append(f"declared {d.name} = {d.value} but verified value is {round(expected, 3)}{hint}")
        else:
            declared_values.append(expected)

    # A number must be a declared, verified value, or a figure the section's own sources state ("a 48% stake").
    for section, (figures, _words) in zip(sections, source_figures):
        for bare, raw, exempt in _scan_numbers(section.text, exempt_years):
            if exempt or abs(Decimal(bare)) in figures:
                continue
            if not any(_matches(bare, v) for v in declared_values):
                findings.append(f"{section.prefix}prose token \"{raw}\" is not a declared, verified value")

    known_ids = {h.id for h in headlines}
    by_sentence = {c.sentence_index: c.headline_ids for c in narrative.citations}
    # A sentence is cited by a marker inside it, like [2], or by a citations entry for its index. Accepting either
    # means a residual difference in how the writer and this check count sentences (an abbreviation such as "Inc."
    # ends a sentence here) cannot fail a paragraph whose every sentence carries a marker.
    for idx, sentence in enumerate(split_sentences(narrative.news_paragraph) if headlines else []):
        inline = [int(marker[1:-1]) for marker in CITATION_PATTERN.findall(sentence)]
        ids = list(dict.fromkeys(inline + list(by_sentence.get(idx, []))))
        if not ids:
            findings.append(f"news sentence {idx + 1} (\"{sentence}\") has no citation; put a marker like [1] in it")
            continue
        for hid in ids:
            if hid not in known_ids:
                findings.append(f"news sentence {idx + 1} cites headline {hid} does not exist")
    findings.extend(_headline_note_findings(narrative, headlines))
    return GroundingResult(ok=not findings, findings=findings)
