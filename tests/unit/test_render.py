def make_ctx(**over):
    from nasdaq_agent.report.render import ReportContext
    from tests.unit.test_grounding import verified, headlines, good_narrative
    base = dict(run_id="run-1", session_date="2026-09-24", session_label="official close", symbol="ACME",
                company="Acme Corp", pct_change=3.738, prev_close=10.70, close=11.10, verified=verified(),
                benchmark_symbol="SPY", narrative=good_narrative(), template_prose=False, headlines=headlines(),
                sentiment_label="positive", sentiment_score=0.6, problems=[], warnings=[],
                background=["Top gainer from Massive; prices from Yahoo Finance."], code_hash="abc123", chart_cid="chart@run-1")
    return ReportContext(**{**base, **over})

def test_render_chart_writes_png(tmp_path):
    from nasdaq_agent.report.chart import render_chart
    p = render_chart(["2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"],
                     [10.0, 10.2, 9.95, 10.6, 10.7, 11.1], tmp_path / "chart.png")
    assert p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

def test_render_report_numbers_come_from_verified():
    from nasdaq_agent.report.render import render_report
    subject, text, html = render_report(make_ctx())
    assert "ACME" in subject and "+3.74%" in subject
    for needle in ["11.00%", "52.87%", "-2.45%", "10.20 pp", "uptrend", "cid:chart@run-1", "abc123", "prices from Yahoo Finance"]:
        assert needle in html, needle
    for part in (text, html):  # plain words only: no reviewed-code aside, no sparkline glyphs
        assert "reviewed code" not in part and "▁" not in part and "█" not in part
    assert "\nFive-day metrics\n" in text and "<h3>Five-day metrics</h3>" in html

def test_template_prose_when_narrative_dropped():
    from nasdaq_agent.report.render import render_report, template_paragraphs
    ctx = make_ctx(narrative=None, template_prose=True)
    trend, news = template_paragraphs(ctx)
    assert "11.00%" in trend and "3 headlines" in news
    _, text, html = render_report(ctx)
    assert trend in text and trend in html


STATUS_ALL_GOOD = "everything worked, and every calculated figure was re-checked by separate code."


def test_a_report_without_problems_says_everything_worked_right_under_the_price():
    # The reader's first question is whether to trust today's email, so the answer sits under the price line, above
    # the chart and the metrics.
    from nasdaq_agent.report.render import render_report
    _, text, html = render_report(make_ctx())
    assert ("ACME (Acme Corp): +3.74%  prev close 10.70 -> close 11.10\n"
            f"Report status: {STATUS_ALL_GOOD}\n\n"
            "Five-day metrics\n") in text
    status = f"<strong>Report status:</strong> {STATUS_ALL_GOOD}"
    assert status in html
    assert html.index("prev close 10.70") < html.index(status) < html.index("cid:chart@run-1") < html.index("<h3>Five-day metrics</h3>")


def test_a_problem_turns_the_status_to_needs_attention_and_is_listed_under_it():
    from nasdaq_agent.report.render import render_report
    _, text, html = render_report(make_ctx(problems=["No chart: it could not be drawn."]))
    assert ("Report status: some things need your attention.\n"
            "  - No chart: it could not be drawn.\n\n"
            "Five-day metrics\n") in text
    assert "<strong>Report status:</strong> some things need your attention." in html
    assert "<li>No chart: it could not be drawn.</li>" in html
    assert "everything worked" not in text and "everything worked" not in html


def test_a_warning_is_labelled_as_one_and_leaves_the_status_clean():
    # A warning is worth knowing but says nothing went wrong with the run.
    from nasdaq_agent.report.render import render_report
    _, text, html = render_report(make_ctx(warnings=["The session was an early close."]))
    assert (f"Report status: {STATUS_ALL_GOOD}\n"
            "  - Warning: The session was an early close.\n\n"
            "Five-day metrics\n") in text
    assert "<li><strong>Warning:</strong> The session was an early close.</li>" in html


def test_problems_are_listed_before_warnings():
    from nasdaq_agent.report.render import render_report
    _, text, html = render_report(make_ctx(problems=["No chart: it could not be drawn."],
                                           warnings=["The session was an early close."]))
    assert "  - No chart: it could not be drawn.\n  - Warning: The session was an early close.\n" in text
    assert html.index("No chart: it could not be drawn.") < html.index("The session was an early close.")


def test_the_footer_says_how_the_report_was_made_and_ends_with_its_reference():
    # Background that rarely changes sits at the bottom, after the news, and the run and code references come last so
    # any email can be traced to its run folder.
    from nasdaq_agent.report.render import render_report
    _, text, html = render_report(make_ctx(background=["Top gainer from Massive; prices from Yahoo Finance.",
                                                       "Closes cross-checked against a second source."]))
    assert text.endswith("\nHow this report was made\n"
                         "  Top gainer from Massive; prices from Yahoo Finance.\n"
                         "  Closes cross-checked against a second source.\n\n"
                         "Reference: run run-1, code abc123")
    assert text.index("\n  [1] ") < text.index("How this report was made")
    footer = html[html.index("How this report was made"):]
    assert "Top gainer from Massive; prices from Yahoo Finance.<br>" in footer
    assert "Closes cross-checked against a second source.<br>" in footer
    assert "Reference: run run-1, code abc123" in footer
    assert html.index("<b>[1]</b>") < html.index("How this report was made")


def test_headline_links_only_for_http_and_https():
    # A headline renders as a link only when its URL scheme is http or https. Any other scheme
    # (javascript:, data:, ...) renders the title as plain text, never as an href.
    from nasdaq_agent.report.render import render_report
    from nasdaq_agent.sources.models import Headline
    ctx = make_ctx(headlines=[
        Headline(id=1, title="Safe https", provider="Reuters", url="https://example.com/a"),
        Headline(id=2, title="Safe http", provider="Reuters", url="http://example.com/b"),
        Headline(id=3, title="Upper HTTPS", provider="Reuters", url="HTTPS://EXAMPLE.COM/c"),
        Headline(id=4, title="JS scheme", provider="Reuters", url="javascript:alert(1)"),
        Headline(id=5, title="Data scheme", provider="Reuters", url="data:text/html,<b>x</b>"),
        Headline(id=6, title="No url", provider="Reuters", url=None),
    ])
    _, _, html = render_report(ctx)
    assert '<a href="https://example.com/a">Safe https</a>' in html
    assert '<a href="http://example.com/b">Safe http</a>' in html
    assert '<a href="HTTPS://EXAMPLE.COM/c">Upper HTTPS</a>' in html
    # non-http(s) schemes: no href at all, and the raw URL is never emitted
    assert 'href="javascript' not in html and "javascript:alert(1)" not in html
    assert 'href="data:' not in html and "data:text/html" not in html
    assert "JS scheme" in html and "Data scheme" in html and "No url" in html  # titles still shown


def test_headline_link_treats_malformed_urls_as_plain_text():
    # urlsplit raises ValueError on some provider URLs (e.g. an unclosed or non-address IPv6 literal).
    # headline_link must not raise -- it returns None so the title renders as plain text.
    from nasdaq_agent.report.render import headline_link
    assert headline_link("http://[x]/story") is None
    assert headline_link("http://[::1") is None
    assert headline_link("https://example.com/ok") == "https://example.com/ok"


def test_render_report_survives_a_malformed_headline_url():
    # A malformed headline URL must not crash render_report (which would end the run unsent at exit 1).
    # The report renders, with the malformed one shown as plain text and no href.
    from nasdaq_agent.report.render import render_report
    from nasdaq_agent.sources.models import Headline
    ctx = make_ctx(headlines=[
        Headline(id=1, title="Bad IPv6 URL", provider="Reuters", url="http://[::1"),
        Headline(id=2, title="Fine URL", provider="Reuters", url="https://example.com/ok"),
    ])
    _, _, html = render_report(ctx)  # must not raise
    assert '<a href="https://example.com/ok">Fine URL</a>' in html
    assert "Bad IPv6 URL" in html and 'href="http://[::1"' not in html


def test_html_output_is_escaped():
    # select_autoescape(["html"]) never matches the template name "report.html.j2", so
    # escaping was silently off. Untrusted headline titles and the model's written summary
    # must not pass through as raw HTML/script.
    from nasdaq_agent.report.render import render_report
    from nasdaq_agent.sources.models import Headline
    from tests.unit.test_grounding import good_narrative
    ctx = make_ctx(
        headlines=[Headline(id=1, title="<script>alert(1)</script>", provider="Reuters")],
        narrative=good_narrative().model_copy(update={"trend_paragraph": "Shares moved on the day. <b>big</b> swing."}),
    )
    _, _, html = render_report(ctx)
    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "<b>big</b>" not in html


def test_news_section_says_there_is_no_news_when_there_are_no_headlines():
    # With no headlines the model writes no news paragraph, and a blank section reads like a rendering fault, so the
    # template says so in one plain sentence. It stays true whether none were found, the sources failed, or news was
    # never fetched; the notes under the status line say which.
    from nasdaq_agent.report.render import render_report
    from tests.unit.test_grounding import good_narrative
    narrative = good_narrative().model_copy(update={"news_paragraph": "", "citations": [], "headline_notes": []})
    ctx = make_ctx(headlines=[], narrative=narrative, sentiment_label=None, sentiment_score=None)
    _, text, html = render_report(ctx)
    assert text.count("No news was available for ACME.") == 1
    assert html.count("No news was available for ACME.") == 1
    assert "model-assessed sentiment" not in text


def test_no_news_sentence_is_absent_when_headlines_exist():
    from nasdaq_agent.report.render import render_report
    _, text, html = render_report(make_ctx())
    assert "No news was available" not in text and "No news was available" not in html


def test_template_prose_without_headlines_says_no_news_exactly_once():
    from nasdaq_agent.report.render import render_report
    ctx = make_ctx(headlines=[], narrative=None, template_prose=True, sentiment_label=None, sentiment_score=None)
    _, text, html = render_report(ctx)
    assert text.count("No news was available for ACME.") == 1 and html.count("No news was available for ACME.") == 1


def test_plain_text_news_heading_sits_on_its_own_line():
    # The text template trims the newline after a block tag, which ran the paragraph onto the "News" heading.
    from nasdaq_agent.report.render import render_report
    from tests.unit.test_grounding import good_narrative
    _, text, _ = render_report(make_ctx())
    assert "News (model-assessed sentiment: positive, 0.60)\nThe company filed" in text
    assert "\n  [1] " in text and "\n  [2] " in text  # each headline on its own line
    narrative = good_narrative().model_copy(update={"news_paragraph": "", "citations": [], "headline_notes": []})
    _, text, _ = render_report(make_ctx(headlines=[], narrative=narrative, sentiment_label=None, sentiment_score=None))
    assert "\nNews\nNo news was available for ACME.\n" in text


def _noted(*ids):
    """The good narrative with one note per id, in the given order."""
    from nasdaq_agent.report.schemas import HeadlineNote
    from tests.unit.test_grounding import good_narrative
    notes = [HeadlineNote(headline_ids=[i], summary=f"Note on headline {i}.") for i in ids]
    return good_narrative().model_copy(update={"headline_notes": notes})


def _headline_block(text):
    return text.split("\nHow this report was made")[0].split("\n  [", 1)[1]


def test_headlines_follow_the_agent_note_order_with_each_note_under_its_headline():
    # The agent chooses the order (compose_report has checked it is most recent first) and writes one note per headline.
    from nasdaq_agent.report.render import render_report
    from tests.unit.test_grounding import dated_headlines
    _, text, html = render_report(make_ctx(headlines=dated_headlines(), narrative=_noted(2, 3, 1)))
    block = _headline_block(text)
    assert block.index("Acme wins FDA clearance") < block.index("Acme presents") < block.index("Acme prices share offering")
    assert "  [2] 2026-09-24 08:00 ET  Acme wins FDA clearance\n      Source: Business Wire\n      Note on headline 2.\n  [3] " in text
    assert html.index("<b>[2]</b>") < html.index("<b>[3]</b>") < html.index("<b>[1]</b>")
    assert "Note on headline 2." in html and "Relevance" not in html


def test_template_prose_lists_the_headlines_most_recent_first_without_notes():
    # Template prose has no agent notes, so there is no agent order either: the code lists the headlines most recent
    # first by publication time.
    from nasdaq_agent.report.render import render_report
    from tests.unit.test_grounding import dated_headlines
    _, text, html = render_report(make_ctx(headlines=dated_headlines(), narrative=None, template_prose=True))
    block = _headline_block(text)
    assert block.index("Acme wins FDA clearance") < block.index("Acme presents") < block.index("Acme prices share offering")
    assert "Note on" not in text and "Note on" not in html


def test_headline_times_are_shown_in_eastern_time():
    from nasdaq_agent.report.render import render_report
    from tests.unit.test_grounding import dated_headlines
    _, text, html = render_report(make_ctx(headlines=dated_headlines(), narrative=_noted(2, 3, 1)))
    assert "  [2] 2026-09-24 08:00 ET  Acme wins FDA clearance\n" in text
    assert "2026-09-24 08:00 ET" in html


def test_a_headline_without_a_time_is_listed_without_one():
    from nasdaq_agent.report.render import render_report
    from nasdaq_agent.sources.models import Headline
    _, text, _ = render_report(make_ctx(headlines=[Headline(id=1, title="Undated story", provider="Yahoo")],
                                        narrative=_noted(1)))
    assert "  [1] Undated story\n      Source: Yahoo\n      Note on headline 1." in text


def test_a_headline_the_notes_miss_is_still_listed_and_a_note_for_a_missing_headline_is_dropped():
    # compose_report makes both impossible for the agent's prose; rendering still never loses a headline or fails.
    from nasdaq_agent.report.render import render_report
    from tests.unit.test_grounding import dated_headlines
    _, text, _ = render_report(make_ctx(headlines=dated_headlines(), narrative=_noted(3, 9)))
    block = _headline_block(text)
    assert block.index("Acme presents") < block.index("Acme wins FDA clearance") < block.index("Acme prices share offering")
    assert "Note on headline 9." not in text


def test_headline_notes_are_escaped_in_html():
    from nasdaq_agent.report.render import render_report
    from nasdaq_agent.report.schemas import HeadlineNote
    from tests.unit.test_grounding import good_narrative, good_notes
    notes = [HeadlineNote(headline_ids=[1], summary="<b>bold</b> claim"), *good_notes()[1:]]
    _, _, html = render_report(make_ctx(narrative=good_narrative().model_copy(update={"headline_notes": notes})))
    assert "&lt;b&gt;bold&lt;/b&gt; claim" in html and "<b>bold</b>" not in html



def test_a_consolidated_story_is_listed_once_naming_the_outlets_its_title_does_not_link():
    # The title links the story's first article, so the line under it names only the other outlets. A story whose
    # title has no link still names every source.
    from nasdaq_agent.report.render import render_report
    from nasdaq_agent.report.schemas import HeadlineNote
    from tests.unit.test_grounding import good_narrative, multi_source_headlines
    notes = [HeadlineNote(headline_ids=[2, 4], summary="Clearance news."), HeadlineNote(headline_ids=[1, 3], summary="Offering news.")]
    narrative = good_narrative().model_copy(update={"headline_notes": notes})
    _, text, html = render_report(make_ctx(headlines=multi_source_headlines(), narrative=narrative))
    assert ("  [1, 3] 2026-09-23 09:00 ET  Acme prices $5 million registered direct offering\n"
            "      https://example.com/offering\n"
            "      Also reported by: Yahoo Finance via Yahoo Finance\n"
            "      Offering news.") in text
    assert "      Sources: Business Wire via Massive; Zacks via Yahoo Finance\n" in text
    assert "Acme Corp announces pricing" not in text  # the duplicate's title is not repeated
    assert text.index("[2, 4]") < text.index("[1, 3]")
    assert ('<a href="https://example.com/offering">Acme prices $5 million registered direct offering</a>'
            "<br><small>Also reported by: Yahoo Finance via Yahoo Finance</small>") in html
    assert html.count('href="https://example.com/offering"') == 1


def test_a_linked_title_has_no_source_line_repeating_its_link():
    # The plain-text part cannot link the title, so the link goes on the line under it, with no "Source:" label.
    from nasdaq_agent.report.render import render_report
    from nasdaq_agent.sources.models import Headline
    filing = Headline(id=1, title="Acme Corp filed a Form 8-K with the SEC", provider="Acme Corp", source="sec",
                      url="https://example.com/filing")
    _, text, html = render_report(make_ctx(headlines=[filing], narrative=_noted(1)))
    assert "  [1] Acme Corp filed a Form 8-K with the SEC\n      https://example.com/filing\n      Note on headline 1." in text
    assert html.count('href="https://example.com/filing"') == 1
    assert "Source:" not in text and "Source:" not in html


def test_a_title_that_cannot_be_linked_keeps_its_source_line():
    # Only http and https links are shown, so this title is plain text and the line under it says where it came from.
    from nasdaq_agent.report.render import render_report
    from nasdaq_agent.sources.models import Headline
    story = Headline(id=1, title="Acme wins FDA clearance", provider="Business Wire", source="massive",
                     url="javascript:alert(1)")
    _, text, html = render_report(make_ctx(headlines=[story], narrative=_noted(1)))
    assert "  [1] Acme wins FDA clearance\n      Source: Business Wire via Massive\n      Note on headline 1." in text
    assert "Acme wins FDA clearance<br><small>Source: Business Wire via Massive</small>" in html


def test_a_story_entry_shows_each_link_once():
    # Two sources can return the same article. A headline whose link the entry already shows is not listed again, and
    # an entry left with no other outlet has no line under its title; a headline without a link is still named.
    from nasdaq_agent.report.render import render_report
    from nasdaq_agent.report.schemas import HeadlineNote
    from nasdaq_agent.sources.models import Headline
    from tests.unit.test_grounding import good_narrative
    headlines = [
        Headline(id=1, title="Acme wins FDA clearance", provider="Business Wire", source="massive", url="https://example.com/fda"),
        Headline(id=2, title="Acme wins FDA clearance", provider="Business Wire", source="yfinance", url="https://example.com/fda"),
        Headline(id=3, title="Acme prices registered direct offering", provider="GlobeNewswire", source="massive",
                 url="https://example.com/offering"),
        Headline(id=4, title="Acme prices registered direct offering", provider="GlobeNewswire", source="yfinance",
                 url="https://example.com/offering"),
        Headline(id=5, title="Acme announces pricing of offering", provider="Zacks", source="yfinance", url="https://example.com/zacks"),
        Headline(id=6, title="Acme announces pricing of offering", provider="Zacks", source="massive", url="https://example.com/zacks"),
        Headline(id=7, title="Acme Corp filed a Form 8-K with the SEC", provider="Acme Corp", source="sec"),
    ]
    notes = [HeadlineNote(headline_ids=[1, 2], summary="Clearance news."),
             HeadlineNote(headline_ids=[3, 4, 5, 6, 7], summary="Offering news.")]
    narrative = good_narrative().model_copy(update={"headline_notes": notes})
    _, text, html = render_report(make_ctx(headlines=headlines, narrative=narrative))
    assert "  [1, 2] Acme wins FDA clearance\n      https://example.com/fda\n      Clearance news.\n" in text
    assert ("  [3, 4, 5, 6, 7] Acme prices registered direct offering\n"
            "      https://example.com/offering\n"
            "      Also reported by: Zacks via Yahoo Finance https://example.com/zacks; Acme Corp via SEC EDGAR\n"
            "      Offering news.") in text
    assert ('<small>Also reported by: <a href="https://example.com/zacks">Zacks via Yahoo Finance</a>; '
            "Acme Corp via SEC EDGAR</small>") in html
    assert html.count("Also reported by") == 1
    assert [html.count(f'href="{url}"') for url in ("https://example.com/fda", "https://example.com/offering",
                                                     "https://example.com/zacks")] == [1, 1, 1]


def test_template_prose_lists_every_headline_on_its_own_with_its_source():
    # Without the agent there is no consolidation: each headline is its own entry, most recent first, still tagged.
    from nasdaq_agent.report.render import render_report
    from tests.unit.test_grounding import multi_source_headlines
    _, text, _ = render_report(make_ctx(headlines=multi_source_headlines(), narrative=None, template_prose=True))
    import re
    assert re.findall(r"\n  \[([\d, ]+)\]", "\n  [" + _headline_block(text)) == ["4", "2", "3", "1"]
    assert "Source: Zacks via Yahoo Finance" in text



def test_a_long_story_summary_is_wrapped_and_indented_in_the_text_part():
    from nasdaq_agent.report.render import render_report
    from nasdaq_agent.report.schemas import HeadlineNote
    from tests.unit.test_grounding import good_narrative, headlines
    summary = ("Acme asked regulators to review its lead drug candidate, a step that usually precedes a decision by "
               "several months. The filing covers the company's main programme, and approval would open its first "
               "commercial market, so investors often treat such filings as a signal of progress.")
    notes = [HeadlineNote(headline_ids=[1], summary=summary), HeadlineNote(headline_ids=[2], summary="Guidance news."),
             HeadlineNote(headline_ids=[3], summary="Upgrade news.")]
    _, text, _ = render_report(make_ctx(headlines=headlines(), narrative=good_narrative().model_copy(update={"headline_notes": notes})))
    lines = [line for line in text.splitlines() if line.startswith("      ") and "Source" not in line][:3]
    assert len(lines) == 3 and all(len(line) <= 102 for line in lines)
    assert " ".join(line.strip() for line in lines).startswith("Acme asked regulators to review its lead drug")


def test_each_metric_sits_on_its_own_line_in_the_text_part():
    # The daily-changes line ended in a loop tag, and trim_blocks removed the line break after it, so a live email read
    # "... +309.64%  average daily change: 60.94%".
    from nasdaq_agent.report.render import render_report
    _, text, _ = render_report(make_ctx())
    block = text.split("Five-day metrics\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert [line.split(":")[0].strip() for line in block] == ["daily changes", "average daily change", "cumulative return",
                                                             "annualised volatility", "max drawdown", "vs SPY", "trend"]
    assert block[0].endswith("+2.00%, -2.45%, +6.53%, +0.94%, +3.74%")


def test_a_filing_is_tagged_with_its_filer_via_sec_edgar():
    from nasdaq_agent.report.render import source_tag
    from nasdaq_agent.sources.models import Headline
    filing = Headline(id=1, title="Wetour Robotics Ltd filed a Form 6-K with the SEC", provider="Wetour Robotics Ltd",
                      source="sec")
    assert source_tag(filing) == "Wetour Robotics Ltd via SEC EDGAR"
