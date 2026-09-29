"""The published documents at the repository root: a README with setup and execution instructions, and a REPORT of
about two and a half pages covering five named topics and nothing else. docs/ is git-ignored (local design material),
so nothing published may depend on it."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The owner chose clarity over a strict two-page limit, so the report may run a little past two A4 pages. Measured by
# rendering it to PDF at 11pt: 1,244 words, with the diagram and three tables, printed to about two and a half pages.
# The cap leaves a little room and stops the report from sprawling.
REPORT_MAX_WORDS = 1275
REPORT_SECTIONS = (
    "## 1. Agent framework and architecture",
    "## 2. How the agent plans and executes the task",
    "## 3. How the agent generates and executes the analysis code",
    "## 4. Challenges and solutions",
    "## 5. Error handling and reliability",
)
README_NEEDLES = ("make install", "nasdaq-agent run", "nasdaq-agent replay", "docker compose", "OPENROUTER_API_KEY",
                  "Yahoo", "robots.txt", "training", "macOS")
PUBLISHED_DOCS = ("README.md", "REPORT.md")
SECTION_HEADING = re.compile(r"^## .+$", re.MULTILINE)
LINK_TARGET = re.compile(r"\]\(([^)#\s]+)")
DOCS_REFERENCE = re.compile(r"(?:\]\(|`)docs/")


def test_readme_gives_setup_and_execution_instructions():
    readme = (ROOT / "README.md").read_text()
    for needle in README_NEEDLES:
        assert needle in readme, needle


def test_report_covers_only_the_five_required_topics_in_order_within_about_two_and_a_half_pages():
    report = (ROOT / "REPORT.md").read_text()
    assert tuple(SECTION_HEADING.findall(report)) == REPORT_SECTIONS
    assert len(report.split()) <= REPORT_MAX_WORDS


def test_published_docs_never_point_into_the_ignored_docs_folder():
    for name in PUBLISHED_DOCS:
        assert not DOCS_REFERENCE.search((ROOT / name).read_text()), name


def test_relative_links_and_images_in_published_docs_resolve():
    for name in PUBLISHED_DOCS:
        for target in LINK_TARGET.findall((ROOT / name).read_text()):
            if "://" in target or target.startswith("mailto:"):
                continue
            assert (ROOT / target).exists(), (name, target)


def test_docs_folder_is_git_ignored():
    ignored = [line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()]
    assert "docs/" in ignored
