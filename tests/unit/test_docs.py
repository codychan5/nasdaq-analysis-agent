"""The published documents at the repository root: a README with setup and execution instructions, and a REPORT of
up to about three pages covering five named topics. docs/ is git-ignored (local design material), so nothing
published may depend on it."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_MAX_WORDS = 1950  # about three pages, at roughly 650 words a page
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
LINK_TARGET = re.compile(r"\]\(([^)#\s]+)")
DOCS_REFERENCE = re.compile(r"(?:\]\(|`)docs/")


def test_readme_gives_setup_and_execution_instructions():
    readme = (ROOT / "README.md").read_text()
    for needle in README_NEEDLES:
        assert needle in readme, needle


def test_report_covers_the_five_required_topics_in_order_within_three_pages():
    report = (ROOT / "REPORT.md").read_text()
    positions = [report.find(heading) for heading in REPORT_SECTIONS]
    assert all(p >= 0 for p in positions), [h for h, p in zip(REPORT_SECTIONS, positions) if p < 0]
    assert positions == sorted(positions)
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
