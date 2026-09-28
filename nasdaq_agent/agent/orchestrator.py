from pathlib import Path

from langchain.agents import create_agent

from ..metrics import DEFINITIONS_TEXT, METRIC_NAMES
from .context import RunContext
from .middleware import build_middleware
from .tools import build_tools
from .tools.common import Deps

PROMPT_PATH = Path(__file__).parent / "prompts" / "orchestrator.md"
KICKOFF_MESSAGE = "Produce today's top NASDAQ gainer report. Begin."


def render_system_prompt() -> str:
    # B8: the prompt is built only from the static metric definitions and names; it never used the
    # `settings` argument it used to take, and the recipient must never appear in the prompt anyway.
    template = PROMPT_PATH.read_text()
    return template.replace("{definitions}", DEFINITIONS_TEXT.strip()).replace("{required_keys}", ", ".join(METRIC_NAMES))


def build_orchestrator(ctx: RunContext, deps: Deps, model, checkpointer=None):
    return create_agent(model=model, tools=build_tools(ctx, deps), system_prompt=render_system_prompt(),
                        middleware=build_middleware(ctx, deps.settings), checkpointer=checkpointer)
