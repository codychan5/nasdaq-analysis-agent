from langchain_core.tools import tool

from ..context import RunContext
from .common import Deps, ok, record_call


def make_give_up(ctx: RunContext, deps: Deps):
    @tool("give_up")
    def give_up(reason: str) -> str:
        """Stop the run because a required step cannot succeed. State the reason; a failure notice will be sent."""
        ctx.give_up_reason = reason
        ctx.progress.gave_up = True
        outcome = ok({"stopped": True, "reason": reason})
        record_call(ctx, deps, "give_up", {"reason": reason}, outcome)
        return outcome
    return give_up
