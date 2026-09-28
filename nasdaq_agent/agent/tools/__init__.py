from langchain_core.tools import BaseTool

from ..context import RunContext
from .code import make_run_python
from .common import Deps
from .compose import make_compose_report
from .gainer import make_find_top_gainer
from .give_up import make_give_up
from .history import make_get_price_history
from .news import make_get_news
from .send import make_send_email
from .sentiment import make_record_sentiment
from .session import make_resolve_session
from .verify import make_verify_analysis


def build_tools(ctx: RunContext, deps: Deps) -> list[BaseTool]:
    return [make_resolve_session(ctx, deps), make_find_top_gainer(ctx, deps), make_get_price_history(ctx, deps),
            make_run_python(ctx, deps), make_verify_analysis(ctx, deps), make_get_news(ctx, deps),
            make_record_sentiment(ctx, deps), make_compose_report(ctx, deps), make_send_email(ctx, deps),
            make_give_up(ctx, deps)]
