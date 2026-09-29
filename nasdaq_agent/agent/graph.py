import hashlib
import logging
import sqlite3
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, TypedDict

from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from pydantic import SecretStr

from ..artifacts import RunDir, configure_logging, new_run_id, redact, validate_run_id
from ..config import Mode, Settings
from ..email.transports import FileTransport, Transport, select_transport
from ..replay.environment import current_environment, environment_differences, sandbox_backend_differences
from ..replay.http_cassette import HttpCassette
from ..replay.llm_cache import JsonFileLLMCache, check_manifest, install_llm_cache
from ..replay.proxies import wrap_sources
from ..replay.recording import (RecordingNotSealable, acquire_recording_lock, discard_recording, release_recording_lock,
                                seal_recording, staging_dir, start_recording)
from ..report.judge import make_judge
from ..sandbox.runner import select_runner
from ..sources.errors import REPLAY_ENVIRONMENT_DIFFERENCES, CassetteMiss
from ..sources.registry import (build_gainer_sources, build_history_sources, build_listing_source, build_news_sources,
                                make_http_client, massive_rate_limiter)
from ..universe import UniverseByDate, load_universe
from .context import CONTEXT_FILENAME, RunContext
from .finalize import (EXIT_CLEAN, EXIT_DEGRADED, EXIT_FAILED, MAX_ERROR_CHARS, finalize, finalize_with,
                       reconcile_delivery, write_summary)
from .llm import PROVIDER_KEY_FIELDS, build_chat_model
from .orchestrator import KICKOFF_MESSAGE, build_orchestrator
from .tools.common import Deps

log = logging.getLogger("nasdaq_agent.graph")
CHECKPOINT_DB = "checkpoints.sqlite"
UNIVERSE_CACHE = "nasdaqlisted.txt"
# Massive's NASDAQ list for each past session a live run needed, one file per day, beside the symbol file.
DAY_LISTS_DIR = "nasdaq_lists"
# Passed to both graph.invoke and orchestrator.invoke so each checkpoint is
# durable (flushed to the sqlite file) before the next step starts -- required for resume-from-
# checkpoint to see a consistent, complete prior step rather than a partially-written one.
DURABILITY = "sync"
# Replay runs with no keys. The chat models are still built in replay -- bind_tools, with_structured_output and the
# cache key all come from the provider class, and ChatGoogleGenerativeAI cannot be built without a credential -- so
# they get this placeholder. It is never sent anywhere: every response comes from the cassette, and a miss raises
# CassetteMiss before any provider call. The llm_string names a key by reference only, so it matches the recording.
REPLAY_PLACEHOLDER_KEY = "replay-placeholder-not-a-real-key"
# The key-gated data sources in sources/registry.py and the Settings field that enables each. SEC needs no key, but its
# source runs only with a declared User-Agent, which replay likewise swaps for the placeholder.
KEYED_SOURCE_KEY_FIELDS = {"massive": "massive_api_key", "alphavantage": "alphavantage_api_key",
                           "sec": "sec_user_agent"}


class GraphState(TypedDict, total=False):
    run_id: str
    exit_code: int
    summary: str


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    exit_code: int
    artifacts_path: str


def _settings_hash(settings: Settings) -> str:
    # The exclude set comes from Settings.secret_field_names(), not a second hand-written list of the
    # secret fields, so the two cannot drift apart.
    return hashlib.sha256(settings.model_dump_json(exclude=set(Settings.secret_field_names())).encode()).hexdigest()[:16]


def _secrets(settings: Settings) -> list[str]:
    return settings.secret_values()


def _transport_for_mode(settings: Settings, outbox_dir: Path) -> Transport:
    """Replay never sends real mail. It always writes to the file outbox, whatever SMTP
    settings the environment holds. Record is a live run and uses the configured transport."""
    if settings.mode == Mode.replay:
        return FileTransport(outbox_dir)
    return select_transport(settings, outbox_dir)


def build_deps(settings: Settings, ctx: RunContext, run_dir: RunDir, clock: Callable[[], datetime],
               cassette=None, judge=None) -> Deps:
    cassette_backed = settings.mode != Mode.live
    if cassette is None and cassette_backed:
        cassette = HttpCassette(settings.cassette_dir, settings.mode.value)
    http = make_http_client(settings, cassette)
    # In record and replay the NASDAQ symbol file always comes through the cassette-backed client, never the local
    # universe cache -- a clean-machine replay must not depend on one, or leave one behind.
    official = load_universe(http.get_text, None if cassette_backed else Path(settings.artifacts_dir) / UNIVERSE_CACHE)
    # Rely on select_runner's own default availability check (docker_backend_available, which requires both a
    # reachable daemon AND a built sandbox image) instead of importing docker_available into this module and passing
    # it explicitly -- a daemon-only check would let "auto" select Docker and then fail every run with
    # "Unable to find image". Without Docker it raises, unless the subprocess sandbox was chosen explicitly, so the
    # run ends at setup with a failure notice before any model call.
    runner = select_runner(settings.sandbox_backend)
    if judge is None:
        judge = make_judge(build_chat_model(settings, "judge"))
    # Massive counts calls per API key, so the gainer, history, news and listing sources share one allowance.
    massive_limiter = massive_rate_limiter()
    # The symbol file lists only today's securities, so a session before its day is filtered with Massive's list for
    # that session. Record and replay save no lists, as they keep no symbol file: the cassette holds the responses.
    listing_source = build_listing_source(settings, cassette, massive_limiter=massive_limiter)
    universe = UniverseByDate(official, listing_source.nasdaq_listing if listing_source is not None else None,
                              None if cassette_backed else Path(settings.artifacts_dir) / DAY_LISTS_DIR)
    gainer_sources = build_gainer_sources(settings, universe, Path(settings.artifacts_dir), cassette,
                                          massive_limiter=massive_limiter)
    # History sources only map symbol spellings, which no list changes.
    history_sources = build_history_sources(settings, official, cassette, massive_limiter=massive_limiter)
    news_sources = build_news_sources(settings, cassette, massive_limiter=massive_limiter)
    if cassette_backed:
        # yfinance bypasses our HTTP client, so those adapters are recorded and replayed at the source interface.
        gainer_sources, history_sources, news_sources = (
            wrap_sources(sources, settings.cassette_dir, settings.mode.value)
            for sources in (gainer_sources, history_sources, news_sources))
    return Deps(settings=settings, universe=universe, gainer_sources=gainer_sources, history_sources=history_sources,
                news_sources=news_sources, runner=runner, transport=_transport_for_mode(settings, run_dir.outbox_dir),
                judge=judge, clock=clock, run_dir=run_dir)


def _has_pending_tasks(graph, config: dict) -> bool:
    """A crash can leave a checkpointed graph mid-step -- the outer graph, or the
    orchestrator's OWN thread (a ReAct loop that is itself checkpointed; durability="sync" makes each
    of its steps durable before the next starts), e.g. an AI tool call recorded but its tool result
    never produced. Resuming must continue that graph with invoke(None, ...) rather than start it
    again with fresh input, which for the orchestrator would append a second kickoff message to a
    conversation that already has a pending tool call in it.

    Pending work is read from StateSnapshot.tasks, not .next. A finished task's
    writes are saved as soon as it finishes, before the step's own checkpoint, and .next leaves out
    every task whose writes were saved. After a kill in that window .next can be empty while the
    step is still unfinished; .tasks still lists it.

    get_state is a real langgraph API; a fake orchestrator in tests may not implement it at all,
    which must mean "nothing to resume", not an AttributeError -- it must never weaken this check
    for the real graphs, which always have get_state.
    """
    get_state = getattr(graph, "get_state", None)
    if get_state is None:
        return False
    return bool(get_state(config).tasks)


def build_graph(orchestrator, ctx: RunContext | None, deps: Deps | None, checkpointer=None, outcome: dict | None = None):
    def init_node(state: GraphState) -> GraphState:
        return {}

    def agent_node(state: GraphState) -> GraphState:
        config = {"configurable": {"thread_id": state["run_id"]}}
        try:
            if _has_pending_tasks(orchestrator, config):
                result = orchestrator.invoke(None, config=config, durability=DURABILITY)
            else:
                result = orchestrator.invoke({"messages": [HumanMessage(KICKOFF_MESSAGE)]}, config=config, durability=DURABILITY)
        except Exception as e:
            # Catch anything the orchestrator raises (model/provider errors, a tool bug that escaped its own
            # handling, ...) so this node always returns normally. An exception here would abort graph.invoke()
            # before "finalize" ever ran, and finalize is what sends the failure notice and writes summary.json --
            # it must always run. Note this is `except Exception`, not `except BaseException`: a BaseException
            # subclass (simulating a killed process, e.g. in a resume test) is deliberately NOT caught here either,
            # exactly as a real kill would not be.
            secrets = deps.settings.secret_values() if deps is not None else []
            # Redact the full message before truncating, so a secret that
            # straddles the truncation boundary can't leave a partial, unredacted fragment behind.
            message = redact(f"agent loop failed: {type(e).__name__}: {e}", secrets)[:MAX_ERROR_CHARS]
            log.exception(message)
            if ctx is not None:
                ctx.errors.append(message)
                ctx.save()
            return {"summary": ""}
        messages = result.get("messages", [])
        summary = messages[-1].content if messages else ""
        if ctx is not None:
            deps.run_dir.write_text("closing_summary.txt", str(summary))
        return {"summary": str(summary)}

    def finalize_node(state: GraphState) -> GraphState:
        code = finalize(ctx, deps) if ctx is not None else 0
        # Record this invocation's own outcome for _handle_graph_invoke_failure
        # to recover if graph.invoke itself still raises afterwards (e.g. persisting the checkpoint
        # for this very step) -- never a stale file from an earlier invocation of this same run.
        if outcome is not None:
            outcome["exit_code"] = code
        return {"exit_code": code}

    g = StateGraph(GraphState)
    g.add_node("init", init_node)
    g.add_node("agent", agent_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "init")
    g.add_edge("init", "agent")
    g.add_edge("agent", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def _prepare(settings: Settings, run_id: str) -> tuple[RunContext, RunDir]:
    """Only the steps that must succeed for anything -- including the
    failure path -- to work at all: creating the run directory, configuring logging, loading or
    creating the context. An exception here propagates; there is no ctx/run_dir yet to route a
    failure notice through, and the command line prints one clean line for it.
    """
    run_dir = RunDir(Path(settings.artifacts_dir), run_id)
    configure_logging(run_dir, _secrets(settings))
    ctx_path = run_dir.path / "context.json"
    ctx = (RunContext.load(run_dir.path) if ctx_path.exists()
           else RunContext.new(run_id, _settings_hash(settings), str(run_dir.path), mode=settings.mode))
    ctx.save()
    return ctx, run_dir


def _replay_settings(settings: Settings, manifest: dict[str, Any]) -> Settings:
    """Replay runs the bundled cassette with no keys, and never holds a real one:
    - every model provider key becomes the placeholder (see REPLAY_PLACEHOLDER_KEY);
    - the data sources are exactly the recording's chain: a key-gated source the manifest lists is rebuilt around the
      placeholder (request hashes leave credentials out, sources/http.py), and one it does not list is left out,
      whatever keys this machine holds;
    - pacing is off: no provider is called, so there is no rate limit to respect.
    """
    recorded = set(manifest.get("keyed_sources", []))
    unknown = recorded - set(KEYED_SOURCE_KEY_FIELDS)
    if unknown:
        raise CassetteMiss(f"the cassette manifest names sources this version does not know: {sorted(unknown)}; "
                           "run `nasdaq-agent record` again")
    placeholder = SecretStr(REPLAY_PLACEHOLDER_KEY)
    update: dict[str, Any] = {"llm_call_delay_seconds": 0}
    update.update({field: placeholder for field in PROVIDER_KEY_FIELDS.values()})
    update.update({field: placeholder if name in recorded else None for name, field in KEYED_SOURCE_KEY_FIELDS.items()})
    return settings.model_copy(update=update)


def _keyed_sources(settings: Settings) -> list[str]:
    return [name for name, field in KEYED_SOURCE_KEY_FIELDS.items() if getattr(settings, field) is not None]


def _note_environment_differences(ctx: RunContext, differences: list[str], scope: ExitStack) -> None:
    """A replay in an environment that differs from the recording's logs a warning, adds a
    run note naming each difference, and makes every CassetteMiss raised for the rest of the run lead with them."""
    if not differences:
        return
    text = f"replay environment differs from the recording ({'; '.join(differences)})"
    log.warning(text)
    ctx.notes.append(text)
    token = REPLAY_ENVIRONMENT_DIFFERENCES.set(REPLAY_ENVIRONMENT_DIFFERENCES.get() + tuple(differences))
    scope.callback(REPLAY_ENVIRONMENT_DIFFERENCES.reset, token)


def _prepare_mode(settings: Settings, now: datetime | None, ctx: RunContext,
                  scope: ExitStack) -> tuple[Settings, datetime | None, dict[str, Any] | None]:
    """Record and replay setup steps, entered on the run's scope so they are undone when the run ends, however it ends.

    Replay: a missing or stale cassette raises CassetteMiss here, which becomes a failure notice and exit 1 like any
    other setup failure. The recording's clock replaces any other, so every date the tools derive -- and every source
    request key -- matches the recording. LangSmith tracing is off (replay makes no network
    calls). Environment differences from the recording are noted. Returns the manifest, so _setup can
    compare the sandbox backend once the deps have chosen one.

    Record: one clock, captured at the start; the manifest stores that instant for replay. Under an exclusive lock
    beside the cassette, the recording goes into a fresh staging directory beside it, which the stores write to and
    which is removed at the end of the run unless the seal swapped it in -- so a failed recording leaves the previous
    cassette exactly as it was. The cassette directory must pass recording.check_cassette_directory's content guard.

    Both install the cassette's LLM cache; the run's scope puts the previous global cache back.
    """
    if settings.mode == Mode.replay:
        manifest = check_manifest(settings.cassette_dir)
        now = datetime.fromisoformat(manifest["now"])
        scope.enter_context(tracing_context(enabled=False))
        _note_environment_differences(ctx, environment_differences(manifest.get("environment"), current_environment()), scope)
        settings = _replay_settings(settings, manifest)
        install_llm_cache(JsonFileLLMCache(settings.cassette_dir, Mode.replay.value))
        return settings, now, manifest
    if settings.mode == Mode.record:
        now = now if now is not None else datetime.now(timezone.utc)
        # One recording at a time per cassette, the lock held for the whole recording and
        # released on the run's scope however it ends -- after the staging directory is cleaned up.
        scope.callback(release_recording_lock, acquire_recording_lock(settings.cassette_dir))
        staging = start_recording(settings.cassette_dir, ctx.run_id)
        scope.callback(discard_recording, staging)
        settings = settings.model_copy(update={"cassette_dir": staging})
        install_llm_cache(JsonFileLLMCache(staging, Mode.record.value))
    return settings, now, None


@contextmanager
def _run_scope() -> Iterator[ExitStack]:
    """Process-wide state a record or replay run changes for its duration -- the global LLM cache
    (set_llm_cache is process-global), LangSmith tracing, the note CassetteMiss messages carry, the staging
    directory -- is entered on this stack by _prepare_mode, and all of it is undone when the run ends, however it ends:
    the previous global cache comes back last."""
    previous_cache = get_llm_cache()
    with ExitStack() as scope:
        scope.callback(set_llm_cache, previous_cache)
        yield scope


def _setup(settings: Settings, ctx: RunContext, run_dir: RunDir, now: datetime | None, model, deps_factory,
          cassette=None, outcome: dict | None = None, *, scope: ExitStack):
    """Every step after the context exists -- building deps/model, opening
    the checkpoint connection, constructing the orchestrator/graph. Raises on failure instead of
    silently leaking the sqlite connection; run_once/resume_run catch it and route to
    finalize_with without ever entering the agent loop.

    The sqlite connection is opened last, after deps and the model exist (so a UniverseError, a
    missing model key, or select_runner raising never opens one at all), and is closed here if
    anything after it raises. On success the connection is returned unclosed -- the caller closes
    it once done with the graph.

    Record and replay are setup steps too (_prepare_mode), run first, on the run's scope. In replay the
    settings the deps and models see carry placeholder keys and zero pacing; in record, the staging directory.
    """
    settings, now, replay_manifest = _prepare_mode(settings, now, ctx, scope)
    clock = (lambda: now) if now is not None else (lambda: datetime.now(timezone.utc))
    deps = deps_factory(settings, ctx, run_dir, clock, cassette=cassette) if cassette is not None else deps_factory(settings, ctx, run_dir, clock)
    if replay_manifest is not None:
        # The sandbox backend is known only now, once the deps have chosen it.
        backend = deps.runner.name if deps.runner is not None else None
        _note_environment_differences(ctx, sandbox_backend_differences(replay_manifest.get("environment"), backend), scope)
    model = model or build_chat_model(settings, "orchestrator")
    # The sqlite3 connection is opened by hand and handed to SqliteSaver(conn)
    # -- not SqliteSaver.from_conn_string, which is a @contextmanager in this package, not a plain
    # constructor -- so run_once/resume_run can close it themselves once done with the graph.
    # test_checkpoints_persist_for_both_the_outer_graph_and_the_orchestrator_threads
    # (test_run_lifecycle.py) opens a fresh connection after a real run_once and finds checkpoints
    # for both the outer graph's thread and the orchestrator's thread.
    conn = sqlite3.connect(str(run_dir.path / CHECKPOINT_DB), check_same_thread=False)
    try:
        checkpointer = SqliteSaver(conn)
        orchestrator = build_orchestrator(ctx, deps, model, checkpointer=checkpointer)
        graph = build_graph(orchestrator, ctx, deps, checkpointer, outcome=outcome)
    except Exception:
        conn.close()
        raise
    return deps, graph, conn


def _transport_for_finalize(settings: Settings, run_dir: RunDir) -> Transport:
    """A transport built independently of Deps, for the two finalize_with call sites that run
    before (or instead of) full setup: an already-resolved resume, and a setup failure. Falls
    back to a bare FileTransport if select_transport itself raises.
    """
    try:
        return _transport_for_mode(settings, run_dir.outbox_dir)
    except Exception:
        return FileTransport(run_dir.outbox_dir)


def _record_failure_and_finalize(label: str, e: Exception, ctx: RunContext, settings: Settings, run_dir: RunDir) -> int:
    """Records "<label>: <type>: <message>" -- redacted before it is cut to MAX_ERROR_CHARS -- in ctx.errors and the
    log, then finalizes without Deps. Called from inside the caller's except block, so exc_info carries the traceback."""
    secrets = settings.secret_values()
    message = redact(f"{label}: {type(e).__name__}: {e}", secrets)[:MAX_ERROR_CHARS]
    log.error(message, exc_info=True)
    ctx.errors.append(message)
    return finalize_with(ctx, settings, run_dir, _transport_for_finalize(settings, run_dir))


def _handle_setup_failure(e: Exception, ctx: RunContext, settings: Settings, run_dir: RunDir) -> int:
    return _record_failure_and_finalize("setup failed", e, ctx, settings, run_dir)


def _handle_graph_invoke_failure(e: Exception, ctx: RunContext, settings: Settings, run_dir: RunDir, outcome: dict) -> int:
    """graph.invoke can still raise after finalize_node's code has already run. With
    durability="sync", graph.invoke waits for each step's checkpoint write before it goes on, so a
    failed write of the finalize step's own checkpoint (of any cause -- a full disk, a locked
    database file, or anything else) raises out of graph.invoke once finalize has already computed
    and recorded its exit code; a failed background write of a task's pending writes is re-raised
    when graph.invoke exits.

    This must never read summary.json to decide what happened, because on a
    resume, summary.json from an EARLIER invocation of this same run already exists -- reading it
    would return that earlier, stale exit code even when finalize_node ran again in THIS
    invocation and recorded a different, current one. `outcome` is a fresh, empty dict created by
    the caller for this invocation only, filled by finalize_node the moment it computes an exit
    code: if it already has one, that is this invocation's true, authoritative outcome, so return
    it directly. Otherwise finalize never recorded an outcome in this invocation, and the failure is
    recorded as "run failed: <type>: <message>" -- setup had already succeeded,
    so it is not a setup failure -- and finalized without Deps. finalize_with reconciles delivery,
    so a report that was, in fact, delivered during this same invocation still yields 0 or 2 and no
    failure notice.
    """
    if "exit_code" in outcome:
        log.error("graph.invoke raised after finalize already recorded this invocation's exit "
                 "code (%s): %s: %s", outcome["exit_code"], type(e).__name__, e, exc_info=True)
        return outcome["exit_code"]
    return _record_failure_and_finalize("run failed", e, ctx, settings, run_dir)


def _seal_recording(settings: Settings, ctx: RunContext, run_dir: RunDir, deps: Deps, exit_code: int) -> int:
    """The staged recording becomes the cassette -- manifest written, entries swapped in -- only after the run finished
    with exit 0 or 2, with no judge failure (the verdict would be missing), and with no configured secret anywhere in
    it (cassettes are committed). The manifest records the environment, which a replay compares with its own, and the
    exit code, which `replay --check` compares with the replay's. A recording that cannot be sealed fails the run
    loudly: an ERROR log line, the reason in ctx.errors and summary.json, and exit 1, though the report itself was
    delivered. Either way the run's scope removes the staging directory; the previous cassette is untouched. Never
    raises."""
    if exit_code not in (EXIT_CLEAN, EXIT_DEGRADED):
        log.error("recording not sealed: the run ended with exit %d; the cassette at %s is unchanged",
                  exit_code, settings.cassette_dir)
        return exit_code
    secrets = settings.secret_values()
    try:
        if ctx.judge_failures:
            raise RecordingNotSealable(f"the judge failed {ctx.judge_failures} time(s) while recording, so its verdict "
                                       "is not in the recording and a replay would fail at it; record again")
        environment = current_environment(deps.runner.name if deps.runner is not None else None)
        seal_recording(staging_dir(settings.cassette_dir, ctx.run_id), settings.cassette_dir, deps.clock().isoformat(),
                       secrets, _keyed_sources(settings), environment, exit_code)
    except Exception as e:
        message = redact(f"recording not sealed: {type(e).__name__}: {e}", secrets)
        log.error(message)
        ctx.errors.append(message[:MAX_ERROR_CHARS])
        write_summary(ctx, run_dir, EXIT_FAILED, secrets)
        return EXIT_FAILED
    log.info("recording sealed: the cassette at %s now holds this run's recording", settings.cassette_dir)
    return exit_code


def run_once(settings: Settings, now: datetime | None = None, model=None, deps_factory=build_deps, cassette=None) -> RunOutcome:
    run_id = new_run_id()
    ctx, run_dir = _prepare(settings, run_id)
    with _run_scope() as scope:
        outcome: dict = {}
        try:
            deps, graph, conn = _setup(settings, ctx, run_dir, now, model, deps_factory, cassette, outcome, scope=scope)
        except Exception as e:
            return RunOutcome(run_id=run_id, exit_code=_handle_setup_failure(e, ctx, settings, run_dir), artifacts_path=ctx.artifacts_path)
        try:
            result = graph.invoke({"run_id": run_id}, config={"configurable": {"thread_id": f"graph-{run_id}"}}, durability=DURABILITY)
            exit_code = result["exit_code"]
        except Exception as e:
            exit_code = _handle_graph_invoke_failure(e, ctx, settings, run_dir, outcome)
        finally:
            conn.close()
        if settings.mode == Mode.record:
            # deps.clock is the record clock _prepare_mode captured at the start: the manifest stores that instant.
            exit_code = _seal_recording(settings, ctx, run_dir, deps, exit_code)
        return RunOutcome(run_id=run_id, exit_code=exit_code, artifacts_path=ctx.artifacts_path)


def _require_resumable(settings: Settings, run_id: str) -> None:
    """The run id comes from the command line and becomes a path, so it is validated, and the run must already exist,
    before _prepare -- RunDir(...) creates the directory. A run is also tied to the mode it started in: otherwise a
    failed replay, resumed with the default mode, would run live and send real mail built from cassette data. Raises;
    nothing is created or written."""
    validate_run_id(run_id)
    context_path = Path(settings.artifacts_dir) / run_id / CONTEXT_FILENAME
    if not context_path.is_file():
        raise FileNotFoundError(f"no run {run_id} to resume: {context_path} does not exist")
    started_in, resuming_in = RunContext.load(context_path.parent).mode, settings.mode
    if started_in == Mode.replay:
        raise ValueError(f"run {run_id} is a replay run (current mode: {resuming_in.value}); replay runs cannot be "
                         "resumed -- run `nasdaq-agent replay` again")
    if Mode.record in (started_in, resuming_in):
        # A resumed recording would stage a new cassette and record only the remainder of the run, so a replay could
        # never reproduce it.
        raise ValueError(f"run {run_id} started in {started_in.value} mode and the current mode is {resuming_in.value}; "
                         "a recording cannot be resumed -- run `nasdaq-agent record` again")
    if started_in != resuming_in:
        raise ValueError(f"run {run_id} started in {started_in.value} mode and cannot be resumed in "
                         f"{resuming_in.value} mode")


def resume_run(settings: Settings, run_id: str, model=None, deps_factory=build_deps, cassette=None,
               now: datetime | None = None) -> RunOutcome:
    """`now` pins the resumed run's clock, as the command line does for a run of AGENT_SESSION_DATE; None is the real
    clock."""
    _require_resumable(settings, run_id)
    ctx, run_dir = _prepare(settings, run_id)
    current_hash = _settings_hash(settings)
    if ctx.settings_hash != current_hash:
        # A warning, not a refusal -- an operator may fix a setting (a timeout, say) and resume on purpose.
        log.warning("settings changed since run %s started (settings hash %s, now %s); resuming with the current "
                    "settings", run_id, ctx.settings_hash, current_hash)
    # Reconcile before the early-return check below, so a resumed run whose
    # report marker says "sent" (context.json merely stale from a crash between
    # marker.complete() and ctx.progress.sent being persisted) never re-enters the agent loop.
    if reconcile_delivery(ctx, run_dir):
        ctx.save()
    if ctx.progress.sent or ctx.progress.gave_up:
        transport = _transport_for_finalize(settings, run_dir)
        return RunOutcome(run_id=run_id, exit_code=finalize_with(ctx, settings, run_dir, transport), artifacts_path=ctx.artifacts_path)
    with _run_scope() as scope:
        outcome: dict = {}
        try:
            deps, graph, conn = _setup(settings, ctx, run_dir, now, model, deps_factory, cassette, outcome, scope=scope)
        except Exception as e:
            return RunOutcome(run_id=run_id, exit_code=_handle_setup_failure(e, ctx, settings, run_dir), artifacts_path=ctx.artifacts_path)
        config = {"configurable": {"thread_id": f"graph-{run_id}"}}
        try:
            # If the outer graph itself stopped mid-way (a task still pending --
            # e.g. the crash hit while "agent" was still running, so that step was never checkpointed
            # as complete, or while "finalize" was), continue it with invoke(None, ...) instead of
            # restarting from "init" with fresh input, which would re-run "init" and "agent" from
            # scratch -- and agent_node would append a second kickoff message to an orchestrator
            # thread that may already have its own pending tool call recorded in it. Pending work
            # is read from .tasks, not .next (see _has_pending_tasks).
            if _has_pending_tasks(graph, config):
                result = graph.invoke(None, config, durability=DURABILITY)
            else:
                result = graph.invoke({"run_id": run_id}, config=config, durability=DURABILITY)
            return RunOutcome(run_id=run_id, exit_code=result["exit_code"], artifacts_path=ctx.artifacts_path)
        except Exception as e:
            return RunOutcome(run_id=run_id, exit_code=_handle_graph_invoke_failure(e, ctx, settings, run_dir, outcome), artifacts_path=ctx.artifacts_path)
        finally:
            conn.close()
