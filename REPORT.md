# NASDAQ Top-Gainer Agent: Report

## 1. Agent framework and architecture

![The run graph: init, an orchestrating agent that calls ten state-gated tools, and finalize](assets/nasdaq-agent-graph.png)

The agent is a LangGraph 1.2 graph with three nodes. `init` is an empty first step: the run's id, output folder and checkpoint store must exist before the graph starts, so the command creates them first. `agent` is a LangChain 1.4 `create_agent` orchestrator with ten tools. `finalize` sends a failure notice if no report went out and sets the exit code.

Gating middleware shows the model only the tools whose preconditions hold, and each tool re-checks its own precondition, so the model can choose the order of the steps but can never run one too early. I chose an agent over a fixed workflow because the task needs judgement at several points: which step comes next, the analysis code and its repairs, the news sentiment and the written summary. The tools and the saved state keep the guarantees.

## 2. How the agent plans and executes the task

**Tool selection.** At each step the model reads the last result and picks the next tool from those currently allowed. The system prompt suggests a typical order but does not enforce it. Fallback between data sources is code, not a model decision: each data tool tries its sources in a fixed order, each at most once, and when every source has failed only `give_up` remains.

**Information flow.** Everything the run knows lives in one typed state object, saved to `context.json` after every tool call. Each tool reads from it and writes back to it:

| Tool | What it does | Reads → writes |
|---|---|---|
| `resolve_session` | picks the last finished day | clock and calendar → the trading day to analyse |
| `find_top_gainer` | finds and checks the gainer | the trading day → top stock and sources tried |
| `get_price_history` | fetches six days of prices | top stock → price files for it and a benchmark |
| `run_python` | runs the model's code | price files → each code run and its output |
| `verify_analysis` | recomputes every metric | code output and prices → verified metrics |
| `get_news` | fetches recent headlines | top stock → headlines, tagged by source |
| `record_sentiment` | records the news sentiment | headlines → a sentiment score, label and reason |
| `compose_report` | fact-checks the summary | verified metrics, headlines → email text and chart |
| `send_email` | sends the report once | report, address in settings → a record of the send |
| `give_up` | ends the run with a reason | nothing → the reason for stopping |

Gainers come from Massive and Alpha Vantage first, then the Yahoo and Nasdaq.com screeners; price history from yfinance, then Massive; news from Massive, Yahoo and, when enabled, SEC filings. "The day" is the last finished trading session, and the top gainer is the literal one among common stocks; a prior close under $5 adds a thin-trading warning. The recipient's address comes only from settings, never from the model or from fetched text.

## 3. How the agent generates and executes the analysis code

`get_price_history` saves the stock's and the benchmark's daily prices as files in the sandbox. The model then writes pandas code that reads them and leaves a `result` dict of seven required metrics. Before it runs, a syntax-tree check rejects imports other than pandas, numpy, math, statistics, json and datetime, calls such as `eval` and `open`, and file readers and writers. The code then runs in one of two sandboxes:

- **Docker, the default:** no network, a read-only file system, 512 MB of memory, one CPU and a non-root user.
- **Subprocess, only by explicit choice:** CPU-time and file limits, but no network isolation and, on macOS, no memory limit. Without Docker the run stops at setup; it never falls back to this weaker sandbox silently.

`run_python` checks that the `result` dict has exactly the required keys, finite numbers and a valid trend. Any error goes back to the model with the error text and the rules, so it can fix its code, within six runs. `verify_analysis` then recomputes every metric with reviewed code, and only those numbers are published.

## 4. Challenges and solutions

| Challenge | Solution |
|---|---|
| **Gating without a fixed pipeline.** A free agent could call `send_email` before the analysis is verified, or skip a step. | Middleware hides each tool until its precondition holds, and each tool re-checks it, so the model still chooses the order but cannot choose an unsafe one. |
| **LLM output is not trusted.** Financial figures must be exact, and a model can be confidently wrong, even when checking its own work. | Reviewed code recomputes every metric within 0.01; every number in the summary must trace to a verified figure or a cited headline; a judge model from another lab, as configured, reviews the summary's meaning. |
| **One source is not trusted.** A provider can skip a day, disagree with another, or report a gain its own prices do not support. | A second source must match every close within 0.5%, or both closes are shown and the run is marked degraded; all six closes differing by one factor is a later split's restatement, so the traded prices are shown instead; each gain is recomputed from the stock's own prices. |
| **Fetched news is not trusted.** A headline could carry injected instructions that steer the code the model writes or try to redirect the email. | Headlines reach the model as quoted data, not instructions; code runs in Docker with no network, a read-only disk and no admin rights, never silently elsewhere; the recipient comes only from settings. |
| **Parallel tool calls.** LangGraph runs the tool calls of one model turn at the same time, on a thread pool, so two could change the shared run state at once. | A per-run lock runs tool calls one at a time, so no two change the shared state at once, and the limits of six code runs and 25 tool calls cannot be exceeded. |

## 5. Error handling and reliability

I apply four rules at every step, so the agent behaves predictably whatever fails:

- **Classify, then respond.** A transient error, such as a timeout, is retried. Bad data, such as a missing day, moves on to another stock or source. A risk to correctness, such as code that keeps failing verification, stops the run, because a wrong report is worse than none. A failed optional part, such as the news or the chart, only degrades the report.
- **Bound everything.** A data call gets three quick attempts, or up to 90 s when rate-limited; the AI model gets a 90-second timeout and two retries, then a fallback model. Each run allows 25 calls to the AI model, 25 tool calls, six code runs and ten minutes, so no stuck source or looping model can hang it.
- **Never fail silently**, so a clean email can be trusted. A clean report exits 0; a degraded report names each gap in the email and exits 2; a run with no report sends a redacted failure notice and exits 1.
- **Make recovery safe.** Checkpoints follow every tool call, so `resume` continues a killed run. The email goes out at most once: a `sending` marker blocks any resend, a failed send is retried only if the mail server certainly did not receive it, and a report that cannot be delivered is kept in the outbox.
