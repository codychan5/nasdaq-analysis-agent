.PHONY: install test test-unit test-graph replay record run schedule compose-up
# make replay needs no .env: replay never sends mail (it writes to the run's file outbox), so a placeholder recipient
# will do. Only the replay command receives it; run and record read the real recipient from .env.
AGENT_EMAIL_TO ?= replay@example.com

install:
	pip install -c constraints.txt -e ".[dev]"
test-unit:
	pytest tests/unit -q
test-graph:
	pytest tests/graph -q
test: test-unit test-graph
replay:
	AGENT_EMAIL_TO=$(AGENT_EMAIL_TO) nasdaq-agent replay
# Precondition: an environment installed with `make install`, i.e. pinned by constraints.txt, the versions CI and
# every replay use -- cache keys contain provider kwargs and tool schemas as those libraries generate them. The
# manifest records the environment, and replay warns about any difference.
record:
	@echo "record: expects the environment pinned by constraints.txt (make install); the manifest records it"
	AGENT_MODE=record nasdaq-agent record
run:
	nasdaq-agent run
schedule:
	nasdaq-agent schedule
compose-up:
	docker compose -f docker/compose.yaml up --build
