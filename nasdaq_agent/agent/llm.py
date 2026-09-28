from typing import Literal

import httpx

from langchain.chat_models import init_chat_model

from ..config import Settings

Purpose = Literal["orchestrator", "judge"]

# Current Anthropic models reject sampling parameters (temperature, top_p, ...) with an
# HTTP 400, so any "anthropic:..." model string must be called with no temperature kwarg.
ANTHROPIC_PREFIX = "anthropic:"
# Provider prefix -> the Settings field holding its key. The key is handed to the model explicitly rather than left
# for the provider class to find in os.environ: pydantic-settings does not export .env values to the environment, so
# a key set only in .env would never reach the model; replay also sends its placeholder key this way (Task 22).
PROVIDER_KEY_FIELDS = {"google_genai": "google_api_key", "anthropic": "anthropic_api_key",
                       "openrouter": "openrouter_api_key"}
# OpenRouter speaks the OpenAI API, so "openrouter:<id>" goes through LangChain's OpenAI client at this endpoint. The
# id after the prefix is OpenRouter's own and may itself contain colons, such as "qwen/qwen3.8-27b:free".
OPENROUTER_PREFIX = "openrouter:"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# OpenRouter sometimes reports an upstream failure inside an HTTP 200 reply, as {"error": {"code": 503, ...}}. The
# client retries only on the HTTP status, so a transient code in the body is copied onto the status and retried.
RETRYABLE_IN_BODY_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _promote_in_body_error(response: httpx.Response) -> None:
    if response.status_code != 200 or "json" not in response.headers.get("content-type", ""):
        return
    response.read()
    try:
        body = response.json()
    except ValueError:
        return
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, int) and code in RETRYABLE_IN_BODY_CODES:
        response.status_code = code


def _require_prefix(model_string: str) -> str:
    if ":" not in model_string:
        raise ValueError(f"model must be 'provider:model', got {model_string!r}; bare names resolve unreliably")
    return model_string


def _init_kwargs(model_string: str) -> dict:
    """Controller correction 4: only send temperature=0 to providers that accept it."""
    return {} if model_string.startswith(ANTHROPIC_PREFIX) else {"temperature": 0}


def _key_kwargs(model_string: str, settings: Settings) -> dict:
    """The provider's key from Settings, still a SecretStr. When unset, nothing is passed and the provider class
    falls back to its own environment lookup, as before."""
    field = PROVIDER_KEY_FIELDS.get(model_string.split(":", 1)[0])
    key = getattr(settings, field) if field else None
    return {"api_key": key} if key is not None else {}


def _request_kwargs(settings: Settings) -> dict:
    """Client-side timeout and retry count, which every provider's LangChain client accepts under these names. On a
    timeout the client closes the connection and retries or raises, so a late answer is never delivered; the
    fallback model, if configured, then takes over."""
    return {"timeout": settings.llm_timeout_seconds, "max_retries": settings.llm_max_retries,
            "max_tokens": settings.llm_max_output_tokens}


def _init_model(model_string: str, settings: Settings):
    """Build one chat model from a "provider:model" string, with its temperature rule, key, timeout and retries."""
    kwargs = {**_init_kwargs(model_string), **_key_kwargs(model_string, settings), **_request_kwargs(settings)}
    if model_string.startswith(OPENROUTER_PREFIX):
        return init_chat_model(model_string[len(OPENROUTER_PREFIX):], model_provider="openai",
                               base_url=OPENROUTER_BASE_URL,
                               http_client=httpx.Client(event_hooks={"response": [_promote_in_body_error]}), **kwargs)
    return init_chat_model(model_string, **kwargs)


def build_chat_model(settings: Settings, purpose: Purpose):
    """One factory for every model call so provider swap and temperature live in one place."""
    model_string = _require_prefix(settings.resolved_judge_model if purpose == "judge" else settings.llm_model)
    model = _init_model(model_string, settings)
    if settings.llm_fallback_model:
        model = model.with_fallbacks([_init_model(_require_prefix(settings.llm_fallback_model), settings)])
    return model
