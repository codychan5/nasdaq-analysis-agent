from typing import Iterator
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedChatModel(GenericFakeChatModel):
    """Plays scripted AIMessages in order. Records the tool names bound before each call so tests can
    assert which tools the gating middleware offered."""
    bound_tool_names: list[list[str]] = []

    def bind_tools(self, tools, **kwargs):
        object.__setattr__(self, "bound_tool_names", self.bound_tool_names + [[t.name for t in tools]])
        return self


def scripted(messages: list[AIMessage]) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(messages))
