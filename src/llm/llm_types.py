"""
Strongly typed data contracts for LLM completions, tool definitions, and dialogue turns.

Provides standard OpenAI-compatible wire protocol structures for llama.cpp server
communication and dataclasses for structured conversational turn generation.
"""

# Import Modules
import typing
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict


class JSONSchemaProperty(TypedDict):
    """
    Schema property descriptor for tool calling parameter validation.
    """

    type: str
    description: NotRequired[str]
    enum: NotRequired[list[str]]
    items: NotRequired[dict[str, typing.Any]]


class ToolParametersSchema(TypedDict):
    """
    JSON Schema parameters object for function calling definitions.
    """

    type: Literal["object"]
    properties: dict[str, JSONSchemaProperty]
    required: list[str]


class FunctionDefinition(TypedDict):
    """
    Specification of a callable tool function for the LLM.
    """

    name: str
    description: str
    parameters: ToolParametersSchema


class ToolDefinition(TypedDict):
    """
    Tool envelope passed in chat completion requests.
    """

    type: Literal["function"]
    function: FunctionDefinition


class FunctionCall(TypedDict):
    """
    Function call payload emitted by the model.
    """

    name: str
    arguments: str


class ToolCall(TypedDict):
    """
    Tool invocation element emitted by the assistant in chat responses.
    """

    id: str
    type: Literal["function"]
    function: FunctionCall


class ChatMessage(TypedDict, total=False):
    """
    A single conversational message within the LLM chat history.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None
    name: str
    tool_calls: list[ToolCall]
    tool_call_id: str


class ChatCompletionChoice(TypedDict):
    """
    A single completion choice returned by the server.
    """

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "tool_calls", "length"] | None


class ChatCompletionResponse(TypedDict):
    """
    Full response returned by OpenAI-compatible chat completion endpoints.
    """

    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]


@dataclass
class GeneratedTurn:
    """
    A single parsed dialogue turn emitted by the LLM dialogue generator.
    """

    speaker_id: str
    text: str
    vocal_style: str = "normal"
