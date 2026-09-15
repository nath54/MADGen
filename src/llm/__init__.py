"""
LLM inference and structured dialogue generation package for speech simulation.
"""

# Import Modules
from src.llm.client import LLMClient
from src.llm.llm_types import (
    ChatMessage,
    GeneratedTurn,
    ToolDefinition,
    ChatCompletionResponse,
)
from src.llm.dialogue_generator import LLMDialogueGenerator

__all__: list[str] = [
    "LLMClient",
    "LLMDialogueGenerator",
    "GeneratedTurn",
    "ChatMessage",
    "ToolDefinition",
    "ChatCompletionResponse",
]
