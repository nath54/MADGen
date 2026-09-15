"""
Procedural dialogue generation engine driven by LLM with thematic and ambiance constraints.

Prompts llama.cpp via structured tool calling to synthesize rich multi-person conversations
incorporating 100k-word dictionary constraints, ambiance settings, and speaker personas.
"""

# Import Modules
import typing

import json
import logging

import httpx

from src.config.models import PersonaConfig
from src.llm.client import LLMClient
from src.llm.llm_types import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    GeneratedTurn,
    ToolCall,
    ToolDefinition,
)
from src.procedural.ambiance_presets import AmbiancePreset
from src.procedural.dialogue_bank import sample_dialogue_text

logger: logging.Logger = logging.getLogger(__name__)

# Structured tool definition ensuring consistent schema output from the LLM
DIALOGUE_SUBMISSION_TOOL: ToolDefinition = {
    "type": "function",
    "function": {
        "name": "submit_dialogue",
        "description": "Submit a generated multi-turn conversation sequence between personas.",
        "parameters": {
            "type": "object",
            "properties": {
                "turns": {
                    "type": "array",
                    "description": "List of chronological spoken dialogue turns.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker_id": {
                                "type": "string",
                                "description": "Unique identifier of the persona speaking.",
                            },
                            "text": {
                                "type": "string",
                                "description": "Spoken text dialogue sentence.",
                            },
                            "vocal_style": {
                                "type": "string",
                                "enum": ["normal", "shouting", "laughter", "interruption"],
                                "description": "Vocal emotion style preset.",
                            },
                        },
                        "required": ["speaker_id", "text", "vocal_style"],
                    },
                },
            },
            "required": ["turns"],
        },
    },
}


def _extract_raw_arguments(response: ChatCompletionResponse) -> str:
    """
    Extract raw JSON argument string from tool call response or content body.

    Args:
        response (ChatCompletionResponse): Raw server completion response.

    Returns:
        str: Raw JSON argument substring, or empty string if absent.
    """

    if not response.get("choices"):
        return ""

    choice: ChatCompletionChoice = response["choices"][0]
    message: ChatMessage = choice.get("message", {})
    tool_calls: list[ToolCall] = message.get("tool_calls", [])

    # Extract arguments from first submit_dialogue tool call
    for call in tool_calls:
        if call.get("function", {}).get("name") == "submit_dialogue":
            return str(call.get("function", {}).get("arguments", ""))

    # Fallback: check content if tool call not present
    raw_content: typing.Any = message.get("content")
    if raw_content:
        content_str: str = str(raw_content)
        if "{" in content_str and "turns" in content_str:
            return content_str[content_str.find("{") : content_str.rfind("}") + 1]

    return ""


class LLMDialogueGenerator:
    """
    Synthesizes contextual conversations for groups using llama.cpp and constraint anchors.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        base_url: str = "http://127.0.0.1:8080/v1",
        timeout_s: float = 60.0,
    ) -> None:
        """
        Initialize the LLM dialogue generator.

        Args:
            client (LLMClient | None): Optional existing LLMClient instance.
            base_url (str): Endpoint URL if creating a new client.
            timeout_s (float): Timeout limit in seconds.
        """

        self.client: LLMClient = (
            client if client is not None else LLMClient(base_url=base_url, timeout_s=timeout_s)
        )

    def is_available(self) -> bool:
        """
        Check whether the underlying LLM inference server is online.

        Returns:
            bool: True if server is healthy, False otherwise.
        """

        return self.client.is_healthy()

    def build_system_prompt(
        self,
        ambiance: AmbiancePreset,
        constraint_words: list[str],
    ) -> str:
        """
        Construct system prompt embedding ambiance roleplay rules and vocabulary constraints.

        Args:
            ambiance (AmbiancePreset): Conversational ambiance preset.
            constraint_words (list[str]): Constraint keywords from 100k dictionary.

        Returns:
            str: Assembled system instructions string.
        """

        constraints_str: str = ", ".join(f"'{w}'" for w in constraint_words)

        prompt: str = (
            "You are an expert dialogue writer for acoustic speech simulation datasets.\n"
            f"Scene Setting: {ambiance.display_name} - {ambiance.description}\n"
            f"Roleplay Guidelines: {ambiance.prompt_guidelines}\n\n"
            f"Thematic Constraints: The conversation MUST naturally incorporate and revolve "
            f"around these concepts or keywords: {constraints_str}.\n"
            "Create organic, conversational exchanges with realistic turn-taking, occasional "
            "laughter, interruptions, or shouts where appropriate according to the scene setting.\n"
            "Always submit your final dialogue by calling the 'submit_dialogue' tool."
        )

        return prompt

    def build_user_prompt(
        self,
        group: list[PersonaConfig],
        target_turns_count: int,
    ) -> str:
        """
        Construct user prompt detailing participating personas and turn count requirements.

        Args:
            group (list[PersonaConfig]): Personas in the conversational group.
            target_turns_count (int): Approximate number of dialogue turns requested.

        Returns:
            str: Assembled user prompt string.
        """

        speakers_info: list[str] = [
            f"- {p.id}: {p.name} (gender: {p.gender}, language: {p.language})"
            for p in group
        ]
        roster_str: str = "\n".join(speakers_info)

        user_prompt: str = (
            f"Please generate a multi-turn conversation of approximately {target_turns_count} "
            f"dialogue turns between the following participants:\n{roster_str}\n\n"
            "Ensure every speaker participates with lines reflecting their identity. "
            "Call the submit_dialogue tool with the list of turns."
        )

        return user_prompt

    def parse_tool_response(
        self,
        response: ChatCompletionResponse,
        group_ids: set[str],
    ) -> list[GeneratedTurn]:
        """
        Parse structured turns from LLM tool call response.

        Args:
            response (ChatCompletionResponse): Raw server completion response.
            group_ids (set[str]): Valid persona IDs in the conversational clique.

        Returns:
            list[GeneratedTurn]: Extracted and validated turns list.
        """

        raw_args: str = _extract_raw_arguments(response)
        if not raw_args:
            return []

        try:
            payload: dict[str, typing.Any] = json.loads(raw_args)
            raw_turns: list[dict[str, typing.Any]] = payload.get("turns", [])
            valid_id_list: list[str] = sorted(list(group_ids))

            parsed_turns: list[GeneratedTurn] = []
            for item in raw_turns:
                spk: str = str(item.get("speaker_id", ""))
                text: str = str(item.get("text", "")).strip()
                style: str = str(item.get("vocal_style", "normal")).lower()

                # Fix invalid speaker ID if mismatched
                if spk not in group_ids and valid_id_list:
                    spk = valid_id_list[len(parsed_turns) % len(valid_id_list)]

                if text:
                    parsed_turns.append(
                        GeneratedTurn(speaker_id=spk, text=text, vocal_style=style)
                    )

            return parsed_turns
        except json.JSONDecodeError as exc:
            logger.warning("Failed parsing dialogue JSON from tool call: %s", exc)
            return []

    def generate_fallback_dialogue(
        self,
        group: list[PersonaConfig],
        target_turns_count: int,
    ) -> list[GeneratedTurn]:
        """
        Synthesize fallback dialogue turns using built-in offline dialogue bank.

        Args:
            group (list[PersonaConfig]): Participating personas.
            target_turns_count (int): Target number of dialogue turns.

        Returns:
            list[GeneratedTurn]: Generated fallback turns.
        """

        turns: list[GeneratedTurn] = []

        for i in range(target_turns_count):
            speaker: PersonaConfig = group[i % len(group)]
            text: str = sample_dialogue_text(language=speaker.language, category="general")
            turns.append(GeneratedTurn(speaker_id=speaker.id, text=text, vocal_style="normal"))

        return turns

    def _generate_dialogue_chunk(
        self,
        group: list[PersonaConfig],
        system_prompt: str,
        target_chunk_count: int,
        group_ids: set[str],
        prior_turns: list[GeneratedTurn],
        temperature: float,
    ) -> list[GeneratedTurn]:
        """
        Synthesize an individual chunk of conversational turns from the LLM.

        Args:
            group (list[PersonaConfig]): Participating personas.
            system_prompt (str): Active system instruction.
            target_chunk_count (int): Turn count requested for this chunk.
            group_ids (set[str]): Valid persona IDs.
            prior_turns (list[GeneratedTurn]): Previously generated turns for context.
            temperature (float): Sampling temperature.

        Returns:
            list[GeneratedTurn]: Synthesized dialogue turns for this chunk.
        """

        if not prior_turns:
            user_prompt: str = self.build_user_prompt(group, target_chunk_count)
        else:
            recent_lines: list[str] = [f"{t.speaker_id}: {t.text}" for t in prior_turns[-3:]]
            context_str: str = "\n".join(recent_lines)
            user_prompt = (
                f"Continue the conversation naturally for {target_chunk_count} more turns.\n"
                f"Recent dialogue context:\n{context_str}\n\n"
                "Ensure participants speak according to their persona. "
                "Call submit_dialogue with the turns."
            )

        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tool_choice = {"type": "function", "function": {"name": "submit_dialogue"}}

        response: ChatCompletionResponse = self.client.send_chat(
            messages=messages,
            tools=[DIALOGUE_SUBMISSION_TOOL],
            tool_choice=tool_choice,
            temperature=temperature,
        )
        return self.parse_tool_response(response, group_ids)

    def generate_group_dialogue(
        self,
        group: list[PersonaConfig],
        ambiance: AmbiancePreset,
        constraint_words: list[str],
        target_turns_count: int = 6,
        temperature: float = 0.7,
    ) -> list[GeneratedTurn]:
        """
        Generate complete conversation turns for a group using LLM or offline fallback.

        Args:
            group (list[PersonaConfig]): Conversational group personas.
            ambiance (AmbiancePreset): Active ambiance preset.
            constraint_words (list[str]): Thematic constraint keywords.
            target_turns_count (int): Target turn count.
            temperature (float): Sampling temperature.

        Returns:
            list[GeneratedTurn]: Chronological list of generated dialogue turns.
        """

        group_ids: set[str] = {p.id for p in group}
        system_prompt: str = self.build_system_prompt(ambiance, constraint_words)
        all_turns: list[GeneratedTurn] = []
        chunk_size: int = 15

        # Attempt chunked LLM generation
        try:
            while len(all_turns) < target_turns_count:
                needed: int = target_turns_count - len(all_turns)
                step_count: int = min(chunk_size, needed)
                chunk_turns: list[GeneratedTurn] = self._generate_dialogue_chunk(
                    group=group,
                    system_prompt=system_prompt,
                    target_chunk_count=step_count,
                    group_ids=group_ids,
                    prior_turns=all_turns,
                    temperature=temperature,
                )
                if not chunk_turns:
                    break
                all_turns.extend(chunk_turns)

            if all_turns:
                logger.info("Successfully generated %d LLM dialogue turns", len(all_turns))
                return all_turns

            logger.warning("LLM returned empty dialogue turns. Using offline fallback.")
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            logger.warning("LLM dialogue generation failed: %s. Using offline fallback.", exc)

        # Pad remaining needed turns with fallback if partially generated
        if all_turns:
            needed = target_turns_count - len(all_turns)
            if needed > 0:
                all_turns.extend(self.generate_fallback_dialogue(group, needed))
            return all_turns

        return self.generate_fallback_dialogue(group, target_turns_count)
