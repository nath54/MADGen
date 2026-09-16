"""
Unit tests for LLM dialogue generator prompt construction, tool parsing, and fallback.
"""

# Import Modules
import json
import unittest

from src.llm.client import LLMClient
from src.config.models import PersonaConfig
from src.procedural.ambiance_presets import get_ambiance_preset
from src.llm.dialogue_generator import (
    parse_script_response,
    LLMDialogueGenerator,
)
from src.llm.llm_types import (
    GeneratedTurn,
    ChatCompletionResponse,
)


class TestLLMDialogue(unittest.TestCase):
    """
    Validation suite for LLM prompt composition, tool response parsing, and fallback.
    """

    def setUp(self) -> None:
        """
        Set up shared test fixtures and personas.
        """

        self.generator: LLMDialogueGenerator = LLMDialogueGenerator()
        self.personas: list[PersonaConfig] = [
            PersonaConfig(id="spk_alice", name="Alice", gender="female", language="en"),
            PersonaConfig(id="spk_bob", name="Bob", gender="male", language="en"),
        ]
        self.ambiance = get_ambiance_preset("kitchen_cooking")
        self.constraint_words = ["rosemary", "saffron", "oven"]

    def test_build_system_prompt_contains_constraints_and_ambiance(self) -> None:
        """
        Verify system prompt embeds ambiance guidelines and constraint keywords.
        """

        prompt: str = self.generator.build_system_prompt(
            ambiance=self.ambiance,
            constraint_words=self.constraint_words,
        )

        self.assertIn("Kitchen Cooking", prompt)
        self.assertIn("rosemary", prompt)
        self.assertIn("saffron", prompt)
        self.assertIn("oven", prompt)
        self.assertIn("<speaker_id> [<vocal_style>]:", prompt)

    def test_build_user_prompt_contains_speaker_metadata(self) -> None:
        """
        Verify user prompt contains speaker names, IDs, genders, and requested turn count.
        """

        user_prompt: str = self.generator.build_user_prompt(
            group=self.personas,
            target_turns_count=6,
        )

        self.assertIn("spk_alice", user_prompt)
        self.assertIn("spk_bob", user_prompt)
        self.assertIn("female", user_prompt)
        self.assertIn("male", user_prompt)
        self.assertIn("6", user_prompt)

    def test_parse_script_response(self) -> None:
        """
        Verify parsing plain-text script lines with vocal styles and speaker mapping.
        """

        raw_text: str = (
            "spk_alice [normal]: Is the saffron ready?\n"
            "- spk_bob [shouting]: Watch out, the pan is hot!\n"
            "* Alice [laughter]: That was close!\n"
            "Bob: Let's turn down the heat.\n"
        )
        turns: list[GeneratedTurn] = parse_script_response(raw_text, self.personas)
        self.assertEqual(len(turns), 4)

        self.assertEqual(turns[0].speaker_id, "spk_alice")
        self.assertEqual(turns[0].text, "Is the saffron ready?")
        self.assertEqual(turns[0].vocal_style, "normal")

        self.assertEqual(turns[1].speaker_id, "spk_bob")
        self.assertEqual(turns[1].text, "Watch out, the pan is hot!")
        self.assertEqual(turns[1].vocal_style, "shouting")

        self.assertEqual(turns[2].speaker_id, "spk_alice")
        self.assertEqual(turns[2].text, "That was close!")
        self.assertEqual(turns[2].vocal_style, "laughter")

        self.assertEqual(turns[3].speaker_id, "spk_bob")
        self.assertEqual(turns[3].text, "Let's turn down the heat.")
        self.assertEqual(turns[3].vocal_style, "normal")

    def test_parse_script_response_formatting_variations(self) -> None:
        """
        Verify robust parsing across numbering, bolding, display names, and colon fallbacks.
        """

        group: list[PersonaConfig] = [
            PersonaConfig(id="speaker_0", name="Speaker 0"),
            PersonaConfig(id="speaker_1", name="Speaker 1"),
        ]
        raw_text: str = (
            "1. [speaker_0] [normal]: First line of dialogue\n"
            "**Speaker 0**: [normal] Second line\n"
            "Speaker 0 (male) [normal]: Third line\n"
            "- Speaker 1 [laughter]: Fourth line\n"
            "* speaker_1 (shouting): Fifth line!\n"
            "speaker_0: Sixth line without style\n"
        )
        turns: list[GeneratedTurn] = parse_script_response(raw_text, group)
        self.assertEqual(len(turns), 6)
        self.assertEqual(turns[0].speaker_id, "speaker_0")
        self.assertEqual(turns[1].speaker_id, "speaker_0")
        self.assertEqual(turns[2].speaker_id, "speaker_0")
        self.assertEqual(turns[3].speaker_id, "speaker_1")
        self.assertEqual(turns[3].vocal_style, "laughter")
        self.assertEqual(turns[4].speaker_id, "speaker_1")
        self.assertEqual(turns[4].vocal_style, "shouting")
        self.assertEqual(turns[5].speaker_id, "speaker_0")
        self.assertEqual(turns[5].vocal_style, "normal")

    def test_parse_script_response_think_tags(self) -> None:
        """
        Verify that <think>...</think> reasoning blocks from thinking models are stripped.
        """

        raw_text: str = (
            "<think>\n"
            "I should write a realistic discussion.\n"
            "Alice asks about saffron.\n"
            "</think>\n"
            "spk_alice [normal]: Let's begin the preparation.\n"
        )
        turns: list[GeneratedTurn] = parse_script_response(raw_text, self.personas)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].speaker_id, "spk_alice")
        self.assertEqual(turns[0].text, "Let's begin the preparation.")

    def test_parse_tool_response_success(self) -> None:
        """
        Verify parsing structured turns from a valid LLM tool call payload.
        """

        mock_payload = {
            "turns": [
                {
                    "speaker_id": "spk_alice",
                    "text": "Is the saffron ready?",
                    "vocal_style": "normal",
                },
                {
                    "speaker_id": "spk_bob",
                    "text": "Watch out, the pan is hot!",
                    "vocal_style": "shouting",
                },
            ]
        }

        mock_response: ChatCompletionResponse = {
            "id": "mock_id",
            "object": "chat.completion",
            "created": 12345,
            "model": "mock_model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_001",
                                "type": "function",
                                "function": {
                                    "name": "submit_dialogue",
                                    "arguments": json.dumps(mock_payload),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        group_ids: set[str] = {"spk_alice", "spk_bob"}
        turns: list[GeneratedTurn] = self.generator.parse_tool_response(mock_response, group_ids)

        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].speaker_id, "spk_alice")
        self.assertEqual(turns[0].text, "Is the saffron ready?")
        self.assertEqual(turns[0].vocal_style, "normal")

        self.assertEqual(turns[1].speaker_id, "spk_bob")
        self.assertEqual(turns[1].vocal_style, "shouting")

    def test_generate_fallback_dialogue(self) -> None:
        """
        Verify fallback dialogue generation creates valid turns using offline bank.
        """

        turns: list[GeneratedTurn] = self.generator.generate_fallback_dialogue(
            group=self.personas,
            target_turns_count=5,
        )

        self.assertEqual(len(turns), 5)
        for t in turns:
            self.assertIn(t.speaker_id, {"spk_alice", "spk_bob"})
            self.assertGreater(len(t.text), 0)

    def test_offline_fallback_on_unreachable_server(self) -> None:
        """
        Verify generate_group_dialogue safely falls back when LLM server is unreachable.
        """

        unreachable_client = LLMClient(base_url="http://127.0.0.1:59999/v1", timeout_s=0.5)
        offline_generator = LLMDialogueGenerator(client=unreachable_client)

        turns = offline_generator.generate_group_dialogue(
            group=self.personas,
            ambiance=self.ambiance,
            constraint_words=self.constraint_words,
            target_turns_count=4,
        )

        # Must not crash, should return fallback turns
        self.assertEqual(len(turns), 4)


if __name__ == "__main__":
    unittest.main()
