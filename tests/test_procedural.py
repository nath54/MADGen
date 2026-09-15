"""
Unit tests for multilingual dialogue banks and procedural conversation generation.
"""

# Import Modules
import unittest

from src.config.models import PersonaConfig
from src.procedural.dialogue_bank import (
    sample_dialogue_text,
    get_supported_languages,
)
from src.procedural.conversation_generator import (
    calculate_turn_timing,
    estimate_utterance_duration,
    partition_conversational_groups,
    generate_conversations_for_personas,
)


class TestProcedural(unittest.TestCase):
    """
    Validation suite for procedural dialogue sampling and timeline sequencing.
    """

    def test_supported_languages(self) -> None:
        """
        Verify multilingual repository includes English, French, Spanish, and German.
        """

        langs = get_supported_languages()
        self.assertIn("en", langs)
        self.assertIn("fr", langs)
        self.assertIn("es", langs)
        self.assertIn("de", langs)

    def test_sample_dialogue_multilingual(self) -> None:
        """
        Verify text sampling retrieves non-empty strings across all supported languages.
        """

        for lang in ["en", "fr", "es", "de"]:
            text_general: str = sample_dialogue_text(language=lang, category="general")
            self.assertGreater(len(text_general), 0)

            text_shout: str = sample_dialogue_text(language=lang, category="shouting")
            self.assertGreater(len(text_shout), 0)

            text_laugh: str = sample_dialogue_text(language=lang, category="laughter")
            self.assertGreater(len(text_laugh), 0)

    def test_estimate_utterance_duration(self) -> None:
        """
        Verify duration estimation increases with word count and remains positive.
        """

        short_phrase: str = "Hello there."
        long_phrase: str = "This is a much longer phrase with many more words to synthesize."

        dur_short: float = estimate_utterance_duration(short_phrase)
        dur_long: float = estimate_utterance_duration(long_phrase)

        self.assertGreater(dur_short, 0.0)
        self.assertGreater(dur_long, dur_short)

    def test_partition_conversational_groups(self) -> None:
        """
        Verify group partitioning splits large parties into cliques of 2 or 3.
        """

        personas: list[PersonaConfig] = [
            PersonaConfig(id=f"p_{i}", name=f"P {i}") for i in range(5)
        ]
        groups = partition_conversational_groups(personas)

        # 5 personas should split into 2 groups (e.g. 3 and 2)
        self.assertEqual(len(groups), 2)
        total_in_groups: int = sum(len(g) for g in groups)
        self.assertEqual(total_in_groups, 5)

    def test_calculate_turn_timing(self) -> None:
        """
        Verify turn timing computes valid forward or overlapping start timestamps.
        """

        prev_end: float = 5.0

        # Normal gap (overlap_rate = 0.0)
        start_normal, is_cut_normal = calculate_turn_timing(prev_end, overlap_rate=0.0)
        self.assertFalse(is_cut_normal)
        self.assertGreater(start_normal, prev_end)

        # Forced overlap (overlap_rate = 1.0)
        start_cut, is_cut = calculate_turn_timing(prev_end, overlap_rate=1.0)
        self.assertTrue(is_cut)
        self.assertLess(start_cut, prev_end)

    def test_generate_conversations_for_personas(self) -> None:
        """
        Verify conversation generation populates persona utterances across timeline.
        """

        personas: list[PersonaConfig] = [
            PersonaConfig(id="alice", name="Alice", language="en"),
            PersonaConfig(id="bob", name="Bob", language="en"),
        ]

        generate_conversations_for_personas(
            personas=personas,
            duration_s=25.0,
            overlap_rate=0.3,
            shout_rate=0.2,
            laugh_rate=0.2,
        )

        # Check that utterances were added
        total_utts: int = len(personas[0].utterances) + len(personas[1].utterances)
        self.assertGreater(total_utts, 2)


if __name__ == "__main__":
    unittest.main()
