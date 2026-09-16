"""
Unit tests for multilingual dialogue banks and procedural conversation generation.
"""

# Import Modules
import typing

import unittest

from src.config.models import PersonaConfig
from src.tts.synthesizer import MockSynthesizer
from src.personas.persona import Persona
from src.personas.manager import synthesize_persona_clips
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

    def test_partition_conversational_groups_disabled(self) -> None:
        """
        Verify group partitioning keeps all personas in a single group when allow_parallel is False.
        """

        personas: list[PersonaConfig] = [
            PersonaConfig(id=f"p_{i}", name=f"P {i}") for i in range(5)
        ]
        groups = partition_conversational_groups(personas, allow_parallel=False)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 5)

    def test_parallel_discussions_multi_phase(self) -> None:
        """
        Verify parallel discussions feature staggered side chats and non-isolated speakers.
        """

        personas: list[PersonaConfig] = [
            PersonaConfig(id=f"spk_{i}", name=f"Speaker {i}", language="en")
            for i in range(4)
        ]
        generate_conversations_for_personas(
            personas=personas,
            min_sentences=30,
            allow_parallel=True,
        )

        all_group_ids: set[int] = {
            u.group_id for p in personas for u in p.utterances
        }
        self.assertIn(0, all_group_ids)
        self.assertIn(1, all_group_ids)

        # Side group (spk_1, spk_2) should have utterances in group 1 and group 0 (no isolation)
        spk_1_groups: set[int] = {u.group_id for u in personas[1].utterances}
        self.assertIn(0, spk_1_groups)
        self.assertIn(1, spk_1_groups)

        # Side conversation should start after t=0.5
        side_start_times: list[float] = [
            u.start_time_s for p in personas for u in p.utterances if u.group_id == 1
        ]
        self.assertTrue(all(t > 1.0 for t in side_start_times))

    def test_no_speaker_self_overlap_invariant(self) -> None:
        """
        Verify that synthesize_persona_clips guarantees zero self-overlap for all speakers.
        """

        personas: list[PersonaConfig] = [
            PersonaConfig(id=f"spk_{i}", name=f"Speaker {i}", language="en")
            for i in range(4)
        ]
        generate_conversations_for_personas(
            personas=personas,
            min_sentences=30,
            allow_parallel=True,
        )
        persona_objs: list[Persona] = [Persona(config=p) for p in personas]
        synthesizer: MockSynthesizer = MockSynthesizer()
        _clips, _ends, records = synthesize_persona_clips(
            personas=persona_objs,
            synthesizer=synthesizer,
            sample_rate=16000,
        )

        # For each persona, check that no clips for the same persona overlap in time
        for p_id in [p.id for p in personas]:
            p_records: list[dict[str, typing.Any]] = sorted(
                [r for r in records if r["speaker_id"] == p_id],
                key=lambda r: float(r["start_time_s"]),
            )
            for i in range(len(p_records) - 1):
                r1: dict[str, typing.Any] = p_records[i]
                r2: dict[str, typing.Any] = p_records[i + 1]
                r1_end: float = float(r1["start_time_s"]) + float(r1["duration_s"])
                r2_start: float = float(r2["start_time_s"])
                self.assertGreaterEqual(
                    round(r2_start, 2),
                    round(r1_end, 2),
                    f"Temporal self-collision detected for persona {p_id}!",
                )


if __name__ == "__main__":
    unittest.main()
