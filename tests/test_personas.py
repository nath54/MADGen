"""
Unit tests for persona behavioral modeling, spatial size, and track synthesis.
"""

# Import Modules
import unittest

from src.personas.persona import Persona
from src.tts.synthesizer import MockSynthesizer
from src.config.models import (
    PersonaConfig,
    UtteranceConfig,
    PersonalityConfig,
)
from src.personas.manager import (
    PersonaTrack,
    PersonaManager,
    calculate_required_samples,
)


class TestPersonas(unittest.TestCase):
    """
    Validation suite for persona properties, emission points, and multi-track timelines.
    """

    def test_persona_point_source(self) -> None:
        """
        Verify persona with size 0.0 produces a single emission point.
        """

        config: PersonaConfig = PersonaConfig(
            id="alice",
            name="Alice",
            position=(2.0, 3.0, 1.5),
            size=0.0,
        )
        persona: Persona = Persona(config=config)

        points = persona.calculate_source_points()
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0], (2.0, 3.0, 1.5))

    def test_persona_extended_source(self) -> None:
        """
        Verify persona with size > 0 produces distributed emission cluster.
        """

        config: PersonaConfig = PersonaConfig(
            id="bob",
            name="Bob",
            position=(2.0, 3.0, 1.5),
            size=0.4,
        )
        persona: Persona = Persona(config=config)

        points = persona.calculate_source_points()
        # Expect center plus 4 surrounding points
        self.assertEqual(len(points), 5)
        self.assertEqual(points[0], (2.0, 3.0, 1.5))

        # Check radius offset
        radius: float = 0.4 / 2.0
        self.assertEqual(points[1], (2.0 + radius, 3.0, 1.5))
        self.assertEqual(points[2], (2.0 - radius, 3.0, 1.5))

    def test_calculate_required_samples(self) -> None:
        """
        Verify timeline sample length calculation including trailing safety padding.
        """

        # End times at 2.0s and 4.5s
        end_times: list[float] = [2.0, 4.5]
        sample_rate: int = 16000
        padding_s: float = 1.0

        samples: int = calculate_required_samples(end_times, sample_rate, padding_s=padding_s)
        expected: int = int((4.5 + 1.0) * 16000)
        self.assertEqual(samples, expected)

    def test_persona_manager_synthesis(self) -> None:
        """
        Verify PersonaManager synthesizes aligned continuous tracks for all personas.
        """

        persona1: Persona = Persona(
            config=PersonaConfig(
                id="alice",
                name="Alice",
                position=(1.0, 1.0, 1.5),
                personality=PersonalityConfig(length_scale=1.0),
                utterances=[
                    UtteranceConfig(text="Hello", start_time_s=0.0),
                ],
            )
        )
        persona2: Persona = Persona(
            config=PersonaConfig(
                id="bob",
                name="Bob",
                position=(3.0, 3.0, 1.5),
                personality=PersonalityConfig(length_scale=1.0),
                utterances=[
                    UtteranceConfig(text="Hi there", start_time_s=1.0),
                ],
            )
        )

        manager: PersonaManager = PersonaManager(personas=[persona1, persona2])
        synthesizer: MockSynthesizer = MockSynthesizer()

        tracks: list[PersonaTrack] = manager.synthesize_all_tracks(
            synthesizer=synthesizer,
            sample_rate=16000,
            padding_s=1.0,
        )

        self.assertEqual(len(tracks), 2)
        self.assertGreater(tracks[0].total_samples, 0)
        self.assertEqual(tracks[0].total_samples, tracks[1].total_samples)


if __name__ == "__main__":
    unittest.main()
