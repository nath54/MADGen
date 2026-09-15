"""
Unit tests for randomized scene generation and acoustic parameter variations.
"""

# Import Modules
import unittest

from src.config.models import DatasetSampleConfig
from src.procedural.variation_generator import (
    build_random_persona,
    generate_random_scene,
    sample_room_dimensions,
    sample_assistant_corner_position,
    sample_persona_spatial_position,
)


class TestVariations(unittest.TestCase):
    """
    Validation suite for procedural room, microphone, and persona parameter sampling.
    """

    def test_sample_room_dimensions(self) -> None:
        """
        Verify sampled room dimensions fall within realistic residential ranges.
        """

        for _ in range(10):
            dim_x, dim_y, dim_z = sample_room_dimensions()
            self.assertGreaterEqual(dim_x, 4.5)
            self.assertLessEqual(dim_x, 9.5)
            self.assertGreaterEqual(dim_y, 4.5)
            self.assertLessEqual(dim_y, 9.5)
            self.assertGreaterEqual(dim_z, 2.5)
            self.assertLessEqual(dim_z, 3.4)

    def test_sample_assistant_corner_position(self) -> None:
        """
        Verify assistant microphone is placed near room corners within room volume.
        """

        room_dims: tuple[float, float, float] = (7.0, 6.0, 2.8)
        for _ in range(10):
            pos_x, pos_y, pos_z = sample_assistant_corner_position(room_dims)
            self.assertGreater(pos_x, 0.0)
            self.assertLess(pos_x, room_dims[0])
            self.assertGreater(pos_y, 0.0)
            self.assertLess(pos_y, room_dims[1])
            self.assertGreaterEqual(pos_z, 0.75)
            self.assertLessEqual(pos_z, 1.1)

    def test_sample_persona_spatial_position(self) -> None:
        """
        Verify sampled speaker positions maintain safe distance from walls and mic.
        """

        room_dims: tuple[float, float, float] = (8.0, 8.0, 3.0)
        mic_pos: tuple[float, float, float] = (0.4, 0.4, 0.85)

        for _ in range(10):
            pos_x, pos_y, pos_z = sample_persona_spatial_position(room_dims, mic_pos)
            self.assertGreaterEqual(pos_x, 0.8)
            self.assertLessEqual(pos_x, 7.2)
            self.assertGreaterEqual(pos_y, 0.8)
            self.assertLessEqual(pos_y, 7.2)
            self.assertGreater(pos_z, 0.0)
            self.assertLess(pos_z, room_dims[2])

            dist_to_mic: float = ((pos_x - mic_pos[0]) ** 2 + (pos_y - mic_pos[1]) ** 2) ** 0.5
            self.assertGreaterEqual(dist_to_mic, 1.0)

    def test_build_random_persona(self) -> None:
        """
        Verify random persona generation sets valid language and personality parameters.
        """

        room_dims: tuple[float, float, float] = (6.0, 5.0, 2.8)
        mic_pos: tuple[float, float, float] = (0.4, 0.4, 0.85)

        persona = build_random_persona(
            persona_index=1,
            language="fr",
            room_dims=room_dims,
            assistant_pos=mic_pos,
        )

        self.assertEqual(persona.id, "speaker_1")
        self.assertEqual(persona.language, "fr")
        self.assertGreaterEqual(persona.personality.length_scale, 0.85)
        self.assertLessEqual(persona.personality.length_scale, 1.25)

    def test_generate_random_scene(self) -> None:
        """
        Verify generate_random_scene produces fully instantiated SceneConfig with distinct voices.
        """

        sample_cfg: DatasetSampleConfig = DatasetSampleConfig(
            sample_id="test_001",
            duration_s=30.0,
            languages=["en", "fr"],
        )

        scene = generate_random_scene(sample_cfg, num_speakers=4)

        self.assertEqual(len(scene.personas), 4)
        self.assertGreater(scene.room.dimensions[0], 0.0)
        self.assertGreater(scene.assistant.position[0], 0.0)

        # Verify all voice models are distinct
        voice_models: list[str] = [p.voice_model for p in scene.personas]
        self.assertEqual(len(voice_models), len(set(voice_models)))

        # Verify gender diversity (both female and male present)
        genders: list[str] = [p.gender for p in scene.personas]
        self.assertIn("female", genders)
        self.assertIn("male", genders)


if __name__ == "__main__":
    unittest.main()
