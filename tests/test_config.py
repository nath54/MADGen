"""
Unit tests for configuration loaders, models, and spatial validation logic.
"""

# Import Modules
from pathlib import Path

import unittest

from src.common.types import MicrophoneType
from src.config.models import SceneConfig
from src.config.loader import (
    parse_persona,
    load_scene_config,
    parse_room_config,
    load_scene_from_dict,
    parse_microphone_config,
    validate_position_in_room,
)


class TestConfig(unittest.TestCase):
    """
    Validation suite for configuration parsing and geometry validation.
    """

    def test_validate_position_in_room_valid(self) -> None:
        """
        Verify within-bounds coordinates pass validation without exceptions.
        """

        # Inside 6x5x2.8 room
        room_dims: tuple[float, float, float] = (6.0, 5.0, 2.8)
        pos: tuple[float, float, float] = (2.0, 3.0, 1.5)

        # Should not raise
        validate_position_in_room(pos, room_dims, "TestEntity")

    def test_validate_position_out_of_bounds(self) -> None:
        """
        Verify coordinates outside room boundaries raise ValueError.
        """

        room_dims: tuple[float, float, float] = (6.0, 5.0, 2.8)

        # X too large
        with self.assertRaises(ValueError):
            validate_position_in_room((7.0, 2.0, 1.0), room_dims, "TestEntity")

        # Y negative
        with self.assertRaises(ValueError):
            validate_position_in_room((2.0, -0.5, 1.0), room_dims, "TestEntity")

        # Z above ceiling
        with self.assertRaises(ValueError):
            validate_position_in_room((2.0, 2.0, 3.5), room_dims, "TestEntity")

    def test_parse_room_config(self) -> None:
        """
        Verify parsing room dictionary into RoomConfig instance.
        """

        raw_dict = {
            "dimensions": [8.0, 7.0, 3.0],
            "absorption": 0.15,
            "max_order": 4,
            "sample_rate": 22050,
        }

        config = parse_room_config(raw_dict)
        self.assertEqual(config.dimensions, (8.0, 7.0, 3.0))
        self.assertEqual(config.absorption, 0.15)
        self.assertEqual(config.max_order, 4)
        self.assertEqual(config.sample_rate, 22050)

    def test_parse_microphone_config(self) -> None:
        """
        Verify parsing microphone dictionary into MicrophoneConfig instance.
        """

        raw_dict = {
            "position": [0.5, 0.5, 0.9],
            "mic_type": "circular",
            "mic_spacing": 0.1,
            "mic_radius": 0.04,
            "num_microphones": 4,
        }

        config = parse_microphone_config(raw_dict)
        self.assertEqual(config.position, (0.5, 0.5, 0.9))
        self.assertEqual(config.mic_type, MicrophoneType.CIRCULAR)
        self.assertEqual(config.num_microphones, 4)

    def test_parse_persona(self) -> None:
        """
        Verify parsing persona dictionary into PersonaConfig instance.
        """

        raw_dict = {
            "id": "charlie",
            "name": "Charlie",
            "voice_model": "en_US-lessac-medium.onnx",
            "speaker_id": 2,
            "position": [3.0, 2.5, 1.7],
            "size": 0.2,
            "personality": {
                "length_scale": 1.1,
                "noise_scale": 0.6,
                "volume": 0.9,
            },
            "utterances": [
                {"text": "Hello world", "start_time_s": 1.0},
            ],
        }

        persona = parse_persona(raw_dict)
        self.assertEqual(persona.id, "charlie")
        self.assertEqual(persona.speaker_id, 2)
        self.assertEqual(persona.size, 0.2)
        self.assertEqual(persona.personality.length_scale, 1.1)
        self.assertEqual(len(persona.utterances), 1)

    def test_load_scene_from_default_file(self) -> None:
        """
        Verify loading default scene configuration JSON from workspace.
        """

        config_path: Path = Path("config/default_scene.json")
        scene: SceneConfig = load_scene_config(config_path)

        # Validate loaded scene structure
        self.assertEqual(len(scene.personas), 2)
        self.assertEqual(scene.assistant.mic_type, MicrophoneType.STEREO)
        self.assertEqual(scene.room.sample_rate, 16000)

    def test_load_scene_from_dict(self) -> None:
        """
        Verify load_scene_from_dict constructs valid SceneConfig from dictionary.
        """

        raw_dict = {
            "room": {"dimensions": [5.0, 4.0, 2.5]},
            "assistant": {"position": [0.3, 0.3, 0.8]},
            "personas": [
                {"id": "p1", "position": [1.5, 1.5, 1.2]},
            ],
        }

        scene: SceneConfig = load_scene_from_dict(raw_dict)
        self.assertEqual(len(scene.personas), 1)
        self.assertEqual(scene.assistant.position, (0.3, 0.3, 0.8))

    def test_load_scene_missing_file_raises(self) -> None:
        """
        Verify missing configuration file path raises FileNotFoundError.
        """

        with self.assertRaises(FileNotFoundError):
            load_scene_config(Path("config/non_existent_file.json"))


if __name__ == "__main__":
    unittest.main()
