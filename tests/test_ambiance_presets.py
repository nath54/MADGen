"""
Unit tests for conversational ambiance presets and acoustic social constraints.
"""

# Import Modules
import unittest

from src.procedural.ambiance_presets import (
    AmbiancePreset,
    get_preset_names,
    get_ambiance_preset,
    get_all_ambiance_presets,
)


class TestAmbiancePresets(unittest.TestCase):
    """
    Validation suite for ambiance presets, prompt rules, and parameter boundaries.
    """

    def test_preset_names_inventory(self) -> None:
        """
        Verify all expected ambiance presets are registered in the inventory.
        """

        names: list[str] = get_preset_names()

        self.assertGreaterEqual(len(names), 12)
        expected_presets: list[str] = [
            "casual_chit_chat",
            "kitchen_cooking",
            "workplace_meeting",
            "heated_argument",
            "party_celebration",
            "smart_assistant_household",
            "gaming_session",
            "interview_podcast",
            "late_night_philosophy",
            "family_dinner",
            "academic_defense",
            "emergency_rush",
        ]
        for name in expected_presets:
            self.assertIn(name, names)

    def test_preset_acoustic_parameters_bounds(self) -> None:
        """
        Verify all presets define valid probabilities and acoustic SNR levels.
        """

        all_presets: dict[str, AmbiancePreset] = get_all_ambiance_presets()

        for name, preset in all_presets.items():
            self.assertEqual(preset.name, name)
            self.assertGreater(len(preset.display_name), 0)
            self.assertGreater(len(preset.prompt_guidelines), 0)

            # Bounds verification
            self.assertGreaterEqual(preset.overlap_rate, 0.0)
            self.assertLessEqual(preset.overlap_rate, 1.0)

            self.assertGreaterEqual(preset.shout_rate, 0.0)
            self.assertLessEqual(preset.shout_rate, 1.0)

            self.assertGreaterEqual(preset.laugh_rate, 0.0)
            self.assertLessEqual(preset.laugh_rate, 1.0)

            self.assertGreaterEqual(preset.ambient_snr_db, 10.0)
            self.assertLessEqual(preset.ambient_snr_db, 40.0)

            self.assertGreaterEqual(preset.num_constraint_words, 1)

    def test_get_ambiance_preset_dispatch(self) -> None:
        """
        Verify retrieval by exact name, random selection, and fallback resolution.
        """

        # Exact match
        cooking = get_ambiance_preset("kitchen_cooking")
        self.assertEqual(cooking.name, "kitchen_cooking")

        # Random match
        random_preset = get_ambiance_preset("random")
        self.assertIsInstance(random_preset, AmbiancePreset)

        # Fallback for unknown preset
        fallback = get_ambiance_preset("nonexistent_preset_xyz")
        self.assertEqual(fallback.name, "casual_chit_chat")


if __name__ == "__main__":
    unittest.main()
