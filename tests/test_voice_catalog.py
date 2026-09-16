"""
Unit tests for the Piper-TTS voice catalog, metadata index, and diverse sampling.
"""

# Import Modules
import unittest

from src.tts.voice_catalog import (
    VoiceProfile,
    CURATED_VOICE_CATALOG,
    infer_gender_from_name,
    load_full_piper_catalog,
    filter_voices_by_language,
    sample_distinct_voice_profiles,
)


class TestVoiceCatalog(unittest.TestCase):
    """
    Validation suite for Piper voice registry and gender-diverse selection.
    """

    def test_infer_gender_from_name(self) -> None:
        """
        Verify gender inference accurately identifies female and male identifiers.
        """

        self.assertEqual(infer_gender_from_name("lessac"), "female")
        self.assertEqual(infer_gender_from_name("siwis"), "female")
        self.assertEqual(infer_gender_from_name("amy"), "female")
        self.assertEqual(infer_gender_from_name("alan"), "male")
        self.assertEqual(infer_gender_from_name("ryan"), "male")
        self.assertEqual(infer_gender_from_name("thorsten"), "male")
        self.assertEqual(infer_gender_from_name("unknown_nonexistent"), "unspecified")

    def test_load_full_piper_catalog(self) -> None:
        """
        Verify catalog loads all available voices with valid metadata properties.
        """

        catalog: list[VoiceProfile] = load_full_piper_catalog()

        self.assertGreaterEqual(len(catalog), len(CURATED_VOICE_CATALOG))
        for vp in catalog:
            self.assertTrue(bool(vp.key))
            self.assertTrue(bool(vp.language))
            self.assertIn(vp.gender, {"female", "male", "unspecified"})

    def test_filter_voices_by_language(self) -> None:
        """
        Verify filtering catalog by language codes correctly isolates target models.
        """

        en_voices: list[VoiceProfile] = filter_voices_by_language(
            catalog=CURATED_VOICE_CATALOG,
            languages=["en"],
        )

        self.assertGreater(len(en_voices), 0)
        for vp in en_voices:
            self.assertTrue(vp.lang_family == "en" or vp.language.startswith("en"))

    def test_sample_distinct_voice_profiles_uniqueness(self) -> None:
        """
        Verify sampled voice profiles contain strictly unique voice keys without replacement.
        """

        sampled = sample_distinct_voice_profiles(
            count=6,
            languages=["en"],
            ensure_gender_diversity=True,
        )

        self.assertEqual(len(sampled), 6)
        keys: list[str] = [vp.key for vp, _, _ in sampled]
        unique_keys: set[str] = set(keys)

        # Assert no two speakers share the same voice model key
        self.assertEqual(len(keys), len(unique_keys))

    def test_sample_distinct_voice_profiles_gender_diversity(self) -> None:
        """
        Verify sampled profiles enforce balanced representation of female and male voices.
        """

        sampled = sample_distinct_voice_profiles(
            count=4,
            languages=["en"],
            ensure_gender_diversity=True,
        )

        genders: list[str] = [gender for _, _, gender in sampled]

        # In 4 speakers with alternating gender, exactly 2 female and 2 male
        self.assertEqual(genders.count("female"), 2)
        self.assertEqual(genders.count("male"), 2)

    def test_resolve_voice_folder_with_info(self) -> None:
        """
        Verify resolve_voice_folder builds BASE_LANG/LANG_DETAIL/persona_name from voice_info.
        """

        from src.tts.voice_downloader import resolve_voice_folder  # pylint: disable=import-outside-toplevel
        from pathlib import Path  # pylint: disable=import-outside-toplevel

        base = Path("data/piper_voices")
        info = {
            "name": "lessac",
            "language": {"code": "en_US", "family": "en"},
        }
        folder = resolve_voice_folder("en_US-lessac-low", base, info)
        self.assertEqual(folder, base / "en" / "en_US" / "lessac")

    def test_resolve_voice_folder_fallback_parsing(self) -> None:
        """
        Verify resolve_voice_folder correctly infers structure from key string alone.
        """

        from src.tts.voice_downloader import resolve_voice_folder  # pylint: disable=import-outside-toplevel
        from pathlib import Path  # pylint: disable=import-outside-toplevel

        base = Path("data/piper_voices")
        folder = resolve_voice_folder("fr_FR-siwis-medium", base, None)
        self.assertEqual(folder, base / "fr" / "fr_FR" / "siwis")

    def test_synthesizer_resolves_nested_model_path(self) -> None:
        """
        Verify PiperSynthesizer locates models nested in hierarchical language directories.
        """

        import tempfile  # pylint: disable=import-outside-toplevel
        from pathlib import Path  # pylint: disable=import-outside-toplevel
        from src.tts.synthesizer import PiperSynthesizer  # pylint: disable=import-outside-toplevel

        with tempfile.TemporaryDirectory() as tmp_dir:
            voices_dir = Path(tmp_dir)
            model_dir = voices_dir / "en" / "en_US" / "lessac"
            model_dir.mkdir(parents=True)
            mock_model = model_dir / "en_US-lessac-low.onnx"
            mock_model.write_bytes(b"dummy onnx")

            synth = PiperSynthesizer(voices_dir=voices_dir, auto_download=False)
            resolved = synth.resolve_model_path("en_US-lessac-low")
            self.assertEqual(resolved, mock_model)


if __name__ == "__main__":
    unittest.main()
