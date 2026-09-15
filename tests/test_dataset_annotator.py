"""
Unit tests for dataset manifest serialization, RTTM generation, and ASR formatting.
"""

# Import Modules
import typing
from pathlib import Path

import json
import unittest

from src.config.models import (
    VocalStyle,
    SceneConfig,
    PersonaConfig,
    UtteranceConfig,
)
from src.dataset.annotator import (
    format_rttm_line,
    export_diarization_rttm,
    export_dataset_manifest,
    build_utterance_metadata,
    export_asr_transcripts_jsonl,
)


class TestDatasetAnnotator(unittest.TestCase):
    """
    Validation suite for ground-truth annotation exporters.
    """

    def test_format_rttm_line(self) -> None:
        """
        Verify RTTM line adheres to NIST standard syntax.
        """

        line: str = format_rttm_line(
            session_id="session_01",
            speaker_id="speaker_alice",
            start_time_s=1.25,
            duration_s=3.50,
        )

        expected: str = "SPEAKER session_01 1 1.25 3.50 <NA> <NA> speaker_alice <NA> <NA>\n"
        self.assertEqual(line, expected)

    def test_build_utterance_metadata(self) -> None:
        """
        Verify metadata dictionary generation captures speaker, text, and vocal style.
        """

        utt: UtteranceConfig = UtteranceConfig(
            text="Watch out!",
            start_time_s=2.0,
            language="en",
            vocal_style=VocalStyle.SHOUTING,
        )
        persona: PersonaConfig = PersonaConfig(
            id="spk_1",
            name="Alice",
            position=(1.5, 2.0, 1.7),
            size=0.25,
        )

        meta: dict[str, typing.Any] = build_utterance_metadata(utt, persona, duration_s=1.5)

        self.assertEqual(meta["speaker_id"], "spk_1")
        self.assertEqual(meta["language"], "en")
        self.assertEqual(meta["vocal_style"], "shouting")
        self.assertEqual(meta["start_time_s"], 2.0)
        self.assertEqual(meta["end_time_s"], 3.5)
        self.assertEqual(meta["text"], "Watch out!")

    def test_export_manifest_and_annotations(self) -> None:
        """
        Verify full dataset manifest and annotation file creation on disk.
        """

        temp_dir: Path = Path("data/datasets/test_sample_export")
        temp_dir.mkdir(parents=True, exist_ok=True)

        scene: SceneConfig = SceneConfig()
        utterances_meta: list[dict[str, typing.Any]] = [
            {
                "speaker_id": "spk_1",
                "start_time_s": 0.5,
                "duration_s": 2.0,
                "text": "Hello",
            },
            {
                "speaker_id": "spk_2",
                "start_time_s": 3.0,
                "duration_s": 1.5,
                "text": "Hi there",
            },
        ]

        export_dataset_manifest(
            sample_dir=temp_dir,
            session_id="test_sess",
            scene_config=scene,
            utterances_meta=utterances_meta,
            audio_duration_s=5.0,
        )

        # Check all files were created
        self.assertTrue((temp_dir / "diarization.rttm").is_file())
        self.assertTrue((temp_dir / "transcripts.jsonl").is_file())
        self.assertTrue((temp_dir / "annotations.json").is_file())

        # Validate JSON manifest
        with (temp_dir / "annotations.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["session_id"], "test_sess")
            self.assertEqual(len(data["utterances"]), 2)

        # Clean up
        for item in temp_dir.iterdir():
            item.unlink()
        temp_dir.rmdir()

    def test_export_rttm_and_jsonl_directly(self) -> None:
        """
        Verify export_diarization_rttm and export_asr_transcripts_jsonl direct execution.
        """

        temp_dir: Path = Path("data/datasets/test_direct_export")
        temp_dir.mkdir(parents=True, exist_ok=True)
        rttm_path: Path = temp_dir / "test.rttm"
        jsonl_path: Path = temp_dir / "test.jsonl"
        records: list[dict[str, typing.Any]] = [
            {"speaker_id": "spk_1", "start_time_s": 0.0, "duration_s": 1.0, "text": "Hi"},
        ]

        export_diarization_rttm(rttm_path, "sess_01", records)
        export_asr_transcripts_jsonl(jsonl_path, records)

        self.assertTrue(rttm_path.is_file())
        self.assertTrue(jsonl_path.is_file())

        rttm_path.unlink()
        jsonl_path.unlink()
        temp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
