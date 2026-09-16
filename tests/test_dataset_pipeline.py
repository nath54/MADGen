"""
Unit tests for the procedural dataset batch generation pipeline.

Verifies single-sample generation, isolated stems export, batch manifest production,
ambiance integration, and thematic dictionary constraints.
"""

# Import Modules
import typing

import json
from pathlib import Path
import tempfile
import unittest

from src.config.models import (
    BatchGenerationConfig,
    ConversationalStyle,
    DatasetSampleConfig,
)
from src.pipeline.dataset_pipeline import (
    generate_dataset_batch,
    generate_single_dataset_sample,
)


class TestDatasetPipeline(unittest.TestCase):
    """
    Test suite for dataset generation pipeline functions and batch coordination.
    """

    def setUp(self) -> None:
        """
        Create isolated temporary workspace directory for test sample artifacts.
        """

        # Create temporary working directory for test fixtures
        tmp_dir: str = self.enterContext(
            tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        )
        self.test_dir: Path = Path(tmp_dir)
        self.voices_dir: Path = self.test_dir / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def test_generate_single_dataset_sample_mock_tts(self) -> None:
        """
        Verify single-sample generation writes mixed audio, stems, and metadata.
        """

        sample_dir: Path = self.test_dir / "sample_0001"
        sample_config: DatasetSampleConfig = DatasetSampleConfig(
            sample_id="sample_0001",
            duration_s=6.0,
            ambiance_preset="kitchen_cooking",
            num_constraint_words=3,
            use_llm=False,
        )

        result_dir: Path = generate_single_dataset_sample(
            sample_config=sample_config,
            num_speakers=2,
            voices_dir=self.voices_dir,
            output_sample_dir=sample_dir,
            use_mock_tts=True,
            export_isolated_stems=True,
        )

        self.assertEqual(result_dir, sample_dir)
        self.assertTrue((sample_dir / "mixed_scene.wav").exists())
        self.assertTrue((sample_dir / "annotations.json").exists())
        self.assertTrue((sample_dir / "diarization.rttm").exists())
        self.assertTrue((sample_dir / "transcripts.jsonl").exists())

        # Verify isolated speaker stems
        stems_dir: Path = sample_dir / "isolated_speakers"
        self.assertTrue(stems_dir.exists())
        wav_stems: list[Path] = list(stems_dir.glob("*.wav"))
        self.assertEqual(len(wav_stems), 2)

        # Verify metadata content
        with (sample_dir / "annotations.json").open("r", encoding="utf-8") as annot_file:
            payload: dict[str, typing.Any] = json.load(annot_file)

        self.assertIn("ambiance", payload)
        self.assertEqual(payload["ambiance"]["name"], "kitchen_cooking")
        self.assertIn("thematic_constraints", payload)
        self.assertEqual(len(payload["thematic_constraints"]), 3)

    def test_generate_dataset_batch(self) -> None:
        """
        Verify generate_dataset_batch produces requested samples and master manifest.
        """

        out_batch: Path = self.test_dir / "batch_out"
        batch_config: BatchGenerationConfig = BatchGenerationConfig(
            num_samples=2,
            duration_range=(5.0, 7.0),
            speakers_range=(2, 2),
            languages=["en"],
            style=ConversationalStyle.MIXED,
            voices_dir=self.voices_dir,
            output_dir=out_batch,
            use_mock_tts=True,
            export_isolated_stems=False,
            use_llm=False,
            ambiance_preset="casual_chit_chat",
            num_constraint_words=2,
        )

        samples: list[Path] = generate_dataset_batch(batch_config)
        self.assertEqual(len(samples), 2)

        master_manifest: Path = out_batch / "dataset_manifest.json"
        self.assertTrue(master_manifest.exists())

        with master_manifest.open("r", encoding="utf-8") as manifest_file:
            data: dict[str, typing.Any] = json.load(manifest_file)

        self.assertEqual(data["total_samples"], 2)
        self.assertEqual(len(data["samples"]), 2)
        self.assertEqual(data["samples"][0]["sample_id"], "sample_001")
        self.assertEqual(data["samples"][1]["sample_id"], "sample_002")

    def test_sequential_batch_generation_appends_and_increments(self) -> None:
        """
        Verify consecutive batch runs auto-increment sample IDs and merge manifest.
        """

        out_batch: Path = self.test_dir / "batch_sequential"
        batch_config: BatchGenerationConfig = BatchGenerationConfig(
            num_samples=2,
            duration_range=(5.0, 7.0),
            speakers_range=(2, 2),
            languages=["en"],
            voices_dir=self.voices_dir,
            output_dir=out_batch,
            use_mock_tts=True,
            export_isolated_stems=False,
            use_llm=False,
        )

        # First run: should produce sample_001 and sample_002
        first_run_samples: list[Path] = generate_dataset_batch(batch_config)
        self.assertEqual(len(first_run_samples), 2)
        self.assertEqual(first_run_samples[0].name, "sample_001")
        self.assertEqual(first_run_samples[1].name, "sample_002")

        # Second run with 1 sample: should detect existing and produce sample_003
        batch_config.num_samples = 1
        second_run_samples: list[Path] = generate_dataset_batch(batch_config)
        self.assertEqual(len(second_run_samples), 1)
        self.assertEqual(second_run_samples[0].name, "sample_003")

        # Verify all 3 sample folders exist
        self.assertTrue((out_batch / "sample_001").is_dir())
        self.assertTrue((out_batch / "sample_002").is_dir())
        self.assertTrue((out_batch / "sample_003").is_dir())

        # Verify merged manifest has all 3 samples
        manifest_path: Path = out_batch / "dataset_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            data: dict[str, typing.Any] = json.load(manifest_file)

        self.assertEqual(data["total_samples"], 3)
        sample_ids: list[str] = [s["sample_id"] for s in data["samples"]]
        self.assertEqual(sample_ids, ["sample_001", "sample_002", "sample_003"])

    def test_generate_sample_100_sentences_auto_duration(self) -> None:
        """
        Verify procedural sample generation with at least 100 sentences and auto duration.
        """

        sample_dir: Path = self.test_dir / "sample_100_turns"
        sample_config: DatasetSampleConfig = DatasetSampleConfig(
            sample_id="sample_100_turns",
            duration_s=None,
            min_sentences=100,
            ambiance_preset="office_meeting",
            use_llm=False,
        )

        result_dir: Path = generate_single_dataset_sample(
            sample_config=sample_config,
            num_speakers=6,
            voices_dir=self.voices_dir,
            output_sample_dir=sample_dir,
            use_mock_tts=True,
            export_isolated_stems=False,
        )

        self.assertEqual(result_dir, sample_dir)
        self.assertTrue((sample_dir / "mixed_scene.wav").exists())
        self.assertTrue((sample_dir / "annotations.json").exists())

        with (sample_dir / "annotations.json").open("r", encoding="utf-8") as f_annot:
            payload: dict[str, typing.Any] = json.load(f_annot)

        utterances: list[dict[str, typing.Any]] = payload.get("utterances", [])
        self.assertGreaterEqual(len(utterances), 100)
        self.assertGreater(payload["duration_s"], 0.0)

        speakers: set[str] = {u["speaker_id"] for u in utterances}
        self.assertEqual(len(speakers), 6)

    def test_generate_dataset_with_disable_effects(self) -> None:
        """
        Verify generating dataset with disable_effects skips vocal effects and ambient noise.
        """

        sample_dir: Path = self.test_dir / "sample_dry"
        sample_config: DatasetSampleConfig = DatasetSampleConfig(
            sample_id="sample_dry",
            duration_s=5.0,
            use_llm=False,
            disable_effects=True,
        )

        result_dir: Path = generate_single_dataset_sample(
            sample_config=sample_config,
            num_speakers=2,
            voices_dir=self.voices_dir,
            output_sample_dir=sample_dir,
            use_mock_tts=True,
            export_isolated_stems=False,
        )

        self.assertEqual(result_dir, sample_dir)
        self.assertTrue((sample_dir / "mixed_scene.wav").exists())
        self.assertTrue((sample_dir / "annotations.json").exists())

        with (sample_dir / "annotations.json").open("r", encoding="utf-8") as f_annot:
            payload: dict[str, typing.Any] = json.load(f_annot)

        self.assertTrue(payload.get("disable_effects"))


if __name__ == "__main__":
    unittest.main()
