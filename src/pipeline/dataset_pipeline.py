"""
Batch dataset generation pipeline producing diverse multi-speaker training samples.

Coordinates procedural room variations, multi-person conversation generation,
multilingual speech synthesis, vocal distortions, spatial acoustic simulation,
and ground-truth dataset annotations.
"""

# Import Modules
import typing
from pathlib import Path

import json
import random
import logging

from src.common.types import AudioArray
from src.personas.persona import Persona
from src.personas.manager import PersonaManager
from src.dataset.annotator import export_dataset_manifest
from src.audio.distortions import add_ambient_room_noise
from src.tts.synthesizer import BaseSynthesizer
from src.pipeline.orchestrator import instantiate_synthesizer
from src.procedural.variation_generator import generate_random_scene
from src.procedural.conversation_generator import generate_conversations_for_personas
from src.config.models import (
    SceneConfig,
    ConversationalStyle,
    DatasetSampleConfig,
)
from src.common.audio_utils import (
    write_wav_file,
    normalize_audio_peak,
)
from src.simulation.simulator import (
    run_acoustic_simulation,
    run_simulation_with_stems,
)

logger: logging.Logger = logging.getLogger(__name__)


def simulate_scene_audio(
    scene: SceneConfig,
    manager: PersonaManager,
    sample_dir: Path,
    synthesizer: BaseSynthesizer,
    export_isolated_stems: bool,
) -> tuple[AudioArray, list[dict[str, typing.Any]]]:
    """
    Synthesize persona tracks and simulate spatial room acoustics.

    Args:
        scene (SceneConfig): Configured room and personas.
        manager (PersonaManager): Persona manager instance.
        sample_dir (Path): Destination sample directory.
        synthesizer (BaseSynthesizer): TTS synthesizer.
        export_isolated_stems (bool): Whether to export isolated spatial tracks.

    Returns:
        tuple[AudioArray, list[dict[str, typing.Any]]]: Mixed spatial audio and utterance metadata.
    """

    # Synthesize multi-persona continuous audio tracks
    tracks = manager.synthesize_all_tracks(
        synthesizer=synthesizer,
        sample_rate=scene.room.sample_rate,
    )
    utterances_meta = manager.get_utterance_metadata()

    # Simulate room acoustics with isolated stems or composite mix
    if export_isolated_stems:
        mix_audio, stems = run_simulation_with_stems(
            room_config=scene.room,
            mic_config=scene.assistant,
            tracks=tracks,
        )

        # Export per-speaker ground-truth stems
        stems_dir: Path = sample_dir / "isolated_speakers"
        stems_dir.mkdir(parents=True, exist_ok=True)

        for speaker_id, stem_data in stems.items():
            stem_path: Path = stems_dir / f"{speaker_id}.wav"
            write_wav_file(stem_path, stem_data, scene.room.sample_rate)

        return (mix_audio, utterances_meta)

    mix_audio = run_acoustic_simulation(
        room_config=scene.room,
        mic_config=scene.assistant,
        tracks=tracks,
    )

    return (mix_audio, utterances_meta)


def generate_single_dataset_sample(
    sample_config: DatasetSampleConfig,
    num_speakers: int,
    voices_dir: Path,
    output_sample_dir: Path,
    use_mock_tts: bool = False,
    export_isolated_stems: bool = True,
) -> Path:
    """
    Generate a single procedural dataset sample variation.

    Args:
        sample_config (DatasetSampleConfig): Sample parameters.
        num_speakers (int): Number of speakers in scene.
        voices_dir (Path): Voices folder.
        output_sample_dir (Path): Destination folder.
        use_mock_tts (bool): Mock fallback mode.
        export_isolated_stems (bool): Export isolated stems for source separation.

    Returns:
        Path: Generated sample directory.
    """

    # Step 1: Generate randomized scene geometry and persona placement
    scene: SceneConfig = generate_random_scene(sample_config, num_speakers)

    # Step 2: Procedurally populate conversations with turn-taking and overlaps
    generate_conversations_for_personas(
        personas=scene.personas,
        duration_s=sample_config.duration_s,
        overlap_rate=sample_config.overlap_rate,
        shout_rate=sample_config.shout_rate,
        laugh_rate=sample_config.laugh_rate,
        style_preset=sample_config.style,
    )

    # Step 3: Initialize speech synthesizer
    synthesizer: BaseSynthesizer = instantiate_synthesizer(
        voices_dir=voices_dir,
        use_mock_tts=use_mock_tts,
    )

    # Step 4: Build PersonaManager and execute simulation
    manager: PersonaManager = PersonaManager(personas=[Persona(config=p) for p in scene.personas])
    mix_audio, utterances_meta = simulate_scene_audio(
        scene=scene,
        manager=manager,
        sample_dir=output_sample_dir,
        synthesizer=synthesizer,
        export_isolated_stems=export_isolated_stems,
    )

    # Step 5: Add background room ambiance and normalize
    mix_audio = add_ambient_room_noise(mix_audio, target_snr_db=sample_config.ambient_snr_db)
    final_mix: AudioArray = normalize_audio_peak(mix_audio, peak_target=0.92)

    # Step 6: Write composite mixed scene WAV
    mixed_wav_path: Path = output_sample_dir / "mixed_scene.wav"
    write_wav_file(mixed_wav_path, final_mix, scene.room.sample_rate)

    # Step 7: Export standard dataset annotations (RTTM, JSONL, Manifest)
    duration_s: float = final_mix.shape[0] / float(scene.room.sample_rate)
    export_dataset_manifest(
        sample_dir=output_sample_dir,
        session_id=sample_config.sample_id,
        scene_config=scene,
        utterances_meta=utterances_meta,
        audio_duration_s=duration_s,
    )

    return output_sample_dir


def generate_batch_sample_item(
    index: int,
    output_dir: Path,
    duration_range: tuple[float, float],
    speakers_range: tuple[int, int],
    languages: list[str],
    style: ConversationalStyle,
    voices_dir: Path,
    use_mock_tts: bool,
    export_isolated_stems: bool,
) -> tuple[Path, dict[str, typing.Any]]:
    """
    Generate an individual sample variation within a batch run.

    Args:
        index (int): Sequence index.
        output_dir (Path): Base output directory.
        duration_range (tuple[float, float]): Range of audio lengths.
        speakers_range (tuple[int, int]): Range of speaker counts.
        languages (list[str]): Allowed languages.
        style (ConversationalStyle): Style preset.
        voices_dir (Path): Voices directory.
        use_mock_tts (bool): Mock fallback mode.
        export_isolated_stems (bool): Isolated stems flag.

    Returns:
        tuple[Path, dict[str, typing.Any]]: Sample path and summary dictionary.
    """

    sample_id: str = f"sample_{index:04d}"
    sample_dir: Path = output_dir / sample_id
    duration_s: float = round(random.uniform(duration_range[0], duration_range[1]), 1)
    num_speakers: int = random.randint(speakers_range[0], speakers_range[1])

    sample_config: DatasetSampleConfig = DatasetSampleConfig(
        sample_id=sample_id,
        duration_s=duration_s,
        overlap_rate=round(random.uniform(0.2, 0.45), 2),
        shout_rate=round(random.uniform(0.1, 0.25), 2),
        laugh_rate=round(random.uniform(0.15, 0.3), 2),
        ambient_snr_db=round(random.uniform(20.0, 32.0), 1),
        style=style,
        languages=languages,
    )

    generate_single_dataset_sample(
        sample_config=sample_config,
        num_speakers=num_speakers,
        voices_dir=voices_dir,
        output_sample_dir=sample_dir,
        use_mock_tts=use_mock_tts,
        export_isolated_stems=export_isolated_stems,
    )

    record: dict[str, typing.Any] = {
        "sample_id": sample_id,
        "duration_s": duration_s,
        "num_speakers": num_speakers,
        "languages": languages,
    }

    return (sample_dir, record)


def write_batch_manifest(
    output_dir: Path,
    num_samples: int,
    summary_records: list[dict[str, typing.Any]],
) -> None:
    """
    Export master summary JSON for completed batch run.

    Args:
        output_dir (Path): Output directory.
        num_samples (int): Total samples count.
        summary_records (list[dict[str, typing.Any]]): Per-sample summaries.
    """

    summary_path: Path = output_dir / "dataset_manifest.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "total_samples": num_samples,
                "samples": summary_records,
            },
            summary_file,
            indent=2,
        )


def generate_dataset_batch(
    num_samples: int,
    duration_range: tuple[float, float],
    speakers_range: tuple[int, int],
    languages: list[str],
    style: ConversationalStyle,
    voices_dir: Path,
    output_dir: Path,
    use_mock_tts: bool = False,
    export_isolated_stems: bool = True,
) -> list[Path]:
    """
    Generate a diverse batch of procedural dataset samples.

    Args:
        num_samples (int): Number of distinct sample variations to generate.
        duration_range (tuple[float, float]): (min_duration, max_duration) in seconds.
        speakers_range (tuple[int, int]): (min_speakers, max_speakers).
        languages (list[str]): Allowed language codes.
        style (ConversationalStyle): Conversational style preset.
        voices_dir (Path): Folder containing Piper ONNX voice models.
        output_dir (Path): Destination root folder.
        use_mock_tts (bool): Whether to use mock synthesizer for fast testing.
        export_isolated_stems (bool): Whether to export isolated ground-truth stems.

    Returns:
        list[Path]: List of generated sample folder paths.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_samples: list[Path] = []
    summary_records: list[dict[str, typing.Any]] = []

    for i in range(1, num_samples + 1):
        logger.info("Generating dataset sample %d of %d...", i, num_samples)
        sample_dir, record = generate_batch_sample_item(
            index=i,
            output_dir=output_dir,
            duration_range=duration_range,
            speakers_range=speakers_range,
            languages=languages,
            style=style,
            voices_dir=voices_dir,
            use_mock_tts=use_mock_tts,
            export_isolated_stems=export_isolated_stems,
        )

        generated_samples.append(sample_dir)
        summary_records.append(record)

    write_batch_manifest(output_dir, num_samples, summary_records)

    return generated_samples
