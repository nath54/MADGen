"""
Batch dataset generation pipeline producing diverse multi-speaker training samples.

Coordinates procedural room variations, multi-person conversation generation,
multilingual speech synthesis, vocal distortions, spatial acoustic simulation,
and ground-truth dataset annotations.
"""

# Import Modules
import typing

import json
import logging
from pathlib import Path
import random

from src.audio.distortions import add_ambient_room_noise
from src.common.audio_utils import (
    normalize_audio_peak,
    write_wav_file,
)
from src.common.types import AudioArray
from src.config.models import (
    BatchGenerationConfig,
    DatasetSampleConfig,
    SceneConfig,
)
from src.dataset.annotator import export_dataset_manifest
from src.llm.dialogue_generator import LLMDialogueGenerator
from src.personas.manager import PersonaManager
from src.personas.persona import Persona
from src.pipeline.orchestrator import instantiate_synthesizer
from src.procedural.ambiance_presets import AmbiancePreset, get_ambiance_preset
from src.procedural.conversation_generator import generate_conversations_for_personas
from src.procedural.variation_generator import generate_random_scene
from src.procedural.word_dictionary import sample_constraint_keywords
from src.simulation.simulator import (
    run_acoustic_simulation,
    run_simulation_with_stems,
)
from src.tts.synthesizer import BaseSynthesizer

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


def _prepare_scene_conversations(
    sample_config: DatasetSampleConfig,
    num_speakers: int,
) -> tuple[SceneConfig, AmbiancePreset, list[str]]:
    """
    Generate scene geometry and populate persona conversations with ambiance and constraints.

    Args:
        sample_config (DatasetSampleConfig): Sample parameters.
        num_speakers (int): Number of personas.

    Returns:
        tuple[SceneConfig, AmbiancePreset, list[str]]: Prepared scene, ambiance, and keywords.
    """

    scene: SceneConfig = generate_random_scene(sample_config, num_speakers)
    ambiance: AmbiancePreset = get_ambiance_preset(sample_config.ambiance_preset)
    num_keywords: int = (
        sample_config.num_constraint_words
        if sample_config.num_constraint_words > 0
        else ambiance.num_constraint_words
    )
    constraint_words: list[str] = sample_constraint_keywords(count=num_keywords)

    llm_generator: LLMDialogueGenerator | None = None
    if sample_config.use_llm:
        llm_generator = LLMDialogueGenerator(base_url=sample_config.llm_url)

    generate_conversations_for_personas(
        personas=scene.personas,
        duration_s=sample_config.duration_s,
        min_sentences=sample_config.min_sentences,
        overlap_rate=ambiance.overlap_rate,
        shout_rate=ambiance.shout_rate,
        laugh_rate=ambiance.laugh_rate,
        style_preset=sample_config.style,
        llm_generator=llm_generator,
        ambiance=ambiance,
        constraint_words=constraint_words,
        allow_parallel=sample_config.allow_parallel,
    )
    return (scene, ambiance, constraint_words)


def _export_sample_artifacts(
    scene: SceneConfig,
    ambiance: AmbiancePreset,
    constraint_words: list[str],
    sample_dir: Path,
    sample_id: str,
    mix_audio: AudioArray,
    utterances_meta: list[dict[str, typing.Any]],
) -> None:
    """
    Normalize, write mixed audio WAV, and export annotations for a sample.

    Args:
        scene (SceneConfig): Scene configuration.
        ambiance (AmbiancePreset): Ambiance preset used.
        constraint_words (list[str]): Constraint keywords.
        sample_dir (Path): Sample output directory.
        sample_id (str): Unique sample session identifier.
        mix_audio (AudioArray): Synthesized composite audio before ambiance noise.
        utterances_meta (list[dict[str, typing.Any]]): Utterance metadata list.
    """

    final_mix: AudioArray = normalize_audio_peak(
        add_ambient_room_noise(mix_audio, target_snr_db=ambiance.ambient_snr_db),
        peak_target=0.92,
    )
    write_wav_file(sample_dir / "mixed_scene.wav", final_mix, scene.room.sample_rate)

    duration_s: float = final_mix.shape[0] / float(scene.room.sample_rate)
    extra_metadata: dict[str, typing.Any] = {
        "ambiance": {
            "name": ambiance.name,
            "display_name": ambiance.display_name,
            "description": ambiance.description,
        },
        "thematic_constraints": constraint_words,
        "total_sentences": len(utterances_meta),
    }
    export_dataset_manifest(
        sample_dir=sample_dir,
        session_id=sample_id,
        scene_config=scene,
        utterances_meta=utterances_meta,
        audio_duration_s=duration_s,
        extra_metadata=extra_metadata,
    )


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

    scene, ambiance, constraint_words = _prepare_scene_conversations(
        sample_config=sample_config,
        num_speakers=num_speakers,
    )

    synthesizer: BaseSynthesizer = instantiate_synthesizer(
        voices_dir=voices_dir,
        use_mock_tts=use_mock_tts,
    )
    manager: PersonaManager = PersonaManager(personas=[Persona(config=p) for p in scene.personas])
    mix_audio, utterances_meta = simulate_scene_audio(
        scene=scene,
        manager=manager,
        sample_dir=output_sample_dir,
        synthesizer=synthesizer,
        export_isolated_stems=export_isolated_stems,
    )

    _export_sample_artifacts(
        scene=scene,
        ambiance=ambiance,
        constraint_words=constraint_words,
        sample_dir=output_sample_dir,
        sample_id=sample_config.sample_id,
        mix_audio=mix_audio,
        utterances_meta=utterances_meta,
    )

    return output_sample_dir


def _build_sample_config_for_batch(
    index: int,
    batch_config: BatchGenerationConfig,
) -> tuple[DatasetSampleConfig, int]:
    """
    Construct a randomized DatasetSampleConfig and speaker count for a batch item.

    Args:
        index (int): Sequence item index.
        batch_config (BatchGenerationConfig): Master batch configuration.

    Returns:
        tuple[DatasetSampleConfig, int]: Initialized sample config and speaker count.
    """

    sample_id: str = f"sample_{index:04d}"
    duration_s: float | None = None
    if batch_config.duration_range is not None:
        duration_s = round(
            random.uniform(batch_config.duration_range[0], batch_config.duration_range[1]),
            1,
        )

    num_speakers: int = random.randint(
        batch_config.speakers_range[0],
        batch_config.speakers_range[1],
    )

    allow_parallel: bool = random.random() < batch_config.parallel_prob

    sample_config: DatasetSampleConfig = DatasetSampleConfig(
        sample_id=sample_id,
        duration_s=duration_s,
        min_sentences=batch_config.min_sentences,
        overlap_rate=round(random.uniform(0.2, 0.45), 2),
        shout_rate=round(random.uniform(0.1, 0.25), 2),
        laugh_rate=round(random.uniform(0.15, 0.3), 2),
        ambient_snr_db=round(random.uniform(20.0, 32.0), 1),
        style=batch_config.style,
        languages=batch_config.languages,
        use_llm=batch_config.use_llm,
        llm_url=batch_config.llm_url,
        ambiance_preset=batch_config.ambiance_preset,
        num_constraint_words=batch_config.num_constraint_words,
        llm_temperature=batch_config.llm_temperature,
        allow_parallel=allow_parallel,
    )
    return (sample_config, num_speakers)


def generate_batch_sample_item(
    index: int,
    batch_config: BatchGenerationConfig,
) -> tuple[Path, dict[str, typing.Any]]:
    """
    Generate an individual sample variation within a batch run.

    Args:
        index (int): Sequence index.
        batch_config (BatchGenerationConfig): Master batch configuration.

    Returns:
        tuple[Path, dict[str, typing.Any]]: Sample path and summary dictionary.
    """

    sample_config, num_speakers = _build_sample_config_for_batch(index, batch_config)
    sample_dir: Path = batch_config.output_dir / sample_config.sample_id

    generate_single_dataset_sample(
        sample_config=sample_config,
        num_speakers=num_speakers,
        voices_dir=batch_config.voices_dir,
        output_sample_dir=sample_dir,
        use_mock_tts=batch_config.use_mock_tts,
        export_isolated_stems=batch_config.export_isolated_stems,
    )

    actual_duration: float = sample_config.duration_s or 0.0
    total_sentences: int = sample_config.min_sentences
    manifest_file: Path = sample_dir / "annotations.json"

    # Read rendered duration and utterance count from manifest if available
    if manifest_file.exists():
        with manifest_file.open("r", encoding="utf-8") as f_meta:
            meta_data: dict[str, typing.Any] = json.load(f_meta)
            actual_duration = float(meta_data.get("duration_s", actual_duration))
            total_sentences = int(
                meta_data.get(
                    "total_sentences",
                    len(meta_data.get("utterances", [])),
                )
            )

    record: dict[str, typing.Any] = {
        "sample_id": sample_config.sample_id,
        "duration_s": actual_duration,
        "total_sentences": total_sentences,
        "num_speakers": num_speakers,
        "languages": batch_config.languages,
        "ambiance": sample_config.ambiance_preset,
        "use_llm": batch_config.use_llm,
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
    batch_config: BatchGenerationConfig,
) -> list[Path]:
    """
    Generate a diverse batch of procedural dataset samples.

    Args:
        batch_config (BatchGenerationConfig): Master batch parameters.

    Returns:
        list[Path]: List of generated sample folder paths.
    """

    batch_config.output_dir.mkdir(parents=True, exist_ok=True)
    generated_samples: list[Path] = []
    summary_records: list[dict[str, typing.Any]] = []

    for i in range(1, batch_config.num_samples + 1):
        logger.info("Generating dataset sample %d of %d...", i, batch_config.num_samples)
        sample_dir, record = generate_batch_sample_item(
            index=i,
            batch_config=batch_config,
        )
        generated_samples.append(sample_dir)
        summary_records.append(record)

    write_batch_manifest(batch_config.output_dir, batch_config.num_samples, summary_records)

    return generated_samples
