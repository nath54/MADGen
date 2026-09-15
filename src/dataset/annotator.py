"""
Ground-truth dataset annotation exporter for speaker diarization, isolation, and ASR.

Produces standard RTTM diarization files, JSONL multi-speaker transcriptions,
and comprehensive JSON scene manifests.
"""

# Import Modules
import typing
from pathlib import Path

import json

from src.config.models import (
    SceneConfig,
    PersonaConfig,
    UtteranceConfig,
)


def format_rttm_line(
    session_id: str,
    speaker_id: str,
    start_time_s: float,
    duration_s: float,
) -> str:
    """
    Format a single standard NIST RTTM speaker diarization record.

    Args:
        session_id (str): Audio recording session identifier.
        speaker_id (str): Unique speaker label.
        start_time_s (float): Start timestamp in seconds.
        duration_s (float): Segment duration in seconds.

    Returns:
        str: Formatted RTTM line.
    """

    # Standard RTTM format: SPEAKER <file> 1 <tbeg> <tdur> <NA> <NA> <spkr> <NA> <NA>
    return (
        f"SPEAKER {session_id} 1 {start_time_s:.2f} {duration_s:.2f} "
        f"<NA> <NA> {speaker_id} <NA> <NA>\n"
    )


def build_utterance_metadata(
    utterance: UtteranceConfig,
    persona: PersonaConfig,
    duration_s: float,
) -> dict[str, typing.Any]:
    """
    Construct a metadata dictionary for a single spoken utterance.

    Args:
        utterance (UtteranceConfig): Utterance specifications.
        persona (PersonaConfig): Speaker emitting the utterance.
        duration_s (float): Segment duration in seconds.

    Returns:
        dict[str, typing.Any]: Utterance metadata record.
    """

    # Assemble comprehensive record
    record: dict[str, typing.Any] = {
        "speaker_id": persona.id,
        "speaker_name": persona.name,
        "gender": persona.gender,
        "language": utterance.language,
        "vocal_style": utterance.vocal_style.value,
        "start_time_s": utterance.start_time_s,
        "end_time_s": round(utterance.start_time_s + duration_s, 2),
        "duration_s": round(duration_s, 2),
        "text": utterance.text,
        "speaker_position": list(persona.position),
        "speaker_size": persona.size,
    }

    return record


def export_diarization_rttm(
    output_path: Path,
    session_id: str,
    utterances_meta: list[dict[str, typing.Any]],
) -> None:
    """
    Export standard NIST RTTM file for speaker diarization benchmarks.

    Args:
        output_path (Path): Path to output .rttm file.
        session_id (str): Recording session name.
        utterances_meta (list[dict[str, typing.Any]]): Utterance metadata list.
    """

    # Sort records chronologically
    sorted_records: list[dict[str, typing.Any]] = sorted(
        utterances_meta,
        key=lambda r: float(r["start_time_s"]),
    )

    # Write RTTM lines
    with output_path.open("w", encoding="utf-8") as file_stream:
        for record in sorted_records:
            line: str = format_rttm_line(
                session_id=session_id,
                speaker_id=str(record["speaker_id"]),
                start_time_s=float(record["start_time_s"]),
                duration_s=float(record["duration_s"]),
            )
            file_stream.write(line)


def export_asr_transcripts_jsonl(
    output_path: Path,
    utterances_meta: list[dict[str, typing.Any]],
) -> None:
    """
    Export chronological JSONL transcript lines for Speech-to-Text training.

    Args:
        output_path (Path): Output .jsonl file path.
        utterances_meta (list[dict[str, typing.Any]]): Utterance records.
    """

    # Sort chronologically by start timestamp
    sorted_records: list[dict[str, typing.Any]] = sorted(
        utterances_meta,
        key=lambda r: float(r["start_time_s"]),
    )

    # Write each record as a JSON line
    with output_path.open("w", encoding="utf-8") as file_stream:
        for record in sorted_records:
            line_str: str = json.dumps(record, ensure_ascii=False)
            file_stream.write(line_str + "\n")


def export_dataset_manifest(
    sample_dir: Path,
    session_id: str,
    scene_config: SceneConfig,
    utterances_meta: list[dict[str, typing.Any]],
    audio_duration_s: float,
) -> None:
    """
    Write full dataset metadata manifest and standard annotations to disk.

    Args:
        sample_dir (Path): Sample destination folder.
        session_id (str): Recording identifier.
        scene_config (SceneConfig): Simulated scene parameters.
        utterances_meta (list[dict[str, typing.Any]]): Utterances list.
        audio_duration_s (float): Total audio length in seconds.
    """

    # Ensure output directory exists
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Export RTTM diarization file
    rttm_path: Path = sample_dir / "diarization.rttm"
    export_diarization_rttm(rttm_path, session_id, utterances_meta)

    # Export JSONL transcripts
    jsonl_path: Path = sample_dir / "transcripts.jsonl"
    export_asr_transcripts_jsonl(jsonl_path, utterances_meta)

    # Compile comprehensive JSON manifest
    manifest: dict[str, typing.Any] = {
        "session_id": session_id,
        "duration_s": round(audio_duration_s, 2),
        "sample_rate": scene_config.room.sample_rate,
        "num_speakers": len(scene_config.personas),
        "room": {
            "dimensions": list(scene_config.room.dimensions),
            "absorption": scene_config.room.absorption,
            "max_order": scene_config.room.max_order,
        },
        "assistant": {
            "position": list(scene_config.assistant.position),
            "mic_type": scene_config.assistant.mic_type.value,
        },
        "speakers": [
            {
                "id": p.id,
                "name": p.name,
                "gender": p.gender,
                "language": p.language,
                "position": list(p.position),
                "size": p.size,
                "voice_model": p.voice_model,
                "speaker_id": p.speaker_id,
            }
            for p in scene_config.personas
        ],
        "utterances": utterances_meta,
    }

    # Save manifest JSON
    manifest_path: Path = sample_dir / "annotations.json"
    with manifest_path.open("w", encoding="utf-8") as json_file:
        json.dump(manifest, json_file, indent=2, ensure_ascii=False)
