"""
Persona management and multi-track audio timeline sequencer.

Coordinates text-to-speech synthesis across multiple personas, applies vocal
distortions (shouting overdrive, laughter modulation), and assembles individual
dialogue turns into temporally aligned continuous audio streams.
"""

# Import Modules
import typing
from dataclasses import dataclass

import logging

import numpy as np

from src.common.types import AudioArray
from src.config.models import VocalStyle
from src.personas.persona import Persona
from src.dataset.annotator import build_utterance_metadata
from src.tts.synthesizer import BaseSynthesizer
from src.common.audio_utils import place_audio_on_timeline
from src.audio.distortions import (
    apply_laughter_modulation,
    apply_soft_clipping_overdrive,
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class PersonaTrack:
    """
    Continuous audio waveform and metadata for a single persona in the scene.
    """

    persona: Persona
    audio: AudioArray
    total_samples: int


def calculate_required_samples(
    utterances_end_times: list[float],
    sample_rate: int,
    padding_s: float = 1.0,
) -> int:
    """
    Calculate the total sample count needed to encompass all utterances plus padding.

    Args:
        utterances_end_times (list[float]): List of end timestamps in seconds.
        sample_rate (int): Sampling frequency in Hertz.
        padding_s (float): Trailing safety silence in seconds.

    Returns:
        int: Total number of samples required for the scene timeline.
    """

    # Handle empty utterance case
    if not utterances_end_times:
        return int(padding_s * sample_rate)

    # Find latest speech termination timestamp
    max_end_time: float = max(utterances_end_times)
    total_time_s: float = max_end_time + padding_s

    return int(total_time_s * sample_rate)


def apply_vocal_style_effects(
    audio_clip: AudioArray,
    vocal_style: VocalStyle,
    sample_rate: int,
) -> AudioArray:
    """
    Apply vocal effects like soft-clipping saturation or laughter tremolo.

    Args:
        audio_clip (AudioArray): Dry synthesized audio clip.
        vocal_style (VocalStyle): Emotional vocal style.
        sample_rate (int): Sampling frequency in Hertz.

    Returns:
        AudioArray: Processed audio clip.
    """

    # Apply soft-clipping overdrive when shouting
    if vocal_style == VocalStyle.SHOUTING:
        return apply_soft_clipping_overdrive(audio_clip, drive_gain=1.6)

    # Apply laughter tremolo modulation when laughing
    if vocal_style == VocalStyle.LAUGHTER:
        return apply_laughter_modulation(audio_clip, sample_rate=sample_rate)

    return audio_clip


def synthesize_single_utterance(
    persona: Persona,
    utt_index: int,
    synthesizer: BaseSynthesizer,
    sample_rate: int,
) -> tuple[AudioArray, float, dict[str, typing.Any]]:
    """
    Synthesize and post-process a single utterance for a persona.

    Args:
        persona (Persona): Speaker.
        utt_index (int): Index of utterance in persona config.
        synthesizer (BaseSynthesizer): TTS engine.
        sample_rate (int): Sample rate.

    Returns:
        tuple[AudioArray, float, dict[str, typing.Any]]: Processed audio, end time, and metadata.
    """

    utt = persona.config.utterances[utt_index]

    logger.info(
        "Synthesizing for '%s' (%s, style=%s): '%s' at %.2fs",
        persona.display_name,
        utt.language,
        utt.vocal_style.value,
        utt.text,
        utt.start_time_s,
    )

    # Synthesize dry speech audio
    raw_clip: AudioArray = synthesizer.synthesize(
        text=utt.text,
        voice_model=persona.voice_model,
        speaker_id=persona.speaker_id,
        personality=persona.config.personality,
        target_sample_rate=sample_rate,
    )

    # Apply emotional vocal distortions
    processed_clip: AudioArray = apply_vocal_style_effects(
        audio_clip=raw_clip,
        vocal_style=utt.vocal_style,
        sample_rate=sample_rate,
    )

    duration_s: float = len(processed_clip) / float(sample_rate)
    end_time_s: float = utt.start_time_s + duration_s

    # Build ground truth metadata record
    metadata: dict[str, typing.Any] = build_utterance_metadata(
        utterance=utt,
        persona=persona.config,
        duration_s=duration_s,
    )

    return (processed_clip, end_time_s, metadata)


def synthesize_persona_clips(
    personas: list[Persona],
    synthesizer: BaseSynthesizer,
    sample_rate: int,
) -> tuple[list[tuple[Persona, AudioArray, float]], list[float], list[dict[str, typing.Any]]]:
    """
    Synthesize all speech clips across all personas.

    Args:
        personas (list[Persona]): Personas with utterances to synthesize.
        synthesizer (BaseSynthesizer): TTS synthesis engine.
        sample_rate (int): Output sampling rate in Hertz.

    Returns:
        tuple[list[tuple[Persona, AudioArray, float]], list[float], list[dict[str, typing.Any]]]:
            Clips, end timestamps, and ground-truth metadata.
    """

    # Collect rendered audio segments, termination timestamps, and metadata
    speech_clips: list[tuple[Persona, AudioArray, float]] = []
    end_times: list[float] = []
    metadata_records: list[dict[str, typing.Any]] = []

    for persona in personas:
        for utt_idx in range(len(persona.config.utterances)):
            clip, end_t, meta = synthesize_single_utterance(
                persona=persona,
                utt_index=utt_idx,
                synthesizer=synthesizer,
                sample_rate=sample_rate,
            )
            speech_clips.append((persona, clip, float(meta["start_time_s"])))
            end_times.append(end_t)
            metadata_records.append(meta)

    return (speech_clips, end_times, metadata_records)


def assemble_persona_tracks(
    personas: list[Persona],
    speech_clips: list[tuple[Persona, AudioArray, float]],
    total_samples: int,
    sample_rate: int,
) -> list[PersonaTrack]:
    """
    Assemble speech clips onto per-persona continuous timeline buffers.

    Args:
        personas (list[Persona]): List of active personas.
        speech_clips (list[tuple[Persona, AudioArray, float]]): Rendered speech clips.
        total_samples (int): Total buffer length in samples.
        sample_rate (int): Sampling frequency in Hertz.

    Returns:
        list[PersonaTrack]: Continuous tracks for each persona.
    """

    # Initialize zero buffers for each persona
    persona_buffers: dict[str, AudioArray] = {
        p.identifier: np.zeros(total_samples, dtype=np.float64)
        for p in personas
    }

    # Overlay each speech segment at its designated start timestamp
    for persona, clip, start_time_s in speech_clips:
        start_sample: int = int(start_time_s * sample_rate)
        buf: AudioArray = persona_buffers[persona.identifier]
        place_audio_on_timeline(buf, clip, start_sample)

    # Package into PersonaTrack structures
    tracks: list[PersonaTrack] = [
        PersonaTrack(
            persona=p,
            audio=persona_buffers[p.identifier],
            total_samples=total_samples,
        )
        for p in personas
    ]

    return tracks


class PersonaManager:
    """
    Manages persona lifecycles and speech rendering for the simulation.
    """

    def __init__(self, personas: list[Persona]) -> None:
        """
        Initialize the persona manager with a list of active personas.

        Args:
            personas (list[Persona]): Personas participating in the scene.
        """

        # Register personas by their unique identifiers and prepare metadata store
        self.personas: list[Persona] = personas
        self.utterances_metadata: list[dict[str, typing.Any]] = []

    def get_utterance_metadata(self) -> list[dict[str, typing.Any]]:
        """
        Retrieve ground-truth metadata records for all synthesized utterances.

        Returns:
            list[dict[str, typing.Any]]: Chronological utterance metadata records.
        """

        return self.utterances_metadata

    def synthesize_all_tracks(
        self,
        synthesizer: BaseSynthesizer,
        sample_rate: int,
        padding_s: float = 1.5,
    ) -> list[PersonaTrack]:
        """
        Synthesize speech for all personas and position them onto aligned tracks.

        Args:
            synthesizer (BaseSynthesizer): TTS synthesis engine.
            sample_rate (int): Audio sampling frequency in Hertz.
            padding_s (float): Trailing room reverberation padding in seconds.

        Returns:
            list[PersonaTrack]: Rendered audio tracks for each persona.
        """

        # Step 1: Synthesize all individual utterances and collect timestamps
        speech_clips, end_times, meta_records = synthesize_persona_clips(
            self.personas,
            synthesizer,
            sample_rate,
        )
        self.utterances_metadata = meta_records

        # Step 2: Determine global scene timeline length
        total_samples: int = calculate_required_samples(
            utterances_end_times=end_times,
            sample_rate=sample_rate,
            padding_s=padding_s,
        )

        # Step 3: Populate continuous timeline buffers for each persona
        tracks: list[PersonaTrack] = assemble_persona_tracks(
            self.personas,
            speech_clips,
            total_samples,
            sample_rate,
        )

        return tracks
