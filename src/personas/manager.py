"""
Persona management and multi-track audio timeline sequencer.

Coordinates text-to-speech synthesis across multiple personas, applies vocal
distortions (shouting overdrive, laughter modulation), and assembles individual
dialogue turns into temporally aligned continuous audio streams.
"""

# Import Modules
from dataclasses import dataclass
import logging
import random
import typing

import numpy as np

from src.audio.distortions import (
    apply_laughter_modulation,
    apply_soft_clipping_overdrive,
)
from src.common.audio_utils import place_audio_on_timeline
from src.common.types import AudioArray
from src.config.models import UtteranceConfig, VocalStyle
from src.dataset.annotator import build_utterance_metadata
from src.personas.persona import Persona
from src.tts.synthesizer import BaseSynthesizer

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


def _synthesize_and_process_utterance(
    persona: Persona,
    utt: UtteranceConfig,
    synthesizer: BaseSynthesizer,
    sample_rate: int,
) -> tuple[AudioArray, float]:
    """
    Synthesize speech audio for an utterance and apply emotional vocal effects.

    Args:
        persona (Persona): Speaker persona emitting the utterance.
        utt (UtteranceConfig): Utterance configuration.
        synthesizer (BaseSynthesizer): Speech synthesis engine.
        sample_rate (int): Target sampling frequency in Hertz.

    Returns:
        tuple[AudioArray, float]: Processed audio clip and exact duration in seconds.
    """

    logger.info(
        "Synthesizing for '%s' (%s, style=%s): '%s'",
        persona.display_name,
        utt.language,
        utt.vocal_style.value,
        utt.text,
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

    return (processed_clip, duration_s)


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

    utt: UtteranceConfig = persona.config.utterances[utt_index]
    processed_clip, duration_s = _synthesize_and_process_utterance(
        persona=persona,
        utt=utt,
        synthesizer=synthesizer,
        sample_rate=sample_rate,
    )
    end_time_s: float = utt.start_time_s + duration_s

    # Build ground truth metadata record
    metadata: dict[str, typing.Any] = build_utterance_metadata(
        utterance=utt,
        persona=persona.config,
        duration_s=duration_s,
    )

    return (processed_clip, end_time_s, metadata)


def _calculate_next_turn_start(
    prev_end_s: float,
    is_interruption: bool,
) -> float:
    """
    Calculate start timestamp for the next turn in a conversational group.

    Args:
        prev_end_s (float): End timestamp of preceding utterance in seconds.
        is_interruption (bool): Whether the upcoming turn interrupts the previous speaker.

    Returns:
        float: Calculated start timestamp in seconds.
    """

    # Overlap previous speaker slightly if interrupting
    if is_interruption:
        return max(0.0, prev_end_s - random.uniform(0.2, 0.6))

    # Conversational turn-taking pause between normal utterances
    return prev_end_s + random.uniform(0.15, 0.60)


def _resolve_speaker_temporal_collision(
    proposed_start_s: float,
    duration_s: float,
    busy_intervals: list[tuple[float, float]],
) -> float:
    """
    Ensure a speaker never overlaps with their own speech in concurrent discussions.

    Args:
        proposed_start_s (float): Desired utterance start time in seconds.
        duration_s (float): Spoken audio duration in seconds.
        busy_intervals (list[tuple[float, float]]): Existing speech intervals for this speaker.

    Returns:
        float: Collision-free start timestamp in seconds.
    """

    start_s: float = proposed_start_s
    for busy_start, busy_end in busy_intervals:
        if not (start_s + duration_s <= busy_start or start_s >= busy_end):
            start_s = max(start_s, busy_end + random.uniform(0.15, 0.45))

    return round(start_s, 2)


def _sequence_group_utterances(
    group_items: list[tuple[Persona, UtteranceConfig]],
    synthesizer: BaseSynthesizer,
    sample_rate: int,
    speaker_busy: dict[str, list[tuple[float, float]]],
) -> tuple[list[tuple[Persona, AudioArray, float]], list[float], list[dict[str, typing.Any]]]:
    """
    Synthesize and dynamically position utterances for a single conversation group.

    Args:
        group_items (list[tuple[Persona, UtteranceConfig]]): Group personas and utterances.
        synthesizer (BaseSynthesizer): Speech synthesizer engine.
        sample_rate (int): Output sampling rate in Hertz.
        speaker_busy (dict[str, list[tuple[float, float]]]): Busy intervals per speaker.

    Returns:
        tuple[list[tuple[Persona, AudioArray, float]], list[float], list[dict[str, typing.Any]]]:
            Clips, end timestamps, and ground-truth metadata records.
    """

    group_items.sort(key=lambda item: (item[1].turn_order, item[1].start_time_s))
    has_order: bool = any(utt.turn_order > 0 for _, utt in group_items)

    clips: list[tuple[Persona, AudioArray, float]] = []
    end_times: list[float] = []
    records: list[dict[str, typing.Any]] = []
    prev_end_s: float = random.uniform(0.3, 0.8)

    for turn_idx, (persona, utt) in enumerate(group_items):
        if turn_idx > 0 and has_order:
            utt.start_time_s = round(
                _calculate_next_turn_start(
                    prev_end_s,
                    utt.vocal_style == VocalStyle.INTERRUPTION,
                ),
                2,
            )
        elif turn_idx == 0 and has_order and utt.start_time_s <= 0.0:
            utt.start_time_s = round(prev_end_s, 2)

        clip, dur_s = _synthesize_and_process_utterance(
            persona=persona,
            utt=utt,
            synthesizer=synthesizer,
            sample_rate=sample_rate,
        )

        # Enforce zero self-overlap across concurrent discussions
        utt.start_time_s = _resolve_speaker_temporal_collision(
            proposed_start_s=utt.start_time_s,
            duration_s=dur_s,
            busy_intervals=speaker_busy.setdefault(persona.identifier, []),
        )

        prev_end_s = utt.start_time_s + dur_s
        speaker_busy.setdefault(persona.identifier, []).append((utt.start_time_s, prev_end_s))

        clips.append((persona, clip, utt.start_time_s))
        end_times.append(prev_end_s)
        records.append(
            build_utterance_metadata(
                utterance=utt,
                persona=persona.config,
                duration_s=dur_s,
            )
        )

    return (clips, end_times, records)


def synthesize_persona_clips(
    personas: list[Persona],
    synthesizer: BaseSynthesizer,
    sample_rate: int,
) -> tuple[list[tuple[Persona, AudioArray, float]], list[float], list[dict[str, typing.Any]]]:
    """
    Synthesize all speech clips across all personas with dynamic synthesis-driven timing.

    Args:
        personas (list[Persona]): Personas with utterances to synthesize.
        synthesizer (BaseSynthesizer): TTS synthesis engine.
        sample_rate (int): Output sampling rate in Hertz.

    Returns:
        tuple[list[tuple[Persona, AudioArray, float]], list[float], list[dict[str, typing.Any]]]:
            Clips, end timestamps, and ground-truth metadata records.
    """

    # Group utterances by conversation clique identifier
    group_map: dict[int, list[tuple[Persona, UtteranceConfig]]] = {}
    for persona in personas:
        for utt in persona.config.utterances:
            group_map.setdefault(utt.group_id, []).append((persona, utt))

    speech_clips: list[tuple[Persona, AudioArray, float]] = []
    end_times: list[float] = []
    metadata_records: list[dict[str, typing.Any]] = []
    speaker_busy: dict[str, list[tuple[float, float]]] = {}

    for _, group_items in sorted(group_map.items()):
        grp_clips, grp_ends, grp_records = _sequence_group_utterances(
            group_items=group_items,
            synthesizer=synthesizer,
            sample_rate=sample_rate,
            speaker_busy=speaker_busy,
        )
        speech_clips.extend(grp_clips)
        end_times.extend(grp_ends)
        metadata_records.extend(grp_records)

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
