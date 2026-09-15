"""
Persona management and multi-track audio timeline sequencer.

Coordinates text-to-speech synthesis across multiple personas and assembles
individual dialogue turns into temporally aligned continuous audio streams.
"""

# Import Modules
from dataclasses import dataclass

import logging

import numpy as np

from src.common.types import AudioArray
from src.personas.persona import Persona
from src.tts.synthesizer import BaseSynthesizer
from src.common.audio_utils import place_audio_on_timeline

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


def synthesize_persona_clips(
    personas: list[Persona],
    synthesizer: BaseSynthesizer,
    sample_rate: int,
) -> tuple[list[tuple[Persona, AudioArray, float]], list[float]]:
    """
    Synthesize all speech clips across all personas.

    Args:
        personas (list[Persona]): Personas with utterances to synthesize.
        synthesizer (BaseSynthesizer): TTS synthesis engine.
        sample_rate (int): Output sampling rate in Hertz.

    Returns:
        tuple[list[tuple[Persona, AudioArray, float]], list[float]]: Synthesized clips and ends.
    """

    # Collect rendered audio segments and their termination timestamps
    speech_clips: list[tuple[Persona, AudioArray, float]] = []
    end_times: list[float] = []

    for persona in personas:
        for utt in persona.config.utterances:
            logger.info(
                "Synthesizing for '%s': '%s' at %.2fs",
                persona.display_name,
                utt.text,
                utt.start_time_s,
            )

            clip: AudioArray = synthesizer.synthesize(
                text=utt.text,
                voice_model=persona.voice_model,
                speaker_id=persona.speaker_id,
                personality=persona.config.personality,
                target_sample_rate=sample_rate,
            )

            duration_s: float = len(clip) / float(sample_rate)
            end_times.append(utt.start_time_s + duration_s)
            speech_clips.append((persona, clip, utt.start_time_s))

    return (speech_clips, end_times)


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

        # Register personas by their unique identifiers
        self.personas: list[Persona] = personas

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
        speech_clips, end_times = synthesize_persona_clips(
            self.personas,
            synthesizer,
            sample_rate,
        )

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
