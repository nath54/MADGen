"""
Acoustic room simulation orchestrator using Pyroomacoustics.

Coordinates placing the smart assistant microphone array and persona sources into
the room, executing ray-tracing and image source simulations, and rendering audio.
Supports generating both mixed spatial audio and isolated ground-truth stems.
"""

# Import Modules
import logging

import numpy as np
import pyroomacoustics as pra

from src.common.types import AudioArray
from src.personas.manager import PersonaTrack
from src.common.constants import EPSILON
from src.common.audio_utils import normalize_audio_peak
from src.simulation.microphone import build_microphone_array
from src.simulation.room_builder import build_room
from src.config.models import (
    RoomConfig,
    MicrophoneConfig,
)

logger: logging.Logger = logging.getLogger(__name__)


def add_persona_sources_to_room(
    room: pra.ShoeBox,
    tracks: list[PersonaTrack],
) -> None:
    """
    Inject persona audio tracks into the simulated room at their spatial locations.

    Extended personas with positive size are modeled using distributed sub-sources.

    Args:
        room (pra.ShoeBox): Configured room simulation instance.
        tracks (list[PersonaTrack]): Audio tracks and spatial properties for each persona.
    """

    # Iterate through each active persona track
    for track in tracks:
        points: list[tuple[float, float, float]] = track.persona.calculate_source_points()
        num_points: int = len(points)

        logger.info(
            "Adding '%s' with %d acoustic emission point(s)",
            track.persona.display_name,
            num_points,
        )

        # Scale signal per point to maintain uniform radiated acoustic power
        scaled_signal: AudioArray = (track.audio / float(num_points)).astype(np.float64)

        # Add each emission point into the room
        for point in points:
            pos_list: list[float] = [point[0], point[1], point[2]]
            room.add_source(pos_list, signal=scaled_signal)


def extract_simulated_signals(room: pra.ShoeBox) -> AudioArray:
    """
    Extract and format multi-channel microphone recordings from simulated room.

    Args:
        room (pra.ShoeBox): Simulated room after room.simulate() execution.

    Returns:
        AudioArray: Formatted audio array of shape (N_samples, N_channels).
    """

    # Extract raw microphone array signals of shape (N_channels, N_samples)
    raw_signals: np.ndarray = room.mic_array.signals

    # Transpose to standard audio shape (N_samples, N_channels)
    audio_out: AudioArray = raw_signals.T.astype(np.float64)

    return audio_out


def run_acoustic_simulation(
    room_config: RoomConfig,
    mic_config: MicrophoneConfig,
    tracks: list[PersonaTrack],
) -> AudioArray:
    """
    Execute full spatial acoustic room simulation for the smart assistant scene.

    Args:
        room_config (RoomConfig): Room dimensions and wall absorption specifications.
        mic_config (MicrophoneConfig): Microphone array parameters and placement.
        tracks (list[PersonaTrack]): Persona audio tracks to simulate.

    Returns:
        AudioArray: Multi-channel audio array recorded by the assistant.
    """

    # Step 1: Construct ShoeBox room
    logger.info(
        "Building room: %s with absorption %.2f",
        room_config.dimensions,
        room_config.absorption,
    )
    room: pra.ShoeBox = build_room(room_config)

    # Step 2: Build and attach assistant microphone array
    logger.info(
        "Adding %s microphone array at %s",
        mic_config.mic_type.value,
        mic_config.position,
    )
    mic_matrix: np.ndarray = build_microphone_array(mic_config)
    room.add_microphone_array(mic_matrix)

    # Step 3: Add persona sources
    add_persona_sources_to_room(room, tracks)

    # Step 4: Run acoustic simulation
    logger.info("Running ray-tracing acoustic propagation simulation...")
    room.simulate()

    # Step 5: Extract and normalize recorded multi-channel audio
    raw_audio: AudioArray = extract_simulated_signals(room)
    output_audio: AudioArray = normalize_audio_peak(raw_audio, peak_target=0.92)
    logger.info("Simulation complete. Output audio shape: %s", output_audio.shape)

    return output_audio


def simulate_single_persona_stem(
    room_config: RoomConfig,
    mic_config: MicrophoneConfig,
    track: PersonaTrack,
) -> AudioArray:
    """
    Simulate acoustic propagation for an isolated individual speaker track.

    Args:
        room_config (RoomConfig): Room parameters.
        mic_config (MicrophoneConfig): Microphone array parameters.
        track (PersonaTrack): Individual speaker track.

    Returns:
        AudioArray: Raw multi-channel spatial audio for this speaker at the mic.
    """

    # Build room and attach microphone array
    room: pra.ShoeBox = build_room(room_config)
    mic_matrix: np.ndarray = build_microphone_array(mic_config)
    room.add_microphone_array(mic_matrix)

    # Add only this speaker's sources
    add_persona_sources_to_room(room, [track])
    room.simulate()

    return extract_simulated_signals(room)


def align_and_sum_stems(
    raw_stems: dict[str, AudioArray],
    target_peak: float,
) -> tuple[AudioArray, dict[str, AudioArray]]:
    """
    Zero-pad stems to uniform length, compute composite mix, and apply scaling.

    Args:
        raw_stems (dict[str, AudioArray]): Per-speaker raw multi-channel audio stems.
        target_peak (float): Peak level for normalized mix.

    Returns:
        tuple[AudioArray, dict[str, AudioArray]]: Normalized mix and aligned stems.
    """

    # Align all stems to maximum length across tracks
    max_len: int = max(s.shape[0] for s in raw_stems.values())
    num_channels: int = next(iter(raw_stems.values())).shape[1]

    aligned_stems: dict[str, AudioArray] = {}
    composite_raw: AudioArray = np.zeros((max_len, num_channels), dtype=np.float64)

    for spk_id, stem in raw_stems.items():
        if stem.shape[0] < max_len:
            padded: AudioArray = np.zeros((max_len, num_channels), dtype=np.float64)
            padded[: stem.shape[0], :] = stem
            aligned_stems[spk_id] = padded
            composite_raw += padded
        else:
            aligned_stems[spk_id] = stem
            composite_raw += stem

    # Compute common scaling factor from composite mix
    max_peak: float = float(np.max(np.abs(composite_raw)))
    scaling_factor: float = target_peak / (max_peak + EPSILON) if max_peak > EPSILON else 1.0

    normalized_mix: AudioArray = (composite_raw * scaling_factor).astype(np.float64)
    normalized_stems: dict[str, AudioArray] = {
        spk_id: (stem * scaling_factor).astype(np.float64)
        for spk_id, stem in aligned_stems.items()
    }

    return (normalized_mix, normalized_stems)


def run_simulation_with_stems(
    room_config: RoomConfig,
    mic_config: MicrophoneConfig,
    tracks: list[PersonaTrack],
    target_peak: float = 0.92,
) -> tuple[AudioArray, dict[str, AudioArray]]:
    """
    Simulate individual speaker stems and combine them into a composite mix.

    Preserves mathematical linearity so that sum of isolated stems equals the mix.

    Args:
        room_config (RoomConfig): Room specifications.
        mic_config (MicrophoneConfig): Microphone array geometry.
        tracks (list[PersonaTrack]): Audio tracks for all personas.
        target_peak (float): Peak target amplitude.

    Returns:
        tuple[AudioArray, dict[str, AudioArray]]: Composite mix and isolated speaker stems.
    """

    # Simulate each speaker independently
    raw_stems: dict[str, AudioArray] = {}
    for track in tracks:
        logger.info("Simulating isolated spatial stem for '%s'...", track.persona.display_name)
        stem_audio: AudioArray = simulate_single_persona_stem(
            room_config=room_config,
            mic_config=mic_config,
            track=track,
        )
        raw_stems[track.persona.identifier] = stem_audio

    # Handle empty tracks
    if not raw_stems:
        empty_out: AudioArray = np.zeros((100, 2), dtype=np.float64)
        return (empty_out, {})

    return align_and_sum_stems(raw_stems=raw_stems, target_peak=target_peak)
