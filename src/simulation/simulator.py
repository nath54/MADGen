"""
Acoustic room simulation orchestrator using Pyroomacoustics.

Coordinates placing the smart assistant microphone array and persona sources into
the room, executing ray-tracing and image source simulations, and rendering audio.
"""

# Import Modules
import logging

import numpy as np
import pyroomacoustics as pra

from src.common.types import AudioArray
from src.personas.manager import PersonaTrack
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

    # Normalize to safe headroom level preventing clipping
    normalized: AudioArray = normalize_audio_peak(audio_out, peak_target=0.92)

    return normalized


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
    output_audio: AudioArray = extract_simulated_signals(room)
    logger.info("Simulation complete. Output audio shape: %s", output_audio.shape)

    return output_audio
