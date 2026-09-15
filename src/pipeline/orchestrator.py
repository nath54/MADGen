"""
End-to-end pipeline orchestrator for smart assistant room acoustics simulation.

Coordinates configuration loading, text-to-speech generation, multi-track alignment,
spatial acoustic simulation, and audio file export.
"""

# Import Modules
from pathlib import Path
from dataclasses import dataclass

import logging

from src.common.types import AudioArray
from src.config.models import SceneConfig
from src.personas.persona import Persona
from src.common.audio_utils import write_wav_file
from src.simulation.simulator import run_acoustic_simulation
from src.tts.synthesizer import (
    BaseSynthesizer,
    MockSynthesizer,
    PiperSynthesizer,
)
from src.personas.manager import (
    PersonaTrack,
    PersonaManager,
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """
    Summary metrics and output information for a completed simulation run.
    """

    output_path: Path
    num_channels: int
    duration_s: float
    sample_rate: int
    num_personas: int


def instantiate_synthesizer(
    voices_dir: Path,
    use_mock_tts: bool,
) -> BaseSynthesizer:
    """
    Create appropriate synthesizer instance based on configuration and voice availability.

    Args:
        voices_dir (Path): Local folder containing Piper ONNX voices.
        use_mock_tts (bool): Flag forcing synthetic offline fallback synthesis.

    Returns:
        BaseSynthesizer: Initialized synthesizer instance.
    """

    # Return mock synthesizer if requested explicitly
    if use_mock_tts:
        logger.info("Using MockSynthesizer (synthetic tones) for offline testing")
        return MockSynthesizer()

    # Attempt using Piper-TTS
    logger.info("Using PiperSynthesizer with voices directory: %s", voices_dir)
    return PiperSynthesizer(voices_dir=voices_dir)


def run_pipeline(
    scene_config: SceneConfig,
    voices_dir: Path,
    output_wav_path: Path,
    use_mock_tts: bool = False,
) -> SimulationResult:
    """
    Execute full simulation workflow from scene specification to exported WAV file.

    Args:
        scene_config (SceneConfig): Parsed and validated scene settings.
        voices_dir (Path): Path to folder holding Piper voice models.
        output_wav_path (Path): File destination for rendered audio.
        use_mock_tts (bool): Whether to use synthetic tones instead of Piper models.

    Returns:
        SimulationResult: Execution metrics and output path.
    """

    # Step 1: Initialize speech synthesizer
    synthesizer: BaseSynthesizer = instantiate_synthesizer(
        voices_dir=voices_dir,
        use_mock_tts=use_mock_tts,
    )

    # Step 2: Build Persona instances
    personas: list[Persona] = [Persona(config=p) for p in scene_config.personas]
    manager: PersonaManager = PersonaManager(personas=personas)

    # Step 3: Synthesize speech and compile continuous timeline tracks
    logger.info("Synthesizing dialogue for %d personas...", len(personas))
    tracks: list[PersonaTrack] = manager.synthesize_all_tracks(
        synthesizer=synthesizer,
        sample_rate=scene_config.room.sample_rate,
    )

    # Step 4: Run spatial acoustic propagation simulation
    output_audio: AudioArray = run_acoustic_simulation(
        room_config=scene_config.room,
        mic_config=scene_config.assistant,
        tracks=tracks,
    )

    # Step 5: Export rendered multi-channel audio to disk
    logger.info("Writing output WAV to %s", output_wav_path)
    write_wav_file(
        output_path=output_wav_path,
        audio_data=output_audio,
        sample_rate=scene_config.room.sample_rate,
    )

    # Calculate final metrics
    num_samples: int = output_audio.shape[0]
    num_channels: int = output_audio.shape[1] if output_audio.ndim > 1 else 1
    duration_s: float = num_samples / float(scene_config.room.sample_rate)

    return SimulationResult(
        output_path=output_wav_path,
        num_channels=num_channels,
        duration_s=duration_s,
        sample_rate=scene_config.room.sample_rate,
        num_personas=len(personas),
    )
