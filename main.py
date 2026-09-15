"""
Main launcher script for smart assistant acoustic simulation with Piper-TTS personas.

Parses command-line options, loads scene configuration, resolves voice dependencies,
and triggers the spatial acoustic simulation pipeline.
"""

# Import Modules
from pathlib import Path

import logging
import argparse

from src.common.types import MicrophoneType
from src.config.loader import load_scene_config
from src.config.models import SceneConfig
from src.tts.voice_downloader import download_piper_voice
from src.pipeline.orchestrator import run_pipeline

logger: logging.Logger = logging.getLogger(__name__)


def build_cli_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line argument parser for the simulation runner.

    Returns:
        argparse.ArgumentParser: Populated parser instance.
    """

    # Initialize parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Simulate multi-persona room acoustics captured by a smart assistant device.",
    )

    # Configuration files and paths
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_scene.json",
        help="Path to JSON scene configuration file (default: config/default_scene.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/smart_assistant_simulation.wav",
        help="Destination path for the rendered WAV file.",
    )
    parser.add_argument(
        "--voices-dir",
        type=str,
        default="data/piper_voices",
        help="Directory where Piper ONNX voices are stored.",
    )

    # Synthesis options
    parser.add_argument(
        "--mock-tts",
        action="store_true",
        help="Use synthetic tones instead of Piper neural voices for testing.",
    )
    parser.add_argument(
        "--download-voice",
        type=str,
        default=None,
        help="Automatically download a specified Piper voice before starting simulation.",
    )

    # CLI parameter overrides
    parser.add_argument(
        "--room-dim",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override room dimensions in meters (e.g. 6.0 5.0 2.8).",
    )
    parser.add_argument(
        "--mic-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override assistant microphone position in meters (e.g. 0.4 0.4 0.85).",
    )
    parser.add_argument(
        "--mic-type",
        type=str,
        choices=["mono", "stereo", "circular"],
        default=None,
        help="Override microphone array geometry type.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Override simulation sampling rate in Hertz.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity level.",
    )

    return parser


def apply_cli_overrides(
    scene_config: SceneConfig,
    args: argparse.Namespace,
) -> None:
    """
    Apply command-line argument overrides onto an existing SceneConfig instance.

    Args:
        scene_config (SceneConfig): Parsed scene configuration to mutate.
        args (argparse.Namespace): Parsed command-line arguments.
    """

    # Apply room dimension override
    if args.room_dim:
        scene_config.room.dimensions = (args.room_dim[0], args.room_dim[1], args.room_dim[2])

    # Apply microphone position override
    if args.mic_pos:
        scene_config.assistant.position = (args.mic_pos[0], args.mic_pos[1], args.mic_pos[2])

    # Apply microphone type override
    if args.mic_type:
        scene_config.assistant.mic_type = MicrophoneType(args.mic_type)

    # Apply sample rate override
    if args.sample_rate:
        scene_config.room.sample_rate = args.sample_rate


def setup_environment(
    voices_dir: Path,
    download_voice_key: str | None,
) -> None:
    """
    Ensure voice repository exists and download initial voice if specified.

    Args:
        voices_dir (Path): Local directory for voice models.
        download_voice_key (str | None): Optional voice key to download.
    """

    # Create storage directory
    voices_dir.mkdir(parents=True, exist_ok=True)

    # Download requested voice model if provided
    if download_voice_key:
        logger.info("Downloading requested voice: %s", download_voice_key)
        download_piper_voice(download_voice_key, voices_dir)


def main() -> None:
    """
    Execute simulation workflow based on command-line invocation.
    """

    # Parse command-line arguments
    parser: argparse.ArgumentParser = build_cli_parser()
    args: argparse.Namespace = parser.parse_args()

    # Configure logger
    numeric_level: int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Define paths
    config_file: Path = Path(args.config)
    output_wav: Path = Path(args.output)
    voices_path: Path = Path(args.voices_dir)

    # Prepare environment and optional downloads
    setup_environment(voices_path, args.download_voice)

    # Load scene configuration
    logger.info("Loading scene configuration from %s", config_file)
    scene_config: SceneConfig = load_scene_config(config_file)

    # Apply any command line overrides
    apply_cli_overrides(scene_config, args)

    # Run end-to-end pipeline
    result = run_pipeline(
        scene_config=scene_config,
        voices_dir=voices_path,
        output_wav_path=output_wav,
        use_mock_tts=args.mock_tts,
    )

    # Display final execution summary
    print("\n" + "=" * 60)
    print("Simulation Completed Successfully!")
    print(f"  - Output file:   {result.output_path}")
    print(f"  - Duration:      {result.duration_s:.2f} seconds")
    print(f"  - Channels:      {result.num_channels}")
    print(f"  - Sampling Rate: {result.sample_rate} Hz")
    print(f"  - Personas:      {result.num_personas}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
