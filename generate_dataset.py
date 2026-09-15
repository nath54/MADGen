"""
Command-line interface for batch procedural audio dataset generation.

Generates diverse multi-speaker acoustic scenes with parallel conversations,
turn-taking interruptions, vocal distortions, and ground-truth annotations
for training and evaluating speaker diarization, source separation, and ASR.
"""

# Import Modules
from pathlib import Path

import logging
import argparse

from src.config.models import ConversationalStyle
from src.pipeline.dataset_pipeline import generate_dataset_batch

logger: logging.Logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct command-line argument parser for dataset generator.

    Returns:
        argparse.ArgumentParser: Populated parser instance.
    """

    # Initialize parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Generate procedural multi-speaker dataset variations with ground-truth labels."
        ),
    )

    # Dataset batch options
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of distinct procedural sample variations to generate (default: 1).",
    )
    parser.add_argument(
        "--duration-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[20.0, 45.0],
        help="Range of duration in seconds per sample (default: 20.0 45.0).",
    )
    parser.add_argument(
        "--speakers-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[2, 4],
        help="Range of concurrent speaker count per sample (default: 2 4).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en"],
        choices=["en", "fr", "es", "de"],
        help="Languages allowed in the generated scenes (default: en).",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="mixed",
        choices=["mixed", "party", "meeting", "argument", "assistant"],
        help="Conversational style setting turn-taking dynamics and emotions (default: mixed).",
    )

    # Storage and synthesis settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets/procedural_run",
        help="Target folder where generated dataset samples will be stored.",
    )
    parser.add_argument(
        "--voices-dir",
        type=str,
        default="data/piper_voices",
        help="Directory where local Piper ONNX voices are located.",
    )
    parser.add_argument(
        "--no-isolated",
        action="store_true",
        help="Disable exporting per-speaker isolated spatial stems (faster generation).",
    )
    parser.add_argument(
        "--mock-tts",
        action="store_true",
        help="Use synthetic tones instead of neural Piper voices for rapid testing.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity level.",
    )

    return parser


def main() -> None:
    """
    Execute batch procedural dataset generation workflow.
    """

    # Parse arguments
    parser: argparse.ArgumentParser = build_argument_parser()
    args: argparse.Namespace = parser.parse_args()

    # Configure logging
    numeric_level: int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Convert paths and parameters
    out_dir: Path = Path(args.output_dir)
    voices_dir: Path = Path(args.voices_dir)
    style_enum: ConversationalStyle = ConversationalStyle(args.style)
    duration_tuple: tuple[float, float] = (args.duration_range[0], args.duration_range[1])
    speakers_tuple: tuple[int, int] = (args.speakers_range[0], args.speakers_range[1])
    export_isolated: bool = not args.no_isolated

    print("\n" + "=" * 65)
    print("Starting Procedural Audio Dataset Generation...")
    print(f"  - Target Samples:  {args.num_samples}")
    print(f"  - Duration Range:  {duration_tuple[0]}s - {duration_tuple[1]}s")
    print(f"  - Speakers Range:  {speakers_tuple[0]} - {speakers_tuple[1]} speakers")
    print(f"  - Languages:       {', '.join(args.languages)}")
    print(f"  - Style Scenario:  {style_enum.value}")
    print(f"  - Isolated Stems:  {export_isolated}")
    print(f"  - Output Folder:   {out_dir}")
    print("=" * 65 + "\n")

    # Run batch generation
    samples: list[Path] = generate_dataset_batch(
        num_samples=args.num_samples,
        duration_range=duration_tuple,
        speakers_range=speakers_tuple,
        languages=args.languages,
        style=style_enum,
        voices_dir=voices_dir,
        output_dir=out_dir,
        use_mock_tts=args.mock_tts,
        export_isolated_stems=export_isolated,
    )

    print("\n" + "=" * 65)
    print("Batch Dataset Generation Complete!")
    print(f"  - Generated {len(samples)} sample variations in: {out_dir}")
    print(f"  - Master Manifest: {out_dir / 'dataset_manifest.json'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
