"""
Command-line interface for batch procedural audio dataset generation.

Generates diverse multi-speaker acoustic scenes with parallel conversations,
turn-taking interruptions, vocal distortions, and ground-truth annotations
for training and evaluating speaker diarization, source separation, and ASR.
"""

# Import Modules
import argparse
import logging
from pathlib import Path

from src.config.models import BatchGenerationConfig, ConversationalStyle
from src.pipeline.dataset_pipeline import generate_dataset_batch
from src.procedural.ambiance_presets import get_preset_names

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
        "--min-sentences",
        type=int,
        default=100,
        help="Minimum total sentences per sample across all speakers (default: 100).",
    )
    parser.add_argument(
        "--duration-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional duration range in seconds (default: None, auto-calculated from speech).",
    )
    parser.add_argument(
        "--speakers-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[4, 10],
        help="Range of concurrent speaker count per sample (default: 4 10).",
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

    # LLM & Ambiance Preset Options
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Generate dynamic dialogue lines using llama.cpp / llama-server.",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        help="Endpoint URL for llama.cpp / OpenAI server (default: http://127.0.0.1:8080/v1).",
    )
    parser.add_argument(
        "--ambiance",
        type=str,
        default="random",
        choices=["random"] + get_preset_names(),
        help="Ambiance and social constraint preset (default: random).",
    )
    parser.add_argument(
        "--constraint-words",
        type=int,
        default=3,
        help="Number of words to sample from 100k dictionary as thematic constraints (default: 3).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for LLM dialogue generation (default: 0.7).",
    )
    parser.add_argument(
        "--parallel-prob",
        type=float,
        default=0.5,
        help="Probability of parallel side conversations appearing in scenes (default: 0.5).",
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
    duration_tuple: tuple[float, float] | None = None
    if args.duration_range is not None:
        duration_tuple = (args.duration_range[0], args.duration_range[1])

    dur_str: str = (
        f"{duration_tuple[0]}s - {duration_tuple[1]}s"
        if duration_tuple is not None
        else "Auto-calculated from speech synthesis"
    )
    speakers_tuple: tuple[int, int] = (args.speakers_range[0], args.speakers_range[1])
    export_isolated: bool = not args.no_isolated

    print("\n" + "=" * 65)
    print("Starting Procedural Audio Dataset Generation...")
    print(f"  - Target Samples:      {args.num_samples}")
    print(f"  - Min Sentences:       {args.min_sentences}")
    print(f"  - Duration:            {dur_str}")
    print(f"  - Speakers Range:      {speakers_tuple[0]} - {speakers_tuple[1]} speakers")
    print(f"  - Languages:           {', '.join(args.languages)}")
    print(f"  - Style Scenario:      {style_enum.value}")
    print(f"  - Ambiance Preset:     {args.ambiance}")
    print(f"  - LLM Dialogues:       {args.use_llm} ({args.llm_url})")
    print(f"  - Constraint Words:    {args.constraint_words} from 100k dictionary")
    print(f"  - Isolated Stems:      {export_isolated}")
    print(f"  - Parallel Prob:       {args.parallel_prob}")
    print(f"  - Output Folder:       {out_dir}")
    print("=" * 65 + "\n")

    # Configure and run batch generation
    batch_config: BatchGenerationConfig = BatchGenerationConfig(
        num_samples=args.num_samples,
        duration_range=duration_tuple,
        min_sentences=args.min_sentences,
        speakers_range=speakers_tuple,
        languages=args.languages,
        style=style_enum,
        voices_dir=voices_dir,
        output_dir=out_dir,
        use_mock_tts=args.mock_tts,
        export_isolated_stems=export_isolated,
        use_llm=args.use_llm,
        llm_url=args.llm_url,
        ambiance_preset=args.ambiance,
        num_constraint_words=args.constraint_words,
        llm_temperature=args.llm_temperature,
        parallel_prob=args.parallel_prob,
    )
    samples: list[Path] = generate_dataset_batch(batch_config)

    print("\n" + "=" * 65)
    print("Batch Dataset Generation Complete!")
    print(f"  - Generated {len(samples)} sample variations in: {out_dir}")
    print(f"  - Master Manifest: {out_dir / 'dataset_manifest.json'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
