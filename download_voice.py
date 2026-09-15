"""
CLI utility to download and manage Piper-TTS voices from HuggingFace.

Allows downloading neural ONNX voice models and accompanying JSON configurations
into the local project voice repository.
"""

# Import Modules
from pathlib import Path

import logging
import argparse

from src.tts.voice_downloader import (
    download_piper_voice,
    fetch_voices_catalog,
    download_voices_for_languages,
)

logger: logging.Logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI argument parser for voice management.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """

    # Create command-line parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Download and manage Piper-TTS voices from HuggingFace.",
    )

    # Add CLI arguments
    parser.add_argument(
        "--voice",
        type=str,
        default="en_US-lessac-low",
        help="Voice identifier to download (e.g. 'en_US-lessac-low', 'en_US-lessac-medium').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Batch download all voices matching specified languages.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "fr", "es", "de"],
        help="Languages to batch download when using --all (default: en fr es de).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/piper_voices",
        help="Target folder for downloaded voice files (default: data/piper_voices).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available voices from the official catalog.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter listed voices by name or language prefix (e.g. 'en_US', 'fr').",
    )

    return parser


def handle_list_voices(search_filter: str | None) -> None:
    """
    Fetch and display catalog voices matching the optional search filter.

    Args:
        search_filter (str | None): Optional substring to filter voice keys.
    """

    # Fetch voice catalog from HuggingFace or local cache
    print("Fetching available Piper voices catalog...")
    catalog = fetch_voices_catalog()
    all_keys: list[str] = sorted(catalog.keys())

    # Apply filter if provided
    matched_keys: list[str] = [
        k for k in all_keys
        if search_filter is None or search_filter.lower() in k.lower()
    ]

    print(f"Found {len(matched_keys)} matching voice(s):")
    for key in matched_keys:
        info = catalog[key]
        lang: str = info.get("language", {}).get("name_english", "Unknown")
        print(f"  - {key} ({lang})")


def execute_download(
    voice_key: str,
    target_dir: Path,
) -> None:
    """
    Download voice ONNX and JSON files to disk.

    Args:
        voice_key (str): Voice identifier.
        target_dir (Path): Destination folder.
    """

    # Initiate voice download
    print(f"Downloading Piper voice '{voice_key}' to '{target_dir}'...")
    onnx_file, json_file = download_piper_voice(voice_key, target_dir)
    print("Successfully downloaded:")
    print(f"  - ONNX Model: {onnx_file}")
    print(f"  - Config JSON: {json_file}")


def main() -> None:
    """
    Main entrypoint for voice download utility.
    """

    # Configure logging output
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Parse arguments
    parser: argparse.ArgumentParser = build_argument_parser()
    args: argparse.Namespace = parser.parse_args()

    # Handle list command
    if args.list or args.filter:
        handle_list_voices(args.filter)
        return

    # Handle batch download all command
    if args.all:
        output_dir: Path = Path(args.output_dir)
        print(f"Batch downloading all voices for languages: {args.languages}...")
        downloaded = download_voices_for_languages(args.languages, output_dir)
        print(f"Finished downloading {len(downloaded)} voice model(s).")
        return

    # Handle single voice download command
    output_dir = Path(args.output_dir)
    execute_download(args.voice, output_dir)


if __name__ == "__main__":
    main()
