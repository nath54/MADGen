"""
CLI utility to download and filter the 100k-word thematic constraint dictionary.

Fetches clean English vocabulary words from the remote repository, filters by length
and alphabetical validity, and caches to data/dictionaries/words_100k.txt.
"""

# Import Modules
from pathlib import Path

import logging
import argparse
import sys

from src.procedural.word_dictionary import (
    DEFAULT_DICTIONARY_PATH,
    download_and_filter_dictionary,
    load_word_dictionary,
)

logger: logging.Logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI argument parser for dictionary downloading.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Download and filter the 100k-word constraint dictionary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DICTIONARY_PATH,
        help="Destination path for the dictionary text file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the local dictionary file already exists.",
    )

    return parser


def main() -> int:
    """
    Execute dictionary download workflow based on CLI arguments.

    Returns:
        int: Exit status code (0 for success, non-zero for error).
    """

    parser: argparse.ArgumentParser = build_argument_parser()
    args: argparse.Namespace = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out_path: Path = args.output

    if out_path.is_file() and out_path.stat().st_size > 0 and not args.force:
        words = load_word_dictionary(dictionary_path=out_path)
        print(f"Dictionary already exists at '{out_path}' with {len(words):,} words.")
        print("Use '--force' to re-download from the remote repository.")
        return 0

    print(f"Downloading and filtering clean vocabulary to '{out_path}'...")
    words = download_and_filter_dictionary(target_path=out_path)

    if not words:
        print("Error: Failed to download or filter dictionary.", file=sys.stderr)
        return 1

    print(f"Successfully saved {len(words):,} words to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
