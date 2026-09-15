"""
Word dictionary manager and thematic constraint keyword sampler for LLM dialogues.

Loads and caches a dictionary of 100,000 common words, providing randomized
thematic constraint keywords to anchor procedural conversation topics and guarantee
lexical diversity in generated speech dialogues.
"""

# Import Modules
from pathlib import Path

import random
import logging
import urllib.error
import urllib.request

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_DICTIONARY_PATH: Path = Path("data/dictionaries/words_100k.txt")
REMOTE_WORDS_URL: str = (
    "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
)

# Built-in fallback keywords if dictionary is missing and offline
FALLBACK_KEYWORDS: list[str] = [
    "astronomy", "architecture", "baking", "clover", "telescope", "labyrinth",
    "submarine", "clockwork", "glacier", "volcano", "compass", "origami",
    "harvest", "symphony", "quarry", "fountain", "meadow", "lantern",
    "corridor", "blueprint", "meteor", "constellation", "kaleidoscope",
    "amber", "obsidian", "canvas", "sculpture", "pendulum", "almanac",
    "beacon", "canyon", "echo", "mirage", "oasis", "parchment", "quiver",
    "radiance", "solitude", "summit", "timber", "velocity", "whisper",
]

_CACHED_WORD_LIST: list[str] = []


def download_and_filter_dictionary(target_path: Path) -> list[str]:
    """
    Download raw word list from remote repository, filter, and save to disk.

    Args:
        target_path (Path): Destination filesystem location.

    Returns:
        list[str]: Filtered words list.
    """

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(
            REMOTE_WORDS_URL,
            headers={"User-Agent": "AudioDatasetGenerator/1.0"},
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            raw_content: str = response.read().decode("utf-8")

        lines: list[str] = raw_content.splitlines()
        filtered: list[str] = [
            w.strip().lower()
            for w in lines
            if w.strip().isalpha() and 4 <= len(w.strip()) <= 12
        ]
        selected: list[str] = filtered[:100000]

        target_path.write_text("\n".join(selected), encoding="utf-8")
        logger.info("Saved %d words to %s", len(selected), target_path)
        return selected
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        logger.warning("Failed downloading word dictionary from remote: %s", err)
        return list(FALLBACK_KEYWORDS)


def load_word_dictionary(
    dictionary_path: Path | None = None,
    force_reload: bool = False,
) -> list[str]:
    """
    Load 100k word dictionary from local cache or remote with in-memory caching.

    Args:
        dictionary_path (Path | None): Custom dictionary file path.
        force_reload (bool): Whether to invalidate the in-memory cache.

    Returns:
        list[str]: Loaded list of vocabulary words.
    """

    if _CACHED_WORD_LIST and not force_reload:
        return list(_CACHED_WORD_LIST)

    path: Path = dictionary_path if dictionary_path is not None else DEFAULT_DICTIONARY_PATH

    # Read from local file if exists
    if path.is_file() and path.stat().st_size > 0:
        try:
            content: str = path.read_text(encoding="utf-8")
            words: list[str] = [w.strip() for w in content.splitlines() if w.strip()]
            if words:
                _CACHED_WORD_LIST.clear()
                _CACHED_WORD_LIST.extend(words)
                return list(_CACHED_WORD_LIST)
        except OSError as err:
            logger.warning("Failed reading local dictionary %s: %s", path, err)

    # Attempt download if missing
    downloaded: list[str] = download_and_filter_dictionary(path)
    _CACHED_WORD_LIST.clear()
    _CACHED_WORD_LIST.extend(downloaded)
    return list(_CACHED_WORD_LIST)


def sample_constraint_keywords(
    count: int = 3,
    dictionary_path: Path | None = None,
) -> list[str]:
    """
    Randomly sample N distinct constraint words from the 100k word dictionary.

    Args:
        count (int): Number of unique keywords to sample.
        dictionary_path (Path | None): Optional custom path to words dictionary.

    Returns:
        list[str]: Sampled thematic constraint keywords.
    """

    words: list[str] = load_word_dictionary(dictionary_path=dictionary_path)
    count = max(1, min(count, len(words)))

    sampled: list[str] = random.sample(words, k=count)
    return sampled
