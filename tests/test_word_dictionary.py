"""
Unit tests for the 100k-word dictionary loader and thematic constraint keyword sampler.
"""

# Import Modules
from pathlib import Path

import unittest

from src.procedural.word_dictionary import (
    load_word_dictionary,
    sample_constraint_keywords,
    DEFAULT_DICTIONARY_PATH,
)


class TestWordDictionary(unittest.TestCase):
    """
    Validation suite for 100k-word dictionary loading and constraint keyword sampling.
    """

    def test_load_word_dictionary(self) -> None:
        """
        Verify word dictionary loads a non-empty list of valid vocabulary strings.
        """

        words: list[str] = load_word_dictionary()

        self.assertGreaterEqual(len(words), 40)
        # If local 100k file exists, verify substantial size
        if DEFAULT_DICTIONARY_PATH.is_file():
            self.assertGreaterEqual(len(words), 50000)

        # Check all words are clean lowercase strings
        for word in words[:50]:
            self.assertTrue(word.isalpha())
            self.assertTrue(word.islower())

    def test_sample_constraint_keywords_uniqueness(self) -> None:
        """
        Verify sampling constraint keywords returns strictly unique non-empty words.
        """

        keywords: list[str] = sample_constraint_keywords(count=4)

        self.assertEqual(len(keywords), 4)
        self.assertEqual(len(keywords), len(set(keywords)))
        for kw in keywords:
            self.assertGreater(len(kw), 0)

    def test_sample_constraint_keywords_fallback(self) -> None:
        """
        Verify sampling constraint keywords falls back safely when path does not exist.
        """

        nonexistent_path: Path = Path("data/dictionaries/nonexistent_file_test.txt")
        keywords: list[str] = sample_constraint_keywords(
            count=3,
            dictionary_path=nonexistent_path,
        )

        self.assertEqual(len(keywords), 3)


if __name__ == "__main__":
    unittest.main()
