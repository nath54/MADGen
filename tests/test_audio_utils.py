"""
Unit tests for audio manipulation and mathematical processing utilities.
"""

# Import Modules
from pathlib import Path

import unittest

import numpy as np
import soundfile as sf

from src.common.types import AudioArray
from src.common.audio_utils import (
    calculate_gcd,
    write_wav_file,
    resample_audio,
    normalize_audio_peak,
    place_audio_on_timeline,
    generate_synthetic_fallback_tone,
)


class TestAudioUtils(unittest.TestCase):
    """
    Validation suite for audio processing helper functions.
    """

    def test_calculate_gcd(self) -> None:
        """
        Verify greatest common divisor arithmetic calculations.
        """

        # Verify coprime and shared divisor cases
        self.assertEqual(calculate_gcd(16000, 22050), 50)
        self.assertEqual(calculate_gcd(44100, 16000), 100)

    def test_resample_identical_sample_rate(self) -> None:
        """
        Ensure identical sampling rates return exact signal copy.
        """

        # Generate sample signal
        orig_sig: AudioArray = np.array([0.1, 0.5, -0.2, 0.8], dtype=np.float64)

        # Resample with identical rates
        resampled: AudioArray = resample_audio(orig_sig, 16000, 16000)

        np.testing.assert_array_almost_equal(orig_sig, resampled)

    def test_resample_different_sample_rate(self) -> None:
        """
        Ensure resampling produces array of expected length.
        """

        # Generate 1-second sine wave at 16000 Hz
        sample_rate_src: int = 16000
        sample_rate_target: int = 8000
        time_grid: np.ndarray = np.linspace(0.0, 1.0, sample_rate_src, endpoint=False)
        signal: AudioArray = np.sin(2.0 * np.pi * 100.0 * time_grid).astype(np.float64)

        # Resample down to 8000 Hz
        resampled: AudioArray = resample_audio(signal, sample_rate_src, sample_rate_target)

        # Target should have approximately half the samples
        self.assertEqual(len(resampled), sample_rate_target)

    def test_normalize_audio_peak(self) -> None:
        """
        Verify peak normalization scales signal to target amplitude without clipping.
        """

        # Input audio with peak of 2.0
        signal: AudioArray = np.array([-2.0, 0.5, 1.0], dtype=np.float64)

        # Normalize to peak 0.9
        target_peak: float = 0.9
        normalized: AudioArray = normalize_audio_peak(signal, peak_target=target_peak)

        # Check maximum absolute value equals target
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), target_peak, places=5)

    def test_normalize_silent_audio(self) -> None:
        """
        Verify silent audio array does not cause zero division error.
        """

        # All zeros
        silent_signal: AudioArray = np.zeros(100, dtype=np.float64)

        # Normalize silent signal
        result: AudioArray = normalize_audio_peak(silent_signal, peak_target=0.95)

        self.assertEqual(float(np.max(np.abs(result))), 0.0)

    def test_place_audio_on_timeline(self) -> None:
        """
        Verify placing an audio clip onto timeline adds samples correctly.
        """

        # Create empty timeline and sample clip
        timeline: AudioArray = np.zeros(10, dtype=np.float64)
        clip: AudioArray = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        # Place clip starting at sample index 4
        updated: AudioArray = place_audio_on_timeline(timeline, clip, 4)

        expected: AudioArray = np.array(
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(updated, expected)

    def test_generate_synthetic_tone(self) -> None:
        """
        Verify fallback synthetic tone produces non-empty audio within peak target.
        """

        # Generate 0.5s tone at 16000 Hz
        sample_rate: int = 16000
        duration_s: float = 0.5
        tone: AudioArray = generate_synthetic_fallback_tone(duration_s, sample_rate)

        # Verify length and maximum amplitude
        expected_len: int = int(duration_s * sample_rate)
        self.assertEqual(len(tone), expected_len)
        self.assertLessEqual(float(np.max(np.abs(tone))), 0.81)

    def test_write_and_read_wav(self) -> None:
        """
        Verify audio file writing and reading round-trip integrity.
        """

        # Create temporary file path
        test_path: Path = Path("data/output/test_roundtrip.wav")

        # Generate stereo sample signal
        sample_rate: int = 16000
        stereo_signal: AudioArray = np.zeros((100, 2), dtype=np.float64)
        stereo_signal[:, 0] = 0.5
        stereo_signal[:, 1] = -0.5

        # Write WAV file
        write_wav_file(test_path, stereo_signal, sample_rate)
        self.assertTrue(test_path.is_file())

        # Read back written file
        loaded_data, read_sr = sf.read(str(test_path))
        self.assertEqual(read_sr, sample_rate)
        self.assertEqual(loaded_data.shape, (100, 2))

        # Clean up temporary file
        if test_path.is_file():
            test_path.unlink()


if __name__ == "__main__":
    unittest.main()
