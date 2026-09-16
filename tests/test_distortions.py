"""
Unit tests for vocal distortions, laughter modulation, and ambient noise generation.
"""

# Import Modules
import unittest

import numpy as np

from src.common.types import AudioArray
from src.audio.distortions import (
    add_ambient_room_noise,
    apply_laughter_modulation,
    generate_ambient_pink_noise,
    apply_soft_clipping_overdrive,
)


class TestDistortions(unittest.TestCase):
    """
    Validation suite for vocal effects and acoustic distortion algorithms.
    """

    def test_soft_clipping_overdrive(self) -> None:
        """
        Verify soft-clipping saturation normalizes to peak target without exceeding.
        """

        # Generate loud sine wave
        time_grid: np.ndarray = np.linspace(0.0, 0.1, 1600, endpoint=False)
        signal: AudioArray = (2.5 * np.sin(2.0 * np.pi * 220.0 * time_grid)).astype(np.float64)

        # Apply overdrive
        target_peak: float = 0.95
        saturated: AudioArray = apply_soft_clipping_overdrive(
            audio_signal=signal,
            drive_gain=2.0,
            target_peak=target_peak,
        )

        # Peak must match target
        self.assertAlmostEqual(float(np.max(np.abs(saturated))), target_peak, places=4)
        self.assertEqual(len(saturated), len(signal))

    def test_laughter_modulation(self) -> None:
        """
        Verify laughter tremolo applies periodic amplitude modulation.
        """

        # 1-second constant amplitude test signal
        sample_rate: int = 16000
        constant_audio: AudioArray = np.ones(sample_rate, dtype=np.float64)

        # Apply 5 Hz laughter modulation
        modulated: AudioArray = apply_laughter_modulation(
            audio_signal=constant_audio,
            sample_rate=sample_rate,
            modulation_freq=5.0,
            depth=0.5,
        )

        self.assertEqual(len(modulated), sample_rate)
        # Min amplitude should be 1.0 - 0.5 = 0.5
        self.assertAlmostEqual(float(np.min(modulated)), 0.5, places=2)
        # Max amplitude should be 1.0
        self.assertAlmostEqual(float(np.max(modulated)), 1.0, places=2)

    def test_generate_ambient_pink_noise(self) -> None:
        """
        Verify pink noise generation produces correct length and finite values.
        """

        num_samples: int = 8000
        noise: AudioArray = generate_ambient_pink_noise(num_samples)

        self.assertEqual(len(noise), num_samples)
        self.assertTrue(np.all(np.isfinite(noise)))

    def test_add_ambient_room_noise_stereo(self) -> None:
        """
        Verify adding ambient noise to stereo signal maintains shape and increases energy.
        """

        # 0.5s stereo signal
        num_samples: int = 8000
        stereo_signal: AudioArray = np.zeros((num_samples, 2), dtype=np.float64)
        stereo_signal[:, 0] = 0.5
        stereo_signal[:, 1] = 0.5

        noisy_signal: AudioArray = add_ambient_room_noise(
            audio_signal=stereo_signal,
            target_snr_db=20.0,
        )

        self.assertEqual(noisy_signal.shape, (num_samples, 2))
        # Noisy signal variance should be greater than zero
        self.assertGreater(float(np.var(noisy_signal)), 0.0)

    def test_vocal_style_effects_disable_and_subtle_defaults(self) -> None:
        """
        Verify that disable_effects leaves signal dry, and default laughter modulation is subtle.
        """

        from src.config.models import VocalStyle  # pylint: disable=import-outside-toplevel
        from src.personas.manager import apply_vocal_style_effects  # pylint: disable=import-outside-toplevel

        sample_rate: int = 16000
        constant_audio: AudioArray = np.ones(sample_rate, dtype=np.float64)

        # With disable_effects=True, even SHOUTING or LAUGHTER returns exact input
        dry_shout: AudioArray = apply_vocal_style_effects(
            audio_clip=constant_audio,
            vocal_style=VocalStyle.SHOUTING,
            sample_rate=sample_rate,
            disable_effects=True,
        )
        np.testing.assert_array_equal(dry_shout, constant_audio)

        dry_laugh: AudioArray = apply_vocal_style_effects(
            audio_clip=constant_audio,
            vocal_style=VocalStyle.LAUGHTER,
            sample_rate=sample_rate,
            disable_effects=True,
        )
        np.testing.assert_array_equal(dry_laugh, constant_audio)

        # Default enabled laughter modulation should be very subtle (depth=0.05, min >= 0.94)
        subtle_laugh: AudioArray = apply_vocal_style_effects(
            audio_clip=constant_audio,
            vocal_style=VocalStyle.LAUGHTER,
            sample_rate=sample_rate,
            disable_effects=False,
        )
        self.assertGreaterEqual(float(np.min(subtle_laugh)), 0.94)


if __name__ == "__main__":
    unittest.main()
