"""
Unit tests for audio analysis, spectral features, SI-SDR, and voice recovery metrics.
"""

# Import Modules
import unittest

import numpy as np

from src.audio.analysis import (
    SampleAnalysisReport,
    SpectralFeatures,
    analyze_speaker_in_mix,
    compute_log_spectrogram,
    compute_power_spectral_density,
    compute_si_sdr,
    compute_spectral_correlation,
    compute_spectral_features,
    compute_stft,
    ensure_1d_channel,
    evaluate_dataset_sample,
)


class TestAudioAnalysis(unittest.TestCase):
    """
    Validation suite for acoustic analysis, SI-SDR calculation, and voice recoverability.
    """

    def setUp(self) -> None:
        """
        Create synthetic test waveforms and sampling rate fixtures.
        """

        self.sample_rate: int = 16000
        self.duration_s: float = 1.0
        time_axis: np.ndarray = np.linspace(
            0,
            self.duration_s,
            int(self.sample_rate * self.duration_s),
            endpoint=False,
        )

        # 440 Hz tone and 1000 Hz tone
        self.tone_a: np.ndarray = 0.5 * np.sin(2.0 * np.pi * 440.0 * time_axis)
        self.tone_b: np.ndarray = 0.5 * np.sin(2.0 * np.pi * 1000.0 * time_axis)

    def test_ensure_1d_channel(self) -> None:
        """
        Verify channel extraction handles 1D and multi-channel 2D audio safely.
        """

        mono = np.array([0.1, 0.2, 0.3])
        extracted_mono = ensure_1d_channel(mono, channel_idx=0)
        np.testing.assert_allclose(mono, extracted_mono)

        stereo = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        ch0 = ensure_1d_channel(stereo, channel_idx=0)
        ch1 = ensure_1d_channel(stereo, channel_idx=1)
        ch_clamped = ensure_1d_channel(stereo, channel_idx=99)

        np.testing.assert_allclose(ch0, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(ch1, [0.9, 0.8, 0.7])
        np.testing.assert_allclose(ch_clamped, [0.9, 0.8, 0.7])

        with self.assertRaises(ValueError):
            ensure_1d_channel(np.zeros((2, 2, 2)))

    def test_compute_stft_and_log_spectrogram(self) -> None:
        """
        Verify STFT shape consistency and log power decibel scaling.
        """

        freqs, times, zxx = compute_stft(self.tone_a, self.sample_rate, n_fft=512, hop_length=128)
        self.assertEqual(len(freqs), 257)
        self.assertGreater(len(times), 10)
        self.assertEqual(zxx.shape, (257, len(times)))

        spec_db = compute_log_spectrogram(zxx, top_db=60.0)
        self.assertEqual(spec_db.shape, zxx.shape)
        self.assertLessEqual(float(np.max(spec_db)) - float(np.min(spec_db)), 60.01)

    def test_compute_power_spectral_density(self) -> None:
        """
        Verify Welch PSD produces peak near fundamental frequency.
        """

        freqs, psd = compute_power_spectral_density(self.tone_a, self.sample_rate, n_per_seg=1024)
        peak_freq = freqs[np.argmax(psd)]
        self.assertAlmostEqual(peak_freq, 440.0, delta=20.0)

    def test_compute_spectral_features(self) -> None:
        """
        Verify spectral centroid, rolloff, flatness, and dynamic range properties.
        """

        freqs, _, zxx = compute_stft(self.tone_a, self.sample_rate, n_fft=512, hop_length=128)
        features: SpectralFeatures = compute_spectral_features(zxx, freqs, self.tone_a)

        # Centroid for a 440 Hz pure tone should be centered near 440 Hz
        self.assertGreater(features.spectral_centroid_hz, 350.0)
        self.assertLess(features.spectral_centroid_hz, 550.0)
        self.assertGreaterEqual(features.spectral_rolloff_hz, 400.0)
        # Flatness of pure tone is close to 0 (harmonic / tonal, not noise-like)
        self.assertLess(features.spectral_flatness, 0.2)
        self.assertGreater(features.crest_factor_db, 2.0)

    def test_compute_si_sdr(self) -> None:
        """
        Verify SI-SDR properties: perfect reconstruction, scale invariance, and noise degradation.
        """

        # Identical signal = very high SI-SDR
        perfect_score = compute_si_sdr(self.tone_a, self.tone_a)
        self.assertGreaterEqual(perfect_score, 80.0)

        # Arbitrary scale factor: SI-SDR is scale-invariant!
        scaled_score = compute_si_sdr(self.tone_a, self.tone_a * 3.7)
        self.assertGreaterEqual(scaled_score, 80.0)

        # Inverted phase is also an orthogonal collinear vector:
        inv_score = compute_si_sdr(self.tone_a, -self.tone_a * 0.5)
        self.assertGreaterEqual(inv_score, 80.0)

        # Interference from orthogonal frequency tone degrades SI-SDR
        noisy = self.tone_a + self.tone_b
        degraded_score = compute_si_sdr(self.tone_a, noisy)
        self.assertLess(degraded_score, 10.0)
        self.assertGreater(degraded_score, -5.0)

        # Silent signal
        silent_score = compute_si_sdr(np.zeros_like(self.tone_a), self.tone_a)
        self.assertEqual(silent_score, 0.0)

    def test_compute_spectral_correlation(self) -> None:
        """
        Verify Pearson correlation between magnitude spectrograms.
        """

        _, _, zxx_a = compute_stft(self.tone_a, self.sample_rate)
        _, _, zxx_b = compute_stft(self.tone_b, self.sample_rate)

        mag_a = np.abs(zxx_a)
        mag_b = np.abs(zxx_b)
        active_all = np.ones(mag_a.shape[1], dtype=bool)

        # Identical magnitudes correlate at 1.0
        corr_self = compute_spectral_correlation(mag_a, mag_a, active_all)
        self.assertAlmostEqual(corr_self, 1.0, places=3)

        # Inactive mask returns 0.0
        corr_inactive = compute_spectral_correlation(
            mag_a,
            mag_b,
            np.zeros(mag_a.shape[1], dtype=bool),
        )
        self.assertEqual(corr_inactive, 0.0)

    def test_analyze_speaker_in_mix_solo(self) -> None:
        """
        Verify solo speaker in mix produces 100% recoverable energy and 0% destroyed.
        """

        stems = {"speaker_1": self.tone_a}
        mix = self.tone_a.copy()

        metrics, mask, recovered = analyze_speaker_in_mix(
            target_stem=self.tone_a,
            all_stems=stems,
            mix_audio=mix,
            target_id="speaker_1",
            sample_rate=self.sample_rate,
        )

        self.assertEqual(metrics.speaker_id, "speaker_1")
        self.assertGreaterEqual(metrics.recoverable_energy_fraction, 0.95)
        self.assertLessEqual(metrics.destroyed_energy_fraction, 0.05)
        self.assertEqual(metrics.overlap_speech_fraction, 0.0)
        self.assertEqual(metrics.solo_speech_fraction, 1.0)
        self.assertGreater(metrics.unprocessed_si_sdr_db, 50.0)
        self.assertEqual(mask.shape[0], 513)
        self.assertGreater(len(recovered), 0)

    def test_analyze_speaker_in_mix_two_speakers(self) -> None:
        """
        Verify two overlapping speakers reflect collision in recoverability and overlap metrics.
        """

        stems = {
            "speaker_1": self.tone_a,
            "speaker_2": self.tone_b,
        }
        mix = self.tone_a + self.tone_b

        metrics, _, _ = analyze_speaker_in_mix(
            target_stem=self.tone_a,
            all_stems=stems,
            mix_audio=mix,
            target_id="speaker_1",
            sample_rate=self.sample_rate,
        )

        # Both speakers are active simultaneously -> overlap should be near 1.0
        self.assertGreaterEqual(metrics.overlap_speech_fraction, 0.9)
        # Because of spectral separation (440 Hz vs 1000 Hz), IRM can separate most energy!
        self.assertGreaterEqual(metrics.recoverable_energy_fraction, 0.5)
        # Oracle recovered SI-SDR should be significantly higher than unprocessed mix SI-SDR
        self.assertGreater(metrics.oracle_recovered_si_sdr_db, metrics.unprocessed_si_sdr_db)
        self.assertGreater(metrics.sdr_improvement_db, 5.0)

    def test_evaluate_dataset_sample(self) -> None:
        """
        Verify complete sample evaluation report generation and dictionary serialization.
        """

        stereo_mix = np.column_stack([self.tone_a + self.tone_b, self.tone_a])
        stems = {
            "speaker_1": np.column_stack([self.tone_a, self.tone_a]),
            "speaker_2": np.column_stack([self.tone_b, self.tone_b * 0.5]),
        }

        report: SampleAnalysisReport = evaluate_dataset_sample(
            sample_id="test_sample_001",
            mix_audio=stereo_mix,
            speaker_stems=stems,
            sample_rate=self.sample_rate,
            channel_idx=0,
        )

        self.assertEqual(report.sample_id, "test_sample_001")
        self.assertEqual(report.channels, 2)
        self.assertEqual(len(report.speaker_metrics), 2)
        self.assertGreaterEqual(report.mean_recoverable_fraction, 0.5)

        data_dict = report.to_dict()
        self.assertIn("speaker_metrics", data_dict)
        self.assertIn("overall_mix_features", data_dict)
        self.assertEqual(data_dict["sample_id"], "test_sample_001")


if __name__ == "__main__":
    unittest.main()
