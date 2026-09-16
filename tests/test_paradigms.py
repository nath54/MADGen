"""
Unit tests for MADGen evaluation paradigms framework.
"""

# Import Modules
import unittest
import numpy as np

from paradigms.common.latency_profiler import LatencyProfiler
from paradigms.common.streaming_feeder import StreamingAudioFeeder
from paradigms.common.metrics_evaluator import (
    compute_frame_level_der,
    compute_wer_metrics,
    DiarizationMetrics,
    ASRMetrics,
)
from paradigms.common.weights_manager import WeightsManager, SileroVADWrapper
from paradigms import get_pipeline


class TestLatencyProfiler(unittest.TestCase):
    """
    Test microsecond layer latency profiler and RTF metrics.
    """

    def test_layer_recording(self) -> None:
        """Verify recording wall-clock time in named layers."""
        profiler = LatencyProfiler()
        profiler.start_pipeline()

        with profiler.record("vad"):
            # Dummy compute
            _ = sum(i for i in range(10000))

        with profiler.record("asr"):
            _ = sum(i * 2 for i in range(20000))

        total_ms = profiler.end_pipeline()
        self.assertGreater(total_ms, 0.0)

        layer_totals = profiler.get_layer_totals_ms()
        self.assertIn("vad", layer_totals)
        self.assertIn("asr", layer_totals)
        self.assertGreater(layer_totals["vad"], 0.0)
        self.assertGreater(layer_totals["asr"], 0.0)

        summary = profiler.summary_dict(audio_duration_s=10.0)
        self.assertIn("rtf", summary)
        self.assertIn("layers", summary)
        self.assertLess(float(str(summary["rtf"])), 1.0)


class TestStreamingAudioFeeder(unittest.TestCase):
    """
    Test streaming audio chunking.
    """

    def test_chunk_iteration_1d(self) -> None:
        """Verify 1D audio chunk slicing."""
        sr = 16000
        duration_s = 1.0
        audio = np.zeros(int(sr * duration_s), dtype=np.float32)
        chunk_ms = 200

        feeder = StreamingAudioFeeder(audio=audio, sample_rate=sr, chunk_duration_ms=chunk_ms)
        chunks = list(feeder.iter_chunks())

        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks[0].start_s, 0.0)
        self.assertAlmostEqual(chunks[0].end_s, 0.2, places=3)
        self.assertTrue(chunks[-1].is_last)

    def test_chunk_iteration_multichannel(self) -> None:
        """Verify multi-channel audio chunk slicing."""
        sr = 16000
        audio = np.zeros((4, sr), dtype=np.float32)
        feeder = StreamingAudioFeeder(audio=audio, sample_rate=sr, chunk_duration_ms=250)
        chunks = list(feeder.iter_chunks())

        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].audio.shape, (4, 4000))


class TestMetricsEvaluator(unittest.TestCase):
    """
    Test DER and WER evaluation metrics.
    """

    def test_der_identical_turns(self) -> None:
        """Identical reference and hypothesis should produce DER = 0.0."""
        ref = [(0.0, 2.0, "speaker_1"), (2.5, 4.0, "speaker_2")]
        hyp = [(0.0, 2.0, "hyp_a"), (2.5, 4.0, "hyp_b")]

        res: DiarizationMetrics = compute_frame_level_der(
            reference_intervals=ref,
            hypothesis_intervals=hyp,
            step_s=0.05,
        )
        self.assertAlmostEqual(res.der, 0.0, places=3)
        self.assertAlmostEqual(res.speaker_confusion_fraction, 0.0, places=3)

    def test_der_missed_speech(self) -> None:
        """Missing hypothesis turns should yield missed speech error."""
        ref = [(0.0, 2.0, "speaker_1")]
        hyp: list[tuple[float, float, str]] = []

        res: DiarizationMetrics = compute_frame_level_der(ref, hyp)
        self.assertGreater(res.der, 0.0)
        self.assertGreater(res.missed_speech_fraction, 0.0)

    def test_wer_calculation(self) -> None:
        """Verify Word Error Rate computation."""
        ref = "bonjour tout le monde"
        hyp = "bonjour le monde"
        res: ASRMetrics = compute_wer_metrics(ref, hyp)
        self.assertGreater(res.wer, 0.0)
        self.assertEqual(res.deletions, 1)

        identical = compute_wer_metrics("test phrase", "test phrase")
        self.assertEqual(identical.wer, 0.0)


class TestWeightsManager(unittest.TestCase):
    """
    Test weights manager and hardware detection.
    """

    def test_device_selection(self) -> None:
        """Verify device selection returns string and compute type."""
        wm = WeightsManager()
        dev = wm.get_best_device()
        self.assertIn(dev, ["cuda", "cpu"])
        comp = wm.get_whisper_compute_type(dev)
        self.assertIsInstance(comp, str)

    def test_silero_vad_wrapper(self) -> None:
        """Verify Silero VAD session and wrapper initialization."""
        wm = WeightsManager()
        vad = wm.ensure_silero_vad_wrapper()
        self.assertIsInstance(vad, SileroVADWrapper)

        # Test single step on dummy zeros
        dummy_chunk = np.zeros(512, dtype=np.float32)
        prob = vad.step(dummy_chunk)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


class TestPipelineInstantiation(unittest.TestCase):
    """
    Test factory instantiation of all 4 paradigms.
    """

    def test_get_all_paradigms(self) -> None:
        """Verify factory instantiates all 4 pipelines."""
        for p in ["1", "2", "3", "4"]:
            pipe = get_pipeline(p)
            self.assertIsNotNone(pipe.name)
            self.assertIsNotNone(pipe.paradigm_id)


if __name__ == "__main__":
    unittest.main()
