"""
Paradigm 1: Diarize-Then-Transcribe Pipeline using Silero VAD and Faster-Whisper.
"""

# Import Modules
from pathlib import Path
from typing import Any, cast
import logging
import numpy as np
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering

from paradigms.common.base_pipeline import (
    BaseParadigmPipeline,
    PipelineOutput,
    TurnHypothesis,
)
from paradigms.common.weights_manager import WeightsManager
from paradigms.common.streaming_feeder import StreamingAudioFeeder

logger: logging.Logger = logging.getLogger(__name__)


class SileroWhisperPipeline(BaseParadigmPipeline):
    """
    Temporal segmentation pipeline: Silero VAD -> Spectral Clustering -> Whisper ASR.
    """

    def __init__(
        self,
        weights_manager: WeightsManager | None = None,
        whisper_model_size: str = "tiny",
        vad_threshold: float = 0.5,
        min_speech_duration_s: float = 0.3,
        min_silence_duration_s: float = 0.3,
    ) -> None:
        """
        Initialize the Silero + Whisper pipeline.

        Args:
            weights_manager (WeightsManager | None): Shared model cache manager.
            whisper_model_size (str): Whisper size ('tiny', 'base', 'small').
            vad_threshold (float): Speech probability threshold.
            min_speech_duration_s (float): Minimum speech chunk length in seconds.
            min_silence_duration_s (float): Minimum silence separating turns in seconds.
        """

        super().__init__(
            name="Silero-VAD + Clustering + Faster-Whisper",
            paradigm_id="diarize_then_transcribe",
        )
        self.weights = weights_manager if weights_manager is not None else WeightsManager()
        self.whisper_model_size: str = whisper_model_size
        self.vad_threshold: float = vad_threshold
        self.min_speech_duration_s: float = min_speech_duration_s
        self.min_silence_duration_s: float = min_silence_duration_s

        self.vad: Any = None
        self.whisper_model: Any = None

    def load_weights(self) -> None:
        """
        Ensure Silero VAD and Whisper weights are loaded into memory.
        """

        if not self.weights_loaded:
            self.vad = self.weights.ensure_silero_vad_wrapper()
            self.whisper_model = self.weights.ensure_whisper_model(self.whisper_model_size)
            self.weights_loaded = True

    def _run_silero_vad(
        self,
        audio_1d: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[tuple[float, float]]:
        """
        Segment audio into speech intervals using Silero VAD wrapper.

        Args:
            audio_1d (np.ndarray): 1D float32 audio signal.
            sample_rate (int): Audio sample rate.

        Returns:
            list[tuple[float, float]]: List of (start_s, end_s) speech intervals.
        """

        if self.vad is None:
            self.vad = self.weights.ensure_silero_vad_wrapper(sample_rate=sample_rate)

        intervals = self.vad.get_speech_intervals(
            audio_1d=audio_1d,
            threshold=self.vad_threshold,
            min_speech_duration_s=self.min_speech_duration_s,
            min_silence_duration_s=self.min_silence_duration_s,
        )
        return cast(list[tuple[float, float]], intervals)

    def _cluster_segments(
        self,
        audio_1d: np.ndarray,
        intervals: list[tuple[float, float]],
        sample_rate: int = 16000,
        max_speakers: int = 6,
    ) -> list[str]:
        """
        Cluster speech intervals into speaker identities using MFCC spectral representations.

        Args:
            audio_1d (np.ndarray): Full audio waveform.
            intervals (list[tuple[float, float]]): Speech intervals.
            sample_rate (int): Sampling frequency.
            max_speakers (int): Maximum cluster count bound.

        Returns:
            list[str]: Speaker label mapped to each segment (e.g. 'speaker_0').
        """

        if not intervals:
            return []

        features: list[np.ndarray] = []
        for s_s, e_s in intervals:
            s_samp = int(s_s * sample_rate)
            e_samp = int(e_s * sample_rate)
            slice_audio = audio_1d[s_samp:e_samp]
            if len(slice_audio) < 256:
                feat = np.zeros(20, dtype=np.float32)
            else:
                # Fast spectral profile
                fft_mag = np.abs(np.fft.rfft(slice_audio[:2048]))
                feat = np.log1p(fft_mag[:20])
            features.append(feat)

        feat_mat = np.array(features)
        n_samples = len(intervals)

        if n_samples == 1:
            return ["speaker_0"]

        n_clusters = min(max_speakers, max(2, int(np.sqrt(n_samples))))
        n_clusters = min(n_clusters, n_samples)

        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(feat_mat)
        return [f"speaker_{lbl}" for lbl in labels]

    def process_offline(
        self,
        audio_path: Path,
        sample_rate: int = 16000,
    ) -> PipelineOutput:
        """
        Execute full offline pipeline on input audio file.

        Args:
            audio_path (Path): Target audio file.
            sample_rate (int): Audio sample rate.

        Returns:
            PipelineOutput: Output containing predicted turns and layer latencies.
        """

        self.load_weights()
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data[0] if data.shape[0] < data.shape[1] else data[:, 0]

        total_dur_s: float = len(data) / float(sr)
        output = PipelineOutput(
            session_id=audio_path.stem,
            paradigm_id=self.paradigm_id,
            pipeline_name=self.name,
            audio_duration_s=total_dur_s,
        )

        output.profiler.start_pipeline()

        # Layer 1: VAD
        with output.profiler.record("vad"):
            intervals = self._run_silero_vad(data, sample_rate=sr)

        # Layer 2: Diarization & Clustering
        with output.profiler.record("diarization"):
            speaker_labels = self._cluster_segments(data, intervals, sample_rate=sr)

        # Layer 3: ASR
        with output.profiler.record("asr"):
            for (s_s, e_s), spk in zip(intervals, speaker_labels):
                s_samp = int(s_s * sr)
                e_samp = int(e_s * sr)
                chunk = data[s_samp:e_samp]
                if len(chunk) < int(0.2 * sr):
                    continue

                segments, _ = self.whisper_model.transcribe(
                    chunk,
                    beam_size=1,
                    language=None,
                    vad_filter=False,
                )
                text = " ".join([seg.text.strip() for seg in segments]).strip()
                if text:
                    output.turns.append(
                        TurnHypothesis(
                            speaker_id=spk,
                            start_s=s_s,
                            end_s=e_s,
                            text=text,
                            confidence=0.9,
                        )
                    )

        output.profiler.end_pipeline()
        return output

    def process_streaming(
        self,
        feeder: StreamingAudioFeeder,
    ) -> PipelineOutput:
        """
        Execute simulated real-time streaming pipeline.

        Args:
            feeder (StreamingAudioFeeder): Active streaming audio feeder.

        Returns:
            PipelineOutput: Final session output with streaming latency metrics.
        """

        self.load_weights()
        output = PipelineOutput(
            session_id="streaming_session",
            paradigm_id=self.paradigm_id,
            pipeline_name=self.name,
            audio_duration_s=feeder.total_duration_s,
        )

        output.profiler.start_pipeline()
        speech_buffer: list[np.ndarray] = []
        in_speech = False
        turn_start_s = 0.0

        for chunk in feeder.iter_chunks():
            with output.profiler.record("buffering"):
                chunk_1d = chunk.audio if chunk.audio.ndim == 1 else chunk.audio[0]

            with output.profiler.record("vad"):
                # Sub-frame VAD check
                chunk_vad = self._run_silero_vad(chunk_1d, sample_rate=chunk.sample_rate)
                chunk_has_speech = len(chunk_vad) > 0

            if chunk_has_speech:
                if not in_speech:
                    in_speech = True
                    turn_start_s = chunk.start_s
                speech_buffer.append(chunk_1d)
            else:
                if in_speech and speech_buffer:
                    # Silence detected: finalize and transcribe turn
                    turn_audio = np.concatenate(speech_buffer)
                    turn_end_s = chunk.end_s

                    with output.profiler.record("asr"):
                        segments, _ = self.whisper_model.transcribe(
                            turn_audio,
                            beam_size=1,
                            language=None,
                        )
                        text = " ".join([s.text.strip() for s in segments]).strip()

                    if text:
                        output.turns.append(
                            TurnHypothesis(
                                speaker_id="speaker_live",
                                start_s=turn_start_s,
                                end_s=turn_end_s,
                                text=text,
                                confidence=0.85,
                            )
                        )

                    speech_buffer.clear()
                    in_speech = False

        # Process any trailing speech
        if speech_buffer:
            turn_audio = np.concatenate(speech_buffer)
            with output.profiler.record("asr"):
                segments, _ = self.whisper_model.transcribe(turn_audio, beam_size=1)
                text = " ".join([s.text.strip() for s in segments]).strip()
            if text:
                output.turns.append(
                    TurnHypothesis(
                        speaker_id="speaker_live",
                        start_s=turn_start_s,
                        end_s=feeder.total_duration_s,
                        text=text,
                        confidence=0.85,
                    )
                )

        output.profiler.end_pipeline()
        return output
