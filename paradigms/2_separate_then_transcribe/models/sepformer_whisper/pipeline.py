"""
Paradigm 2: SpeechBrain SepFormer Neural Separation + Faster-Whisper Pipeline.
"""

# Import Modules
from pathlib import Path
from typing import Any
import logging
import numpy as np
import scipy.signal
import soundfile as sf
import torch

from paradigms.common.base_pipeline import (
    BaseParadigmPipeline,
    PipelineOutput,
    TurnHypothesis,
)
from paradigms.common.weights_manager import WeightsManager
from paradigms.common.streaming_feeder import StreamingAudioFeeder

logger: logging.Logger = logging.getLogger(__name__)


class SepformerWhisperPipeline(BaseParadigmPipeline):
    """
    Neural blind source separation with SpeechBrain SepFormer + Faster-Whisper.
    """

    def __init__(
        self,
        weights_manager: WeightsManager | None = None,
        whisper_model_size: str = "tiny",
    ) -> None:
        """
        Initialize SepFormer + Whisper pipeline.

        Args:
            weights_manager (WeightsManager | None): Shared model cache manager.
            whisper_model_size (str): Faster-Whisper model variation.
        """

        super().__init__(
            name="SpeechBrain SepFormer + Faster-Whisper",
            paradigm_id="separate_then_transcribe",
        )
        self.weights = weights_manager if weights_manager is not None else WeightsManager()
        self.whisper_model_size: str = whisper_model_size
        self.whisper_model: Any = None
        self.sepformer_model: Any = None
        self.vad_wrapper: Any = None

    def load_weights(self) -> None:
        """
        Ensure SepFormer and Whisper weights are loaded into memory.
        """

        if not self.weights_loaded:
            self.whisper_model = self.weights.ensure_whisper_model(self.whisper_model_size)
            self.vad_wrapper = self.weights.ensure_silero_vad_wrapper()
            # SpeechBrain SepFormer weights are downloaded on demand
            try:
                self.sepformer_model = self.weights.ensure_sepformer_model()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("SepFormer initialization deferred: %s", exc)
            self.weights_loaded = True

    def _separate_audio(
        self,
        audio_1d: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[np.ndarray]:
        """
        Separate mixed 1D audio into 2 speaker streams using SepFormer.

        Args:
            audio_1d (np.ndarray): Input mixture waveform.
            sample_rate (int): Sampling frequency.

        Returns:
            list[np.ndarray]: List of 2 separated 16kHz waveforms.
        """

        if self.sepformer_model is None:
            self.sepformer_model = self.weights.ensure_sepformer_model()

        # SepFormer WSJ02mix operates at 8kHz
        target_sr = 8000
        num_target_samples = int(len(audio_1d) * target_sr / sample_rate)
        audio_8k = scipy.signal.resample(audio_1d, num_target_samples)

        tensor_in = torch.from_numpy(audio_8k).float().unsqueeze(0)
        with torch.no_grad():
            est_sources = self.sepformer_model.separate_batch(tensor_in)
            # Shape: [1, samples, 2]
            est_sources = est_sources.squeeze(0).cpu().numpy()

        separated_streams: list[np.ndarray] = []
        for ch in range(est_sources.shape[-1]):
            source_8k = est_sources[:, ch]
            # Resample back to 16kHz for Whisper
            source_16k = scipy.signal.resample(source_8k, len(audio_1d))
            separated_streams.append(source_16k.astype(np.float32))

        return separated_streams

    def process_offline(
        self,
        audio_path: Path,
        sample_rate: int = 16000,
    ) -> PipelineOutput:
        """
        Execute SepFormer separation and Whisper transcription.

        Args:
            audio_path (Path): Path to mixed audio file.
            sample_rate (int): Audio sampling rate.

        Returns:
            PipelineOutput: Output containing predicted turns and latencies.
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

        # Layer 1: Neural Separation
        with output.profiler.record("separation"):
            separated_streams = self._separate_audio(data, sample_rate=sr)

        # Layer 2 & 3: Per-stream VAD & ASR
        for spk_idx, sep_wav in enumerate(separated_streams):
            spk_label = f"speaker_{spk_idx + 1}"

            with output.profiler.record("vad"):
                intervals = self.vad_wrapper.get_speech_intervals(
                    sep_wav,
                    threshold=0.4,
                    min_speech_duration_s=0.25,
                    min_silence_duration_s=0.3,
                )

            with output.profiler.record("asr"):
                for s_s, e_s in intervals:
                    s_samp = int(s_s * sr)
                    e_samp = int(e_s * sr)
                    chunk = sep_wav[s_samp:e_samp]
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
                                speaker_id=spk_label,
                                start_s=s_s,
                                end_s=e_s,
                                text=text,
                                confidence=0.9,
                            )
                        )

        output.turns.sort(key=lambda t: t.start_s)
        output.profiler.end_pipeline()
        return output

    def process_streaming(
        self,
        feeder: StreamingAudioFeeder,
    ) -> PipelineOutput:
        """
        Execute simulated real-time streaming separation pipeline.
        """

        self.load_weights()
        output = PipelineOutput(
            session_id="streaming_session_sepformer",
            paradigm_id=self.paradigm_id,
            pipeline_name=self.name,
            audio_duration_s=feeder.total_duration_s,
        )

        output.profiler.start_pipeline()
        speech_buffer: list[np.ndarray] = []
        turn_start_s = 0.0
        in_speech = False

        for chunk in feeder.iter_chunks():
            with output.profiler.record("buffering"):
                chunk_1d = chunk.audio if chunk.audio.ndim == 1 else chunk.audio[0]

            with output.profiler.record("vad"):
                prob = self.vad_wrapper.step(chunk_1d[:512])
                has_speech = prob > 0.5

            if has_speech:
                if not in_speech:
                    in_speech = True
                    turn_start_s = chunk.start_s
                speech_buffer.append(chunk_1d)
            else:
                if in_speech and speech_buffer:
                    turn_audio = np.concatenate(speech_buffer)
                    turn_end_s = chunk.end_s

                    with output.profiler.record("separation"):
                        # In streaming, run separation on isolated multi-speaker burst
                        sep_streams = self._separate_audio(turn_audio, sample_rate=chunk.sample_rate)

                    with output.profiler.record("asr"):
                        for idx, s_wav in enumerate(sep_streams):
                            segments, _ = self.whisper_model.transcribe(s_wav, beam_size=1)
                            text = " ".join([s.text.strip() for s in segments]).strip()
                            if text:
                                output.turns.append(
                                    TurnHypothesis(
                                        speaker_id=f"speaker_stream_{idx+1}",
                                        start_s=turn_start_s,
                                        end_s=turn_end_s,
                                        text=text,
                                        confidence=0.85,
                                    )
                                )

                    speech_buffer.clear()
                    in_speech = False

        output.profiler.end_pipeline()
        return output
