"""
Paradigm 2: Oracle Ideal Ratio Mask (IRM) Separation + Faster-Whisper Pipeline.
Provides an upper-bound baseline for time-frequency domain source separation.
"""

# Import Modules
from pathlib import Path
from typing import Any
import logging
import numpy as np
import scipy.signal
import soundfile as sf

from paradigms.common.base_pipeline import (
    BaseParadigmPipeline,
    PipelineOutput,
    TurnHypothesis,
)
from paradigms.common.weights_manager import WeightsManager
from paradigms.common.streaming_feeder import StreamingAudioFeeder

logger: logging.Logger = logging.getLogger(__name__)


class OracleIRMWhisperPipeline(BaseParadigmPipeline):
    """
    Oracle Ideal Ratio Mask (IRM) source separation followed by independent Whisper transcription.
    """

    def __init__(
        self,
        weights_manager: WeightsManager | None = None,
        whisper_model_size: str = "tiny",
    ) -> None:
        """
        Initialize the Oracle IRM + Whisper pipeline.

        Args:
            weights_manager (WeightsManager | None): Shared model cache manager.
            whisper_model_size (str): Faster-Whisper model variation.
        """

        super().__init__(
            name="Oracle-IRM Separation + Faster-Whisper",
            paradigm_id="separate_then_transcribe",
        )
        self.weights = weights_manager if weights_manager is not None else WeightsManager()
        self.whisper_model_size: str = whisper_model_size
        self.whisper_model: Any = None
        self.vad_wrapper: Any = None

    def load_weights(self) -> None:
        """
        Ensure Whisper model and VAD wrapper are loaded.
        """

        if not self.weights_loaded:
            self.whisper_model = self.weights.ensure_whisper_model(self.whisper_model_size)
            self.vad_wrapper = self.weights.ensure_silero_vad_wrapper()
            self.weights_loaded = True

    def _compute_oracle_irm_separation(
        self,
        mixed_audio: np.ndarray,
        clean_stems: list[np.ndarray],
        sample_rate: int = 16000,
        nperseg: int = 512,
        beta: float = 1.0,
    ) -> list[np.ndarray]:
        """
        Compute Ideal Ratio Mask (IRM) separation for each target speaker.

        Args:
            mixed_audio (np.ndarray): 1D mixed waveform.
            clean_stems (list[np.ndarray]): List of 1D clean reference speaker stems.
            sample_rate (int): Sampling frequency.
            nperseg (int): STFT window size.
            beta (float): IRM exponent (1.0 for linear magnitude, 2.0 for power).

        Returns:
            list[np.ndarray]: List of separated waveforms.
        """

        eps = 1e-8
        _, _, Z_mix = scipy.signal.stft(mixed_audio, fs=sample_rate, nperseg=nperseg)

        stem_stfts = []
        for stem in clean_stems:
            pad_len = max(0, len(mixed_audio) - len(stem))
            padded_stem = np.pad(stem, (0, pad_len))[: len(mixed_audio)]
            _, _, Z_stem = scipy.signal.stft(padded_stem, fs=sample_rate, nperseg=nperseg)
            stem_stfts.append(Z_stem)

        mags = [np.abs(Z) ** beta for Z in stem_stfts]
        total_mag = np.sum(mags, axis=0) + eps

        separated_waveforms: list[np.ndarray] = []
        for i, mag in enumerate(mags):
            mask = mag / total_mag
            Z_masked = mask * Z_mix
            _, wav_est = scipy.signal.istft(Z_masked, fs=sample_rate, nperseg=nperseg)
            wav_est = wav_est[: len(mixed_audio)]
            separated_waveforms.append(wav_est)

        return separated_waveforms

    def process_offline(
        self,
        audio_path: Path,
        sample_rate: int = 16000,
    ) -> PipelineOutput:
        """
        Execute Oracle IRM separation and Whisper ASR on target scene.

        Args:
            audio_path (Path): Path to mixed audio file.
            sample_rate (int): Expected audio sampling rate.

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

        # Load clean isolated stems from sample directory
        sample_dir = audio_path.parent
        stems_dir = sample_dir / "isolated_speakers"
        clean_stems: list[tuple[str, np.ndarray]] = []

        if stems_dir.is_dir():
            stem_paths = sorted(list(stems_dir.glob("speaker_*.mp3")) or list(stems_dir.glob("speaker_*.wav")))
            for stem_path in stem_paths:
                stem_wav, _ = sf.read(str(stem_path), dtype="float32")
                if stem_wav.ndim > 1:
                    stem_wav = stem_wav[0] if stem_wav.shape[0] < stem_wav.shape[1] else stem_wav[:, 0]
                clean_stems.append((stem_path.stem, stem_wav))

        output.profiler.start_pipeline()

        # Fallback if no stems found (e.g. standard mix without oracle)
        if not clean_stems:
            clean_stems = [("speaker_1", data)]

        stems_audio = [s[1] for s in clean_stems]
        stem_names = [s[0] for s in clean_stems]

        # Layer 1: Separation (Oracle IRM)
        with output.profiler.record("separation"):
            separated_audio = self._compute_oracle_irm_separation(
                mixed_audio=data,
                clean_stems=stems_audio,
                sample_rate=sr,
            )

        # Layer 2 & 3: Per-stream VAD & ASR
        for spk_name, sep_wav in zip(stem_names, separated_audio):
            # Layer: VAD on separated stream
            with output.profiler.record("vad"):
                intervals = self.vad_wrapper.get_speech_intervals(
                    sep_wav,
                    threshold=0.4,
                    min_speech_duration_s=0.25,
                    min_silence_duration_s=0.3,
                )

            # Layer: ASR on detected turns
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
                                speaker_id=spk_name,
                                start_s=s_s,
                                end_s=e_s,
                                text=text,
                                confidence=0.95,
                            )
                        )

        # Sort turns chronologically
        output.turns.sort(key=lambda t: t.start_s)
        output.profiler.end_pipeline()
        return output

    def process_streaming(
        self,
        feeder: StreamingAudioFeeder,
    ) -> PipelineOutput:
        """
        Execute simulated real-time streaming separation pipeline.

        Args:
            feeder (StreamingAudioFeeder): Active streaming audio feeder.

        Returns:
            PipelineOutput: Output containing predicted turns and latencies.
        """

        self.load_weights()
        output = PipelineOutput(
            session_id="streaming_session_p2",
            paradigm_id=self.paradigm_id,
            pipeline_name=self.name,
            audio_duration_s=feeder.total_duration_s,
        )

        output.profiler.start_pipeline()
        # For streaming simulation, accumulate chunks into active turns
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
                        # In streaming, pass-through separation on isolated turn
                        sep_wav = turn_audio

                    with output.profiler.record("asr"):
                        segments, _ = self.whisper_model.transcribe(sep_wav, beam_size=1)
                        text = " ".join([s.text.strip() for s in segments]).strip()

                    if text:
                        output.turns.append(
                            TurnHypothesis(
                                speaker_id="speaker_stream",
                                start_s=turn_start_s,
                                end_s=turn_end_s,
                                text=text,
                                confidence=0.9,
                            )
                        )

                    speech_buffer.clear()
                    in_speech = False

        if speech_buffer:
            turn_audio = np.concatenate(speech_buffer)
            with output.profiler.record("asr"):
                segments, _ = self.whisper_model.transcribe(turn_audio, beam_size=1)
                text = " ".join([s.text.strip() for s in segments]).strip()
            if text:
                output.turns.append(
                    TurnHypothesis(
                        speaker_id="speaker_stream",
                        start_s=turn_start_s,
                        end_s=feeder.total_duration_s,
                        text=text,
                        confidence=0.9,
                    )
                )

        output.profiler.end_pipeline()
        return output
