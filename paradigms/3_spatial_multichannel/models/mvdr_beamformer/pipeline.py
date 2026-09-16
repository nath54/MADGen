"""
Paradigm 3: Multi-Channel Spatial Array Processing (MVDR Beamforming) + Faster-Whisper Pipeline.
Pure DSP mathematical array processing (requires 0MB model weights download).
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


class MVDRBeamformerPipeline(BaseParadigmPipeline):
    """
    Spatial array beamforming (Delay-and-Sum & MVDR) followed by Silero VAD and Faster-Whisper.
    """

    def __init__(
        self,
        weights_manager: WeightsManager | None = None,
        whisper_model_size: str = "tiny",
        array_radius_m: float = 0.05,
        speed_of_sound: float = 343.0,
        num_spatial_beams: int = 4,
    ) -> None:
        """
        Initialize the MVDR beamformer pipeline.

        Args:
            weights_manager (WeightsManager | None): Shared model cache manager.
            whisper_model_size (str): Faster-Whisper model size.
            array_radius_m (float): Circular mic array radius in meters (MADGen default: 0.05m).
            speed_of_sound (float): Acoustic wave velocity in m/s.
            num_spatial_beams (int): Number of angular spatial beams to steer around 360 degrees.
        """

        super().__init__(
            name="Spatial MVDR Beamformer + Faster-Whisper",
            paradigm_id="spatial_multichannel",
        )
        self.weights = weights_manager if weights_manager is not None else WeightsManager()
        self.whisper_model_size: str = whisper_model_size
        self.array_radius_m: float = array_radius_m
        self.speed_of_sound: float = speed_of_sound
        self.num_spatial_beams: int = num_spatial_beams

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

    def _compute_steering_vector(
        self,
        freqs: np.ndarray,
        azimuth_rad: float,
        num_channels: int,
    ) -> np.ndarray:
        """
        Compute theoretical array steering vectors for a circular microphone array.

        Args:
            freqs (np.ndarray): Frequency vector in Hz [F].
            azimuth_rad (float): Target steering angle in radians.
            num_channels (int): Number of microphone channels M.

        Returns:
            np.ndarray: Complex steering vectors of shape [num_channels, F].
        """

        mic_angles = np.linspace(0, 2 * np.pi, num_channels, endpoint=False)
        delays = (self.array_radius_m / self.speed_of_sound) * np.cos(azimuth_rad - mic_angles)
        # Phase shift: exp(-j * 2 * pi * f * delay)
        # Shape: [M, 1] * [1, F] -> [M, F]
        phase = -2.0 * np.pi * np.outer(delays, freqs)
        steering = np.exp(1j * phase)
        # Normalize: d^H * d = M
        steering /= np.sqrt(num_channels)
        return steering

    def _beamform_multichannel(
        self,
        audio_multi: np.ndarray,
        sample_rate: int = 16000,
        nperseg: int = 512,
    ) -> list[tuple[float, np.ndarray]]:
        """
        Apply spatial MVDR / Delay-and-Sum beamforming across angular directions.

        Args:
            audio_multi (np.ndarray): Multi-channel audio array [M, samples].
            sample_rate (int): Sampling rate.
            nperseg (int): STFT window size.

        Returns:
            list[tuple[float, np.ndarray]]: List of (azimuth_deg, beamformed_audio_1d).
        """

        num_channels, num_samples = audio_multi.shape
        if num_channels == 1:
            return [(0.0, audio_multi[0])]

        # STFT on all channels: [M, F, T]
        f_vec, _, z_first = scipy.signal.stft(audio_multi[0], fs=sample_rate, nperseg=nperseg)
        num_freqs = len(f_vec)
        num_frames = z_first.shape[-1]
        z_multi = np.zeros((num_channels, num_freqs, num_frames), dtype=np.complex64)
        z_multi[0] = z_first

        for m in range(1, num_channels):
            _, _, z_m = scipy.signal.stft(audio_multi[m], fs=sample_rate, nperseg=nperseg)
            z_multi[m] = z_m

        # Compute sample covariance matrix per frequency bin: R[f] = 1/T * Z_multi[f] * Z_multi[f]^H
        # Regularization diagonal loading
        diag_reg = 1e-4 * np.eye(num_channels, dtype=np.complex64)
        cov_inv = np.zeros((num_freqs, num_channels, num_channels), dtype=np.complex64)

        for f_idx in range(num_freqs):
            # Shape [M, T]
            z_f = z_multi[:, f_idx, :]
            r_f = (z_f @ z_f.conj().T) / float(num_frames) + diag_reg
            cov_inv[f_idx] = np.linalg.pinv(r_f)

        # Steer beams around the circle
        beam_angles_deg = np.linspace(0, 360, self.num_spatial_beams, endpoint=False)
        beamformed_streams: list[tuple[float, np.ndarray]] = []

        for angle_deg in beam_angles_deg:
            angle_rad = float(np.deg2rad(angle_deg))
            steering = self._compute_steering_vector(f_vec, angle_rad, num_channels)

            # MVDR weights: w[f] = (R^-1 * d) / (d^H * R^-1 * d)
            # Shape: [M, F]
            z_beam = np.zeros((num_freqs, num_frames), dtype=np.complex64)

            for f_idx in range(num_freqs):
                d_f = steering[:, f_idx]  # [M]
                r_inv_f = cov_inv[f_idx]  # [M, M]
                num = r_inv_f @ d_f       # [M]
                denom = np.real(d_f.conj().T @ num) + 1e-8
                w_f = num / denom          # [M]
                # Apply beamforming weight vector: y[f, t] = w^H * x[f, t]
                z_beam[f_idx, :] = w_f.conj().T @ z_multi[:, f_idx, :]

            # Invert back to time domain
            _, wav_beam = scipy.signal.istft(z_beam, fs=sample_rate, nperseg=nperseg)
            wav_beam = wav_beam[:num_samples].astype(np.float32)
            beamformed_streams.append((angle_deg, wav_beam))

        return beamformed_streams

    def process_offline(
        self,
        audio_path: Path,
        sample_rate: int = 16000,
    ) -> PipelineOutput:
        """
        Execute spatial multi-channel beamforming and Whisper transcription.

        Args:
            audio_path (Path): Path to multi-channel mixed WAV.
            sample_rate (int): Sampling rate.

        Returns:
            PipelineOutput: Output containing predicted turns and latencies.
        """

        self.load_weights()
        data, sr = sf.read(str(audio_path), dtype="float32")
        # Ensure shape [channels, samples]
        if data.ndim == 1:
            multi_data = data[np.newaxis, :]
        elif data.shape[0] > data.shape[1]:
            multi_data = data.T
        else:
            multi_data = data

        num_samples = multi_data.shape[-1]
        total_dur_s: float = num_samples / float(sr)

        output = PipelineOutput(
            session_id=audio_path.stem,
            paradigm_id=self.paradigm_id,
            pipeline_name=self.name,
            audio_duration_s=total_dur_s,
            metadata={"channels": multi_data.shape[0], "beams": self.num_spatial_beams},
        )

        output.profiler.start_pipeline()

        # Layer 1 & 2: Spatial Covariance Tracking and MVDR Beamforming
        with output.profiler.record("beamforming"):
            beam_streams = self._beamform_multichannel(multi_data, sample_rate=sr)

        # Layer 3: VAD and ASR on each spatial beam
        for angle_deg, beam_wav in beam_streams:
            spk_label = f"beam_{int(angle_deg)}deg"

            with output.profiler.record("vad"):
                intervals = self.vad_wrapper.get_speech_intervals(
                    beam_wav,
                    threshold=0.45,
                    min_speech_duration_s=0.25,
                    min_silence_duration_s=0.3,
                )

            with output.profiler.record("asr"):
                for s_s, e_s in intervals:
                    s_samp = int(s_s * sr)
                    e_samp = int(e_s * sr)
                    chunk = beam_wav[s_samp:e_samp]
                    if len(chunk) < int(0.25 * sr):
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
        Execute simulated real-time streaming spatial beamforming pipeline.
        """

        self.load_weights()
        output = PipelineOutput(
            session_id="streaming_session_spatial",
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
                chunk_audio = chunk.audio
                lead_ch = chunk_audio[0] if chunk_audio.ndim > 1 else chunk_audio

            with output.profiler.record("vad"):
                prob = self.vad_wrapper.step(lead_ch[:512])
                has_speech = prob > 0.5

            if has_speech:
                if not in_speech:
                    in_speech = True
                    turn_start_s = chunk.start_s
                speech_buffer.append(chunk_audio)
            else:
                if in_speech and speech_buffer:
                    if chunk_audio.ndim > 1:
                        turn_audio = np.concatenate(speech_buffer, axis=1)
                    else:
                        turn_audio = np.concatenate(speech_buffer)

                    turn_end_s = chunk.end_s

                    with output.profiler.record("beamforming"):
                        if turn_audio.ndim > 1:
                            beam_out = self._beamform_multichannel(turn_audio, sample_rate=chunk.sample_rate)
                        else:
                            beam_out = [(0.0, turn_audio)]

                    with output.profiler.record("asr"):
                        # Transcribe dominant beam
                        best_beam_wav = beam_out[0][1]
                        segments, _ = self.whisper_model.transcribe(best_beam_wav, beam_size=1)
                        text = " ".join([s.text.strip() for s in segments]).strip()

                    if text:
                        output.turns.append(
                            TurnHypothesis(
                                speaker_id="speaker_spatial",
                                start_s=turn_start_s,
                                end_s=turn_end_s,
                                text=text,
                                confidence=0.9,
                            )
                        )

                    speech_buffer.clear()
                    in_speech = False

        output.profiler.end_pipeline()
        return output
