"""
Paradigm 4: Real-Time Streaming Smart Assistant Pipeline.
Simulates live microphone stream processing with low-latency chunked VAD,
online speaker tracking, and real-time Whisper ASR.
"""

# Import Modules
from pathlib import Path
from typing import Any
import logging
import numpy as np
import soundfile as sf

from paradigms.common.base_pipeline import (
    BaseParadigmPipeline,
    PipelineOutput,
    TurnHypothesis,
)
from paradigms.common.weights_manager import WeightsManager
from paradigms.common.streaming_feeder import StreamingAudioFeeder

logger: logging.Logger = logging.getLogger(__name__)


class StreamingSmartAssistantPipeline(BaseParadigmPipeline):
    """
    Real-time simulated smart assistant pipeline processing chunked audio in chronological order.
    """

    def __init__(
        self,
        weights_manager: WeightsManager | None = None,
        whisper_model_size: str = "tiny",
        chunk_ms: int = 200,
        vad_threshold: float = 0.5,
        hangover_silence_chunks: int = 2,
    ) -> None:
        """
        Initialize the streaming smart assistant pipeline.

        Args:
            weights_manager (WeightsManager | None): Shared model cache manager.
            whisper_model_size (str): Faster-Whisper model variation.
            chunk_ms (int): Audio buffer ingestion chunk size in ms.
            vad_threshold (float): VAD speech activation threshold.
            hangover_silence_chunks (int): Number of silent chunks required to trigger end of turn.
        """

        super().__init__(
            name="Real-Time Streaming Assistant (Chunked VAD + Online ASR)",
            paradigm_id="realtime_streaming",
        )
        self.weights = weights_manager if weights_manager is not None else WeightsManager()
        self.whisper_model_size: str = whisper_model_size
        self.chunk_ms: int = chunk_ms
        self.vad_threshold: float = vad_threshold
        self.hangover_silence_chunks: int = hangover_silence_chunks

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

    def process_streaming(
        self,
        feeder: StreamingAudioFeeder,
    ) -> PipelineOutput:
        """
        Execute simulated real-time streaming smart assistant pipeline chunk-by-chunk.

        Args:
            feeder (StreamingAudioFeeder): Active streaming audio feeder.

        Returns:
            PipelineOutput: Output containing predicted turns and layer-by-layer latencies.
        """

        self.load_weights()
        self.vad_wrapper.reset_states()

        output = PipelineOutput(
            session_id="realtime_streaming_session",
            paradigm_id=self.paradigm_id,
            pipeline_name=self.name,
            audio_duration_s=feeder.total_duration_s,
            metadata={"chunk_duration_ms": feeder.chunk_duration_ms},
        )

        output.profiler.start_pipeline()

        speech_buffer: list[np.ndarray] = []
        turn_start_s = 0.0
        in_speech = False
        silence_chunks_count = 0
        active_speaker_idx = 1

        for chunk in feeder.iter_chunks():
            # Layer 1: Buffering & Acquisition
            with output.profiler.record("buffering"):
                if chunk.audio.ndim > 1:
                    chunk_1d = (
                        chunk.audio[:, 0]
                        if chunk.audio.shape[0] > chunk.audio.shape[1]
                        else chunk.audio[0]
                    )
                else:
                    chunk_1d = chunk.audio

            # Layer 2: Chunk VAD
            with output.profiler.record("vad"):
                # Subdivide chunk into 512-sample windows for VAD
                w_size = 512
                n_wins = max(1, len(chunk_1d) // w_size)
                chunk_probs = []
                for w_idx in range(n_wins):
                    sub_frame = chunk_1d[w_idx * w_size : (w_idx + 1) * w_size]
                    prob = self.vad_wrapper.step(sub_frame)
                    chunk_probs.append(prob)
                chunk_speech = max(chunk_probs) >= self.vad_threshold if chunk_probs else False

            # Layer 3: Online Speaker Tracking
            with output.profiler.record("speaker_tracking"):
                if chunk_speech:
                    if not in_speech:
                        in_speech = True
                        turn_start_s = chunk.start_s
                        # Alternate active speaker tracker state on new turns
                        active_speaker_idx = 1 if len(output.turns) % 2 == 0 else 2
                    silence_chunks_count = 0
                    speech_buffer.append(chunk_1d)
                else:
                    if in_speech:
                        silence_chunks_count += 1
                        speech_buffer.append(chunk_1d)

            # Layer 4: Turn Boundary Detection & ASR Dispatch
            if in_speech and silence_chunks_count >= self.hangover_silence_chunks:
                turn_audio = np.concatenate(speech_buffer)
                turn_end_s = chunk.end_s - (silence_chunks_count * chunk.duration_s)

                with output.profiler.record("asr"):
                    if len(turn_audio) >= int(0.3 * chunk.sample_rate):
                        segments, _ = self.whisper_model.transcribe(
                            turn_audio,
                            beam_size=1,
                            language=None,
                        )
                        text = " ".join([s.text.strip() for s in segments]).strip()
                    else:
                        text = ""

                if text:
                    output.turns.append(
                        TurnHypothesis(
                            speaker_id=f"speaker_{active_speaker_idx}",
                            start_s=turn_start_s,
                            end_s=turn_end_s,
                            text=text,
                            confidence=0.92,
                        )
                    )

                speech_buffer.clear()
                in_speech = False
                silence_chunks_count = 0

        # Flush any trailing speech at end of stream
        if in_speech and speech_buffer:
            turn_audio = np.concatenate(speech_buffer)
            with output.profiler.record("asr"):
                if len(turn_audio) >= int(0.3 * feeder.sample_rate):
                    segments, _ = self.whisper_model.transcribe(turn_audio, beam_size=1)
                    text = " ".join([s.text.strip() for s in segments]).strip()
                else:
                    text = ""
            if text:
                output.turns.append(
                    TurnHypothesis(
                        speaker_id=f"speaker_{active_speaker_idx}",
                        start_s=turn_start_s,
                        end_s=feeder.total_duration_s,
                        text=text,
                        confidence=0.92,
                    )
                )

        output.profiler.end_pipeline()
        return output

    def process_offline(
        self,
        audio_path: Path,
        sample_rate: int = 16000,
    ) -> PipelineOutput:
        """
        Execute streaming pipeline over offline audio file by feeding sequential chunks.
        """

        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data[:, 0] if data.shape[0] > data.shape[1] else data[0]

        feeder = StreamingAudioFeeder(
            audio=data,
            sample_rate=sr,
            chunk_duration_ms=self.chunk_ms,
        )
        output = self.process_streaming(feeder)
        output.session_id = audio_path.stem
        return output
