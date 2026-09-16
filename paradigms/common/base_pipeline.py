"""
Abstract base class and data interfaces for evaluation paradigm pipelines.
"""

# Import Modules
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paradigms.common.latency_profiler import LatencyProfiler
from paradigms.common.streaming_feeder import StreamingAudioFeeder


@dataclass
class TurnHypothesis:
    """
    Hypothesized speaker turn record from recognition and diarization pipeline.
    """

    speaker_id: str
    start_s: float
    end_s: float
    text: str
    confidence: float = 1.0


@dataclass
class PipelineOutput:
    """
    Structured outcome of an evaluation pipeline execution.
    """

    session_id: str
    paradigm_id: str
    pipeline_name: str
    audio_duration_s: float
    turns: list[TurnHypothesis] = field(default_factory=list)
    profiler: LatencyProfiler = field(default_factory=LatencyProfiler)
    der: float | None = None
    wer: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize result to JSON-compatible dictionary.
        """

        turns_data = [
            {
                "speaker_id": t.speaker_id,
                "start_s": round(t.start_s, 2),
                "end_s": round(t.end_s, 2),
                "text": t.text,
                "confidence": round(t.confidence, 3),
            }
            for t in self.turns
        ]

        return {
            "session_id": self.session_id,
            "paradigm_id": self.paradigm_id,
            "pipeline_name": self.pipeline_name,
            "audio_duration_s": round(self.audio_duration_s, 2),
            "der": round(self.der, 4) if self.der is not None else None,
            "wer": round(self.wer, 4) if self.wer is not None else None,
            "latency_summary": self.profiler.summary_dict(self.audio_duration_s),
            "turns_count": len(self.turns),
            "turns": turns_data,
            "metadata": self.metadata,
        }


class BaseParadigmPipeline(ABC):
    """
    Abstract interface governing multi-speaker detection, diarization, and ASR pipelines.
    """

    def __init__(self, name: str, paradigm_id: str) -> None:
        """
        Initialize base pipeline.

        Args:
            name (str): Human-readable pipeline identifier.
            paradigm_id (str): Canonical paradigm code.
        """

        self.name: str = name
        self.paradigm_id: str = paradigm_id
        self.weights_loaded: bool = False

    @abstractmethod
    def load_weights(self) -> None:
        """
        Ensure all neural weights required by this pipeline are downloaded and loaded.
        """

    @abstractmethod
    def process_offline(
        self,
        audio_path: Path,
        sample_rate: int = 16000,
    ) -> PipelineOutput:
        """
        Execute full-file offline multi-speaker recognition and diarization.

        Args:
            audio_path (Path): Path to input mixed WAV audio file.
            sample_rate (int): Expected audio sampling rate.

        Returns:
            PipelineOutput: Output containing predicted turns and layer latencies.
        """

    @abstractmethod
    def process_streaming(
        self,
        feeder: StreamingAudioFeeder,
    ) -> PipelineOutput:
        """
        Execute simulated real-time streaming smart assistant pipeline chunk-by-chunk.

        Args:
            feeder (StreamingAudioFeeder): Active streaming audio buffer feeder.

        Returns:
            PipelineOutput: Final session output with streaming latency metrics.
        """
