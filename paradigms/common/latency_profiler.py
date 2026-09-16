"""
High-resolution layer-by-layer execution time profiler and Real-Time Factor (RTF) tracker.
"""

# Import Modules
from typing import Iterator
from contextlib import contextmanager
import time


class LayerLatencyRecord:
    """
    Latency metrics collection for an individual processing layer.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize layer record.

        Args:
            name (str): Label identifying processing layer.
        """

        self.name: str = name
        self.durations_ms: list[float] = []

    def add_duration(self, duration_ms: float) -> None:
        """
        Record a single execution duration in milliseconds.

        Args:
            duration_ms (float): Execution time in milliseconds.
        """

        self.durations_ms.append(duration_ms)

    @property
    def total_ms(self) -> float:
        """
        Total cumulative execution time in milliseconds.
        """

        return sum(self.durations_ms)

    @property
    def mean_ms(self) -> float:
        """
        Mean execution duration in milliseconds.
        """

        if not self.durations_ms:
            return 0.0
        return self.total_ms / len(self.durations_ms)

    @property
    def count(self) -> int:
        """
        Number of times this layer was executed.
        """

        return len(self.durations_ms)


class LatencyProfiler:
    """
    Wall-clock latency profiler tracking execution times across pipeline stages.
    """

    def __init__(self) -> None:
        """
        Initialize latency profiler.
        """

        self.layers: dict[str, LayerLatencyRecord] = {}
        self.start_time: float = 0.0
        self.total_pipeline_time_ms: float = 0.0

    @contextmanager
    def record(self, layer_name: str) -> Iterator[None]:
        """
        Context manager to record elapsed wall-clock time for a named processing layer.

        Args:
            layer_name (str): Processing layer identifier (e.g. 'vad', 'asr', 'diarization').

        Yields:
            None
        """

        if layer_name not in self.layers:
            self.layers[layer_name] = LayerLatencyRecord(layer_name)

        t_start: float = time.perf_counter()
        try:
            yield
        finally:
            t_end: float = time.perf_counter()
            elapsed_ms: float = (t_end - t_start) * 1000.0
            self.layers[layer_name].add_duration(elapsed_ms)

    def start_pipeline(self) -> None:
        """
        Mark beginning of overall end-to-end pipeline execution.
        """

        self.start_time = time.perf_counter()

    def end_pipeline(self) -> float:
        """
        Mark completion of overall pipeline and return total elapsed milliseconds.

        Returns:
            float: Total elapsed time in milliseconds.
        """

        if self.start_time > 0:
            self.total_pipeline_time_ms = (time.perf_counter() - self.start_time) * 1000.0
        else:
            self.total_pipeline_time_ms = sum(rec.total_ms for rec in self.layers.values())
        return self.total_pipeline_time_ms

    def get_layer_totals_ms(self) -> dict[str, float]:
        """
        Return mapping of layer names to their total cumulative execution times.

        Returns:
            dict[str, float]: Total execution time per layer in milliseconds.
        """

        return {k: v.total_ms for k, v in self.layers.items()}

    def compute_rtf(self, audio_duration_s: float) -> float:
        """
        Compute Real-Time Factor (RTF).

        RTF = (Total Computation Time in Seconds) / (Audio Duration in Seconds).
        An RTF < 1.0 indicates faster-than-real-time execution.

        Args:
            audio_duration_s (float): Duration of processed audio in seconds.

        Returns:
            float: Computed Real-Time Factor.
        """

        if audio_duration_s <= 0:
            return 0.0
        total_compute_s: float = self.total_pipeline_time_ms / 1000.0
        return total_compute_s / audio_duration_s

    def summary_dict(self, audio_duration_s: float = 0.0) -> dict[str, object]:
        """
        Generate structured summary dictionary of latency statistics.

        Args:
            audio_duration_s (float): Total audio duration processed.

        Returns:
            dict[str, object]: Latency statistics dictionary.
        """

        layer_data: dict[str, dict[str, float]] = {}
        for name, rec in self.layers.items():
            pct = (
                (rec.total_ms / self.total_pipeline_time_ms * 100.0)
                if self.total_pipeline_time_ms > 0
                else 0.0
            )
            layer_data[name] = {
                "total_ms": round(rec.total_ms, 2),
                "mean_ms": round(rec.mean_ms, 2),
                "count": float(rec.count),
                "percentage": round(pct, 1),
            }

        rtf_val = self.compute_rtf(audio_duration_s) if audio_duration_s > 0 else 0.0

        return {
            "total_latency_ms": round(self.total_pipeline_time_ms, 2),
            "audio_duration_s": round(audio_duration_s, 2),
            "rtf": round(rtf_val, 3),
            "layers": layer_data,
        }
