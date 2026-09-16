"""
Streaming audio buffer chunker simulating real-time smart assistant microphone acquisition.
"""

# Import Modules
from typing import Iterator
import time
import numpy as np


class StreamingAudioChunk:
    """
    Data record for an individual streaming audio slice.
    """

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start_s: float,
        end_s: float,
        chunk_idx: int,
        is_last: bool = False,
    ) -> None:
        """
        Initialize streaming chunk container.

        Args:
            audio (np.ndarray): Audio array for this slice (1D or multi-channel).
            sample_rate (int): Audio sampling frequency in Hertz.
            start_s (float): Timestamp offset in seconds at start of chunk.
            end_s (float): Timestamp offset in seconds at end of chunk.
            chunk_idx (int): Sequential chunk index.
            is_last (bool): Whether this chunk terminates the stream.
        """

        self.audio: np.ndarray = audio
        self.sample_rate: int = sample_rate
        self.start_s: float = start_s
        self.end_s: float = end_s
        self.chunk_idx: int = chunk_idx
        self.is_last: bool = is_last

    @property
    def duration_s(self) -> float:
        """
        Duration of this chunk in seconds.
        """

        return self.end_s - self.start_s


class StreamingAudioFeeder:
    """
    Slices continuous audio into sequential chunks to simulate streaming hardware input.
    """

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 200,
        simulate_clock: bool = False,
    ) -> None:
        """
        Initialize streaming audio feeder.

        Args:
            audio (np.ndarray): Complete audio signal (1D [samples] or 2D [channels, samples]).
            sample_rate (int): Sampling rate in Hz.
            chunk_duration_ms (int): Duration of each streaming buffer in milliseconds.
            simulate_clock (bool): If True, sleeps between chunks to mimic real-time pace.
        """

        self.audio: np.ndarray = audio
        self.sample_rate: int = sample_rate
        self.chunk_duration_ms: int = chunk_duration_ms
        self.simulate_clock: bool = simulate_clock
        self.chunk_samples: int = int((chunk_duration_ms / 1000.0) * sample_rate)

        # Determine total sample count along time axis
        self.total_samples: int = audio.shape[-1]
        self.total_duration_s: float = self.total_samples / float(sample_rate)

    def iter_chunks(self) -> Iterator[StreamingAudioChunk]:
        """
        Iterate over streaming chunks in sequential chronological order.

        Yields:
            StreamingAudioChunk: Next chronological audio slice.
        """

        num_chunks: int = max(1, int(np.ceil(self.total_samples / self.chunk_samples)))

        for idx in range(num_chunks):
            start_sample: int = idx * self.chunk_samples
            end_sample: int = min(self.total_samples, start_sample + self.chunk_samples)

            if start_sample >= self.total_samples:
                break

            # Slice along last dimension
            if self.audio.ndim == 1:
                chunk_data = self.audio[start_sample:end_sample]
            else:
                chunk_data = self.audio[:, start_sample:end_sample]

            start_s: float = start_sample / float(self.sample_rate)
            end_s: float = end_sample / float(self.sample_rate)
            is_last: bool = (idx == num_chunks - 1) or (end_sample >= self.total_samples)

            if self.simulate_clock:
                chunk_dur: float = end_s - start_s
                time.sleep(chunk_dur)

            yield StreamingAudioChunk(
                audio=chunk_data,
                sample_rate=self.sample_rate,
                start_s=start_s,
                end_s=end_s,
                chunk_idx=idx,
                is_last=is_last,
            )
