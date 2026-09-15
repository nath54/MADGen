"""
Audio processing utilities including resampling, normalization, and WAV I/O.

Provides functions to manipulate audio waveforms, adjust sampling rates with
polyphase filtering, align utterances onto a global timeline, and export files.
"""

# Import Modules
from pathlib import Path

import math

import numpy as np
import soundfile as sf
import scipy.signal

from src.common.types import AudioArray
from src.common.constants import EPSILON


def calculate_gcd(
    val_a: int,
    val_b: int,
) -> int:
    """
    Compute greatest common divisor of two integers.

    Args:
        val_a (int): First integer.
        val_b (int): Second integer.

    Returns:
        int: Greatest common divisor.
    """

    # Compute greatest common divisor
    return math.gcd(val_a, val_b)


def resample_audio(
    audio_signal: AudioArray,
    orig_sr: int,
    target_sr: int,
) -> AudioArray:
    """
    Resample a single-channel audio signal to a target sample rate.

    Args:
        audio_signal (AudioArray): 1D array of input audio samples.
        orig_sr (int): Original sampling rate in Hertz.
        target_sr (int): Target sampling rate in Hertz.

    Returns:
        AudioArray: Resampled audio array.
    """

    # Check for identical sample rates
    if orig_sr == target_sr:
        return audio_signal.copy()

    # Calculate simplification factors for rational resampling
    common_divisor: int = calculate_gcd(orig_sr, target_sr)
    up_factor: int = target_sr // common_divisor
    down_factor: int = orig_sr // common_divisor

    # Perform polyphase filtering resampling
    resampled: AudioArray = scipy.signal.resample_poly(
        audio_signal,
        up=up_factor,
        down=down_factor,
    ).astype(np.float64)

    return resampled


def normalize_audio_peak(
    audio_signal: AudioArray,
    peak_target: float = 0.95,
) -> AudioArray:
    """
    Normalize audio array to a specified peak amplitude preventing clipping.

    Args:
        audio_signal (AudioArray): Audio samples to normalize.
        peak_target (float): Maximum desired absolute amplitude.

    Returns:
        AudioArray: Peak-normalized audio samples.
    """

    # Compute maximum absolute amplitude across signal
    max_val: float = float(np.max(np.abs(audio_signal)))

    # Prevent division by zero if audio is silent
    if max_val < EPSILON:
        return audio_signal.copy()

    # Scale signal to target peak
    scaling_factor: float = peak_target / (max_val + EPSILON)
    normalized: AudioArray = (audio_signal * scaling_factor).astype(np.float64)

    return normalized


def place_audio_on_timeline(
    timeline: AudioArray,
    audio_clip: AudioArray,
    start_sample: int,
) -> AudioArray:
    """
    Add an audio clip onto a timeline starting at a specific sample index.

    Args:
        timeline (AudioArray): Full timeline audio buffer.
        audio_clip (AudioArray): Audio segment to add.
        start_sample (int): Zero-indexed insertion sample point.

    Returns:
        AudioArray: Updated timeline audio buffer.
    """

    # Calculate end boundary on timeline
    end_sample: int = start_sample + len(audio_clip)

    # Add clip samples into timeline buffer
    timeline[start_sample:end_sample] += audio_clip

    return timeline


def write_wav_file(
    output_path: Path,
    audio_data: AudioArray,
    sample_rate: int,
) -> None:
    """
    Save floating-point audio data to a WAV file.

    Args:
        output_path (Path): Path to output file.
        audio_data (AudioArray): Multi-channel or mono audio array.
        sample_rate (int): Audio sampling frequency in Hertz.
    """

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write audio data to disk in 32-bit floating point format
    sf.write(str(output_path), audio_data, sample_rate, subtype="FLOAT")


def generate_synthetic_fallback_tone(
    duration_s: float,
    sample_rate: int,
    base_freq: float = 220.0,
) -> AudioArray:
    """
    Generate synthetic speech-like harmonic tone for offline testing.

    Args:
        duration_s (float): Duration in seconds.
        sample_rate (int): Sampling frequency in Hertz.
        base_freq (float): Fundamental frequency in Hertz.

    Returns:
        AudioArray: Synthesized tone with speech envelope.
    """

    # Generate time grid
    num_samples: int = int(duration_s * sample_rate)
    time_points: AudioArray = np.linspace(
        0.0,
        duration_s,
        num_samples,
        endpoint=False,
        dtype=np.float64,
    )

    # Compute harmonic carrier waves simulating vocal cords
    carrier: AudioArray = (
        0.5 * np.sin(2.0 * np.pi * base_freq * time_points)
        + 0.3 * np.sin(4.0 * np.pi * base_freq * time_points)
        + 0.2 * np.sin(6.0 * np.pi * base_freq * time_points)
    )

    # Apply pseudo-syllabic amplitude modulation at 4 Hz
    modulation: AudioArray = 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * time_points))

    # Apply smooth Hann envelope to eliminate edge clicks
    envelope: AudioArray = np.sin(np.pi * time_points / duration_s) ** 2

    # Combine carrier, modulation, and window envelope
    synthetic_signal: AudioArray = carrier * modulation * envelope

    return normalize_audio_peak(synthetic_signal, peak_target=0.8)
