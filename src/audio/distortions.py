"""
Acoustic distortions and vocal effects for realistic speech generation.

Provides non-linear soft-clipping overdrive for shouting, amplitude tremolo
modulation for laughter, and ambient room noise injection.
"""

# Import Modules
import numpy as np

from src.common.types import AudioArray
from src.common.constants import EPSILON
from src.common.audio_utils import normalize_audio_peak


def apply_soft_clipping_overdrive(
    audio_signal: AudioArray,
    drive_gain: float = 1.6,
    target_peak: float = 0.95,
) -> AudioArray:
    """
    Apply non-linear soft-clipping saturation simulating vocal cord and mic overdrive.

    Args:
        audio_signal (AudioArray): Input audio samples.
        drive_gain (float): Preamplifier drive gain multiplier.
        target_peak (float): Peak level after normalization.

    Returns:
        AudioArray: Saturated and normalized audio waveform.
    """

    # Apply input pre-amp drive
    driven_signal: AudioArray = (audio_signal * drive_gain).astype(np.float64)

    # Apply hyperbolic tangent non-linear soft saturation curve
    saturated: AudioArray = np.tanh(driven_signal)

    # Restore normalized headroom
    return normalize_audio_peak(saturated, peak_target=target_peak)


def apply_laughter_modulation(
    audio_signal: AudioArray,
    sample_rate: int,
    modulation_freq: float = 5.0,
    depth: float = 0.4,
) -> AudioArray:
    """
    Apply pseudo-periodic amplitude tremolo mimicking laughing while speaking.

    Args:
        audio_signal (AudioArray): Input dry speech audio samples.
        sample_rate (int): Sampling frequency in Hertz.
        modulation_freq (float): Diaphragm laughter pulse frequency in Hertz (typically 4-6 Hz).
        depth (float): Tremolo modulation depth between 0.0 and 1.0.

    Returns:
        AudioArray: Laughter-modulated audio waveform.
    """

    # Calculate time grid for signal
    num_samples: int = len(audio_signal)
    time_points: AudioArray = np.linspace(
        0.0,
        num_samples / float(sample_rate),
        num_samples,
        endpoint=False,
        dtype=np.float64,
    )

    # Compute laughter amplitude envelope (4-6 Hz pulses)
    angular_freq: float = 2.0 * np.pi * modulation_freq
    tremolo_envelope: AudioArray = 1.0 - depth * (0.5 + 0.5 * np.sin(angular_freq * time_points))

    # Apply modulation to signal
    modulated: AudioArray = (audio_signal * tremolo_envelope).astype(np.float64)

    return modulated


def generate_ambient_pink_noise(num_samples: int) -> AudioArray:
    """
    Generate pink noise (1/f spectral density) simulating room air and ventilation.

    Args:
        num_samples (int): Required number of noise samples.

    Returns:
        AudioArray: 1D pink noise array.
    """

    # Generate white Gaussian noise
    white_noise: AudioArray = np.random.randn(num_samples).astype(np.float64)

    # Transform to frequency domain
    freq_spectrum: np.ndarray = np.fft.rfft(white_noise)

    # Compute 1/sqrt(f) filter curve for pink noise power distribution
    freqs: np.ndarray = np.fft.rfftfreq(num_samples)
    freqs[0] = 1.0
    pink_filter: np.ndarray = 1.0 / np.sqrt(freqs)
    pink_filter[0] = 0.0

    # Apply filter and transform back to time domain
    filtered_spectrum: np.ndarray = freq_spectrum * pink_filter
    pink_time: AudioArray = np.fft.irfft(filtered_spectrum, n=num_samples).astype(np.float64)

    return pink_time


def add_ambient_room_noise(
    audio_signal: AudioArray,
    target_snr_db: float = 25.0,
) -> AudioArray:
    """
    Add realistic ambient background noise at a designated signal-to-noise ratio.

    Args:
        audio_signal (AudioArray): Multi-channel or mono audio array.
        target_snr_db (float): Desired signal-to-noise ratio in decibels.

    Returns:
        AudioArray: Composite audio with room ambiance added.
    """

    # Calculate signal root-mean-square amplitude
    signal_rms: float = float(np.sqrt(np.mean(audio_signal**2)))

    # If signal is silent, return original
    if signal_rms < EPSILON:
        return audio_signal.copy()

    # Calculate required noise RMS based on target SNR
    snr_linear: float = 10.0 ** (target_snr_db / 20.0)
    target_noise_rms: float = signal_rms / (snr_linear + EPSILON)

    # Generate pink ambient noise
    num_samples: int = audio_signal.shape[0]

    if audio_signal.ndim == 1:
        raw_noise: AudioArray = generate_ambient_pink_noise(num_samples)
        current_noise_rms: float = float(np.sqrt(np.mean(raw_noise**2)))
        scaled_noise: AudioArray = raw_noise * (target_noise_rms / (current_noise_rms + EPSILON))
        return (audio_signal + scaled_noise).astype(np.float64)

    # Multi-channel handling
    num_channels: int = audio_signal.shape[1]
    noise_channels: list[AudioArray] = []

    for _ in range(num_channels):
        raw_ch_noise: AudioArray = generate_ambient_pink_noise(num_samples)
        ch_noise_rms: float = float(np.sqrt(np.mean(raw_ch_noise**2)))
        scaled_ch: AudioArray = raw_ch_noise * (target_noise_rms / (ch_noise_rms + EPSILON))
        noise_channels.append(scaled_ch)

    composite_noise: AudioArray = np.column_stack(noise_channels).astype(np.float64)

    return (audio_signal + composite_noise).astype(np.float64)
