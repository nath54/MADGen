"""
Acoustic and spectrogram analysis module for multi-speaker audio evaluation.

Provides STFT spectral analysis, psychoacoustic feature extraction, Scale-Invariant
Signal-to-Distortion Ratio (SI-SDR), and oracle time-frequency masking to quantify
voice recoverability versus destructive interference.
"""

# Import Modules
from dataclasses import asdict, dataclass
import typing

import numpy as np
import scipy.signal

from src.common.constants import EPSILON
from src.common.types import AudioArray


@dataclass
class SpectralFeatures:
    """
    Timbral and dynamic spectral metrics extracted from an audio signal.
    """

    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    spectral_flatness: float
    crest_factor_db: float
    rms_energy_db: float
    dynamic_range_db: float


@dataclass
class SpeakerRecoveryMetrics:
    """
    Evaluation metrics quantifying voice preservation vs destruction in a mix.
    """

    speaker_id: str
    active_duration_s: float
    speech_frames_count: int
    unprocessed_si_sdr_db: float
    oracle_recovered_si_sdr_db: float
    sdr_improvement_db: float
    recoverable_energy_fraction: float
    destroyed_energy_fraction: float
    overlap_speech_fraction: float
    solo_speech_fraction: float
    spectral_correlation: float
    active_sir_db: float


@dataclass
class SampleAnalysisReport:
    """
    Comprehensive acoustic and comparative evaluation report for a scene.
    """

    sample_id: str
    sample_rate: int
    duration_s: float
    channels: int
    overall_mix_features: SpectralFeatures
    stems_sum_snr_db: float
    speaker_metrics: list[SpeakerRecoveryMetrics]
    mean_recoverable_fraction: float
    mean_destroyed_fraction: float
    total_overlapping_frames_fraction: float

    def to_dict(self) -> dict[str, typing.Any]:
        """
        Convert report to a JSON-serializable dictionary.

        Returns:
            dict[str, typing.Any]: Serialized representation.
        """
        return asdict(self)


def ensure_1d_channel(audio: AudioArray, channel_idx: int = 0) -> np.ndarray:
    """
    Extract a single 1D audio channel from a potentially multi-channel array.

    Args:
        audio (AudioArray): 1D or 2D audio array.
        channel_idx (int): Zero-indexed channel to select.

    Returns:
        np.ndarray: 1D float64 audio array.
    """

    arr: np.ndarray = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        safe_channel: int = max(0, min(channel_idx, arr.shape[1] - 1))
        return arr[:, safe_channel]
    raise ValueError(f"Unsupported audio array dimension: {arr.ndim}")


def compute_stft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT) for an audio signal.

    Args:
        audio (np.ndarray): 1D audio samples.
        sample_rate (int): Sampling rate in Hertz.
        n_fft (int): FFT window length in samples.
        hop_length (int): Frame shift in samples.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (frequencies, times, complex STFT).
    """

    step_overlap: int = n_fft - hop_length
    freqs, times, zxx = scipy.signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=step_overlap,
        padded=True,
    )
    return freqs, times, zxx


def compute_log_spectrogram(
    stft_zxx: np.ndarray,
    ref_power: float = 1.0,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Convert complex STFT matrix into decibel power spectrogram.

    Args:
        stft_zxx (np.ndarray): Complex STFT array of shape (freq_bins, time_frames).
        ref_power (float): Reference power scaling value.
        top_db (float): Maximum dynamic range in dB below the peak.

    Returns:
        np.ndarray: Log-magnitude spectrogram in decibels.
    """

    magnitude_sq: np.ndarray = np.abs(stft_zxx) ** 2
    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(magnitude_sq, EPSILON) / ref_power)
    peak_db: float = float(np.max(log_spec))
    clipped_spec: np.ndarray = np.maximum(log_spec, peak_db - top_db)
    return clipped_spec


def compute_power_spectral_density(
    audio: np.ndarray,
    sample_rate: int,
    n_per_seg: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch Power Spectral Density (PSD) estimate.

    Args:
        audio (np.ndarray): 1D audio samples.
        sample_rate (int): Sampling rate in Hertz.
        n_per_seg (int): Segment length for Welch averaging.

    Returns:
        tuple[np.ndarray, np.ndarray]: (frequencies, power_density_values).
    """

    freqs, psd = scipy.signal.welch(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=min(len(audio), n_per_seg),
    )
    return freqs, psd


def _compute_centroid_and_rolloff(
    power: np.ndarray,
    freq_bins: np.ndarray,
) -> tuple[float, float]:
    """
    Helper to calculate average spectral centroid and 85% energy rolloff.

    Args:
        power (np.ndarray): STFT power matrix (frequencies x time).
        freq_bins (np.ndarray): Frequency vector in Hertz.

    Returns:
        tuple[float, float]: (average_centroid_hz, average_rolloff_hz).
    """

    total_power_per_frame: np.ndarray = np.sum(power, axis=0) + EPSILON
    weights: np.ndarray = freq_bins[:, np.newaxis]
    frame_centroids: np.ndarray = np.sum(weights * power, axis=0) / total_power_per_frame

    cumulative_power: np.ndarray = np.cumsum(power, axis=0)
    energy_threshold: np.ndarray = 0.85 * cumulative_power[-1, :]
    rolloff_indices: np.ndarray = np.argmax(cumulative_power >= energy_threshold, axis=0)
    frame_rolloff: np.ndarray = freq_bins[rolloff_indices]

    return float(np.mean(frame_centroids)), float(np.mean(frame_rolloff))


def _compute_flatness_and_dynamics(
    power: np.ndarray,
    audio_raw: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Helper to calculate spectral flatness, crest factor, RMS, and dynamic range.

    Args:
        power (np.ndarray): STFT power matrix.
        audio_raw (np.ndarray): Original 1D audio waveform.

    Returns:
        tuple[float, float, float, float]: (flatness, crest_db, rms_db, dynamic_range_db).
    """

    arithmetic_mean: np.ndarray = np.mean(power, axis=0) + EPSILON
    log_power: np.ndarray = np.log(np.maximum(power, EPSILON))
    geometric_mean: np.ndarray = np.exp(np.mean(log_power, axis=0))
    flatness: float = float(np.mean(geometric_mean / arithmetic_mean))

    peak_amp: float = float(np.max(np.abs(audio_raw)))
    rms_val: float = float(np.sqrt(np.mean(audio_raw**2)))
    crest_db: float = 20.0 * np.log10((peak_amp + EPSILON) / (rms_val + EPSILON))
    rms_db: float = 20.0 * np.log10(rms_val + EPSILON)

    frame_rms: np.ndarray = np.sqrt(np.mean(power, axis=0))
    valid_frames: np.ndarray = frame_rms[frame_rms > EPSILON]
    dyn_range: float = (
        float(np.percentile(20.0 * np.log10(valid_frames), 95)
              - np.percentile(20.0 * np.log10(valid_frames), 5))
        if len(valid_frames) > 0 else 0.0
    )

    return flatness, crest_db, rms_db, dyn_range


def compute_spectral_features(
    stft_zxx: np.ndarray,
    freq_bins: np.ndarray,
    audio_raw: np.ndarray,
) -> SpectralFeatures:
    """
    Extract timbral, spectral, and dynamic metrics from STFT and raw audio.

    Args:
        stft_zxx (np.ndarray): Complex STFT array.
        freq_bins (np.ndarray): Frequency vector in Hertz.
        audio_raw (np.ndarray): Original 1D audio samples.

    Returns:
        SpectralFeatures: Computed psychoacoustic features.
    """

    power: np.ndarray = np.abs(stft_zxx) ** 2
    centroid, rolloff = _compute_centroid_and_rolloff(power, freq_bins)
    flatness, crest_db, rms_db, dyn_range = _compute_flatness_and_dynamics(power, audio_raw)

    return SpectralFeatures(
        spectral_centroid_hz=round(centroid, 2),
        spectral_rolloff_hz=round(rolloff, 2),
        spectral_flatness=round(flatness, 4),
        crest_factor_db=round(crest_db, 2),
        rms_energy_db=round(rms_db, 2),
        dynamic_range_db=round(dyn_range, 2),
    )


def compute_si_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR in decibels).

    SI-SDR is the standard metric in speech separation benchmarks (BSS-eval, Conv-TasNet).
    It orthogonally projects the estimate onto the reference vector to isolate true target
    scaling from orthogonal residual distortion.

    Args:
        reference (np.ndarray): Clean ground-truth 1D reference signal.
        estimate (np.ndarray): Processed or mixed 1D estimate signal.

    Returns:
        float: SI-SDR metric in decibels (capped within [-50.0, 100.0]).
    """

    ref_zero_mean: np.ndarray = reference - np.mean(reference)
    est_zero_mean: np.ndarray = estimate - np.mean(estimate)

    ref_energy: float = float(np.dot(ref_zero_mean, ref_zero_mean))
    if ref_energy < EPSILON:
        return 0.0

    # Orthogonal projection coefficient: alpha = <est, ref> / ||ref||^2
    alpha: float = float(np.dot(est_zero_mean, ref_zero_mean)) / ref_energy
    target_component: np.ndarray = alpha * ref_zero_mean
    residual_distortion: np.ndarray = est_zero_mean - target_component

    target_energy: float = float(np.dot(target_component, target_component))
    residual_energy: float = float(np.dot(residual_distortion, residual_distortion))

    if residual_energy < EPSILON:
        return 100.0

    si_sdr_val: float = 10.0 * np.log10((target_energy + EPSILON) / (residual_energy + EPSILON))
    return float(np.clip(si_sdr_val, -50.0, 100.0))


def compute_spectral_correlation(
    mag_reference: np.ndarray,
    mag_estimate: np.ndarray,
    active_mask: np.ndarray,
) -> float:
    """
    Compute Pearson correlation between reference and estimate spectral magnitude envelopes.

    Args:
        mag_reference (np.ndarray): Reference magnitude spectrogram.
        mag_estimate (np.ndarray): Estimate magnitude spectrogram.
        active_mask (np.ndarray): Boolean mask indicating active speech frames.

    Returns:
        float: Pearson correlation coefficient [-1.0, 1.0].
    """

    if not np.any(active_mask):
        return 0.0

    ref_active: np.ndarray = mag_reference[:, active_mask].ravel()
    est_active: np.ndarray = mag_estimate[:, active_mask].ravel()

    ref_std: float = float(np.std(ref_active))
    est_std: float = float(np.std(est_active))

    if ref_std < EPSILON or est_std < EPSILON:
        return 0.0

    correlation_matrix: np.ndarray = np.corrcoef(ref_active, est_active)
    return float(np.clip(correlation_matrix[0, 1], -1.0, 1.0))


def _compute_stems_activity_and_sum(
    all_stems: dict[str, np.ndarray],
    target_id: str,
    shape: tuple[int, int],
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    Compute total stem magnitude sum, per-speaker active masks, and interference power.
    """

    sum_all_stems_mag = np.zeros(shape, dtype=np.float64)
    active_masks: dict[str, np.ndarray] = {}
    interference_energy = np.zeros(shape[1], dtype=np.float64)

    for spk_id, stem_audio in all_stems.items():
        _, _, spk_zxx = compute_stft(stem_audio, sample_rate, n_fft, hop_length)
        spk_mag: np.ndarray = np.abs(spk_zxx)
        sum_all_stems_mag += spk_mag

        spk_frame_power: np.ndarray = np.sum(spk_mag**2, axis=0)
        peak_pwr: float = float(np.max(spk_frame_power)) if len(spk_frame_power) > 0 else 0.0
        active_masks[spk_id] = spk_frame_power > (peak_pwr * 1e-4)

        if spk_id != target_id:
            interference_energy += spk_frame_power

    return sum_all_stems_mag, active_masks, interference_energy


def _compute_overlap_ratios(
    target_active: np.ndarray,
    active_masks: dict[str, np.ndarray],
    target_id: str,
) -> tuple[float, float, int]:
    """
    Calculate overlap fraction, solo fraction, and active frame count.
    """

    total_active_frames: int = int(np.sum(target_active))
    if total_active_frames == 0:
        return 0.0, 1.0, 0

    overlapping_frames: int = 0
    for frame_idx, is_active in enumerate(target_active):
        if is_active:
            other_active = any(
                mask[frame_idx] for s_id, mask in active_masks.items() if s_id != target_id
            )
            if other_active:
                overlapping_frames += 1

    overlap_frac = overlapping_frames / total_active_frames
    solo_frac = (total_active_frames - overlapping_frames) / total_active_frames
    return float(overlap_frac), float(solo_frac), total_active_frames


def _reconstruct_oracle_audio(
    mask: np.ndarray,
    zxx_mix: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """
    Reconstruct time-domain audio via inverse STFT from masked complex spectrum.
    """

    recovered_zxx = mask * zxx_mix
    _, recovered_audio = scipy.signal.istft(
        recovered_zxx,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
    )
    return np.asarray(recovered_audio, dtype=np.float64)


def _extract_active_signal(
    audio: np.ndarray,
    target_active: np.ndarray,
    hop_length: int,
) -> np.ndarray:
    """
    Extract concatenated waveform samples corresponding to active speech frames.
    """

    segments: list[np.ndarray] = []
    for idx, is_active in enumerate(target_active):
        if is_active:
            start_s: int = idx * hop_length
            end_s: int = start_s + hop_length
            if start_s < len(audio):
                segments.append(audio[start_s : min(end_s, len(audio))])
    if segments:
        return np.concatenate(segments)
    return audio


def _calculate_energy_recovery_fraction(
    ideal_ratio_mask: np.ndarray,
    mag_target: np.ndarray,
) -> float:
    """
    Calculate fraction of spectral energy preserved by ideal ratio mask.
    """

    target_power: np.ndarray = mag_target**2
    target_total: float = float(np.sum(target_power))
    recovered_energy: float = float(np.sum((ideal_ratio_mask**2) * target_power))
    ratio: float = (
        recovered_energy / (target_total + EPSILON) if target_total > EPSILON else 1.0
    )
    return float(np.clip(ratio, 0.0, 1.0))


def _compute_recovery_and_sdr(
    target_stem: np.ndarray,
    recovered_audio: np.ndarray,
    mix_audio: np.ndarray,
    ideal_ratio_mask: np.ndarray,
    mag_target: np.ndarray,
    target_active: np.ndarray,
    hop_length: int,
) -> tuple[float, float, float]:
    """
    Compute raw SI-SDR, oracle recovered SI-SDR, and recoverable energy fraction.
    """

    min_len: int = min(len(target_stem), len(recovered_audio), len(mix_audio))
    act_tgt = _extract_active_signal(target_stem[:min_len], target_active, hop_length)
    act_mix = _extract_active_signal(mix_audio[:min_len], target_active, hop_length)
    act_rec = _extract_active_signal(recovered_audio[:min_len], target_active, hop_length)

    raw_sdr: float = compute_si_sdr(act_tgt, act_mix)
    rec_sdr: float = compute_si_sdr(act_tgt, act_rec)
    rec_frac: float = _calculate_energy_recovery_fraction(ideal_ratio_mask, mag_target)

    return raw_sdr, rec_sdr, rec_frac


def _compute_sir_and_correlation(
    mag_target: np.ndarray,
    zxx_mix: np.ndarray,
    interf_energy: np.ndarray,
    target_active: np.ndarray,
) -> tuple[float, float]:
    """
    Compute active SIR and spectral correlation between target and mix.
    """

    target_pwr_active: float = float(np.sum(np.sum(mag_target**2, axis=0)[target_active]))
    interf_pwr_active: float = float(np.sum(interf_energy[target_active]))
    active_sir: float = (
        10.0 * np.log10((target_pwr_active + EPSILON) / (interf_pwr_active + EPSILON))
        if interf_pwr_active > EPSILON else 50.0
    )
    spec_corr: float = compute_spectral_correlation(mag_target, np.abs(zxx_mix), target_active)
    return active_sir, spec_corr


def _build_speaker_metrics(
    target_id: str,
    total_frames: int,
    frame_dur: float,
    sdr_tuple: tuple[float, float, float],
    stats_tuple: tuple[float, float, float, float],
) -> SpeakerRecoveryMetrics:
    """
    Construct SpeakerRecoveryMetrics record from evaluated components.
    """

    raw_sdr, rec_sdr, rec_frac = sdr_tuple
    overlap_frac, solo_frac, sir_db, spec_corr = stats_tuple

    return SpeakerRecoveryMetrics(
        speaker_id=target_id,
        active_duration_s=round(total_frames * frame_dur, 2),
        speech_frames_count=total_frames,
        unprocessed_si_sdr_db=round(raw_sdr, 2),
        oracle_recovered_si_sdr_db=round(rec_sdr, 2),
        sdr_improvement_db=round(rec_sdr - raw_sdr, 2),
        recoverable_energy_fraction=round(rec_frac, 4),
        destroyed_energy_fraction=round(1.0 - rec_frac, 4),
        overlap_speech_fraction=round(overlap_frac, 4),
        solo_speech_fraction=round(solo_frac, 4),
        spectral_correlation=round(spec_corr, 4),
        active_sir_db=round(sir_db, 2),
    )


def _prepare_target_and_oracle(
    target_stem: np.ndarray,
    all_stems: dict[str, np.ndarray],
    mix_audio: np.ndarray,
    target_id: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    Compute STFTs, stems activity, ideal ratio mask, and oracle reconstructed audio.
    """

    _, _, zxx_target = compute_stft(target_stem, sample_rate, n_fft, hop_length)
    _, _, zxx_mix = compute_stft(mix_audio, sample_rate, n_fft, hop_length)
    mag_target: np.ndarray = np.abs(zxx_target)

    sum_all_mag, active_masks, interf_energy = _compute_stems_activity_and_sum(
        all_stems, target_id, mag_target.shape, sample_rate, n_fft, hop_length
    )

    mask = np.clip(mag_target / (sum_all_mag + EPSILON), 0.0, 1.0)
    recovered_audio = _reconstruct_oracle_audio(
        mask, zxx_mix, sample_rate, n_fft, hop_length
    )
    return mag_target, zxx_mix, mask, recovered_audio, active_masks, interf_energy


def _evaluate_speaker_stats(
    target_stem: np.ndarray,
    mix_audio: np.ndarray,
    target_id: str,
    prep_data: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray
    ],
    hop_length: int,
    sample_rate: int,
) -> SpeakerRecoveryMetrics:
    """
    Evaluate SDR, SIR, overlap, and correlation from prepared spectral data.
    """

    mag_target: np.ndarray = prep_data[0]
    active_masks: dict[str, np.ndarray] = prep_data[4]
    target_active: np.ndarray = active_masks.get(
        target_id, np.zeros(mag_target.shape[1], dtype=bool)
    )

    overlap_frac, solo_frac, total_frames = _compute_overlap_ratios(
        target_active, active_masks, target_id
    )
    sdr_tuple = _compute_recovery_and_sdr(
        target_stem, prep_data[3], mix_audio, prep_data[2], mag_target, target_active, hop_length
    )
    sir_db, spec_corr = _compute_sir_and_correlation(
        mag_target, prep_data[1], prep_data[5], target_active
    )

    return _build_speaker_metrics(
        target_id=target_id,
        total_frames=total_frames,
        frame_dur=hop_length / float(sample_rate),
        sdr_tuple=sdr_tuple,
        stats_tuple=(overlap_frac, solo_frac, sir_db, spec_corr),
    )


def analyze_speaker_in_mix(
    target_stem: np.ndarray,
    all_stems: dict[str, np.ndarray],
    mix_audio: np.ndarray,
    target_id: str,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> tuple[SpeakerRecoveryMetrics, np.ndarray, np.ndarray]:
    """
    Quantify voice recoverability vs destruction for a single speaker stem in the mixture.

    Args:
        target_stem (np.ndarray): 1D ground-truth audio for this speaker.
        all_stems (dict[str, np.ndarray]): All speaker stems keyed by ID.
        mix_audio (np.ndarray): Complete mixed scene audio.
        target_id (str): Identifier of the target speaker.
        sample_rate (int): Sampling frequency in Hertz.
        n_fft (int): FFT size for time-frequency analysis.
        hop_length (int): STFT hop size.

    Returns:
        tuple[SpeakerRecoveryMetrics, np.ndarray, np.ndarray]:
            (Metrics, Ideal Ratio Mask matrix, Oracle Reconstructed audio).
    """

    prep_data = _prepare_target_and_oracle(
        target_stem, all_stems, mix_audio, target_id, sample_rate, n_fft, hop_length
    )
    metrics = _evaluate_speaker_stats(
        target_stem, mix_audio, target_id, prep_data, hop_length, sample_rate
    )
    return metrics, prep_data[2], prep_data[3]


def _compute_stems_sum_snr(stems_1d: dict[str, np.ndarray], mix_1d: np.ndarray) -> float:
    """
    Compute SNR between stems sum and mixed scene residual.
    """

    if not stems_1d:
        return 0.0

    aligned_len = min(len(mix_1d), min(len(s) for s in stems_1d.values()))
    stems_matrix = np.array([s[:aligned_len] for s in stems_1d.values()])
    stems_sum = np.sum(stems_matrix, axis=0)
    residual = mix_1d[:aligned_len] - stems_sum
    sum_pwr = float(np.mean(stems_sum**2))
    res_pwr = float(np.mean(residual**2))

    if res_pwr < EPSILON:
        return 60.0
    return float(10.0 * np.log10(sum_pwr / res_pwr))


def _compute_mean_statistics(
    metrics: list[SpeakerRecoveryMetrics],
) -> tuple[float, float, float]:
    """
    Calculate average recoverable fraction, destroyed fraction, and overlap fraction.
    """

    if not metrics:
        return 1.0, 0.0, 0.0
    mean_rec = float(np.mean([m.recoverable_energy_fraction for m in metrics]))
    mean_dest = float(np.mean([m.destroyed_energy_fraction for m in metrics]))
    mean_ovl = float(np.mean([m.overlap_speech_fraction for m in metrics]))
    return mean_rec, mean_dest, mean_ovl


def _evaluate_all_stems(
    stems_1d: dict[str, np.ndarray],
    mix_1d: np.ndarray,
    sample_rate: int,
) -> list[SpeakerRecoveryMetrics]:
    """
    Evaluate each speaker stem against the mixed audio.
    """

    results: list[SpeakerRecoveryMetrics] = []
    for spk_id, stem_audio in sorted(stems_1d.items()):
        metrics, _, _ = analyze_speaker_in_mix(
            target_stem=stem_audio,
            all_stems=stems_1d,
            mix_audio=mix_1d,
            target_id=spk_id,
            sample_rate=sample_rate,
        )
        results.append(metrics)
    return results


def evaluate_dataset_sample(
    sample_id: str,
    mix_audio: AudioArray,
    speaker_stems: dict[str, AudioArray],
    sample_rate: int,
    channel_idx: int = 0,
) -> SampleAnalysisReport:
    """
    Perform complete acoustic and comparative evaluation of a dataset sample.

    Args:
        sample_id (str): Sample session identifier.
        mix_audio (AudioArray): Multi-channel or mono mixed scene audio.
        speaker_stems (dict[str, AudioArray]): Isolated stems mapped by speaker ID.
        sample_rate (int): Audio sampling frequency in Hertz.
        channel_idx (int): Primary microphone channel to evaluate.

    Returns:
        SampleAnalysisReport: Full evaluation report including recovery and destruction metrics.
    """

    mix_1d: np.ndarray = ensure_1d_channel(mix_audio, channel_idx=channel_idx)
    stems_1d: dict[str, np.ndarray] = {
        spk_id: ensure_1d_channel(stem, channel_idx=channel_idx)
        for spk_id, stem in speaker_stems.items()
    }

    freqs, _, zxx_mix = compute_stft(mix_1d, sample_rate)
    mix_features: SpectralFeatures = compute_spectral_features(zxx_mix, freqs, mix_1d)
    speaker_metrics: list[SpeakerRecoveryMetrics] = _evaluate_all_stems(
        stems_1d, mix_1d, sample_rate
    )
    mean_rec, mean_dest, mean_ovl = _compute_mean_statistics(speaker_metrics)

    num_channels: int = (
        mix_audio.shape[1] if isinstance(mix_audio, np.ndarray) and mix_audio.ndim > 1 else 1
    )

    return SampleAnalysisReport(
        sample_id=sample_id,
        sample_rate=sample_rate,
        duration_s=round(len(mix_1d) / float(sample_rate), 2),
        channels=num_channels,
        overall_mix_features=mix_features,
        stems_sum_snr_db=round(_compute_stems_sum_snr(stems_1d, mix_1d), 2),
        speaker_metrics=speaker_metrics,
        mean_recoverable_fraction=round(mean_rec, 4),
        mean_destroyed_fraction=round(mean_dest, 4),
        total_overlapping_frames_fraction=round(mean_ovl, 4),
    )
