"""
Audio spectrogram and voice recovery/destruction analysis CLI tool.

Analyzes mixed spatial audio scenes and evaluates individual speaker stems against
the mixture using STFT spectrograms, psychoacoustic metrics, SI-SDR, and oracle
time-frequency masking.
"""

# Import Modules
import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from src.audio.analysis import (
    SampleAnalysisReport,
    SpeakerRecoveryMetrics,
    analyze_speaker_in_mix,
    compute_log_spectrogram,
    compute_power_spectral_density,
    compute_stft,
    ensure_1d_channel,
    evaluate_dataset_sample,
)

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger: logging.Logger = logging.getLogger("analyze_audio")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for acoustic and spectrogram evaluation.

    Returns:
        argparse.Namespace: Parsed argument parameters.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Analyze spectrograms and voice recoverability vs destruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=None,
        help="Path to a specific dataset sample folder (e.g. data/output/sample_001).",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Identifier of sample within --output-dir (e.g. sample_001).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Base directory containing dataset samples.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Batch analyze all samples located in --output-dir.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Zero-indexed microphone channel to analyze.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Custom path to save the spectrogram analysis plot.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Custom path to save the JSON analysis report.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for exported visualization image.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating graphical spectrogram plots.",
    )

    return parser.parse_args()


def load_sample_audio_and_stems(
    sample_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """
    Load mixed scene audio and all available isolated speaker stems.

    Args:
        sample_dir (Path): Sample folder containing mixed_scene.wav and isolated_speakers/.

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray], int]: (mix_data, stems_dict, sample_rate).

    Raises:
        FileNotFoundError: If mixed_scene.mp3 or mixed_scene.wav is missing.
    """

    mix_file: Path = sample_dir / "mixed_scene.mp3"
    if not mix_file.is_file():
        mix_file = sample_dir / "mixed_scene.wav"
    if not mix_file.is_file():
        raise FileNotFoundError(f"Missing mixed_scene.mp3 or mixed_scene.wav in {sample_dir}")

    mix_data, sample_rate = sf.read(str(mix_file))

    stems_dir: Path = sample_dir / "isolated_speakers"
    stems_dict: dict[str, np.ndarray] = {}

    if stems_dir.is_dir():
        stem_files = sorted(list(stems_dir.glob("*.mp3")) or list(stems_dir.glob("*.wav")))
        for stem_file in stem_files:
            spk_id: str = stem_file.stem
            stem_audio, _ = sf.read(str(stem_file))
            stems_dict[spk_id] = stem_audio

    return mix_data, stems_dict, sample_rate


def _plot_mixed_spectrogram(
    axis: Axes,
    mix_1d: np.ndarray,
    sample_rate: int,
    report: SampleAnalysisReport,
) -> None:
    """
    Render log-magnitude spectrogram for the mixed scene.
    """

    freqs, times, zxx_mix = compute_stft(mix_1d, sample_rate, n_fft=1024, hop_length=256)
    spec_db = compute_log_spectrogram(zxx_mix, top_db=70.0)

    im_mix = axis.pcolormesh(
        times,
        freqs / 1000.0,
        spec_db,
        shading="gouraud",
        cmap="magma",
    )
    cbar = plt.colorbar(im_mix, ax=axis, pad=0.01)
    cbar.set_label("Power (dB)", fontsize=9)
    axis.set_title(
        f"Mixed Scene Spectrogram: {report.sample_id} "
        f"(Dur: {report.duration_s}s, "
        f"Centroid: {report.overall_mix_features.spectral_centroid_hz:.0f} Hz, "
        f"Stems SNR: {report.stems_sum_snr_db:.1f} dB)",
        fontsize=11,
        fontweight="bold",
    )
    axis.set_xlabel("Time (s)", fontsize=9)
    axis.set_ylabel("Frequency (kHz)", fontsize=9)
    axis.set_ylim(0, sample_rate / 2000.0)


def _plot_psd_comparison(
    axis: Axes,
    mix_1d: np.ndarray,
    stems_1d: dict[str, np.ndarray],
    sample_rate: int,
) -> None:
    """
    Render Power Spectral Density (PSD) comparison between mix and stems.
    """

    psd_freqs, psd_mix = compute_power_spectral_density(mix_1d, sample_rate)
    axis.plot(
        psd_freqs,
        10.0 * np.log10(psd_mix + 1e-12),
        label="Mixed Scene",
        color="black",
        linewidth=2.0,
        alpha=0.9,
    )

    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(stems_1d))))  # pylint: disable=no-member
    for idx, (spk_id, stem_audio) in enumerate(stems_1d.items()):
        _, psd_stem = compute_power_spectral_density(stem_audio, sample_rate)
        axis.plot(
            psd_freqs,
            10.0 * np.log10(psd_stem + 1e-12),
            label=spk_id,
            color=palette[idx % len(palette)],
            alpha=0.6,
            linewidth=1.0,
        )

    axis.set_title("Power Spectral Density (PSD)", fontsize=10, fontweight="bold")
    axis.set_xlabel("Frequency (Hz)", fontsize=8)
    axis.set_ylabel("Power Density (dB/Hz)", fontsize=8)
    axis.set_xlim(0, 8000)
    axis.grid(True, linestyle="--", alpha=0.5)
    if len(stems_1d) <= 6:
        axis.legend(fontsize=7, loc="upper right")


def _plot_recovery_bars(
    axis: Axes,
    report: SampleAnalysisReport,
) -> None:
    """
    Render horizontal stacked bar chart of recoverable vs destroyed vocal energy.
    """

    spk_ids = [m.speaker_id for m in report.speaker_metrics]
    rec_pct = [m.recoverable_energy_fraction * 100.0 for m in report.speaker_metrics]
    dest_pct = [m.destroyed_energy_fraction * 100.0 for m in report.speaker_metrics]
    y_pos = np.arange(len(spk_ids))

    axis.barh(y_pos, rec_pct, label="Recoverable Energy (%)", color="#2ecc71", alpha=0.85)
    axis.barh(
        y_pos,
        dest_pct,
        left=rec_pct,
        label="Destroyed / Masked (%)",
        color="#e74c3c",
        alpha=0.85,
    )
    axis.set_yticks(y_pos)
    axis.set_yticklabels(spk_ids, fontsize=8)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Energy Proportion (%)", fontsize=8)
    axis.set_title(
        f"Voice Recoverability vs Destruction (Mean Recoverable: "
        f"{report.mean_recoverable_fraction * 100.0:.1f}%)",
        fontsize=10,
        fontweight="bold",
    )
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(axis="x", linestyle="--", alpha=0.5)


def _plot_stem_spectrogram(
    axis: Axes,
    audio: np.ndarray,
    sample_rate: int,
    title: str,
) -> None:
    """
    Render spectrogram for a single stem.
    """

    freqs, times, zxx = compute_stft(audio, sample_rate, n_fft=1024, hop_length=256)
    spec = compute_log_spectrogram(zxx, top_db=70.0)
    im_stem = axis.pcolormesh(
        times, freqs / 1000.0, spec,
        shading="gouraud", cmap="viridis",
    )
    plt.colorbar(im_stem, ax=axis, pad=0.01)
    axis.set_title(title, fontsize=10, fontweight="bold")
    axis.set_xlabel("Time (s)", fontsize=8)
    axis.set_ylabel("Frequency (kHz)", fontsize=8)
    axis.set_ylim(0, sample_rate / 2000.0)


def _plot_primary_comparison(
    fig: Figure,
    grid_spec: GridSpec,
    stems_tuple: tuple[np.ndarray, np.ndarray],
    metric: SpeakerRecoveryMetrics,
    sample_rate: int,
) -> None:
    """
    Render comparative subplots for ground-truth stem vs oracle recovered audio.
    """

    target_stem, p_recovered = stems_tuple
    p_id: str = metric.speaker_id

    ax_ref = fig.add_subplot(grid_spec[2, 0])
    _plot_stem_spectrogram(
        ax_ref,
        target_stem,
        sample_rate,
        f"Ground-Truth Isolated Stem: {p_id} (Active: {metric.active_duration_s}s)",
    )

    ax_rec = fig.add_subplot(grid_spec[2, 1])
    _plot_stem_spectrogram(
        ax_rec,
        p_recovered,
        sample_rate,
        f"Oracle Recovered: {p_id} (Gain: +{metric.sdr_improvement_db} dB, "
        f"Rec: {metric.recoverable_energy_fraction * 100.0:.1f}%)",
    )


def _plot_speaker_spectrograms(
    fig: Figure,
    grid_spec: GridSpec,
    mix_1d: np.ndarray,
    stems_1d: dict[str, np.ndarray],
    sample_rate: int,
    report: SampleAnalysisReport,
) -> None:
    """
    Render comparative spectrograms for the primary speaker (clean stem vs oracle recovered).
    """

    if not (report.speaker_metrics and stems_1d):
        return

    primary_metric = max(report.speaker_metrics, key=lambda m: m.active_duration_s)
    p_id = primary_metric.speaker_id
    target_stem = stems_1d[p_id]

    _, _, p_recovered = analyze_speaker_in_mix(
        target_stem=target_stem,
        all_stems=stems_1d,
        mix_audio=mix_1d,
        target_id=p_id,
        sample_rate=sample_rate,
    )

    _plot_primary_comparison(
        fig=fig,
        grid_spec=grid_spec,
        stems_tuple=(target_stem, p_recovered),
        metric=primary_metric,
        sample_rate=sample_rate,
    )


def generate_spectrogram_plot(
    mix_1d: np.ndarray,
    stems_1d: dict[str, np.ndarray],
    sample_rate: int,
    report: SampleAnalysisReport,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Generate multi-panel publication-grade spectrogram and voice recovery plot.

    Args:
        mix_1d (np.ndarray): 1D mixed audio signal.
        stems_1d (dict[str, np.ndarray]): 1D stems mapped by speaker ID.
        sample_rate (int): Sampling frequency in Hertz.
        report (SampleAnalysisReport): Computed evaluation report.
        output_path (Path): Destination PNG image path.
        dpi (int): Output plot resolution.
    """

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid_spec = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.2])

    ax_mix = fig.add_subplot(grid_spec[0, :])
    _plot_mixed_spectrogram(ax_mix, mix_1d, sample_rate, report)

    ax_psd = fig.add_subplot(grid_spec[1, 0])
    _plot_psd_comparison(ax_psd, mix_1d, stems_1d, sample_rate)

    ax_bar = fig.add_subplot(grid_spec[1, 1])
    _plot_recovery_bars(ax_bar, report)

    _plot_speaker_spectrograms(
        fig, grid_spec, mix_1d, stems_1d, sample_rate, report
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)
    logger.info("Saved spectrogram analysis plot to %s", output_path)


def print_cli_dashboard(report: SampleAnalysisReport) -> None:
    """
    Render formatted ASCII dashboard summarizing acoustic and recovery metrics.

    Args:
        report (SampleAnalysisReport): Computed evaluation report.
    """

    sep: str = "=" * 88
    subsep: str = "-" * 88

    print("\n" + sep)
    print(f" ACOUSTIC & SPECTROGRAM ANALYSIS REPORT: {report.sample_id}")
    print(sep)
    print(
        f" Duration: {report.duration_s:.2f}s | Sample Rate: {report.sample_rate} Hz | "
        f"Channels: {report.channels} | Stems-to-Residual SNR: {report.stems_sum_snr_db:.2f} dB"
    )
    print(subsep)
    print(
        f" Spectral Centroid: {report.overall_mix_features.spectral_centroid_hz:.1f} Hz | "
        f"Rolloff (85%): {report.overall_mix_features.spectral_rolloff_hz:.1f} Hz"
    )
    print(
        f" Spectral Flatness:  {report.overall_mix_features.spectral_flatness:.4f}    | "
        f"Crest Factor:  {report.overall_mix_features.crest_factor_db:.2f} dB | "
        f"Dyn Range: {report.overall_mix_features.dynamic_range_db:.1f} dB"
    )
    print(sep)
    print(
        f"{'Speaker ID':<12} | {'Active (s)':<10} | {'Overlap %':<9} | "
        f"{'Mix SI-SDR':<10} | {'Oracle SDR':<10} | {'Gain':<8} | "
        f"{'Recover %':<9} | {'Destroy %':<9}"
    )
    print(subsep)

    for m in report.speaker_metrics:
        print(
            f"{m.speaker_id:<12} | "
            f"{m.active_duration_s:<10.1f} | "
            f"{m.overlap_speech_fraction * 100.0:<8.1f}% | "
            f"{m.unprocessed_si_sdr_db:<9.1f}dB | "
            f"{m.oracle_recovered_si_sdr_db:<9.1f}dB | "
            f"{f'+{m.sdr_improvement_db:.1f}dB':<8} | "
            f"{m.recoverable_energy_fraction * 100.0:<8.1f}% | "
            f"{m.destroyed_energy_fraction * 100.0:<8.1f}%"
        )

    print(subsep)
    print(
        f" OVERALL SUMMARY: "
        f"Mean Recoverable Energy: {report.mean_recoverable_fraction * 100.0:.1f}% | "
        f"Mean Destroyed Energy: {report.mean_destroyed_fraction * 100.0:.1f}% | "
        f"Mean Overlap: {report.total_overlapping_frames_fraction * 100.0:.1f}%"
    )
    print(sep + "\n")

def _export_analysis_artifacts(
    sample_dir: Path,
    report: SampleAnalysisReport,
    mix_audio: np.ndarray,
    stems_dict: dict[str, np.ndarray],
    sample_rate: int,
    output_plot: Path | None,
    output_report: Path | None,
    no_plot: bool,
    dpi: int,
    channel_idx: int,
) -> None:
    """
    Save JSON evaluation report and generate spectrogram plot images.
    """

    report_file: Path = output_report or (sample_dir / "analysis_report.json")
    with report_file.open("w", encoding="utf-8") as f_out:
        json.dump(report.to_dict(), f_out, indent=2)
    logger.info("Saved JSON report to %s", report_file)

    if not no_plot:
        plot_file: Path = output_plot or (sample_dir / "spectrogram_analysis.png")
        mix_1d = ensure_1d_channel(mix_audio, channel_idx=channel_idx)
        stems_1d = {
            s_id: ensure_1d_channel(s_audio, channel_idx=channel_idx)
            for s_id, s_audio in stems_dict.items()
        }
        generate_spectrogram_plot(
            mix_1d=mix_1d,
            stems_1d=stems_1d,
            sample_rate=sample_rate,
            report=report,
            output_path=plot_file,
            dpi=dpi,
        )


def analyze_single_sample(
    sample_dir: Path,
    channel_idx: int = 0,
    output_plot: Path | None = None,
    output_report: Path | None = None,
    no_plot: bool = False,
    dpi: int = 150,
) -> SampleAnalysisReport:
    """
    Execute full evaluation workflow for a single dataset sample directory.

    Args:
        sample_dir (Path): Sample folder.
        channel_idx (int): Audio channel to evaluate.
        output_plot (Path | None): Custom path for plot PNG.
        output_report (Path | None): Custom path for JSON report.
        no_plot (bool): Whether to bypass generating plot image.
        dpi (int): Image resolution.

    Returns:
        SampleAnalysisReport: Computed evaluation metrics.
    """

    logger.info("Analyzing sample in: %s", sample_dir)

    mix_audio, stems_dict, sample_rate = load_sample_audio_and_stems(sample_dir)
    report: SampleAnalysisReport = evaluate_dataset_sample(
        sample_id=sample_dir.name,
        mix_audio=mix_audio,
        speaker_stems=stems_dict,
        sample_rate=sample_rate,
        channel_idx=channel_idx,
    )

    _export_analysis_artifacts(
        sample_dir, report, mix_audio, stems_dict, sample_rate,
        output_plot, output_report, no_plot, dpi, channel_idx
    )

    print_cli_dashboard(report)
    return report


def _resolve_target_directories(args: argparse.Namespace) -> list[Path]:
    """
    Determine target sample directories from CLI options.
    """

    if args.all:
        if not args.output_dir.is_dir():
            logger.error("Base output directory '%s' does not exist", args.output_dir)
            return []
        return [
            entry for entry in sorted(args.output_dir.iterdir())
            if entry.is_dir() and ((entry / "mixed_scene.mp3").is_file() or (entry / "mixed_scene.wav").is_file())
        ]

    if args.sample_dir:
        return [args.sample_dir]

    if args.sample_id:
        return [args.output_dir / args.sample_id]

    default_sample = args.output_dir / "sample_001"
    if default_sample.is_dir():
        return [default_sample]

    return []


def main() -> int:
    """
    Main entry point for command-line audio analysis tool.

    Returns:
        int: Exit status code (0 for success, non-zero for errors).
    """

    args: argparse.Namespace = parse_arguments()

    try:
        targets = _resolve_target_directories(args)
        if not targets:
            logger.error("No valid sample targets found. Check --output-dir or --sample-id.")
            return 1

        for target_dir in targets:
            analyze_single_sample(
                sample_dir=target_dir,
                channel_idx=args.channel,
                output_plot=args.output_plot,
                output_report=args.output_report,
                no_plot=args.no_plot,
                dpi=args.dpi,
            )

        return 0

    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Analysis failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
