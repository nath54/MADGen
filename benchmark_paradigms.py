"""
Master Benchmark CLI: Evaluates all multi-speaker speech processing paradigms
across MADGen generated scenes with layer-by-layer latency profiling and comparative metrics.
"""

# Import Modules
from pathlib import Path
import argparse
import json
import logging
import soundfile as sf

from paradigms import get_pipeline
from paradigms.common.weights_manager import WeightsManager
from paradigms.common.streaming_feeder import StreamingAudioFeeder
from paradigms.common.base_pipeline import PipelineOutput
from paradigms.common.metrics_evaluator import (
    parse_rttm_file,
    parse_transcripts_jsonl,
    compute_frame_level_der,
    compute_wer_metrics,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger: logging.Logger = logging.getLogger(__name__)


def resolve_sample_dir(sample_id_or_dir: str) -> Path:
    """
    Resolve sample directory from either sample ID or full path.
    """

    p = Path(sample_id_or_dir)
    if p.is_dir():
        return p
    output_candidate = Path("data/output") / sample_id_or_dir
    if output_candidate.is_dir():
        return output_candidate
    raise FileNotFoundError(f"Cannot find sample directory: {sample_id_or_dir}")


def print_comparison_dashboard(  # pylint: disable=too-many-locals
    sample_id: str,
    duration_s: float,
    channels: int,
    results: list[PipelineOutput],
) -> None:
    """
    Render ASCII comparison table showing DER, WER, layer latencies, and RTF.
    """

    width = 112
    print("\n" + "=" * width)
    print(
        f" MADGen PARADIGMS BENCHMARK REPORT: {sample_id} "
        f"(Duration: {duration_s:.1f}s, Channels: {channels})"
    )
    print("=" * width)
    header = (
        f"{'Paradigm / Model':<32} | {'DER (%)':>8} | {'WER (%)':>8} | "
        f"{'VAD (ms)':>9} | {'Proc/Sep (ms)':>13} | {'ASR (ms)':>9} | "
        f"{'Total (s)':>10} | {'RTF':>7}"
    )
    print(header)
    print("-" * width)

    for out in results:
        der_str = f"{out.der * 100.0:>7.1f}%" if out.der is not None else "    N/A "
        wer_str = f"{out.wer * 100.0:>7.1f}%" if out.wer is not None else "    N/A "

        layers = out.profiler.get_layer_totals_ms()
        vad_ms = layers.get("vad", 0.0)

        # Diarization / Separation / Beamforming / Online tracking
        proc_ms = (
            layers.get("diarization", 0.0)
            + layers.get("separation", 0.0)
            + layers.get("beamforming", 0.0)
            + layers.get("speaker_tracking", 0.0)
        )
        asr_ms = layers.get("asr", 0.0)
        total_s = out.profiler.total_pipeline_time_ms / 1000.0
        rtf = out.profiler.compute_rtf(out.audio_duration_s)

        row = (
            f"{out.pipeline_name[:32]:<32} | {der_str:>8} | {wer_str:>8} | "
            f"{vad_ms:>9.1f} | {proc_ms:>13.1f} | {asr_ms:>9.1f} | {total_s:>9.2f}s | {rtf:>6.3f}x"
        )
        print(row)

    print("=" * width + "\n")


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """
    CLI entry point for master paradigm benchmarking.
    """

    parser = argparse.ArgumentParser(
        description="MADGen Multi-Speaker Paradigm Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default="sample_001",
        help="Sample identifier (e.g. sample_001, sample_002, sample_007) or directory path",
    )
    parser.add_argument(
        "--paradigms",
        nargs="+",
        default=["1", "2", "3", "4"],
        help="Paradigms to benchmark (options: 1, 2, 3, 4)",
    )
    parser.add_argument(
        "--whisper-size",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium"],
        help="Faster-Whisper model variation",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Simulate real-time streaming microphone ingestion across all paradigms",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=200,
        help="Streaming chunk buffer size in milliseconds",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional destination path for JSON benchmark output",
    )

    args = parser.parse_args()
    sample_dir = resolve_sample_dir(args.sample_id)
    audio_path = sample_dir / "mixed_scene.mp3"
    if not audio_path.is_file():
        audio_path = sample_dir / "mixed_scene.wav"
    if not audio_path.is_file():
        audio_path = sample_dir / "mixed.wav"

    if not audio_path.is_file():
        logger.error("Mixed audio file not found in %s (expected mixed_scene.mp3 or mixed_scene.wav)", sample_dir)
        return

    info = sf.info(str(audio_path))
    audio_data, sr = sf.read(str(audio_path), dtype="float32")
    channels = info.channels
    duration_s = info.duration

    logger.info(
        "Beginning MADGen Paradigm Benchmark on %s (%.1fs, %d channels, sr=%d)...",
        sample_dir.name,
        duration_s,
        channels,
        sr,
    )

    # Load references for DER and WER
    rttm_path = sample_dir / "diarization.rttm"
    jsonl_path = sample_dir / "transcripts.jsonl"
    ref_turns = parse_rttm_file(rttm_path) if rttm_path.is_file() else []
    ref_records = parse_transcripts_jsonl(jsonl_path) if jsonl_path.is_file() else []
    ref_full_text = " ".join([r.get("text", "") for r in ref_records]).strip()

    weights = WeightsManager()
    results: list[PipelineOutput] = []

    for p_key in args.paradigms:
        logger.info("--> Running Paradigm %s...", p_key)
        try:
            pipeline = get_pipeline(
                p_key,
                weights_manager=weights,
                whisper_model_size=args.whisper_size,
            )
            if args.streaming:
                feeder_audio = (
                    audio_data.T
                    if audio_data.ndim > 1 and audio_data.shape[0] > audio_data.shape[1]
                    else audio_data
                )
                feeder = StreamingAudioFeeder(
                    audio=feeder_audio,
                    sample_rate=sr,
                    chunk_duration_ms=args.chunk_ms,
                )
                output = pipeline.process_streaming(feeder)
            else:
                output = pipeline.process_offline(audio_path)

            output.session_id = sample_dir.name

            # Calculate DER
            if ref_turns:
                hyp_turns = [(t.start_s, t.end_s, t.speaker_id) for t in output.turns]
                der_eval = compute_frame_level_der(
                    reference_intervals=ref_turns,
                    hypothesis_intervals=hyp_turns,
                    max_duration_s=output.audio_duration_s,
                )
                output.der = der_eval.der

            # Calculate WER
            if ref_full_text:
                hyp_full_text = " ".join([t.text for t in output.turns]).strip()
                wer_eval = compute_wer_metrics(ref_full_text, hyp_full_text)
                output.wer = wer_eval.wer

            results.append(output)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Paradigm %s failed: %s", p_key, exc)

    print_comparison_dashboard(
        sample_id=sample_dir.name,
        duration_s=duration_s,
        channels=channels,
        results=results,
    )

    out_json_path = (
        Path(args.output_json)
        if args.output_json
        else sample_dir / "benchmark_report.json"
    )
    report_dict = {
        "sample_id": sample_dir.name,
        "audio_duration_s": duration_s,
        "channels": channels,
        "sample_rate": sr,
        "whisper_size": args.whisper_size,
        "streaming_mode": args.streaming,
        "results": [r.to_dict() for r in results],
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f_out:
        json.dump(report_dict, f_out, indent=2)
    logger.info("Saved benchmark report to %s", out_json_path)


if __name__ == "__main__":
    main()
