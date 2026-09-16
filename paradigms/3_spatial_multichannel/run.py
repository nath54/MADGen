"""
Runner script for Paradigm 3: Spatial Multi-Channel Processing.
"""

# Import Modules
import sys
from pathlib import Path
import argparse
import json
import logging
import soundfile as sf

_current_dir = Path(__file__).resolve().parent
_workspace_dir = _current_dir.parent.parent
if str(_workspace_dir) not in sys.path:
    sys.path.insert(0, str(_workspace_dir))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from models.mvdr_beamformer.pipeline import MVDRBeamformerPipeline
from paradigms.common.weights_manager import WeightsManager
from paradigms.common.streaming_feeder import StreamingAudioFeeder
from paradigms.common.metrics_evaluator import (
    parse_rttm_file,
    parse_transcripts_jsonl,
    compute_frame_level_der,
    compute_wer_metrics,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """
    CLI entry point for Paradigm 3 evaluation.
    """

    parser = argparse.ArgumentParser(description="Evaluate Paradigm 3: Spatial Multi-Channel Processing")
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="data/output/sample_002",
        help="Path to sample directory containing multi-channel mixed_scene.wav",
    )
    parser.add_argument(
        "--whisper-size",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium"],
        help="Faster-Whisper model size",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=4,
        help="Number of angular spatial beams to steer",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Simulate streaming real-time input using StreamingAudioFeeder",
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
        help="Optional path to save JSON evaluation report",
    )

    args = parser.parse_args()
    sample_dir = Path(args.sample_dir)
    audio_path = sample_dir / "mixed_scene.mp3"
    if not audio_path.is_file():
        audio_path = sample_dir / "mixed_scene.wav"
    if not audio_path.is_file():
        audio_path = sample_dir / "mixed.wav"

    if not audio_path.is_file():
        logger.error("Mixed audio file not found in %s (expected mixed_scene.mp3 or mixed_scene.wav)", sample_dir)
        return

    logger.info("Initializing Paradigm 3 Pipeline (Beams: %d, Whisper size: %s)...", args.beams, args.whisper_size)
    weights = WeightsManager()
    pipeline = MVDRBeamformerPipeline(
        weights_manager=weights,
        whisper_model_size=args.whisper_size,
        num_spatial_beams=args.beams,
    )

    if args.streaming:
        logger.info("Executing streaming simulation (%d ms chunks)...", args.chunk_ms)
        audio_data, sr = sf.read(str(audio_path), dtype="float32")
        feeder = StreamingAudioFeeder(
            audio=audio_data.T if audio_data.ndim > 1 and audio_data.shape[0] > audio_data.shape[1] else audio_data,
            sample_rate=sr,
            chunk_duration_ms=args.chunk_ms,
        )
        output = pipeline.process_streaming(feeder)
    else:
        logger.info("Executing offline spatial beamforming pipeline...")
        output = pipeline.process_offline(audio_path)

    # Evaluate metrics if reference files exist
    rttm_path = sample_dir / "diarization.rttm"
    jsonl_path = sample_dir / "transcripts.jsonl"

    if rttm_path.is_file():
        ref_turns = parse_rttm_file(rttm_path)
        hyp_turns = [(t.start_s, t.end_s, t.speaker_id) for t in output.turns]
        der_metrics = compute_frame_level_der(
            reference_intervals=ref_turns,
            hypothesis_intervals=hyp_turns,
            max_duration_s=output.audio_duration_s,
        )
        output.der = der_metrics.der
        logger.info("Diarization Error Rate (DER): %.2f%%", output.der * 100.0)

    if jsonl_path.is_file():
        records = parse_transcripts_jsonl(jsonl_path)
        ref_full_text = " ".join([r.get("text", "") for r in records]).strip()
        hyp_full_text = " ".join([t.text for t in output.turns]).strip()
        wer_metrics = compute_wer_metrics(ref_full_text, hyp_full_text)
        output.wer = wer_metrics.wer
        logger.info("Word Error Rate (WER): %.2f%%", output.wer * 100.0)

    summary = output.to_dict()
    print("\n" + "=" * 70)
    print(f" PARADIGM 3 EVALUATION REPORT: {output.session_id}")
    print("=" * 70)
    print(f"Pipeline:         {output.pipeline_name}")
    print(f"Audio Duration:   {output.audio_duration_s:.2f}s")
    print(f"Channels:         {output.metadata.get('channels', 1)}")
    print(f"Total Latency:    {output.profiler.total_pipeline_time_ms:.2f}ms")
    print(f"Real-Time Factor: {output.profiler.compute_rtf(output.audio_duration_s):.3f}x")
    if output.der is not None:
        print(f"DER:              {output.der * 100.0:.2f}%")
    if output.wer is not None:
        print(f"WER:              {output.wer * 100.0:.2f}%")
    print(f"Detected Turns:   {len(output.turns)}")
    print("-" * 70)
    print("Layer Latency Breakdown:")
    for layer, stats in summary["latency_summary"]["layers"].items():
        print(
            f"  - {layer:<14}: {stats['total_ms']:>8.2f} ms ({stats['percentage']:>5.1f}%) | "
            f"mean: {stats['mean_ms']:>6.2f} ms (x{int(stats['count'])})"
        )
    print("=" * 70)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f_out:
            json.dump(summary, f_out, indent=2)
        logger.info("Saved report to %s", out_p)


if __name__ == "__main__":
    main()
