"""
Batch benchmark evaluator across all 10 MADGen samples.
Generates comprehensive benchmark metrics (DER, WER, RTF, Latency) for each sample.
"""

# Import Modules
import sys
from pathlib import Path
import json
import logging
import soundfile as sf

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from paradigms import get_pipeline
from paradigms.common.weights_manager import WeightsManager
from paradigms.common.metrics_evaluator import (
    parse_rttm_file,
    parse_transcripts_jsonl,
    compute_frame_level_der,
    compute_wer_metrics,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    weights = WeightsManager()
    output_dir = Path("data/output")
    master_results = []

    sample_dirs = sorted(output_dir.glob("sample_*"))

    for s_dir in sample_dirs:
        audio_path = s_dir / "mixed_scene.mp3"
        if not audio_path.is_file():
            audio_path = s_dir / "mixed_scene.wav"
        if not audio_path.is_file():
            audio_path = s_dir / "mixed.wav"
        if not audio_path.is_file():
            continue

        info = sf.info(str(audio_path))
        rttm_path = s_dir / "diarization.rttm"
        jsonl_path = s_dir / "transcripts.jsonl"
        ann_path = s_dir / "annotations.json"

        # Load metadata
        lang = "en"
        spk_count = 1
        if ann_path.is_file():
            try:
                with ann_path.open() as f:
                    ann = json.load(f)
                spks = ann.get("speakers", [])
                spk_count = len(spks) or ann.get("num_speakers", 1)
                langs = sorted(list({s.get("language", "en") for s in spks}))
                lang = ", ".join(langs) if langs else "en"
            except Exception:
                pass

        ref_turns = parse_rttm_file(rttm_path) if rttm_path.is_file() else []
        ref_records = parse_transcripts_jsonl(jsonl_path) if jsonl_path.is_file() else []
        ref_text = " ".join([r.get("text", "") for r in ref_records]).strip()

        logger.info("Benchmarking %s (%.1fs, %d ch, %d spks, %s)...", s_dir.name, info.duration, info.channels, spk_count, lang)

        sample_benchmark = {
            "sample_id": s_dir.name,
            "duration_s": round(info.duration, 1),
            "channels": info.channels,
            "speakers": spk_count,
            "language": lang,
            "paradigms": {},
        }

        # Paradigms to evaluate: 1 (Diarize-Then-Transcribe) and 4 (Realtime Streaming)
        for p_key in ["1", "4"]:
            try:
                pipe = get_pipeline(p_key, weights_manager=weights, whisper_model_size="tiny")
                out = pipe.process_offline(audio_path)

                if ref_turns:
                    hyp_turns = [(t.start_s, t.end_s, t.speaker_id) for t in out.turns]
                    der_eval = compute_frame_level_der(
                        reference_intervals=ref_turns,
                        hypothesis_intervals=hyp_turns,
                        max_duration_s=out.audio_duration_s,
                    )
                    out.der = der_eval.der

                if ref_text:
                    hyp_text = " ".join([t.text for t in out.turns]).strip()
                    wer_eval = compute_wer_metrics(ref_text, hyp_text)
                    out.wer = wer_eval.wer

                rtf = out.profiler.compute_rtf(out.audio_duration_s)
                total_s = out.profiler.total_pipeline_time_ms / 1000.0

                sample_benchmark["paradigms"][pipe.name] = {
                    "der_pct": round(out.der * 100.0, 1) if out.der is not None else None,
                    "wer_pct": round(out.wer * 100.0, 1) if out.wer is not None else None,
                    "rtf": round(rtf, 3),
                    "latency_s": round(total_s, 2),
                    "turns_detected": len(out.turns),
                }
            except Exception as exc:
                logger.error("Failed paradigm %s on %s: %s", p_key, s_dir.name, exc)

        master_results.append(sample_benchmark)

    summary_file = output_dir / "all_samples_benchmark.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
    logger.info("Saved all samples benchmark to %s", summary_file)


if __name__ == "__main__":
    main()
