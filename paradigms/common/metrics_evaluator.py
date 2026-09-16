"""
Benchmark metrics evaluation engine computing Diarization Error Rate (DER)
and Word Error Rate (WER).
"""

# Import Modules
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging
import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class DiarizationMetrics:
    """
    Detailed diarization evaluation metrics.
    """

    der: float
    missed_speech_fraction: float
    false_alarm_fraction: float
    speaker_confusion_fraction: float
    ground_truth_duration_s: float
    hypothesis_duration_s: float


@dataclass
class ASRMetrics:
    """
    Speech-to-text transcription evaluation metrics.
    """

    wer: float
    insertions: int
    deletions: int
    substitutions: int
    word_count: int


def parse_rttm_file(rttm_path: Path) -> list[tuple[float, float, str]]:
    """
    Parse NIST RTTM file into list of (start_s, end_s, speaker_id) intervals.

    Args:
        rttm_path (Path): Path to standard RTTM file.

    Returns:
        list[tuple[float, float, str]]: Extracted speech intervals.
    """

    intervals: list[tuple[float, float, str]] = []
    if not rttm_path.is_file():
        return intervals

    with rttm_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                try:
                    start_s = float(parts[3])
                    dur_s = float(parts[4])
                    spk_id = parts[7]
                    intervals.append((start_s, start_s + dur_s, spk_id))
                except (ValueError, IndexError):
                    continue

    return intervals


def parse_transcripts_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """
    Load ground-truth transcript records from JSONL file.

    Args:
        jsonl_path (Path): Path to transcripts.jsonl.

    Returns:
        list[dict[str, Any]]: Loaded transcript lines.
    """

    records: list[dict[str, Any]] = []
    if not jsonl_path.is_file():
        return records

    with jsonl_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue

    return records


def compute_frame_level_der(  # pylint: disable=too-many-locals,too-many-branches
    reference_intervals: list[tuple[float, float, str]],
    hypothesis_intervals: list[tuple[float, float, str]],
    step_s: float = 0.05,
    max_duration_s: float | None = None,
) -> DiarizationMetrics:
    """
    Compute frame-level Diarization Error Rate (DER) with optimal speaker mapping.

    Args:
        reference_intervals (list[tuple[float, float, str]]): Ground-truth turns.
        hypothesis_intervals (list[tuple[float, float, str]]): Predicted turns.
        step_s (float): Frame resolution step in seconds (default 50ms).
        max_duration_s (float | None): Total session duration.

    Returns:
        DiarizationMetrics: Detailed DER statistics.
    """

    if not reference_intervals and not hypothesis_intervals:
        return DiarizationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Determine timeline extent
    all_ends = [t[1] for t in reference_intervals] + [t[1] for t in hypothesis_intervals]
    total_dur = max(all_ends) if all_ends else 0.0
    if max_duration_s is not None:
        total_dur = max(total_dur, max_duration_s)

    num_frames = int(np.ceil(total_dur / step_s))
    if num_frames == 0:
        return DiarizationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Frame masks: frame_idx -> set of active speakers
    ref_frames: list[set[str]] = [set() for _ in range(num_frames)]
    hyp_frames: list[set[str]] = [set() for _ in range(num_frames)]

    for start_s, end_s, spk in reference_intervals:
        s_idx = max(0, int(start_s / step_s))
        e_idx = min(num_frames, int(np.ceil(end_s / step_s)))
        for idx in range(s_idx, e_idx):
            ref_frames[idx].add(spk)

    for start_s, end_s, spk in hypothesis_intervals:
        s_idx = max(0, int(start_s / step_s))
        e_idx = min(num_frames, int(np.ceil(end_s / step_s)))
        for idx in range(s_idx, e_idx):
            hyp_frames[idx].add(spk)

    # Optimal Hungarian mapping between hypothesis speakers and reference speakers
    hyp_spks = sorted(list({spk for _, _, spk in hypothesis_intervals}))
    ref_spks = sorted(list({spk for _, _, spk in reference_intervals}))

    # Co-occurrence cost matrix
    cooccur = np.zeros((len(hyp_spks), len(ref_spks)), dtype=np.int32)
    for h_set, r_set in zip(hyp_frames, ref_frames):
        for h_idx, h_spk in enumerate(hyp_spks):
            if h_spk in h_set:
                for r_idx, r_spk in enumerate(ref_spks):
                    if r_spk in r_set:
                        cooccur[h_idx, r_idx] += 1

    # Greedy or linear sum assignment matching
    if len(hyp_spks) > 0 and len(ref_spks) > 0:
        cost_matrix = -cooccur
        h_ind, r_ind = linear_sum_assignment(cost_matrix)
        mapping = {hyp_spks[h]: ref_spks[r] for h, r in zip(h_ind, r_ind)}
    else:
        mapping = {}

    # Map hypothesis frames
    mapped_hyp_frames: list[set[str]] = [
        {mapping.get(s, s) for s in h_set} for h_set in hyp_frames
    ]

    # Calculate frame-level errors
    total_ref_speech = 0
    missed_speech = 0
    false_alarm = 0
    confusion = 0

    for r_set, h_set in zip(ref_frames, mapped_hyp_frames):
        num_r = len(r_set)
        num_h = len(h_set)
        total_ref_speech += num_r

        if num_r > num_h:
            missed_speech += num_r - num_h
        elif num_h > num_r:
            false_alarm += num_h - num_r

        # Speaker confusion
        overlap_correct = len(r_set.intersection(h_set))
        potential_matches = min(num_r, num_h)
        confusion += potential_matches - overlap_correct

    total_ref_frames = max(1, total_ref_speech)
    der = (missed_speech + false_alarm + confusion) / float(total_ref_frames)

    return DiarizationMetrics(
        der=float(der),
        missed_speech_fraction=float(missed_speech / total_ref_frames),
        false_alarm_fraction=float(false_alarm / total_ref_frames),
        speaker_confusion_fraction=float(confusion / total_ref_frames),
        ground_truth_duration_s=float(total_ref_speech * step_s),
        hypothesis_duration_s=float(sum(len(h) for h in hyp_frames) * step_s),
    )


def compute_wer_metrics(
    reference_text: str,
    hypothesis_text: str,
) -> ASRMetrics:
    """
    Compute standard Word Error Rate (WER) using Levenshtein distance.

    Args:
        reference_text (str): Ground-truth reference string.
        hypothesis_text (str): Predicted transcription string.

    Returns:
        ASRMetrics: Computed WER and error breakdown.
    """

    clean_ref = reference_text.strip()
    clean_hyp = hypothesis_text.strip()

    if not clean_ref and not clean_hyp:
        return ASRMetrics(0.0, 0, 0, 0, 0)
    if not clean_ref:
        words = clean_hyp.split()
        return ASRMetrics(1.0, len(words), 0, 0, len(words))

    wer_score = float(jiwer.wer(clean_ref, clean_hyp))
    word_count = len(clean_ref.split())

    # Levenshtein breakdown
    out = jiwer.process_words(clean_ref, clean_hyp)

    return ASRMetrics(
        wer=wer_score,
        insertions=out.insertions,
        deletions=out.deletions,
        substitutions=out.substitutions,
        word_count=word_count,
    )
