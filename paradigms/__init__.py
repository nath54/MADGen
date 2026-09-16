"""
Multi-Speaker Audio Recognition & Diarization Evaluation Paradigms for MADGen.
"""

# Import Modules
import importlib
from typing import Any, cast
from paradigms.common.base_pipeline import BaseParadigmPipeline
from paradigms.common.weights_manager import WeightsManager

PARADIGM_MODULE_MAP: dict[str, tuple[str, str]] = {
    "1": (
        "paradigms.1_diarize_then_transcribe.models.silero_whisper.pipeline",
        "SileroWhisperPipeline",
    ),
    "diarize_then_transcribe": (
        "paradigms.1_diarize_then_transcribe.models.silero_whisper.pipeline",
        "SileroWhisperPipeline",
    ),
    "2": (
        "paradigms.2_separate_then_transcribe.models.oracle_irm_whisper.pipeline",
        "OracleIRMWhisperPipeline",
    ),
    "separate_then_transcribe": (
        "paradigms.2_separate_then_transcribe.models.oracle_irm_whisper.pipeline",
        "OracleIRMWhisperPipeline",
    ),
    "3": (
        "paradigms.3_spatial_multichannel.models.mvdr_beamformer.pipeline",
        "MVDRBeamformerPipeline",
    ),
    "spatial_multichannel": (
        "paradigms.3_spatial_multichannel.models.mvdr_beamformer.pipeline",
        "MVDRBeamformerPipeline",
    ),
    "4": (
        "paradigms.4_realtime_streaming.models.streaming_assistant.pipeline",
        "StreamingSmartAssistantPipeline",
    ),
    "realtime_streaming": (
        "paradigms.4_realtime_streaming.models.streaming_assistant.pipeline",
        "StreamingSmartAssistantPipeline",
    ),
}


def get_pipeline(
    paradigm_key: str | int,
    weights_manager: WeightsManager | None = None,
    whisper_model_size: str = "tiny",
    **kwargs: Any,
) -> BaseParadigmPipeline:
    """
    Factory function to instantiate any paradigm pipeline by identifier.

    Args:
        paradigm_key (str | int): Identifier (e.g. '1', '2', '3', '4' or name).
        weights_manager (WeightsManager | None): Shared model cache.
        whisper_model_size (str): Faster-Whisper model size.
        **kwargs: Extra parameters forwarded to pipeline constructor.

    Returns:
        BaseParadigmPipeline: Loaded pipeline instance.
    """

    key_str = str(paradigm_key).lower().strip()
    if key_str not in PARADIGM_MODULE_MAP:
        raise ValueError(
            f"Unknown paradigm key: {paradigm_key}. Supported: {list(PARADIGM_MODULE_MAP.keys())}"
        )

    mod_path, class_name = PARADIGM_MODULE_MAP[key_str]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, class_name)
    instance = cls(weights_manager=weights_manager, whisper_model_size=whisper_model_size, **kwargs)
    return cast(BaseParadigmPipeline, instance)
