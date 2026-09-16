"""
Text-to-speech synthesis engine wrapping Piper-TTS with fallback support.

Provides synthesis capabilities from text to floating point audio arrays while
applying vocal personality modifiers like speaking rate, pitch variability, and gain.
"""

# Import Modules
from abc import ABC, abstractmethod
from pathlib import Path

import logging

import numpy as np
from piper.voice import PiperVoice
from piper.config import SynthesisConfig

from src.common.types import AudioArray
from src.config.models import PersonalityConfig
from src.tts.voice_downloader import download_piper_voice
from src.common.audio_utils import (
    resample_audio,
    generate_synthetic_fallback_tone,
)

logger: logging.Logger = logging.getLogger(__name__)


class BaseSynthesizer(ABC):
    """
    Abstract interface for speech synthesis implementations.
    """

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_model: str,
        speaker_id: int | None,
        personality: PersonalityConfig,
        target_sample_rate: int,
    ) -> AudioArray:
        """
        Synthesize text into a single-channel floating-point audio array.

        Args:
            text (str): Spoken phrase to synthesize.
            voice_model (str): Name or path of the voice model.
            speaker_id (int | None): Speaker ID for multi-speaker models.
            personality (PersonalityConfig): Vocal personality configuration.
            target_sample_rate (int): Desired output sampling rate.

        Returns:
            AudioArray: Synthesized audio waveform.
        """


class PiperSynthesizer(BaseSynthesizer):
    """
    Piper-TTS synthesis engine utilizing local ONNX neural voice models.
    """

    def __init__(
        self,
        voices_dir: Path,
        auto_download: bool = True,
    ) -> None:
        """
        Initialize Piper synthesis engine with a directory of voice models.

        Args:
            voices_dir (Path): Directory containing .onnx and .onnx.json files.
            auto_download (bool): Automatically download missing models from HuggingFace.
        """

        # Store voices directory and initialize voice cache
        self.voices_dir: Path = voices_dir
        self.auto_download: bool = auto_download
        self.voice_cache: dict[str, PiperVoice] = {}

    def resolve_model_path(self, voice_model: str) -> Path:
        """
        Locate full filesystem path for a requested voice model name.

        Args:
            voice_model (str): Filename or relative path of the model.

        Returns:
            Path: Resolved absolute path to the ONNX file.

        Raises:
            FileNotFoundError: If model file cannot be found or downloaded.
        """

        # Check direct path first
        direct_path: Path = Path(voice_model)
        if direct_path.is_file():
            return direct_path

        # Check inside voices directory
        candidate_path: Path = self.voices_dir / voice_model
        if candidate_path.is_file():
            return candidate_path

        # Check with .onnx suffix if omitted
        if not voice_model.endswith(".onnx"):
            candidate_with_ext: Path = self.voices_dir / f"{voice_model}.onnx"
            if candidate_with_ext.is_file():
                return candidate_with_ext

        # Recursive search in voices_dir across hierarchical subdirectories
        target_filename: str = (
            voice_model if voice_model.endswith(".onnx") else f"{voice_model}.onnx"
        )
        target_stem: str = (
            voice_model[:-5] if voice_model.endswith(".onnx") else voice_model
        )

        if self.voices_dir.is_dir():
            for file_path in self.voices_dir.rglob("*.onnx"):
                if (
                    file_path.name == target_filename
                    or file_path.stem == target_stem
                    or file_path.parent.name == voice_model
                ):
                    return file_path

        # Attempt automatic download from HuggingFace
        if self.auto_download:
            voice_key: str = (
                voice_model[:-5] if voice_model.endswith(".onnx") else voice_model
            )
            voice_key = Path(voice_key).name
            logger.info(
                "Voice '%s' not found locally in '%s'. Attempting automatic download...",
                voice_key,
                self.voices_dir,
            )
            try:
                downloaded_onnx, _ = download_piper_voice(voice_key, self.voices_dir)
                if downloaded_onnx.is_file():
                    logger.info("Successfully downloaded '%s' automatically", voice_key)
                    return downloaded_onnx
            except (KeyError, ValueError, OSError, RuntimeError) as exc:
                logger.warning(
                    "Automatic download for '%s' failed (%s). Checking fallbacks...",
                    voice_key,
                    exc,
                )

        # Fallback to any existing ONNX model in voices_dir (recursive)
        if self.voices_dir.is_dir():
            available_models: list[Path] = list(self.voices_dir.rglob("*.onnx"))
            if available_models:
                logger.warning(
                    "Model '%s' not found in '%s'. Falling back to local '%s'",
                    voice_model,
                    self.voices_dir,
                    available_models[0].name,
                )
                return available_models[0]

        raise FileNotFoundError(
            f"Voice model '{voice_model}' was not found in '{self.voices_dir}'. "
            "Please download it using download_voice.py."
        )

    def load_voice(self, model_path: Path) -> PiperVoice:
        """
        Load or retrieve a cached PiperVoice model instance.

        Args:
            model_path (Path): Path to .onnx model file.

        Returns:
            PiperVoice: Loaded Piper voice instance.
        """

        # Check in-memory cache
        path_key: str = str(model_path.resolve())
        if path_key in self.voice_cache:
            return self.voice_cache[path_key]

        # Locate accompanying configuration JSON file
        config_path: Path = Path(f"{model_path}.json")
        if not config_path.is_file():
            # Check alternative naming: model_name.onnx.json
            alt_config_path: Path = model_path.with_suffix(".onnx.json")
            if alt_config_path.is_file():
                config_path = alt_config_path

        logger.info("Loading Piper voice model from %s", model_path)

        # Load voice model into memory
        loaded_voice: PiperVoice = PiperVoice.load(
            str(model_path),
            config_path=str(config_path) if config_path.is_file() else None,
        )

        self.voice_cache[path_key] = loaded_voice
        return loaded_voice

    def synthesize(
        self,
        text: str,
        voice_model: str,
        speaker_id: int | None,
        personality: PersonalityConfig,
        target_sample_rate: int,
    ) -> AudioArray:
        """
        Synthesize text into an audio array matching target_sample_rate.

        Args:
            text (str): Text to synthesize.
            voice_model (str): Name or path of voice model.
            speaker_id (int | None): Speaker ID for multi-speaker models.
            personality (PersonalityConfig): Vocal personality parameters.
            target_sample_rate (int): Output sampling rate in Hertz.

        Returns:
            AudioArray: Synthesized audio waveform.
        """

        # Resolve voice model path and load voice
        model_path: Path = self.resolve_model_path(voice_model)
        voice: PiperVoice = self.load_voice(model_path)

        # Build synthesis configuration from personality attributes
        syn_config: SynthesisConfig = SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=personality.length_scale,
            noise_scale=personality.noise_scale,
            noise_w_scale=personality.noise_w_scale,
            volume=personality.volume,
        )

        # Generate audio chunks
        chunk_list: list[AudioArray] = []
        source_sr: int = 16000

        for chunk in voice.synthesize(text, syn_config=syn_config):
            source_sr = chunk.sample_rate
            chunk_list.append(chunk.audio_float_array.astype(np.float64))

        # Handle empty speech case
        if not chunk_list:
            return np.zeros(0, dtype=np.float64)

        # Concatenate audio chunks
        raw_audio: AudioArray = np.concatenate(chunk_list)

        # Resample to target simulation sample rate if necessary
        return resample_audio(raw_audio, source_sr, target_sample_rate)


class MockSynthesizer(BaseSynthesizer):
    """
    Offline fallback synthesizer generating synthetic tones for testing.
    """

    def synthesize(
        self,
        text: str,
        voice_model: str,
        speaker_id: int | None,
        personality: PersonalityConfig,
        target_sample_rate: int,
    ) -> AudioArray:
        """
        Generate synthetic speech-like tone proportional to text length.

        Args:
            text (str): Utterance text used to determine duration.
            voice_model (str): Voice model name (unused in mock).
            speaker_id (int | None): Speaker ID (unused in mock).
            personality (PersonalityConfig): Personality determining tone pitch and speed.
            target_sample_rate (int): Output sampling rate in Hertz.

        Returns:
            AudioArray: Generated synthetic audio waveform.
        """

        # Unused arguments logged for debug clarity
        _ = (voice_model, speaker_id)

        # Estimate duration from word count and speaking rate
        words_count: int = max(1, len(text.split()))
        duration_s: float = (words_count * 0.35) * personality.length_scale

        # Base frequency modified slightly by persona personality
        base_pitch: float = 200.0 * personality.noise_scale

        # Generate speech-like tone
        tone: AudioArray = generate_synthetic_fallback_tone(
            duration_s,
            target_sample_rate,
            base_freq=base_pitch,
        )

        return tone * personality.volume
