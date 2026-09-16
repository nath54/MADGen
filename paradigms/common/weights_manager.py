"""
Centralized automated model weights management, downloading, and caching.

Ensures zero-setup out-of-the-box execution by detecting missing model checkpoints
and automatically fetching them with progress feedback, hardware-adaptive compute type
selection, and local disk caching.
"""

# Import Modules
from pathlib import Path
from typing import Any
import logging
import urllib.request

import numpy as np
import torch
import onnxruntime as ort
import ctranslate2
from faster_whisper import WhisperModel

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR: Path = Path("models_cache")
SILERO_VAD_URL: str = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)


class SileroVADWrapper:
    """
    Pure-NumPy execution wrapper around Silero VAD v5 ONNX session.
    """

    def __init__(self, session: ort.InferenceSession, sample_rate: int = 16000) -> None:
        self.session: ort.InferenceSession = session
        self.sample_rate: int = sample_rate
        self.window_size: int = 512 if sample_rate == 16000 else 256
        self.context_size: int = 64 if sample_rate == 16000 else 32
        self.sr_tensor: np.ndarray = np.array(sample_rate, dtype=np.int64)
        self.state: np.ndarray = np.zeros((2, 1, 128), dtype=np.float32)
        self.context: np.ndarray = np.zeros((1, self.context_size), dtype=np.float32)

    def reset_states(self) -> None:
        """Reset internal RNN states and streaming context buffer."""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, self.context_size), dtype=np.float32)

    def step(self, chunk: np.ndarray) -> float:
        """
        Process a single frame chunk and return speech probability.
        """
        if len(chunk) != self.window_size:
            if len(chunk) < self.window_size:
                chunk = np.pad(chunk, (0, self.window_size - len(chunk)))
            else:
                chunk = chunk[: self.window_size]

        chunk_2d = chunk.astype(np.float32)[np.newaxis, :]
        x = np.concatenate([self.context, chunk_2d], axis=1)
        out, new_state = self.session.run(
            None,
            {"input": x, "state": self.state, "sr": self.sr_tensor},
        )
        self.state = new_state
        self.context = x[:, -self.context_size :]
        return float(out[0][0])

    def get_speech_intervals(  # pylint: disable=too-many-locals
        self,
        audio_1d: np.ndarray,
        threshold: float = 0.5,
        min_speech_duration_s: float = 0.25,
        min_silence_duration_s: float = 0.3,
    ) -> list[tuple[float, float]]:
        """
        Segment 1D audio array into speech intervals with hysteresis thresholding.
        """
        self.reset_states()
        num_windows = len(audio_1d) // self.window_size
        probs = np.zeros(num_windows, dtype=np.float32)

        for i in range(num_windows):
            chunk = audio_1d[i * self.window_size : (i + 1) * self.window_size]
            probs[i] = self.step(chunk)

        intervals: list[tuple[float, float]] = []
        in_speech = False
        start_idx = 0
        min_speech_frames = max(
            1, int((min_speech_duration_s * self.sample_rate) / self.window_size)
        )
        min_silence_frames = max(
            1, int((min_silence_duration_s * self.sample_rate) / self.window_size)
        )
        silence_count = 0

        for idx, prob in enumerate(probs):
            if prob >= threshold:
                if not in_speech:
                    in_speech = True
                    start_idx = idx
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        end_idx = idx - silence_count
                        if (end_idx - start_idx) >= min_speech_frames:
                            s_s = (start_idx * self.window_size) / self.sample_rate
                            e_s = (end_idx * self.window_size) / self.sample_rate
                            intervals.append((s_s, e_s))
                        in_speech = False
                        silence_count = 0

        if in_speech and (len(probs) - start_idx) >= min_speech_frames:
            intervals.append((
                (start_idx * self.window_size) / self.sample_rate,
                len(audio_1d) / self.sample_rate,
            ))

        return intervals



class WeightsManager:
    """
    Automated weights manager handling downloads, ctransformers, and neural model instances.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """
        Initialize the weights manager.

        Args:
            cache_dir (Path | None): Custom cache directory path (defaults to models_cache/).
        """

        self.cache_dir: Path = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._silero_session: ort.InferenceSession | None = None
        self._whisper_models: dict[str, WhisperModel] = {}
        self._sepformer_model: Any = None

    def get_best_device(self) -> str:
        """
        Determine optimal hardware compute device (CUDA GPU or CPU).

        Returns:
            str: 'cuda' if GPU with supported compute capability (>= 7.0) is available, else 'cpu'.
        """

        if torch.cuda.is_available():
            try:
                cap = torch.cuda.get_device_capability(0)
                if cap[0] >= 7:
                    return "cuda"
                logger.info(
                    "Detected GPU compute capability %s (< 7.0). Using CPU with INT8 optimization.",
                    cap,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Failed checking CUDA device capability: %s", exc)

        return "cpu"

    def get_whisper_compute_type(self, device: str) -> str:
        """
        Select optimal CTranslate2 compute type matching hardware capabilities.

        Args:
            device (str): Compute device ('cuda' or 'cpu').

        Returns:
            str: Supported compute type ('float16', 'float32', 'int8').
        """

        try:
            supported = ctranslate2.get_supported_compute_types(device)
            if device == "cuda":
                if "float16" in supported:
                    return "float16"
                if "float32" in supported:
                    return "float32"
            else:
                if "int8" in supported:
                    return "int8"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed probing CTranslate2 compute types: %s", exc)

        return "default"

    def ensure_silero_vad(self) -> ort.InferenceSession:
        """
        Ensure Silero VAD ONNX model weights are present, downloading if absent.

        Returns:
            ort.InferenceSession: Active ONNX Runtime inference session.
        """

        if self._silero_session is not None:
            return self._silero_session

        vad_path: Path = self.cache_dir / "silero_vad.onnx"

        if not vad_path.is_file() or vad_path.stat().st_size < 1000:
            logger.info("Silero VAD weights not found. Downloading to %s...", vad_path)
            try:
                urllib.request.urlretrieve(SILERO_VAD_URL, str(vad_path))
                logger.info("Successfully downloaded Silero VAD weights (%s)", vad_path)
            except Exception as exc:
                logger.error(
                    "Failed downloading Silero VAD weights from %s: %s", SILERO_VAD_URL, exc
                )
                raise

        # Initialize ONNX inference session
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._silero_session = ort.InferenceSession(str(vad_path), sess_options=opts)
        return self._silero_session

    def ensure_silero_vad_wrapper(self, sample_rate: int = 16000) -> SileroVADWrapper:
        """
        Ensure Silero VAD session is loaded and return high-level inference wrapper.

        Args:
            sample_rate (int): Target sampling frequency.

        Returns:
            SileroVADWrapper: Active VAD wrapper with streaming state tracking.
        """

        sess = self.ensure_silero_vad()
        return SileroVADWrapper(session=sess, sample_rate=sample_rate)


    def ensure_whisper_model(
        self,
        model_size: str = "tiny",
        device: str | None = None,
    ) -> WhisperModel:
        """
        Ensure Faster-Whisper model is present, downloading from Hugging Face if needed.

        Args:
            model_size (str): Whisper model variation ('tiny', 'base', 'small', 'medium').
            device (str | None): Device to run on (None for auto-detection).

        Returns:
            WhisperModel: Loaded and ready Faster-Whisper instance.
        """

        active_device: str = device if device is not None else self.get_best_device()
        compute_type: str = self.get_whisper_compute_type(active_device)
        cache_key: str = f"{model_size}_{active_device}_{compute_type}"

        if cache_key in self._whisper_models:
            return self._whisper_models[cache_key]

        target_dir: Path = self.cache_dir / f"whisper_{model_size}"
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing WhisperModel (%s) on %s [%s]...",
            model_size,
            active_device,
            compute_type,
        )

        try:
            model = WhisperModel(
                model_size_or_path=model_size,
                device=active_device,
                compute_type=compute_type,
                download_root=str(target_dir),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Failed loading WhisperModel on %s (%s). Falling back to CPU...",
                active_device,
                exc,
            )
            model = WhisperModel(
                model_size_or_path=model_size,
                device="cpu",
                compute_type="int8",
                download_root=str(target_dir),
            )

        self._whisper_models[cache_key] = model
        return model

    def ensure_sepformer_model(self, device: str | None = None) -> Any:
        """
        Ensure SpeechBrain SepFormer weights are downloaded and instantiated.

        Args:
            device (str | None): Device to run on (None for auto-detection).

        Returns:
            SepformerSeparation: Loaded SpeechBrain SepFormer model.
        """

        if self._sepformer_model is not None:
            return self._sepformer_model

        active_device: str = device if device is not None else self.get_best_device()
        target_dir: Path = self.cache_dir / "sepformer_wsj02mix"
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing SpeechBrain SepFormer on %s...", active_device)
        from speechbrain.inference.separation import (  # pylint: disable=import-outside-toplevel
            SepformerSeparation,
        )

        run_opts = {"device": active_device} if active_device == "cuda" else {"device": "cpu"}
        self._sepformer_model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir=str(target_dir),
            run_opts=run_opts,
        )
        return self._sepformer_model
