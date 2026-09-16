"""
Downloader utility for Piper-TTS voice models and configuration metadata.

Retrieves official voice models from HuggingFace and stores them into the designated
voice directory for local text-to-speech synthesis.
"""

# Import Modules
import typing
from pathlib import Path

import json
import logging
import urllib.error
import urllib.request

logger: logging.Logger = logging.getLogger(__name__)

VOICES_INDEX_URL: str = "https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json"
VOICES_BASE_URL: str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
LOCAL_VOICES_PATH: Path = Path("data/piper_voices/voices.json")


def fetch_voices_catalog() -> dict[str, typing.Any]:
    """
    Fetch the catalog mapping of all available Piper-TTS models from HuggingFace.

    Returns:
        dict[str, typing.Any]: Parsed JSON catalog dictionary.

    Raises:
        RuntimeError: If downloading or parsing fails.
    """

    # Check local cached copy first
    if LOCAL_VOICES_PATH.is_file():
        try:
            with LOCAL_VOICES_PATH.open("r", encoding="utf-8") as file_stream:
                catalog_dict: dict[str, typing.Any] = json.load(file_stream)
                return catalog_dict
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read local voices.json: %s", exc)

    # Fetch catalog payload via HTTP
    try:
        with urllib.request.urlopen(VOICES_INDEX_URL, timeout=15) as response:
            catalog_bytes: bytes = response.read()
            remote_dict: dict[str, typing.Any] = json.loads(catalog_bytes.decode("utf-8"))
            return remote_dict
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Failed to fetch Piper voices catalog: {exc}") from exc


def download_file_if_missing(
    remote_url: str,
    target_file: Path,
) -> None:
    """
    Download a remote file to a local path if not already present.

    Args:
        remote_url (str): Source HTTP URL.
        target_file (Path): Destination filesystem location.
    """

    # Check if file exists and is non-empty
    if target_file.is_file() and target_file.stat().st_size > 0:
        logger.info("File already exists: %s", target_file)
        return

    # Ensure parent folder exists
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Perform download stream
    logger.info("Downloading from %s to %s", remote_url, target_file)

    with urllib.request.urlopen(remote_url, timeout=60) as response:
        with target_file.open("wb") as out_file:
            while True:
                chunk: bytes = response.read(65536)
                if not chunk:
                    break
                out_file.write(chunk)

    logger.info("Saved %s successfully", target_file.name)


def resolve_voice_folder(
    voice_key: str,
    target_dir: Path,
    voice_info: dict[str, typing.Any] | None = None,
) -> Path:
    """
    Resolve structured destination folder for a Piper voice model.

    Returns path adhering to BASE_LANG/LANG_DETAIL/persona_name.

    Args:
        voice_key (str): Identifier of the voice (e.g. 'en_US-lessac-low').
        target_dir (Path): Base directory for Piper voices.
        voice_info (dict[str, typing.Any] | None): Optional metadata from voices.json.

    Returns:
        Path: Resolved directory path for the voice files.
    """

    family: str = ""
    code: str = ""
    name: str = ""

    if voice_info is not None:
        lang_dict: dict[str, typing.Any] = voice_info.get("language", {})
        code = str(lang_dict.get("code", ""))
        family = str(lang_dict.get("family", code.split("_", maxsplit=1)[0]))
        name = str(voice_info.get("name", ""))

    if not family or not code or not name:
        parts: list[str] = voice_key.split("-")
        code = parts[0] if parts else "en_US"
        family = code.split("_", maxsplit=1)[0]
        name = parts[1] if len(parts) > 1 else "speaker"

    # Avoid duplicating subdirectories if target_dir already points to persona folder
    if target_dir.name == name:
        return target_dir

    return target_dir / family / code / name


def download_piper_voice(
    voice_key: str,
    target_dir: Path,
) -> tuple[Path, Path]:
    """
    Download both .onnx model and .onnx.json metadata files for a Piper voice.

    Args:
        voice_key (str): Identifier of the voice (e.g. 'en_US-lessac-low').
        target_dir (Path): Directory where voice files should be saved.

    Returns:
        tuple[Path, Path]: Tuple containing (onnx_path, json_path).

    Raises:
        KeyError: If voice_key is not found in the official catalog.
    """

    # Retrieve catalog
    catalog: dict[str, typing.Any] = fetch_voices_catalog()

    # Verify voice key exists
    if voice_key not in catalog:
        raise KeyError(
            f"Voice '{voice_key}' not found in Piper catalog. "
            f"Available sample: {list(catalog.keys())[:5]}"
        )

    # Extract remote relative paths
    voice_info: dict[str, typing.Any] = catalog[voice_key]
    files_dict: dict[str, typing.Any] = voice_info.get("files", {})

    # Locate onnx and json paths in catalog
    onnx_rel_path: str | None = None
    json_rel_path: str | None = None

    for rel_path in files_dict.keys():
        if rel_path.endswith(".onnx"):
            onnx_rel_path = rel_path
        elif rel_path.endswith(".onnx.json"):
            json_rel_path = rel_path

    if not onnx_rel_path or not json_rel_path:
        raise ValueError(f"Incomplete files for voice {voice_key} in catalog")

    # Construct structured destination paths
    voice_folder: Path = resolve_voice_folder(voice_key, target_dir, voice_info)
    dest_onnx: Path = voice_folder / f"{voice_key}.onnx"
    dest_json: Path = voice_folder / f"{voice_key}.onnx.json"

    # Download ONNX model file
    download_file_if_missing(VOICES_BASE_URL + onnx_rel_path, dest_onnx)

    # Download config JSON file
    download_file_if_missing(VOICES_BASE_URL + json_rel_path, dest_json)

    return (dest_onnx, dest_json)


def download_voices_for_languages(
    languages: list[str],
    target_dir: Path,
) -> list[str]:
    """
    Batch download all available Piper voices matching specified languages.

    Args:
        languages (list[str]): Language code prefixes (e.g. ['en', 'fr']).
        target_dir (Path): Destination directory.

    Returns:
        list[str]: Successfully downloaded voice keys.
    """

    catalog: dict[str, typing.Any] = fetch_voices_catalog()
    norm_langs: set[str] = {lang.lower() for lang in languages}
    matching_keys: list[str] = []

    for key, info in catalog.items():
        lang_dict: dict[str, typing.Any] = info.get("language", {})
        code: str = str(lang_dict.get("code", ""))
        family: str = str(lang_dict.get("family", code.split("_", maxsplit=1)[0]))

        if family.lower() in norm_langs or code.lower() in norm_langs:
            matching_keys.append(key)

    downloaded: list[str] = []

    for voice_key in matching_keys:
        try:
            download_piper_voice(voice_key, target_dir)
            downloaded.append(voice_key)
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as err:
            logger.warning("Failed downloading voice '%s': %s", voice_key, err)

    return downloaded
