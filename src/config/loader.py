"""
Configuration loading and validation logic for acoustic scenes.

Parses scene configuration files from JSON format, validates spatial bounds,
and constructs fully validated SceneConfig instances.
"""

# Import Modules
import typing
from pathlib import Path

import json

from src.common.types import MicrophoneType
from src.config.models import (
    RoomConfig,
    SceneConfig,
    PersonaConfig,
    UtteranceConfig,
    MicrophoneConfig,
    PersonalityConfig,
)


def validate_position_in_room(
    position: tuple[float, float, float],
    room_dims: tuple[float, float, float],
    entity_name: str,
) -> None:
    """
    Ensure spatial coordinates reside safely within room boundaries.

    Args:
        position (tuple[float, float, float]): 3D coordinates (x, y, z).
        room_dims (tuple[float, float, float]): Room dimensions (Lx, Ly, Lz).
        entity_name (str): Label for error reporting.

    Raises:
        ValueError: If position falls outside room boundaries.
    """

    # Extract coordinates and room boundaries
    pos_x: float = position[0]
    pos_y: float = position[1]
    pos_z: float = position[2]
    max_x: float = room_dims[0]
    max_y: float = room_dims[1]
    max_z: float = room_dims[2]

    # Verify X axis bounds
    if not 0.0 < pos_x < max_x:
        raise ValueError(
            f"{entity_name} X coordinate {pos_x} is out of room bounds (0, {max_x})"
        )

    # Verify Y axis bounds
    if not 0.0 < pos_y < max_y:
        raise ValueError(
            f"{entity_name} Y coordinate {pos_y} is out of room bounds (0, {max_y})"
        )

    # Verify Z axis bounds
    if not 0.0 < pos_z < max_z:
        raise ValueError(
            f"{entity_name} Z coordinate {pos_z} is out of room bounds (0, {max_z})"
        )


def parse_room_config(raw_room: dict[str, typing.Any]) -> RoomConfig:
    """
    Parse room parameters dictionary into a RoomConfig dataclass.

    Args:
        raw_room (dict[str, typing.Any]): Raw dictionary from JSON configuration.

    Returns:
        RoomConfig: Parsed room configuration.
    """

    # Extract raw room dimensions
    dim_list: list[float] = [float(v) for v in raw_room.get("dimensions", [6.0, 5.0, 2.8])]
    dims: tuple[float, float, float] = (dim_list[0], dim_list[1], dim_list[2])

    # Extract acoustic parameters
    absorption: float = float(raw_room.get("absorption", 0.2))
    max_order: int = int(raw_room.get("max_order", 3))
    sample_rate: int = int(raw_room.get("sample_rate", 16000))

    return RoomConfig(
        dimensions=dims,
        absorption=absorption,
        max_order=max_order,
        sample_rate=sample_rate,
    )


def parse_microphone_config(raw_assistant: dict[str, typing.Any]) -> MicrophoneConfig:
    """
    Parse smart assistant parameters into a MicrophoneConfig dataclass.

    Args:
        raw_assistant (dict[str, typing.Any]): Raw assistant dictionary.

    Returns:
        MicrophoneConfig: Parsed microphone configuration.
    """

    # Extract assistant coordinates
    pos_list: list[float] = [float(v) for v in raw_assistant.get("position", [0.4, 0.4, 0.85])]
    pos: tuple[float, float, float] = (pos_list[0], pos_list[1], pos_list[2])

    # Extract microphone geometry attributes
    mic_type_str: str = str(raw_assistant.get("mic_type", "stereo")).lower()
    mic_type: MicrophoneType = MicrophoneType(mic_type_str)
    mic_spacing: float = float(raw_assistant.get("mic_spacing", 0.08))
    mic_radius: float = float(raw_assistant.get("mic_radius", 0.035))
    num_mics: int = int(raw_assistant.get("num_microphones", 2))

    return MicrophoneConfig(
        position=pos,
        mic_type=mic_type,
        mic_spacing=mic_spacing,
        mic_radius=mic_radius,
        num_microphones=num_mics,
    )


def parse_utterance(raw_utterance: dict[str, typing.Any]) -> UtteranceConfig:
    """
    Parse a single utterance dictionary.

    Args:
        raw_utterance (dict[str, typing.Any]): Raw utterance dictionary.

    Returns:
        UtteranceConfig: Parsed utterance dataclass.
    """

    # Extract text content and start timestamp
    text: str = str(raw_utterance.get("text", ""))
    start_time_s: float = float(raw_utterance.get("start_time_s", 0.0))

    return UtteranceConfig(text=text, start_time_s=start_time_s)


def parse_persona(raw_persona: dict[str, typing.Any]) -> PersonaConfig:
    """
    Parse a single persona dictionary.

    Args:
        raw_persona (dict[str, typing.Any]): Raw persona dictionary.

    Returns:
        PersonaConfig: Parsed persona dataclass.
    """

    # Extract identifier and display name
    persona_id: str = str(raw_persona.get("id", "persona"))
    name: str = str(raw_persona.get("name", "Persona"))
    voice_model: str = str(raw_persona.get("voice_model", "en_US-lessac-low.onnx"))

    # Extract speaker index if applicable
    raw_spk: typing.Any = raw_persona.get("speaker_id")
    speaker_id: int | None = int(raw_spk) if raw_spk is not None else None

    # Extract language and vocal gender
    language: str = str(raw_persona.get("language", "en"))
    gender: str = str(raw_persona.get("gender", "unspecified"))

    # Extract spatial placement and size
    pos_list: list[float] = [float(v) for v in raw_persona.get("position", [2.0, 2.0, 1.5])]
    pos: tuple[float, float, float] = (pos_list[0], pos_list[1], pos_list[2])
    size: float = float(raw_persona.get("size", 0.0))

    # Extract personality parameters
    raw_personality: dict[str, typing.Any] = raw_persona.get("personality", {})
    personality: PersonalityConfig = PersonalityConfig(
        length_scale=float(raw_personality.get("length_scale", 1.0)),
        noise_scale=float(raw_personality.get("noise_scale", 0.667)),
        noise_w_scale=float(raw_personality.get("noise_w_scale", 0.8)),
        volume=float(raw_personality.get("volume", 1.0)),
    )

    # Extract dialogue utterances
    raw_utterances: list[dict[str, typing.Any]] = raw_persona.get("utterances", [])
    utterances: list[UtteranceConfig] = [parse_utterance(u) for u in raw_utterances]

    return PersonaConfig(
        id=persona_id,
        name=name,
        voice_model=voice_model,
        speaker_id=speaker_id,
        language=language,
        gender=gender,
        position=pos,
        size=size,
        personality=personality,
        utterances=utterances,
    )


def load_scene_from_dict(raw_data: dict[str, typing.Any]) -> SceneConfig:
    """
    Convert a dictionary into a validated SceneConfig object.

    Args:
        raw_data (dict[str, typing.Any]): Raw loaded dictionary.

    Returns:
        SceneConfig: Fully validated scene configuration.
    """

    # Parse components
    room: RoomConfig = parse_room_config(raw_data.get("room", {}))
    assistant: MicrophoneConfig = parse_microphone_config(raw_data.get("assistant", {}))
    personas: list[PersonaConfig] = [
        parse_persona(p) for p in raw_data.get("personas", [])
    ]

    # Validate assistant position within room
    validate_position_in_room(assistant.position, room.dimensions, "Assistant Microphone")

    # Validate all persona positions within room
    for persona in personas:
        validate_position_in_room(persona.position, room.dimensions, f"Persona '{persona.name}'")

    return SceneConfig(room=room, assistant=assistant, personas=personas)


def load_scene_config(file_path: Path) -> SceneConfig:
    """
    Load and parse a JSON scene configuration file.

    Args:
        file_path (Path): Path to JSON configuration file.

    Returns:
        SceneConfig: Validated scene configuration.

    Raises:
        FileNotFoundError: If configuration file does not exist.
    """

    # Ensure file exists on filesystem
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {file_path}")

    # Read and parse JSON content
    with file_path.open("r", encoding="utf-8") as file_stream:
        raw_data: dict[str, typing.Any] = json.load(file_stream)

    return load_scene_from_dict(raw_data)
