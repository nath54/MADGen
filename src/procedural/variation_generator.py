"""
Randomized scene variation generator for diverse training and evaluation datasets.

Procedurally synthesizes room dimensions, acoustic wall absorptions, assistant
corner placements, microphone array geometries, and multi-speaker positions.
"""

# Import Modules
import random

from src.common.types import MicrophoneType
from src.tts.voice_catalog import (
    VoiceProfile,
    sample_distinct_voice_profiles,
)
from src.config.models import (
    RoomConfig,
    SceneConfig,
    PersonaConfig,
    MicrophoneConfig,
    PersonalityConfig,
    DatasetSampleConfig,
)

# Mapping from language code to recommended default Piper voice models
LANGUAGE_VOICE_MAP: dict[str, list[str]] = {
    "en": ["en_US-lessac-low.onnx", "en_US-lessac-medium.onnx"],
    "fr": ["fr_FR-siwis-medium.onnx"],
    "es": ["es_ES-davefx-medium.onnx"],
    "de": ["de_DE-karlsson-low.onnx", "de_DE-thorsten-medium.onnx"],
}


def sample_room_dimensions(num_speakers: int = 4) -> tuple[float, float, float]:
    """
    Generate randomized 3D room dimensions (Lx, Ly, Lz) in meters.

    Args:
        num_speakers (int): Number of speakers to accommodate in the room.

    Returns:
        tuple[float, float, float]: Room width, length, and ceiling height.
    """

    # Width and length scaled to accommodate speaker count
    base_min: float = 5.0 if num_speakers <= 4 else 5.0 + (num_speakers - 4) * 0.5
    base_max: float = 8.5 if num_speakers <= 4 else 8.5 + (num_speakers - 4) * 0.6
    dim_x: float = round(random.uniform(base_min, base_max), 2)
    dim_y: float = round(random.uniform(base_min, base_max), 2)
    dim_z: float = round(random.uniform(2.5, 3.4), 2)
    return (dim_x, dim_y, dim_z)


def sample_assistant_corner_position(
    room_dims: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Select one of four room corners and generate safe placement coordinates.

    Args:
        room_dims (tuple[float, float, float]): Dimensions (Lx, Ly, Lz) of the room.

    Returns:
        tuple[float, float, float]: Coordinates (x, y, z) for the assistant device.
    """

    # Pick wall clearance between 0.3m and 0.55m
    offset_x: float = round(random.uniform(0.3, 0.55), 2)
    offset_y: float = round(random.uniform(0.3, 0.55), 2)

    # Table or countertop height between 0.75m and 1.1m
    height_z: float = round(random.uniform(0.75, 1.1), 2)

    # Pick one of the four room corners (0: bottom-left, 1: bottom-right, 2: top-left, 3: top-right)
    corner_index: int = random.randint(0, 3)

    if corner_index == 0:
        return (offset_x, offset_y, height_z)
    if corner_index == 1:
        return (room_dims[0] - offset_x, offset_y, height_z)
    if corner_index == 2:
        return (offset_x, room_dims[1] - offset_y, height_z)
    return (room_dims[0] - offset_x, room_dims[1] - offset_y, height_z)


def sample_persona_spatial_position(
    room_dims: tuple[float, float, float],
    assistant_pos: tuple[float, float, float],
    existing_positions: list[tuple[float, float, float]] | None = None,
) -> tuple[float, float, float]:
    """
    Generate randomized 3D position for a speaker ensuring clearance from walls and mic.

    Args:
        room_dims (tuple[float, float, float]): Room dimensions (Lx, Ly, Lz).
        assistant_pos (tuple[float, float, float]): Assistant microphone position.
        existing_positions (list[tuple[float, float, float]] | None): Other speaker positions.

    Returns:
        tuple[float, float, float]: Safe 3D spatial position coordinates.
    """

    margin: float = 0.8
    min_dist_to_mic: float = 1.0
    min_dist_to_speakers: float = 0.70

    pos_x: float = room_dims[0] / 2.0
    pos_y: float = room_dims[1] / 2.0
    pos_z: float = 1.70

    for iteration in range(60):
        pos_x = round(random.uniform(margin, room_dims[0] - margin), 2)
        pos_y = round(random.uniform(margin, room_dims[1] - margin), 2)
        pos_z = round(
            random.uniform(1.65, 1.80) if random.random() < 0.6 else random.uniform(1.10, 1.30),
            2,
        )

        dx: float = pos_x - assistant_pos[0]
        dy: float = pos_y - assistant_pos[1]
        if (dx**2 + dy**2) ** 0.5 < min_dist_to_mic:
            continue

        # Check distance to already placed speakers
        req_dist: float = min_dist_to_speakers if iteration < 40 else min_dist_to_speakers * 0.6
        if existing_positions and any(
            ((pos_x - ox) ** 2 + (pos_y - oy) ** 2) ** 0.5 < req_dist
            for ox, oy, _ in existing_positions
        ):
            continue

        return (pos_x, pos_y, pos_z)

    return (pos_x, pos_y, pos_z)


def build_random_persona(
    persona_index: int,
    language: str,
    room_dims: tuple[float, float, float],
    assistant_pos: tuple[float, float, float],
    voice_model: str | None = None,
    speaker_id: int | None = None,
    gender: str = "unspecified",
    existing_positions: list[tuple[float, float, float]] | None = None,
) -> PersonaConfig:
    """
    Generate a single randomized persona configuration.

    Args:
        persona_index (int): Numeric identifier index.
        language (str): Language assigned to persona.
        room_dims (tuple[float, float, float]): Room dimensions.
        assistant_pos (tuple[float, float, float]): Assistant position.
        voice_model (str | None): Optional specific ONNX voice model filename.
        speaker_id (int | None): Optional multi-speaker ID.
        gender (str): Vocal gender representation ('female' or 'male').
        existing_positions (list[tuple[float, float, float]] | None): Placed speakers.

    Returns:
        PersonaConfig: Randomized persona.
    """

    # Pick voice model mapped to language if not provided
    if voice_model is None:
        voice_models: list[str] = LANGUAGE_VOICE_MAP.get(
            language,
            ["en_US-lessac-low.onnx"],
        )
        selected_model: str = random.choice(voice_models)
    else:
        selected_model = voice_model

    # Position in room with inter-speaker clearance
    pos: tuple[float, float, float] = sample_persona_spatial_position(
        room_dims=room_dims,
        assistant_pos=assistant_pos,
        existing_positions=existing_positions,
    )

    # 35% chance point source (0.0), 65% chance extended physical size (0.15 - 0.40m)
    size: float = 0.0 if random.random() < 0.35 else round(random.uniform(0.15, 0.40), 2)

    # Randomized vocal personality traits
    personality: PersonalityConfig = PersonalityConfig(
        length_scale=round(random.uniform(0.85, 1.25), 2),
        noise_scale=round(random.uniform(0.50, 0.85), 2),
        noise_w_scale=round(random.uniform(0.65, 0.95), 2),
        volume=round(random.uniform(0.85, 1.20), 2),
    )

    display_name: str = (
        f"Speaker {persona_index} ({gender})"
        if gender != "unspecified"
        else f"Speaker {persona_index}"
    )

    return PersonaConfig(
        id=f"speaker_{persona_index}",
        name=display_name,
        voice_model=selected_model,
        speaker_id=speaker_id,
        language=language,
        gender=gender,
        position=pos,
        size=size,
        personality=personality,
    )


def build_random_room(num_speakers: int = 4) -> RoomConfig:
    """
    Synthesize randomized ShoeBox room geometry and reverberation parameters.

    Args:
        num_speakers (int): Number of speakers to accommodate in room sizing.

    Returns:
        RoomConfig: Randomized room configuration.
    """

    room_dims: tuple[float, float, float] = sample_room_dimensions(num_speakers)
    absorption: float = round(random.uniform(0.10, 0.38), 2)
    max_order: int = random.choice([2, 3, 4])

    return RoomConfig(
        dimensions=room_dims,
        absorption=absorption,
        max_order=max_order,
        sample_rate=16000,
    )


def build_random_assistant(
    room_dims: tuple[float, float, float],
) -> MicrophoneConfig:
    """
    Synthesize randomized assistant microphone placement and geometry.

    Args:
        room_dims (tuple[float, float, float]): Room dimensions (Lx, Ly, Lz).

    Returns:
        MicrophoneConfig: Configured assistant microphone array.
    """

    mic_pos: tuple[float, float, float] = sample_assistant_corner_position(room_dims)
    mic_type: MicrophoneType = random.choice(
        [MicrophoneType.MONO, MicrophoneType.STEREO, MicrophoneType.CIRCULAR]
    )

    return MicrophoneConfig(
        position=mic_pos,
        mic_type=mic_type,
        mic_spacing=round(random.uniform(0.06, 0.12), 2),
        mic_radius=round(random.uniform(0.03, 0.05), 2),
        num_microphones=4 if mic_type == MicrophoneType.CIRCULAR else 2,
    )


def generate_random_scene(
    sample_config: DatasetSampleConfig,
    num_speakers: int,
) -> SceneConfig:
    """
    Construct fully randomized SceneConfig based on dataset variation parameters.

    Args:
        sample_config (DatasetSampleConfig): Sample generation parameters.
        num_speakers (int): Number of concurrent speakers to instantiate.

    Returns:
        SceneConfig: Fully initialized randomized scene configuration.
    """

    # Step 1: Synthesize room scaled to speaker count and assistant placement
    room: RoomConfig = build_random_room(num_speakers)
    assistant: MicrophoneConfig = build_random_assistant(room.dimensions)

    # Step 2: Sample strictly distinct voice profiles with balanced gender diversity
    personas: list[PersonaConfig] = []
    allowed_langs: list[str] = sample_config.languages if sample_config.languages else ["en"]

    sampled_profiles: list[tuple[VoiceProfile, int | None, str]] = sample_distinct_voice_profiles(
        count=num_speakers,
        languages=allowed_langs,
        ensure_gender_diversity=True,
    )

    existing_positions: list[tuple[float, float, float]] = []
    for idx, (v_prof, spk_id, gender) in enumerate(sampled_profiles, start=1):
        persona: PersonaConfig = build_random_persona(
            persona_index=idx,
            language=v_prof.lang_family,
            room_dims=room.dimensions,
            assistant_pos=assistant.position,
            voice_model=f"{v_prof.key}.onnx",
            speaker_id=spk_id,
            gender=gender,
            existing_positions=existing_positions,
        )
        existing_positions.append(persona.position)
        personas.append(persona)

    return SceneConfig(room=room, assistant=assistant, personas=personas)
