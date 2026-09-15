"""
Data model definitions for acoustic scene configurations.

Defines strongly-typed dataclasses representing rooms, smart assistant microphones,
personas, speech personalities, and spoken utterances.
"""

# Import Modules
from dataclasses import field, dataclass

from src.common.types import MicrophoneType
from src.common.constants import (
    DEFAULT_MAX_ORDER,
    DEFAULT_ABSORPTION,
    DEFAULT_MIC_RADIUS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MIC_SPACING,
)


@dataclass
class PersonalityConfig:
    """
    Vocal personality parameters mapping to Piper-TTS synthesis settings.
    """

    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8
    volume: float = 1.0


@dataclass
class UtteranceConfig:
    """
    A single spoken utterance segment associated with a timeline start time.
    """

    text: str = ""
    start_time_s: float = 0.0


@dataclass
class PersonaConfig:
    """
    Physical and vocal attributes of a persona situated within the room.
    """

    id: str = "persona"
    name: str = "Persona"
    voice_model: str = "en_US-lessac-low.onnx"
    speaker_id: int | None = None
    position: tuple[float, float, float] = (2.0, 2.0, 1.5)
    size: float = 0.0
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    utterances: list[UtteranceConfig] = field(default_factory=list)


@dataclass
class MicrophoneConfig:
    """
    Geometry and acoustic placement parameters for the assistant microphone.
    """

    position: tuple[float, float, float] = (0.4, 0.4, 0.85)
    mic_type: MicrophoneType = MicrophoneType.STEREO
    mic_spacing: float = DEFAULT_MIC_SPACING
    mic_radius: float = DEFAULT_MIC_RADIUS
    num_microphones: int = 2


@dataclass
class RoomConfig:
    """
    Acoustic dimensions and reverberation properties of the ShoeBox room.
    """

    dimensions: tuple[float, float, float] = (6.0, 5.0, 2.8)
    absorption: float = DEFAULT_ABSORPTION
    max_order: int = DEFAULT_MAX_ORDER
    sample_rate: int = DEFAULT_SAMPLE_RATE


@dataclass
class SceneConfig:
    """
    Composite configuration representing the complete simulation environment.
    """

    room: RoomConfig = field(default_factory=RoomConfig)
    assistant: MicrophoneConfig = field(default_factory=MicrophoneConfig)
    personas: list[PersonaConfig] = field(default_factory=list)
