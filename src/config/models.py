"""
Data model definitions for acoustic scene configurations.

Defines strongly-typed dataclasses representing rooms, smart assistant microphones,
personas, speech personalities, spoken utterances, and procedural dataset generation.
"""

# Import Modules
from dataclasses import field, dataclass
import enum
from pathlib import Path

from src.common.constants import (
    DEFAULT_ABSORPTION,
    DEFAULT_MAX_ORDER,
    DEFAULT_MIC_RADIUS,
    DEFAULT_MIC_SPACING,
    DEFAULT_SAMPLE_RATE,
)
from src.common.types import MicrophoneType


class VocalStyle(str, enum.Enum):
    """
    Vocal expression style applied to an utterance segment.
    """

    NORMAL = "normal"
    SHOUTING = "shouting"
    LAUGHTER = "laughter"
    INTERRUPTION = "interruption"


class ConversationalStyle(str, enum.Enum):
    """
    Social scenario setting conversational turn-taking and emotional intensity.
    """

    PARTY = "party"
    MEETING = "meeting"
    ARGUMENT = "argument"
    ASSISTANT = "assistant"
    MIXED = "mixed"


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
    language: str = "en"
    vocal_style: VocalStyle = VocalStyle.NORMAL
    group_id: int = 0
    turn_order: int = 0


@dataclass
class PersonaConfig:
    """
    Physical and vocal attributes of a persona situated within the room.
    """

    id: str = "persona"
    name: str = "Persona"
    voice_model: str = "en_US-lessac-low.onnx"
    speaker_id: int | None = None
    language: str = "en"
    gender: str = "unspecified"
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


@dataclass
class DatasetSampleConfig:
    """
    Parameters governing procedural generation of a single dataset sample variation.
    """

    sample_id: str = "sample_001"
    duration_s: float | None = None
    min_sentences: int = 100
    overlap_rate: float = 0.20
    shout_rate: float = 0.04
    laugh_rate: float = 0.05
    ambient_snr_db: float = 38.0
    style: ConversationalStyle = ConversationalStyle.MIXED
    languages: list[str] = field(default_factory=lambda: ["en"])
    use_llm: bool = False
    llm_url: str = "http://127.0.0.1:8080/v1"
    ambiance_preset: str = "random"
    num_constraint_words: int = 3
    llm_temperature: float = 0.7
    allow_parallel: bool = True
    disable_effects: bool = False


@dataclass
class BatchGenerationConfig:
    """
    Configuration parameters governing procedural dataset batch generation.
    """

    num_samples: int = 10
    duration_range: tuple[float, float] | None = None
    min_sentences: int = 100
    speakers_range: tuple[int, int] = (4, 10)
    languages: list[str] = field(default_factory=lambda: ["en"])
    style: ConversationalStyle = ConversationalStyle.MIXED
    voices_dir: Path = field(default_factory=lambda: Path("data/piper_voices"))
    output_dir: Path = field(default_factory=lambda: Path("data/output"))
    use_mock_tts: bool = False
    export_isolated_stems: bool = True
    use_llm: bool = False
    llm_url: str = "http://127.0.0.1:8080/v1"
    ambiance_preset: str = "random"
    num_constraint_words: int = 3
    llm_temperature: float = 0.7
    parallel_prob: float = 0.5
    disable_effects: bool = False
