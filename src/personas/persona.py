"""
Persona entity modeling identity, vocal personality, and room presence.

Encapsulates individual speaker behavior, dialogue utterances, and the physical
dispersion of sound waves corresponding to the persona's physical size in the room.
"""

# Import Modules
from dataclasses import dataclass

from src.config.models import PersonaConfig


@dataclass
class Persona:
    """
    Representation of a distinct vocal persona located within the acoustic scene.
    """

    config: PersonaConfig

    @property
    def identifier(self) -> str:
        """
        Unique string identifier for the persona.

        Returns:
            str: Identifier string.
        """

        return self.config.id

    @property
    def display_name(self) -> str:
        """
        Human-readable persona display name.

        Returns:
            str: Name of the persona.
        """

        return self.config.name

    @property
    def voice_model(self) -> str:
        """
        Piper-TTS voice model file name.

        Returns:
            str: Voice model filename or path.
        """

        return self.config.voice_model

    @property
    def speaker_id(self) -> int | None:
        """
        Speaker identifier within a multi-speaker voice model.

        Returns:
            int | None: Speaker ID or None for single-speaker models.
        """

        return self.config.speaker_id

    @property
    def position(self) -> tuple[float, float, float]:
        """
        Central 3D spatial position of the persona in the room.

        Returns:
            tuple[float, float, float]: Coordinate tuple (x, y, z) in meters.
        """

        return self.config.position

    @property
    def size(self) -> float:
        """
        Effective physical acoustic size of the persona in meters.

        Returns:
            float: Spatial radius or aperture in meters.
        """

        return self.config.size

    def calculate_source_points(self) -> list[tuple[float, float, float]]:
        """
        Compute acoustic emission points representing the persona's physical size.

        When size is zero, returns a single point source. When size is positive,
        models an extended acoustic aperture via a cluster of distributed points.

        Returns:
            list[tuple[float, float, float]]: List of 3D emission coordinates.
        """

        # Return single point source if size is negligible
        center_x: float = self.position[0]
        center_y: float = self.position[1]
        center_z: float = self.position[2]

        if self.size <= 0.0:
            return [(center_x, center_y, center_z)]

        # Calculate dispersion radius from physical size
        radius: float = self.size / 2.0

        # Construct spatial cluster modeling head and torso acoustic radiation
        points: list[tuple[float, float, float]] = [
            (center_x, center_y, center_z),
            (center_x + radius, center_y, center_z),
            (center_x - radius, center_y, center_z),
            (center_x, center_y + radius, center_z),
            (center_x, center_y - radius, center_z),
        ]

        return points
