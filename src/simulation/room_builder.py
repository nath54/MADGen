"""
Room acoustic environment construction using Pyroomacoustics.

Configures ShoeBox geometry, boundary absorption coefficients, reflection orders,
and simulation sampling frequency.
"""

# Import Modules
import pyroomacoustics as pra

from src.config.models import RoomConfig


def create_room_material(absorption: float) -> pra.Material:
    """
    Construct a Pyroomacoustics Material object with a specified absorption coefficient.

    Args:
        absorption (float): Acoustic absorption coefficient between 0.0 and 1.0.

    Returns:
        pra.Material: Configured wall material.

    Raises:
        ValueError: If absorption is outside the valid range [0.0, 1.0].
    """

    # Validate absorption bounds
    if not 0.0 <= absorption <= 1.0:
        raise ValueError(
            f"Absorption coefficient must be between 0.0 and 1.0, got {absorption}"
        )

    # Instantiate wall material
    return pra.Material(absorption)


def build_room(room_config: RoomConfig) -> pra.ShoeBox:
    """
    Instantiate and configure a Pyroomacoustics ShoeBox room.

    Args:
        room_config (RoomConfig): Acoustic specifications for the room.

    Returns:
        pra.ShoeBox: Initialized ShoeBox simulation instance.
    """

    # Create uniform boundary material
    wall_material: pra.Material = create_room_material(room_config.absorption)

    # Convert dimension tuple to list format required by Pyroomacoustics
    room_dims: list[float] = [
        room_config.dimensions[0],
        room_config.dimensions[1],
        room_config.dimensions[2],
    ]

    # Instantiate ShoeBox room
    room: pra.ShoeBox = pra.ShoeBox(
        room_dims,
        fs=room_config.sample_rate,
        max_order=room_config.max_order,
        materials=wall_material,
    )

    return room
