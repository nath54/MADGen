"""
Microphone array geometry builder simulating smart assistant listening devices.

Constructs 3D spatial coordinate matrices for mono, stereo, and circular microphone
arrays placed in designated room locations such as corners.
"""

# Import Modules
import numpy as np

from src.config.models import MicrophoneConfig
from src.common.types import (
    MicrophoneType,
    MicArrayMatrix,
)


def build_mono_array(center_pos: tuple[float, float, float]) -> MicArrayMatrix:
    """
    Construct a single omnidirectional microphone coordinate matrix.

    Args:
        center_pos (tuple[float, float, float]): Position (x, y, z) in meters.

    Returns:
        MicArrayMatrix: 3x1 coordinate matrix.
    """

    # Create 3x1 column vector for single microphone
    return np.array([[center_pos[0]], [center_pos[1]], [center_pos[2]]], dtype=np.float64)


def build_stereo_array(
    center_pos: tuple[float, float, float],
    spacing: float,
) -> MicArrayMatrix:
    """
    Construct a two-channel stereo microphone array spaced symmetrically.

    Args:
        center_pos (tuple[float, float, float]): Center point (x, y, z) in meters.
        spacing (float): Total separation distance between microphones in meters.

    Returns:
        MicArrayMatrix: 3x2 coordinate matrix.
    """

    # Calculate half spacing along the X axis
    half_dist: float = spacing / 2.0
    pos_x: float = center_pos[0]
    pos_y: float = center_pos[1]
    pos_z: float = center_pos[2]

    # Construct coordinates for left (-X) and right (+X) microphones
    mic_left: list[float] = [pos_x - half_dist, pos_y, pos_z]
    mic_right: list[float] = [pos_x + half_dist, pos_y, pos_z]

    # Combine into 3x2 matrix
    mic_matrix: MicArrayMatrix = np.array(
        [
            [mic_left[0], mic_right[0]],
            [mic_left[1], mic_right[1]],
            [mic_left[2], mic_right[2]],
        ],
        dtype=np.float64,
    )

    return mic_matrix


def build_circular_array(
    center_pos: tuple[float, float, float],
    radius: float,
    num_microphones: int,
) -> MicArrayMatrix:
    """
    Construct a horizontal circular microphone ring typical of smart speakers.

    Args:
        center_pos (tuple[float, float, float]): Center point (x, y, z) in meters.
        radius (float): Radius of the circular ring in meters.
        num_microphones (int): Number of microphone elements in the ring.

    Returns:
        MicArrayMatrix: 3xN coordinate matrix.
    """

    # Generate angular positions evenly spaced around circle
    angles: np.ndarray = np.linspace(0.0, 2.0 * np.pi, num_microphones, endpoint=False)

    # Compute X, Y offsets and constant Z height
    pos_x: float = center_pos[0]
    pos_y: float = center_pos[1]
    pos_z: float = center_pos[2]

    coords_x: list[float] = [pos_x + radius * float(np.cos(theta)) for theta in angles]
    coords_y: list[float] = [pos_y + radius * float(np.sin(theta)) for theta in angles]
    coords_z: list[float] = [pos_z for _ in angles]

    # Assemble 3xN matrix
    mic_matrix: MicArrayMatrix = np.array(
        [coords_x, coords_y, coords_z],
        dtype=np.float64,
    )

    return mic_matrix


def build_microphone_array(config: MicrophoneConfig) -> MicArrayMatrix:
    """
    Build the complete microphone coordinate array matrix from configuration.

    Args:
        config (MicrophoneConfig): Microphone configuration parameters.

    Returns:
        MicArrayMatrix: 3xN microphone coordinates array.

    Raises:
        ValueError: If an unknown microphone array type is provided.
    """

    # Dispatch to appropriate geometry builder
    if config.mic_type == MicrophoneType.MONO:
        return build_mono_array(config.position)

    if config.mic_type == MicrophoneType.STEREO:
        return build_stereo_array(config.position, config.mic_spacing)

    if config.mic_type == MicrophoneType.CIRCULAR:
        return build_circular_array(
            config.position,
            config.mic_radius,
            config.num_microphones,
        )

    raise ValueError(f"Unsupported microphone type: {config.mic_type}")
