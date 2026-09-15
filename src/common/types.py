"""
Common type definitions and enumerations for acoustic simulations.

Provides type aliases for multi-dimensional audio arrays, spatial 3D coordinates,
and microphone geometry enumerations.
"""

# Import Modules
import enum
import typing

import numpy as np
from numpy.typing import NDArray

# Audio signal array type alias representing 64-bit floating point audio
AudioArray: typing.TypeAlias = NDArray[np.float64]

# Spatial 3D coordinate represented as a tuple of x, y, z in meters
Point3D: typing.TypeAlias = tuple[float, float, float]

# Spatial 3D vector array represented as a NumPy float64 array
Vector3D: typing.TypeAlias = NDArray[np.float64]

# Microphone array geometry matrix of shape (3, N_mics)
MicArrayMatrix: typing.TypeAlias = NDArray[np.float64]


class MicrophoneType(str, enum.Enum):
    """
    Supported microphone array geometry types for the simulated device.
    """

    MONO = "mono"
    STEREO = "stereo"
    CIRCULAR = "circular"
