"""
Physical constants and default acoustic configuration parameters.

Provides acoustic constants such as the speed of sound, default sampling rates,
and standard geometry offsets for simulated smart assistant devices.
"""

# Physical acoustics constants
SPEED_OF_SOUND_MPS: float = 343.0

# Audio sampling rate defaults
DEFAULT_SAMPLE_RATE: int = 16000
HIGH_DEF_SAMPLE_RATE: int = 44100

# Room acoustics simulation defaults
DEFAULT_MAX_ORDER: int = 3
DEFAULT_ABSORPTION: float = 0.2

# Smart assistant placement defaults
DEFAULT_CORNER_OFFSET: float = 0.4
DEFAULT_DEVICE_HEIGHT: float = 0.85

# Microphone array geometry defaults
DEFAULT_MIC_SPACING: float = 0.08
DEFAULT_MIC_RADIUS: float = 0.035

# Numerical stability threshold
EPSILON: float = 1e-9
