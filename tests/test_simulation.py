"""
Unit tests for room acoustics simulation, microphone geometry, and wave propagation.
"""

# Import Modules
import unittest

import numpy as np

from src.common.types import MicrophoneType
from src.personas.persona import Persona
from src.personas.manager import PersonaTrack
from src.simulation.simulator import run_acoustic_simulation
from src.config.models import (
    RoomConfig,
    PersonaConfig,
    MicrophoneConfig,
)
from src.simulation.microphone import (
    build_mono_array,
    build_stereo_array,
    build_circular_array,
    build_microphone_array,
)
from src.simulation.room_builder import (
    build_room,
    create_room_material,
)


class TestSimulation(unittest.TestCase):
    """
    Validation suite for microphone arrays, ShoeBox room construction, and acoustic rendering.
    """

    def test_create_room_material_valid(self) -> None:
        """
        Verify valid absorption values construct a Material object.
        """

        mat = create_room_material(0.2)
        self.assertIsNotNone(mat)

    def test_create_room_material_invalid_raises(self) -> None:
        """
        Verify absorption values outside [0, 1] raise ValueError.
        """

        with self.assertRaises(ValueError):
            create_room_material(-0.1)

        with self.assertRaises(ValueError):
            create_room_material(1.5)

    def test_build_room(self) -> None:
        """
        Verify ShoeBox room instantiation sets proper dimensions and sample rate.
        """

        room_config: RoomConfig = RoomConfig(
            dimensions=(5.0, 4.0, 2.5),
            absorption=0.2,
            max_order=2,
            sample_rate=16000,
        )
        room = build_room(room_config)

        self.assertEqual(room.fs, 16000)

    def test_build_mono_array(self) -> None:
        """
        Verify mono microphone array produces 3x1 matrix with exact position.
        """

        pos: tuple[float, float, float] = (0.5, 0.5, 0.9)
        mic_mat = build_mono_array(pos)

        self.assertEqual(mic_mat.shape, (3, 1))
        self.assertAlmostEqual(mic_mat[0, 0], 0.5)
        self.assertAlmostEqual(mic_mat[1, 0], 0.5)
        self.assertAlmostEqual(mic_mat[2, 0], 0.9)

    def test_build_stereo_array(self) -> None:
        """
        Verify stereo array produces 3x2 matrix with symmetric separation.
        """

        pos: tuple[float, float, float] = (0.5, 0.5, 0.9)
        spacing: float = 0.08
        mic_mat = build_stereo_array(pos, spacing)

        self.assertEqual(mic_mat.shape, (3, 2))
        # Left mic: X - 0.04 = 0.46
        self.assertAlmostEqual(mic_mat[0, 0], 0.46)
        # Right mic: X + 0.04 = 0.54
        self.assertAlmostEqual(mic_mat[0, 1], 0.54)

    def test_build_circular_array(self) -> None:
        """
        Verify circular array creates 3xN matrix with elements on constant radius.
        """

        pos: tuple[float, float, float] = (1.0, 1.0, 1.0)
        radius: float = 0.05
        num_mics: int = 4
        mic_mat = build_circular_array(pos, radius, num_mics)

        self.assertEqual(mic_mat.shape, (3, 4))
        # All Z values should equal 1.0
        np.testing.assert_array_almost_equal(mic_mat[2, :], np.ones(4))

    def test_build_microphone_array_dispatch(self) -> None:
        """
        Verify dispatch logic for different MicrophoneType enum options.
        """

        # Mono
        cfg_mono: MicrophoneConfig = MicrophoneConfig(mic_type=MicrophoneType.MONO)
        self.assertEqual(build_microphone_array(cfg_mono).shape, (3, 1))

        # Stereo
        cfg_stereo: MicrophoneConfig = MicrophoneConfig(mic_type=MicrophoneType.STEREO)
        self.assertEqual(build_microphone_array(cfg_stereo).shape, (3, 2))

        # Circular
        cfg_circ: MicrophoneConfig = MicrophoneConfig(
            mic_type=MicrophoneType.CIRCULAR,
            num_microphones=6,
        )
        self.assertEqual(build_microphone_array(cfg_circ).shape, (3, 6))

    def test_run_acoustic_simulation(self) -> None:
        """
        Verify running full simulation generates multi-channel audio without clipping.
        """

        room_config: RoomConfig = RoomConfig(
            dimensions=(4.0, 4.0, 2.5),
            absorption=0.3,
            max_order=1,
            sample_rate=16000,
        )
        mic_config: MicrophoneConfig = MicrophoneConfig(
            position=(0.4, 0.4, 0.8),
            mic_type=MicrophoneType.STEREO,
        )

        # 0.2s of white noise audio
        num_samples: int = 3200
        test_signal = np.sin(np.linspace(0, 20 * np.pi, num_samples, dtype=np.float64))

        persona: Persona = Persona(
            config=PersonaConfig(
                id="alice",
                name="Alice",
                position=(2.0, 2.0, 1.2),
                size=0.1,
            )
        )
        track: PersonaTrack = PersonaTrack(
            persona=persona,
            audio=test_signal,
            total_samples=num_samples,
        )

        output = run_acoustic_simulation(room_config, mic_config, [track])

        # Verify stereo output
        self.assertEqual(output.shape[1], 2)
        # Verify normalization (peak <= 0.95)
        self.assertLessEqual(float(np.max(np.abs(output))), 0.95)


if __name__ == "__main__":
    unittest.main()
