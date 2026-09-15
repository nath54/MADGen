import numpy as np
import pyroomacoustics as pra
import soundfile as sf

fs = 44100
# Simulation d'une pièce 10m x 10m x 3m avec légère réverbération
room_dim = [10.0, 10.0, 3.0]
# Un coefficient uniforme sur tous les murs
room = pra.ShoeBox(
    room_dim,
    fs=fs,
    max_order=3,
    materials=pra.Material(0.2)
)

# 1. Position et orientation du récepteur (tête / oreilles)
listener_pos = np.array([5.0, 5.0, 1.5])
ear_distance = 0.18  # 18 cm

# Oreille gauche (-X) et oreille droite (+X)
mic_left = listener_pos + np.array([-ear_distance / 2, 0.0, 0.0])
mic_right = listener_pos + np.array([ear_distance / 2, 0.0, 0.0])

mic_array = np.c_[mic_left, mic_right]  # Matrice 3 x 2
room.add_microphone_array(mic_array)

# 2. Ajout des sources (signal NumPy 1D + coordonnée 3D)
sources = [
    {"pos": [3.0, 7.0, 1.5], "signal": np.random.randn(fs * 2)},
    {"pos": [7.0, 4.0, 2.0], "signal": np.random.randn(fs * 2)},
]

for src in sources:
    room.add_source(src["pos"], signal=src["signal"])

# 3. Calcul de la propagation acoustique et rendu stéréo
room.simulate()

# room.mic_array.signals est un ndarray de forme (2, N_samples)
stereo_out = room.mic_array.signals.T

# Normalisation pour éviter le clipping
stereo_out /= np.max(np.abs(stereo_out)) + 1e-9

# Sauvegarde
sf.write("scene_spatiale.wav", stereo_out, fs)