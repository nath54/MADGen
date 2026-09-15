# Smart Assistant Room Acoustics & Piper-TTS Simulation

A spatial audio simulation framework simulating smart assistant devices (e.g. Amazon Echo, Apple HomePod, Google Nest) placed in the corner of a room, listening to multiple distinct personas speaking with neural Piper-TTS voices, individual speech personalities, and physical room presence.

---

## Key Features

- **Multi-Persona Speech Generation**:
  - Distinct voice models powered by [Piper-TTS](https://github.com/rhasspy/piper).
  - Fine-grained vocal personalities: speaking rate (`length_scale`), emotional inflection / variability (`noise_scale`), rhythm cadence (`noise_w_scale`), and vocal effort / loudness (`volume`).
  - Staggered dialogue turn-taking and continuous timeline sequencing.
- **Physical Room Presence & Size**:
  - 3D spatial position in meters `[x, y, z]` (e.g., standing adults, seated persons, children).
  - Extended acoustic aperture modeling (`size` in meters): Models head, mouth, and torso radiation via distributed spatial clusters. Point source modeling when `size == 0.0`.
- **Smart Assistant Device Simulation**:
  - Positioned in room corners (e.g., on tables, countertops, shelves).
  - Configurable microphone array geometries:
    - `mono`: Single omnidirectional microphone.
    - `stereo`: Spaced microphone pair (e.g., 8 cm or 18 cm binaural separation).
    - `circular`: Circular multi-microphone array (e.g., 4 or 6 mics in a 3.5 cm radius ring).
- **Room Acoustic Physics via Pyroomacoustics**:
  - Image Source Model (ISM) and ray-tracing propagation.
  - Configurable room dimensions, wall materials, absorption coefficients, and reflection orders.
  - Multi-channel audio capture with automatic peak normalization and safe headroom protection.
- **Automated Voice Management**:
  - Direct downloader tool (`download_voice.py` or `--download-voice` option in `main.py`) fetching official ONNX models and configurations from HuggingFace.
  - Offline fallback synthesizer (`MockSynthesizer`) for instant dry-run testing without needing model files.
- **Strict Clean Code Quality**:
  - Strictly follows `code_style.txt` standards.
  - 100% type annotations with `mypy --strict` compliance (0 errors).
  - 10.00/10 rating on `pylint` with 0 warnings.
  - 28 unit tests with 100% pass rate.

---

## Project Structure

```
audio tests/
├── config/
│   └── default_scene.json          # Complete scene configuration
├── data/
│   ├── piper_voices/               # Downloaded ONNX voices and JSON configs
│   └── output/                     # Rendered spatial multi-channel WAV files
├── src/
│   ├── common/
│   │   ├── constants.py            # Physics constants and simulation defaults
│   │   ├── types.py                # Type aliases (AudioArray, Point3D, MicArrayMatrix)
│   │   └── audio_utils.py          # Resampling, peak normalization, WAV I/O
│   ├── config/
│   │   ├── models.py               # Dataclasses (RoomConfig, PersonaConfig, SceneConfig)
│   │   └── loader.py               # JSON scene parser & geometric boundary validation
│   ├── personas/
│   │   ├── persona.py              # Persona entity and spatial cluster calculation
│   │   └── manager.py              # Multi-persona timeline sequencer and track builder
│   ├── tts/
│   │   ├── synthesizer.py          # Piper-TTS synthesis engine & mock fallback
│   │   └── voice_downloader.py     # HuggingFace voice downloader
│   ├── simulation/
│   │   ├── room_builder.py         # ShoeBox room geometry & materials
│   │   ├── microphone.py           # Smart assistant microphone array geometry
│   │   └── simulator.py            # Pyroomacoustics simulation orchestrator
│   └── pipeline/
│       └── orchestrator.py         # End-to-end simulation coordinator
├── tests/
│   ├── test_audio_utils.py         # Tests for audio math, resampling, and WAV I/O
│   ├── test_config.py              # Tests for scene loading and coordinate bounds
│   ├── test_personas.py            # Tests for persona dispersion and timeline
│   └── test_simulation.py          # Tests for room building and microphone arrays
├── main.py                         # CLI entrypoint with argparse & config overrides
├── download_voice.py               # CLI tool to download Piper voices
├── mypy.ini                        # Strict type-checking configuration
├── .pylintrc                       # Code style and linting configuration
└── code_style.txt                  # Strict style guidelines reference
```

---

## Setup & Prerequisites

Ensure the virtual environment is activated:

```bash
source .venv/bin/activate
```

Dependencies installed:
- `python >= 3.12`
- `numpy`
- `scipy`
- `soundfile`
- `pyroomacoustics`
- `piper-tts`

---

## Downloading Piper Voices

Voices are stored in `data/piper_voices/`. You can browse and download any official voice from HuggingFace using `download_voice.py`:

```bash
# List available voices matching a search pattern (e.g. English or French)
python download_voice.py --list --filter en_US

# Download a specific voice model (downloads .onnx and .onnx.json)
python download_voice.py --voice en_US-lessac-low --output-dir data/piper_voices
```

Available voices include:
- `en_US-lessac-low` / `en_US-lessac-medium` (English, female)
- `en_US-bryce-medium` (English, male)
- `en_GB-alan-medium` (British English, male)
- `fr_FR-siwis-medium` (French, female)
- And hundreds more from the Piper model catalog.

---

## Running Simulations

### 1. Standard Simulation with Piper-TTS

Run the simulation using the default scene (`config/default_scene.json`):

```bash
python main.py
```

Output is saved to `data/output/smart_assistant_simulation.wav`.

### 2. Fast Dry-Run (Mock TTS)

To test room acoustics and geometry without downloading neural voice models:

```bash
python main.py --mock-tts
```

### 3. Command-Line Overrides

Override room dimensions, microphone corner positions, array types, or output paths directly from the CLI:

```bash
# Place assistant in a different corner with a 4-microphone circular ring
python main.py --mic-type circular --mic-pos 0.3 0.3 0.9 --output data/output/corner_circular.wav

# Simulate a larger room (8m x 6m x 3m)
python main.py --room-dim 8.0 6.0 3.0 --output data/output/large_living_room.wav
```

---

## Scene Configuration (`config/default_scene.json`)

Scenes are defined in clean JSON format:

```json
{
  "room": {
    "dimensions": [6.0, 5.0, 2.8],
    "absorption": 0.2,
    "max_order": 3,
    "sample_rate": 16000
  },
  "assistant": {
    "position": [0.4, 0.4, 0.85],
    "mic_type": "stereo",
    "mic_spacing": 0.08,
    "mic_radius": 0.035,
    "num_microphones": 2
  },
  "personas": [
    {
      "id": "alice",
      "name": "Alice",
      "voice_model": "en_US-lessac-low.onnx",
      "speaker_id": null,
      "position": [2.5, 3.5, 1.65],
      "size": 0.25,
      "personality": {
        "length_scale": 0.95,
        "noise_scale": 0.667,
        "noise_w_scale": 0.8,
        "volume": 1.0
      },
      "utterances": [
        {
          "text": "Echo, what is the weather like outside today?",
          "start_time_s": 0.5
        },
        {
          "text": "Thank you, that sounds wonderful.",
          "start_time_s": 4.0
        }
      ]
    },
    {
      "id": "bob",
      "name": "Bob",
      "voice_model": "en_US-lessac-low.onnx",
      "speaker_id": null,
      "position": [4.5, 2.0, 1.15],
      "size": 0.3,
      "personality": {
        "length_scale": 1.15,
        "noise_scale": 0.5,
        "noise_w_scale": 0.9,
        "volume": 0.85
      },
      "utterances": [
        {
          "text": "Can you also remind me to water the plants at six?",
          "start_time_s": 6.5
        }
      ]
    }
  ]
}
```

---

## Verification & Quality Assurance

Run the test suite and static analysis tools:

```bash
# 1. Type verification with strict mypy
python -m mypy --config-file mypy.ini src main.py download_voice.py tests

# 2. Code style and quality check with pylint
python -m pylint --rcfile=.pylintrc src main.py download_voice.py tests

# 3. Unit test suite
python -m unittest discover -s tests -p "test_*.py" -v
```

Current results:
- **mypy**: `Success: no issues found in 27 source files` (strict mode)
- **pylint**: `10.00 / 10.00`
- **unittest**: `28 tests passed (0 failures, 0 errors)`
