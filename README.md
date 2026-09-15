# Smart Assistant Room Acoustics & Procedural Multi-Speaker Dataset Generator

A high-fidelity spatial audio simulation and dataset generation framework for training and evaluating **Speech-to-Text (ASR)**, **Speaker Diarization**, and **Speaker Isolation / Source Separation** models.

Simulates smart assistant listening devices (e.g. Amazon Echo, Apple HomePod, Google Nest) placed in room corners capturing complex multi-person conversational scenes with neural [Piper-TTS](https://github.com/rhasspy/piper) voices, real-life vocal distortions, multilingual dialogues, and procedural sample variations.

---

## Key Capabilities

### 1. Procedural Conversation Generation & High Sample Variations
- **Batch Generation (`generate_dataset.py`)**: Generate large batches ($N$ samples) of diverse audio variations with a single CLI command.
- **Randomized Room Acoustics**: Room dimensions (4.5m to 9.5m), wall absorption ($\alpha = 0.10$ to $0.38$), and reflection orders (2 to 4).
- **Assistant Microphone Placement**: Automatically placed in random corners and heights (0.75m to 1.1m) with configurable geometries (`mono`, `stereo`, `circular`).
- **Randomized Speaker Configurations**: Variable speaker count (2 to 8), 3D spatial placements (standing/seated, near/far field), physical acoustic apertures (`size`: 0.0m point source to 0.45m head/torso cluster), and vocal personalities.
- **Conversational Dynamics & Overlaps**: Parallel conversation cliques with natural turn-taking and realistic interruptions / overlaps (people cutting each other off). Tunable styles: `mixed`, `party`, `meeting`, `argument`, `assistant`.

### 2. Strictly Distinct Voices, Gender Diversity & Complete Voice Catalog
- **Full Catalog of 176 Piper Voices**: Complete registry of all **176 official Piper neural voice models** and **2,712 distinct speakers** across **57 languages**.
- **Guaranteed Distinct Voices per Scene**: In every generated scene, no two speakers will ever share the same voice model or speaker ID (selection without replacement).
- **Balanced Gender Diversity**: Personas alternate and balance female and male voices, creating contrast in fundamental pitch ($F_0$), formants, and vocal timbre—essential for source separation and diarization models.
- **Automatic & Bulk Downloading**: Missing voice models are downloaded automatically on-the-fly during generation, or in bulk via `python download_voice.py --all --languages en fr es de`.

### 3. Real-Life Vocal Distortions & Dynamics
- **Shouting & Vocal Overdrive**: Dynamic volume boost and soft-clipping non-linear saturation (hyperbolic tangent sigmoid) simulating vocal cord strain and pre-amp saturation.
- **Laughter & Chuckles**: Interspersed dialogue laughter and rhythmic 5 Hz amplitude tremolo modulation simulating laughing while speaking.
- **Room Ambiance**: Subtle ambient background pink/brownian noise (HVAC ventilation / room air) at configurable SNR levels (15 dB to 35 dB).

### 4. Multi-Lingual Capacities
- **Multi-Lingual Dialogue Repository**: Built-in conversational corpora across **English (`en`)**, **French (`fr`)**, **Spanish (`es`)**, and **German (`de`)**.
- **57 Languages Supported**: Any language code present in the Piper catalog can be utilized for generation.

### 5. Ground-Truth Dataset Packaging (ASR, Diarization, Source Separation)
For every generated sample, exports:
- `mixed_scene.wav`: Full composite multi-channel spatial recording at the smart assistant microphone.
- `isolated_speakers/<speaker_id>.wav`: Ground-truth spatial track of each speaker recorded at the microphone (for training source separation models like SepFormer / Conv-TasNet).
- `diarization.rttm`: Standard NIST RTTM format for speaker diarization benchmarks.
- `transcripts.jsonl`: Multi-speaker ASR transcript lines with timestamps, speaker IDs, and gender labels.
- `annotations.json`: Detailed manifest with millisecond timestamps, text, speaker metadata, gender, voice model, and 3D coordinates.

---

## Project Structure

```
audio tests/
├── config/
│   └── default_scene.json          # Standard scene configuration
├── data/
│   ├── piper_voices/               # Neural ONNX voice models (auto-downloaded)
│   │   └── voices.json             # Official registry of 176 voices & 2,712 speakers
│   ├── output/                     # Standalone simulation outputs
│   └── datasets/                   # Generated procedural dataset batches
├── src/
│   ├── common/
│   │   ├── constants.py            # Physics constants & defaults
│   │   ├── types.py                # Type aliases (AudioArray, Point3D, MicArrayMatrix)
│   │   └── audio_utils.py          # Resampling, normalization, WAV I/O
│   ├── config/
│   │   ├── models.py               # Dataclasses (SceneConfig, DatasetSampleConfig, VocalStyle)
│   │   └── loader.py               # JSON scene parser & geometric boundary validation
│   ├── audio/
│   │   └── distortions.py          # Soft-clipping overdrive, laughter tremolo, ambient room noise
│   ├── procedural/
│   │   ├── dialogue_bank.py        # Multilingual conversational bank (EN, FR, ES, DE)
│   │   ├── conversation_generator.py # Turn-taking, interruptions, parallel conversation groups
│   │   └── variation_generator.py  # Enforces unique voices, gender diversity, and room variations
│   ├── dataset/
│   │   └── annotator.py            # Manifest exporter, RTTM generator, JSONL transcripts
│   ├── personas/
│   │   ├── persona.py              # Persona entity and spatial cluster calculation
│   │   └── manager.py              # Multi-persona timeline sequencer with vocal effects
│   ├── tts/
│   │   ├── voice_catalog.py        # Complete catalog of 176 Piper models with gender metadata
│   │   ├── voice_downloader.py     # HuggingFace voice downloader with bulk download support
│   │   └── synthesizer.py          # Piper-TTS engine with automatic on-the-fly downloader
│   ├── simulation/
│   │   ├── room_builder.py         # ShoeBox room geometry & materials
│   │   ├── microphone.py           # Smart assistant microphone array geometries
│   │   └── simulator.py            # Pyroomacoustics simulation & isolated stem generator
│   └── pipeline/
│       ├── orchestrator.py         # Standard simulation pipeline
│       └── dataset_pipeline.py     # Batch dataset generation pipeline
├── tests/                          # 52 unit tests across 9 test suites
├── generate_dataset.py             # CLI for procedural batch dataset generation
├── main.py                         # CLI for single-scene simulation runs
├── download_voice.py               # CLI tool to download Piper voices (single or batch)
├── mypy.ini                        # Strict type-checking configuration
└── .pylintrc                       # Code style configuration (100 char limit)
```

---

## Quickstart & Usage

### 1. Generating a Batch of Procedural Dataset Variations

Generate a batch of diverse dataset samples with isolated stems, RTTM diarization, and ASR transcripts:

```bash
# Generate 10 diverse sample variations (duration 20-40s, 2-4 speakers, English & French)
python generate_dataset.py \
    --num-samples 10 \
    --duration-range 20.0 40.0 \
    --speakers-range 2 4 \
    --languages en fr \
    --style mixed \
    --output-dir data/datasets/my_training_dataset
```

Each sample folder (`sample_0001`, `sample_0002`, ...) contains:
```
data/datasets/my_training_dataset/sample_0001/
├── mixed_scene.wav            # Spatial multi-channel recording at the mic
├── annotations.json           # Time-aligned manifest with speaker genders & positions
├── diarization.rttm           # NIST standard diarization ground truth
├── transcripts.jsonl          # Standard ASR training lines
└── isolated_speakers/         # Ground-truth spatial stems for source separation
    ├── speaker_1.wav
    └── speaker_2.wav
```

### 2. Managing & Batch Downloading Piper Voices

List voices or batch-download all models for target languages:

```bash
# List available voices matching 'fr' or 'en'
python download_voice.py --list --filter fr

# Batch download all voices for English, French, Spanish, and German
python download_voice.py --all --languages en fr es de

# Download a specific voice model
python download_voice.py --voice fr_FR-siwis-medium
```

### 3. Running a Single Configured Scene

Run the default scene from `config/default_scene.json`:

```bash
python main.py
```

Override parameters directly:
```bash
# Circular 4-microphone array in a larger room
python main.py --mic-type circular --room-dim 8.0 6.0 3.0 --output data/output/custom_room.wav
```

### 4. Fast Dry-Run (Mock TTS)

To test geometry, conversation logic, and acoustics rapidly without neural TTS:

```bash
python generate_dataset.py --num-samples 5 --mock-tts --output-dir data/datasets/fast_test
```

---

## Code Style & Verification

The codebase strictly adheres to `code_style.txt`:
- **Static Type Checking**: `mypy --strict` passes with **0 issues** across all 43 source files.
- **Linting & Quality**: `pylint` achieves a perfect **10.00 / 10.00** rating with zero warnings.
- **Unit Testing**: **52 unit tests** across all 9 test suites run and pass in ~0.03s.

Run all verifications:
```bash
python -m mypy --config-file mypy.ini src main.py download_voice.py generate_dataset.py tests
python -m pylint --rcfile=.pylintrc src main.py download_voice.py generate_dataset.py tests
python -m unittest discover -s tests -p "test_*.py" -v
```
