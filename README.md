# Smart Assistant Room Acoustics & Procedural Multi-Speaker Dataset Generator

A high-fidelity spatial audio simulation and dataset generation framework for training and evaluating **Speech-to-Text (ASR)**, **Speaker Diarization**, and **Speaker Isolation / Source Separation** models.

Simulates smart assistant listening devices (e.g. Amazon Echo, Apple HomePod, Google Nest) placed in room corners capturing complex multi-person conversational scenes with neural [Piper-TTS](https://github.com/rhasspy/piper) voices, dynamic **LLM-generated dialogues** (`llama.cpp`), **100,000-word dictionary constraints**, **12 rich ambiance presets**, real-life vocal distortions, multilingual dialogues, and procedural sample variations.

---

## Key Capabilities

### 1. Dynamic LLM Dialogue Generation (`llama.cpp`)
- **Interactive Tool-Calling Engine**: Prompts local `llama.cpp` servers (`llama-server`) via OpenAI-compatible endpoints (`/v1/chat/completions`) using structured JSON function calling (`submit_dialogue`).
- **Context-Aware Personas**: Prompts incorporate each speaker's assigned name, gender, voice, and conversational clique, producing natural multi-turn exchanges with spontaneous interruptions and laughs.
- **Robust Offline Fallback**: Automatically falls back to the built-in multilingual conversational bank if the LLM server is offline, unreachable, or disabled.

### 2. 100,000-Word Dictionary & Thematic Constraints
- **100k Clean English Dictionary (`data/dictionaries/words_100k.txt`)**: Curated English vocabulary list of 100,000 words (lengths 4–12 characters).
- **Randomized Keyword Anchors**: Procedurally samples unique constraint keywords passed into the LLM prompt to anchor conversations to unexpected, highly diverse themes and subjects.

### 3. 12 Rich Conversational Ambiance Presets
Fine-tuned presets that configure LLM roleplay guidelines, turn-taking overlap rates, shouting/laughing rates, and background acoustic SNR:
- `casual_chit_chat`: Relaxed living room banter about weekend plans and daily life.
- `kitchen_cooking`: People preparing dinner, discussing recipes, ingredients, and kitchen timers.
- `workplace_meeting`: Professional sprint review, deadlines, deliverables, and slide decks.
- `heated_argument`: Intense disagreement over a broken promise with rapid overlapping interruptions.
- `party_celebration`: High-energy social celebration, loud toasts, overlapping banter, and laughter.
- `smart_assistant_household`: Busy household with smart home queries ("Hey Alexa", "Turn off lights").
- `gaming_session`: Multiplayer gaming with tactical callouts, excitement, and panic.
- `interview_podcast`: Host and guest discussing career highlights and audience questions.
- `late_night_philosophy`: Thoughtful late-night discussion on ethics, artificial intelligence, and reality.
- `family_dinner`: Multigenerational dinner table with kids talking over adults and passing food.
- `academic_defense`: Thesis defense panel discussing methodology, citations, and validity.
- `emergency_rush`: Urgent coordination during a crisis with quick shouting instructions.

### 4. Long Conversational Samples & Synthesis-Driven Timeline
- **Long Procedural Sequences (100+ Sentences)**: Generates comprehensive multi-speaker exchanges with at least 100 sentences per sample across parallel conversation cliques (2 to 3 speakers per group).
- **Automatic Duration from Speech Synthesis**: Scene duration and turn timestamps are calculated dynamically from the actual waveforms synthesized by Piper-TTS (or mock TTS). Subsequent speaker turns and interruptions automatically synchronize with preceding speech durations.
- **Chunked LLM Generation**: Robustly generates lengthy conversations in continuation chunks, avoiding token limits on local LLM engines.

### 5. Procedural Room Variations & Batch Dataset Generation
- **Batch Generation (`generate_dataset.py`)**: Generate large batches ($N$ samples) of diverse audio variations with a single CLI command.
- **Dynamic Room Dimensions & Spatial Separation**: Rooms scale dynamically (5.0m to 12.0m) to accommodate 4 to 10 concurrent speakers while guaranteeing $\ge 0.70\text{m}$ inter-speaker spatial clearance and safe distance from walls and microphones.
- **Assistant Microphone Placement**: Placed in random room corners and heights (0.75m to 1.1m) with configurable geometries (`mono`, `stereo`, `circular`).
- **Randomized Speaker Configurations**: 4 to 10 concurrent speakers, 3D spatial placements (standing/seated, near/far field), physical acoustic apertures (`size`: 0.0m point source to 0.45m head/torso cluster), and vocal personalities.

### 6. Strictly Distinct Voices, Gender Diversity & Complete Voice Catalog
- **Full Catalog of 176 Piper Voices**: Complete registry of all **176 official Piper neural voice models** and **2,712 distinct speakers** across **57 languages**.
- **Guaranteed Distinct Voices per Scene**: In every generated scene with 4 to 10 speakers, no two speakers ever share the same voice model or speaker ID (selection without replacement).
- **Balanced Gender Diversity**: Personas alternate and balance female and male voices, creating contrast in fundamental pitch ($F_0$), formants, and vocal timbre—essential for source separation and diarization models.
- **Automatic & Bulk Downloading**: Missing voice models are downloaded automatically on-the-fly during generation, or in bulk via `python download_voice.py --all --languages en fr es de`.

### 6. Real-Life Vocal Distortions & Dynamics
- **Shouting & Vocal Overdrive**: Dynamic volume boost and soft-clipping non-linear saturation (hyperbolic tangent sigmoid) simulating vocal cord strain and pre-amp saturation.
- **Laughter & Chuckles**: Interspersed dialogue laughter and rhythmic 5 Hz amplitude tremolo modulation simulating laughing while speaking.
- **Room Ambiance**: Ambient background noise (HVAC ventilation / room air) tailored to each ambiance preset's target SNR level.

### 7. Ground-Truth Dataset Packaging (ASR, Diarization, Source Separation)
For every generated sample, exports:
- `mixed_scene.wav`: Full composite multi-channel spatial recording at the smart assistant microphone.
- `isolated_speakers/<speaker_id>.wav`: Ground-truth spatial track of each speaker recorded at the microphone (for training source separation models like SepFormer / Conv-TasNet).
- `diarization.rttm`: Standard NIST RTTM format for speaker diarization benchmarks.
- `transcripts.jsonl`: Multi-speaker ASR transcript lines with timestamps, speaker IDs, and gender labels.
- `annotations.json`: Detailed manifest with millisecond timestamps, text, speaker metadata, gender, voice model, 3D coordinates, ambiance preset, and thematic constraint keywords.

---

## Project Structure

```
audio tests/
├── config/
│   └── default_scene.json          # Standard scene configuration
├── data/
│   ├── dictionaries/
│   │   └── words_100k.txt          # 100,000 clean English constraint words
│   ├── piper_voices/               # Neural ONNX voice models (auto-downloaded)
│   │   └── voices.json             # Official registry of 176 voices & 2,712 speakers
│   ├── output/                     # Standalone simulation outputs
│   └── datasets/                   # Generated procedural dataset batches
├── src/
│   ├── audio/
│   │   └── distortions.py          # Soft-clipping overdrive, laughter tremolo, ambient room noise
│   ├── common/
│   │   ├── constants.py            # Physics constants & defaults
│   │   ├── types.py                # Type aliases (AudioArray, Point3D, MicArrayMatrix)
│   │   └── audio_utils.py          # Resampling, normalization, WAV I/O
│   ├── config/
│   │   ├── models.py               # Dataclasses (SceneConfig, DatasetSampleConfig, BatchGenerationConfig)
│   │   └── loader.py               # JSON scene parser & geometric boundary validation
│   ├── dataset/
│   │   └── annotator.py            # Manifest exporter, RTTM generator, JSONL transcripts
│   ├── llm/
│   │   ├── client.py               # llama.cpp HTTP client with health probing & tool calling
│   │   ├── dialogue_generator.py   # LLM dialogue synthesis with tool-calling schema & fallback
│   │   └── llm_types.py            # Strongly-typed TypedDicts for chat completions & tools
│   ├── personas/
│   │   ├── persona.py              # Persona entity and spatial cluster calculation
│   │   └── manager.py              # Multi-persona timeline sequencer with vocal effects
│   ├── pipeline/
│   │   ├── dataset_pipeline.py     # Batch dataset generation pipeline
│   │   └── orchestrator.py         # Standard simulation pipeline
│   ├── procedural/
│   │   ├── ambiance_presets.py     # 12 rich conversational ambiance presets
│   │   ├── conversation_generator.py # Turn-taking, interruptions, parallel cliques, LLM sequencing
│   │   ├── dialogue_bank.py        # Multilingual conversational bank (EN, FR, ES, DE)
│   │   ├── variation_generator.py  # Enforces unique voices, gender diversity, and room variations
│   │   └── word_dictionary.py      # 100k-word dictionary loader & keyword constraint sampler
│   ├── simulation/
│   │   ├── microphone.py           # Smart assistant microphone array geometries
│   │   ├── room_builder.py         # ShoeBox room geometry & materials
│   │   └── simulator.py            # Pyroomacoustics simulation & isolated stem generator
│   └── tts/
│       ├── synthesizer.py          # Piper-TTS engine with automatic on-the-fly downloader
│       ├── voice_catalog.py        # Complete catalog of 176 Piper models with gender metadata
│       └── voice_downloader.py     # HuggingFace voice downloader with bulk download support
├── tests/                          # 67 unit tests across 14 test suites
├── generate_dataset.py             # CLI for procedural batch dataset generation
├── main.py                         # CLI for single-scene simulation runs
├── download_voice.py               # CLI tool to download Piper voices (single or batch)
├── mypy.ini                        # Strict type-checking configuration
└── .pylintrc                       # Code style configuration (100 char limit)
```

---

## Quickstart & Usage

### 1. Generating Procedural Datasets with LLM Dialogues & Ambiance

Generate dataset samples powered by a local `llama.cpp` server, 100k-word constraints, and kitchen ambiance:

```bash
python generate_dataset.py \
    --num-samples 5 \
    --min-sentences 100 \
    --speakers-range 4 8 \
    --use-llm \
    --llm-url http://127.0.0.1:8080/v1 \
    --ambiance kitchen_cooking \
    --constraint-words 3 \
    --output-dir data/datasets/kitchen_dataset
```

Sample variations can also select randomly among all 12 ambiance presets:
```bash
python generate_dataset.py \
    --num-samples 20 \
    --ambiance random \
    --use-llm \
    --output-dir data/datasets/diverse_ambiances
```

Each sample folder (`sample_0001`, `sample_0002`, ...) contains:
```
data/datasets/kitchen_dataset/sample_0001/
├── mixed_scene.wav            # Spatial multi-channel recording at the mic
├── annotations.json           # Time-aligned manifest with speaker genders, positions, and constraints
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

To test geometry, LLM dialogues, and acoustics rapidly without neural speech synthesis:

```bash
python generate_dataset.py --num-samples 3 --use-llm --mock-tts --output-dir data/datasets/fast_test
```

---

## Code Style & Verification

The codebase strictly adheres to `code_style.txt`:
- **Static Type Checking**: `mypy --strict` passes with **0 issues** across all 54 source files.
- **Linting & Quality**: `pylint` achieves a perfect **10.00 / 10.00** rating across all source and test files.
- **Unit Testing**: **69 unit tests** across all 14 test suites run and pass.

Run all verifications:
```bash
./.venv/bin/mypy --strict src main.py download_voice.py generate_dataset.py tests
./.venv/bin/pylint --rcfile=.pylintrc src main.py download_voice.py generate_dataset.py tests
./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```
