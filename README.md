# Smart Assistant Room Acoustics & Procedural Multi-Speaker Dataset Generator

A high-fidelity spatial audio simulation and procedural dataset generation framework for training and evaluating **Speech-to-Text (ASR)**, **Speaker Diarization**, and **Speech Separation / Source Isolation** models (e.g. SepFormer, Conv-TasNet, Whisper, PyAnnote).

Simulates smart assistant listening devices (e.g. Amazon Echo, Apple HomePod, Google Nest) placed in room corners capturing complex multi-person conversational scenes with neural [Piper-TTS](https://github.com/rhasspy/piper) voices, dynamic **LLM-generated dialogues** (`llama.cpp`), **100,000-word dictionary constraints**, **12 rich ambiance presets**, real-life vocal distortions, multilingual dialogues (English, French, Spanish, German), and procedural sample variations.

Includes a comprehensive **acoustic and spectrogram analysis engine** (`analyze_audio.py`) that quantifies **voice recoverability vs destruction** (STFT, Welch PSD, active speech SI-SDR, and Ideal Ratio Masking oracle separation).

---

## Table of Contents
1. [Key Capabilities](#key-capabilities)
2. [Installation & Setup](#installation--setup)
3. [CLI Reference & Argument Guide](#cli-reference--argument-guide)
   - [`generate_dataset.py` (Dataset Generation)](#1-generate_datasetpy---procedural-batch-dataset-generation)
   - [`analyze_audio.py` (Acoustic & Spectrogram Analysis)](#2-analyze_audiopy---acoustic--spectrogram-analysis)
   - [`download_voice.py` (Voice Model Management)](#3-download_voicepy---piper-voice-management)
   - [`main.py` (Single-Scene Simulation)](#4-mainpy---single-scene-simulation)
4. [Dataset Output Structure](#dataset-output-structure)
5. [12 Conversational Ambiance Presets](#12-conversational-ambiance-presets)
6. [Multilingual Support & Voice Catalog](#multilingual-support--voice-catalog)
7. [Acoustic Recovery & Destruction Analysis](#acoustic-recovery--destruction-analysis)
8. [Common Recipes & Cookbook](#common-recipes--cookbook)
9. [Project Architecture](#project-architecture)
10. [Verification & Code Quality](#verification--code-quality)

---

## Key Capabilities

### 1. Dynamic LLM Dialogue Generation (`llama.cpp`)
- **Fast Reasoning-Bypass Inference**: Prompts local `llama-server` instances via OpenAI-compatible endpoints (`/v1/chat/completions`). Uses an assistant prefill technique (`<think>\n</think>\n`) to bypass lengthy chain-of-thought tokens on reasoning models (e.g. `Qwen3.6-35B`), dropping generation latency from ~7 minutes to **~3 seconds per 15-turn chunk**.
- **Context-Aware Personas**: Prompts incorporate each speaker's assigned name, gender, language, voice model, and conversational clique, producing natural multi-turn exchanges with spontaneous interruptions, laughter, and overlapping speech.
- **Robust Offline Fallback**: Automatically falls back to the built-in multilingual conversational bank (`src/procedural/dialogue_bank.py`) if the LLM server is offline, unreachable, or disabled (`--no-use-llm`).

### 2. 100,000-Word Dictionary & Thematic Anchors
- **Curated 100k Dictionary (`data/dictionaries/words_100k.txt`)**: Curated clean vocabulary list (lengths 4–12 characters).
- **Thematic Anchors**: Procedurally samples unique constraint keywords passed into the LLM prompt to anchor conversations to unexpected, highly diverse themes, preventing repetitive synthetic conversations.

### 3. Procedural Room Variations & Non-Overlapping Consecutive Batches
- **Sequential Output (`sample_001`, `sample_002`, ...)**: Automatically detects existing sample folders in `data/output/` and auto-increments indices without overwriting previous runs.
- **Dynamic 3D Geometry**: Rooms scale dynamically (5.0m to 12.0m) to accommodate 2 to 10 concurrent speakers with guaranteed inter-speaker spatial clearance ($\ge 0.70\text{m}$) and safe distance from walls and microphones.
- **Virtual Smart Assistant Microphones**: Placed in room corners and heights (0.75m to 1.1m) with configurable geometries (`mono`, `stereo`, `circular`).

### 4. Distinct Voices & Balanced Gender Diversity
- **Piper Neural Voice Models**: High-performance, fast neural text-to-speech using ONNX Runtime.
- **Hierarchical Storage**: Models organized cleanly into `data/piper_voices/BASE_LANG/LANG_DETAIL/persona_name/`.
- **Gender Balance**: Alternates male and female voices across scenes, creating rich contrast in fundamental pitch ($F_0$), formants, and vocal timbre—critical for source separation and diarization models.

### 5. Acoustic Spectrogram & Voice Recoverability Analysis (`analyze_audio.py`)
- **Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)**: Evaluates in-mix speech degradation over active speech frames.
- **Ideal Ratio Masking (IRM) Oracle Recovery**: Quantifies how much vocal energy is **recoverable vs destroyed** by acoustic overlap, reverberation, and background noise.
- **Automated Dashboards & Publication Plots**: Emits terminal summary tables, structured JSON reports (`analysis_report.json`), and high-resolution spectrogram comparison plots (`spectrogram_analysis.png`).

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.10+** (Python 3.12 recommended)
- **libsndfile** and standard audio libraries (typically pre-installed on Linux/macOS)

### 2. Clone & Virtual Environment Setup

```bash
# Clone the repository
git clone https://github.com/nath54/audio-tests.git
cd "audio tests"

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. (Optional) Starting the Local LLM Server

To generate dynamic, unpredictable conversations instead of template sentences, start a local OpenAI-compatible server such as `llama-server` (from [llama.cpp](https://github.com/ggerganov/llama.cpp)):

```bash
# Example launching Qwen3.6-35B on port 8080
llama-server -m models/Qwen3.6-35B-A3B-Q4_K_M.gguf --port 8080 -c 8192 --jinja
```

> **Note**: If `llama-server` is not running, the system will gracefully detect the offline status and fall back to the built-in offline dialogue bank without crashing.

---

## CLI Reference & Argument Guide

### 1. `generate_dataset.py` - Procedural Batch Dataset Generation

Primary tool to generate batches of multi-speaker audio scenes with ground-truth labels.

```bash
./.venv/bin/python generate_dataset.py [OPTIONS]
```

#### Complete Arguments Reference:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--num-samples` | `int` | `10` | Total number of unique dataset sample variations to generate. |
| `--min-sentences` | `int` | `100` | Minimum number of dialogue turns across the whole scene. |
| `--duration-range` | `float float` | `None` | Target min and max audio duration in seconds (e.g. `10.0 30.0`). If omitted, duration is auto-calculated from speech synthesis. |
| `--speakers-range` | `int int` | `4 10` | Minimum and maximum number of concurrent speakers per scene. |
| `--languages` | `str [str ...]` | `en` | Languages allowed in generated scenes. Choices: `en`, `fr`, `es`, `de`. |
| `--style` | `str` | `mixed` | Conversational style scenario setting turn-taking dynamics and emotions. Choices: `mixed`, `party`, `meeting`, `argument`, `assistant`. |
| `--use-llm` / `--no-use-llm` | `flag` | `True` | Whether to generate dynamic dialogue lines via local LLM server. (Use `--no-use-llm` to force offline dialogue bank). |
| `--llm-url` | `str` | `http://127.0.0.1:8080/v1` | Endpoint URL for the `llama-server` / OpenAI-compatible API. |
| `--ambiance` | `str` | `random` | Conversational ambiance preset (sets roleplay guidelines and acoustics). Choices: `random` or any of the 12 preset names. |
| `--constraint-words` | `int` | `3` | Number of keywords sampled from the 100k dictionary to anchor dialogue themes. |
| `--llm-temperature` | `float` | `0.7` | Sampling temperature for LLM dialogue generation. |
| `--parallel-prob` | `float` | `0.5` | Probability of parallel side conversations occurring concurrently with the main discussion. |
| `--disable-effects` | `flag` | `False` | Disables vocal distortion/saturation, room noise, and heavy reverb for dry, pristine speech audio. |
| `--output-dir` | `str` | `data/output` | Target base directory where sample folders (`sample_001`, ...) are saved. |
| `--voices-dir` | `str` | `data/piper_voices` | Directory where Piper ONNX models are stored and auto-downloaded. |
| `--no-isolated` | `flag` | `False` | Disable exporting per-speaker isolated spatial stems (faster generation). |
| `--mock-tts` | `flag` | `False` | Use synthetic beep tones instead of neural Piper voices for rapid testing. |
| `--log-level` | `str` | `INFO` | Logging verbosity level. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

---

### 2. `analyze_audio.py` - Acoustic & Spectrogram Analysis

Evaluates generated samples, computes SI-SDR degradation, oracle Ideal Ratio Masking recoverability, and exports publication-grade plots.

```bash
./.venv/bin/python analyze_audio.py [OPTIONS]
```

#### Complete Arguments Reference:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--sample-id` | `str` | `None` | Identifier of sample within `--output-dir` (e.g. `sample_001`). |
| `--sample-dir` | `Path` | `None` | Direct path to a specific dataset sample folder (e.g. `data/output/sample_001`). |
| `--output-dir` | `Path` | `data/output` | Base directory containing dataset samples when resolving `--sample-id` or `--all`. |
| `--all` | `flag` | `False` | Batch analyze all samples found in `--output-dir`. |
| `--channel` | `int` | `0` | Zero-indexed microphone channel to analyze (e.g. `0` for primary channel). |
| `--output-plot` | `Path` | `None` | Custom path to save the spectrogram PNG (defaults to `<sample_dir>/spectrogram_analysis.png`). |
| `--output-report` | `Path` | `None` | Custom path to save the JSON analysis report (defaults to `<sample_dir>/analysis_report.json`). |
| `--dpi` | `int` | `150` | Resolution in dots-per-inch for exported visualization plots. |
| `--no-plot` | `flag` | `False` | Skip generating the graphical plot and only output the terminal dashboard and JSON report. |

---

### 3. `download_voice.py` - Piper Voice Management

Manages downloading official Piper neural voices from HuggingFace into the local hierarchical directory.

```bash
./.venv/bin/python download_voice.py [OPTIONS]
```

#### Complete Arguments Reference:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--voice` | `str` | `en_US-lessac-low` | Voice model key to download (e.g. `fr_FR-siwis-medium`, `en_US-bryce-medium`). |
| `--all` | `flag` | `False` | Batch download all available voices matching the specified `--languages`. |
| `--languages` | `str [str ...]` | `en fr es de` | Language codes to download when using `--all`. |
| `--list` | `flag` | `False` | List all available voice models in the official catalog. |
| `--filter` | `str` | `None` | Filter listed voices by name or language prefix (e.g. `fr`, `female`). |
| `--output-dir` | `str` | `data/piper_voices` | Target root folder for downloaded models. |

---

### 4. `main.py` - Single-Scene Simulation

Renders a single acoustic simulation from a static JSON configuration file.

```bash
./.venv/bin/python main.py [OPTIONS]
```

#### Complete Arguments Reference:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `config/default_scene.json` | Path to JSON scene configuration file. |
| `--output` | `str` | `data/output/smart_assistant_simulation.wav` | Destination path for the rendered WAV file. |
| `--voices-dir` | `str` | `data/piper_voices` | Directory where Piper ONNX voices are stored. |
| `--room-dim` | `float float float` | `None` | Override room dimensions in meters: `X Y Z` (e.g. `7.0 5.0 2.8`). |
| `--mic-pos` | `float float float` | `None` | Override assistant microphone position: `X Y Z` (e.g. `0.4 0.4 0.85`). |
| `--mic-type` | `str` | `None` | Override microphone array geometry (`mono`, `stereo`, `circular`). |
| `--sample-rate` | `int` | `None` | Override simulation sampling rate in Hz (e.g. `16000`). |
| `--mock-tts` | `flag` | `False` | Use synthetic beep tones instead of neural Piper voices. |
| `--download-voice`| `str` | `None` | Download a specified Piper voice before running simulation. |
| `--log-level` | `str` | `INFO` | Logging verbosity level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Dataset Output Structure

Every generated dataset batch produces numbered sample folders inside `data/output/` alongside a master manifest:

```
data/output/
├── dataset_manifest.json          # Master dataset manifest with summary of all samples
├── sample_001/
│   ├── mixed_scene.wav            # Multi-channel composite spatial audio recording
│   ├── annotations.json           # Detailed manifest (speaker 3D coordinates, genders, models, words)
│   ├── diarization.rttm           # Standard NIST RTTM diarization ground truth
│   ├── transcripts.jsonl          # Time-aligned ASR transcripts with speaker IDs
│   ├── spectrogram_analysis.png   # (When analyzed) 5-panel acoustic evaluation plot
│   ├── analysis_report.json       # (When analyzed) JSON metrics for SI-SDR and recoverability
│   └── isolated_speakers/         # Ground-truth isolated spatial stems for source separation
│       ├── speaker_1.wav
│       ├── speaker_2.wav
│       └── ...
└── sample_002/
    └── ...
```

---

## 12 Conversational Ambiance Presets

Select an ambiance via `--ambiance <name>` or use `--ambiance random`:

| Preset Name | Scene Setting | Conversational Dynamics |
| :--- | :--- | :--- |
| `casual_chit_chat` | Living room conversation | Relaxed banter about daily life, low overlap, occasional laughter. |
| `kitchen_cooking` | Dinner preparation | Cooking instructions, recipes, timers, moderate interruptions. |
| `workplace_meeting` | Professional conference room | Sprint review, action items, deliverables, professional register. |
| `heated_argument` | Disagreement | High overlap, frequent shouting, rapid aggressive interruptions. |
| `party_celebration` | Lively party | Loud social atmosphere, frequent laughter, toasts, high energy. |
| `smart_assistant_household` | Smart home | Family members issuing smart assistant voice commands and queries. |
| `gaming_session` | Competitive multiplayer | Tactical callouts, excitement, panic, shouts, and quick confirmations. |
| `interview_podcast` | Studio podcast | Host and guest structured Q&A, professional pacing, minimal overlap. |
| `late_night_philosophy` | Late-night philosophical chat | Contemplative, relaxed pace, deep abstract thoughts and musings. |
| `family_dinner` | Multigenerational dinner | Overlapping conversation, passing dishes, interruptions, storytelling. |
| `academic_defense` | Thesis defense panel | Formal academic inquiry, technical explanations, rigorous questions. |
| `emergency_rush` | Crisis coordination | Urgent shouted commands, high-stress instructions, rapid pacing. |

---

## Multilingual Support & Voice Catalog

Voices are organized hierarchically: `data/piper_voices/<lang_family>/<language_code>/<persona_name>/`.

### French (`--languages fr`)
All 7 official French Piper voice models are supported:
- `fr_FR-siwis-medium` (Female, 1 speaker)
- `fr_FR-siwis-low` (Female, 1 speaker)
- `fr_FR-tom-medium` (Male, 1 speaker)
- `fr_FR-gilles-low` (Male, 1 speaker)
- `fr_FR-mls_1840-low` (Male, 1 speaker)
- `fr_FR-upmc-medium` (Mixed, 2 speakers: Jessica & Pierre)
- `fr_FR-mls-medium` (Mixed, 125 distinct French speakers)

### English (`--languages en`)
- `en_US-lessac-low`, `en_US-lessac-medium`, `en_US-amy-medium`, `en_US-bryce-medium`, `en_US-john-medium`, `en_US-norman-medium`, `en_GB-cori-high`, `en_GB-alba-medium`, etc.

### Spanish (`--languages es`) & German (`--languages de`)
- Spanish: `es_ES-davefx-medium`, `es_ES-sharvard-medium`
- German: `de_DE-karlsson-low`, `de_DE-thorsten-medium`

---

## Acoustic Recovery & Destruction Analysis

The analysis tool [`analyze_audio.py`](analyze_audio.py) computes:
1. **In-Mix SI-SDR ($\text{dB}$)**: Signal-to-Distortion Ratio of each speaker inside the raw mixed audio.
2. **Oracle Ideal Ratio Masking (IRM)**: Time-frequency oracle separation:
   $$M_i(t, f) = \frac{|S_i(t, f)|}{\sum_j |S_j(t, f)| + \epsilon}$$
3. **Recoverable Energy Fraction (%)**: Proportion of vocal energy preserved and separable.
4. **Destroyed Energy Fraction (%)**: Energy irrecoverably lost to acoustic overlap and destructive interference.
5. **Separation Gain ($\Delta\text{SI-SDR}$)**: Upper-bound separation improvement achievable via time-frequency masking.

---

## Common Recipes & Cookbook

### 1. Generating a French Multi-Speaker Dataset
```bash
./.venv/bin/python generate_dataset.py \
    --languages fr \
    --num-samples 5 \
    --speakers-range 3 6 \
    --min-sentences 35 \
    --disable-effects
```

### 2. Generating a High-Overlap Heated Argument in English
```bash
./.venv/bin/python generate_dataset.py \
    --languages en \
    --num-samples 3 \
    --ambiance heated_argument \
    --style argument \
    --speakers-range 3 5 \
    --min-sentences 40
```

### 3. Rapid Pipeline Verification (Mock TTS)
```bash
./.venv/bin/python generate_dataset.py \
    --num-samples 2 \
    --mock-tts \
    --min-sentences 10 \
    --no-use-llm
```

### 4. Evaluating a Sample with Acoustic Analysis
```bash
# Analyze sample_001
./.venv/bin/python analyze_audio.py --sample-id sample_001

# Batch analyze all generated samples in data/output
./.venv/bin/python analyze_audio.py --all
```

---

## Project Architecture

```
audio tests/
├── requirements.txt                # Core runtime & dev dependencies
├── mypy.ini                        # Strict static type checker configuration
├── .pylintrc                       # Code style rules (100 char limit)
├── generate_dataset.py             # CLI: Procedural batch dataset generation
├── analyze_audio.py                # CLI: Acoustic spectrogram & voice recovery analysis
├── download_voice.py               # CLI: Piper voice downloader & manager
├── main.py                         # CLI: Single-scene spatial acoustic simulation
├── config/
│   └── default_scene.json          # Example single-scene configuration
├── data/
│   ├── dictionaries/
│   │   └── words_100k.txt          # 100,000 clean English constraint words
│   ├── piper_voices/               # Hierarchical neural voice repository
│   └── output/                     # Generated dataset samples (sample_001, ...)
├── src/
│   ├── audio/
│   │   ├── analysis.py             # STFT, Welch PSD, SI-SDR, and IRM recoverability
│   │   └── distortions.py          # Vocal overdrive, laughter tremolo, ambient noise
│   ├── common/
│   │   ├── constants.py            # Physics constants & acoustic parameters
│   │   ├── types.py                # Strongly-typed aliases (AudioArray, Point3D)
│   │   └── audio_utils.py          # Resampling, normalization, and WAV I/O
│   ├── config/
│   │   ├── models.py               # Dataclasses (SceneConfig, BatchGenerationConfig, etc.)
│   │   └── loader.py               # JSON parser & boundary validator
│   ├── dataset/
│   │   └── annotator.py            # Manifest exporter, RTTM diarization, JSONL transcripts
│   ├── llm/
│   │   ├── client.py               # HTTP client with health probing & tool calling
│   │   ├── dialogue_generator.py   # LLM screenplay synthesis with reasoning bypass
│   │   └── llm_types.py            # Strongly-typed completions & turn definitions
│   ├── personas/
│   │   ├── persona.py              # Persona entity and spatial aperture cluster
│   │   └── manager.py              # Multi-persona timeline sequencer & effects
│   ├── pipeline/
│   │   ├── dataset_pipeline.py     # Batch dataset generation pipeline
│   │   └── orchestrator.py         # Single-scene simulation pipeline
│   ├── procedural/
│   │   ├── ambiance_presets.py     # 12 rich conversational ambiance presets
│   │   ├── conversation_generator.py # Parallel discussion cliques & turn sequencing
│   │   ├── dialogue_bank.py        # Offline multilingual fallback dialogues (EN, FR, ES, DE)
│   │   ├── variation_generator.py  # Procedural room geometries & voice distribution
│   │   └── word_dictionary.py      # 100k-word dictionary loader & keyword sampler
│   ├── simulation/
│   │   ├── microphone.py           # Virtual assistant microphone arrays (mono, stereo, circular)
│   │   ├── room_builder.py         # ShoeBox room acoustics & wall absorption
│   │   └── simulator.py            # Pyroomacoustics simulation & stem generator
│   └── tts/
│       ├── synthesizer.py          # Piper-TTS engine with automatic on-the-fly downloader
│       ├── voice_catalog.py        # Complete registry of 176 Piper models with metadata
│       └── voice_downloader.py     # HuggingFace voice downloader with bulk download support
└── tests/                          # 92 unit tests across 16 test suites
```

---

## Verification & Code Quality

The codebase enforces strict type safety and quality standards:
- **Static Type Checking**: `mypy --strict` passes with **0 errors** across all source files.
- **Linter & Formatting**: `pylint` achieves a perfect **10.00 / 10.00** rating across all source files.
- **Unit Testing**: **92 automated unit tests** across 16 test suites covering acoustics, mathematics, parsing, and pipeline integration.

```bash
# Run static type checking
./.venv/bin/mypy src generate_dataset.py analyze_audio.py download_voice.py main.py tests

# Run linter
./.venv/bin/pylint src generate_dataset.py analyze_audio.py download_voice.py main.py tests

# Run all 92 unit tests
./.venv/bin/python -m unittest discover -s tests
```
