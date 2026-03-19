# CLAUDE.md

## Project Overview

**Voxtral Terminal Transcriber (notai)** — A Python terminal application for live meeting transcription using Mistral's Voxtral model with speaker diarization. Built with PyTorch, Hugging Face Transformers, and Rich for terminal UI.

## Directory Structure

```
notai/
├── src/
│   ├── main.py            # Entry point: VoxtralApp class, CLI arg parsing, Rich UI
│   ├── audio.py           # Audio streaming: MicrophoneStream, FileAudioStream, MockAudioStream, VAD
│   ├── transcription.py   # Transcription: VoxtralTranscriber (Whisper fallback), MockTranscriber
│   ├── diarization.py     # Speaker ID: PyannoteDiarizer, SimpleDiarizer fallback
│   ├── config.py          # Configuration loading from notai.json with defaults
│   └── export.py          # Transcript export: text, JSON, SRT formats
├── tests/
│   └── test_basic.py      # Unit tests for all components
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions: lint (ruff) + test (pytest)
├── pyproject.toml         # Project metadata, dependencies, ruff & pytest config
├── requirements.txt       # Python dependencies (pip install)
└── README.md
```

## Tech Stack

- **Python 3.10+**
- **PyTorch / torchaudio** — model inference and audio processing
- **Hugging Face Transformers** — Voxtral and Whisper ASR models
- **pyannote.audio** — speaker diarization
- **PyAudio** — microphone capture (requires system `portaudio` library)
- **Rich** — terminal UI with live-updating display
- **numpy / scipy / librosa** — audio utilities

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install with dev tools (ruff, pytest)
pip install -e ".[dev]"

# Run live transcription
python src/main.py

# Run in mock mode (no microphone/GPU needed)
python src/main.py --mock

# Transcribe an audio file
python src/main.py --input-file recording.wav

# Run with real speaker diarization
python src/main.py --auth-token <HF_TOKEN>

# Auto-export transcript on exit
python src/main.py --mock --export text
python src/main.py --mock --export json
python src/main.py --mock --export srt

# Use a custom config file
python src/main.py --config path/to/notai.json

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## Configuration

The app loads settings from `notai.json` (current directory) or `~/.notai.json`. All fields are optional — defaults are used for missing keys.

```json
{
  "sample_rate": 16000,
  "chunk_size": 512,
  "silence_duration_threshold": 1.0,
  "min_turn_duration": 0.5,
  "vad_threshold": 0.01,
  "model_id": "mistralai/voxtral-transcribe-2.4b",
  "language": "english",
  "max_transcript_display": 15,
  "refresh_rate": 10,
  "export_format": "text",
  "export_path": "./transcript"
}
```

## Architecture

The app uses a **modular, component-based** design with abstract base classes and fallback implementations:

- **Audio** → `AudioStream` (ABC) with `MicrophoneStream`, `FileAudioStream`, and `MockAudioStream`
- **Transcription** → `BaseTranscriber` with `VoxtralTranscriber` (falls back to Whisper) and `MockTranscriber`
- **Diarization** → `BaseDiarizer` with `PyannoteDiarizer` and `SimpleDiarizer`
- **Config** → `load_config()` reads `notai.json` with defaults
- **Export** → `export_transcript()` writes text/JSON/SRT files

**Key patterns:**
- Background worker thread (`_processing_worker`) processes audio via queues
- Voice Activity Detection (VAD) uses energy-based thresholds
- Turn detection via silence duration (1.0s threshold) and minimum duration (0.5s)
- Rich Live display refreshes at 10fps with a scrolling transcript table
- Graceful fallbacks: PyAudio → Mock, Voxtral → Whisper, PyannoteDiarizer → SimpleDiarizer

## Code Conventions

- **Classes**: PascalCase (`VoxtralApp`, `MicrophoneStream`)
- **Methods/functions**: snake_case (`get_chunk`, `is_speech`)
- **Constants**: UPPER_SNAKE_CASE (`SAMPLE_RATE`, `CHUNK_SIZE`)
- **Private methods**: leading underscore (`_processing_worker`, `_read_loop`)
- **Imports**: stdlib first, then third-party, then local modules
- **Logging**: Python `logging` module
- **Linter**: ruff (config in pyproject.toml)
- **Line length**: 120

## Testing

- Framework: pytest (with unittest-style test classes)
- Tests use mock/fallback components (no GPU or microphone required)
- Run with: `pytest tests/ -v`
- Test config: `pythonpath = ["src"]` in pyproject.toml
- Coverage: VAD, transcription, diarization, audio streams, config loading, export (text/JSON/SRT)

## CI/CD

GitHub Actions runs on push/PR to master:
- **lint** job: `ruff check` and `ruff format --check`
- **test** job: `pytest tests/ -v`

## System Dependencies

- `portaudio19-dev` (Ubuntu/Debian) or `portaudio` (macOS via Homebrew) — required for PyAudio
- CUDA-capable GPU recommended for real transcription (CPU fallback available)
