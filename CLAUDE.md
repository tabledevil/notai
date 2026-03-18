# CLAUDE.md

## Project Overview

**Voxtral Terminal Transcriber** — A Python terminal application for live meeting transcription using Mistral's Voxtral model with speaker diarization. Built with PyTorch, Hugging Face Transformers, and Rich for terminal UI.

## Directory Structure

```
notai/
├── src/
│   ├── main.py            # Entry point: VoxtralApp class, CLI arg parsing, Rich UI
│   ├── audio.py           # Audio streaming: MicrophoneStream, MockAudioStream, VAD
│   ├── transcription.py   # Transcription: VoxtralTranscriber (Whisper fallback), MockTranscriber
│   └── diarization.py     # Speaker ID: PyannoteDiarizer, SimpleDiarizer fallback
├── tests/
│   └── test_basic.py      # Unit tests for all components
├── requirements.txt       # Python dependencies
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

# Run live transcription
python src/main.py

# Run in mock mode (no microphone/GPU needed)
python src/main.py --mock

# Run with real speaker diarization
python src/main.py --auth-token <HF_TOKEN>

# Run tests
python -m unittest tests/test_basic.py
```

## Architecture

The app uses a **modular, component-based** design with abstract base classes and fallback implementations:

- **Audio** → `AudioStream` (ABC) with `MicrophoneStream` and `MockAudioStream`
- **Transcription** → `BaseTranscriber` with `VoxtralTranscriber` (falls back to Whisper) and `MockTranscriber`
- **Diarization** → `BaseDiarizer` with `PyannoteDiarizer` and `SimpleDiarizer`

**Key patterns:**
- Background worker thread (`_processing_worker`) processes audio via queues
- Voice Activity Detection (VAD) uses energy-based thresholds
- Turn detection via silence duration (1.0s threshold) and minimum duration (0.5s)
- Rich Live display refreshes at 10fps with a scrolling transcript table

## Code Conventions

- **Classes**: PascalCase (`VoxtralApp`, `MicrophoneStream`)
- **Methods/functions**: snake_case (`get_chunk`, `is_speech`)
- **Constants**: UPPER_SNAKE_CASE (`SAMPLE_RATE`, `CHUNK_SIZE`)
- **Private methods**: leading underscore (`_processing_worker`, `_read_loop`)
- **Imports**: stdlib first, then third-party, then local modules
- **Logging**: Python `logging` module in transcription.py and diarization.py

## Testing

- Framework: Python `unittest` (standard library)
- Tests use mock/fallback components (no GPU or microphone required)
- Run with: `python -m unittest tests/test_basic.py`
- Tests import from `src/` via `sys.path.append`

## System Dependencies

- `portaudio19-dev` (Ubuntu/Debian) or `portaudio` (macOS via Homebrew) — required for PyAudio
- CUDA-capable GPU recommended for real transcription (CPU fallback available)
