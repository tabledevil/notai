# Voxtral Terminal Transcriber

A Python-based terminal application for live meeting transcription using Mistral's Voxtral model (or compatible transformers) and speaker diarization.

## Features

- **Live Transcription**: Real-time speech-to-text.
- **Speaker Diarization**: Distinguishes between speakers (requires Hugging Face token for `pyannote.audio`).
- **Terminal UI**: Beautiful, scrolling interface using `rich`.
- **Mock Mode**: Fully functional mock mode for testing without a microphone or GPU.

## Prerequisites

- Python 3.10+
- `portaudio` (Required for PyAudio)
  - Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
  - macOS: `brew install portaudio`

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Live Mode (Requires Microphone)
```bash
python src/main.py
```

### Live Mode with Real Diarization
To use the state-of-the-art `pyannote/speaker-diarization` model, you need a Hugging Face token with access to the model.

```bash
python src/main.py --auth-token YOUR_HF_TOKEN
```

### Mock Mode (Testing)
Simulates audio input and transcription.
```bash
python src/main.py --mock
```

## Architecture

- **Audio Engine**: `pyaudio` for capture, `webrtcvad` (or energy-based fallback) for voice detection.
- **Transcription**: Hugging Face `transformers` pipeline (Defaults to Voxtral, falls back to Whisper).
- **Diarization**: `pyannote.audio` pipeline.
- **UI**: `rich` library with `Live` display.
