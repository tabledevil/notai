import json
import logging
from pathlib import Path

logger = logging.getLogger("Config")

DEFAULT_CONFIG = {
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
    "export_path": "./transcript",
}

CONFIG_FILENAME = "notai.json"


def find_config_file():
    """Search for config file in current dir, then home dir."""
    candidates = [
        Path.cwd() / CONFIG_FILENAME,
        Path.home() / f".{CONFIG_FILENAME}",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def get_best_device():
    """Detect the best available compute device: CUDA > MPS (Apple Silicon) > CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_dtype(device):
    """Return the optimal dtype for the given device."""
    try:
        import torch
    except ImportError:
        return None

    if device == "cuda":
        return torch.float16
    # MPS supports float16 but some ops are more stable with float32
    if device == "mps":
        return torch.float32
    return torch.float32


def load_config(config_path=None):
    """Load configuration from file, falling back to defaults."""
    config = dict(DEFAULT_CONFIG)

    path = Path(config_path) if config_path else find_config_file()
    if path and path.is_file():
        try:
            with open(path) as f:
                user_config = json.load(f)
            config.update(user_config)
            logger.info(f"Loaded config from {path}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load config from {path}: {e}")

    return config
