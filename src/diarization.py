import logging
import random

import numpy as np

logger = logging.getLogger("SpeakerDiarizer")


class BaseDiarizer:
    def identify_speaker(self, audio_segment: np.ndarray) -> str:
        raise NotImplementedError


class PyannoteDiarizer(BaseDiarizer):
    def __init__(self, auth_token=None):
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise ImportError("pyannote.audio not installed.") from e

        from config import get_best_device

        self.device = torch.device(get_best_device())
        try:
            # Note: access token is required for pyannote/speaker-diarization
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
            self.pipeline.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load pyannote pipeline: {e}") from e

    def identify_speaker(self, audio_segment: np.ndarray) -> str:
        # Pyannote expects a file or a tensor in specific format
        # This is complex to integrate in real-time on short chunks without context.
        # For a simple turn-based system, we might just return a speaker embedding hash.
        # This is a placeholder for the real implementation which requires buffering.
        return "Unknown"


class SimpleDiarizer(BaseDiarizer):
    """
    A simple heuristic or mock diarizer.
    For the mock, it randomly assigns Speaker A or B.
    In a real simple implementation, this could use cosine similarity of embeddings
    from a small model (like pyannote/embedding) to cluster chunks.
    """

    def __init__(self):
        self.speakers = ["Speaker A", "Speaker B", "Speaker C"]
        self.last_speaker = None

    def identify_speaker(self, audio_segment: np.ndarray) -> str:
        # For POC: Randomly switch speakers occasionally
        if self.last_speaker is None or random.random() < 0.3:
            self.last_speaker = random.choice(self.speakers)
        return self.last_speaker


def load_diarizer(use_mock=True, auth_token=None):
    if not use_mock and auth_token:
        try:
            return PyannoteDiarizer(auth_token)
        except Exception as e:
            logger.warning(f"Falling back to SimpleDiarizer: {e}")
    return SimpleDiarizer()
