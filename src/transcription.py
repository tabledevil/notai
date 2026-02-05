import time
import numpy as np
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoxtralTranscriber")

class BaseTranscriber:
    def transcribe(self, audio_data: np.ndarray) -> str:
        raise NotImplementedError

class VoxtralTranscriber(BaseTranscriber):
    def __init__(self, model_id="mistralai/voxtral-transcribe-2.4b", device="cpu"):
        logger.info(f"Loading model: {model_id} on {device}")
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            import torch
        except ImportError:
            raise ImportError("transformers or torch not installed.")

        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            # Attempt to load the model
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)

            # Create pipeline for easier inference
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"Failed to load Voxtral model: {e}")
            logger.warning("Falling back to OpenAI Whisper (small) due to Voxtral load failure.")
            # Fallback to a known model if Voxtral fails (e.g. invalid repo ID)
            fallback_id = "openai/whisper-tiny"
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=fallback_id,
                chunk_length_s=30,
                device=device,
            )

    def transcribe(self, audio_data: np.ndarray) -> str:
        # Pipeline expects dictionary or file path, but can also take numpy array if formatted right
        # For simplicity, we assume audio_data is float32 mono

        # Note: audio_data should be 16kHz
        result = self.pipe(audio_data, generate_kwargs={"language": "english"})
        return result["text"].strip()

class MockTranscriber(BaseTranscriber):
    """Simulates transcription for testing without heavy models."""
    def __init__(self):
        self.sentences = [
            "Hello, is this working?",
            "I think the connection is stable now.",
            "We should proceed with the deployment.",
            "The Voxtral model is quite impressive.",
            "Can you hear me clearly?",
            "Let's move to the next topic.",
            "This is a simulated transcription.",
            "Speaker diarization is challenging."
        ]

    def transcribe(self, audio_data: np.ndarray) -> str:
        # Simulate processing delay
        time.sleep(0.5)
        # Return a random sentence
        return random.choice(self.sentences)

def load_transcriber(use_mock=False):
    if use_mock:
        return MockTranscriber()
    try:
        return VoxtralTranscriber()
    except Exception as e:
        logger.error(f"Could not load real transcriber: {e}")
        return MockTranscriber()
