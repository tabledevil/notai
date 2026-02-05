import time
import queue
import threading
import numpy as np
from abc import ABC, abstractmethod

# Try importing pyaudio, handle failure gracefully for headless/mock environments
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

class AudioStream(ABC):
    """Abstract base class for audio streams."""

    def __init__(self, rate=16000, chunk_size=1024):
        self.rate = rate
        self.chunk_size = chunk_size
        self.closed = False
        self.queue = queue.Queue()

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def get_chunk(self):
        """Get the next chunk of audio data. Returns None if stream is closed."""
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return None

class MicrophoneStream(AudioStream):
    """Stream audio from system microphone using PyAudio."""

    def __init__(self, rate=16000, chunk_size=1024, device_index=None):
        super().__init__(rate, chunk_size)
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio is not installed. Use MockAudioStream or install pyaudio.")

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.device_index = device_index
        self._stop_event = threading.Event()
        self._thread = None

    def _callback(self, in_data, frame_count, time_info, status):
        # Convert bytes to float32 numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def _read_loop(self):
        """Blocking read loop if callback isn't used."""
        try:
            while not self._stop_event.is_set():
                if self.stream.is_active():
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    self.queue.put(audio_data)
        except Exception as e:
            print(f"Error in microphone stream: {e}")

    def start(self):
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

class MockAudioStream(AudioStream):
    """Generates synthetic audio (speech-like bursts) for testing."""

    def __init__(self, rate=16000, chunk_size=1024):
        super().__init__(rate, chunk_size)
        self._stop_event = threading.Event()
        self._thread = None

    def _generate_audio(self):
        t = 0
        while not self._stop_event.is_set():
            # Simulate "speech" (bursts of noise/sine) and "silence"
            # Cycle: 3s talk, 2s silence
            cycle_pos = (t * self.chunk_size / self.rate) % 5.0

            if cycle_pos < 3.0:
                # Active speech: Random noise modulated
                noise = np.random.normal(0, 0.1, self.chunk_size).astype(np.float32)
                # Add some sine components to make it look like a signal
                sine = 0.5 * np.sin(2 * np.pi * 440 * np.arange(self.chunk_size) / self.rate)
                data = (noise + sine) * 0.5
            else:
                # Silence
                data = np.zeros(self.chunk_size, dtype=np.float32)

            self.queue.put(data)
            t += 1
            time.sleep(self.chunk_size / self.rate)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._generate_audio)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

class VAD:
    """Simple energy-based Voice Activity Detection."""

    def __init__(self, threshold=0.01, sample_rate=16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.buffer = []

    def is_speech(self, chunk):
        """Returns True if chunk energy is above threshold."""
        energy = np.mean(np.abs(chunk))
        return energy > self.threshold
