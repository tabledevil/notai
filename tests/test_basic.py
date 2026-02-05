import sys
import os
import time
import threading
import unittest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from audio import MockAudioStream, VAD
from transcription import MockTranscriber
from diarization import SimpleDiarizer

class TestComponents(unittest.TestCase):
    def test_vad(self):
        vad = VAD(threshold=0.5)
        silent_chunk = [0.0] * 512
        loud_chunk = [1.0] * 512
        self.assertFalse(vad.is_speech(silent_chunk))
        self.assertTrue(vad.is_speech(loud_chunk))

    def test_mock_transcriber(self):
        t = MockTranscriber()
        res = t.transcribe([0.0])
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 0)

    def test_mock_diarizer(self):
        d = SimpleDiarizer()
        res = d.identify_speaker([0.0])
        self.assertIn("Speaker", res)

    def test_audio_stream(self):
        stream = MockAudioStream()
        stream.start()
        time.sleep(0.1)
        chunk = stream.get_chunk()
        self.assertIsNotNone(chunk)
        stream.stop()

if __name__ == '__main__':
    unittest.main()
