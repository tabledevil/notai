import os
import sys
import time
import unittest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from audio import VAD, MockAudioStream
from diarization import SimpleDiarizer
from transcription import MockTranscriber


class TestVAD(unittest.TestCase):
    def test_silent_chunk_detected(self):
        vad = VAD(threshold=0.5)
        silent_chunk = [0.0] * 512
        self.assertFalse(vad.is_speech(silent_chunk))

    def test_loud_chunk_detected(self):
        vad = VAD(threshold=0.5)
        loud_chunk = [1.0] * 512
        self.assertTrue(vad.is_speech(loud_chunk))

    def test_below_threshold(self):
        vad = VAD(threshold=0.1)
        below = [0.05] * 512
        self.assertFalse(vad.is_speech(below))

    def test_custom_threshold(self):
        vad = VAD(threshold=0.001)
        quiet = [0.005] * 512
        self.assertTrue(vad.is_speech(quiet))


class TestMockTranscriber(unittest.TestCase):
    def test_returns_string(self):
        t = MockTranscriber()
        res = t.transcribe([0.0])
        self.assertIsInstance(res, str)

    def test_returns_nonempty(self):
        t = MockTranscriber()
        res = t.transcribe([0.0])
        self.assertTrue(len(res) > 0)

    def test_returns_from_sentence_list(self):
        t = MockTranscriber()
        res = t.transcribe([0.0])
        self.assertIn(res, t.sentences)


class TestSimpleDiarizer(unittest.TestCase):
    def test_returns_speaker_label(self):
        d = SimpleDiarizer()
        res = d.identify_speaker([0.0])
        self.assertIn("Speaker", res)

    def test_speaker_in_known_set(self):
        d = SimpleDiarizer()
        res = d.identify_speaker([0.0])
        self.assertIn(res, d.speakers)

    def test_consistency(self):
        """Speaker should stay the same sometimes (70% chance)."""
        d = SimpleDiarizer()
        d.identify_speaker([0.0])
        first = d.last_speaker
        self.assertIsNotNone(first)


class TestMockAudioStream(unittest.TestCase):
    def test_start_and_get_chunk(self):
        stream = MockAudioStream()
        stream.start()
        time.sleep(0.1)
        chunk = stream.get_chunk()
        self.assertIsNotNone(chunk)
        stream.stop()

    def test_chunk_length(self):
        stream = MockAudioStream(chunk_size=256)
        stream.start()
        time.sleep(0.1)
        chunk = stream.get_chunk()
        self.assertIsNotNone(chunk)
        self.assertEqual(len(chunk), 256)
        stream.stop()

    def test_stop_is_clean(self):
        stream = MockAudioStream()
        stream.start()
        time.sleep(0.05)
        stream.stop()
        # Should not raise or hang


class TestConfig(unittest.TestCase):
    def test_load_defaults(self):
        from config import DEFAULT_CONFIG, load_config

        config = load_config("/nonexistent/path.json")
        self.assertEqual(config["sample_rate"], DEFAULT_CONFIG["sample_rate"])
        self.assertEqual(config["chunk_size"], DEFAULT_CONFIG["chunk_size"])

    def test_default_values_present(self):
        from config import DEFAULT_CONFIG

        required_keys = [
            "sample_rate",
            "chunk_size",
            "silence_duration_threshold",
            "min_turn_duration",
            "vad_threshold",
            "export_format",
        ]
        for key in required_keys:
            self.assertIn(key, DEFAULT_CONFIG)


class TestExport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = "/tmp/notai_test_export"
        os.makedirs(self.tmpdir, exist_ok=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_items(self):
        # Inline class to avoid importing from main
        class Item:
            def __init__(self, timestamp, speaker, text):
                self.timestamp = timestamp
                self.speaker = speaker
                self.text = text

        return [
            Item("00:00:01", "Speaker A", "Hello world"),
            Item("00:00:05", "Speaker B", "Hi there"),
        ]

    def test_export_text(self):
        from export import export_transcript

        items = self._make_items()
        path = export_transcript(items, fmt="text", base_path=f"{self.tmpdir}/out")
        self.assertTrue(path.endswith(".txt"))
        with open(path) as f:
            content = f.read()
        self.assertIn("Hello world", content)
        self.assertIn("Speaker A", content)

    def test_export_json(self):
        import json

        from export import export_transcript

        items = self._make_items()
        path = export_transcript(items, fmt="json", base_path=f"{self.tmpdir}/out")
        self.assertTrue(path.endswith(".json"))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["text"], "Hello world")

    def test_export_srt(self):
        from export import export_transcript

        items = self._make_items()
        path = export_transcript(items, fmt="srt", base_path=f"{self.tmpdir}/out")
        self.assertTrue(path.endswith(".srt"))
        with open(path) as f:
            content = f.read()
        self.assertIn("Hello world", content)

    def test_export_empty(self):
        from export import export_transcript

        path = export_transcript([], fmt="text", base_path=f"{self.tmpdir}/out")
        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
