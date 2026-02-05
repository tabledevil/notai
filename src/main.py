import argparse
import time
import threading
import queue
import numpy as np
from datetime import datetime

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.text import Text
from rich.table import Table

from audio import MicrophoneStream, MockAudioStream, VAD
from transcription import load_transcriber
from diarization import load_diarizer

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32ms
SILENCE_DURATION_THRESHOLD = 1.0  # Seconds of silence to trigger end-of-turn
MIN_TURN_DURATION = 0.5 # Minimum speech duration to process

def create_layout() -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    return layout

class TranscriptItem:
    def __init__(self, timestamp, speaker, text):
        self.timestamp = timestamp
        self.speaker = speaker
        self.text = text

class VoxtralApp:
    def __init__(self, args):
        self.args = args
        self.console = Console()
        self.layout = create_layout()
        self.transcript_history = []
        self.status_message = "Initializing..."
        self.is_running = True

        # Audio buffer for current turn
        self.current_turn_audio = []
        self.silence_counter = 0
        self.is_speech_active = False

        # Queues for threading
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Components
        self.vad = VAD(threshold=0.01)

        # Initialize Audio
        if args.mock:
            self.audio_stream = MockAudioStream(rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
        else:
            try:
                self.audio_stream = MicrophoneStream(rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
            except Exception as e:
                self.console.print(f"[red]Error initializing microphone:[/red] {e}")
                self.console.print("[yellow]Falling back to Mock Audio Stream.[/yellow]")
                self.audio_stream = MockAudioStream(rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)

        # Initialize Models (lazy load in thread to avoid blocking startup too long,
        # but for simplicity we load here with status updates)
        self.status_message = "Loading Models..."
        self.transcriber = load_transcriber(use_mock=args.mock)

        use_mock_diarizer = args.mock or (not args.auth_token)
        self.diarizer = load_diarizer(use_mock=use_mock_diarizer, auth_token=args.auth_token)

        self.status_message = "Ready. Listening..."

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

    def _processing_worker(self):
        """Background thread to process audio turns."""
        while self.is_running:
            try:
                audio_data = self.processing_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if len(audio_data) == 0:
                continue

            # Concatenate chunks
            full_audio = np.concatenate(audio_data)

            # Transcribe
            self.status_message = "Transcribing..." # Note: This is racy, but acceptable for TUI status
            try:
                text = self.transcriber.transcribe(full_audio)

                # Diarize
                speaker = self.diarizer.identify_speaker(full_audio)

                # Send result back
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.result_queue.put(TranscriptItem(timestamp, speaker, text))
            except Exception as e:
                self.console.print(f"[red]Error in processing:[/red] {e}")
            finally:
                self.status_message = "Listening..."
                self.processing_queue.task_done()

    def update_ui(self) -> Layout:
        # Check for new results
        while not self.result_queue.empty():
            item = self.result_queue.get()
            self.transcript_history.append(item)

        # Header
        self.layout["header"].update(
            Panel(f"Voxtral Live Transcription | Status: {self.status_message}", style="bold white on blue")
        )

        # Main Transcript
        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Time", style="dim cyan", width=10)
        table.add_column("Speaker", style="bold green", width=15)
        table.add_column("Text", style="white")

        # Show last 15 entries
        for item in self.transcript_history[-15:]:
            table.add_row(item.timestamp, item.speaker, item.text)

        self.layout["main"].update(
            Panel(table, title="Live Transcript", border_style="green")
        )

        # Footer
        mode_str = "MOCK MODE" if isinstance(self.audio_stream, MockAudioStream) else "LIVE MODE"
        diar_str = "REAL DIARIZATION" if self.args.auth_token else "SIMPLE DIARIZATION"
        footer_text = f"Press Ctrl+C to exit | {mode_str} | {diar_str}"
        self.layout["footer"].update(
            Panel(footer_text, style="dim white")
        )

        return self.layout

    def run(self):
        self.audio_stream.start()

        chunks_per_second = SAMPLE_RATE / CHUNK_SIZE
        silence_threshold_chunks = int(SILENCE_DURATION_THRESHOLD * chunks_per_second)
        min_turn_chunks = int(MIN_TURN_DURATION * chunks_per_second)

        try:
            with Live(self.layout, refresh_per_second=10, screen=True) as live:
                while self.is_running:
                    chunk = self.audio_stream.get_chunk()

                    if chunk is None:
                        if self.args.mock:
                             time.sleep(0.1)
                             continue
                        break

                    is_speech = self.vad.is_speech(chunk)

                    if is_speech:
                        self.is_speech_active = True
                        self.silence_counter = 0
                        self.current_turn_audio.append(chunk)
                        # We don't update status here to avoid flickering "Transcribing" overwrites
                    else:
                        if self.is_speech_active:
                            self.silence_counter += 1
                            self.current_turn_audio.append(chunk)

                            if self.silence_counter > silence_threshold_chunks:
                                # End of turn
                                if len(self.current_turn_audio) > min_turn_chunks:
                                    # Copy data and send to queue
                                    self.processing_queue.put(list(self.current_turn_audio))

                                self.current_turn_audio = []
                                self.is_speech_active = False
                        else:
                            pass

                    live.update(self.update_ui())

        except KeyboardInterrupt:
            self.is_running = False
        finally:
            self.audio_stream.stop()

def main():
    parser = argparse.ArgumentParser(description="Voxtral Terminal Transcriber")
    parser.add_argument("--mock", action="store_true", help="Use mock audio and model")
    parser.add_argument("--auth-token", type=str, help="Hugging Face auth token for real diarization (pyannote)", default=None)
    args = parser.parse_args()

    app = VoxtralApp(args)
    app.run()

if __name__ == "__main__":
    main()
