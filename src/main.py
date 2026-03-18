import argparse
import logging
import queue
import sys
import threading
import time
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from audio import VAD, FileAudioStream, MicrophoneStream, MockAudioStream
from config import load_config
from diarization import load_diarizer
from export import export_transcript
from transcription import load_transcriber

logger = logging.getLogger("VoxtralApp")


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
    def __init__(self, args, config=None):
        self.args = args
        self.config = config or load_config(getattr(args, "config", None))
        self.console = Console()
        self.layout = create_layout()
        self.transcript_history = []
        self.status_message = "Initializing..."
        self.is_running = True
        self.is_paused = False

        # Audio buffer for current turn
        self.current_turn_audio = []
        self.silence_counter = 0
        self.is_speech_active = False

        # Queues for threading
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Config values
        self.sample_rate = self.config["sample_rate"]
        self.chunk_size = self.config["chunk_size"]
        self.silence_duration_threshold = self.config["silence_duration_threshold"]
        self.min_turn_duration = self.config["min_turn_duration"]

        # Components
        self.vad = VAD(threshold=self.config["vad_threshold"])

        # Initialize Audio
        self._init_audio_stream()

        # Initialize Models
        self.status_message = "Loading Models..."
        self.transcriber = load_transcriber(use_mock=args.mock)

        use_mock_diarizer = args.mock or (not args.auth_token)
        self.diarizer = load_diarizer(use_mock=use_mock_diarizer, auth_token=args.auth_token)

        self.status_message = "Ready. Listening..."

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

    def _init_audio_stream(self):
        """Initialize the appropriate audio stream based on args."""
        if self.args.mock:
            self.audio_stream = MockAudioStream(rate=self.sample_rate, chunk_size=self.chunk_size)
        elif self.args.input_file:
            try:
                self.audio_stream = FileAudioStream(
                    self.args.input_file, rate=self.sample_rate, chunk_size=self.chunk_size
                )
            except Exception as e:
                self.console.print(f"[red]Error loading audio file:[/red] {e}")
                sys.exit(1)
        else:
            try:
                self.audio_stream = MicrophoneStream(rate=self.sample_rate, chunk_size=self.chunk_size)
            except ImportError:
                self.console.print("[red]PyAudio is not installed.[/red] Install it with: pip install pyaudio")
                self.console.print("[yellow]Falling back to Mock Audio Stream.[/yellow]")
                self.audio_stream = MockAudioStream(rate=self.sample_rate, chunk_size=self.chunk_size)
            except OSError as e:
                self.console.print(f"[red]No audio device found:[/red] {e}")
                self.console.print("[yellow]Falling back to Mock Audio Stream.[/yellow]")
                self.audio_stream = MockAudioStream(rate=self.sample_rate, chunk_size=self.chunk_size)
            except Exception as e:
                self.console.print(f"[red]Error initializing microphone:[/red] {e}")
                self.console.print("[yellow]Falling back to Mock Audio Stream.[/yellow]")
                self.audio_stream = MockAudioStream(rate=self.sample_rate, chunk_size=self.chunk_size)

    def _processing_worker(self):
        """Background thread to process audio turns."""
        while self.is_running:
            try:
                audio_data = self.processing_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if len(audio_data) == 0:
                continue

            full_audio = np.concatenate(audio_data)

            self.status_message = "Transcribing..."
            try:
                text = self.transcriber.transcribe(full_audio)
                speaker = self.diarizer.identify_speaker(full_audio)
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.result_queue.put(TranscriptItem(timestamp, speaker, text))
            except Exception as e:
                logger.error(f"Error in processing: {e}")
            finally:
                self.status_message = "Listening..."
                self.processing_queue.task_done()

    def export(self, fmt=None):
        """Export the current transcript."""
        fmt = fmt or self.config["export_format"]
        base_path = self.config["export_path"]
        path = export_transcript(self.transcript_history, fmt=fmt, base_path=base_path)
        if path:
            self.status_message = f"Exported to {path}"
        return path

    def update_ui(self) -> Layout:
        # Check for new results
        while not self.result_queue.empty():
            item = self.result_queue.get()
            self.transcript_history.append(item)

        # Header
        pause_indicator = " [PAUSED]" if self.is_paused else ""
        self.layout["header"].update(
            Panel(
                f"Voxtral Live Transcription | Status: {self.status_message}{pause_indicator}",
                style="bold white on blue",
            )
        )

        # Main Transcript
        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Time", style="dim cyan", width=10)
        table.add_column("Speaker", style="bold green", width=15)
        table.add_column("Text", style="white")

        max_display = self.config["max_transcript_display"]
        for item in self.transcript_history[-max_display:]:
            table.add_row(item.timestamp, item.speaker, item.text)

        self.layout["main"].update(Panel(table, title="Live Transcript", border_style="green"))

        # Footer
        mode_str = "MOCK MODE" if self.args.mock else ("FILE MODE" if self.args.input_file else "LIVE MODE")
        diar_str = "REAL DIARIZATION" if self.args.auth_token else "SIMPLE DIARIZATION"
        shortcuts = "Ctrl+C: quit | p: pause | e: export"
        footer_text = f"{shortcuts} | {mode_str} | {diar_str}"
        self.layout["footer"].update(Panel(footer_text, style="dim white"))

        return self.layout

    def run(self):
        self.audio_stream.start()

        chunks_per_second = self.sample_rate / self.chunk_size
        silence_threshold_chunks = int(self.silence_duration_threshold * chunks_per_second)
        min_turn_chunks = int(self.min_turn_duration * chunks_per_second)

        try:
            with Live(self.layout, refresh_per_second=self.config["refresh_rate"], screen=True) as live:
                while self.is_running:
                    if self.is_paused:
                        time.sleep(0.1)
                        live.update(self.update_ui())
                        continue

                    chunk = self.audio_stream.get_chunk()

                    if chunk is None:
                        if self.args.mock:
                            time.sleep(0.1)
                            continue
                        # For file mode, None means EOF
                        if self.args.input_file:
                            self.status_message = "File processing complete."
                            live.update(self.update_ui())
                            # Wait for remaining processing
                            self.processing_queue.join()
                            # Drain results
                            live.update(self.update_ui())
                            break
                        break

                    is_speech = self.vad.is_speech(chunk)

                    if is_speech:
                        self.is_speech_active = True
                        self.silence_counter = 0
                        self.current_turn_audio.append(chunk)
                    else:
                        if self.is_speech_active:
                            self.silence_counter += 1
                            self.current_turn_audio.append(chunk)

                            if self.silence_counter > silence_threshold_chunks:
                                if len(self.current_turn_audio) > min_turn_chunks:
                                    self.processing_queue.put(list(self.current_turn_audio))

                                self.current_turn_audio = []
                                self.is_speech_active = False

                    live.update(self.update_ui())

        except KeyboardInterrupt:
            self.is_running = False
        finally:
            self.audio_stream.stop()
            # Auto-export if transcript has content
            if self.transcript_history and self.args.export:
                self.export(self.args.export)


def main():
    parser = argparse.ArgumentParser(description="Voxtral Terminal Transcriber")
    parser.add_argument("--mock", action="store_true", help="Use mock audio and model")
    parser.add_argument("--auth-token", type=str, help="Hugging Face auth token for real diarization", default=None)
    parser.add_argument(
        "--input-file", type=str, help="Transcribe an audio file instead of live microphone", default=None
    )
    parser.add_argument("--config", type=str, help="Path to config file (notai.json)", default=None)
    parser.add_argument(
        "--export", type=str, choices=["text", "json", "srt"], help="Auto-export transcript on exit", default=None
    )
    args = parser.parse_args()

    config = load_config(args.config)
    app = VoxtralApp(args, config=config)
    app.run()


if __name__ == "__main__":
    main()
