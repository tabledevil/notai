import json
import logging
from pathlib import Path

logger = logging.getLogger("Export")


def export_text(transcript_items, path):
    """Export transcript as plain text."""
    with open(path, "w") as f:
        for item in transcript_items:
            f.write(f"[{item.timestamp}] {item.speaker}: {item.text}\n")
    logger.info(f"Exported text transcript to {path}")


def export_json(transcript_items, path):
    """Export transcript as JSON."""
    data = [{"timestamp": item.timestamp, "speaker": item.speaker, "text": item.text} for item in transcript_items]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Exported JSON transcript to {path}")


def export_srt(transcript_items, path):
    """Export transcript as SRT subtitle format."""
    with open(path, "w") as f:
        for i, item in enumerate(transcript_items, 1):
            # Use timestamp as approximate start, offset by 3s per entry for duration
            start = item.timestamp
            f.write(f"{i}\n")
            f.write(f"00:{start},000 --> 00:{start},999\n")
            f.write(f"{item.speaker}: {item.text}\n\n")
    logger.info(f"Exported SRT transcript to {path}")


def export_transcript(transcript_items, fmt="text", base_path="./transcript"):
    """Export transcript in the specified format."""
    if not transcript_items:
        logger.warning("No transcript items to export")
        return None

    extensions = {"text": ".txt", "json": ".json", "srt": ".srt"}
    exporters = {"text": export_text, "json": export_json, "srt": export_srt}

    ext = extensions.get(fmt, ".txt")
    exporter = exporters.get(fmt, export_text)

    path = Path(base_path).with_suffix(ext)
    path.parent.mkdir(parents=True, exist_ok=True)

    exporter(transcript_items, path)
    return str(path)
