"""SRT subtitle generator from VAD-aligned translation entries."""

from pathlib import Path


def _fmt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm."""
    h   = int(seconds // 3600)
    m   = int((seconds % 3600) // 60)
    s   = int(seconds % 60)
    ms  = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(
    entries: list[tuple[float, float, str]],
    output_path: Path,
) -> Path:
    """Write an SRT subtitle file.

    Args:
        entries:     List of (start_seconds, end_seconds, text) tuples.
        output_path: Where to write the .srt file.

    Returns:
        output_path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    idx = 1
    with open(output_path, "w", encoding="utf-8") as f:
        for start, end, text in entries:
            text = text.strip()
            if not text:
                continue
            f.write(f"{idx}\n")
            f.write(f"{_fmt_time(start)} --> {_fmt_time(end)}\n")
            f.write(f"{text}\n\n")
            idx += 1
    return output_path
