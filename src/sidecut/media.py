"""Reading media files: probing, fingerprinting, subtitles, audio windows.

Everything here shells out to ffprobe / ffmpeg and never writes to the media.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts", ".m2ts", ".wmv", ".webm", ".mpg", ".mpeg"}
TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
ENGLISH = {"eng", "en", "english"}


class MediaError(RuntimeError):
    pass


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    text: str


def _run(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, timeout=timeout)
    except FileNotFoundError as err:
        raise MediaError(f"{args[0]} is not installed") from err
    except subprocess.TimeoutExpired as err:
        raise MediaError(f"{args[0]} timed out after {timeout}s") from err


def probe(path: str | os.PathLike) -> dict:
    result = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                  timeout=300)
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed on {path}: {result.stderr.decode(errors='replace').strip()[:300]}")
    return json.loads(result.stdout)


def duration_ms(info: dict) -> int | None:
    value = info.get("format", {}).get("duration")
    try:
        return int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return None


def oshash(path: str | os.PathLike) -> str:
    """OpenSubtitles hash: file size plus checksums of the first and last 64 KiB.
    Cheap on network shares because it reads 128 KiB regardless of file size."""
    block = 65536
    size = os.path.getsize(path)
    if size < block * 2:
        raise MediaError(f"{path}: too small to fingerprint")
    value = size
    with open(path, "rb") as fh:
        for offset in (0, size - block):
            fh.seek(offset)
            data = fh.read(block)
            for (chunk,) in struct.iter_unpack("<Q", data):
                value = (value + chunk) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _lang(stream: dict) -> str:
    return str(stream.get("tags", {}).get("language", "")).lower()


def _title(stream: dict) -> str:
    return str(stream.get("tags", {}).get("title", "")).lower()


def pick_audio_stream(info: dict, language: str = "eng") -> tuple[int, dict] | None:
    """Returns (index among audio streams, stream). Prefers the requested
    language, skips commentary tracks, then takes the default or first."""
    wanted = ENGLISH if language.lower() in ENGLISH else {language.lower()}
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        return None

    def score(item: tuple[int, dict]) -> tuple:
        position, stream = item
        commentary = "comment" in _title(stream) or stream.get("disposition", {}).get("comment")
        return (
            0 if _lang(stream) in wanted else (1 if not _lang(stream) or _lang(stream) == "und" else 2),
            1 if commentary else 0,
            0 if stream.get("disposition", {}).get("default") else 1,
            position,
        )

    return min(enumerate(audio), key=score)


def pick_text_subtitle(info: dict, language: str = "eng") -> tuple[int, dict] | None:
    """Returns (index among subtitle streams, stream) for the best text subtitle
    in the requested language, or None. Image subtitles (PGS, VobSub) are no use
    here. Forced tracks only carry foreign-language lines, so they rank last."""
    wanted = ENGLISH if language.lower() in ENGLISH else {language.lower()}
    subs = [(i, s) for i, s in enumerate(s for s in info.get("streams", []) if s.get("codec_type") == "subtitle")
            if s.get("codec_name") in TEXT_SUB_CODECS and _lang(s) in wanted | {"", "und"}]
    if not subs:
        return None

    def score(item: tuple[int, dict]) -> tuple:
        position, stream = item
        forced = stream.get("disposition", {}).get("forced") or "forced" in _title(stream)
        return (0 if _lang(stream) in wanted else 1, 1 if forced else 0, position)

    return min(subs, key=score)


_SRT_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{1,3})")
_MARKUP = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\r?\n\s*\r?\n", text.replace("﻿", "")):
        lines = [line for line in block.splitlines() if line.strip()]
        for i, line in enumerate(lines):
            m = _SRT_TIME.search(line)
            if not m:
                continue
            h1, m1, s1, f1, h2, m2, s2, f2 = m.groups()
            start = (int(h1) * 3600 + int(m1) * 60 + int(s1)) * 1000 + int(f1.ljust(3, "0"))
            end = (int(h2) * 3600 + int(m2) * 60 + int(s2)) * 1000 + int(f2.ljust(3, "0"))
            body = _MARKUP.sub("", " ".join(lines[i + 1:])).strip()
            if body and end > start:
                cues.append(Cue(start, end, body))
            break
    cues.sort(key=lambda c: c.start_ms)
    return cues


def find_sidecar_subtitle(video: str | os.PathLike, language: str = "eng") -> Path | None:
    """Looks for Movie.srt, Movie.en.srt, Movie.eng.srt, Movie.en.sdh.srt, ..."""
    video = Path(video)
    wanted = ENGLISH if language.lower() in ENGLISH else {language.lower()}
    best: tuple[int, Path] | None = None
    try:
        siblings = list(video.parent.iterdir())
    except OSError:
        return None
    for candidate in siblings:
        if candidate.suffix.lower() != ".srt" or not candidate.name.startswith(video.stem):
            continue
        parts = candidate.name[len(video.stem):].lower().split(".")
        parts = [p for p in parts if p and p != "srt"]
        if "forced" in parts:
            continue
        if not parts:
            rank = 1
        elif any(p in wanted for p in parts):
            rank = 0
        else:
            continue
        if best is None or rank < best[0]:
            best = (rank, candidate)
    return best[1] if best else None


def read_subtitles(video: str | os.PathLike, info: dict | None = None, language: str = "eng",
                   timeout: int = 3600) -> tuple[list[Cue], str]:
    """Returns (cues, where they came from). Empty list when no text subtitles exist."""
    sidecar = find_sidecar_subtitle(video, language)
    if sidecar:
        return parse_srt(sidecar.read_text(encoding="utf-8", errors="replace")), f"sidecar:{sidecar.name}"
    info = info or probe(video)
    picked = pick_text_subtitle(info, language)
    if not picked:
        return [], ""
    index, stream = picked
    result = _run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", f"0:s:{index}", "-f", "srt", "-"],
                  timeout=timeout)
    if result.returncode != 0:
        raise MediaError(f"could not extract subtitles: {result.stderr.decode(errors='replace').strip()[:300]}")
    return parse_srt(result.stdout.decode("utf-8", errors="replace")), f"embedded:{stream.get('codec_name')}#{index}"


def _dialogue_filter(stream: dict | None) -> list[str]:
    """In 5.1 and 7.1 mixes dialogue sits in the centre channel; using it alone
    keeps music and effects out of the recogniser's way."""
    layout = str((stream or {}).get("channel_layout", ""))
    channels = int((stream or {}).get("channels") or 0)
    if channels >= 6 and ("5.1" in layout or "7.1" in layout or "6.1" in layout):
        return ["-af", "pan=mono|c0=FC"]
    return ["-ac", "1"]


def read_audio(video: str | os.PathLike, start_ms: int, length_ms: int, audio_index: int = 0,
               stream: dict | None = None, sample_rate: int = 16000, timeout: int = 600):
    """Decode a window of audio to mono float32 at `sample_rate`. Seeks by
    index, so short windows are fast even in very large files."""
    import numpy as np

    args = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0, start_ms) / 1000:.3f}", "-i", str(video),
            "-t", f"{length_ms / 1000:.3f}", "-map", f"0:a:{audio_index}", "-vn", "-sn",
            *_dialogue_filter(stream), "-ar", str(sample_rate), "-f", "s16le", "-"]
    result = _run(args, timeout=timeout)
    if result.returncode != 0:
        raise MediaError(f"could not decode audio: {result.stderr.decode(errors='replace').strip()[:300]}")
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def iter_videos(root: str | os.PathLike):
    root = Path(root)
    if root.is_file():
        yield root
        return
    for directory, _, files in os.walk(root):
        for name in sorted(files):
            if Path(name).suffix.lower() in VIDEO_EXTENSIONS and not name.startswith("."):
                yield Path(directory) / name
