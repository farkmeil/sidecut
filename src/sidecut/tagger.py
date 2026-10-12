"""Language tagger: find profanity in a video and write it as filter segments.

Method
  1. Read text subtitles (sidecar .srt or an embedded text track) and find the
     lines that contain listed words. Subtitles are only used to decide where
     to listen; their timing is too loose to mute single words.
  2. Run speech recognition with word timestamps on a few seconds of audio
     around each such line, and match the word list against what was said.
     That gives a tight interval per word.
  3. If a subtitle line shows a listed word that recognition did not hear, the
     whole line is tagged instead, so nothing the subtitles show slips through.

With no text subtitles (image-based PGS, or none at all) the whole soundtrack
is transcribed instead. That is slower and relies on recognition alone.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import media
from .asr import Transcriber
from .model import Filter, MediaRef, Segment, default_generator
from .wordlist import TimedMatch, Wordlist, find_in_words

WHISPER_LANG = {"eng": "en", "en": "en", "english": "en", "spa": "es", "fre": "fr", "fra": "fr", "ger": "de",
                "deu": "de"}


@dataclass
class TagOptions:
    language: str = "eng"
    window_pad_ms: int = 1500
    cue_slack_ms: int = 600
    full: bool = False
    chunk_ms: int = 300_000
    chunk_overlap_ms: int = 2000
    min_word_ms: int = 300  # recognisers often clip short words; widen anything shorter than this
    allow_full: bool = True  # transcribe the whole soundtrack when there are no text subtitles
    vad: bool = True         # in whole-soundtrack mode, skip stretches without speech


class NoSubtitles(media.MediaError):
    """The video has no text subtitles and whole-soundtrack mode is off."""


def guess_identity(path: str | os.PathLike) -> dict:
    """Title/year/season/episode from the usual 'Title (Year)/file' and
    'Show/Season 01/Show - S01E02 - Name' layouts. Best effort only."""
    path = Path(path)
    out: dict = {}
    episode = re.search(r"[Ss](\d{1,2})[ ._-]?[Ee](\d{1,3})", path.name)
    if episode:
        out["season"], out["episode"] = int(episode.group(1)), int(episode.group(2))
        show_dir = path.parent
        if re.match(r"(?i)^(season|series|specials)\b", show_dir.name):
            show_dir = show_dir.parent
        name = show_dir.name
    else:
        name = path.parent.name if path.parent.name else path.stem
    year = re.search(r"\((\d{4})\)", name)
    if year:
        out["year"] = int(year.group(1))
        name = name[: year.start()]
    name = re.sub(r"\s*[\[{].*$", "", name).strip(" .-_")
    if name:
        out["title"] = name
    return out


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _dedupe(matches: list[TimedMatch]) -> list[TimedMatch]:
    """Chunk overlaps can hear the same word twice."""
    out: list[TimedMatch] = []
    for match in sorted(matches, key=lambda m: (m.start_ms, m.end_ms)):
        if out and match.tag == out[-1].tag and match.start_ms < out[-1].end_ms:
            if match.probability > out[-1].probability:
                out[-1] = match
            continue
        out.append(match)
    return out


def tag_language(video: str | os.PathLike, transcriber: Transcriber, wordlist: Wordlist,
                 options: TagOptions | None = None,
                 log: Callable[[str], None] = lambda message: None) -> tuple[Filter, dict]:
    """Returns (filter, report). Never modifies the video."""
    options = options or TagOptions()
    started = time.monotonic()
    video = Path(video)
    info = media.probe(video)
    total_ms = media.duration_ms(info)
    if not total_ms:
        raise media.MediaError(f"{video}: could not read duration")
    picked = media.pick_audio_stream(info, options.language)
    if not picked:
        raise media.MediaError(f"{video}: no audio stream")
    audio_index, audio_stream = picked
    whisper_language = WHISPER_LANG.get(options.language.lower(), options.language.lower()[:2])

    cues: list[media.Cue] = []
    subtitle_source = ""
    if not options.full:
        cues, subtitle_source = media.read_subtitles(video, info, options.language)
        if not cues and not options.allow_full:
            raise NoSubtitles(f"{video.name}: no text subtitles")
    report: dict = {
        "method": "subtitle+asr" if cues else "asr-full",
        "subtitle_source": subtitle_source,
        "audio_stream": audio_index,
        "audio_codec": audio_stream.get("codec_name"),
    }

    def listen(start_ms: int, end_ms: int, vad: bool = False) -> list[TimedMatch]:
        start_ms = max(0, start_ms)
        end_ms = min(total_ms, end_ms)
        if end_ms - start_ms < 200:
            return []
        audio = media.read_audio(video, start_ms, end_ms - start_ms, audio_index, audio_stream)
        if len(audio) == 0:
            return []
        words = transcriber.words(audio, whisper_language, vad=vad)
        return [TimedMatch(m.start_ms + start_ms, m.end_ms + start_ms, m.severity, m.tag, m.probability)
                for m in find_in_words(wordlist, words)]

    segments: list[Segment] = []
    heard: list[TimedMatch] = []
    if cues:
        hit_cues = [(cue, found) for cue in cues if (found := wordlist.find(cue.text))]
        windows = _merge_windows([(cue.start_ms - options.window_pad_ms, cue.end_ms + options.window_pad_ms)
                                  for cue, _ in hit_cues])
        log(f"{len(cues)} subtitle lines, {len(hit_cues)} with listed words, {len(windows)} audio windows "
            f"({sum(e - s for s, e in windows) / 1000:.0f}s to transcribe)")
        for number, (start, end) in enumerate(windows, 1):
            heard.extend(listen(start, end))
            if number % 25 == 0:
                log(f"  window {number}/{len(windows)}")
        heard = _dedupe(heard)
        whole_lines = 0
        for cue, found in hit_cues:
            lo, hi = cue.start_ms - options.cue_slack_ms, cue.end_ms + options.cue_slack_ms
            located = sum(1 for m in heard if lo <= (m.start_ms + m.end_ms) // 2 <= hi)
            if located >= len(found):
                continue
            worst = max(found, key=lambda m: m.severity)
            whole_lines += 1
            segments.append(Segment(
                start_ms=cue.start_ms, end_ms=min(cue.end_ms, total_ms), category="language",
                severity=worst.severity, action="mute", tags=[worst.tag],
                note="whole subtitle line: word not located in audio", source="subtitle", confidence=0.5,
            ))
        report.update(subtitle_lines=len(cues), lines_with_hits=len(hit_cues), windows=len(windows),
                      whole_line_fallbacks=whole_lines)
    else:
        log("no text subtitles found; transcribing the whole soundtrack" if not options.full
            else "transcribing the whole soundtrack")
        step = options.chunk_ms
        position = 0
        chunk_number = 0
        chunks = max(1, -(-total_ms // step))
        while position < total_ms:
            chunk_number += 1
            heard.extend(listen(position, position + step + options.chunk_overlap_ms, vad=options.vad))
            log(f"  chunk {chunk_number}/{chunks}")
            position += step
        heard = _dedupe(heard)
        report.update(chunks=chunks)

    source = "subtitle+asr" if cues else "asr"
    for match in heard:
        start_ms, end_ms = match.start_ms, match.end_ms
        short_by = options.min_word_ms - (end_ms - start_ms)
        if short_by > 0:
            start_ms = max(0, start_ms - short_by // 2)
            end_ms = min(total_ms, start_ms + options.min_word_ms)
        segments.append(Segment(
            start_ms=start_ms, end_ms=end_ms, category="language", severity=match.severity,
            action="mute", tags=[match.tag], source=source, confidence=match.probability,
        ))
    segments.sort(key=lambda s: (s.start_ms, s.end_ms))

    identity = guess_identity(video)
    try:
        fingerprint = media.oshash(video)
    except (OSError, media.MediaError):
        fingerprint = ""
    ref = MediaRef(
        title=identity.get("title", ""), year=identity.get("year"), season=identity.get("season"),
        episode=identity.get("episode"), duration_ms=total_ms, size=os.path.getsize(video), oshash=fingerprint,
        audio_language=options.language,
    )
    generator = default_generator()
    generator["language_tagger"] = {
        "method": report["method"],
        "model": getattr(transcriber, "model_name", ""),
        "subtitles": subtitle_source,
    }
    report.update(
        words_located=len(heard),
        segments=len(segments),
        by_tag=_count(segments),
        elapsed_s=round(time.monotonic() - started, 1),
    )
    return Filter(media=ref, segments=segments, generator=generator), report


def _count(segments: list[Segment]) -> dict[str, int]:
    out: dict[str, int] = {}
    for segment in segments:
        key = segment.tags[0] if segment.tags else segment.category
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: -item[1]))


def merge_language(existing: Filter, fresh: Filter) -> Filter:
    """Re-tagging replaces machine-made language segments and keeps everything
    else (hand-tagged scenes, hand-added language segments)."""
    machine = {"subtitle", "subtitle+asr", "asr"}
    kept = [s for s in existing.segments if not (s.category == "language" and s.source in machine)]
    generator = dict(existing.generator)
    generator.update(fresh.generator)
    media_ref = fresh.media
    if existing.media.ids:
        media_ref.ids = dict(existing.media.ids)
    return Filter(media=media_ref, segments=kept + fresh.segments, generator=generator)
