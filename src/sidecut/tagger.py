"""Language tagger: find profanity in a video and write it as filter segments.

Method
  1. Read subtitles (a sidecar .srt, an embedded text track, or image
     subtitles put through OCR) and find the lines that contain listed words.
     Subtitles only decide where to listen; their timing is too loose to mute
     single words.
  2. Align the text of those lines to the audio around them (forced
     alignment), which places every word of the line to within a few tens of
     milliseconds. A word the aligner could not place convincingly (the audio
     does not say what the subtitle shows) is left to the next steps.
  3. Run speech recognition with word timestamps on the same audio and match
     the word list against what was said. This catches words the subtitles
     left out, and times words the aligner could not place.
  4. If a subtitle line shows a listed word that neither step placed, the
     whole line is tagged instead, so nothing the subtitles show slips through.
  5. Subtitles leave words out (a cleaned-up transcript, a dropped "fucking"),
     so the whole soundtrack is transcribed as well and anything heard that
     the steps above did not already place is tagged too.

With no usable subtitles at all, step 5 is all there is.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import media
from .align import Aligner, spoken_words
from .asr import Transcriber
from .model import Filter, MediaRef, Segment, default_generator
from .wordlist import TimedMatch, Word, Wordlist, find_in_words

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
    ocr: bool = True         # read image subtitles (Blu-ray PGS) by OCR when there are no text ones
    vad: bool = True         # in whole-soundtrack mode, skip stretches without speech
    asr: bool = True         # run recognition on subtitle windows as well as alignment
    thorough: bool = True    # with subtitles, transcribe the whole soundtrack too for words they leave out
    align_min_score: float = 0.4   # below this an aligned word is treated as not placed
    align_window_max_ms: int = 45_000
    asr_slack_ms: tuple[int, int] = (130, 150)  # recognised words start and end early by about this much


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


def _alignment_windows(windows: list[tuple[int, int]], cues: list[media.Cue], pad_ms: int,
                       max_ms: int) -> list[tuple[int, int]]:
    """Widen listening windows so every subtitle line they touch is inside
    them whole, since the aligner is told the text of those lines. Windows
    that would grow past `max_ms` are split between lines instead."""
    out: list[tuple[int, int]] = []
    for start, end in windows:
        inside = [c for c in cues if c.end_ms > start and c.start_ms < end]
        lo = min([start] + [c.start_ms - pad_ms for c in inside])
        hi = max([end] + [c.end_ms + pad_ms for c in inside])
        if hi - lo <= max_ms or not inside:
            out.append((lo, hi))
            continue
        group: list[media.Cue] = []
        for cue in inside:
            if group and cue.end_ms + pad_ms - (group[0].start_ms - pad_ms) > max_ms:
                out.append((group[0].start_ms - pad_ms, group[-1].end_ms + pad_ms))
                group = []
            group.append(cue)
        out.append((group[0].start_ms - pad_ms, group[-1].end_ms + pad_ms))
    return sorted(out)


def _same_word(a: TimedMatch, b: TimedMatch, slack_ms: int = 250) -> bool:
    return a.start_ms < b.end_ms + slack_ms and b.start_ms < a.end_ms + slack_ms


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
                 log: Callable[[str], None] = lambda message: None,
                 aligner: Aligner | None = None) -> tuple[Filter, dict]:
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
        try:
            cues, subtitle_source = media.read_subtitles(video, info, options.language, ocr=options.ocr)
        except media.MediaError as err:
            if not options.allow_full:
                raise NoSubtitles(f"{video.name}: subtitles unreadable: {err}") from err
            log(f"subtitles unreadable ({err}); carrying on without them")
        if not cues and not options.allow_full:
            raise NoSubtitles(f"{video.name}: no text subtitles")
    if cues and aligner is None:
        method = "subtitle+asr"
    elif cues:
        method = "subtitle+align+asr" if options.asr else "subtitle+align"
    else:
        method = "asr-full"
    if cues and options.thorough:
        method += "+full"
    report: dict = {
        "method": method,
        "subtitle_source": subtitle_source,
        "audio_stream": audio_index,
        "audio_codec": audio_stream.get("codec_name"),
    }

    def clip(start_ms: int, end_ms: int) -> tuple[int, int]:
        return max(0, start_ms), min(total_ms, end_ms)

    def hear(audio, offset_ms: int, vad: bool = False) -> list[TimedMatch]:
        words = transcriber.words(audio, whisper_language, vad=vad)
        return [TimedMatch(m.start_ms + offset_ms, m.end_ms + offset_ms, m.severity, m.tag, m.probability)
                for m in find_in_words(wordlist, words)]

    def listen(start_ms: int, end_ms: int, vad: bool = False) -> list[TimedMatch]:
        start_ms, end_ms = clip(start_ms, end_ms)
        if end_ms - start_ms < 200:
            return []
        audio = media.read_audio(video, start_ms, end_ms - start_ms, audio_index, audio_stream)
        return hear(audio, start_ms, vad) if len(audio) else []

    segments: list[Segment] = []
    heard: list[TimedMatch] = []
    aligned: list[TimedMatch] = []
    extra: list[TimedMatch] = []   # heard in the whole soundtrack, shown by no subtitle

    def listen_to_everything() -> list[TimedMatch]:
        step = options.chunk_ms
        position = 0
        chunk_number = 0
        chunks = max(1, -(-total_ms // step))
        found: list[TimedMatch] = []
        while position < total_ms:
            chunk_number += 1
            found.extend(listen(position, position + step + options.chunk_overlap_ms, vad=options.vad))
            log(f"  chunk {chunk_number}/{chunks}")
            position += step
        report.update(chunks=chunks)
        return _dedupe(found)

    if cues:
        hit_cues = [(index, cue, found) for index, cue in enumerate(cues) if (found := wordlist.find(cue.text))]
        windows = _merge_windows([(cue.start_ms - options.window_pad_ms, cue.end_ms + options.window_pad_ms)
                                  for _, cue, _ in hit_cues])
        if aligner is not None:
            windows = _alignment_windows(windows, cues, options.window_pad_ms, options.align_window_max_ms)
        log(f"{len(cues)} subtitle lines, {len(hit_cues)} with listed words, {len(windows)} audio windows "
            f"({sum(e - s for s, e in windows) / 1000:.0f}s to listen to)")
        hit_index = {index for index, _, _ in hit_cues}
        aligned_by_cue: dict[int, list[TimedMatch]] = {}
        for number, (start, end) in enumerate(windows, 1):
            start, end = clip(start, end)
            if end - start < 200:
                continue
            audio = media.read_audio(video, start, end - start, audio_index, audio_stream)
            if len(audio) == 0:
                continue
            if aligner is not None:
                inside = [(index, cue) for index, cue in enumerate(cues)
                          if cue.start_ms >= start and cue.end_ms <= end and index not in aligned_by_cue]
                words: list[str] = []
                owners: list[int] = []
                for index, cue in inside:
                    spoken = spoken_words(cue.text)
                    words.extend(spoken)
                    owners.extend([index] * len(spoken))
                placed = aligner.align(audio, words) if words else []
                for index, cue in inside:
                    aligned_by_cue[index] = []
                    if index not in hit_index:
                        continue
                    cue_words = [Word(w.start_ms + start, w.end_ms + start, w.text, w.probability)
                                 for w, owner in zip(placed, owners) if owner == index]
                    aligned_by_cue[index] = [m for m in find_in_words(wordlist, cue_words)
                                             if m.probability >= options.align_min_score]
                    aligned.extend(aligned_by_cue[index])
            if options.asr or aligner is None:
                heard.extend(hear(audio, start))
            if number % 25 == 0:
                log(f"  window {number}/{len(windows)}")
        heard = [m for m in _dedupe(heard) if not any(_same_word(m, a) for a in aligned)]
        whole_lines = 0
        for index, cue, found in hit_cues:
            lo, hi = cue.start_ms - options.cue_slack_ms, cue.end_ms + options.cue_slack_ms
            located = len(aligned_by_cue.get(index, [])) + sum(1 for m in heard if lo <= (m.start_ms + m.end_ms) // 2 <= hi)
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
                      words_aligned=len(aligned), whole_line_fallbacks=whole_lines)
        if options.thorough:
            log("listening to the whole soundtrack as well, for words the subtitles leave out")
            whole_lines_spans = [(s.start_ms, s.end_ms) for s in segments]
            placed = aligned + heard
            for match in listen_to_everything():
                middle = (match.start_ms + match.end_ms) // 2
                if any(_same_word(match, p) for p in placed) or any(a <= middle <= b for a, b in whole_lines_spans):
                    continue
                extra.append(match)
            report.update(words_unsubtitled=len(extra))
    else:
        log("no text subtitles found; transcribing the whole soundtrack" if not options.full
            else "transcribing the whole soundtrack")
        heard = listen_to_everything()

    before, after = options.asr_slack_ms
    for match, source in [(m, "subtitle+asr" if cues else "asr") for m in heard] + [(m, "asr") for m in extra]:
        # recognised words come out a little early at both ends (measured in tools/timing_check.py)
        start_ms, end_ms = max(0, match.start_ms - before), min(total_ms, match.end_ms + after)
        short_by = options.min_word_ms - (end_ms - start_ms)
        if short_by > 0:
            start_ms = max(0, start_ms - short_by // 2)
            end_ms = min(total_ms, start_ms + options.min_word_ms)
        segments.append(Segment(
            start_ms=start_ms, end_ms=end_ms, category="language", severity=match.severity,
            action="mute", tags=[match.tag], source=source, confidence=match.probability,
        ))
    for match in aligned:
        start_ms, end_ms = match.start_ms, min(total_ms, match.end_ms)
        short_by = options.min_word_ms - (end_ms - start_ms)
        if short_by > 0:
            start_ms = max(0, start_ms - short_by // 2)
            end_ms = min(total_ms, start_ms + options.min_word_ms)
        segments.append(Segment(
            start_ms=start_ms, end_ms=end_ms, category="language", severity=match.severity,
            action="mute", tags=[match.tag], source="subtitle+align", confidence=round(match.probability, 3),
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
        "aligner": getattr(aligner, "model_name", "") if aligner is not None else "",
        "subtitles": subtitle_source,
    }
    report.update(
        words_located=len(heard) + len(aligned) + len(extra),
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
    machine = {"subtitle", "subtitle+asr", "subtitle+align", "asr"}
    kept = [s for s in existing.segments if not (s.category == "language" and s.source in machine)]
    generator = dict(existing.generator)
    generator.update(fresh.generator)
    media_ref = fresh.media
    if existing.media.ids:
        media_ref.ids = dict(existing.media.ids)
    return Filter(media=media_ref, segments=kept + fresh.segments, generator=generator)
