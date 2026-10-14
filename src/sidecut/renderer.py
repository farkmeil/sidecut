"""Render filtered copies of a library with ffmpeg: a 'family' tree that
mirrors the original folder layout, so Plex matches it the same way.

Mutes are baked into the audio (sample-exact) and the video is copied
untouched; skips need a re-encode of the video, so they are slow and off by
default. Files without a filter, and clean files, are linked rather than
copied so the family tree is complete without a second copy of everything.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import compile as compiler
from . import media, model
from .config import FilterStore
from .model import FilterError
from .profile import Profile, resolve

KEEP_CONTAINER = {".mkv", ".mp4", ".m4v", ".webm", ".mov"}


@dataclass
class RenderOptions:
    profile: Profile
    source_root: Path
    dest_root: Path
    with_skips: bool = False      # re-encode video to cut scenes (slow); otherwise skips are ignored
    clean: str = "link"           # what to do with files the profile leaves untouched: link | copy | skip
    untagged: str = "skip"        # files with no filter: skip | link | copy
    force: bool = False
    dry_run: bool = False
    language: str = "eng"
    timeout: int = 6 * 3600


@dataclass
class RenderReport:
    rendered: list[Path] = field(default_factory=list)
    linked: list[Path] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    up_to_date: list[Path] = field(default_factory=list)


def output_path(video: Path, options: RenderOptions) -> Path:
    relative = video.relative_to(options.source_root)
    target = options.dest_root / relative
    if video.suffix.lower() not in KEEP_CONTAINER:
        target = target.with_suffix(".mkv")
    return target


def audio_codec_for(stream: dict) -> list[str]:
    """Surround stays surround (E-AC-3, which every Plex client plays); stereo
    becomes AAC. Lossless originals cannot be kept: ffmpeg has no TrueHD or
    DTS-HD encoder."""
    channels = int(stream.get("channels") or 2)
    if channels >= 6:
        return ["eac3", "-b:a", "640k", "-ac", "6"]
    return ["aac", "-b:a", "256k"]


def _link_or_copy(video: Path, target: Path, how: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    if how == "copy":
        shutil.copy2(video, target)
        return
    try:
        os.link(video, target)
    except OSError:
        os.symlink(video.absolute(), target)


def render_library(store: FilterStore, options: RenderOptions,
                   log: Callable[[str], None] = lambda message: None) -> RenderReport:
    report = RenderReport()
    videos = list(media.iter_videos(options.source_root))
    for number, video in enumerate(videos, 1):
        target = output_path(video, options)
        filter_path = store.find(store.to_plex(str(video.absolute())))
        label = f"[{number}/{len(videos)}] {video.relative_to(options.source_root)}"
        if not filter_path:
            if options.untagged == "skip":
                report.skipped.append((video, "no filter"))
                log(f"{label}: no filter, skipped")
                continue
            log(f"{label}: no filter, {options.untagged}")
            if not options.dry_run:
                _link_or_copy(video, target, options.untagged)
            report.linked.append(target)
            continue
        try:
            flt = model.load(filter_path)
        except FilterError as err:
            report.failed.append((video, f"bad filter: {err}"))
            log(f"{label}: FAILED bad filter: {err}")
            continue
        try:
            info = media.probe(video)
        except media.MediaError as err:
            report.failed.append((video, str(err)))
            log(f"{label}: FAILED {err}")
            continue
        duration = media.duration_ms(info) or 0
        same_cut, why = flt.media.matches(duration, video.stat().st_size)
        if not same_cut:
            report.skipped.append((video, f"filter is for a different cut: {why}"))
            log(f"{label}: filter is for a different cut of this file ({why}), skipped")
            continue
        plan = resolve(flt, options.profile, duration)
        skips = [a for a in plan if a.kind == "skip"]
        if skips and not options.with_skips:
            plan = [a for a in plan if a.kind != "skip"]
        if not plan:
            if options.clean == "skip":
                report.skipped.append((video, "nothing to filter"))
                log(f"{label}: nothing to filter, skipped")
                continue
            log(f"{label}: nothing to filter, {options.clean}ed")
            if not options.dry_run:
                _link_or_copy(video, target, options.clean)
            report.linked.append(target)
            continue
        if target.exists() and not target.is_symlink() and not options.force \
                and target.stat().st_mtime >= filter_path.stat().st_mtime:
            report.up_to_date.append(target)
            log(f"{label}: up to date")
            continue
        picked = media.pick_audio_stream(info, flt.media.audio_language or options.language)
        if not picked:
            report.failed.append((video, "no audio stream"))
            log(f"{label}: FAILED no audio stream")
            continue
        audio_index, audio_stream = picked
        mutes = [a for a in plan if a.kind == "mute"]
        what = f"{len(mutes)} mutes" + (f", {len(skips)} skips cut" if skips and options.with_skips else "")
        if skips and not options.with_skips:
            what += f" ({len(skips)} skips ignored: --with-skips re-encodes video)"
        log(f"{label}: {what}")
        if options.dry_run:
            report.rendered.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part" + target.suffix)
        args = compiler.to_ffmpeg(plan, str(video), str(partial), audio_stream=audio_index,
                                  audio_codec=audio_codec_for(audio_stream),
                                  subtitles=target.suffix.lower() == ".mkv")
        args[1:1] = ["-loglevel", "error"]
        started = time.monotonic()
        try:
            result = subprocess.run(args, capture_output=True, timeout=options.timeout)
        except subprocess.TimeoutExpired:
            partial.unlink(missing_ok=True)
            report.failed.append((video, "ffmpeg timed out"))
            log(f"{label}: FAILED ffmpeg timed out")
            continue
        if result.returncode != 0 or not partial.exists() or partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            detail = result.stderr.decode(errors="replace").strip()[-300:]
            report.failed.append((video, detail))
            log(f"{label}: FAILED {detail}")
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        partial.replace(target)
        report.rendered.append(target)
        log(f"    {time.monotonic() - started:.0f}s -> {target}")
    return report
