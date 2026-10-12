"""The filter file: data model, validation, load and save.

See SPEC.md for the format. Times are integer milliseconds from the start of
the file the filter was made for.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import SPEC_VERSION, __version__

CATEGORIES = ("language", "sex_nudity", "violence", "substances", "frightening", "other")
ACTIONS = ("mute", "skip")
DEFAULT_ACTION = {
    "language": "mute",
    "sex_nudity": "skip",
    "violence": "skip",
    "substances": "skip",
    "frightening": "skip",
    "other": "skip",
}
SUFFIX = ".sidecut.json"


class FilterError(ValueError):
    """A filter file that does not follow the spec."""


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    category: str
    severity: int = 2
    action: str | None = None
    tags: list[str] = field(default_factory=list)
    note: str = ""
    source: str = ""
    confidence: float | None = None

    @property
    def suggested_action(self) -> str:
        return self.action or DEFAULT_ACTION[self.category]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "category": self.category,
            "severity": self.severity,
        }
        if self.action:
            d["action"] = self.action
        if self.tags:
            d["tags"] = list(self.tags)
        if self.note:
            d["note"] = self.note
        if self.source:
            d["source"] = self.source
        if self.confidence is not None:
            d["confidence"] = round(float(self.confidence), 3)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], where: str = "segment") -> "Segment":
        if not isinstance(d, dict):
            raise FilterError(f"{where}: must be an object")
        for key in ("start_ms", "end_ms", "category"):
            if key not in d:
                raise FilterError(f"{where}: missing '{key}'")
        start, end = d["start_ms"], d["end_ms"]
        for name, value in (("start_ms", start), ("end_ms", end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise FilterError(f"{where}: '{name}' must be an integer number of milliseconds")
        if start < 0:
            raise FilterError(f"{where}: start_ms is negative")
        if end <= start:
            raise FilterError(f"{where}: end_ms must be after start_ms")
        category = d["category"]
        if category not in CATEGORIES:
            raise FilterError(f"{where}: unknown category '{category}' (use one of {', '.join(CATEGORIES)})")
        severity = d.get("severity", 2)
        if isinstance(severity, bool) or not isinstance(severity, int) or not 1 <= severity <= 3:
            raise FilterError(f"{where}: severity must be 1, 2 or 3")
        action = d.get("action")
        if action is not None and action not in ACTIONS:
            raise FilterError(f"{where}: unknown action '{action}' (use one of {', '.join(ACTIONS)})")
        tags = d.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise FilterError(f"{where}: tags must be a list of strings")
        confidence = d.get("confidence")
        if confidence is not None and not (isinstance(confidence, (int, float)) and 0 <= confidence <= 1):
            raise FilterError(f"{where}: confidence must be between 0 and 1")
        return cls(
            start_ms=start,
            end_ms=end,
            category=category,
            severity=severity,
            action=action,
            tags=list(tags),
            note=str(d.get("note", "")),
            source=str(d.get("source", "")),
            confidence=confidence,
        )


@dataclass
class MediaRef:
    """Which file a filter was made for.

    Timings only hold for one specific cut of a title, so the fingerprint
    (duration, size, oshash) is what players check before applying a filter.
    """

    title: str = ""
    year: int | None = None
    ids: dict[str, str] = field(default_factory=dict)
    season: int | None = None
    episode: int | None = None
    edition: str = ""
    duration_ms: int | None = None
    size: int | None = None
    oshash: str = ""
    audio_language: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for key in ("title", "year", "season", "episode", "edition", "duration_ms", "size", "oshash", "audio_language"):
            value = getattr(self, key)
            if value not in (None, "", {}):
                d[key] = value
        if self.ids:
            d["ids"] = dict(self.ids)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MediaRef":
        if not isinstance(d, dict):
            raise FilterError("media: must be an object")
        ids = d.get("ids", {})
        if not isinstance(ids, dict):
            raise FilterError("media.ids: must be an object")
        for key in ("duration_ms", "size", "year", "season", "episode"):
            value = d.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise FilterError(f"media.{key}: must be a non-negative integer")
        return cls(
            title=str(d.get("title", "")),
            year=d.get("year"),
            ids={str(k): str(v) for k, v in ids.items()},
            season=d.get("season"),
            episode=d.get("episode"),
            edition=str(d.get("edition", "")),
            duration_ms=d.get("duration_ms"),
            size=d.get("size"),
            oshash=str(d.get("oshash", "")),
            audio_language=str(d.get("audio_language", "")),
        )

    def matches(self, duration_ms: int | None = None, size: int | None = None,
                oshash: str | None = None, tolerance_ms: int = 2000) -> tuple[bool, str]:
        """Is this the same cut the filter was made for? Returns (ok, reason)."""
        if oshash and self.oshash:
            if oshash.lower() != self.oshash.lower():
                return False, "file hash differs from the one the filter was made for"
            return True, "hash match"
        if size is not None and self.size is not None and size != self.size:
            return False, f"file size {size} differs from filter's {self.size}"
        if duration_ms is not None and self.duration_ms is not None:
            if abs(duration_ms - self.duration_ms) > tolerance_ms:
                return False, f"duration {duration_ms} ms differs from filter's {self.duration_ms} ms"
            return True, "duration match"
        if size is not None and self.size is not None:
            return True, "size match"
        return True, "no fingerprint to compare"


@dataclass
class Filter:
    media: MediaRef = field(default_factory=MediaRef)
    segments: list[Segment] = field(default_factory=list)
    generator: dict[str, Any] = field(default_factory=dict)
    version: str = SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"sidecut": self.version, "media": self.media.to_dict()}
        if self.generator:
            d["generator"] = dict(self.generator)
        d["segments"] = [s.to_dict() for s in sorted(self.segments, key=lambda s: (s.start_ms, s.end_ms))]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Filter":
        if not isinstance(d, dict):
            raise FilterError("filter: top level must be an object")
        version = d.get("sidecut")
        if not isinstance(version, str):
            raise FilterError("filter: missing 'sidecut' version string")
        if version.split(".")[0] != SPEC_VERSION.split(".")[0]:
            raise FilterError(f"filter: spec version {version} is not supported by this build ({SPEC_VERSION})")
        segments_raw = d.get("segments")
        if not isinstance(segments_raw, list):
            raise FilterError("filter: 'segments' must be a list")
        media = MediaRef.from_dict(d.get("media", {}))
        segments = [Segment.from_dict(s, f"segments[{i}]") for i, s in enumerate(segments_raw)]
        if media.duration_ms:
            for i, s in enumerate(segments):
                if s.start_ms >= media.duration_ms + 1000:
                    raise FilterError(f"segments[{i}]: starts after the end of the media")
        generator = d.get("generator", {})
        if not isinstance(generator, dict):
            raise FilterError("filter: 'generator' must be an object")
        return cls(media=media, segments=segments, generator=generator, version=version)

    def summary(self) -> dict[str, dict[int, int]]:
        out: dict[str, dict[int, int]] = {}
        for s in self.segments:
            out.setdefault(s.category, {}).setdefault(s.severity, 0)
            out[s.category][s.severity] += 1
        return out


def load(path: str | os.PathLike) -> Filter:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as err:
            raise FilterError(f"{path}: not valid JSON ({err})") from err
    try:
        return Filter.from_dict(raw)
    except FilterError as err:
        raise FilterError(f"{path}: {err}") from err


def save(flt: Filter, path: str | os.PathLike) -> None:
    """Write atomically so a player never reads a half-written filter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(flt.to_dict(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def default_generator(tool: str = "sidecut") -> dict[str, Any]:
    import datetime

    return {
        "tool": f"{tool} {__version__}",
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_time(text: str | int | float) -> int:
    """'1:02:03.5', '62:03', '3723.5' (seconds) or '3723500ms' -> milliseconds."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return int(round(float(text) * 1000))
    text = str(text).strip()
    if text.endswith("ms"):
        return int(text[:-2])
    m = _TIME_RE.match(text)
    if not m:
        raise ValueError(f"cannot read time '{text}'")
    a, b, c = m.groups()
    if a is not None and b is not None:
        hours, minutes = int(a), int(b)
    elif a is not None:
        hours, minutes = 0, int(a)
    else:
        hours, minutes = 0, 0
    return int(round((hours * 3600 + minutes * 60 + float(c)) * 1000))


def format_time(ms: int) -> str:
    ms = max(0, int(ms))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{millis:03d}"
