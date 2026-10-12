"""Tiered word list: loading and matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


class WordlistError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    severity: int
    tag: str
    pattern: re.Pattern


@dataclass(frozen=True)
class Match:
    start: int  # character offsets into the normalised text
    end: int
    severity: int
    tag: str


_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "‐": "-", "‑": "-",
                              "–": "-", "—": " "})


def normalise(text: str) -> str:
    """Lower-case, straighten quotes. Length is preserved so that character
    offsets in the result are valid in the original."""
    return text.translate(_APOSTROPHES).lower()


class Wordlist:
    def __init__(self, entries: list[Entry]):
        self.entries = entries

    @classmethod
    def parse(cls, text: str, name: str = "wordlist") -> "Wordlist":
        entries = []
        for number, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) != 3 or parts[0] not in ("1", "2", "3"):
                raise WordlistError(f"{name}:{number}: expected 'severity tag pattern'")
            severity, tag, pattern = int(parts[0]), parts[1], parts[2].strip()
            # a space in a pattern matches spaces or a hyphen; ' ?' makes it optional
            body = pattern.replace(" ?", r"[\s-]?").replace(" ", r"[\s-]+")
            try:
                compiled = re.compile(rf"(?<![\w'])(?:{body})(?![\w])", re.IGNORECASE)
            except re.error as err:
                raise WordlistError(f"{name}:{number}: bad pattern '{pattern}': {err}") from err
            entries.append(Entry(severity, tag, compiled))
        if not entries:
            raise WordlistError(f"{name}: no entries")
        return cls(entries)

    @classmethod
    def load(cls, path: str | None = None) -> "Wordlist":
        if path:
            return cls.parse(Path(path).read_text(encoding="utf-8"), str(path))
        text = resources.files("sidecut").joinpath("data/wordlist.en.txt").read_text(encoding="utf-8")
        return cls.parse(text, "built-in wordlist")

    def find(self, text: str) -> list[Match]:
        """Non-overlapping matches. Where candidates overlap, the longest wins,
        then the most severe."""
        norm = normalise(text)
        candidates = [
            Match(m.start(), m.end(), entry.severity, entry.tag)
            for entry in self.entries
            for m in entry.pattern.finditer(norm)
        ]
        candidates.sort(key=lambda c: (-(c.end - c.start), -c.severity, c.start))
        chosen: list[Match] = []
        for cand in candidates:
            if all(cand.end <= c.start or cand.start >= c.end for c in chosen):
                chosen.append(cand)
        chosen.sort(key=lambda c: c.start)
        return chosen


@dataclass(frozen=True)
class Word:
    start_ms: int
    end_ms: int
    text: str
    probability: float = 1.0


@dataclass(frozen=True)
class TimedMatch:
    start_ms: int
    end_ms: int
    severity: int
    tag: str
    probability: float


_EDGE_PUNCT = re.compile(r"^[^\w']+|[^\w']+$")


def find_in_words(wordlist: Wordlist, words: list[Word]) -> list[TimedMatch]:
    """Match the word list against recognised words and return timed hits."""
    pieces: list[str] = []
    spans: list[tuple[int, int, int]] = []  # (char start, char end, word index)
    cursor = 0
    for index, word in enumerate(words):
        token = _EDGE_PUNCT.sub("", normalise(word.text.strip()))
        if not token:
            continue
        if pieces:
            cursor += 1
        spans.append((cursor, cursor + len(token), index))
        pieces.append(token)
        cursor += len(token)
    text = " ".join(pieces)
    out: list[TimedMatch] = []
    for match in wordlist.find(text):
        covered = [index for start, end, index in spans if start < match.end and end > match.start]
        if not covered:
            continue
        first, last = words[covered[0]], words[covered[-1]]
        probability = min(words[i].probability for i in covered)
        out.append(TimedMatch(first.start_ms, max(last.end_ms, first.start_ms + 1), match.severity, match.tag,
                              probability))
    return out
