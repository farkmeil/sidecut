"""Forced alignment: given a stretch of audio and the words that are spoken in
it, find when each word is said.

The tagger uses this on subtitle lines. Recognition alone has to *hear* a
listed word to time it; alignment only has to place a word it has been told
is there, which is a much easier job, so the timing is tighter and far fewer
lines fall back to a whole-line mute.

The model is wav2vec2-base-960h (English, character level) exported to ONNX,
run with onnxruntime. A character-level CTC model emits one short spike per
letter and a word-separator spike between words, and a Viterbi pass over the
emissions, constrained to the known text, gives the frame of every spike.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

from .wordlist import Word

SAMPLE_RATE = 16000

# Published with the project so no torch is needed at run time; tools/export_aligner.py makes them.
MODEL_FILES = {
    "wav2vec2-base-960h.onnx": {
        "url": "https://github.com/farkmeil/sidecut/releases/download/aligner-v1/wav2vec2-base-960h.onnx",
        "sha256": "39fc86fc441848139db0a739f7a4cff0f8f35ff40a936a99cc3f07ddc36c4bc6",
    },
    "labels.json": {
        "url": "https://github.com/farkmeil/sidecut/releases/download/aligner-v1/labels.json",
        "sha256": "08e444212dde233080da83509eca0d2b4e53647ea631d261a87c854d62c45311",
    },
}
DEFAULT_MODEL = "wav2vec2-base-960h.onnx"


class AlignError(RuntimeError):
    pass


class Aligner(Protocol):
    model_name: str

    def align(self, audio: np.ndarray, words: list[str]) -> list[Word]:
        """One Word per input word, in order, timed within `audio` (16 kHz mono
        float32). A word that could not be placed has probability 0."""


_NOISE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|♪[^♪]*♪|♪|#")
_SPEAKER = re.compile(r"(?:^|(?<=[-–—]\s)|(?<=\s))[A-Z][A-Z0-9.'&]*(?: [A-Z0-9.'&]+){0,2}:\s")
_DASHES = re.compile(r"(?:^|\s)[-–—]+(?=\s|$)")
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def spoken_words(text: str) -> list[str]:
    """The words a subtitle line expects to be heard: no sound descriptions,
    speaker labels, music marks or punctuation. Word order is preserved."""
    text = _NOISE.sub(" ", text)
    text = _SPEAKER.sub(" ", text)
    text = _DASHES.sub(" ", text)
    return [m.group(0) for m in _WORD.finditer(text)]


def ctc_align(log_probs: np.ndarray, tokens: list[int],
              blank: int) -> tuple[list[tuple[int, int]], np.ndarray] | None:
    """Viterbi forced alignment of a CTC emission matrix [frames, labels] to a
    token sequence. Returns per token the half-open frame span in which it is
    emitted, plus the label the path emits at every frame, or None when the
    audio is too short to hold the sequence."""
    frames = log_probs.shape[0]
    length = len(tokens)
    if length == 0:
        return [], np.full(frames, blank, dtype=np.int64)
    states = 2 * length + 1
    labels = np.full(states, blank, dtype=np.int64)
    labels[1::2] = tokens
    skip_ok = np.zeros(states, dtype=bool)
    for s in range(3, states, 2):
        skip_ok[s] = tokens[(s - 1) // 2] != tokens[(s - 3) // 2]
    neg = np.float32(-1e30)
    score = np.full(states, neg, dtype=np.float32)
    score[0] = log_probs[0, blank]
    score[1] = log_probs[0, tokens[0]]
    back = np.zeros((frames, states), dtype=np.int8)
    for t in range(1, frames):
        prev1 = np.empty_like(score)
        prev1[0] = neg
        prev1[1:] = score[:-1]
        prev2 = np.full_like(score, neg)
        prev2[2:] = score[:-2]
        prev2[~skip_ok] = neg
        best = np.maximum(np.maximum(score, prev1), prev2)
        back[t] = np.where(best == score, 0, np.where(best == prev1, 1, 2))
        score = best + log_probs[t, labels]
    finals = [states - 1, states - 2]
    end = max(finals, key=lambda s: score[s])
    if score[end] <= neg / 2:
        return None
    path = np.empty(frames, dtype=np.int64)
    state = end
    for t in range(frames - 1, 0, -1):
        path[t] = state
        state -= int(back[t, state])
    path[0] = state
    spans: list[tuple[int, int]] = []
    for s in range(1, states, 2):
        where = np.flatnonzero(path == s)
        if len(where) == 0:
            return None  # cannot happen on a valid path, but never guess
        spans.append((int(where[0]), int(where[-1]) + 1))
    return spans, labels[path]


class OnnxAligner:
    """Runs the exported wav2vec2 CTC model with onnxruntime."""

    def __init__(self, model: str | Path, labels: str | Path | None = None, threads: int = 0):
        try:
            import onnxruntime as ort
        except ImportError as err:  # pragma: no cover
            raise AlignError("forced alignment needs onnxruntime: pip install 'sidecut[tagger]'") from err
        model = Path(model)
        labels = Path(labels) if labels else model.parent / "labels.json"
        meta = json.loads(Path(labels).read_text(encoding="utf-8"))
        self.model_name = model.stem
        self.labels: list[str] = meta["labels"]
        self.blank: int = meta["blank"]
        self.separator: int = self.labels.index(meta.get("word_separator", "|"))
        self.frame_ms: float = float(meta.get("frame_ms", 20))
        self.normalize: bool = bool(meta.get("normalize", True))
        self.vocab = {label: index for index, label in enumerate(self.labels) if len(label) == 1}
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def tokens(self, word: str) -> list[int]:
        return [self.vocab[c] for c in word.upper().replace("’", "'") if c in self.vocab]

    def emissions(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < SAMPLE_RATE:
            audio = np.concatenate([audio, np.zeros(SAMPLE_RATE - len(audio), dtype=np.float32)])
        if self.normalize:
            audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        return self.session.run(None, {self.input_name: audio[None, :]})[0][0]

    def align(self, audio: np.ndarray, words: list[str]) -> list[Word]:
        if not words:
            return []
        sequence, slots = token_sequence([self.tokens(w) for w in words], self.separator)
        if not sequence:
            return [Word(0, 0, w, 0.0) for w in words]
        log_probs = self.emissions(audio)
        aligned = ctc_align(log_probs, sequence, self.blank)
        if aligned is None:
            return [Word(0, 0, w, 0.0) for w in words]
        spans, path = aligned
        return place_words(words, slots, sequence, spans, path, log_probs, self.separator, self.frame_ms)


def token_sequence(per_word: list[list[int]], separator: int) -> tuple[list[int], list[tuple[int, int] | None]]:
    """Join per-word tokens with the separator. Returns the sequence and, per
    word, its token index range (None for a word with no alignable letters)."""
    sequence: list[int] = []
    slots: list[tuple[int, int] | None] = []
    for toks in per_word:
        if not toks:
            slots.append(None)
            continue
        if sequence:
            sequence.append(separator)
        slots.append((len(sequence), len(sequence) + len(toks)))
        sequence.extend(toks)
    return sequence, slots


def place_words(words: list[str], slots: list[tuple[int, int] | None], sequence: list[int],
                spans: list[tuple[int, int]], path: np.ndarray, log_probs: np.ndarray, separator: int,
                frame_ms: float, end_slack_ms: int = 150, max_word_ms: int = 1500) -> list[Word]:
    """Turn token spans into timed words.

    A word runs from its first letter's spike to the word-separator spike that
    follows it, which the model emits as the word ends; but no further than
    `end_slack_ms` past the last letter, since after a pause the separator can
    come late. The score is how closely the forced path followed the model's
    free choice over the word's frames (1 = exactly). A word stretched past
    `max_word_ms` is a sign the text does not match the audio there, so it is
    reported as not placed."""
    # how much the forced path had to give up against the model's free choice, per frame
    forced = log_probs[np.arange(len(path)), path]
    free = log_probs.max(axis=1)
    slack_frames = max(1, int(round(end_slack_ms / frame_ms)))
    out: list[Word] = []
    last_end_ms = 0
    for word, slot in zip(words, slots):
        if slot is None:
            out.append(Word(last_end_ms, last_end_ms, word, 0.0))
            continue
        first, last = slot
        start_frame = spans[first][0]
        end_frame = spans[last - 1][1]
        if last < len(sequence) and sequence[last] == separator:
            end_frame = min(max(end_frame, spans[last][0]), end_frame + slack_frames)
        score = float(np.exp(np.mean(forced[start_frame:end_frame] - free[start_frame:end_frame])))
        start_ms = int(round(start_frame * frame_ms))
        end_ms = int(round(end_frame * frame_ms))
        if end_ms - start_ms > max_word_ms:
            score = 0.0
        out.append(Word(start_ms, max(end_ms, start_ms + 1), word, score))
        last_end_ms = end_ms
    return out


def model_path(model_dir: str | Path | None, name: str = DEFAULT_MODEL) -> Path:
    base = Path(model_dir) if model_dir else Path.home() / ".cache" / "sidecut"
    return base / "aligner" / name


def ensure_model(model_dir: str | Path | None, name: str = DEFAULT_MODEL, log=lambda message: None) -> Path:
    """Download the alignment model files on first use. Returns the model path."""
    target = model_path(model_dir, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    for filename in (name, "labels.json"):
        path = target.parent / filename
        spec = MODEL_FILES.get(filename)
        if path.exists() or not spec:
            continue
        log(f"downloading alignment model {filename}")
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(spec["url"], timeout=120) as response, open(tmp, "wb") as out:
            digest = hashlib.sha256()
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
        if spec["sha256"] and digest.hexdigest() != spec["sha256"]:
            tmp.unlink()
            raise AlignError(f"{filename}: download did not match the expected checksum")
        tmp.replace(path)
    if not target.exists():
        raise AlignError(f"alignment model not found: {target}")
    return target
