"""Measure how well the recognition step times words, against ground truth.

Speech is synthesised one word at a time with espeak-ng and assembled at known
offsets, so the true start and end of every target word is known exactly. The
audio then goes through the same recogniser and word-list matching the tagger
uses. Synthetic speech is not film dialogue, but it gives real numbers for
the timing error that mute padding has to absorb.

Needs: espeak-ng, ffmpeg, and sidecut installed with the tagger extra.
Usage: python tools/timing_check.py [model ...]        (default: small.en)
"""
import os
import random
import statistics
import subprocess
import sys
import tempfile

import numpy as np

from sidecut.asr import WhisperTranscriber
from sidecut.wordlist import Wordlist, find_in_words

SR = 16000
TARGETS = ["banana", "purple", "window", "dragon", "yellow", "rocket", "hell", "damn"]
FILLER = ("the people were walking down to see what happened over there and nobody could tell them why it "
          "was so late in the evening when they finally came back home from work that day").split()
WL = Wordlist.parse("\n".join(f"1 t {w}" for w in TARGETS))
PADS = [(0, 0), (150, 150), (250, 300)]
_cache = {}


def say(word):
    if word not in _cache:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        subprocess.run(["espeak-ng", "-v", "en-us", "-s", "165", "-w", path, word], check=True)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                             capture_output=True, check=True).stdout
        os.unlink(path)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
        loud = np.where(np.abs(audio) > 0.02)[0]
        _cache[word] = audio[loud[0]:loud[-1] + 1] if len(loud) else audio
    return _cache[word]


def sentence(rng):
    words = [rng.choice(FILLER) for _ in range(rng.randint(8, 14))]
    spots = sorted(rng.sample(range(2, len(words) - 1), 2))
    for spot in spots:
        words[spot] = rng.choice(TARGETS)
    parts = [np.zeros(int(SR * 0.4), dtype=np.float32)]
    position = len(parts[0])
    truth = []
    for index, word in enumerate(words):
        clip = say(word)
        if index in spots:
            truth.append((position * 1000 / SR, (position + len(clip)) * 1000 / SR))
        gap = np.zeros(int(SR * rng.uniform(0.04, 0.12)), dtype=np.float32)
        parts += [clip, gap]
        position += len(clip) + len(gap)
    parts.append(np.zeros(int(SR * 0.4), dtype=np.float32))
    return np.concatenate(parts), truth


def run(model_name, noise_db, sentences=40, seed=7):
    rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)
    transcriber = WhisperTranscriber(model_name, threads=8)
    start_errors, end_errors, missed, total, extra = [], [], 0, 0, 0
    for _ in range(sentences):
        audio, truth = sentence(rng)
        if noise_db is not None:
            noise = np.convolve(noise_rng.standard_normal(len(audio)).astype(np.float32), np.ones(8) / 8, mode="same")
            level = np.sqrt(np.mean(audio[np.abs(audio) > 0.02] ** 2))
            audio = audio + noise * (level / (np.sqrt(np.mean(noise ** 2)) * 10 ** (noise_db / 20)))
        found = find_in_words(WL, transcriber.words(audio.astype(np.float32)))
        used = set()
        for true_start, true_end in truth:
            total += 1
            near = [i for i, m in enumerate(found)
                    if i not in used and m.start_ms < true_end + 400 and m.end_ms > true_start - 400]
            if not near:
                missed += 1
                continue
            best = min(near, key=lambda i: abs(found[i].start_ms - true_start))
            used.add(best)
            start_errors.append(found[best].start_ms - true_start)
            end_errors.append(found[best].end_ms - true_end)
        extra += len(found) - len(used)

    def within(values, share):
        ordered = sorted(abs(v) for v in values)
        return ordered[min(len(ordered) - 1, int(len(ordered) * share))]

    covered = "/".join(
        f"{sum(1 for s, e in zip(start_errors, end_errors) if s - before <= 0 and e + after >= 0) / len(start_errors):.0%}"
        for before, after in PADS)
    print(f"{model_name:9s} noise {'none' if noise_db is None else str(noise_db) + ' dB':5s} | "
          f"{total - missed}/{total} words found, {extra} false | "
          f"start error median {statistics.median(start_errors):+.0f} ms (90% within {within(start_errors, .9):.0f}) | "
          f"end error median {statistics.median(end_errors):+.0f} ms (90% within {within(end_errors, .9):.0f}) | "
          f"whole word inside the mute with padding {' '.join(f'{b}+{a}' for b, a in PADS)}: {covered}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["small.en"]:
        for noise in (None, 10):
            run(name, noise)
