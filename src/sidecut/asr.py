"""Speech recognition with word timestamps (faster-whisper), loaded lazily so
the rest of sidecut works without it installed."""

from __future__ import annotations

import os
from typing import Protocol

from .wordlist import Word


class Transcriber(Protocol):
    def words(self, audio, language: str = "en") -> list[Word]:
        """Recognise speech in mono 16 kHz float32 audio. Times are relative
        to the start of the audio."""


class WhisperTranscriber:
    def __init__(self, model: str = "small.en", device: str = "cpu", compute_type: str = "int8",
                 threads: int = 0, model_dir: str | None = None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as err:  # pragma: no cover - depends on install
            raise RuntimeError(
                "speech recognition needs faster-whisper: pip install 'sidecut[tagger]'"
            ) from err
        self.model_name = model
        self._model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            cpu_threads=threads or min(16, os.cpu_count() or 4),
            download_root=model_dir or os.environ.get("SIDECUT_MODEL_DIR") or None,
        )

    def words(self, audio, language: str = "en") -> list[Word]:
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        out: list[Word] = []
        for segment in segments:
            for word in segment.words or []:
                out.append(Word(int(round(word.start * 1000)), int(round(word.end * 1000)), word.word,
                                float(word.probability)))
        return out
