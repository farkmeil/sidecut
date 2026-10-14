"""Forced alignment: the Viterbi pass on synthetic emissions (no model needed)
and the subtitle text clean-up."""

import numpy as np
import pytest

from sidecut.align import ctc_align, spoken_words

BLANK = 0


def emissions(frames, spikes, labels=6, sure=0.9):
    """A CTC-like emission matrix: blank everywhere except a spike of `label`
    at the frames given in `spikes` ({frame: label})."""
    probs = np.full((frames, labels), (1 - sure) / (labels - 1), dtype=np.float32)
    for t in range(frames):
        probs[t, spikes.get(t, BLANK)] = sure
    return np.log(probs)


def test_tokens_land_on_their_spikes():
    # "c a t" as labels 3, 1, 5 spoken at frames 4, 7 and 11 in 20 frames
    log_probs = emissions(20, {4: 3, 7: 1, 11: 5})
    spans, path = ctc_align(log_probs, [3, 1, 5], BLANK)
    assert spans == [(4, 5), (7, 8), (11, 12)]
    assert list(path[:4]) == [BLANK] * 4 and path[4] == 3 and path[11] == 5


def test_repeated_letters_need_a_blank_between_them():
    # "l l" (label 2 twice) at frames 3 and 6; a path cannot merge them
    log_probs = emissions(12, {3: 2, 6: 2})
    spans, _ = ctc_align(log_probs, [2, 2], BLANK)
    assert spans == [(3, 4), (6, 7)]


def test_a_token_held_over_several_frames_is_one_span():
    log_probs = emissions(10, {2: 4, 3: 4, 4: 4})
    spans, _ = ctc_align(log_probs, [4], BLANK)
    assert spans == [(2, 5)]


def test_forcing_a_token_the_audio_lacks_still_gives_a_path_but_a_poor_one():
    log_probs = emissions(10, {2: 3, 7: 5})
    spans, path = ctc_align(log_probs, [3, 1, 5], BLANK)   # label 1 was never emitted
    assert spans[0] == (2, 3) and spans[2] == (7, 8)
    assert spans[1][0] >= 3 and spans[1][1] <= 7
    forced = log_probs[np.arange(10), path]
    free = log_probs.max(axis=1)
    assert forced.sum() < free.sum() - 1.0  # the forced path paid for the missing letter


def test_too_little_audio_for_the_text_is_reported():
    assert ctc_align(emissions(2, {}), [1, 1, 1], BLANK) is None
    assert ctc_align(emissions(5, {}), [], BLANK)[0] == []


@pytest.mark.parametrize("text, words", [
    ("Go to hell, you son of a bitch.", ["Go", "to", "hell", "you", "son", "of", "a", "bitch"]),
    ("- What?\n- Nothing.", ["What", "Nothing"]),
    ("[LAUGHS] Oh, shit.", ["Oh", "shit"]),
    ("TED: It's fine. (sighs)", ["It's", "fine"]),
    ("♪ la la la ♪ Damn it", ["Damn", "it"]),
    ("JAMIE TARTT: What the fuck?", ["What", "the", "fuck"]),
    ("I said: no.", ["I", "said", "no"]),
    ("Goddamn it!", ["Goddamn", "it"]),
    ("", []),
])
def test_spoken_words(text, words):
    assert spoken_words(text) == words


def test_model_download_checks_the_checksum(tmp_path, monkeypatch):
    import hashlib
    import io
    import urllib.request

    from sidecut import align

    served = {"m.onnx": b"model-bytes", "labels.json": b"{}"}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: Response(served[url.rsplit("/", 1)[1]]))
    monkeypatch.setattr(align, "MODEL_FILES", {
        "m.onnx": {"url": "https://x/m.onnx", "sha256": hashlib.sha256(b"model-bytes").hexdigest()},
        "labels.json": {"url": "https://x/labels.json", "sha256": ""},
    })
    path = align.ensure_model(tmp_path, "m.onnx")
    assert path == tmp_path / "aligner" / "m.onnx" and path.read_bytes() == b"model-bytes"
    assert (tmp_path / "aligner" / "labels.json").read_bytes() == b"{}"
    assert align.ensure_model(tmp_path, "m.onnx") == path          # second call: nothing to fetch

    served["m.onnx"] = b"tampered"
    with pytest.raises(align.AlignError, match="checksum"):
        align.ensure_model(tmp_path / "other", "m.onnx")
    assert not list((tmp_path / "other" / "aligner").glob("m.onnx*"))


def test_word_ends_follow_the_separator_but_not_across_a_pause():
    from sidecut.align import place_words, token_sequence

    sep, a, b = 4, 1, 2
    # "a b": letter a at frame 2, separator at frame 4, letter b at frame 30 (long pause before it)
    sequence, slots = token_sequence([[a], [b]], sep)
    log_probs = emissions(40, {2: a, 4: sep, 30: b})
    spans, path = ctc_align(log_probs, sequence, 0)
    first, second = place_words(["a", "b"], slots, sequence, spans, path, log_probs, sep, 20)
    assert (first.start_ms, first.end_ms) == (40, 80)           # runs to the separator spike
    assert (second.start_ms, second.end_ms) == (600, 620)
    assert first.probability > 0.9 and second.probability > 0.9
    # separator far after the letter: the word ends at most 150 ms after its last letter
    log_probs = emissions(40, {2: a, 20: sep, 30: b})
    spans, path = ctc_align(log_probs, sequence, 0)
    first, _ = place_words(["a", "b"], slots, sequence, spans, path, log_probs, sep, 20)
    assert first.end_ms == 60 + 140 or first.end_ms == 60 + 160  # last letter end (60) plus ~150 ms, in 20 ms frames
    # a word smeared over more than 1.5 s is not trusted
    sequence, slots = token_sequence([[a, b]], sep)
    log_probs = emissions(120, {2: a, 100: b})
    spans, path = ctc_align(log_probs, sequence, 0)
    [word] = place_words(["ab"], slots, sequence, spans, path, log_probs, sep, 20)
    assert word.end_ms - word.start_ms > 1500 and word.probability == 0.0
    # a word with no alignable letters is reported, unplaced, without breaking its neighbours
    sequence, slots = token_sequence([[a], [], [b]], sep)
    log_probs = emissions(40, {2: a, 4: sep, 30: b})
    spans, path = ctc_align(log_probs, sequence, 0)
    placed = place_words(["a", "123", "b"], slots, sequence, spans, path, log_probs, sep, 20)
    assert [w.probability > 0 for w in placed] == [True, False, True]
