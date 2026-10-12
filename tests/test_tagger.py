"""The tagger with a scripted recogniser standing in for Whisper. What is
"spoken" in the fake soundtrack is defined here, so the tests can check that
the right audio is listened to and the right intervals come out."""

import pytest

from sidecut import media, model
from sidecut.tagger import TagOptions, guess_identity, merge_language, tag_language
from sidecut.wordlist import Word, Wordlist

WL = Wordlist.load()

# (start_ms, end_ms, text) of every word "spoken" in the test movie
SPOKEN = [
    (2100, 2500, "Hello"), (2500, 2900, "there"),
    (10300, 10600, "What"), (10600, 10750, "the"), (10750, 11200, "fuck"), (11200, 11400, "is"), (11400, 11800, "that"),
    (20200, 20400, "Oh"), (20400, 20900, "shit"), (21100, 21400, "Damn"), (21400, 21600, "it"),
    # the subtitle at 30 s says "hell" and "son of a bitch"; only the second is audible
    (31500, 31700, "you"), (31700, 31900, "son"), (31900, 32000, "of"), (32000, 32100, "a"), (32100, 32600, "bitch"),
    # said on screen but missing from the subtitles
    (36000, 36500, "bastard"),
]


class Window:
    """Stands in for decoded audio; remembers which part of the file it is."""

    def __init__(self, start_ms, length_ms):
        self.start_ms, self.length_ms = start_ms, length_ms

    def __len__(self):
        return self.length_ms


class Scripted:
    model_name = "scripted"

    def __init__(self, spoken=SPOKEN):
        self.spoken = spoken
        self.listened = []
        self.vad_used = []

    def words(self, audio, language="en", vad=False):
        self.listened.append((audio.start_ms, audio.start_ms + audio.length_ms))
        self.vad_used.append(vad)
        return [Word(s - audio.start_ms, e - audio.start_ms, " " + text, 0.9)
                for s, e, text in self.spoken if audio.start_ms <= s and e <= audio.start_ms + audio.length_ms]


@pytest.fixture
def fake_audio(monkeypatch):
    monkeypatch.setattr(media, "read_audio",
                        lambda video, start_ms, length_ms, *a, **k: Window(start_ms, length_ms))


def spans(flt):
    return [(s.start_ms, s.end_ms, s.tags[0], s.severity, s.source) for s in flt.segments]


def test_subtitles_choose_where_to_listen_and_speech_gives_the_timing(clips, fake_audio):
    asr = Scripted()
    flt, report = tag_language(clips["movie"], asr, WL)
    located = [span for span in spans(flt) if span[4] == "subtitle+asr"]
    assert located == [
        (10750, 11200, "f-word", 3, "subtitle+asr"),
        (20400, 20900, "s-word", 2, "subtitle+asr"),
        (21100, 21400, "damn", 1, "subtitle+asr"),
        (31700, 32600, "b-word", 2, "subtitle+asr"),
    ]
    # "hell" is in the subtitle at 30 s but was not heard: the whole line is tagged as well
    [line] = [span for span in spans(flt) if span[4] == "subtitle"]
    assert abs(line[0] - 30_000) <= 50 and abs(line[1] - 33_000) <= 50 and line[3] == 2
    assert all(s.category == "language" and s.action == "mute" for s in flt.segments)
    # only the three lines with listed words were listened to, with padding
    assert len(asr.listened) == 3
    assert sum(end - start for start, end in asr.listened) < 20_000
    assert report["method"] == "subtitle+asr" and report["whole_line_fallbacks"] == 1
    assert report["lines_with_hits"] == 3 and report["subtitle_lines"] == 5


def test_filter_records_which_file_it_was_made_for(clips, fake_audio):
    flt, _ = tag_language(clips["movie"], Scripted(), WL)
    assert flt.media.title == "Test Movie" and flt.media.year == 2001
    assert abs(flt.media.duration_ms - 45_000) < 300
    assert flt.media.size == clips["movie"].stat().st_size
    assert flt.media.oshash == media.oshash(clips["movie"])
    assert flt.generator["language_tagger"]["method"] == "subtitle+asr"
    assert model.Filter.from_dict(flt.to_dict()).to_dict() == flt.to_dict()


def test_no_subtitles_means_the_whole_soundtrack_is_transcribed(clips, fake_audio):
    spoken = [(3000, 3400, "shit"), (12000, 12500, "hello"), (15000, 15600, "asshole")]
    asr = Scripted(spoken)
    flt, report = tag_language(clips["episode"], asr, WL, TagOptions(chunk_ms=8_000, chunk_overlap_ms=1_000))
    assert report["method"] == "asr-full" and report["chunks"] == 3
    assert spans(flt) == [(3000, 3400, "s-word", 2, "asr"), (15000, 15600, "a-word", 2, "asr")]
    assert flt.media.season == 2 and flt.media.episode == 5 and flt.media.title == "Some Show"
    # chunks cover the file end to end
    assert asr.listened[0][0] == 0 and asr.listened[-1][1] >= 19_900


def test_full_mode_hears_what_subtitles_leave_out_and_overlaps_are_not_double_counted(clips, fake_audio):
    asr = Scripted()
    flt, report = tag_language(clips["movie"], asr, WL,
                               TagOptions(full=True, chunk_ms=10_000, chunk_overlap_ms=2_000))
    found = [(s.start_ms, s.tags[0]) for s in flt.segments]
    assert (36000, "insult") in found                      # not in the subtitles
    assert found.count((10750, "f-word")) == 1             # sits in an overlap between chunks
    assert all(s.source == "asr" for s in flt.segments)


def test_speech_detection_is_used_only_for_whole_soundtrack_mode(clips, fake_audio):
    asr = Scripted()
    tag_language(clips["movie"], asr, WL)
    assert asr.vad_used and not any(asr.vad_used)          # short windows: listen to everything
    asr = Scripted([])
    tag_language(clips["episode"], asr, WL)
    assert asr.vad_used and all(asr.vad_used)
    asr = Scripted([])
    tag_language(clips["episode"], asr, WL, TagOptions(vad=False))
    assert not any(asr.vad_used)


def test_no_full_refuses_videos_without_text_subtitles(clips, fake_audio):
    from sidecut.tagger import NoSubtitles

    with pytest.raises(NoSubtitles):
        tag_language(clips["episode"], Scripted([]), WL, TagOptions(allow_full=False))
    flt, _ = tag_language(clips["movie"], Scripted(), WL, TagOptions(allow_full=False))
    assert flt.segments


def test_very_short_words_are_widened(clips, fake_audio):
    spoken = [(3000, 3120, "hell"), (15000, 15600, "asshole")]
    flt, _ = tag_language(clips["episode"], Scripted(spoken), WL)
    assert [(s.start_ms, s.end_ms) for s in flt.segments] == [(2910, 3210), (15000, 15600)]
    flt, _ = tag_language(clips["episode"], Scripted(spoken), WL, TagOptions(min_word_ms=0))
    assert [(s.start_ms, s.end_ms) for s in flt.segments] == [(3000, 3120), (15000, 15600)]


def test_clean_file_gives_an_empty_filter(clips, fake_audio):
    flt, report = tag_language(clips["episode"], Scripted([]), WL)
    assert flt.segments == [] and report["segments"] == 0


def test_retagging_keeps_hand_made_segments(clips, fake_audio):
    first, _ = tag_language(clips["movie"], Scripted(), WL)
    first.segments.append(model.Segment(5_000, 9_000, "sex_nudity", 3, source="manual"))
    first.segments.append(model.Segment(40_000, 40_500, "language", 2, source="manual", tags=["s-word"]))
    first.media.ids = {"imdb": "tt1234567"}
    again, _ = tag_language(clips["movie"], Scripted(SPOKEN[:7]), WL)
    merged = merge_language(first, again)
    kinds = [(s.category, s.source) for s in merged.segments]
    assert ("sex_nudity", "manual") in kinds and ("language", "manual") in kinds
    assert sum(1 for s in merged.segments if s.source in ("subtitle+asr", "subtitle")) == len(again.segments)
    assert merged.media.ids == {"imdb": "tt1234567"}


@pytest.mark.parametrize("path, expected", [
    ("/media/movies/Heat (1995)/Heat.1995.2160p.mkv", {"title": "Heat", "year": 1995}),
    ("/media/movies/Batman & Robin (1997)/x.mkv", {"title": "Batman & Robin", "year": 1997}),
    ("/media/movies/Dune (2021) {tmdb-438631}/Dune.mkv", {"title": "Dune", "year": 2021}),
    ("/media/tv/The Office (US)/Season 03/The.Office.S03E07.mkv", {"title": "The Office (US)", "season": 3, "episode": 7}),
    ("/media/tv/Bluey (2018)/Bluey - s01e02 - Hospital.mkv", {"title": "Bluey", "year": 2018, "season": 1, "episode": 2}),
])
def test_guess_identity(path, expected):
    assert guess_identity(path) == expected
