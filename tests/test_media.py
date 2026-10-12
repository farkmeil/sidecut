import shutil
import subprocess

import pytest

from sidecut import media
from sidecut.media import parse_srt


def test_parse_srt_strips_markup_and_reads_times():
    cues = parse_srt("﻿1\n00:00:01,500 --> 00:00:03,000\n<i>Hello</i> {\\an8}there\nsecond line\n\n"
                     "2\n00:01:00.250 --> 00:01:02.000\n- Hi.\n\n3\n00:00:09,000 --> 00:00:08,000\nbackwards\n")
    assert [(c.start_ms, c.end_ms, c.text) for c in cues] == [
        (1500, 3000, "Hello there second line"), (60250, 62000, "- Hi.")]


def test_probe_duration_and_stream_choice(clips):
    info = media.probe(clips["movie"])
    assert abs(media.duration_ms(info) - 45_000) < 300
    index, stream = media.pick_audio_stream(info)
    assert index == 0 and stream["codec_name"] == "aac"
    index, stream = media.pick_text_subtitle(info)
    assert index == 0 and stream["codec_name"] in ("subrip", "srt")
    assert media.pick_text_subtitle(media.probe(clips["episode"])) is None


def test_stream_choice_prefers_language_and_avoids_commentary_and_forced():
    info = {"streams": [
        {"codec_type": "audio", "tags": {"language": "fre"}, "disposition": {"default": 1}},
        {"codec_type": "audio", "tags": {"language": "eng", "title": "Director's Commentary"}, "disposition": {}},
        {"codec_type": "audio", "tags": {"language": "eng"}, "disposition": {}},
        {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
        {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "eng", "title": "Forced"},
         "disposition": {"forced": 1}},
        {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "spa"}},
        {"codec_type": "subtitle", "codec_name": "ass", "tags": {"language": "eng", "title": "SDH"}},
    ]}
    assert media.pick_audio_stream(info)[0] == 2
    assert media.pick_text_subtitle(info)[0] == 3   # the SDH track, not PGS, not forced, not Spanish
    assert media.pick_audio_stream({"streams": []}) is None


def test_reads_embedded_subtitles(clips):
    cues, source = media.read_subtitles(clips["movie"])
    assert source.startswith("embedded:")
    assert len(cues) == 5
    assert cues[1].text == "What the fuck is that?"
    assert abs(cues[1].start_ms - 10_000) <= 50  # muxing shifts timestamps by a frame or so


def test_sidecar_subtitle_wins_and_is_matched_by_name(clips, tmp_path):
    video = tmp_path / "Film.mkv"
    shutil.copy(clips["episode"], video)
    assert media.read_subtitles(video) == ([], "")
    (tmp_path / "Film.es.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhola\n")
    (tmp_path / "Film.en.forced.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nforced\n")
    (tmp_path / "Other.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nother\n")
    assert media.find_sidecar_subtitle(video) is None
    (tmp_path / "Film.en.sdh.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nright one\n")
    cues, source = media.read_subtitles(video)
    assert source == "sidecar:Film.en.sdh.srt" and cues[0].text == "right one"


PGS_INFO = {"streams": [
    {"codec_type": "video"},
    {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "fre"}},
    {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng", "title": "Forced"},
     "disposition": {"forced": 1}},
    {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
    {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng", "title": "Commentary"}},
    {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng", "title": "SDH"}},
    {"codec_type": "subtitle", "codec_name": "dvd_subtitle", "tags": {"language": "eng"}},
]}


def test_image_subtitle_candidates():
    assert [index for index, _ in media.pick_image_subtitles(PGS_INFO)] == [2, 4]   # not French, forced, commentary, VobSub
    assert media.pick_image_subtitles({"streams": []}) == []


def test_image_subtitles_are_read_by_ocr_only_when_nothing_better_exists(clips, tmp_path, monkeypatch):
    calls = []

    def fake_ocr(video, candidates, language="eng", timeout=3600):
        calls.append([index for index, _ in candidates])
        return [media.Cue(1000, 2000, "read from the picture")], candidates[-1][0]

    video = tmp_path / "Disc.mkv"
    shutil.copy(clips["episode"], video)
    monkeypatch.setattr(media, "ocr_subtitles", fake_ocr)
    monkeypatch.setattr(media, "ocr_available", lambda: True)
    cues, source = media.read_subtitles(video, PGS_INFO)
    assert cues[0].text == "read from the picture" and source == "ocr:hdmv_pgs_subtitle#4" and calls == [[2, 4]]
    assert media.read_subtitles(video, PGS_INFO, ocr=False) == ([], "")            # switched off
    monkeypatch.setattr(media, "ocr_available", lambda: False)
    assert media.read_subtitles(video, PGS_INFO) == ([], "")                       # tools not installed
    monkeypatch.setattr(media, "ocr_available", lambda: True)
    (tmp_path / "Disc.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nreal text\n")
    assert media.read_subtitles(video, PGS_INFO)[1] == "sidecar:Disc.en.srt"       # text always wins
    assert len(calls) == 1


def test_ocr_picks_the_fullest_track_and_survives_empty_ones(tmp_path, monkeypatch):
    ran = []

    def fake_run(args, timeout=None):
        ran.append(args[0])
        if args[0] == "ffmpeg":
            outputs = [a for a in args if str(a).endswith(".sup")]
            sizes = fake_run.sizes
            for path, size in zip(outputs, sizes):
                open(path, "wb").write(b"x" * size)
        else:
            sup = args[-1]
            fake_run.ripped = sup
            open(sup[:-4] + ".srt", "w").write("1\n00:00:05,000 --> 00:00:06,000\nhello\n")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(media, "_run", fake_run)
    candidates = [(2, {"codec_name": "hdmv_pgs_subtitle"}), (4, {"codec_name": "hdmv_pgs_subtitle"})]
    fake_run.sizes = [10, 500]
    cues, used = media.ocr_subtitles(tmp_path / "x.mkv", candidates)
    assert used == 4 and cues[0].text == "hello" and "track4.en.sup" in fake_run.ripped
    assert ran == ["ffmpeg", "pgsrip"]                      # one pass over the file, one OCR run
    fake_run.sizes = [0, 0]
    assert media.ocr_subtitles(tmp_path / "x.mkv", candidates) == ([], None)


def test_read_audio_window(clips):
    np = pytest.importorskip("numpy")
    info = media.probe(clips["movie"])
    index, stream = media.pick_audio_stream(info)
    audio = media.read_audio(clips["movie"], 10_000, 3_000, index, stream)
    assert audio.dtype == np.float32
    assert abs(len(audio) - 48_000) < 1_000      # 3 s at 16 kHz
    assert 0.01 < float(np.abs(audio).max()) <= 1.0


def test_surround_uses_the_centre_channel(clips):
    info = media.probe(clips["episode"])
    index, stream = media.pick_audio_stream(info)
    assert media._dialogue_filter(stream) == ["-af", "pan=mono|c0=FC"]
    assert media._dialogue_filter({"channels": 2, "channel_layout": "stereo"}) == ["-ac", "1"]
    audio = media.read_audio(clips["episode"], 5_000, 2_000, index, stream)
    assert abs(len(audio) - 32_000) < 1_000


def test_oshash_is_stable_and_content_sensitive(clips, tmp_path):
    first = media.oshash(clips["movie"])
    assert len(first) == 16 and first == media.oshash(clips["movie"])
    assert first != media.oshash(clips["episode"])
    tiny = tmp_path / "tiny.mkv"
    tiny.write_bytes(b"x" * 100)
    with pytest.raises(media.MediaError):
        media.oshash(tiny)


def test_oshash_known_vector(tmp_path):
    # size + sum of little-endian uint64 words of first and last 64 KiB
    data = bytes(range(256)) * 1024  # 262144 bytes
    path = tmp_path / "v.bin"
    path.write_bytes(data)
    import struct
    expected = len(data)
    for chunk in (data[:65536], data[-65536:]):
        for (word,) in struct.iter_unpack("<Q", chunk):
            expected = (expected + word) & 0xFFFFFFFFFFFFFFFF
    assert media.oshash(path) == f"{expected:016x}"


def test_iter_videos(clips):
    found = [p.name for p in media.iter_videos(clips["root"])]
    assert found == ["Test Movie (2001).mkv", "Some Show - S02E05 - Pilot.mkv"]
    assert [p.name for p in media.iter_videos(clips["movie"])] == ["Test Movie (2001).mkv"]


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(media.MediaError, match="ffprobe failed"):
        media.probe(tmp_path / "nope.mkv")
