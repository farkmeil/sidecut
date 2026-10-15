import json

import pytest

from sidecut import align, asr, cli, media, model

from test_tagger import Scripted, ScriptedAligner, Window

_real_read_audio = media.read_audio   # the env fixture swaps in a stub; the render test needs the real one


@pytest.fixture
def env(tmp_path, monkeypatch, clips):
    config = tmp_path / "sidecut.yaml"
    config.write_text(f"filters:\n  dir: {tmp_path}/filters\n  sidecar: false\n"
                      f"path_map:\n  /media: {clips['root']}\n"
                      "users:\n  Kids: family\n")
    monkeypatch.setenv("SIDECUT_CONFIG", str(config))
    monkeypatch.delenv("PLEX_TOKEN", raising=False)
    monkeypatch.setattr(asr, "WhisperTranscriber", lambda *a, **k: Scripted())
    monkeypatch.setattr(align, "ensure_model", lambda *a, **k: tmp_path / "fake.onnx")
    monkeypatch.setattr(align, "OnnxAligner", lambda path, threads=0: ScriptedAligner())
    monkeypatch.setattr(media, "read_audio", lambda video, start_ms, length_ms, *a, **k: Window(start_ms, length_ms))
    return tmp_path


def test_tag_show_compile_round_trip(env, clips, capsys):
    assert cli.main(["tag", str(clips["root"] / "movies")]) == 0
    out = capsys.readouterr().out
    assert "5 language segments" in out and "f-word 1" in out
    stored = env / "filters/media/movies/Test Movie (2001)/Test Movie (2001).sidecut.json"
    assert stored.exists()                       # central store mirrors the path Plex reports

    assert cli.main(["tag", str(clips["movie"])]) == 0
    assert "skip (already tagged)" in capsys.readouterr().out

    assert cli.main(["add", str(clips["movie"]), "--start", "0:00:38", "--end", "0:00:41.5",
                     "--category", "sex_nudity", "--severity", "3", "--tag", "nudity"]) == 0
    capsys.readouterr()
    flt = model.load(stored)
    assert [s.source for s in flt.segments if s.category == "sex_nudity"] == ["manual"]

    assert cli.main(["show", str(clips["movie"]), "--profile", "family", "--list"]) == 0
    out = capsys.readouterr().out
    assert "Test Movie (2001)" in out and "1 skips" in out and "skip 0:00:37.700 - 0:00:41.650" in out

    assert cli.main(["compile", str(stored), "--profile", "family", "--format", "edl"]) == 0
    edl = capsys.readouterr().out.strip().splitlines()
    assert edl[-1] == "37.700\t41.650\t0" and edl[0].endswith("\t1")

    assert cli.main(["compile", str(stored), "--profile", "language", "--format", "plexautoskip", "--key", "25549"]) == 0
    markers = json.loads(capsys.readouterr().out)["markers"]["25549"]
    assert all(m["mode"] == "volume" for m in markers)

    # re-tagging on request keeps the hand-made scene
    assert cli.main(["tag", "--force", str(clips["movie"])]) == 0
    capsys.readouterr()
    assert any(s.source == "manual" for s in model.load(stored).segments)


def test_tag_to_explicit_file_and_sidecar(env, clips, tmp_path, capsys):
    out = tmp_path / "one.sidecut.json"
    assert cli.main(["tag", str(clips["movie"]), "--out", str(out)]) == 0
    assert out.exists()
    assert cli.main(["tag", str(clips["root"]), "--out", str(out)]) == 2      # two videos, one output
    assert cli.main(["tag", str(tmp_path / "empty-dir")]) == 1
    capsys.readouterr()


def test_no_full_skips_and_reports_videos_without_subtitles(env, clips, capsys):
    assert cli.main(["tag", "--no-full", str(clips["root"])]) == 0
    out = capsys.readouterr().out
    assert "skipped: no usable subtitles" in out and "1 video(s) skipped" in out
    assert (env / "filters/media/movies/Test Movie (2001)/Test Movie (2001).sidecut.json").exists()
    assert not list((env / "filters").rglob("Some Show*"))


def test_validate_reports_each_file(env, tmp_path, capsys):
    good = tmp_path / "good.sidecut.json"
    good.write_text(json.dumps({"sidecut": "0.1", "segments": []}))
    bad = tmp_path / "bad.sidecut.json"
    bad.write_text(json.dumps({"sidecut": "0.1", "segments": [{"start_ms": 9, "end_ms": 3, "category": "language"}]}))
    assert cli.main(["validate", str(good)]) == 0
    assert cli.main(["validate", str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert "ok   " in out and "BAD  " in out and "end_ms must be after start_ms" in out


def test_alignment_is_used_unless_switched_off(env, clips, capsys, monkeypatch):
    stored = env / "filters/media/movies/Test Movie (2001)/Test Movie (2001).sidecut.json"
    assert cli.main(["tag", str(clips["movie"])]) == 0
    sources = {s.source for s in model.load(stored).segments}
    assert "subtitle+align" in sources and "subtitle+asr" not in sources
    assert cli.main(["tag", "--force", "--no-align", str(clips["movie"])]) == 0
    sources = {s.source for s in model.load(stored).segments}
    assert "subtitle+align" not in sources and "subtitle+asr" in sources
    # the model cannot be fetched: say so once and carry on with recognition
    def offline(*args, **kwargs):
        raise align.AlignError("download failed: no network")

    monkeypatch.setattr(align, "ensure_model", offline)
    assert cli.main(["tag", "--force", str(clips["movie"])]) == 0
    assert "forced alignment unavailable" in capsys.readouterr().err
    assert "subtitle+asr" in {s.source for s in model.load(stored).segments}


def test_helpful_errors(env, clips, capsys):
    assert cli.main(["show", str(clips["episode"])]) == 2
    assert "no filter found" in capsys.readouterr().err
    assert cli.main(["tag", str(clips["movie"])]) == 0
    capsys.readouterr()
    assert cli.main(["compile", str(clips["movie"]), "--profile", "nope", "--format", "edl"]) == 2
    assert "unknown profile 'nope'" in capsys.readouterr().err
    assert cli.main(["watch"]) == 2
    assert "no Plex token" in capsys.readouterr().err
    assert cli.main(["probe"]) == 2
    capsys.readouterr()


def test_profiles_listing(env, capsys):
    assert cli.main(["profiles"]) == 0
    out = capsys.readouterr().out
    assert "family: language mute" in out and "Kids -> family" in out


def test_probe_against_the_lab(env, monkeypatch, capsys):
    from fakes import FakePlayer, FakePlex

    plex, player = FakePlex(), FakePlayer(start_ms=65_000, volume=80)
    try:
        plex.add_session(player, "Kids", "/media/movies/Test Movie (2001)/Test Movie (2001).mkv")
        config = env / "sidecut.yaml"
        config.write_text(config.read_text() + f"plex:\n  url: {plex.url}\nplayers:\n  Test SHIELD: {{port: {player.port}}}\n")
        monkeypatch.setenv("PLEX_TOKEN", "tok")
        assert cli.main(["probe", "--test-volume"]) == 0
        out = capsys.readouterr().out
        assert "Kids on Test SHIELD: Test Movie" in out
        assert "direct position poll: ok, at 0:01:05" in out and "volume 80" in out
        assert "volume command: accepted" in out
        assert "proxy  position poll: FAILED" in out      # this player is not registered with the server
        assert player.volume == 80
        assert player.of("seekTo") == []                  # probing never moves playback unless asked
    finally:
        player.close()
        plex.close()


def test_render_writes_a_family_tree(env, clips, capsys, tmp_path):
    assert cli.main(["tag", str(clips["root"])]) == 0          # both test clips get language segments
    capsys.readouterr()
    out = tmp_path / "family"
    assert cli.main(["render", str(clips["root"]), "--to", str(out), "--profile", "family", "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "would render 2, linked 0" in text and "mutes" in text
    assert not out.exists()

    assert cli.main(["render", str(clips["root"]), "--to", str(out), "--profile", "family"]) == 0
    text = capsys.readouterr().out
    assert "rendered 2, linked 0" in text
    rendered = out / "movies/Test Movie (2001)/Test Movie (2001).mkv"
    episode = out / "tv/Some Show/Season 02/Some Show - S02E05 - Pilot.mkv"
    assert rendered.is_file() and not rendered.is_symlink() and episode.is_file()
    surround = [s for s in media.probe(episode)["streams"] if s["codec_type"] == "audio"][0]
    assert surround["codec_name"] == "eac3" and surround["channels"] == 6     # 5.1 stays 5.1
    info = media.probe(rendered)
    assert abs(media.duration_ms(info) - 45_000) < 300            # mutes only: nothing cut
    assert [s["codec_type"] for s in info["streams"]] == ["video", "audio", "subtitle"]
    # the muted stretch is silent, the rest is not
    import numpy as np
    loud = _real_read_audio(rendered, 5_000, 1_000)
    quiet = _real_read_audio(rendered, 10_700, 300)             # inside the f-word mute (10_620..11_350 padded)
    assert float(np.abs(loud).max()) > 0.05 and float(np.abs(quiet).max()) < 0.01

    # a second run does nothing; --force renders again
    assert cli.main(["render", str(clips["root"]), "--to", str(out), "--profile", "family"]) == 0
    assert "rendered 0, linked 0, up to date 2" in capsys.readouterr().out
    assert cli.main(["render", str(clips["root"]), "--to", str(out), "--profile", "family", "--force"]) == 0
    assert "rendered 2" in capsys.readouterr().out
    # one video given: only that one is rendered, into the folder given
    assert cli.main(["render", str(clips["movie"]), "--to", str(tmp_path / "one"), "--profile", "family"]) == 0
    assert "rendered 1, linked 0" in capsys.readouterr().out
    assert sorted(p.name for p in (tmp_path / "one").iterdir()) == ["Test Movie (2001).mkv"]
    # a profile that filters nothing links everything; untagged files stay out unless asked
    assert cli.main(["render", str(clips["root"]), "--to", str(tmp_path / "off"), "--profile", "off"]) == 0
    assert "rendered 0, linked 2" in capsys.readouterr().out
