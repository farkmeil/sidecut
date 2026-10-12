import json

import pytest

from sidecut import asr, cli, media, model

from test_tagger import Scripted, Window


@pytest.fixture
def env(tmp_path, monkeypatch, clips):
    config = tmp_path / "sidecut.yaml"
    config.write_text(f"filters:\n  dir: {tmp_path}/filters\n  sidecar: false\n"
                      f"path_map:\n  /media: {clips['root']}\n"
                      "users:\n  Kids: family\n")
    monkeypatch.setenv("SIDECUT_CONFIG", str(config))
    monkeypatch.delenv("PLEX_TOKEN", raising=False)
    monkeypatch.setattr(asr, "WhisperTranscriber", lambda *a, **k: Scripted())
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


def test_validate_reports_each_file(env, tmp_path, capsys):
    good = tmp_path / "good.sidecut.json"
    good.write_text(json.dumps({"sidecut": "0.1", "segments": []}))
    bad = tmp_path / "bad.sidecut.json"
    bad.write_text(json.dumps({"sidecut": "0.1", "segments": [{"start_ms": 9, "end_ms": 3, "category": "language"}]}))
    assert cli.main(["validate", str(good)]) == 0
    assert cli.main(["validate", str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert "ok   " in out and "BAD  " in out and "end_ms must be after start_ms" in out


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
