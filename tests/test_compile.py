import json
import shlex
import subprocess

import pytest

from sidecut import compile as compiler
from sidecut import media
from sidecut.profile import Action

PLAN = [Action(5_000, 6_000, "mute", ("language:f-word",)), Action(10_000, 20_000, "skip", ("sex_nudity",)),
        Action(30_250, 30_900, "mute", ("language:s-word",))]


def test_edl_is_what_kodi_reads():
    assert compiler.to_edl(PLAN) == "5.000\t6.000\t1\n10.000\t20.000\t0\n30.250\t30.900\t1\n"
    assert compiler.to_edl([]) == ""


def test_plexautoskip_markers():
    out = compiler.to_plexautoskip(PLAN, "imdb://tt0096895")
    assert out == {"markers": {"imdb://tt0096895": [
        {"start": 5_000, "end": 6_000, "mode": "volume"},
        {"start": 10_000, "end": 20_000, "mode": "skip"},
        {"start": 30_250, "end": 30_900, "mode": "volume"},
    ]}}
    assert json.loads(compiler.render(PLAN, "plexautoskip", key=25549))["markers"]["25549"]
    with pytest.raises(ValueError, match="needs a key"):
        compiler.render(PLAN, "plexautoskip")


def test_plan_json():
    out = json.loads(compiler.render(PLAN, "plan"))
    assert out["stats"] == {"skips": 1, "skipped_ms": 10_000, "mutes": 2, "muted_ms": 1_650}
    assert out["actions"][1] == {"start_ms": 10_000, "end_ms": 20_000, "action": "skip", "reasons": ["sex_nudity"]}


def test_ffmpeg_command_is_shell_safe():
    line = compiler.render(PLAN, "ffmpeg", src="/m/It's a film (2001).mkv", dst="/out/clean.mkv")
    args = shlex.split(line)
    assert args[0] == "ffmpeg" and args[-1] == "/out/clean.mkv" and "/m/It's a film (2001).mkv" in args
    with pytest.raises(ValueError):
        compiler.render(PLAN, "ffmpeg")
    with pytest.raises(ValueError, match="unknown format"):
        compiler.render(PLAN, "vhs")


def _loudness(path, start_s, length_s):
    """Mean volume in dB of a slice of the first audio stream."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-ss", str(start_s), "-t", str(length_s), "-i",
                          str(path), "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "mean_volume" in line:
            return float(line.split("mean_volume:")[1].split("dB")[0])
    raise AssertionError(out)


def test_mute_only_render_copies_video_and_silences_the_words(clips, tmp_path):
    plan = [a for a in PLAN if a.kind == "mute"]
    out = tmp_path / "muted.mkv"
    args = compiler.to_ffmpeg(plan, str(clips["movie"]), str(out))
    assert args[args.index("-c:v") + 1] == "copy"
    subprocess.run(args, check=True, capture_output=True)
    info = media.probe(out)
    assert abs(media.duration_ms(info) - 45_000) < 300          # nothing cut
    assert any(s["codec_type"] == "subtitle" for s in info["streams"])  # subtitles kept
    assert _loudness(out, 5.2, 0.6) < -70                        # inside the first mute
    assert _loudness(out, 30.4, 0.4) < -70                       # inside the second
    assert _loudness(out, 7.0, 1.0) > -40                        # normal sound elsewhere
    assert _loudness(out, 29.0, 1.0) > -40


def test_render_with_skips_cuts_the_scene_and_keeps_mutes_in_the_right_place(clips, tmp_path):
    out = tmp_path / "cut.mkv"
    args = compiler.to_ffmpeg(PLAN, str(clips["movie"]), str(out), video_codec=("libx264", "-preset", "ultrafast"))
    subprocess.run(args, check=True, capture_output=True)
    info = media.probe(out)
    assert abs(media.duration_ms(info) - 35_000) < 400           # 45 s minus the 10 s scene
    assert _loudness(out, 5.2, 0.6) < -70                        # mute before the cut: unchanged position
    assert _loudness(out, 20.45, 0.3) < -70                      # mute after the cut: 10 s earlier now
    assert _loudness(out, 30.4, 0.4) > -40                       # and not where it used to be


def test_nothing_to_do_is_a_straight_copy():
    args = compiler.to_ffmpeg([], "in.mkv", "out.mkv")
    assert args[args.index("-c:a") + 1] == "copy" and "-af" not in args
