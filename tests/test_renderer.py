"""The renderer's audio sync check: rendered audio must sit where the source
audio sits, or the file is rejected."""

import subprocess

import pytest

from sidecut import renderer

np = pytest.importorskip("numpy")


def _noise_clip(path, seconds=40, delay_ms=0):
    """A video with a non-repeating soundtrack (so cross-correlation has one
    clear peak), optionally with the audio delayed."""
    audio_filter = f"adelay={delay_ms}|{delay_ms}" if delay_ms else "anull"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x90:rate=10",
        "-f", "lavfi", "-i", f"anoisesrc=color=pink:seed=7:duration={seconds}:sample_rate=16000",
        "-af", audio_filter, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-ac", "2", str(path),
    ], check=True)


def test_sync_check_sees_a_delayed_soundtrack(tmp_path):
    source, same, late = tmp_path / "src.mkv", tmp_path / "same.mkv", tmp_path / "late.mkv"
    _noise_clip(source)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-c", "copy", str(same)],
                   check=True)
    _noise_clip(late, delay_ms=300)
    ok = renderer.check_sync(source, same, 40_000)
    assert ok is not None and ok.checked >= 3 and ok.worst_ms < 20
    off = renderer.check_sync(source, late, 40_000)
    assert off is not None and off.checked >= 3 and 250 < off.worst_ms < 350
    assert renderer.check_sync(source, same, 5_000) is None    # too short to check


def test_render_rejects_a_file_whose_audio_drifted(tmp_path, monkeypatch, clips):
    from sidecut import model
    from sidecut.config import FilterStore
    from sidecut.model import Filter, MediaRef, Segment
    from sidecut.profile import load_profiles

    video = clips["movie"]
    store = FilterStore(tmp_path / "filters", sidecar=False, path_map={})
    flt = Filter(media=MediaRef(title="Test Movie", duration_ms=45_000, size=video.stat().st_size),
                 segments=[Segment(10_000, 10_500, "language", severity=3, tags=["f-word"])])
    model.save(flt, store.path_for_video(video))
    monkeypatch.setattr(renderer, "check_sync", lambda *a, **k: renderer.SyncCheck([-3.0, 85.0, 2.0], 3))
    options = renderer.RenderOptions(profile=load_profiles(None)["family"], source_root=video.parent,
                                     dest_root=tmp_path / "out")
    messages = []
    report = renderer.render_library(store, options, log=messages.append)
    assert report.rendered == [] and len(report.failed) == 1
    assert "out of step" in report.failed[0][1] and "85 ms" in report.failed[0][1]
    assert not list((tmp_path / "out").rglob("*.mkv"))        # nothing half-done left behind
