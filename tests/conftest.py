import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import subprocess

import pytest

SRT = """1
00:00:02,000 --> 00:00:04,000
Hello there, how are you?

2
00:00:10,000 --> 00:00:12,500
<i>What the fuck is that?</i>

3
00:00:20,000 --> 00:00:22,000
Oh, shit. Damn it.

4
00:00:30,000 --> 00:00:33,000
Go to hell, you son of a bitch.

5
00:00:40,000 --> 00:00:42,000
Nothing to see here.
"""


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    """Small synthetic videos: one with an embedded English text subtitle
    track and stereo sound, one with 5.1 sound and no subtitles."""
    root = tmp_path_factory.mktemp("clips")
    movie_dir = root / "movies" / "Test Movie (2001)"
    movie_dir.mkdir(parents=True)
    srt = root / "subs.srt"
    srt.write_text(SRT)
    with_subs = movie_dir / "Test Movie (2001).mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=duration=45:size=160x90:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=45",
            "-i", str(srt), "-map", "0:v", "-map", "1:a", "-map", "2:s",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "10", "-c:a", "aac", "-ac", "2", "-c:s", "srt",
            "-metadata:s:a:0", "language=eng", "-metadata:s:s:0", "language=eng", str(with_subs))
    show_dir = root / "tv" / "Some Show" / "Season 02"
    show_dir.mkdir(parents=True)
    surround = show_dir / "Some Show - S02E05 - Pilot.mkv"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=duration=20:size=160x90:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=20",
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "ac3", "-ac", "6", "-metadata:s:a:0", "language=eng", str(surround))
    return {"root": root, "movie": with_subs, "episode": surround, "srt": srt}
