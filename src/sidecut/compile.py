"""Turn a resolved plan into something a player or encoder understands.

Targets:
  edl          Kodi / MPlayer edit decision list (cut and mute), read natively by Kodi
  plexautoskip custom.json markers for PlexAutoSkip
  ffmpeg       argument list that renders a filtered copy of the file
  plan         plain JSON list of actions (what the sidecut enforcer uses)
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .profile import Action, plan_stats

EDL_CUT = 0
EDL_MUTE = 1


def to_edl(actions: Sequence[Action]) -> str:
    """Kodi reads `<video name>.edl` next to the video. Times are seconds."""
    lines = []
    for a in actions:
        code = EDL_CUT if a.kind == "skip" else EDL_MUTE
        lines.append(f"{a.start_ms / 1000:.3f}\t{a.end_ms / 1000:.3f}\t{code}")
    return "\n".join(lines) + ("\n" if lines else "")


def to_plexautoskip(actions: Sequence[Action], key: str) -> dict[str, Any]:
    """A custom.json fragment for PlexAutoSkip.

    `key` is a Plex ratingKey or a GUID such as imdb://tt0096895. Merge the
    "markers" object into your existing custom.json.
    """
    markers = [
        {"start": a.start_ms, "end": a.end_ms, "mode": "skip" if a.kind == "skip" else "volume"}
        for a in actions
    ]
    return {"markers": {str(key): markers}}


def to_plan(actions: Sequence[Action]) -> dict[str, Any]:
    return {"actions": [a.to_dict() for a in actions], "stats": plan_stats(list(actions))}


def _between(actions: Sequence[Action]) -> str:
    return "+".join(f"between(t,{a.start_ms / 1000:.3f},{a.end_ms / 1000:.3f})" for a in actions)


def to_ffmpeg(actions: Sequence[Action], src: str, dst: str, audio_stream: int = 0,
              video_codec: Sequence[str] = ("libx264", "-crf", "18", "-preset", "medium"),
              audio_codec: Sequence[str] = ("aac", "-b:a", "256k")) -> list[str]:
    """Arguments for an ffmpeg run that writes a filtered copy of `src`.

    With mutes only, video is copied untouched and only the chosen audio
    stream is re-encoded. Skips need frame-accurate cuts, so video is
    re-encoded too. Subtitles are dropped when there are skips because their
    timing would no longer line up.
    """
    mutes = [a for a in actions if a.kind == "mute"]
    skips = [a for a in actions if a.kind == "skip"]
    args = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", src]
    audio_filters = []
    if mutes:
        audio_filters.append(f"volume=enable='{_between(mutes)}':volume=0")
    if not skips:
        args += ["-map", "0:v:0?", "-map", f"0:a:{audio_stream}", "-map", "0:s?", "-c:v", "copy", "-c:s", "copy"]
        if audio_filters:
            args += ["-af", ",".join(audio_filters), "-c:a", *audio_codec]
        else:
            args += ["-c:a", "copy"]
        return args + [dst]
    keep = f"not({_between(skips)})"
    audio_filters.append(f"aselect='{keep}',asetpts=N/SR/TB")
    args += [
        "-map", "0:v:0", "-map", f"0:a:{audio_stream}",
        "-vf", f"select='{keep}',setpts=N/FRAME_RATE/TB",
        "-af", ",".join(audio_filters),
        "-c:v", *video_codec, "-c:a", *audio_codec, "-sn",
    ]
    return args + [dst]


def render(actions: Sequence[Action], fmt: str, **kwargs: Any) -> str:
    """Render a plan in one of the text formats."""
    if fmt == "edl":
        return to_edl(actions)
    if fmt == "plexautoskip":
        key = kwargs.get("key")
        if not key:
            raise ValueError("plexautoskip output needs a key (Plex ratingKey or GUID like imdb://tt0096895)")
        return json.dumps(to_plexautoskip(actions, key), indent=2) + "\n"
    if fmt == "plan":
        return json.dumps(to_plan(actions), indent=2) + "\n"
    if fmt == "ffmpeg":
        import shlex

        src, dst = kwargs.get("src"), kwargs.get("dst")
        if not src or not dst:
            raise ValueError("ffmpeg output needs the source video and an output path")
        return shlex.join(to_ffmpeg(actions, src, dst, audio_stream=kwargs.get("audio_stream", 0))) + "\n"
    raise ValueError(f"unknown format '{fmt}'")
