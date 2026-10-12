"""Configuration file (YAML) and the filter store."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine import Tuning
from .model import SUFFIX
from .profile import Profile, load_profiles


class ConfigError(ValueError):
    pass


@dataclass
class PlayerPrefs:
    port: int | None = None      # Companion port on the device; default depends on the app
    control: str = "auto"        # auto | direct | proxy
    mute: str = "volume"         # volume | skip | off
    mute_fallback: str = "skip"  # if volume commands fail: skip | off
    default_volume: int = 100


@dataclass
class Config:
    plex_url: str = "http://127.0.0.1:32400"
    plex_token: str = ""
    filters_dir: str = "/data/filters"
    sidecar: bool = True
    path_map: dict[str, str] = field(default_factory=dict)  # Plex path prefix -> local path prefix
    enforce: bool = False
    users: dict[str, str] = field(default_factory=dict)
    default_profile: str = "off"
    players: dict[str, PlayerPrefs] = field(default_factory=dict)
    profiles: dict[str, Profile] = field(default_factory=dict)
    tuning: Tuning = field(default_factory=Tuning)
    state_file: str = "/data/state.json"
    use_websocket: bool = True
    session_poll_s: float = 5.0

    def profile_for_user(self, user: str) -> Profile:
        lowered = {k.lower(): v for k, v in self.users.items()}
        name = lowered.get((user or "").lower(), lowered.get("*", self.default_profile))
        if name not in self.profiles:
            raise ConfigError(f"user '{user}' is mapped to unknown profile '{name}'")
        return self.profiles[name]

    def prefs_for_player(self, title: str, identifier: str) -> PlayerPrefs:
        return self.players.get(identifier) or self.players.get(title) or self.players.get("*") or PlayerPrefs()


def _parse_player(name: str, raw: Any) -> PlayerPrefs:
    if not isinstance(raw, dict):
        raise ConfigError(f"players.{name}: must be an object")
    unknown = set(raw) - {"port", "control", "mute", "mute_fallback", "default_volume"}
    if unknown:
        raise ConfigError(f"players.{name}: unknown key(s) {', '.join(sorted(unknown))}")
    prefs = PlayerPrefs(**raw)
    if prefs.control not in ("auto", "direct", "proxy"):
        raise ConfigError(f"players.{name}.control: use auto, direct or proxy")
    if prefs.mute not in ("volume", "skip", "off"):
        raise ConfigError(f"players.{name}.mute: use volume, skip or off")
    if prefs.mute_fallback not in ("skip", "off"):
        raise ConfigError(f"players.{name}.mute_fallback: use skip or off")
    return prefs


def load_config(path: str | os.PathLike | None, env: dict[str, str] | None = None) -> Config:
    import yaml

    env = os.environ if env is None else env
    raw: dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
    known = {"plex", "filters", "path_map", "enforce", "users", "default_profile", "players", "profiles", "tuning",
             "state_file", "use_websocket", "session_poll_s"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown config key(s): {', '.join(sorted(unknown))}")

    plex = raw.get("plex") or {}
    filters = raw.get("filters") or {}
    config = Config()
    config.plex_url = str(env.get("PLEX_URL") or plex.get("url") or config.plex_url).rstrip("/")
    token = env.get("PLEX_TOKEN") or plex.get("token") or ""
    token_file = env.get("PLEX_TOKEN_FILE") or plex.get("token_file")
    if not token and token_file and os.path.exists(token_file):
        token = Path(token_file).read_text(encoding="utf-8").strip()
    config.plex_token = str(token)
    config.filters_dir = str(env.get("SIDECUT_FILTERS") or filters.get("dir") or config.filters_dir)
    config.sidecar = bool(filters.get("sidecar", True))
    config.path_map = {str(k).rstrip("/") or "/": str(v).rstrip("/") or "/" for k, v in (raw.get("path_map") or {}).items()}
    config.enforce = bool(raw.get("enforce", False))
    if env.get("SIDECUT_ENFORCE", "").lower() in ("1", "true", "yes", "on"):
        config.enforce = True
    config.users = {str(k): str(v) if v not in (False, None) else "off" for k, v in (raw.get("users") or {}).items()}
    default_profile = raw.get("default_profile", "off")
    config.default_profile = "off" if default_profile in (False, None) else str(default_profile)
    config.players = {str(k): _parse_player(str(k), v) for k, v in (raw.get("players") or {}).items()}
    config.profiles = load_profiles(raw.get("profiles"))
    tuning_raw = raw.get("tuning") or {}
    unknown = set(tuning_raw) - set(Tuning.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"tuning: unknown key(s) {', '.join(sorted(unknown))}")
    config.tuning = Tuning(**tuning_raw)
    config.state_file = str(raw.get("state_file") or os.path.join(os.path.dirname(config.filters_dir.rstrip("/")) or ".", "state.json"))
    config.use_websocket = bool(raw.get("use_websocket", True))
    config.session_poll_s = float(raw.get("session_poll_s", 5.0))
    for user, name in list(config.users.items()) + [("default_profile", config.default_profile)]:
        if name not in config.profiles:
            raise ConfigError(f"'{user}' is mapped to unknown profile '{name}' "
                              f"(known: {', '.join(sorted(config.profiles))})")
    return config


class FilterStore:
    """Where filters live.

    A filter for /media/movies/Heat (1995)/Heat.mkv is looked up as
      1. /media/movies/Heat (1995)/Heat.sidecut.json      (sidecar, if enabled)
      2. <dir>/media/movies/Heat (1995)/Heat.sidecut.json (central store)
    The central store mirrors the path as Plex reports it, so the media
    folders themselves can stay read-only.
    """

    def __init__(self, directory: str | os.PathLike, sidecar: bool = True, path_map: dict[str, str] | None = None):
        self.directory = Path(directory)
        self.sidecar = sidecar
        self.path_map = dict(path_map or {})

    def to_local(self, plex_path: str) -> str:
        for prefix, local in sorted(self.path_map.items(), key=lambda item: -len(item[0])):
            if plex_path == prefix or plex_path.startswith(prefix.rstrip("/") + "/"):
                return local.rstrip("/") + plex_path[len(prefix.rstrip("/")):]
        return plex_path

    def to_plex(self, local_path: str) -> str:
        for prefix, local in sorted(self.path_map.items(), key=lambda item: -len(item[1])):
            if local_path == local or local_path.startswith(local.rstrip("/") + "/"):
                return prefix.rstrip("/") + local_path[len(local.rstrip("/")):]
        return local_path

    def store_path(self, plex_path: str) -> Path:
        relative = Path(plex_path.lstrip("/"))
        return self.directory / relative.parent / (relative.stem + SUFFIX)

    def sidecar_path(self, plex_path: str) -> Path:
        local = Path(self.to_local(plex_path))
        return local.parent / (local.stem + SUFFIX)

    def find(self, plex_path: str) -> Path | None:
        candidates = ([self.sidecar_path(plex_path)] if self.sidecar else []) + [self.store_path(plex_path)]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
        return None

    def path_for_video(self, local_video: str | os.PathLike) -> Path:
        """Where the tagger writes the filter for a video it can see locally."""
        return self.store_path(self.to_plex(str(Path(local_video).absolute())))
