"""Plex I/O: session list, push notifications, and remote control of players.

Players are controlled with the Plex Companion protocol, the same HTTP
commands the Plex mobile app uses as a remote: /player/playback/seekTo,
/player/playback/setParameters (volume) and /player/timeline/poll. A command
goes straight to the device on the LAN, or through the server when the device
is registered with it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable
from xml.etree import ElementTree

import requests

from . import __version__

log = logging.getLogger("sidecut.plex")

# Companion ports by app. Android and Android TV share one.
CLIENT_PORTS = {
    "Plex for Android (TV)": 32500,
    "Plex for Android (Mobile)": 32500,
    "Plex for iOS": 32500,
    "Plex for Apple TV": 32500,
    "Plex for Roku": 8324,
    "Plex for Windows": 32700,
    "Plex for Mac": 32700,
}
DEFAULT_CLIENT_PORT = 32500
CONTROLLER_ID = f"sidecut-{uuid.uuid5(uuid.NAMESPACE_DNS, 'sidecut.enforcer').hex[:16]}"


class PlexError(RuntimeError):
    pass


@dataclass
class SessionInfo:
    session_key: str
    rating_key: str
    title: str
    kind: str
    user: str
    player_id: str
    player_title: str
    player_product: str
    player_address: str
    player_state: str
    local: bool
    view_offset: int
    duration_ms: int | None
    file: str | None
    size: int | None

    @property
    def key(self) -> str:
        return f"{self.session_key}:{self.player_id}"

    @property
    def label(self) -> str:
        return f"{self.user or '?'} on {self.player_title or self.player_product or '?'}: {self.title}"


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_session(item: dict[str, Any]) -> SessionInfo:
    player = item.get("Player") or {}
    user = item.get("User") or {}
    session = item.get("Session") or {}
    part: dict[str, Any] = {}
    medias = item.get("Media") or []
    chosen = next((m for m in medias if m.get("selected")), medias[0] if medias else {})
    parts = chosen.get("Part") or []
    if parts:
        part = parts[0]
    title = item.get("title", "")
    if item.get("type") == "episode" and item.get("grandparentTitle"):
        season, episode = item.get("parentIndex"), item.get("index")
        code = f" S{int(season):02d}E{int(episode):02d}" if season is not None and episode is not None else ""
        title = f"{item['grandparentTitle']}{code} {title}".strip()
    local = player.get("local")
    if local is None:
        local = session.get("location") == "lan"
    return SessionInfo(
        session_key=str(item.get("sessionKey", "")),
        rating_key=str(item.get("ratingKey", "")),
        title=title,
        kind=str(item.get("type", "")),
        user=str(user.get("title", "")),
        player_id=str(player.get("machineIdentifier", "")),
        player_title=str(player.get("title", "")),
        player_product=str(player.get("product", "")),
        player_address=str(player.get("address", "")),
        player_state=str(player.get("state", "")),
        local=str(local).lower() in ("1", "true"),
        view_offset=_int(item.get("viewOffset")) or 0,
        duration_ms=_int(part.get("duration")) or _int(chosen.get("duration")) or _int(item.get("duration")),
        file=part.get("file"),
        size=_int(part.get("size")),
    )


class PlexServer:
    def __init__(self, url: str, token: str, timeout: float = 6.0, http: requests.Session | None = None):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.http = http or requests.Session()

    def headers(self, **extra: str) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "X-Plex-Client-Identifier": CONTROLLER_ID,
            "X-Plex-Product": "sidecut",
            "X-Plex-Version": __version__,
            "X-Plex-Device-Name": "sidecut",
        }
        if self.token:
            headers["X-Plex-Token"] = self.token
        headers.update(extra)
        return headers

    def get(self, path: str, params: dict[str, Any] | None = None, timeout: float | None = None,
            headers: dict[str, str] | None = None) -> requests.Response:
        try:
            response = self.http.get(self.url + path, params=params, headers=self.headers(**(headers or {})),
                                     timeout=timeout or self.timeout)
        except requests.RequestException as err:
            raise PlexError(f"cannot reach Plex at {self.url}: {type(err).__name__}") from err
        if response.status_code == 401:
            raise PlexError("Plex rejected the token (401)")
        if response.status_code >= 400:
            raise PlexError(f"Plex answered {response.status_code} for {path}")
        return response

    def json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            container = self.get(path, params).json().get("MediaContainer", {})
        except (ValueError, AttributeError) as err:
            raise PlexError(f"Plex sent an unreadable answer for {path}") from err
        return container if isinstance(container, dict) else {}

    def identity(self) -> dict[str, Any]:
        return self.json("/identity")

    def sessions(self) -> list[SessionInfo]:
        out = []
        for item in self.json("/status/sessions").get("Metadata") or []:
            try:
                out.append(parse_session(item))
            except Exception:  # noqa: BLE001 - one unreadable entry must not hide the others
                log.warning("skipping a session entry that could not be read")
        return out

    def clients(self) -> list[dict[str, Any]]:
        return list(self.json("/clients").get("Server", []))

    def websocket_url(self) -> str:
        return self.url.replace("http", "ws", 1) + "/:/websockets/notifications"


@dataclass
class Timeline:
    time_ms: int
    state: str
    volume: int | None


class PlayerLink:
    """Remote control of one player. Raises PlexError when a command fails."""

    def __init__(self, server: PlexServer, info: SessionInfo, port: int | None = None, control: str = "auto",
                 timeout: float = 2.5):
        self.server = server
        self.info = info
        self.port = port or CLIENT_PORTS.get(info.player_product, DEFAULT_CLIENT_PORT)
        self.timeout = timeout
        self._command_id = 0
        if control == "auto":
            # try the device first, then ask the server to relay
            self.routes = ["direct", "proxy"] if info.player_address else ["proxy"]
        else:
            self.routes = [control]
        self.route: str | None = None

    def _send(self, route: str, command: str, params: dict[str, Any]) -> requests.Response:
        self._command_id += 1
        params = dict(params, commandID=self._command_id)
        headers = {
            "X-Plex-Client-Identifier": CONTROLLER_ID,
            "X-Plex-Device-Name": "sidecut",
            "X-Plex-Product": "sidecut",
            "X-Plex-Version": __version__,
            "X-Plex-Target-Client-Identifier": self.info.player_id,
        }
        if route == "direct":
            url = f"http://{self.info.player_address}:{self.port}/player/{command}"
        else:
            url = f"{self.server.url}/player/{command}"
            headers["X-Plex-Token"] = self.server.token
        try:
            response = self.server.http.get(url, params=params, headers=headers, timeout=self.timeout)
        except requests.RequestException as err:
            raise PlexError(f"{route}: {type(err).__name__}") from err
        if response.status_code >= 400:
            raise PlexError(f"{route}: HTTP {response.status_code}")
        return response

    def command(self, command: str, **params: Any) -> requests.Response:
        """Send on the route that last worked, else try each in turn."""
        order = [self.route] if self.route else list(self.routes)
        errors = []
        for route in order:
            try:
                response = self._send(route, command, params)
                if self.route != route:
                    log.info("%s: controlling via %s", self.info.player_title, route)
                    self.route = route
                return response
            except PlexError as err:
                errors.append(str(err))
        if self.route and len(self.routes) > 1:
            self.route = None  # let the next command try the other route again
        raise PlexError(f"{command} failed ({'; '.join(errors)})")

    def poll(self) -> Timeline | None:
        response = self.command("timeline/poll", wait=0)
        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as err:
            raise PlexError("timeline: unreadable answer") from err
        for element in root.iter("Timeline"):
            if element.get("type") == "video" and element.get("state") not in (None, "stopped"):
                return Timeline(time_ms=_int(element.get("time")) or 0, state=element.get("state", ""),
                                volume=_int(element.get("volume")))
        return None

    def seek(self, offset_ms: int) -> None:
        self.command("playback/seekTo", offset=int(offset_ms), type="video")

    def set_volume(self, level: int) -> None:
        self.command("playback/setParameters", volume=int(level), type="video")


class AlertListener(threading.Thread):
    """Server push channel. Calls `on_playing(session_key, player_id, offset_ms,
    state)` for every playback report the server relays. Reconnects forever."""

    def __init__(self, server: PlexServer, on_playing: Callable[[str, str, int, str], None]):
        super().__init__(daemon=True, name="sidecut-alerts")
        self.server = server
        self.on_playing = on_playing
        self.connected = threading.Event()
        self._halt = threading.Event()
        self._ws = None

    def handle(self, message: str | bytes) -> None:
        try:
            container = json.loads(message).get("NotificationContainer", {})
        except (ValueError, AttributeError):
            return
        if container.get("type") != "playing":
            return
        for note in container.get("PlaySessionStateNotification", []):
            offset = _int(note.get("viewOffset"))
            if offset is None:
                continue
            self.on_playing(str(note.get("sessionKey", "")), str(note.get("clientIdentifier", "")), offset,
                            str(note.get("state", "")))

    def run(self) -> None:  # pragma: no cover - needs a live server
        import websocket

        delay = 1.0
        while not self._halt.is_set():
            self._ws = websocket.WebSocketApp(
                self.server.websocket_url(),
                header=[f"X-Plex-Token: {self.server.token}"],
                on_open=lambda ws: (self.connected.set(), log.info("listening for playback reports")),
                on_message=lambda ws, message: self.handle(message),
                on_error=lambda ws, err: log.warning("notification channel error: %s", type(err).__name__),
            )
            started = time.monotonic()
            self._ws.run_forever(ping_interval=30, ping_timeout=10)
            self.connected.clear()
            if self._halt.is_set():
                break
            delay = 1.0 if time.monotonic() - started > 60 else min(delay * 2, 30.0)
            log.warning("notification channel closed; reconnecting in %.0fs", delay)
            self._halt.wait(delay)

    def stop(self) -> None:
        self._halt.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001 - shutting down
                pass
