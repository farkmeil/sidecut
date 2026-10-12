"""A fake Plex server and a fake Companion player, speaking real HTTP on
localhost, for exercising the enforcer end to end without any Plex install."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class FakePlayer:
    """Plays one item in real time and obeys Companion commands."""

    def __init__(self, start_ms: int = 0, volume: int = 100, honours_volume: bool = True,
                 accepts_commands: bool = True, identifier: str = "player-1"):
        self.identifier = identifier
        self.volume = volume
        self.honours_volume = honours_volume
        self.accepts_commands = accepts_commands
        self.state = "playing"
        self._base_ms = float(start_ms)
        self._base_at = time.monotonic()
        self.commands: list[tuple[float, str, dict]] = []  # (media position when received, command, params)
        self.headers_seen: list[dict] = []
        self.lock = threading.Lock()
        player = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def do_GET(self):
                url = urlparse(self.path)
                params = {k: v[0] for k, v in parse_qs(url.query).items()}
                player.headers_seen.append(dict(self.headers))
                if url.path == "/player/timeline/poll":
                    body = (
                        '<?xml version="1.0"?><MediaContainer commandID="{c}" location="fullScreenVideo">'
                        '<Timeline type="music" state="stopped"/>'
                        '<Timeline type="video" state="{s}" time="{t}" duration="600000" volume="{v}" '
                        'ratingKey="42" machineIdentifier="server"/>'
                        '<Timeline type="photo" state="stopped"/></MediaContainer>'
                    ).format(c=params.get("commandID", "0"), s=player.state, t=int(player.position()), v=player.volume)
                    return self._send(200, body, "text/xml")
                if not player.accepts_commands:
                    return self._send(500, "nope")
                with player.lock:
                    player.commands.append((player.position(), url.path, params))
                    if url.path == "/player/playback/seekTo":
                        player.seek(int(params["offset"]))
                    elif url.path == "/player/playback/setParameters" and "volume" in params:
                        if player.honours_volume:
                            player.volume = int(params["volume"])
                    else:
                        return self._send(404, "unknown")
                return self._send(200, "OK")  # Android answers with a bare OK, not XML

            def _send(self, code, body, kind="text/plain"):
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = _Server(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def position(self) -> float:
        if self.state != "playing":
            return self._base_ms
        return self._base_ms + (time.monotonic() - self._base_at) * 1000

    def seek(self, to_ms: int) -> None:
        self._base_ms = float(to_ms)
        self._base_at = time.monotonic()

    def pause(self) -> None:
        self._base_ms = self.position()
        self.state = "paused"

    def of(self, command: str) -> list[tuple[float, dict]]:
        return [(pos, params) for pos, path, params in self.commands if path.endswith(command)]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class FakePlex:
    """Serves /identity, /status/sessions and /clients, and relays /player/*
    to the fake player like a real server does for registered clients."""

    def __init__(self, token: str = "tok"):
        self.token = token
        self.sessions: list[dict] = []
        self.registered: dict[str, FakePlayer] = {}
        self.requests: list[str] = []
        plex = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urlparse(self.path)
                plex.requests.append(url.path)
                if self.headers.get("X-Plex-Token") != plex.token:
                    return self._send(401, "unauthorized")
                if url.path == "/identity":
                    return self._json({"MediaContainer": {"version": "1.43.4-fake", "machineIdentifier": "server"}})
                if url.path == "/status/sessions":
                    items = [item() if callable(item) else item for item in plex.sessions]
                    return self._json({"MediaContainer": {"size": len(items), "Metadata": items}})
                if url.path == "/clients":
                    servers = [{"name": p.identifier, "machineIdentifier": p.identifier} for p in plex.registered.values()]
                    return self._json({"MediaContainer": {"size": len(servers), "Server": servers}})
                if url.path.startswith("/player/"):
                    target = plex.registered.get(self.headers.get("X-Plex-Target-Client-Identifier", ""))
                    if target is None:
                        return self._send(404, "Unable to find player with identifier")
                    import requests

                    upstream = requests.get(f"http://127.0.0.1:{target.port}{self.path}", timeout=2)
                    return self._send(upstream.status_code, upstream.text,
                                      upstream.headers.get("Content-Type", "text/plain"))
                return self._send(404, "not found")

            def _json(self, payload):
                return self._send(200, json.dumps(payload), "application/json")

            def _send(self, code, body, kind="text/plain"):
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = _Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def add_session(self, player: FakePlayer, user: str, file: str, title: str = "Test Movie",
                    duration_ms: int = 600_000, size: int = 123456789, local: bool = True,
                    session_key: str = "7", address: str = "127.0.0.1") -> None:
        def item():
            return {
                "sessionKey": session_key, "ratingKey": "42", "type": "movie", "title": title,
                "duration": duration_ms, "viewOffset": int(player.position()),
                "User": {"id": "1", "title": user},
                "Player": {"address": address, "machineIdentifier": player.identifier,
                           "product": "Plex for Android (TV)", "title": "Test SHIELD", "state": player.state,
                           "local": local},
                "Session": {"id": "abc", "location": "lan" if local else "wan"},
                "Media": [{"duration": duration_ms, "Part": [{"file": file, "size": size, "duration": duration_ms}]}],
            }

        self.sessions.append(item)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
