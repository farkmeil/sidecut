"""The live enforcer: watches Plex sessions and applies each viewer's plan.

With `enforce: false` (the default) it only observes: it works out exactly
what it would do and logs it, and sends nothing to any player except
read-only position polls.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from . import model
from .config import Config, FilterStore
from .engine import PLAYING, STOPPED, Cmd, SessionController, convert_mutes
from .model import format_time
from .plexio import AlertListener, PlayerLink, PlexError, PlexServer, SessionInfo
from .profile import plan_stats, resolve

log = logging.getLogger("sidecut.enforcer")


@dataclass
class Live:
    info: SessionInfo
    controller: SessionController | None
    link: PlayerLink | None
    profile: str = ""
    why_idle: str = ""
    busy: threading.Lock = field(default_factory=threading.Lock)
    observed: set[int] = field(default_factory=set)
    poll_failures: int = 0
    volume_ok_saved: bool | None = None


class Enforcer:
    def __init__(self, config: Config, server: PlexServer, store: FilterStore | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 link_factory: Callable[[SessionInfo], PlayerLink] | None = None):
        self.config = config
        self.server = server
        self.store = store or FilterStore(config.filters_dir, config.sidecar, config.path_map)
        self.clock = clock
        self.link_factory = link_factory or self._make_link
        self.sessions: dict[str, Live] = {}
        self.lock = threading.RLock()
        self._refresh_wanted = threading.Event()
        self._pending: dict[str, tuple[int, str, float]] = {}
        self._last_refresh = float("-inf")
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="sidecut-player")
        self._halt = threading.Event()
        self.listener: AlertListener | None = None

    # ------------------------------------------------------------ session setup

    def _make_link(self, info: SessionInfo) -> PlayerLink:
        prefs = self.config.prefs_for_player(info.player_title, info.player_id)
        control = prefs.control
        if control == "auto" and not info.local:
            control = "proxy"  # a remote device cannot be reached directly
        return PlayerLink(self.server, info, port=prefs.port, control=control)

    def _build(self, info: SessionInfo, now: float) -> Live:
        profile = self.config.profile_for_user(info.user)
        if profile.is_off:
            return Live(info, None, None, profile.name, "user has no filtering profile")
        if info.kind not in ("movie", "episode", "clip"):
            return Live(info, None, None, profile.name, f"not video ({info.kind})")
        if not info.file:
            return Live(info, None, None, profile.name, "server did not report a file path")
        path = self.store.find(info.file)
        if not path:
            return Live(info, None, None, profile.name, "no filter for this file")
        try:
            flt = model.load(path)
        except (OSError, model.FilterError) as err:
            return Live(info, None, None, profile.name, f"filter unreadable: {err}")
        same, reason = flt.media.matches(duration_ms=info.duration_ms, size=info.size)
        if not same:
            return Live(info, None, None, profile.name, f"filter is for a different cut ({reason})")
        plan = resolve(flt, profile, info.duration_ms)
        prefs = self.config.prefs_for_player(info.player_title, info.player_id)
        known = self._load_state().get("players", {}).get(info.player_id, {}).get("volume_ok")
        mute_mode = prefs.mute
        if mute_mode == "volume" and known is False and self.config.enforce:
            mute_mode = prefs.mute_fallback  # learned earlier that this player ignores volume commands
        plan = convert_mutes(plan, mute_mode)
        if not plan:
            return Live(info, None, None, profile.name, "nothing to filter under this profile")
        controller = SessionController(
            key=info.key, plan=plan, duration_ms=info.duration_ms, tuning=self.config.tuning,
            mute_fallback=prefs.mute_fallback, default_volume=prefs.default_volume,
            probe_volume=self.config.enforce and known is None, verify_mute=self.config.enforce,
        )
        controller.on_report(info.view_offset, info.player_state or PLAYING, now, "alert")
        live = Live(info, controller, self.link_factory(info), profile.name)
        stats = plan_stats(plan)
        log.info("%s [%s]: %d skips (%s), %d mutes (%s)%s", info.label, profile.name, stats["skips"],
                 format_time(stats["skipped_ms"]), stats["mutes"], format_time(stats["muted_ms"]),
                 "" if self.config.enforce else " - observing only")
        return live

    def refresh(self, now: float | None = None) -> None:
        """Sync with the server's list of active sessions."""
        now = self.clock() if now is None else now
        try:
            infos = self.server.sessions()
        except PlexError as err:
            log.warning("could not list sessions: %s", err)
            return
        except Exception:  # noqa: BLE001 - an odd answer from the server must not stop the enforcer
            log.exception("could not read the session list")
            return
        self._last_refresh = now
        with self.lock:
            seen = set()
            for info in infos:
                seen.add(info.key)
                live = self.sessions.get(info.key)
                if live is None or live.info.rating_key != info.rating_key:
                    if live is not None:
                        self._close(live, "item changed")
                    try:
                        live = self._build(info, now)
                    except Exception as err:  # noqa: BLE001 - leave this session alone, keep serving the others
                        log.exception("%s: could not set up filtering", info.label)
                        live = Live(info, None, None, "", f"internal error: {type(err).__name__}")
                    self.sessions[info.key] = live
                    if live.controller is None:
                        log.info("%s: not filtering (%s)", info.label, live.why_idle)
                    pending = self._pending.pop(info.key, None)
                    if pending and live.controller:
                        live.controller.on_report(pending[0], pending[1], pending[2], "alert")
                elif live.controller is not None:
                    live.controller.on_report(info.view_offset, info.player_state or live.controller.state, now,
                                              "coarse")
            for key in [k for k in self.sessions if k not in seen]:
                self._close(self.sessions.pop(key), "session ended")
            self._pending = {k: v for k, v in self._pending.items() if now - v[2] < 30}

    # ------------------------------------------------------------------ inputs

    def on_playing(self, session_key: str, player_id: str, offset_ms: int, state: str) -> None:
        """A playback report pushed by the server."""
        now = self.clock()
        key = f"{session_key}:{player_id}"
        with self.lock:
            live = self.sessions.get(key)
            if live is None:
                self._pending[key] = (offset_ms, state, now)
                self._refresh_wanted.set()
                return
            if live.controller is not None:
                live.controller.on_report(offset_ms, state, now, "alert")
            if state == STOPPED:
                self._refresh_wanted.set()

    # ------------------------------------------------------------------ output

    def _execute(self, live: Live, cmd: Cmd) -> None:
        controller, link, name = live.controller, live.link, live.info.label
        assert controller is not None and link is not None
        if cmd.kind == "poll":
            started = self.clock()
            try:
                timeline = link.poll()
            except PlexError as err:
                live.poll_failures += 1
                controller.last_direct_at = self.clock()
                if live.poll_failures == 3:
                    controller.can_poll = False
                    log.warning("%s: player does not answer position polls (%s); timing will be coarser", name, err)
                return
            live.poll_failures = 0
            finished = self.clock()
            with self.lock:
                if timeline is None:
                    controller.last_direct_at = finished
                else:
                    controller.on_report(timeline.time_ms, timeline.state, (started + finished) / 2, "direct",
                                         timeline.volume)
            return

        position = controller.position(self.clock())
        if cmd.kind == "seek":
            assert cmd.value is not None
            if not self.config.enforce:
                log.info("%s: would skip %s -> %s (%s)", name, format_time(position), format_time(cmd.value),
                         cmd.reason)
                return
            try:
                link.seek(cmd.value)
            except PlexError as err:
                log.warning("%s: skip failed (%s)", name, err)
                with self.lock:
                    controller.on_seek_failed(self.clock())
                return
            with self.lock:
                controller.on_seek_sent(cmd.value, self.clock())
            log.info("%s: skipped %s -> %s (%s)", name, format_time(position), format_time(cmd.value), cmd.reason)
        elif cmd.kind == "volume":
            assert cmd.value is not None
            muting = cmd.value == 0
            if not self.config.enforce:
                log.info("%s: would %s at %s (%s)", name, "mute" if muting else "unmute", format_time(position),
                         cmd.reason)
                with self.lock:
                    controller.on_volume_sent(cmd, self.clock())
                return
            if muting or cmd.reason == "probe":
                self._remember_volume(live, controller.volume or controller.default_volume)
            try:
                link.set_volume(cmd.value)
            except PlexError as err:
                log.warning("%s: volume command failed (%s)", name, err)
                with self.lock:
                    controller.on_volume_failed(cmd, self.clock())
                return
            with self.lock:
                controller.on_volume_sent(cmd, self.clock())
            if not muting and cmd.reason != "probe":
                self._forget_volume(live)
            if cmd.reason in ("probe", "probe-restore"):
                log.debug("%s: volume check (%s -> %d)", name, cmd.reason, cmd.value)
            else:
                log.info("%s: %s at %s (%s)", name, "muted" if muting else "unmuted", format_time(position),
                         cmd.reason)

    def _service(self, live: Live) -> None:
        try:
            controller = live.controller
            if controller is None:
                return
            with self.lock:
                cmds = controller.step(self.clock())
                if not self.config.enforce:
                    cmds = self._once_per_visit(live, cmds)
            for cmd in cmds:
                self._execute(live, cmd)
            with self.lock:
                while controller.log:
                    log.warning("%s: %s", live.info.label, controller.log.pop(0))
                if self.config.enforce and controller.volume_ok is not None \
                        and live.volume_ok_saved != controller.volume_ok:
                    live.volume_ok_saved = controller.volume_ok
                    self._remember_capability(live, controller.volume_ok)
        except Exception:  # noqa: BLE001 - one bad session must not stop the rest
            log.exception("%s: unexpected error", live.info.label)
        finally:
            live.busy.release()

    def _once_per_visit(self, live: Live, cmds: list[Cmd]) -> list[Cmd]:
        """In observe mode nothing moves the player, so a skip would be
        proposed on every tick. Report each one once per visit."""
        controller = live.controller
        assert controller is not None
        position = controller.position(self.clock())
        lead = controller.tuning.skip_lead_ms
        live.observed = {i for i in live.observed
                         if controller.plan[i].start_ms - lead <= position < controller.plan[i].end_ms}
        out = []
        for cmd in cmds:
            if cmd.kind != "seek":
                out.append(cmd)
                continue
            index = controller._index_of_skip(position)
            controller.seek_attempts.clear()
            if index is not None and index not in live.observed:
                live.observed.add(index)
                out.append(cmd)
        return out

    def _close(self, live: Live, reason: str) -> None:
        controller = live.controller
        if controller is None:
            return
        for cmd in controller.release():
            if self.config.enforce and live.link is not None:
                try:
                    live.link.set_volume(cmd.value or controller.default_volume)
                    self._forget_volume(live)
                    log.info("%s: volume restored (%s)", live.info.label, reason)
                except PlexError as err:
                    log.warning("%s: could not restore volume (%s)", live.info.label, err)
            controller.on_volume_sent(cmd, self.clock())
        log.info("%s: done (%s)", live.info.label, reason)

    # ------------------------------------------------- state kept between runs

    def _load_state(self) -> dict:
        try:
            with open(self.config.state_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _save_state(self, state: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.config.state_file) or ".", exist_ok=True)
            tmp = self.config.state_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh)
            os.replace(tmp, self.config.state_file)
        except OSError as err:
            log.debug("could not write state file: %s", err)

    def _remember_volume(self, live: Live, volume: int) -> None:
        """Note the volume a player had before we touch it, so a crash cannot
        leave it silent."""
        assert live.link is not None
        state = self._load_state()
        state.setdefault("muted", {})[live.info.player_id] = {
            "title": live.info.player_title, "product": live.info.player_product,
            "address": live.info.player_address, "port": live.link.port, "volume": volume,
        }
        self._save_state(state)

    def _forget_volume(self, live: Live) -> None:
        state = self._load_state()
        if state.get("muted", {}).pop(live.info.player_id, None) is not None:
            self._save_state(state)

    def _remember_capability(self, live: Live, volume_ok: bool) -> None:
        state = self._load_state()
        state.setdefault("players", {})[live.info.player_id] = {
            "title": live.info.player_title, "product": live.info.player_product, "volume_ok": volume_ok,
        }
        self._save_state(state)
        log.info("%s: player %s remote volume", live.info.player_title or live.info.player_id,
                 "honours" if volume_ok else "does not honour")

    def restore_leftover_mutes(self) -> None:
        """If a previous run died while a player was muted, give it its volume back."""
        state = self._load_state()
        leftovers = state.get("muted", {})
        for player_id, entry in list(leftovers.items()):
            info = SessionInfo("", "", "", "", "", player_id, entry.get("title", ""), entry.get("product", ""),
                               entry.get("address", ""), "", True, 0, None, None, None)
            try:
                PlayerLink(self.server, info, port=entry.get("port")).set_volume(int(entry.get("volume", 100)))
                log.info("%s: restored volume left muted by a previous run", entry.get("title") or player_id)
            except PlexError as err:
                log.warning("%s: could not restore volume left muted by a previous run (%s)",
                            entry.get("title") or player_id, err)
        if leftovers:
            state["muted"] = {}
            self._save_state(state)

    # --------------------------------------------------------------- main loop

    def tick(self) -> None:
        now = self.clock()
        if self._refresh_wanted.is_set() or now - self._last_refresh >= self.config.session_poll_s:
            self._refresh_wanted.clear()
            self.refresh(now)
        with self.lock:
            active = [live for live in self.sessions.values() if live.controller is not None]
        for live in active:
            if live.busy.acquire(blocking=False):
                self._pool.submit(self._service, live)

    def run(self, tick_s: float = 0.05) -> None:  # pragma: no cover - loop around tick()
        identity = self.server.identity()
        log.info("connected to Plex %s; %s", identity.get("version", "?"),
                 "ENFORCING" if self.config.enforce else "observing only (enforce: false)")
        if self.config.enforce:
            self.restore_leftover_mutes()
        if self.config.use_websocket:
            self.listener = AlertListener(self.server, self.on_playing)
            self.listener.start()
        try:
            while not self._halt.is_set():
                try:
                    self.tick()
                except Exception:  # noqa: BLE001 - keep running; the next tick starts clean
                    log.exception("unexpected error; continuing")
                    self._halt.wait(1.0)
                self._halt.wait(tick_s)
        finally:
            self.shutdown()

    def stop(self) -> None:
        self._halt.set()

    def shutdown(self) -> None:
        if self.listener is not None:
            self.listener.stop()
        with self.lock:
            for live in self.sessions.values():
                self._close(live, "shutting down")
            self.sessions.clear()
        self._pool.shutdown(wait=False)
