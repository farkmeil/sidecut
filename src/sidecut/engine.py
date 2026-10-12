"""The enforcer's decision logic for one playback session.

Pure state machine: it is told what the player reported and what time it is,
and answers with commands (seek, set volume, poll). No network, no clock of
its own, so it can be tested exhaustively. `enforcer` supplies the I/O.

Players report their position only every few seconds, so position is
estimated from the last report plus elapsed time, and the player is polled
directly when an action is close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .profile import DEFAULT_MERGE_GAP_MS, Action, _merge

PLAYING, PAUSED, BUFFERING, STOPPED = "playing", "paused", "buffering", "stopped"
NEVER = float("-inf")


@dataclass
class Tuning:
    skip_lead_ms: int = 350        # send a seek this long before the segment starts
    mute_lead_ms: int = 200        # send a mute this long before the segment starts
    resync_window_ms: int = 12000  # poll the player directly when an action is this close
    resync_every_ms: int = 1000
    seek_settle_ms: int = 5000     # after a seek, distrust reports from inside the skipped range this long
    min_skip_ms: int = 400         # not worth seeking for less than this
    end_guard_ms: int = 1500       # never seek closer than this to the end of the file
    max_seek_attempts: int = 3
    stale_after_ms: int = 60000    # forget a session that has gone quiet
    coarse_jump_ms: int = 15000    # a coarse report this far from the estimate means the viewer seeked
    volume_check_ms: int = 600     # how long a player gets to show a volume change in its timeline
    volume_check_timeout_ms: int = 3000


@dataclass(frozen=True)
class Cmd:
    kind: str  # "seek" | "volume" | "poll"
    value: int | None = None
    reason: str = ""


def convert_mutes(actions: Sequence[Action], mode: str) -> list[Action]:
    """How to treat mutes on a player: 'volume' keeps them, 'skip' turns them
    into skips (for players that cannot be muted remotely), 'off' drops them."""
    if mode == "volume":
        return list(actions)
    if mode == "off":
        return [a for a in actions if a.kind == "skip"]
    if mode != "skip":
        raise ValueError(f"unknown mute mode '{mode}'")
    merged = _merge([(a.start_ms, a.end_ms, a.reasons) for a in actions], DEFAULT_MERGE_GAP_MS["skip"])
    return [Action(s, e, "skip", r) for s, e, r in merged]


@dataclass
class SessionController:
    key: str
    plan: list[Action]
    duration_ms: int | None = None
    tuning: Tuning = field(default_factory=Tuning)
    can_poll: bool = True
    default_volume: int = 100
    mute_fallback: str = "skip"    # if the player will not take volume commands: "skip" | "off"
    probe_volume: bool = False     # check volume control with a 1-point nudge before it is needed
    verify_mute: bool = True       # confirm each mute in the player's own timeline

    state: str = BUFFERING
    offset_ms: int = 0
    offset_at: float = 0.0
    last_report_at: float = 0.0
    last_direct_at: float = NEVER
    volume: int | None = None      # last volume the player reported while we were not muting
    muted: bool = False
    mute_sent_at: float = NEVER
    saved_volume: int | None = None
    volume_ok: bool | None = None  # None: not known yet
    volume_failures: int = 0
    mute_disabled: bool = False
    seek_origin: int | None = None
    seek_target: int | None = None
    seek_until: float = 0.0
    seek_attempts: dict[int, int] = field(default_factory=dict)
    gave_up: set[int] = field(default_factory=set)
    hold_until: float = 0.0        # back off after a failed command
    log: list[str] = field(default_factory=list)
    _probe: tuple[int, int, float] | None = None   # (nudged value, original, sent at)
    _probe_restore: int | None = None

    def __post_init__(self) -> None:
        self.plan = sorted(self.plan, key=lambda a: (a.start_ms, a.end_ms))

    # ------------------------------------------------------------------ input

    def position(self, now: float) -> int:
        if self.state != PLAYING:
            return self.offset_ms
        estimate = self.offset_ms + int((now - self.offset_at) * 1000)
        if self.duration_ms:
            estimate = min(estimate, self.duration_ms)
        return estimate

    def on_report(self, offset_ms: int, state: str, now: float, source: str = "alert",
                  volume: int | None = None) -> bool:
        """Feed a position report. source: 'alert' (server push, timely),
        'direct' (asked the player just now), or 'coarse' (server session list,
        may be several seconds old). Returns False if the position was ignored."""
        self.last_report_at = now
        if source == "direct":
            self.last_direct_at = now
            if volume is not None:
                self._on_volume_seen(volume, now)
        if source == "coarse":
            same_state = state == self.state
            if same_state and (state != PLAYING or abs(offset_ms - self.position(now)) < self.tuning.coarse_jump_ms):
                return False
        if self.seek_target is not None:
            inside = self.seek_origin is not None and self.seek_origin <= offset_ms < self.seek_target - 250
            if inside and now < self.seek_until:
                return False  # the player has not caught up with our seek yet
            self.seek_origin = self.seek_target = None
        self.offset_ms = max(0, int(offset_ms))
        self.offset_at = now
        self.state = state
        # A report from beyond a skip proves the seek worked (or the viewer moved on):
        # forget earlier failures so the skip is enforced again if they come back to it.
        for index in [i for i in set(self.seek_attempts) | self.gave_up if self.plan[i].end_ms <= self.offset_ms]:
            self.seek_attempts.pop(index, None)
            self.gave_up.discard(index)
        return True

    def _on_volume_seen(self, volume: int, now: float) -> None:
        check = self.tuning.volume_check_ms / 1000
        if self._probe is not None:
            nudged, original, sent_at = self._probe
            if volume == nudged:
                self.volume_ok = True
                self._probe, self._probe_restore = None, original
            elif now - sent_at >= self.tuning.volume_check_timeout_ms / 1000:
                self._probe, self._probe_restore = None, original
                self._volume_unsupported("player accepts volume commands but does not apply them")
            return
        if self._probe_restore is not None:
            return
        if self.muted:
            if self.verify_mute and volume > 0 and now - self.mute_sent_at >= check:
                self.muted = False
                self._volume_unsupported("player accepted a mute but kept playing sound")
            elif volume == 0:
                self.volume_ok = True
            return
        self.volume = volume

    def on_seek_sent(self, target_ms: int, now: float) -> None:
        self.seek_origin = self.position(now)
        self.seek_target = target_ms
        self.seek_until = now + self.tuning.seek_settle_ms / 1000
        self.offset_ms = target_ms
        self.offset_at = now

    def on_seek_failed(self, now: float) -> None:
        self.hold_until = now + 1.0

    def on_volume_sent(self, cmd: Cmd, now: float) -> None:
        """A volume command was delivered (the player answered OK)."""
        self.volume_failures = 0
        if cmd.reason == "probe":
            assert cmd.value is not None and self.volume is not None
            self._probe = (cmd.value, self.volume, now)
        elif cmd.reason == "probe-restore":
            self._probe_restore = None
        elif cmd.value == 0:
            if not self.muted:
                self.saved_volume = self.volume if self.volume else self.default_volume
            self.muted = True
            self.mute_sent_at = now
        else:
            self.muted = False
            self.saved_volume = None
            self._probe = self._probe_restore = None

    def on_volume_failed(self, cmd: Cmd, now: float) -> None:
        """The player refused a volume command or could not be reached."""
        self.hold_until = now + 0.25
        self.volume_failures += 1
        if self.volume_failures < 2:
            return
        if cmd.reason == "probe-restore":
            self._probe_restore = None
        elif cmd.reason == "restore":
            self.muted = False  # nothing more we can do; do not retry forever
            self._probe = self._probe_restore = None
        else:
            self._probe = None
            self._volume_unsupported("player refuses volume commands")

    def _volume_unsupported(self, why: str) -> None:
        if self.mute_disabled:
            return
        self.mute_disabled = True
        self.volume_ok = False
        if self.mute_fallback == "skip":
            self.plan = sorted(convert_mutes(self.plan, "skip"), key=lambda a: (a.start_ms, a.end_ms))
            self.log.append(f"{why}; muting by skipping instead")
        else:
            self.plan = [a for a in self.plan if a.kind == "skip"]
            self.log.append(f"{why}; mutes are off for this session")
        self.seek_attempts.clear()
        self.gave_up.clear()

    def is_stale(self, now: float) -> bool:
        return (now - self.last_report_at) * 1000 > self.tuning.stale_after_ms

    # ----------------------------------------------------------------- output

    def _index_of_skip(self, pos: int) -> int | None:
        lead = self.tuning.skip_lead_ms
        for index, action in enumerate(self.plan):
            if action.kind != "skip" or index in self.gave_up:
                continue
            if action.start_ms - lead <= pos < action.end_ms:
                return index
            if action.start_ms - lead > pos:
                break
        return None

    def _in_mute(self, pos: int) -> Action | None:
        lead = self.tuning.mute_lead_ms
        for action in self.plan:
            if action.kind == "mute" and action.start_ms - lead <= pos < action.end_ms:
                return action
            if action.start_ms - lead > pos:
                break
        return None

    def _next_start(self, pos: int, kind: str | None = None) -> int | None:
        for action in self.plan:
            if action.start_ms > pos and (kind is None or action.kind == kind):
                return action.start_ms
        return None

    def _has_mutes(self) -> bool:
        return any(a.kind == "mute" for a in self.plan)

    def step(self, now: float) -> list[Cmd]:
        """What to send to the player right now. Call every few tens of ms."""
        if self.state == STOPPED:
            return self.release()
        if self.state != PLAYING or now < self.hold_until:
            return []
        pos = self.position(now)

        index = self._index_of_skip(pos)
        if index is not None:
            action = self.plan[index]
            target = action.end_ms
            if self.duration_ms:
                target = min(target, self.duration_ms - self.tuning.end_guard_ms)
            if target - pos >= self.tuning.min_skip_ms:
                attempts = self.seek_attempts.get(index, 0)
                if attempts >= self.tuning.max_seek_attempts:
                    self.gave_up.add(index)
                    self.log.append(f"gave up on skip at {action.start_ms} ms after {attempts} attempts")
                else:
                    self.seek_attempts[index] = attempts + 1
                    return [Cmd("seek", target, ",".join(action.reasons))]

        if self._probe is not None and (now - self._probe[2]) * 1000 > self.tuning.volume_check_timeout_ms + 2000:
            # no timeline came back to judge the nudge by: undo it and stay undecided
            self._probe, self._probe_restore = None, self._probe[1]

        cmds: list[Cmd] = []
        mute = self._in_mute(pos)
        if self._probe_restore is not None:
            cmds.append(Cmd("volume", self._probe_restore, "probe-restore"))
        elif mute and not self.muted and self._probe is None:
            cmds.append(Cmd("volume", 0, ",".join(mute.reasons)))
        elif not mute and self.muted:
            cmds.append(Cmd("volume", self.saved_volume or self.default_volume, "restore"))
        elif (self.probe_volume and self.volume_ok is None and self._probe is None and not self.muted
              and self.volume is not None and self.can_poll and self._has_mutes()):
            upcoming_mute = self._next_start(pos, "mute")
            if upcoming_mute is None or upcoming_mute - pos > 6000:
                cmds.append(Cmd("volume", self.volume - 1 if self.volume > 0 else 1, "probe"))

        if self.can_poll:
            since = (now - self.last_direct_at) * 1000
            upcoming = self._next_start(pos)
            near = upcoming is not None and upcoming - pos <= self.tuning.resync_window_ms
            settling = self.seek_target is not None and now < self.seek_until
            if self._probe is not None and since >= 250:
                cmds.append(Cmd("poll"))
            elif since >= self.tuning.resync_every_ms and (
                    near or self.muted or mute or settling or self.last_direct_at == NEVER):
                cmds.append(Cmd("poll"))
        return cmds

    def release(self) -> list[Cmd]:
        """Give the volume back. Call when a session ends for any reason."""
        if self._probe is not None:
            return [Cmd("volume", self._probe[1], "restore")]
        if self._probe_restore is not None:
            return [Cmd("volume", self._probe_restore, "restore")]
        if self.muted:
            return [Cmd("volume", self.saved_volume or self.default_volume, "restore")]
        return []
