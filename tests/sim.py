"""A simulated player and clock for testing the session controller.

The player's true position advances with virtual time. It reports its
position to the "server" every `report_every` seconds (like a real client),
answers direct polls exactly, and applies seek and volume commands after a
configurable latency, which is where real players leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sidecut.engine import PAUSED, PLAYING, STOPPED, SessionController


@dataclass
class SimPlayer:
    position: float = 0.0          # ms, true position
    state: str = PLAYING
    volume: int = 100
    report_every: float = 10.0     # seconds between timeline reports to the server
    seek_latency: float = 0.30     # seconds from command to picture change
    volume_latency: float = 0.08
    accepts_volume: bool = True    # answers OK to volume commands
    applies_volume: bool = True    # ...and actually changes the volume
    accepts_seek: bool = True
    answers_polls: bool = True
    pending: list = field(default_factory=list)  # (due_time, kind, value)
    seeks: list = field(default_factory=list)
    volumes: list = field(default_factory=list)
    polls: int = 0


@dataclass
class Exposure:
    """What the viewer actually saw and heard, in ms of media time."""
    seen: list = field(default_factory=list)     # (media_ms, audible)

    def shown_ms(self, start: int, end: int) -> int:
        return sum(1 for ms, _ in self.seen if start <= ms < end)

    def heard_ms(self, start: int, end: int) -> int:
        return sum(1 for ms, audible in self.seen if audible and start <= ms < end)


def run(controller: SessionController, player: SimPlayer, seconds: float, script=None, dt: float = 0.01,
        start_time: float = 100.0, observe: bool = False) -> Exposure:
    """Drive controller and player together. `script` maps a time offset in
    seconds to a callable(player, controller, now) for viewer actions."""
    exposure = Exposure()
    script = dict(script or {})
    now = start_time
    next_report = now
    steps = int(round(seconds / dt))
    for step in range(steps):
        now = start_time + step * dt
        elapsed = round(step * dt, 6)
        for at in sorted(k for k in script if k <= elapsed):
            script.pop(at)(player, controller, now)
            next_report = now  # a viewer action makes the client report at once
        # commands in flight
        for item in [p for p in player.pending if p[0] <= now]:
            player.pending.remove(item)
            _, kind, value = item
            if kind == "seek":
                player.position = float(value)
                next_report = now
            else:
                player.volume = value
        # the client's periodic report, relayed by the server
        if now >= next_report:
            controller.on_report(int(player.position), player.state, now, "alert")
            next_report = now + player.report_every
        for cmd in controller.step(now):
            if cmd.kind == "poll":
                player.polls += 1
                if player.answers_polls:
                    controller.on_report(int(player.position), player.state, now, "direct", player.volume)
                else:
                    controller.last_direct_at = now
            elif cmd.kind == "seek":
                player.seeks.append((round(elapsed, 3), cmd.value))
                if observe:
                    continue
                if player.accepts_seek:
                    player.pending.append((now + player.seek_latency, "seek", cmd.value))
                    controller.on_seek_sent(cmd.value, now)
                else:
                    controller.on_seek_failed(now)
            elif cmd.kind == "volume":
                player.volumes.append((round(elapsed, 3), cmd.value))
                if observe:
                    controller.on_volume_sent(cmd, now)
                    continue
                if player.accepts_volume:
                    if player.applies_volume:
                        player.pending.append((now + player.volume_latency, "volume", cmd.value))
                    controller.on_volume_sent(cmd, now)
                else:
                    controller.on_volume_failed(cmd, now)
        if player.state == PLAYING:
            for ms in range(int(player.position), int(player.position + dt * 1000)):
                exposure.seen.append((ms, player.volume > 0))
            player.position += dt * 1000
        if player.state == STOPPED:
            break
    return exposure


def pause(player, controller, now):
    player.state = PAUSED


def resume(player, controller, now):
    player.state = PLAYING


def stop(player, controller, now):
    player.state = STOPPED


def jump(to_ms):
    def action(player, controller, now):
        player.position = float(to_ms)
    return action
