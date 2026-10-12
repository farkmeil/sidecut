"""End-to-end: the enforcer against a fake Plex server and a fake player over
real HTTP in real time. These prove the wiring and the timing in the lab; they
do not prove anything about how a particular real player behaves."""

import json
import threading
import time

import pytest

from sidecut import model
from sidecut.config import Config, PlayerPrefs
from sidecut.enforcer import Enforcer
from sidecut.engine import Tuning
from sidecut.model import Filter, MediaRef, Segment
from sidecut.plexio import PlexServer
from sidecut.profile import load_profiles

from fakes import FakePlayer, FakePlex

MEDIA = "/media/movies/Test Movie (2001)/Test Movie.mkv"


@pytest.fixture
def lab(tmp_path):
    plex = FakePlex()
    players = []

    def make_player(**kwargs):
        player = FakePlayer(**kwargs)
        players.append(player)
        return player

    def make_config(enforce=True, users=None, **player_prefs):
        config = Config(
            plex_url=plex.url, plex_token="tok", filters_dir=str(tmp_path / "filters"), sidecar=False,
            enforce=enforce, users=users or {"Kids": "family"}, profiles=load_profiles(None),
            state_file=str(tmp_path / "state.json"), use_websocket=False, session_poll_s=0.5,
            tuning=Tuning(),
        )
        if players:
            config.players["Test SHIELD"] = PlayerPrefs(port=players[-1].port, **player_prefs)
        return config

    def write_filter(segments, size=123456789, duration_ms=600_000):
        flt = Filter(media=MediaRef(title="Test Movie", duration_ms=duration_ms, size=size), segments=segments)
        path = tmp_path / "filters" / "media/movies/Test Movie (2001)/Test Movie.sidecut.json"
        model.save(flt, path)
        return path

    yield type("Lab", (), {"plex": plex, "player": staticmethod(make_player), "config": staticmethod(make_config),
                           "filter": staticmethod(write_filter), "tmp": tmp_path})
    for player in players:
        player.close()
    plex.close()


def run_for(enforcer: Enforcer, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        enforcer.tick()
        time.sleep(0.02)


SEGMENTS = [
    Segment(1_500, 2_100, "language", severity=3, tags=["f-word"]),
    Segment(4_000, 9_000, "sex_nudity", severity=2),
]


def test_mutes_the_word_and_skips_the_scene(lab):
    player = lab.player()
    lab.filter(SEGMENTS)
    lab.plex.add_session(player, "Kids", MEDIA)
    enforcer = Enforcer(lab.config(), PlexServer(lab.plex.url, "tok"))
    run_for(enforcer, 5.5)
    enforcer.shutdown()

    volumes = player.of("setParameters")
    assert [int(p["volume"]) for _, p in volumes[:2]] == [0, 100]
    muted_at, unmuted_at = volumes[0][0], volumes[1][0]
    assert 900 <= muted_at <= 1_500, muted_at            # before the word (1500), not long before
    assert 2_100 <= unmuted_at <= 2_900, unmuted_at      # after the word (2100), not long after

    seeks = player.of("seekTo")
    assert len(seeks) == 1
    seek_at, params = seeks[0]
    assert int(params["offset"]) == 9_150                 # segment end plus the profile's padding
    assert 3_200 <= seek_at <= 4_000, seek_at             # before the scene starts
    assert params["type"] == "video" and "commandID" in params
    assert player.position() > 9_150
    assert player.volume == 100
    # commands went straight to the device and did not carry the server token
    assert all("X-Plex-Token" not in h for h in player.headers_seen)
    assert all(h.get("X-Plex-Target-Client-Identifier") == "player-1" for h in player.headers_seen)


def test_observe_mode_sends_nothing_but_position_polls(lab, caplog):
    player = lab.player()
    lab.filter(SEGMENTS)
    lab.plex.add_session(player, "Kids", MEDIA)
    enforcer = Enforcer(lab.config(enforce=False), PlexServer(lab.plex.url, "tok"))
    with caplog.at_level("INFO", logger="sidecut.enforcer"):
        run_for(enforcer, 4.6)
    enforcer.shutdown()
    assert player.commands == []
    assert player.volume == 100
    text = caplog.text
    assert "would mute" in text and "would unmute" in text
    assert text.count("would skip") == 1
    assert "observing only" in text


def test_other_users_are_left_alone(lab):
    player = lab.player()
    lab.filter(SEGMENTS)
    lab.plex.add_session(player, "farkmeil", MEDIA)
    enforcer = Enforcer(lab.config(), PlexServer(lab.plex.url, "tok"))
    run_for(enforcer, 2.5)
    enforcer.shutdown()
    assert player.commands == [] and player.headers_seen == []


def test_filter_for_a_different_cut_is_not_applied(lab, caplog):
    player = lab.player()
    lab.filter(SEGMENTS, size=999)  # the library file has another size
    lab.plex.add_session(player, "Kids", MEDIA)
    enforcer = Enforcer(lab.config(), PlexServer(lab.plex.url, "tok"))
    with caplog.at_level("INFO", logger="sidecut.enforcer"):
        run_for(enforcer, 2.5)
    enforcer.shutdown()
    assert player.commands == []
    assert "different cut" in caplog.text


def test_no_filter_means_no_action(lab, caplog):
    player = lab.player()
    lab.plex.add_session(player, "Kids", MEDIA)
    enforcer = Enforcer(lab.config(), PlexServer(lab.plex.url, "tok"))
    with caplog.at_level("INFO", logger="sidecut.enforcer"):
        run_for(enforcer, 1.0)
    enforcer.shutdown()
    assert player.commands == []
    assert "no filter for this file" in caplog.text


def test_commands_can_be_relayed_through_the_server(lab):
    player = lab.player()
    lab.plex.registered[player.identifier] = player
    lab.filter([Segment(2_000, 6_000, "sex_nudity", severity=2)])
    lab.plex.add_session(player, "Kids", MEDIA, address="203.0.113.9")  # device not reachable directly
    enforcer = Enforcer(lab.config(control="proxy"), PlexServer(lab.plex.url, "tok"))
    run_for(enforcer, 3.0)
    enforcer.shutdown()
    seeks = player.of("seekTo")
    assert len(seeks) == 1 and int(seeks[0][1]["offset"]) == 6_150
    assert seeks[0][0] < 2_000
    assert any(path.startswith("/player/") for path in lab.plex.requests)


def test_player_that_ignores_volume_gets_skips_and_is_remembered(lab):
    player = lab.player(honours_volume=False)
    lab.filter([Segment(1_500, 2_500, "language", severity=3, tags=["f-word"]),
                Segment(5_000, 5_600, "language", severity=3, tags=["f-word"])])
    lab.plex.add_session(player, "Kids", MEDIA)
    config = lab.config()
    enforcer = Enforcer(config, PlexServer(lab.plex.url, "tok"))
    run_for(enforcer, 5.5)
    enforcer.shutdown()
    seeks = player.of("seekTo")
    assert seeks and int(seeks[-1][1]["offset"]) == 5_900   # the second word was skipped instead
    assert seeks[-1][0] < 5_000
    state = json.loads((lab.tmp / "state.json").read_text())
    assert state["players"]["player-1"]["volume_ok"] is False

    # next session with this player: mutes are skips from the first word on
    player2 = lab.player(honours_volume=False)
    lab.plex.sessions.clear()
    lab.plex.add_session(player2, "Kids", MEDIA, session_key="8")
    config.players["Test SHIELD"] = PlayerPrefs(port=player2.port)
    enforcer = Enforcer(config, PlexServer(lab.plex.url, "tok"))
    run_for(enforcer, 2.0)
    enforcer.shutdown()
    assert player2.of("setParameters") == []
    first_at, first = player2.of("seekTo")[0]
    assert first_at < 1_500                       # before the first word this time
    assert int(first["offset"]) == 5_900          # the two words are close enough to be one skip


def test_volume_is_restored_when_playback_stops_mid_mute(lab):
    player = lab.player(volume=70)
    lab.filter([Segment(1_000, 30_000, "language", severity=3)])
    lab.plex.add_session(player, "Kids", MEDIA)
    enforcer = Enforcer(lab.config(), PlexServer(lab.plex.url, "tok"))
    run_for(enforcer, 2.0)
    assert player.volume == 0
    assert json.loads((lab.tmp / "state.json").read_text())["muted"]["player-1"]["volume"] == 70
    lab.plex.sessions.clear()  # viewer pressed stop
    run_for(enforcer, 1.2)
    enforcer.shutdown()
    assert player.volume == 70
    assert json.loads((lab.tmp / "state.json").read_text())["muted"] == {}


def test_volume_left_muted_by_a_crash_is_restored_on_start(lab):
    player = lab.player(volume=0)
    config = lab.config()
    (lab.tmp / "state.json").write_text(json.dumps({"muted": {"player-1": {
        "title": "Test SHIELD", "product": "Plex for Android (TV)", "address": "127.0.0.1",
        "port": player.port, "volume": 85}}}))
    enforcer = Enforcer(config, PlexServer(lab.plex.url, "tok"))
    enforcer.restore_leftover_mutes()
    assert player.volume == 85
    assert json.loads((lab.tmp / "state.json").read_text())["muted"] == {}


def test_unreachable_player_does_not_stall_or_spam(lab, caplog):
    player = lab.player(accepts_commands=False)
    lab.filter([Segment(1_000, 6_000, "sex_nudity", severity=2)])
    lab.plex.add_session(player, "Kids", MEDIA)
    enforcer = Enforcer(lab.config(), PlexServer(lab.plex.url, "tok"))
    with caplog.at_level("WARNING", logger="sidecut.enforcer"):
        run_for(enforcer, 4.5)
    enforcer.shutdown()
    assert caplog.text.count("skip failed") == 3
    assert "gave up on skip" in caplog.text


def test_push_report_for_unknown_session_triggers_discovery(lab):
    player = lab.player(start_ms=0)
    lab.filter([Segment(60_000, 61_000, "sex_nudity", severity=2)])
    config = lab.config()
    config.session_poll_s = 3600  # only push reports will reveal the session
    enforcer = Enforcer(config, PlexServer(lab.plex.url, "tok"))
    enforcer.tick()
    assert enforcer.sessions == {}
    lab.plex.add_session(player, "Kids", MEDIA)
    enforcer.on_playing("7", "player-1", 58_000, "playing")
    player.seek(58_000)
    run_for(enforcer, 2.6)
    enforcer.shutdown()
    assert len(player.of("seekTo")) == 1


def test_bad_token_is_reported_plainly(lab):
    from sidecut.plexio import PlexError

    with pytest.raises(PlexError, match="rejected the token"):
        PlexServer(lab.plex.url, "wrong").sessions()
