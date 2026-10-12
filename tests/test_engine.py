from sidecut.engine import PLAYING, SessionController, Tuning, convert_mutes
from sidecut.profile import Action

from sim import SimPlayer, jump, pause, resume, run, stop


def controller(plan, **kwargs):
    return SessionController(key="1:p", plan=list(plan), duration_ms=kwargs.pop("duration_ms", 600_000), **kwargs)


SKIP = Action(20_000, 35_000, "skip", ("sex_nudity",))
MUTE = Action(8_000, 8_600, "mute", ("language:f-word",))


def test_skip_hides_the_whole_segment():
    player = SimPlayer(position=10_000)
    seen = run(controller([SKIP]), player, seconds=20)
    assert seen.shown_ms(20_000, 35_000) == 0
    assert len(player.seeks) == 1 and player.seeks[0][1] == 35_000
    # nothing before the segment was lost beyond the lead time
    assert seen.shown_ms(10_000, 19_600) == 9_600
    assert seen.shown_ms(35_000, 36_000) == 1_000


def test_mute_covers_the_word_and_restores_volume():
    player = SimPlayer(position=0)
    seen = run(controller([MUTE]), player, seconds=12)
    assert seen.heard_ms(8_000, 8_600) == 0
    assert player.volume == 100
    assert [v for _, v in player.volumes] == [0, 100]
    # muted for little more than the word itself
    silent = sum(1 for ms, audible in seen.seen if not audible)
    assert 600 <= silent <= 1_100


def test_restores_the_volume_the_player_had():
    player = SimPlayer(position=0, volume=65)
    run(controller([MUTE]), player, seconds=12)
    assert player.volume == 65


def test_position_is_tracked_between_sparse_reports():
    # reports every 10 s; the skip starts 7 s after the last one
    player = SimPlayer(position=3_000, report_every=10.0, answers_polls=False)
    seen = run(controller([SKIP], can_poll=False), player, seconds=25)
    assert seen.shown_ms(20_000, 35_000) == 0


def test_starting_inside_a_skip_jumps_out_at_once():
    player = SimPlayer(position=25_000)
    seen = run(controller([SKIP]), player, seconds=5)
    assert seen.shown_ms(20_000, 35_000) <= 400  # only the command latency
    assert player.seeks[0][1] == 35_000


def test_rewinding_into_a_skip_is_skipped_again_every_time():
    player = SimPlayer(position=15_000)
    script = {t: jump(22_000) for t in (10.0, 14.0, 18.0, 22.0, 26.0)}
    seen = run(controller([SKIP]), player, seconds=30, script=script)
    assert len(player.seeks) == 6
    # each rewind shows at most the command latency plus the report delay
    assert seen.shown_ms(20_000, 35_000) <= 5 * 450


def test_viewer_seeking_past_a_skip_needs_no_action():
    player = SimPlayer(position=5_000)
    run(controller([SKIP]), player, seconds=10, script={2.0: jump(40_000)})
    assert player.seeks == []


def test_pause_inside_a_mute_stays_muted_then_restores():
    player = SimPlayer(position=7_500)
    script = {0.7: pause, 5.0: resume}
    seen = run(controller([MUTE]), player, seconds=8, script=script)
    assert seen.heard_ms(8_000, 8_600) == 0
    assert player.volume == 100


def test_stopping_while_muted_gives_the_volume_back():
    player = SimPlayer(position=7_500)
    ctl = controller([Action(8_000, 60_000, "mute", ("language",))])
    run(ctl, player, seconds=3, script={1.5: stop})
    assert [v for _, v in player.volumes] == [0, 100]
    assert not ctl.muted


def test_release_when_session_vanishes():
    ctl = controller([Action(0, 60_000, "mute", ("language",))])
    ctl.on_report(1_000, PLAYING, 1.0)
    cmd = ctl.step(1.0)[0]
    assert cmd.value == 0
    ctl.on_volume_sent(cmd, 1.0)
    assert [c.value for c in ctl.release()] == [100]


def test_two_skips_close_together_are_both_taken():
    plan = [Action(10_000, 15_000, "skip", ("violence",)), Action(17_500, 22_000, "skip", ("violence",))]
    player = SimPlayer(position=5_000)
    seen = run(controller(plan), player, seconds=15)
    assert seen.shown_ms(10_000, 15_000) == 0 and seen.shown_ms(17_500, 22_000) == 0
    assert [target for _, target in player.seeks] == [15_000, 22_000]


def test_mute_right_after_a_skip():
    plan = [Action(10_000, 15_000, "skip", ("violence",)), Action(15_300, 15_900, "mute", ("language",))]
    player = SimPlayer(position=8_000)
    seen = run(controller(plan), player, seconds=8)
    assert seen.shown_ms(10_000, 15_000) == 0
    assert seen.heard_ms(15_300, 15_900) == 0
    assert player.volume == 100


def test_skip_near_the_end_stays_inside_the_file():
    plan = [Action(50_000, 60_000, "skip", ("sex_nudity",))]
    player = SimPlayer(position=45_000)
    run(controller(plan, duration_ms=60_000), player, seconds=6)
    assert player.seeks[0][1] == 58_500


def test_a_player_that_ignores_seeks_is_not_hammered():
    player = SimPlayer(position=19_000, accepts_seek=False)
    ctl = controller([SKIP])
    run(ctl, player, seconds=10)
    assert len(player.seeks) == 3  # max_seek_attempts, spaced out
    gaps = [b[0] - a[0] for a, b in zip(player.seeks, player.seeks[1:])]
    assert all(gap >= 0.9 for gap in gaps)
    assert any("gave up" in line for line in ctl.log)


def test_seek_that_silently_does_nothing_is_retried():
    class Deaf(SimPlayer):
        pass

    player = Deaf(position=19_000)
    ctl = controller([SKIP])
    # accept the command but never apply it
    player.seek_latency = 10_000
    run(ctl, player, seconds=14)
    assert 2 <= len(player.seeks) <= 3


def test_player_without_volume_control_falls_back_to_skipping():
    plan = [Action(8_000, 8_600, "mute", ("language",)), Action(30_000, 30_500, "mute", ("language",))]
    player = SimPlayer(position=5_000, accepts_volume=False)
    ctl = controller(plan)
    seen = run(ctl, player, seconds=30)
    assert ctl.mute_disabled
    assert any("skipping instead" in line for line in ctl.log)
    # the second word is handled by a skip
    assert seen.shown_ms(30_000, 30_500) == 0
    assert player.volume == 100


def test_player_that_says_ok_but_keeps_playing_sound_is_caught():
    plan = [Action(8_000, 9_500, "mute", ("language",)), Action(30_000, 30_500, "mute", ("language",))]
    player = SimPlayer(position=5_000, applies_volume=False)
    ctl = controller(plan)
    seen = run(ctl, player, seconds=30)
    assert ctl.mute_disabled and ctl.volume_ok is False
    assert any("kept playing sound" in line for line in ctl.log)
    assert seen.shown_ms(30_000, 30_500) == 0  # later words are skipped instead


def test_volume_probe_confirms_control_before_the_first_mute():
    plan = [Action(60_000, 60_600, "mute", ("language",))]
    player = SimPlayer(position=5_000, volume=80)
    ctl = controller(plan, probe_volume=True)
    seen = run(ctl, player, seconds=10)
    assert ctl.volume_ok is True
    assert [v for _, v in player.volumes] == [79, 80]  # one-point nudge, then back
    assert player.volume == 80
    # the nudge never silenced anything
    assert all(audible for _, audible in seen.seen)


def test_volume_probe_finds_a_player_that_ignores_volume_before_any_word_leaks():
    plan = [Action(60_000, 60_600, "mute", ("language",))]
    player = SimPlayer(position=45_000, applies_volume=False)
    ctl = controller(plan, probe_volume=True)
    seen = run(ctl, player, seconds=20)
    assert ctl.volume_ok is False and ctl.mute_disabled
    assert seen.shown_ms(60_000, 60_600) == 0  # handled as a skip, first time
    assert player.volume == 100


def test_volume_probe_waits_when_a_mute_is_imminent():
    plan = [Action(8_000, 8_600, "mute", ("language",)), Action(90_000, 90_500, "mute", ("language",))]
    player = SimPlayer(position=5_000)
    ctl = controller(plan, probe_volume=True)
    seen = run(ctl, player, seconds=8)
    assert seen.heard_ms(8_000, 8_600) == 0
    assert ctl.volume_ok is True  # the real mute was confirmed in the timeline
    assert [v for _, v in player.volumes] == [0, 100]


def test_probe_is_undone_if_the_session_ends_mid_probe():
    plan = [Action(60_000, 60_600, "mute", ("language",))]
    ctl = controller(plan, probe_volume=True)
    ctl.on_report(1_000, PLAYING, 1.0, "direct", volume=70)
    probe = next(c for c in ctl.step(1.1) if c.kind == "volume")
    assert probe.value == 69 and probe.reason == "probe"
    ctl.on_volume_sent(probe, 1.1)
    [restore] = ctl.release()
    assert restore.value == 70
    ctl.on_volume_sent(restore, 1.2)
    assert ctl.release() == []


def test_fallback_off_leaves_playback_alone():
    player = SimPlayer(position=5_000, accepts_volume=False)
    ctl = controller([MUTE, Action(30_000, 30_500, "mute", ("language",))], mute_fallback="off")
    run(ctl, player, seconds=30)
    assert player.seeks == [] and ctl.plan == []


def test_observe_mode_proposes_but_never_moves_the_player():
    player = SimPlayer(position=15_000)
    ctl = controller([SKIP, Action(40_000, 40_500, "mute", ("language",))], verify_mute=False)
    seen = run(ctl, player, seconds=30, observe=True)
    assert not ctl.mute_disabled
    assert seen.shown_ms(20_000, 35_000) == 15_000  # nothing was skipped
    assert player.volume == 100


def test_polls_only_when_something_is_coming():
    plan = [Action(300_000, 301_000, "skip", ("violence",))]
    player = SimPlayer(position=0)
    run(controller(plan), player, seconds=60)
    assert player.polls == 1  # the initial sync only
    player = SimPlayer(position=285_000)
    run(controller(plan), player, seconds=10)
    assert 5 <= player.polls <= 12


def test_stale_report_after_a_seek_is_ignored():
    ctl = controller([SKIP])
    ctl.on_report(19_800, PLAYING, 10.0)
    [cmd] = [c for c in ctl.step(10.0) if c.kind == "seek"]
    ctl.on_seek_sent(cmd.value, 10.0)
    assert ctl.on_report(20_100, PLAYING, 10.2) is False      # old position still in flight
    assert ctl.position(10.2) >= 35_000
    assert ctl.on_report(35_150, PLAYING, 10.6) is True       # player caught up
    assert not [c for c in ctl.step(10.7) if c.kind == "seek"]


def test_backward_report_after_a_seek_means_the_viewer_rewound():
    ctl = controller([SKIP])
    ctl.on_report(19_800, PLAYING, 10.0)
    ctl.on_seek_sent(35_000, 10.0)
    assert ctl.on_report(5_000, PLAYING, 11.0) is True
    assert ctl.position(11.0) == 5_000


def test_coarse_reports_do_not_disturb_a_good_estimate():
    ctl = controller([SKIP])
    ctl.on_report(1_000, PLAYING, 10.0)
    assert ctl.on_report(1_000, PLAYING, 14.0, "coarse") is False  # server list is just old
    assert ctl.position(14.0) == 5_000
    assert ctl.on_report(120_000, PLAYING, 15.0, "coarse") is True  # a real jump
    assert ctl.on_report(120_000, "paused", 16.0, "coarse") is True  # a state change


def test_convert_mutes_modes():
    plan = [Action(1_000, 1_500, "mute"), Action(1_600, 2_000, "mute"), Action(9_000, 12_000, "skip")]
    assert convert_mutes(plan, "volume") == plan
    assert [a.kind for a in convert_mutes(plan, "off")] == ["skip"]
    skips = convert_mutes(plan, "skip")
    assert all(a.kind == "skip" for a in skips)
    assert (skips[0].start_ms, skips[0].end_ms) == (1_000, 2_000)


def test_slow_player_leak_is_bounded_by_latency():
    # a sluggish player: 900 ms from command to picture change
    player = SimPlayer(position=10_000, seek_latency=0.9)
    seen = run(controller([SKIP]), player, seconds=20)
    leak = seen.shown_ms(20_000, 35_000)
    assert 400 <= leak <= 700  # latency minus the 350 ms lead
    # raising the lead closes the gap
    player = SimPlayer(position=10_000, seek_latency=0.9)
    seen = run(controller([SKIP], tuning=Tuning(skip_lead_ms=1000)), player, seconds=20)
    assert seen.shown_ms(20_000, 35_000) == 0
