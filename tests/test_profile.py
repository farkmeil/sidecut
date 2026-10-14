import pytest

from sidecut.model import Filter, MediaRef, Segment
from sidecut.profile import Action, ProfileError, load_profiles, parse_profile, plan_stats, resolve

PROFILES = load_profiles(None)


def flt(*segments, duration_ms=600_000):
    return Filter(media=MediaRef(duration_ms=duration_ms), segments=list(segments))


def spans(actions):
    return [(a.kind, a.start_ms, a.end_ms) for a in actions]


def test_family_preset_mutes_language_and_skips_sex_but_not_mild_violence():
    plan = resolve(flt(
        Segment(10_000, 10_500, "language", 1, tags=["hell"]),
        Segment(60_000, 90_000, "sex_nudity", 2),
        Segment(120_000, 125_000, "violence", 2),
        Segment(200_000, 210_000, "violence", 3),
        Segment(300_000, 305_000, "substances", 3),
    ), PROFILES["family"])
    assert spans(plan) == [
        ("mute", 9_880, 10_650),
        ("skip", 59_700, 90_150),
        ("skip", 199_700, 210_150),
    ]


def test_off_profile_does_nothing_and_language_profile_only_mutes():
    f = flt(Segment(10_000, 10_500, "language", 3), Segment(60_000, 90_000, "sex_nudity", 3))
    assert resolve(f, PROFILES["off"]) == []
    assert PROFILES["off"].is_off and not PROFILES["family"].is_off
    assert [a.kind for a in resolve(f, PROFILES["language"])] == ["mute"]


def test_nearby_mutes_merge_and_distant_ones_do_not():
    plan = resolve(flt(
        Segment(10_000, 10_300, "language", 3), Segment(10_700, 11_000, "language", 3),
        Segment(20_000, 20_300, "language", 3),
    ), PROFILES["language"])
    assert spans(plan) == [("mute", 9_880, 11_150), ("mute", 19_880, 20_450)]


def test_skips_close_together_become_one_so_the_player_seeks_once():
    plan = resolve(flt(Segment(10_000, 20_000, "sex_nudity", 2), Segment(21_500, 30_000, "sex_nudity", 2)),
                   PROFILES["family"])
    assert spans(plan) == [("skip", 9_700, 30_150)]


def test_a_skip_swallows_mutes_inside_it_and_trims_ones_that_straddle():
    plan = resolve(flt(
        Segment(50_000, 80_000, "sex_nudity", 2),
        Segment(60_000, 60_500, "language", 3),        # inside the skip: gone
        Segment(79_900, 81_000, "language", 3),        # straddles the end: trimmed
    ), PROFILES["family"])
    assert spans(plan) == [("skip", 49_700, 80_150), ("mute", 80_150, 81_150)]
    for a, b in zip(plan, plan[1:]):
        assert a.end_ms <= b.start_ms


def test_actions_never_overlap_and_stay_inside_the_file():
    plan = resolve(flt(Segment(0, 400, "language", 3), Segment(599_800, 600_000, "sex_nudity", 3),
                       duration_ms=600_000), PROFILES["family"])
    assert plan[0].start_ms == 0
    assert plan[-1].end_ms == 600_000


def test_tag_rules_override_category_rules():
    profiles = load_profiles({"relaxed": {"extends": "family", "tags": {"hell": "off", "damn": False}},
                              "devout": {"extends": "family", "tags": {"deity": "on"}}})
    f = flt(Segment(1_000, 1_400, "language", 1, tags=["hell"]), Segment(5_000, 5_400, "language", 3, tags=["f-word"]),
            Segment(9_000, 9_400, "language", 1, tags=["deity"]))
    assert [a.start_ms for a in resolve(f, profiles["relaxed"])] == [4_880]
    assert [a.start_ms for a in resolve(f, profiles["family"])] == [880, 4_880]      # deity is opt-in
    assert [a.start_ms for a in resolve(f, profiles["devout"])] == [880, 4_880, 8_880]


def test_profile_can_change_action_severity_and_padding():
    profiles = load_profiles({"mine": {
        "rules": {"language": {"min_severity": 2, "action": "skip"}, "violence": 3, "sex_nudity": "mute"},
        "pad_ms": {"skip": [1000, 500], "mute": 0},
    }})
    plan = resolve(flt(Segment(10_000, 10_400, "language", 1), Segment(20_000, 20_400, "language", 2),
                       Segment(40_000, 45_000, "sex_nudity", 1)), profiles["mine"])
    assert spans(plan) == [("skip", 19_000, 20_900), ("mute", 40_000, 45_000)]


def test_segment_action_is_only_a_suggestion():
    f = flt(Segment(10_000, 12_000, "language", 3, action="skip"))
    assert resolve(f, PROFILES["family"])[0].kind == "mute"        # profile says mute
    loose = parse_profile("loose", {"rules": {"language": {"min_severity": 1}}})
    assert resolve(f, loose)[0].kind == "skip"                      # profile has no opinion


def test_reasons_and_stats():
    plan = resolve(flt(Segment(10_000, 10_300, "language", 3, tags=["f-word"]),
                       Segment(10_400, 10_700, "language", 2, tags=["s-word"]),
                       Segment(60_000, 90_000, "sex_nudity", 2)), PROFILES["family"])
    assert plan[0].reasons == ("language:f-word", "language:s-word")
    assert plan_stats(plan) == {"skips": 1, "skipped_ms": 30_450, "mutes": 1, "muted_ms": 970}


@pytest.mark.parametrize("raw, message", [
    ({"rules": {"swearing": "on"}}, "unknown category"),
    ({"rules": {"language": {"min_severity": 7}}}, "min_severity"),
    ({"rules": {"language": {"action": "bleep"}}}, "action must be"),
    ({"extends": "nope"}, "unknown profile"),
    ({"pad_ms": {"skip": [1]}}, "before_ms, after_ms"),
    ({"colour": "red"}, "unknown key"),
])
def test_bad_profiles_are_rejected(raw, message):
    with pytest.raises(ProfileError, match=message):
        load_profiles({"p": raw})


def test_action_dataclass():
    assert Action(1, 5, "mute").length_ms == 4
