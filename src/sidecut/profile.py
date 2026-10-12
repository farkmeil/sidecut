"""Viewer profiles: who gets what filtered, and how.

A filter describes what is in a file. A profile is one viewer's choice about
which of those things to mute or skip. `resolve` combines the two into a plan:
a sorted list of non-overlapping mute and skip intervals a player can enforce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import ACTIONS, CATEGORIES, Filter, Segment

DEFAULT_PAD_MS = {"mute": (150, 150), "skip": (300, 150)}
DEFAULT_MERGE_GAP_MS = {"mute": 250, "skip": 2000}
MIN_ACTION_MS = 40


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    min_severity: int = 1
    action: str | None = None  # None: use what the segment suggests

    def applies(self, segment: Segment) -> bool:
        return segment.severity >= self.min_severity


@dataclass
class Profile:
    name: str
    rules: dict[str, Rule | None] = field(default_factory=dict)
    tags: dict[str, Rule | None] = field(default_factory=dict)
    pad_ms: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(DEFAULT_PAD_MS))
    merge_gap_ms: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MERGE_GAP_MS))

    @property
    def is_off(self) -> bool:
        return not any(self.rules.values()) and not any(self.tags.values())

    def rule_for(self, segment: Segment) -> Rule | None:
        """Tag rules win over category rules. A tag set to off exempts the
        segment unless another of its tags has a rule that is on."""
        tag_hits = [self.tags[t] for t in segment.tags if t in self.tags]
        if tag_hits:
            live = [r for r in tag_hits if r is not None]
            return live[0] if live else None
        return self.rules.get(segment.category)


@dataclass(frozen=True)
class Action:
    start_ms: int
    end_ms: int
    kind: str  # "mute" | "skip"
    reasons: tuple[str, ...] = ()

    @property
    def length_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms, "action": self.kind, "reasons": list(self.reasons)}


PRESETS: dict[str, dict[str, Any]] = {
    "off": {"rules": {}},
    "language": {"rules": {"language": {"min_severity": 1, "action": "mute"}}, "tags": {"deity": False}},
    "family": {
        "rules": {
            "language": {"min_severity": 1, "action": "mute"},
            "sex_nudity": {"min_severity": 1, "action": "skip"},
            "violence": {"min_severity": 3, "action": "skip"},
        },
        "tags": {"deity": False},
    },
    "strict": {
        "rules": {
            "language": {"min_severity": 1, "action": "mute"},
            "sex_nudity": {"min_severity": 1, "action": "skip"},
            "violence": {"min_severity": 2, "action": "skip"},
            "substances": {"min_severity": 2, "action": "skip"},
            "frightening": {"min_severity": 3, "action": "skip"},
        },
        "tags": {"deity": False},
    },
}


def _parse_rule(raw: Any, where: str) -> Rule | None:
    if raw is None or raw is False or (isinstance(raw, str) and raw.lower() in ("off", "no", "none", "false")):
        return None
    if raw is True or (isinstance(raw, str) and raw.lower() in ("on", "yes", "true")):
        return Rule()
    if isinstance(raw, str) and raw in ACTIONS:
        return Rule(action=raw)
    if isinstance(raw, int) and not isinstance(raw, bool):
        raw = {"min_severity": raw}
    if not isinstance(raw, dict):
        raise ProfileError(f"{where}: expected off, on, mute, skip or an object")
    unknown = set(raw) - {"min_severity", "action"}
    if unknown:
        raise ProfileError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
    severity = raw.get("min_severity", 1)
    if isinstance(severity, bool) or not isinstance(severity, int) or not 1 <= severity <= 3:
        raise ProfileError(f"{where}: min_severity must be 1, 2 or 3")
    action = raw.get("action")
    if action is not None and action not in ACTIONS:
        raise ProfileError(f"{where}: action must be one of {', '.join(ACTIONS)}")
    return Rule(min_severity=severity, action=action)


def _parse_pair(raw: Any, where: str) -> tuple[int, int]:
    if isinstance(raw, int) and not isinstance(raw, bool):
        raw = [raw, raw]
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in raw)):
        raise ProfileError(f"{where}: expected [before_ms, after_ms]")
    return int(raw[0]), int(raw[1])


def parse_profile(name: str, raw: dict[str, Any] | None, known: dict[str, Profile] | None = None) -> Profile:
    raw = dict(raw or {})
    known = known or {}
    unknown = set(raw) - {"extends", "rules", "tags", "pad_ms", "merge_gap_ms"}
    if unknown:
        raise ProfileError(f"profile '{name}': unknown key(s) {', '.join(sorted(unknown))}")
    base_name = raw.get("extends")
    if base_name:
        if base_name in known:
            base = known[base_name]
        elif base_name in PRESETS:
            base = parse_profile(base_name, PRESETS[base_name])
        else:
            raise ProfileError(f"profile '{name}': extends unknown profile '{base_name}'")
        profile = Profile(name=name, rules=dict(base.rules), tags=dict(base.tags),
                          pad_ms=dict(base.pad_ms), merge_gap_ms=dict(base.merge_gap_ms))
    else:
        profile = Profile(name=name)
    for category, rule_raw in (raw.get("rules") or {}).items():
        if category not in CATEGORIES:
            raise ProfileError(f"profile '{name}': unknown category '{category}'")
        profile.rules[category] = _parse_rule(rule_raw, f"profile '{name}' rule '{category}'")
    for tag, rule_raw in (raw.get("tags") or {}).items():
        profile.tags[str(tag)] = _parse_rule(rule_raw, f"profile '{name}' tag '{tag}'")
    for kind, pair in (raw.get("pad_ms") or {}).items():
        if kind not in ACTIONS:
            raise ProfileError(f"profile '{name}': pad_ms has unknown action '{kind}'")
        profile.pad_ms[kind] = _parse_pair(pair, f"profile '{name}' pad_ms.{kind}")
    for kind, gap in (raw.get("merge_gap_ms") or {}).items():
        if kind not in ACTIONS or isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
            raise ProfileError(f"profile '{name}': merge_gap_ms.{kind} must be a non-negative integer")
        profile.merge_gap_ms[kind] = gap
    return profile


def load_profiles(raw: dict[str, Any] | None) -> dict[str, Profile]:
    """Presets plus whatever the config defines. Config entries may extend
    presets or each other (in order of definition)."""
    profiles = {name: parse_profile(name, spec) for name, spec in PRESETS.items()}
    for name, spec in (raw or {}).items():
        if spec is None or spec is False:
            spec = {}
        if not isinstance(spec, dict):
            raise ProfileError(f"profile '{name}': must be an object")
        profiles[str(name)] = parse_profile(str(name), spec, profiles)
    return profiles


def _merge(intervals: list[tuple[int, int, tuple[str, ...]]], gap: int) -> list[tuple[int, int, tuple[str, ...]]]:
    out: list[tuple[int, int, tuple[str, ...]]] = []
    for start, end, reasons in sorted(intervals):
        if out and start <= out[-1][1] + gap:
            prev = out[-1]
            merged = prev[2] + tuple(r for r in reasons if r not in prev[2])
            out[-1] = (prev[0], max(prev[1], end), merged)
        else:
            out.append((start, end, reasons))
    return out


def _subtract(mutes: list[tuple[int, int, tuple[str, ...]]],
              skips: list[tuple[int, int, tuple[str, ...]]]) -> list[tuple[int, int, tuple[str, ...]]]:
    """Remove from each mute the parts a skip already covers."""
    out = []
    for start, end, reasons in mutes:
        pieces = [(start, end)]
        for s_start, s_end, _ in skips:
            nxt = []
            for a, b in pieces:
                if s_end <= a or s_start >= b:
                    nxt.append((a, b))
                    continue
                if a < s_start:
                    nxt.append((a, s_start))
                if s_end < b:
                    nxt.append((s_end, b))
            pieces = nxt
        out.extend((a, b, reasons) for a, b in pieces if b - a >= MIN_ACTION_MS)
    return out


def resolve(flt: Filter, profile: Profile, duration_ms: int | None = None) -> list[Action]:
    """Apply a profile to a filter. Returns sorted, non-overlapping actions."""
    duration = duration_ms or flt.media.duration_ms
    buckets: dict[str, list[tuple[int, int, tuple[str, ...]]]] = {"mute": [], "skip": []}
    for segment in flt.segments:
        rule = profile.rule_for(segment)
        if rule is None or not rule.applies(segment):
            continue
        kind = rule.action or segment.suggested_action
        before, after = profile.pad_ms.get(kind, DEFAULT_PAD_MS[kind])
        start = max(0, segment.start_ms - before)
        end = segment.end_ms + after
        if duration:
            end = min(end, duration)
        if end - start < MIN_ACTION_MS:
            continue
        label = segment.category if not segment.tags else f"{segment.category}:{segment.tags[0]}"
        buckets[kind].append((start, end, (label,)))
    skips = _merge(buckets["skip"], profile.merge_gap_ms.get("skip", DEFAULT_MERGE_GAP_MS["skip"]))
    mutes = _merge(buckets["mute"], profile.merge_gap_ms.get("mute", DEFAULT_MERGE_GAP_MS["mute"]))
    mutes = _subtract(mutes, skips)
    actions = [Action(s, e, "skip", r) for s, e, r in skips] + [Action(s, e, "mute", r) for s, e, r in mutes]
    actions.sort(key=lambda a: (a.start_ms, a.end_ms))
    return actions


def plan_stats(actions: list[Action]) -> dict[str, Any]:
    skips = [a for a in actions if a.kind == "skip"]
    mutes = [a for a in actions if a.kind == "mute"]
    return {
        "skips": len(skips),
        "skipped_ms": sum(a.length_ms for a in skips),
        "mutes": len(mutes),
        "muted_ms": sum(a.length_ms for a in mutes),
    }
