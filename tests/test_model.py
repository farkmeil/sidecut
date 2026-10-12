import json
from pathlib import Path

import pytest

from sidecut import model
from sidecut.model import Filter, FilterError, MediaRef, Segment, format_time, parse_time

ROOT = Path(__file__).parent.parent


def test_example_loads_and_round_trips(tmp_path):
    flt = model.load(ROOT / "examples/example.sidecut.json")
    assert len(flt.segments) == 4
    assert flt.segments[2].suggested_action == "skip" and flt.segments[0].suggested_action == "mute"
    out = tmp_path / "deep/dir/x.sidecut.json"
    model.save(flt, out)
    assert model.load(out).to_dict() == flt.to_dict()
    assert not list(out.parent.glob("*.tmp"))


def test_schema_accepts_what_we_write_and_rejects_bad_files():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schema/sidecut-filter-0.1.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    example = json.loads((ROOT / "examples/example.sidecut.json").read_text())
    jsonschema.validate(example, schema)
    jsonschema.validate(model.load(ROOT / "examples/example.sidecut.json").to_dict(), schema)
    for broken in (
        {"segments": []},
        {"sidecut": "0.1", "segments": [{"start_ms": 0, "end_ms": 5, "category": "swearing"}]},
        {"sidecut": "0.1", "segments": [{"start_ms": -1, "end_ms": 5, "category": "language"}]},
        {"sidecut": "0.1", "segments": [{"start_ms": 0, "end_ms": 5, "category": "language", "severity": 4}]},
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)


@pytest.mark.parametrize("raw, message", [
    ({"segments": []}, "missing 'sidecut'"),
    ({"sidecut": "2.0", "segments": []}, "not supported"),
    ({"sidecut": "0.1"}, "'segments' must be a list"),
    ({"sidecut": "0.1", "segments": [{"start_ms": 5, "end_ms": 5, "category": "language"}]}, "after start_ms"),
    ({"sidecut": "0.1", "segments": [{"start_ms": 1.5, "end_ms": 5, "category": "language"}]}, "integer"),
    ({"sidecut": "0.1", "segments": [{"start_ms": 1, "end_ms": 5, "category": "nope"}]}, "unknown category"),
    ({"sidecut": "0.1", "segments": [{"start_ms": 1, "end_ms": 5, "category": "language", "severity": 0}]}, "severity"),
    ({"sidecut": "0.1", "segments": [{"start_ms": 1, "end_ms": 5, "category": "language", "action": "bleep"}]}, "unknown action"),
    ({"sidecut": "0.1", "media": {"duration_ms": 1000},
      "segments": [{"start_ms": 5000, "end_ms": 6000, "category": "language"}]}, "after the end"),
])
def test_invalid_filters_are_rejected_with_a_reason(raw, message):
    with pytest.raises(FilterError, match=message):
        Filter.from_dict(raw)


def test_later_minor_versions_and_unknown_fields_are_tolerated():
    flt = Filter.from_dict({"sidecut": "0.7", "future": 1, "segments": [
        {"start_ms": 1, "end_ms": 5, "category": "language", "blur_box": [0, 0, 1, 1]}]})
    assert len(flt.segments) == 1


def test_not_json(tmp_path):
    bad = tmp_path / "x.sidecut.json"
    bad.write_text("{nope")
    with pytest.raises(FilterError, match="not valid JSON"):
        model.load(bad)


@pytest.mark.parametrize("text, ms", [
    ("1:02:03.5", 3_723_500), ("62:03", 3_723_000), ("3723.5", 3_723_500), ("90", 90_000),
    ("1500ms", 1500), (12.25, 12_250), ("0:00:00.001", 1),
])
def test_parse_time(text, ms):
    assert parse_time(text) == ms


def test_parse_time_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_time("soon")


def test_format_time():
    assert format_time(3_723_500) == "1:02:03.500"
    assert format_time(0) == "0:00:00.000"


def test_fingerprint_matching():
    ref = MediaRef(duration_ms=5_400_000, size=1000, oshash="8e245d9679d31e12")
    assert ref.matches(oshash="8E245D9679D31E12")[0]
    assert not ref.matches(oshash="0000000000000000")[0]
    assert ref.matches(duration_ms=5_400_900, size=1000)[0]
    assert not ref.matches(duration_ms=5_400_000, size=1001)[0]
    assert not ref.matches(duration_ms=5_410_000)[0]
    assert ref.matches()[0]
    assert MediaRef().matches(duration_ms=1, size=2)[0]  # nothing recorded, nothing to contradict


def test_summary_counts():
    flt = Filter(segments=[Segment(0, 1, "language", 3), Segment(2, 3, "language", 3), Segment(4, 5, "violence", 1)])
    assert flt.summary() == {"language": {3: 2}, "violence": {1: 1}}
