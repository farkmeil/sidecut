# sidecut filter format, version 0.1

A sidecut filter is a small JSON file that lists time ranges in **one specific
video file** and says what is in each range. It makes no decisions. A viewer's
*profile* decides what to mute or skip, and a player enforces the result.

Keeping those three things apart is the point of the format:

| Thing | Answers | Lives in |
|---|---|---|
| Filter | What is in this file, and when? | `*.sidecut.json`, shareable |
| Profile | What does this viewer not want? | The viewer's own config |
| Enforcer | How does this player mute or skip? | Player, plugin or encoder |

One filter serves every household. Nobody has to agree on what is acceptable,
only on what is there.

## File name and location

`<video file name without extension>.sidecut.json`, in either place:

1. Next to the video (a sidecar, like a subtitle file).
2. In a central folder that mirrors the library's paths, so media folders can
   stay read-only: `<store>/<path as the media server reports it>.sidecut.json`.

Encoding is UTF-8.

## Top level

```json
{
  "sidecut": "0.1",
  "media": { },
  "generator": { },
  "segments": [ ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `sidecut` | yes | Spec version, `"major.minor"`. |
| `segments` | yes | List of segments. May be empty: "checked, nothing found". |
| `media` | no, but strongly advised | Which file this was made for. |
| `generator` | no | Free-form notes on how the file was made. |

Readers must ignore fields they do not know, at every level. Readers must
refuse a file whose major version they do not support.

## `media`: which file

Timings only hold for one cut of a title. A streaming release, a disc remux
and a director's cut of the same film all differ. So a filter records a
fingerprint of the file it was timed against, and an enforcer checks it before
acting.

| Field | Meaning |
|---|---|
| `duration_ms` | Duration of the file in milliseconds. |
| `size` | File size in bytes. |
| `oshash` | OpenSubtitles hash, 16 hex digits: file size plus the 64-bit little-endian sums of the first and last 64 KiB. Cheap to compute on network shares. |
| `title`, `year`, `season`, `episode`, `edition` | For people and for search. |
| `ids` | External ids as strings, e.g. `{"imdb": "tt0096895", "tmdb": "268"}`. |
| `audio_language` | Language of the audio track that language segments were timed against. |

Matching rule for enforcers: if both sides have `oshash`, it decides. Otherwise
`size` must be equal when both are known, and `duration_ms` must agree within
2 seconds when both are known. If nothing can be compared the filter may be
applied. A filter that fails the check must not be applied.

## `segments`

```json
{ "start_ms": 754120, "end_ms": 754680, "category": "language", "severity": 3,
  "action": "mute", "tags": ["f-word"], "source": "subtitle+asr", "confidence": 0.94 }
```

| Field | Required | Meaning |
|---|---|---|
| `start_ms`, `end_ms` | yes | Integers, milliseconds from the start of the file. `end_ms` > `start_ms`. The tight extent of the content, with no safety margin. |
| `category` | yes | One of the categories below. |
| `severity` | no, default 2 | 1 mild, 2 moderate, 3 strong. |
| `action` | no | Suggested handling: `mute` (silence audio, picture continues) or `skip` (jump over). Default is `mute` for `language`, `skip` for everything else. A profile may override it. |
| `tags` | no | Free-form finer labels, e.g. `f-word`, `blasphemy`, `nudity`, `gore`. Profiles can switch individual tags on or off. |
| `note` | no | Human note, e.g. what plot is lost by skipping. |
| `source` | no | How the segment was made: `manual`, `subtitle`, `subtitle+asr`, `asr`, or a tool name. |
| `confidence` | no | 0 to 1, for machine-made segments. |

Segments may overlap and may be listed in any order.

Segments hold the tight extent on purpose. Safety margins depend on the player
(how fast it can react), so they belong to the profile and the enforcer.

### Categories

These follow the five headings parents already know from film advisories.

| Category | Covers | Severity guide |
|---|---|---|
| `language` | Profanity, slurs, blasphemy, crude terms. | 1 mild words, 2 moderate, 3 strongest words and slurs. |
| `sex_nudity` | Sexual activity, nudity, strong innuendo. | 1 innuendo or brief partial nudity, 2 implied sex or clear nudity, 3 explicit. |
| `violence` | Violence and gore. | 1 mild or comic, 2 realistic with injury, 3 graphic. |
| `substances` | Alcohol, drugs, smoking. | 1 incidental, 2 prominent use, 3 hard drug use shown. |
| `frightening` | Intense or disturbing scenes. | 1 mild peril, 2 sustained, 3 disturbing. |
| `other` | Anything else worth marking. Use `tags`. | |

## Profiles (informative)

A profile maps categories and tags to actions. This is sidecut's own layout;
other tools may store profiles however they like.

```yaml
profiles:
  kids:
    rules:
      language:   {min_severity: 1, action: mute}
      sex_nudity: {min_severity: 1, action: skip}
      violence:   {min_severity: 2, action: skip}
    tags:
      hell: off          # a tag rule beats its category rule
    pad_ms:
      mute: [250, 300]   # margin before and after, per action
      skip: [300, 150]
```

Resolving a filter against a profile gives a *plan*: padded intervals, merged
where they touch, with mutes removed where a skip already covers them. The plan
is sorted and never overlaps.

## Enforcement targets (informative)

| Target | Skip | Mute |
|---|---|---|
| Kodi / MPlayer EDL (`.edl` next to the video) | action `0` | action `1` |
| PlexAutoSkip `custom.json` | `"mode": "skip"` | `"mode": "volume"` |
| ffmpeg render | `select` / `aselect` | `volume=enable='between(t,a,b)':volume=0` |
| sidecut enforcer (Plex Companion) | `playback/seekTo` | `playback/setParameters?volume=0` |

## Versioning

Minor versions add optional fields only. A reader for 0.1 can read any 0.x
file. A major version change may break readers.

A JSON Schema for this version is in
[`schema/sidecut-filter-0.1.schema.json`](schema/sidecut-filter-0.1.schema.json).

## Not in 0.1

- Regions of the picture to blur or cover.
- Replacement audio (bleeps, alternate takes).
- Mapping one filter onto a different cut of the same title by audio alignment.
