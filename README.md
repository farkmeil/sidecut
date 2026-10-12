# sidecut

Open content filters for media you own.

Streaming services such as VidAngel and ClearPlay let a family mute language
or skip scenes, but only inside their apps and with their data. sidecut does
the same job for a personal library (Plex, Kodi, plain files) with open parts:

1. **An open filter format.** A small JSON file per video that lists *what is
   in it and when*: language, sex and nudity, violence, and so on. It makes no
   judgement. See [SPEC.md](SPEC.md).
2. **Profiles.** Each viewer chooses what to mute or skip. One filter serves
   every household.
3. **A language tagger.** Finds profanity automatically from subtitles plus
   speech recognition, timed to the word.
4. **Enforcers.** A live enforcer for Plex that mutes and skips on the player
   during playback, plus exports for Kodi (EDL), PlexAutoSkip and ffmpeg.

```
 video ──► tagger ──► Movie.sidecut.json ──┬──► Plex enforcer (live mute / skip)
           (or by hand)        +           ├──► Kodi .edl
                            profile        ├──► PlexAutoSkip custom.json
                                           └──► ffmpeg (filtered copy)
```

## Status

Version 0.1. Early, and honest about it:

| Part | State |
|---|---|
| Filter format, profiles, validation | Working, tested |
| Kodi EDL, PlexAutoSkip and ffmpeg output | Working, tested (ffmpeg renders checked by measuring the result) |
| Language tagger | Working. Tested with a scripted recogniser, measured against ground truth (below), and run on real films |
| Plex enforcer | Working against a simulated server and player. Real players vary: run `sidecut probe` on yours and start in observe mode |
| Scene tagging (nudity, violence) | By hand for now (`sidecut add`). Automatic tagging is planned |
| Shared filter library | Not yet. The format is built for it |

What it cannot do: change what Plex itself streams. Plex has no plugin system
any more, so the live enforcer is a separate process that watches sessions and
remote-controls the player, the way the Plex phone app does. That works for
players on your own network that accept remote control. For anything else,
render a filtered copy with ffmpeg.

## Quick start (Docker)

```sh
git clone https://github.com/farkmeil/sidecut && cd sidecut
cp docker-compose.example.yml docker-compose.yml     # set PLEX_URL and your media path
mkdir -p config data && cp config/sidecut.example.yaml config/sidecut.yaml
echo "PLEX_TOKEN=your-token" > .env
docker compose build
```

Tag a film, look at the result, then start the enforcer in observe mode:

```sh
docker compose run --rm sidecut tag "/media/movies/Some Film (1999)"
docker compose run --rm sidecut show "/media/movies/Some Film (1999)/Some Film.mkv" --profile family --list
docker compose up -d          # observe only until you set enforce: true
docker compose logs -f
```

In observe mode the log shows exactly what would happen and nothing is sent to
any player:

```
Kids on Living Room SHIELD: Some Film [family]: 2 skips (0:01:31.250), 41 mutes (0:00:38.900) - observing only
Kids on Living Room SHIELD: Some Film: would mute at 0:12:34.050 (language:s-word)
Kids on Living Room SHIELD: Some Film: would skip 0:45:10.700 -> 0:46:36.150 (sex_nudity:nudity)
```

When it looks right, set `enforce: true` in `config/sidecut.yaml` and restart.

## Commands

| Command | Does |
|---|---|
| `sidecut tag PATH...` | Tag language in videos or whole folders. Skips ones already done. |
| `sidecut add VIDEO --start 1:02:03 --end 1:03:10 --category sex_nudity --severity 2` | Add a segment by hand. |
| `sidecut show VIDEO [--profile P] [--list]` | Summarise a filter, or what a profile does with it. |
| `sidecut compile VIDEO --profile P --format edl\|plexautoskip\|ffmpeg\|plan` | Emit a player format. |
| `sidecut validate FILE...` | Check filter files against the spec. |
| `sidecut profiles` | List profiles and which user gets which. |
| `sidecut watch [--enforce]` | Run the Plex enforcer. |
| `sidecut probe [--test-volume] [--test-seek]` | Check what the players currently playing accept. |

## How the language tagger works

1. It reads text subtitles (a sidecar `.srt` or an embedded text track) and
   finds the lines containing listed words. Subtitle timing is too loose to
   mute single words, so subtitles only decide *where to listen*.
2. It runs Whisper with word timestamps on a few seconds of audio around each
   of those lines and matches the word list against what was actually said.
   In surround mixes it listens to the centre channel, where dialogue sits.
3. If a subtitle shows a listed word that recognition did not hear, the whole
   line is tagged, so nothing the subtitles show slips through.

With no text subtitles (image-based PGS on disc rips, or none) it transcribes
the whole soundtrack instead. That is slower and depends on recognition alone;
adding an `.srt` next to the file gives better results. `--no-full` skips such
files and lists them.

### How accurate is the timing

`tools/timing_check.py` measures it against ground truth: speech synthesised
word by word at known offsets, then run through the same recogniser and
matching. With the default `small.en` model on 80 target words:

| | Clean | With noise (10 dB) |
|---|---|---|
| Words found | 68 of 80 | 45 of 80 |
| Word start error, median (90% within) | 84 ms early (226 ms) | 124 ms early (210 ms) |
| Word end error, median (90% within) | 65 ms early (135 ms) | 70 ms early (142 ms) |
| Found words wholly inside the mute, no padding | 1% | 0% |
| Found words wholly inside the mute, default padding | 94% | 98% |

So padding is not optional: the default is 250 ms before and 300 ms after each
word. When it falls short, what is heard is an edge of the word, not the word.
On a real film, two model sizes agreed on word ends within 160 ms for 90% of
words.

The "words found" row is why subtitles matter. With subtitles, a word the
recogniser misses still gets its whole line muted. In whole-soundtrack mode a
missed word is simply missed. Synthetic speech is harder to recognise than
most film dialogue, so real recall should be better than this, but it will not
be perfect.

The word list is a plain text file in three tiers, grouped into families
(`f-word`, `blasphemy`, `hell`, ...) that profiles can switch on or off. See
[`src/sidecut/data/wordlist.en.txt`](src/sidecut/data/wordlist.en.txt). Pass
your own with `--wordlist`.

Filters never contain the words themselves, only the family and the time.

## How the Plex enforcer works

- It learns about playback from the server's notification channel and session
  list, and works out who is watching what on which player.
- The viewer's Plex user name picks a profile. Users without one are left
  alone.
- The filter is found by the file's path and checked against the file's size
  and duration, so a filter is never applied to a different cut.
- Shortly before a segment it polls the player for its exact position, then
  sends a seek (skip) or a volume change (mute) using the Plex Companion
  remote-control protocol.
- It checks the player really applied a mute. A player that ignores volume
  commands gets short skips instead, and that is remembered.
- It restores volume when playback stops, and after a crash.

Timing is good, not frame-exact: a command takes a few hundred milliseconds to
take effect, so every action is sent early and padded. Tune `pad_ms` per
profile and `tuning` per install. If you need exact cuts, Kodi with an EDL or
an ffmpeg render gives them.

Requirements on the player: on the same network as the server, and "Advertise
as player" (Plex Companion) enabled in the Plex app's settings.

## Using it without Plex

```sh
sidecut compile Film.sidecut.json --profile family --format edl --out "Film.edl"   # Kodi reads this next to Film.mkv
sidecut compile Film.sidecut.json --profile family --format ffmpeg --video Film.mkv --render-to Film.clean.mkv
```

## Development

```sh
pip install -e '.[dev,plex]'
pytest
```

The tests include a virtual-time player simulation for the timing logic, and
an end-to-end lab with a fake Plex server and fake player over real HTTP.

## Legal note

sidecut never modifies or redistributes media. A filter is a list of
timestamps and labels about a file you already have. Check the law where you
live before sharing filtered *copies* of anything.

## Credits

The enforcer's approach follows the trail blazed by
[PlexAutoSkip](https://github.com/mdhiggins/PlexAutoSkip), and the language
tagger's by [cleanvid](https://github.com/mmguero/cleanvid) and
[monkeyplug](https://github.com/mmguero/monkeyplug). Kodi's
[EDL support](https://kodi.wiki/view/Edit_decision_list) is what makes the
exact path possible.

MIT licensed.
