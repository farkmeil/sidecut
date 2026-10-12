"""Command line: sidecut tag | add | show | validate | compile | watch | probe | profiles"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path

from . import __version__, compile as compiler, media, model
from .config import Config, ConfigError, FilterStore, load_config
from .model import Filter, FilterError, MediaRef, Segment, format_time, parse_time
from .profile import ProfileError, plan_stats, resolve


def _config(args: argparse.Namespace) -> Config:
    explicit = getattr(args, "config", None)
    path = explicit or os.environ.get("SIDECUT_CONFIG")
    if not path and os.path.exists("/config/sidecut.yaml"):
        path = "/config/sidecut.yaml"
    if path and not explicit and not os.path.exists(path):
        print(f"note: no config at {path}; using defaults", file=sys.stderr)
        path = None
    return load_config(path)


def _store(config: Config) -> FilterStore:
    return FilterStore(config.filters_dir, config.sidecar, config.path_map)


def _out(text: str, path: str | None) -> None:
    if path and path != "-":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _resolve_filter_arg(target: str, config: Config) -> Path:
    """Accept either a filter file or a video (its filter is looked up)."""
    path = Path(target)
    if path.name.endswith(model.SUFFIX) or path.suffix == ".json":
        return path
    store = _store(config)
    found = store.find(store.to_plex(str(path.absolute())))
    if not found:
        raise FilterError(f"no filter found for {target}")
    return found


# ------------------------------------------------------------------- commands

def cmd_tag(args: argparse.Namespace) -> int:
    from .asr import WhisperTranscriber
    from .tagger import NoSubtitles, TagOptions, merge_language, tag_language
    from .wordlist import Wordlist

    config = _config(args)
    store = _store(config)
    wordlist = Wordlist.load(args.wordlist)
    videos = [video for target in args.paths for video in media.iter_videos(target)]
    if not videos:
        print("no video files found", file=sys.stderr)
        return 1
    if args.out and len(videos) != 1:
        print("--out needs exactly one video", file=sys.stderr)
        return 2
    todo = []
    for video in videos:
        if args.out:
            target = Path(args.out)
        elif args.sidecar:
            target = video.parent / (video.stem + model.SUFFIX)
        else:
            target = store.path_for_video(video)
        if target.exists() and not args.force:
            try:
                existing = model.load(target)
                if any(s.category == "language" and s.source in ("subtitle", "subtitle+asr", "asr")
                       for s in existing.segments) or existing.generator.get("language_tagger"):
                    print(f"skip (already tagged): {video.name}")
                    continue
            except FilterError:
                pass
        todo.append((video, target))
    if not todo:
        return 0
    transcriber = WhisperTranscriber(args.model, threads=args.threads, device=args.device,
                                     compute_type=args.compute_type)
    failures = 0
    untitled = []
    for number, (video, target) in enumerate(todo, 1):
        print(f"[{number}/{len(todo)}] {video.name}", flush=True)
        try:
            options = TagOptions(language=args.language, full=args.full, allow_full=not args.no_full,
                                 vad=not args.no_vad)
            flt, report = tag_language(video, transcriber, wordlist, options,
                                       log=lambda message: print(f"    {message}", flush=True))
            if target.exists():
                try:
                    flt = merge_language(model.load(target), flt)
                except FilterError:
                    pass
            model.save(flt, target)
        except NoSubtitles:
            untitled.append(video)
            print("    skipped: no text subtitles (--no-full)", flush=True)
            continue
        except (media.MediaError, OSError) as err:
            failures += 1
            print(f"    FAILED: {err}", file=sys.stderr, flush=True)
            continue
        tags = ", ".join(f"{tag} {count}" for tag, count in report["by_tag"].items()) or "none"
        print(f"    {report['segments']} language segments ({tags}); {report['method']}; "
              f"{report['elapsed_s']}s -> {target}", flush=True)
    if untitled:
        print(f"{len(untitled)} video(s) skipped for lack of text subtitles. Add an .srt next to them, "
              f"or run again without --no-full to transcribe their whole soundtrack.", flush=True)
    return 1 if failures else 0


def cmd_add(args: argparse.Namespace) -> int:
    config = _config(args)
    target = Path(args.filter)
    if not (target.name.endswith(model.SUFFIX) or target.suffix == ".json"):
        video = target
        store = _store(config)
        target = store.find(store.to_plex(str(video.absolute()))) or store.path_for_video(video)
        if not target.exists():
            info = media.probe(video)
            from .tagger import guess_identity

            identity = guess_identity(video)
            ref = MediaRef(title=identity.get("title", ""), year=identity.get("year"),
                           season=identity.get("season"), episode=identity.get("episode"),
                           duration_ms=media.duration_ms(info), size=os.path.getsize(video),
                           oshash=media.oshash(video))
            model.save(Filter(media=ref, generator=model.default_generator()), target)
    flt = model.load(target)
    segment = Segment.from_dict({
        "start_ms": parse_time(args.start), "end_ms": parse_time(args.end), "category": args.category,
        "severity": args.severity, **({"action": args.action} if args.action else {}),
        "tags": args.tag or [], "note": args.note or "", "source": "manual",
    })
    flt.segments.append(segment)
    model.save(flt, target)
    print(f"added {segment.category} {format_time(segment.start_ms)}-{format_time(segment.end_ms)} to {target}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    config = _config(args)
    path = _resolve_filter_arg(args.filter, config)
    flt = model.load(path)
    ref = flt.media
    print(f"{ref.title or path.name}{f' ({ref.year})' if ref.year else ''}"
          f"{f'  S{ref.season:02d}E{ref.episode:02d}' if ref.season is not None and ref.episode is not None else ''}")
    print(f"  file: {format_time(ref.duration_ms or 0)}  {ref.size or '?'} bytes  oshash {ref.oshash or '-'}")
    for category, by_severity in sorted(flt.summary().items()):
        counts = ", ".join(f"severity {sev}: {count}" for sev, count in sorted(by_severity.items()))
        print(f"  {category:<12} {counts}")
    if args.profile:
        profile = config.profiles.get(args.profile)
        if not profile:
            raise ProfileError(f"unknown profile '{args.profile}' (known: {', '.join(sorted(config.profiles))})")
        plan = resolve(flt, profile)
        stats = plan_stats(plan)
        print(f"  under '{profile.name}': {stats['skips']} skips ({format_time(stats['skipped_ms'])}), "
              f"{stats['mutes']} mutes ({format_time(stats['muted_ms'])})")
        if args.list:
            for action in plan:
                print(f"    {action.kind:<4} {format_time(action.start_ms)} - {format_time(action.end_ms)}  "
                      f"{', '.join(action.reasons)}")
    elif args.list:
        for s in sorted(flt.segments, key=lambda s: s.start_ms):
            print(f"    {format_time(s.start_ms)} - {format_time(s.end_ms)}  {s.category} sev {s.severity} "
                  f"{','.join(s.tags)} {s.note}".rstrip())
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    bad = 0
    for target in args.filters:
        try:
            flt = model.load(target)
            print(f"ok   {target} ({len(flt.segments)} segments)")
        except (FilterError, OSError) as err:
            bad += 1
            print(f"BAD  {err}")
    return 1 if bad else 0


def cmd_compile(args: argparse.Namespace) -> int:
    config = _config(args)
    path = _resolve_filter_arg(args.filter, config)
    flt = model.load(path)
    profile = config.profiles.get(args.profile)
    if not profile:
        raise ProfileError(f"unknown profile '{args.profile}' (known: {', '.join(sorted(config.profiles))})")
    plan = resolve(flt, profile)
    key = args.key
    if not key and args.format == "plexautoskip":
        for scheme in ("imdb", "tmdb", "tvdb"):
            if scheme in flt.media.ids:
                key = f"{scheme}://{flt.media.ids[scheme]}"
                break
    _out(compiler.render(plan, args.format, key=key, src=args.video, dst=args.render_to,
                         audio_stream=args.audio_stream), args.out)
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    config = _config(args)
    for name, profile in config.profiles.items():
        rules = ", ".join(
            f"{category} {rule.action or 'default'} (severity {rule.min_severity}+)"
            for category, rule in profile.rules.items() if rule
        ) or "nothing filtered"
        tags = ", ".join(f"{tag} {'off' if rule is None else 'on'}" for tag, rule in profile.tags.items())
        print(f"{name}: {rules}{f'; tags: {tags}' if tags else ''}")
    if config.users:
        print("users: " + ", ".join(f"{user} -> {name}" for user, name in config.users.items()))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from .enforcer import Enforcer
    from .plexio import PlexError, PlexServer

    config = _config(args)
    if args.enforce:
        config.enforce = True
    if not config.plex_token:
        print("no Plex token: set PLEX_TOKEN or plex.token in the config", file=sys.stderr)
        return 2
    enforcer = Enforcer(config, PlexServer(config.plex_url, config.plex_token))
    signal.signal(signal.SIGTERM, lambda *_: enforcer.stop())
    try:
        enforcer.run()
    except PlexError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Show live sessions and check what each player accepts."""
    from .plexio import PlayerLink, PlexError, PlexServer

    config = _config(args)
    if not config.plex_token:
        print("no Plex token: set PLEX_TOKEN or plex.token in the config", file=sys.stderr)
        return 2
    server = PlexServer(config.plex_url, config.plex_token)
    identity = server.identity()
    print(f"Plex {identity.get('version', '?')} at {config.plex_url}")
    registered = {c.get("machineIdentifier"): c for c in server.clients()}
    print(f"players registered for remote control: "
          f"{', '.join(c.get('name', '?') for c in registered.values()) or 'none right now'}")
    sessions = server.sessions()
    if not sessions:
        print("nothing is playing. Start something on the player you want to check and run this again.")
        return 0
    store = _store(config)
    for info in sessions:
        profile = config.profile_for_user(info.user)
        found = store.find(info.file) if info.file else None
        print(f"\n{info.label}")
        print(f"  player: {info.player_product} at {info.player_address or '?'} "
              f"({'local' if info.local else 'remote'}), state {info.player_state}, "
              f"registered: {'yes' if info.player_id in registered else 'no'}")
        print(f"  profile: {profile.name}; filter: {found or 'none'}")
        print(f"  file: {info.file}")
        prefs = config.prefs_for_player(info.player_title, info.player_id)
        for route in ("direct", "proxy"):
            link = PlayerLink(server, info, port=prefs.port, control=route)
            try:
                timeline = link.poll()
            except PlexError as err:
                print(f"  {route:<6} position poll: FAILED ({err})")
                continue
            if timeline is None:
                print(f"  {route:<6} position poll: answered, but no video timeline")
                continue
            print(f"  {route:<6} position poll: ok, at {format_time(timeline.time_ms)} {timeline.state}, "
                  f"volume {timeline.volume}")
            if timeline.state == "playing":
                import time as _time

                t0, base, drifts, values = _time.monotonic(), timeline.time_ms, [], [timeline.time_ms]
                for _ in range(5):
                    _time.sleep(0.3)
                    try:
                        sample = link.poll()
                    except PlexError:
                        break
                    if sample is None:
                        break
                    values.append(sample.time_ms)
                    drifts.append((sample.time_ms - base) - (_time.monotonic() - t0) * 1000)
                if drifts:
                    coarse = all(v % 1000 == 0 for v in values)
                    print(f"  {route:<6} position accuracy: within {max(abs(d) for d in drifts):.0f} ms of real time "
                          f"over {len(drifts)} polls{' (reported in whole seconds)' if coarse else ''}")
            if args.test_volume and timeline.volume is not None:
                try:
                    link.set_volume(timeline.volume)
                    print(f"  {route:<6} volume command: accepted (re-sent current volume {timeline.volume}, "
                          f"no audible change)")
                except PlexError as err:
                    print(f"  {route:<6} volume command: FAILED ({err})")
            if args.test_seek:
                try:
                    link.seek(timeline.time_ms)
                    print(f"  {route:<6} seek command: accepted (re-seeked to the current position)")
                except PlexError as err:
                    print(f"  {route:<6} seek command: FAILED ({err})")
    return 0


# ---------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sidecut", description="Open content filters for media you own.")
    parser.add_argument("--version", action="version", version=f"sidecut {__version__}")
    parser.add_argument("-c", "--config", help="config file (default: $SIDECUT_CONFIG or /config/sidecut.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tag", help="find profanity in videos and write language segments")
    p.add_argument("paths", nargs="+", help="video files or folders")
    p.add_argument("--out", help="write the filter here (single video only)")
    p.add_argument("--sidecar", action="store_true", help="write next to the video instead of the central store")
    p.add_argument("--force", action="store_true", help="re-tag videos that already have language segments")
    p.add_argument("--full", action="store_true", help="transcribe the whole soundtrack, ignore subtitles")
    p.add_argument("--no-full", action="store_true",
                   help="skip videos without text subtitles instead of transcribing their whole soundtrack")
    p.add_argument("--no-vad", action="store_true",
                   help="whole-soundtrack mode: do not skip stretches without speech (slower)")
    p.add_argument("--language", default="eng")
    p.add_argument("--model", default=os.environ.get("SIDECUT_MODEL", "small.en"))
    p.add_argument("--device", default=os.environ.get("SIDECUT_DEVICE", "cpu"))
    p.add_argument("--compute-type", default=os.environ.get("SIDECUT_COMPUTE_TYPE", "int8"))
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--wordlist", help="your own word list file")
    p.set_defaults(func=cmd_tag)

    p = sub.add_parser("add", help="add one segment by hand")
    p.add_argument("filter", help="filter file, or a video (its filter is found or created)")
    p.add_argument("--start", required=True, help="e.g. 1:02:03.5")
    p.add_argument("--end", required=True)
    p.add_argument("--category", required=True, choices=model.CATEGORIES)
    p.add_argument("--severity", type=int, default=2, choices=(1, 2, 3))
    p.add_argument("--action", choices=model.ACTIONS)
    p.add_argument("--tag", action="append")
    p.add_argument("--note")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("show", help="summarise a filter")
    p.add_argument("filter", help="filter file or video")
    p.add_argument("--profile", help="also show what this profile would do")
    p.add_argument("--list", action="store_true", help="list every segment or action")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("validate", help="check filter files against the spec")
    p.add_argument("filters", nargs="+")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("compile", help="apply a profile and emit a player format")
    p.add_argument("filter", help="filter file or video")
    p.add_argument("--profile", required=True)
    p.add_argument("--format", required=True, choices=("edl", "plexautoskip", "plan", "ffmpeg"))
    p.add_argument("--key", help="plexautoskip: Plex ratingKey or GUID (imdb://tt...)")
    p.add_argument("--video", help="ffmpeg: source video")
    p.add_argument("--render-to", help="ffmpeg: output video")
    p.add_argument("--audio-stream", type=int, default=0)
    p.add_argument("--out", help="write here instead of stdout")
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("profiles", help="list profiles and user mappings")
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("watch", help="run the live Plex enforcer")
    p.add_argument("--enforce", action="store_true", help="act on players (default is observe only)")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("probe", help="check what the players currently playing will accept")
    p.add_argument("--test-volume", action="store_true", help="re-send the current volume (no audible change)")
    p.add_argument("--test-seek", action="store_true", help="seek to the current position (brief hiccup)")
    p.set_defaults(func=cmd_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for noisy in ("urllib3", "websocket", "httpx", "httpcore", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    try:
        return int(args.func(args) or 0)
    except (FilterError, ProfileError, ConfigError, media.MediaError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
