import pytest

from sidecut.config import ConfigError, FilterStore, load_config

YAML = """
plex:
  url: http://10.0.0.5:32400/
filters:
  dir: {root}/filters
path_map:
  /media: {root}/library
enforce: false
users:
  Kids: littles
  Guest: off
  "*": language
players:
  "Living Room SHIELD": {{port: 32500, mute: skip}}
profiles:
  littles:
    extends: family
    rules:
      violence: {{min_severity: 2}}
    tags: {{hell: off}}
tuning:
  skip_lead_ms: 500
"""


def write(tmp_path, text=YAML):
    path = tmp_path / "sidecut.yaml"
    path.write_text(text.format(root=tmp_path))
    return path


def test_loads_everything(tmp_path):
    config = load_config(write(tmp_path), env={"PLEX_TOKEN": "secret"})
    assert config.plex_url == "http://10.0.0.5:32400" and config.plex_token == "secret"
    assert config.enforce is False
    assert config.profile_for_user("kids").name == "littles"          # names are case-insensitive
    assert config.profile_for_user("Guest").is_off                    # YAML `off` means no filtering
    assert config.profile_for_user("somebody new").name == "language"  # the "*" entry
    assert config.profiles["littles"].rules["violence"].min_severity == 2
    assert config.profiles["littles"].tags["hell"] is None
    assert config.prefs_for_player("Living Room SHIELD", "abc").mute == "skip"
    assert config.prefs_for_player("Other", "abc").mute == "volume"
    assert config.tuning.skip_lead_ms == 500 and config.tuning.mute_lead_ms == 200
    assert config.state_file == f"{tmp_path}/state.json"


def test_env_overrides_and_token_file(tmp_path):
    (tmp_path / "token").write_text("from-file\n")
    config = load_config(write(tmp_path), env={"PLEX_URL": "http://plex:32400", "PLEX_TOKEN_FILE": str(tmp_path / "token"),
                                               "SIDECUT_ENFORCE": "true"})
    assert config.plex_url == "http://plex:32400" and config.plex_token == "from-file" and config.enforce is True


def test_defaults_without_a_file():
    config = load_config(None, env={})
    assert config.enforce is False and config.default_profile == "off"
    assert config.profile_for_user("anyone").is_off
    assert {"family", "language", "strict", "off"} <= set(config.profiles)


@pytest.mark.parametrize("text, message", [
    ("users: {Kids: nonexistent}\n", "unknown profile 'nonexistent'"),
    ("enforced: true\n", "unknown config key"),
    ("tuning: {warp: 9}\n", "tuning: unknown key"),
    ("players: {TV: {mute: loud}}\n", "use volume, skip or off"),
    ("players: {TV: {colour: red}}\n", "unknown key"),
    ("- a\n- b\n", "top level must be a mapping"),
])
def test_mistakes_are_caught_at_load(tmp_path, text, message):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_config(path, env={})


def test_store_paths_and_lookup(tmp_path):
    library = tmp_path / "library/movies/Heat (1995)"
    library.mkdir(parents=True)
    video = library / "Heat.1995.mkv"
    video.write_bytes(b"x")
    store = FilterStore(tmp_path / "filters", sidecar=True, path_map={"/media": str(tmp_path / "library")})
    plex_path = "/media/movies/Heat (1995)/Heat.1995.mkv"
    assert store.to_local(plex_path) == str(video)
    assert store.to_plex(str(video)) == plex_path
    assert store.to_local("/elsewhere/x.mkv") == "/elsewhere/x.mkv"
    central = tmp_path / "filters/media/movies/Heat (1995)/Heat.1995.sidecut.json"
    assert store.store_path(plex_path) == central
    assert store.path_for_video(video) == central
    assert store.find(plex_path) is None
    central.parent.mkdir(parents=True)
    central.write_text("{}")
    assert store.find(plex_path) == central
    sidecar = library / "Heat.1995.sidecut.json"
    sidecar.write_text("{}")
    assert store.find(plex_path) == sidecar                      # a sidecar wins
    assert FilterStore(tmp_path / "filters", sidecar=False,
                       path_map={"/media": str(tmp_path / "library")}).find(plex_path) == central
