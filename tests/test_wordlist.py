import pytest

from sidecut.wordlist import Word, Wordlist, WordlistError, find_in_words

WL = Wordlist.load()


def tags(text):
    return [(m.tag, m.severity) for m in WL.find(text)]


@pytest.mark.parametrize("text, expected", [
    ("What the fuck is that?", [("f-word", 3)]),
    ("You FUCKING idiot", [("f-word", 3)]),
    ("motherfucker!", [("f-word", 3)]),
    ("mother fucker", [("f-word", 3)]),
    ("That's bullshit and you know it", [("s-word", 2)]),
    ("Oh, shit.", [("s-word", 2)]),
    ("son of a bitch", [("b-word", 2)]),
    ("He's an asshole", [("a-word", 2)]),
    ("kick his ass", [("a-word", 1)]),
    ("you dumb-ass", [("a-word", 1)]),
    ("Damn it! Go to hell.", [("damn", 1), ("hell", 1)]),
    ("goddamn it", [("blasphemy", 2)]),
    ("God damn it", [("blasphemy", 2)]),
    ("Oh my God!", [("blasphemy", 1)]),
    ("Oh, my God.", [("blasphemy", 1)]),
    ("Jesus Christ, Frank", [("blasphemy", 2)]),
    ("Jesus, that was close", [("deity", 1)]),
    ("for God’s sake", [("blasphemy", 1)]),
    ("I'm pissed off", [("piss", 1)]),
    ("what a load of crap", [("crap", 1)]),
])
def test_finds(text, expected):
    assert tags(text) == expected


@pytest.mark.parametrize("text", [
    "Hello there, class is in session",
    "Please pass the glasses, I must assess the damage",
    "Mr. Dickens ordered a cocktail in Scunthorpe",
    "The raccoon sat on the shelf",
    "He hit the ball over the hill",
    "They assassinated the ambassador at the embassy",
    "Shell casings on the hellenic coast",
    "God bless you. Thank God you're here.",
    "a basement full of bass guitars",
    "the Christmas tree in Essex",
    "tit for tat",
])
def test_leaves_innocent_text_alone(text):
    assert tags(text) == []


def test_longest_match_wins_and_matches_do_not_overlap():
    found = WL.find("that goddamn son of a bitch")
    assert [(m.tag) for m in found] == ["blasphemy", "b-word"]
    for a, b in zip(found, found[1:]):
        assert a.end <= b.start


def test_offsets_point_at_the_original_text():
    text = "Well, SHIT happens"
    [m] = WL.find(text)
    assert text[m.start:m.end] == "SHIT"


def test_custom_list_and_errors(tmp_path):
    path = tmp_path / "mine.txt"
    path.write_text("# mine\n2 minced  fudge\\w*\n1 minced  gosh darn\n")
    mine = Wordlist.load(str(path))
    assert [m.tag for m in mine.find("Oh fudgesicles, gosh-darn it")] == ["minced", "minced"]
    with pytest.raises(WordlistError, match="expected 'severity tag pattern'"):
        Wordlist.parse("fudge\n")
    with pytest.raises(WordlistError, match="bad pattern"):
        Wordlist.parse("1 x (unclosed\n")
    with pytest.raises(WordlistError, match="no entries"):
        Wordlist.parse("# nothing\n")


def words(*items):
    return [Word(start, end, text, prob) for start, end, text, prob in items]


def test_timed_matches_use_the_word_boundaries():
    heard = words((0, 300, " What", 0.99), (300, 450, " the", 0.99), (450, 900, " fuck,", 0.87),
                  (900, 1200, " man?", 0.95))
    [m] = find_in_words(WL, heard)
    assert (m.start_ms, m.end_ms, m.tag, m.severity) == (450, 900, "f-word", 3)
    assert m.probability == pytest.approx(0.87)


def test_phrases_span_several_recognised_words():
    heard = words((1000, 1200, " Oh,", 0.9), (1200, 1400, " my", 0.8), (1400, 1900, " God!", 0.95),
                  (2500, 2700, " son", 0.9), (2700, 2800, " of", 0.9), (2800, 2850, " a", 0.9), (2850, 3300, " bitch.", 0.7))
    found = find_in_words(WL, heard)
    assert [(m.start_ms, m.end_ms, m.tag) for m in found] == [(1000, 1900, "blasphemy"), (2500, 3300, "b-word")]
    assert found[0].probability == pytest.approx(0.8)


def test_recogniser_noise_is_tolerated():
    heard = words((0, 100, "", 0.5), (100, 200, " ...", 0.5), (200, 600, " Bullshit!", 0.9), (600, 700, " -", 0.2))
    assert [(m.start_ms, m.end_ms) for m in find_in_words(WL, heard)] == [(200, 600)]
    assert find_in_words(WL, []) == []
