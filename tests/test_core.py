"""Behavioral checks for the native scanner against independent small oracles."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product
import random

import pytest

from dagger import Tagger
from dagger.model import compile_model


NOUN = "N,*,*,*"
VERB = "V,*,*,*"
PARTICLE = "P,*,*,*"


def pattern(surface, pos=NOUN, previous=None, shift=None, strip=0, suffix="", reading="*"):
    return surface, previous, len(surface) if shift is None else shift, pos, strip, suffix, reading


def make_tagger(tmp_path, patterns):
    path = tmp_path / "test.dgr"
    compile_model(patterns, path)
    return Tagger(path)


def test_right_context_matches_without_consuming_it(tmp_path):
    tagger = make_tagger(tmp_path, [
        pattern("甲", NOUN), pattern("乙", NOUN),
        pattern("甲乙", VERB, shift=1),
    ])
    tokens = tagger.analyze("甲乙")
    assert [(t.surface, t.pos, t.start, t.end) for t in tokens] == [
        ("甲", VERB, 0, 1), ("乙", NOUN, 1, 2),
    ]
    assert tagger.analyze("甲")[0].pos == NOUN


def test_longer_surface_wins_before_previous_pos_specificity(tmp_path):
    tagger = make_tagger(tmp_path, [
        pattern("前", PARTICLE), pattern("甲", NOUN),
        pattern("甲", VERB, previous=PARTICLE), pattern("甲乙", NOUN),
    ])
    assert [(t.surface, t.pos) for t in tagger.analyze("前甲")] == [("前", PARTICLE), ("甲", VERB)]
    assert [(t.surface, t.pos) for t in tagger.analyze("前甲乙")] == [("前", PARTICLE), ("甲乙", NOUN)]


def test_same_surface_context_wins_and_bos_is_distinct(tmp_path):
    tagger = make_tagger(tmp_path, [
        pattern("前", PARTICLE), pattern("甲", NOUN),
        pattern("甲", VERB, previous=PARTICLE),
        pattern("甲", PARTICLE, previous=""),
    ])
    assert tagger.analyze("甲")[0].pos == PARTICLE
    assert [t.pos for t in tagger.analyze("前甲")] == [PARTICLE, VERB]
    # A new sentence must not inherit the previous sentence's last POS.
    assert tagger.parse_batch_bytes("前甲\n甲\n".encode()) == (tagger.parse("前甲") + tagger.parse("甲")).encode()


def test_unknown_numbers_letters_and_spaces_are_preserved_and_grouped(tmp_path):
    tagger = make_tagger(tmp_path, [pattern("猫")])
    text = "abc１２3猫 ＡＢz🙂"
    tokens = tagger.analyze(text)
    assert [t.surface for t in tokens] == ["abc", "１２3", "猫", " ", "ＡＢz", "🙂"]
    assert "".join(t.surface for t in tokens) == text
    assert [t.is_unknown for t in tokens] == [True, True, False, True, True, True]
    assert tokens[1].pos == "名詞,数詞,*,*"
    assert tokens[3].pos == "特殊,空白,*,*"
    assert tokens[-1].pos == "特殊,記号,*,*"
    assert all(t.lemma == "*" for t in tokens if t.is_unknown)


def test_non_bmp_unicode_offsets_and_lemma_edit(tmp_path):
    tagger = make_tagger(tmp_path, [
        pattern("𠮷野家"),
        pattern("食べた", VERB, strip=1, suffix="る", reading="タベタ"),
    ])
    tokens = tagger.analyze("𠮷野家食べた🙂")
    assert [(t.surface, t.start, t.end) for t in tokens] == [
        ("𠮷野家", 0, 3), ("食べた", 3, 6), ("🙂", 6, 7),
    ]
    assert tokens[1].lemma == "食べる"
    assert tokens[1].reading == "タベタ"
    assert "食べた\tV,*,*,*,食べる,タベタ,*\n" in tagger.parse("食べた")


def test_batch_empty_lines_crlf_and_unterminated_last_line(tmp_path):
    tagger = make_tagger(tmp_path, [pattern("猫"), pattern("犬")])
    data = "猫\r\n\n犬".encode()
    assert tagger.parse_batch_bytes(data) == (tagger.parse("猫") + tagger.parse("") + tagger.parse("犬")).encode()
    assert tagger.parse_batch_bytes(data, wakati=True) == "猫 \n\n犬 \n".encode()
    assert tagger.parse_batch_bytes(b"") == b""
    assert tagger.parse_batch_bytes(b"\n") == b"EOS\n"
    assert tagger.count_batch_bytes(data) == 2


@pytest.mark.parametrize("invalid", [b"\xff", b"\xe3\x81", b"\xc0\xaf", b"\xed\xa0\x80", b"\xf4\x90\x80\x80"])
def test_invalid_utf8_is_rejected_before_native_scan(tmp_path, invalid):
    tagger = make_tagger(tmp_path, [pattern("猫")])
    with pytest.raises(UnicodeDecodeError):
        tagger.parse_batch_bytes(invalid)
    with pytest.raises(UnicodeDecodeError):
        tagger.count_batch_bytes(invalid)


def test_count_and_output_apis_do_the_same_segmentation(tmp_path):
    tagger = make_tagger(tmp_path, [pattern("猫"), pattern("東京"), pattern("大学")])
    texts = ["", "猫", "東京大学", "abc123猫", "𠮷🙂🐕", "カタカナ", " \t　", "１２.３"]
    for text in texts:
        assert tagger.count(text) == len(tagger.tokenize(text)) == len(tagger.analyze(text))
    batch = "\n".join(texts).encode()
    assert tagger.count_batch_bytes(batch) == sum(len(tagger.tokenize(text)) for text in texts)


def reference_analyze(patterns, text):
    """Literal feature matching oracle, independent of trie layout and IDs.

    All characters have generic one-character rules in the random test, so
    this oracle deliberately has no unknown-word or character-class policy.
    """
    result = []
    previous = ""
    at = 0
    while at < len(text):
        matches = [p for p in patterns if text.startswith(p[0], at) and p[1] in (None, previous)]
        chosen = max(matches, key=lambda p: (len(p[0]), p[1] is not None))
        _surface, _previous, shift, pos, strip, suffix, reading = chosen
        surface = text[at:at + shift]
        lemma = (surface[:-strip] if strip else surface) + suffix
        result.append((surface, pos, lemma, reading, at, at + shift, False))
        previous = pos
        at += shift
    return result


def test_random_small_languages_match_literal_feature_oracle(tmp_path):
    rng = random.Random(107)
    alphabet = "甲乙丙"
    pos_names = [NOUN, VERB, PARTICLE]
    queries = ["".join(chars) for size in range(1, 6) for chars in product(alphabet, repeat=size)]
    for _ in range(4):
        rules = {(char, None): pattern(char, rng.choice(pos_names)) for char in alphabet}
        for _ in range(90):
            surface = "".join(rng.choices(alphabet, k=rng.randint(1, 4)))
            previous = rng.choice([None, ""] + pos_names)
            shift = rng.randint(1, len(surface))
            strip = rng.randint(0, shift)
            rules[surface, previous] = pattern(surface, rng.choice(pos_names), previous,
                                                shift, strip, rng.choice(["", "丁"]), "ヨミ")
        patterns = list(rules.values())
        tagger = make_tagger(tmp_path, patterns)
        for text in queries:
            actual = [(t.surface, t.pos, t.lemma, t.reading, t.start, t.end, t.is_unknown)
                      for t in tagger.analyze(text)]
            assert actual == reference_analyze(patterns, text)


def test_one_tagger_can_be_shared_by_multiple_threads(tmp_path):
    tagger = make_tagger(tmp_path, [
        pattern("甲乙", VERB, shift=1), pattern("乙"), pattern("𠮷"),
        pattern("猫"), pattern("甲", PARTICLE),
    ])
    texts = [text * 20 for text in ("甲乙𠮷123abc猫", "猫甲乙🙂", "𠮷abc　猫", "")]
    expected = {text: (tagger.analyze(text), tagger.parse(text), tagger.count(text)) for text in texts}

    def run(text):
        return text, (tagger.analyze(text), tagger.parse(text), tagger.count(text))

    with ThreadPoolExecutor(max_workers=4) as pool:
        for text, result in pool.map(run, texts * 16):
            assert result == expected[text]
