"""Specification-level checks of pattern learning and lossless pruning."""

from itertools import product
import random

import pytest

from dagger._training import extract_patterns


def token(surface, pos="N", lemma=None, reading="*"):
    return surface, pos, surface if lemma is None else lemma, reading


def lookup(patterns, text, previous):
    applicable = [
        pattern
        for pattern in patterns
        if text.startswith(pattern[0])
        and pattern[1] in (None, previous)
    ]
    if not applicable:
        return None
    match = max(applicable, key=lambda p: (len(p[0]), p[1] is not None))
    return match[2:]


def test_right_context_is_not_consumed():
    sentences = [[token("の", "A")]] * 3 + [
        [token("の", "P"), token("な", "ADJ")]
    ]
    patterns, _ = extract_patterns(sentences, max_pattern_chars=2)
    assert lookup(patterns, "の", "")[1] == "A"
    assert lookup(patterns, "のな", "")[:2] == (1, "P")
    assert any(p[0] == "のな" and p[2] == 1 for p in patterns)


def test_segmentation_is_marginalized_before_tag_selection():
    sentences = (
        [[token("甲", "A"), token("乙", "Z")]] * 4
        + [[token("甲", "B"), token("乙", "Z")]] * 4
        + [[token("甲乙", "C")]] * 7
    )
    patterns, _ = extract_patterns(sentences, max_pattern_chars=2, prune=False)
    # A joint argmax would incorrectly select (2, C), whose count is 7.
    # The shift=1 count is 4+4=8, so shift=1 wins; A wins the POS tie.
    assert lookup(patterns, "甲乙", "")[:2] == (1, "A")


def test_pos_is_marginalized_over_reading_and_lemma():
    sentences = (
        [[token("甲", "A", reading="イ")]] * 3
        + [[token("甲", "A", reading="ロ")]] * 3
        + [[token("甲", "B", reading="ハ")]] * 5
    )
    patterns, _ = extract_patterns(sentences)
    assert lookup(patterns, "甲", "")[1:] == ("A", 0, "", "イ")


def test_shift_tie_prefers_the_longer_word():
    sentences = [
        [token("甲", "N"), token("乙", "N")],
        [token("甲乙", "N")],
    ]
    patterns, _ = extract_patterns(sentences, max_pattern_chars=2)
    assert lookup(patterns, "甲乙", "")[0] == 2


def test_previous_pos_and_bos_are_separate_features():
    sentences = [
        [token("前", "P"), token("甲", "A")],
        [token("後", "Q"), token("甲", "B")],
        [token("甲", "C")],
    ]
    patterns, _ = extract_patterns(sentences, max_pattern_chars=2)
    assert lookup(patterns, "甲", "P")[1] == "A"
    assert lookup(patterns, "甲", "Q")[1] == "B"
    assert lookup(patterns, "甲", "")[1] == "C"


def test_dictionary_unseen_tag_uses_corpus_prior():
    sentences = [[token("甲", "N"), token("乙", "N"), token("食", "V")]]
    dictionary = [token("未知", "V"), token("未知", "N")]
    patterns, stats = extract_patterns(sentences, dictionary)
    assert lookup(patterns, "未知", "")[1] == "N"
    assert stats["dictionary_entries"] == 2


def test_a_longer_surface_overrides_shorter_pos_context():
    sentences = [
        [token("前", "P"), token("甲", "A")],
        [token("甲乙", "B")],
    ]
    patterns, _ = extract_patterns(sentences, max_pattern_chars=2)
    assert lookup(patterns, "甲", "P")[1] == "A"
    assert lookup(patterns, "甲乙", "P")[:2] == (2, "B")


def test_pruning_keeps_a_generic_pattern_that_resets_previous_pos():
    sentences = (
        [[token("前", "P"), token("甲", "Y"), token("丙")]] * 20
        + [[token("前", "P"), token("甲", "X"), token("乙")]] * 10
        + [[token("後", "Q"), token("甲", "X"), token("丁")]] * 50
    )
    full, _ = extract_patterns(sentences, max_pattern_chars=2, prune=False)
    pruned, _ = extract_patterns(sentences, max_pattern_chars=2, prune=True)
    assert lookup(full, "甲", "P")[1] == "Y"
    assert lookup(full, "甲乙", "P")[1] == "X"
    assert lookup(pruned, "甲乙", "P") == lookup(full, "甲乙", "P")
    assert len(pruned) < len(full)


def test_pruning_preserves_all_predictions_on_small_random_languages():
    rng = random.Random(204)
    alphabet = "甲乙丙"
    queries = ["".join(chars) for n in range(1, 4) for chars in product(alphabet, repeat=n)]
    for _ in range(12):
        sentences = []
        for _ in range(25):
            sentence = []
            for _ in range(rng.randint(1, 4)):
                word = "".join(rng.choices(alphabet, k=rng.randint(1, 2)))
                sentence.append(token(word, rng.choice(["N", "V", "P"])))
            sentences.append(sentence)
        full, _ = extract_patterns(sentences, max_pattern_chars=3, prune=False)
        pruned, _ = extract_patterns(sentences, max_pattern_chars=3, prune=True)
        for query in queries:
            for previous in ("", "N", "V", "P", "UNSEEN"):
                assert lookup(pruned, query, previous) == lookup(full, query, previous)


def test_character_lengths_lemma_edit_and_long_words():
    sentences = [[token("𠮷野家", "N"), token("食べた", "V", "食べる", "タベタ")]]
    patterns, stats = extract_patterns(sentences, max_pattern_chars=1)
    assert lookup(patterns, "𠮷野家", "")[0] == 3
    assert lookup(patterns, "食べた", "N") == (3, "V", 1, "る", "タベタ")
    assert stats["max_pattern_chars"] == 1
    assert stats["corpus_max_word_chars"] == 3


def test_dictionary_only_training_and_empty_training():
    patterns, _ = extract_patterns([], [token("単語", "N")])
    assert lookup(patterns, "単語", "")[:2] == (2, "N")
    assert extract_patterns([])[0] == []


def test_output_does_not_depend_on_corpus_iteration_order():
    sentences = [
        [token("甲", "N"), token("乙", "P")],
        [token("甲乙", "V")],
        [token("甲", "N"), token("丙", "P")],
    ]
    patterns, _ = extract_patterns(sentences, max_pattern_chars=3)
    reversed_patterns, _ = extract_patterns(list(reversed(sentences)), max_pattern_chars=3)
    assert patterns == reversed_patterns


@pytest.mark.parametrize("kwargs", [{"min_count": 0}, {"max_pattern_chars": 0}, {"max_pattern_chars": True}])
def test_invalid_learning_options(kwargs):
    with pytest.raises(ValueError):
        extract_patterns([], **kwargs)


@pytest.mark.parametrize("row", [("", "N", "", "*"), ("甲", "", "甲", "*"), ("甲", "N"), ("甲", "N", 7, "*")])
def test_invalid_token_records(row):
    with pytest.raises((ValueError, TypeError)):
        extract_patterns([[row]])
