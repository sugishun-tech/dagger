import importlib.util
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("dagger_evaluate", _path)
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


def token(surface, pos="名詞,普通名詞,*,*", lemma=None):
    return surface, pos, surface if lemma is None else lemma, "*"


def test_exact_span_not_only_boundary():
    gold = [[token("東京"), token("大学"), token("へ")]]
    predicted = [[token("東京大学"), token("へ")]]
    result = evaluation.evaluate_corpora(gold, predicted)
    metric = result["metrics"]["segmentation"]
    assert metric == {"correct": 1, "predicted": 2, "gold": 3,
                      "precision": 0.5, "recall": 1 / 3, "f1": 0.4}


def test_pos_hierarchy_and_lemma_are_scored_separately():
    gold = [[token("読んだ", "動詞,*,子音動詞マ行,タ形", "読む")]]
    predicted = [[token("読んだ", "動詞,*,母音動詞,基本形", "読んだ")]]
    result = evaluation.evaluate_corpora(gold, predicted)["metrics"]
    assert result["segmentation"]["f1"] == 1
    assert result["pos1"]["f1"] == 1
    assert result["full_pos"]["f1"] == 0
    assert result["lemma"]["f1"] == 0


def test_micro_average_weights_tokens_not_sentences():
    gold = [[token("A")], [token("B"), token("C"), token("D")]]
    predicted = [[token("A")], [token("BCD")]]
    assert evaluation.evaluate_corpora(gold, predicted)["metrics"]["segmentation"]["f1"] == 1 / 3


def test_unicode_and_repeated_surface_positions():
    gold = [[token("𠮷"), token("🙂"), token("𠮷")]]
    assert evaluation._span_sets(gold[0])["segmentation"] == {(0, 1), (1, 2), (2, 3)}
    assert evaluation.evaluate_corpora(gold, gold)["metrics"]["full_pos"]["correct"] == 3


def test_reconstruction_mismatch_is_not_silently_ignored():
    with pytest.raises(ValueError, match="reconstruct"):
        evaluation.evaluate_corpora([[token("甲 乙")]], [[token("甲乙")]])


def test_sentence_mismatch_is_not_truncated_by_zip():
    with pytest.raises(ValueError, match="sentence counts"):
        evaluation.evaluate_corpora([[token("a")], [token("b")]], [[token("a")]])


def test_empty_sentences_and_missing_final_eos():
    parsed = list(evaluation.read_mecab_lines([
        "EOS\n", "猫\t名詞,普通名詞,*,*,猫,ねこ,*\n", "EOS\n",
        "犬\t名詞,普通名詞,*,*,犬\n",
    ]))
    assert len(parsed) == 3
    assert parsed[0] == []
    assert parsed[2][0][3] == "*"
    assert evaluation.evaluate_corpora([[]], [[]])["metrics"]["segmentation"]["f1"] == 1


def test_invalid_empty_token_rejected():
    with pytest.raises(ValueError, match="empty tokens"):
        evaluation.evaluate_corpora([[token("")]], [[token("")]])
