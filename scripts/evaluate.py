#!/usr/bin/env python3
"""Evaluate exact token spans. This script also accepts another analyzer's output."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# surface, four-level POS, lemma, reading; offsets are Unicode code points.
Morph = tuple[str, str, str, str]


def read_mecab_lines(lines: Iterable[str]) -> Iterator[list[Morph]]:
    """Read the JUMAN/MeCab feature layout used by this project.

    Each line is surface TAB pos,subpos,conjugation,form,lemma,reading[,extra].
    EOS terminates one sentence, including an empty sentence. A final nonempty
    sentence without EOS is accepted; blank physical lines are ignored.
    """
    sentence: list[Morph] = []
    for number, raw in enumerate(lines, 1):
        line = raw.rstrip("\r\n")
        if line == "EOS":
            yield sentence
            sentence = []
        elif not line:
            continue
        else:
            try:
                surface, feature_string = line.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"line {number}: expected surface TAB features") from exc
            features = feature_string.split(",")
            if not surface or len(features) < 5:
                raise ValueError(f"line {number}: empty surface or fewer than five features")
            sentence.append((surface, ",".join(features[:4]), features[4],
                             features[5] if len(features) > 5 else "*"))
    if sentence:
        yield sentence


def read_mecab(path: str | Path) -> list[list[Morph]]:
    with Path(path).open(encoding="utf-8") as stream:
        return list(read_mecab_lines(stream))


def _span_sets(sentence: Sequence[Morph]) -> dict[str, set[tuple]]:
    spans: dict[str, set[tuple]] = {
        "segmentation": set(), "pos1": set(), "full_pos": set(), "lemma": set(),
    }
    start = 0
    for surface, pos, lemma, _reading in sentence:
        if not surface:
            raise ValueError("empty tokens have no valid span")
        end = start + len(surface)
        spans["segmentation"].add((start, end))
        spans["pos1"].add((start, end, pos.split(",", 1)[0]))
        spans["full_pos"].add((start, end, pos))
        spans["lemma"].add((start, end, lemma))
        start = end
    return spans


def _metric(correct: int, predicted: int, gold: int) -> dict:
    # An empty corpus has perfect agreement. Empty vs nonempty has zero F1.
    precision = correct / predicted if predicted else float(gold == 0)
    recall = correct / gold if gold else float(predicted == 0)
    f1 = 2 * correct / (predicted + gold) if predicted + gold else 1.0
    return {"correct": correct, "predicted": predicted, "gold": gold,
            "precision": precision, "recall": recall, "f1": f1}


def evaluate_corpora(gold: Sequence[Sequence[Morph]],
                     predicted: Sequence[Sequence[Morph]]) -> dict:
    """Micro P/R/F1 for exact token spans, with strict text reconstruction.

    ``lemma`` requires the same span and lemma, independently of POS. It is not
    accuracy on oracle gold tokens. No sentence, whitespace, or unknown token
    is silently removed from the denominator.
    """
    if len(gold) != len(predicted):
        raise ValueError(f"sentence counts differ: gold={len(gold)}, predicted={len(predicted)}")
    correct: Counter = Counter()
    predicted_count = 0
    gold_count = 0
    exact_sentences: Counter = Counter()
    for index, (expected, actual) in enumerate(zip(gold, predicted), 1):
        expected_text = "".join(token[0] for token in expected)
        actual_text = "".join(token[0] for token in actual)
        if expected_text != actual_text:
            raise ValueError(f"sentence {index}: predictions do not reconstruct the gold text")
        e = _span_sets(expected)
        a = _span_sets(actual)
        gold_count += len(expected)
        predicted_count += len(actual)
        for key in e:
            correct[key] += len(e[key] & a[key])
            exact_sentences[key] += e[key] == a[key]
    names = ("segmentation", "pos1", "full_pos", "lemma")
    return {
        "sentences": len(gold),
        "gold_tokens": gold_count,
        "predicted_tokens": predicted_count,
        "offset_unit": "Unicode code point",
        "averaging": "micro, exact token span",
        "metrics": {name: _metric(correct[name], predicted_count, gold_count)
                    for name in names},
        "exact_sentences": {name: exact_sentences[name] for name in names},
    }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred", type=Path, help="analyzer output in MeCab/JUMAN layout")
    source.add_argument("--model", type=Path, help="dagger model to evaluate")
    parser.add_argument("--output", type=Path, help="write JSON in addition to stdout")
    args = parser.parse_args(argv)
    gold = read_mecab(args.gold)
    if args.pred:
        predicted = read_mecab(args.pred)
    else:
        from dagger import Tagger
        tagger = Tagger(args.model)
        predicted = [[(t.surface, t.pos, t.lemma, t.reading)
                      for t in tagger.analyze("".join(m[0] for m in sentence))]
                     for sentence in gold]
    result = evaluate_corpora(gold, predicted)
    result["source"] = {"gold": str(args.gold), "gold_sha256": sha256(args.gold),
                        "prediction": str(args.pred or args.model),
                        "prediction_sha256": sha256(args.pred or args.model)}
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
