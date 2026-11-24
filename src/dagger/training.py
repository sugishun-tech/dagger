"""Model training from MeCab-style annotated text and CSV dictionaries."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from ._training import extract_patterns

__all__ = ["extract_patterns", "train"]


def train(
    corpus_path,
    model_path,
    dictionary_path=None,
    max_pattern_chars=None,
    prune=True,
    min_count=1,
):
    """Train and write a model; return reproducibility metadata.

    Only the specified corpus is read. Development and test corpora are never
    discovered or merged automatically.
    """
    from .corpus import read_dictionary, read_mecab
    from .model import compile_model

    started = perf_counter()
    sentences = list(read_mecab(corpus_path))
    dictionary = read_dictionary(dictionary_path) if dictionary_path is not None else ()
    patterns, stats = extract_patterns(
        sentences,
        dictionary=dictionary,
        max_pattern_chars=max_pattern_chars,
        prune=prune,
        min_count=min_count,
    )
    stats["corpus_file"] = Path(corpus_path).name
    stats["dictionary_file"] = Path(dictionary_path).name if dictionary_path is not None else None
    stats["extraction_seconds"] = perf_counter() - started
    compile_model(patterns, model_path, metadata=stats)
    stats["training_seconds"] = perf_counter() - started
    return stats
