"""Strict readers for JUMAN-style MeCab training data and dictionaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

Entry = tuple[str, str, str, str]  # surface, four-level POS, lemma, reading


def read_mecab(path: str | Path) -> Iterator[list[Entry]]:
    """Yield sentences from ``surface<TAB>pos1,pos2,type,form,lemma,reading,*``.

    ``EOS`` ends a sentence. A final sentence without EOS is accepted. Blank
    lines and comment lines beginning with ``# `` are ignored. Surfaces are
    preserved exactly; whitespace is not silently stripped from them.
    """
    path = Path(path)
    sentence: list[Entry] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.rstrip("\r\n")
            if line == "EOS":
                if sentence:
                    yield sentence
                    sentence = []
                continue
            if not line or (line.startswith("# ") and "\t" not in line):
                continue
            surface, separator, features = line.partition("\t")
            fields = features.split(",")
            if not separator or not surface or len(fields) < 6:
                raise ValueError(
                    f"{path}:{number}: expected a surface, TAB, four POS fields, "
                    "lemma and reading"
                )
            if "\t" in features or "\x00" in line:
                raise ValueError(f"{path}:{number}: embedded TAB or NUL is not supported")
            yield_entry = (surface, ",".join(fields[:4]), fields[4], fields[5])
            sentence.append(yield_entry)
    if sentence:
        yield sentence


def read_dictionary(path: str | Path) -> Iterator[Entry]:
    """Yield entries from a UTF-8 MeCab-JUMAN CSV file or directory.

    The first four columns are surface, left ID, right ID and cost. The next
    columns are the four POS fields, lemma and reading. CSV quoting is honored.
    A directory is read in filename order to make tie breaking reproducible.
    """
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"{path}: no dictionary CSV files")
    for source in files:
        with source.open(encoding="utf-8-sig", newline="") as stream:
            for number, fields in enumerate(csv.reader(stream), 1):
                if not fields:
                    continue
                if len(fields) < 10 or not fields[0]:
                    raise ValueError(f"{source}:{number}: expected MeCab-JUMAN CSV columns")
                if any(any(char in value for char in "\r\n\t\x00") for value in fields):
                    raise ValueError(f"{source}:{number}: embedded control character")
                yield fields[0], ",".join(fields[4:8]), fields[8], fields[9]
