#!/usr/bin/env python3
"""Fetch pinned public resources and independently convert them for dagger.

No Jagger source or conversion script is used. Only standard-library modules
are needed. The original KWDLC text is downloaded locally, never bundled here.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

KWDLC_COMMIT = "c6ae49d29eca4e1134c6676d682adf5ed6a25f5a"
KWDLC_URL = f"https://codeload.github.com/ku-nlp/KWDLC/tar.gz/{KWDLC_COMMIT}"
KWDLC_SHA256 = "0aec50c1ae4e01258567cdd5edbc794e9885cd776603b05d7da5462dcda6b81f"
DICTIONARY_URL = (
    "https://deb.debian.org/debian/pool/main/m/mecab-jumandic/"
    "mecab-jumandic_7.0-20130310.orig.tar.gz"
)
DICTIONARY_SHA256 = "eaf216758edee9a159bc3d02507007318686b9537943268c4565cc1f9ef07f15"
SID = re.compile(r"^#\s*S-ID:(\S+)")
CHUNK = re.compile(r"^[*+]\s+-?\d+[A-Z](?:\s|$)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path, expected_sha256: str) -> Path:
    """Download atomically, accepting a cached file only after verification."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != expected_sha256:
            raise ValueError(f"SHA256 mismatch for cached resource: {target}")
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "dagger-resource-preparation/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        digest = sha256(temporary)
        if digest != expected_sha256:
            raise ValueError(f"SHA256 mismatch for {url}: {digest}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def extract(archive: Path, destination: Path) -> Path:
    """Extract regular files/directories only and reject paths outside root."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            resolved = (root / member.name).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"Archive path escapes extraction directory: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
        # Entries were validated above. Using explicit copies also works on
        # Python versions predating tarfile's data extraction filter.
        for member in members:
            target = root / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
    directories = sorted(path for path in destination.iterdir() if path.is_dir())
    if len(directories) != 1:
        raise ValueError(f"Expected one top-level directory in {archive}")
    return directories[0]


def read_ids(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate document ID: {path}")
    return set(values)


def convert_corpus(corpus: Path, output: Path) -> dict:
    split_root = corpus / "id" / "split_for_pas"
    split_ids = {split: read_ids(split_root / f"{split}.id") for split in ("train", "dev", "test")}
    id_to_split: dict[str, str] = {}
    for split, identifiers in split_ids.items():
        for identifier in identifiers:
            if identifier in id_to_split:
                raise ValueError(f"Document appears in multiple splits: {identifier}")
            id_to_split[identifier] = split

    counts = {split: {"documents": 0, "sentences": 0, "tokens": 0} for split in split_ids}
    seen_documents: set[str] = set()
    seen_sentences: set[str] = set()
    files = sorted(corpus.glob("knp/*/*.knp"))
    if not files:
        raise ValueError(f"No KWDLC knp/*/*.knp files found: {corpus}")
    output.mkdir(parents=True, exist_ok=True)
    input_digest = hashlib.sha256()
    with contextlib.ExitStack() as stack:
        tagged = {split: stack.enter_context((output / f"{split}.mecab").open("w", encoding="utf-8", newline="\n")) for split in split_ids}
        plain = {split: stack.enter_context((output / f"{split}.txt").open("w", encoding="utf-8", newline="\n")) for split in split_ids}
        for path in files:
            raw = path.read_bytes()
            input_digest.update(path.relative_to(corpus).as_posix().encode("utf-8") + b"\0" + raw + b"\0")
            split = None
            sentence = None
            surfaces: list[str] = []
            for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                match = SID.match(line)
                if match:
                    if sentence is not None:
                        raise ValueError(f"{path}:{number}: new sentence before EOS")
                    sentence = match.group(1)
                    document = path.stem
                    if not sentence.startswith(document + "-"):
                        raise ValueError(f"{path}:{number}: S-ID does not match its document filename")
                    if document not in id_to_split:
                        raise ValueError(f"{path}:{number}: document missing from standard split: {document}")
                    if sentence in seen_sentences:
                        raise ValueError(f"Duplicate sentence ID: {sentence}")
                    seen_sentences.add(sentence)
                    split = id_to_split[document]
                    if document not in seen_documents:
                        seen_documents.add(document)
                        counts[split]["documents"] += 1
                    continue
                if not line or line.startswith("#") or CHUNK.match(line):
                    continue
                if line == "EOS":
                    if sentence is None or not surfaces:
                        raise ValueError(f"{path}:{number}: EOS without a nonempty sentence")
                    tagged[split].write("EOS\n")
                    plain[split].write("".join(surfaces) + "\n")
                    counts[split]["sentences"] += 1
                    sentence, surfaces = None, []
                    continue
                if sentence is None:
                    raise ValueError(f"{path}:{number}: token before S-ID")
                fields = line.split()
                if len(fields) < 11:
                    raise ValueError(f"{path}:{number}: invalid JUMAN token line")
                surface, reading, lemma = fields[:3]
                pos = [fields[index] for index in (3, 5, 7, 9)]
                # Same tag convention as Jagger/JUMAN++ training: undefined
                # words receive the ordinary-noun tag, not a separate POS.
                if pos[0] == "未定義語":
                    pos = ["名詞", "普通名詞", "*", "*"]
                features = ",".join(pos + [lemma, reading, "*"])
                tagged[split].write(surface + "\t" + features + "\n")
                surfaces.append(surface)
                counts[split]["tokens"] += 1
            if sentence is not None:
                raise ValueError(f"{path}: final sentence has no EOS")
    missing = set(id_to_split) - seen_documents
    if missing:
        raise ValueError(f"Missing KWDLC documents: {sorted(missing)[:5]}")
    return {"counts": counts, "knp_content_sha256": input_digest.hexdigest()}


def convert_dictionary(dictionary: Path, output: Path, encoding: str = "utf-8") -> dict:
    files = sorted(dictionary.glob("*.csv")) if dictionary.is_dir() else [dictionary]
    if not files:
        raise ValueError(f"No dictionary CSV files found: {dictionary}")
    rows = corrections = 0
    replacements = {"で\ufffd": "です", "ま\ufffd": "まい", "こと\ufffd": "ことだ"}
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for path in files:
            # The original AuxV file contains invalid UTF-8 byte sequences,
            # not literal U+FFFD characters. Decode those to a sentinel first;
            # only the explicitly recognized six entries below may repair it.
            with path.open(encoding=encoding, errors="replace", newline="") as source:
                for number, fields in enumerate(csv.reader(source), 1):
                    if not fields:
                        continue
                    if len(fields) != 11 or not fields[0]:
                        raise ValueError(f"{path}:{number}: expected 11-column MeCab-JUMAN CSV")
                    if any(any(char in value for char in "\r\n\t\x00") for value in fields):
                        raise ValueError(f"{path}:{number}: embedded control character")
                    # Correct six known AuxV rows without importing an upstream
                    # patch or modifying any downloaded file.
                    if fields[0] in replacements and fields[4] == "助動詞":
                        original = fields[0]
                        if fields[8] != original or fields[9] != original:
                            raise ValueError(f"Unexpected damaged AuxV entry: {path}:{number}")
                        fields[0] = fields[8] = fields[9] = replacements[original]
                        corrections += 1
                    if any("\ufffd" in value for value in fields):
                        raise ValueError(f"Unrecognized replacement character: {path}:{number}")
                    writer.writerow(fields)
                    rows += 1
    return {"rows": rows, "corrected_auxiliary_rows": corrections, "sha256": sha256(output)}


def local_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--kwdlc", type=Path, help="Existing KWDLC directory (downloads pinned commit if omitted)")
    parser.add_argument("--dictionary", type=Path, help="Existing MeCab-JUMAN CSV directory or file")
    parser.add_argument("--dictionary-encoding", default="utf-8", help="Encoding of a local dictionary, e.g. euc_jp")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.output / ".sources"
    corpus = args.kwdlc
    if corpus is None:
        archive = fetch(KWDLC_URL, source_dir / f"KWDLC-{KWDLC_COMMIT}.tar.gz", KWDLC_SHA256)
        corpus = extract(archive, source_dir / "kwdlc")
    dictionary = args.dictionary
    if dictionary is None:
        archive = fetch(DICTIONARY_URL, source_dir / "mecab-jumandic-7.0-20130310.tar.gz", DICTIONARY_SHA256)
        dictionary = extract(archive, source_dir / "dictionary")

    # Complete and validate all outputs before replacing existing prepared data.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=args.output) as temp:
        stage = Path(temp)
        corpus_metadata = convert_corpus(corpus, stage)
        dictionary_metadata = convert_dictionary(dictionary, stage / "dictionary.csv", args.dictionary_encoding)
        metadata = {
            "kwdlc": {
                "repository": "https://github.com/ku-nlp/KWDLC",
                "commit": local_commit(corpus) if args.kwdlc else KWDLC_COMMIT,
                "downloaded_archive_sha256": KWDLC_SHA256 if args.kwdlc is None else None,
                "split": "id/split_for_pas",
                **corpus_metadata,
            },
            "dictionary": {
                "name": "MeCab-JUMAN 7.0-20130310" if args.dictionary is None else "local MeCab-JUMAN dictionary",
                "source_url": DICTIONARY_URL if args.dictionary is None else None,
                "downloaded_archive_sha256": DICTIONARY_SHA256 if args.dictionary is None else None,
                **dictionary_metadata,
            },
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(stage.iterdir())},
        }
        (stage / "sources.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for path in sorted(stage.iterdir()):
            path.replace(args.output / path.name)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
