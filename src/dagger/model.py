"""Portable, checked model container; inference does not use pickle or NumPy."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import tempfile

MAGIC = b"DAGGER1\0"
HEADER = struct.Struct("<8sQQ32s8x")
UNKNOWN_POS = (
    "名詞,普通名詞,*,*",
    "名詞,数詞,*,*",
    "特殊,記号,*,*",
    "特殊,空白,*,*",
)


def compile_model(patterns, model_path, metadata=None):
    """Compile (surface, previous POS, shift, POS, strip, suffix, reading)."""
    from ._core import build_double_array, classify_word

    if sys.byteorder != "little":
        raise RuntimeError("This release supports little-endian hosts only")
    patterns = list(patterns)
    if not patterns:
        raise ValueError("No patterns: supply a nonempty corpus or dictionary")
    frequency = Counter()
    positions = {""}
    for surface, previous, shift, pos, strip, suffix, reading in patterns:
        if (not isinstance(surface, str) or not surface or type(shift) is not int
                or not 0 < shift <= len(surface)):
            raise ValueError("Each pattern needs a nonempty surface and valid shift")
        if not isinstance(pos, str) or len(pos.split(",")) != 4:
            raise ValueError("POS must have exactly four comma-separated fields")
        if type(strip) is not int or not 0 <= strip <= shift:
            raise ValueError("Invalid lemma strip length")
        if previous is not None and (not isinstance(previous, str) or
                (previous != "" and len(previous.split(",")) != 4) or
                any(c in previous for c in "\n\r\t\0")):
            raise ValueError("Previous POS must be None, BOS or a four-field POS")
        if any(c in surface for c in "\n\r\t\0"):
            raise ValueError("Pattern surfaces must not contain line delimiters or NUL")
        for value in (pos, suffix, reading):
            if not isinstance(value, str) or any(c in value for c in "\n\r\t\0"):
                raise ValueError("Invalid output field")
        surface.encode("utf-8")
        frequency.update(surface)
        positions.add(pos)
        if previous is not None:
            positions.add(previous)
    positions.update(UNKNOWN_POS)
    pos_names = [""] + sorted(positions - {""})
    pos_ids = {name: i for i, name in enumerate(pos_names)}
    characters = sorted(frequency, key=lambda c: (-frequency[c], ord(c)))
    char_ids = {c: i + 1 for i, c in enumerate(characters)}
    context_offset = len(characters) + 1

    labels = []
    label_ids = {}

    def label_id(label):
        if label not in label_ids:
            label_ids[label] = len(labels)
            labels.append(label)
        return label_ids[label]

    entries = {}
    for surface, previous, shift, pos, strip, suffix, reading in patterns:
        word = surface[:shift]
        encoded_word = word.encode("utf-8")
        byte_shift = len(encoded_word)
        byte_strip = len(word[len(word) - strip:].encode("utf-8")) if strip else 0
        label = (shift, byte_shift, pos_ids[pos], strip, byte_strip, suffix, reading,
                 classify_word(encoded_word))
        key = tuple(char_ids[c] for c in surface)
        if previous is not None:
            key += (context_offset + pos_ids[previous],)
        value = label_id(label)
        if key in entries and entries[key] != value:
            raise ValueError("Conflicting duplicate pattern")
        entries[key] = value
    unknown_labels = [label_id((0, 0, pos_ids[pos], 0, 0, "", "*", kind))
                      for pos, kind in zip(UNKNOWN_POS, (0, 1, 5, 4))]
    ordered = sorted(entries.items())
    del entries, patterns
    nodes = build_double_array(ordered, len(characters))
    manifest = {
        "format": 1, "node_width": 4, "node_count": len(nodes) // 16,
        "characters": [ord(c) for c in characters], "pos": pos_names,
        "labels": labels, "unknown_labels": unknown_labels,
        "context_offset": context_offset, "patterns": len(ordered),
        "training": metadata or {},
    }
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    padding = bytes((-len(raw)) % 64)
    body_hash = sha256(raw + padding + nodes).digest()
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(HEADER.pack(MAGIC, len(raw), len(nodes), body_hash))
            handle.write(raw)
            handle.write(padding)
            handle.write(nodes)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return manifest


class Model:
    """Immutable mmap-backed model. Cython checks all indices again when loading."""

    def __init__(self, path):
        from ._core import Engine

        if sys.byteorder != "little":
            raise RuntimeError("This release supports little-endian hosts only")
        self.path = str(path)
        with open(path, "rb") as handle:
            if os.fstat(handle.fileno()).st_size < HEADER.size:
                raise ValueError("Truncated dagger model")
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            magic, json_size, nodes_size, digest = HEADER.unpack_from(mapping)
            offset = HEADER.size + json_size + (-json_size) % 64
            if magic != MAGIC or offset + nodes_size != len(mapping) or nodes_size % 16:
                raise ValueError("Invalid dagger model header")
            view = memoryview(mapping)
            actual = sha256(view[HEADER.size:]).digest()
            view.release()
            if actual != digest:
                raise ValueError("Model checksum mismatch")
            manifest = json.loads(mapping[HEADER.size:HEADER.size + json_size])
            data = memoryview(mapping)[offset:offset + nodes_size].cast("i")
            engine = Engine(data, manifest)
        except Exception:
            # A failed extension constructor may still be retained in the
            # exception traceback. Let its exported views release the mmap
            # when that traceback dies; never hide the original validation error.
            try:
                mapping.close()
            except BufferError:
                pass
            raise
        self.manifest = manifest
        self._mapping = mapping
        self._view = data
        self.engine = engine


def _validate_manifest(m, nodes_size):
    try:
        if m["format"] != 1 or m["node_width"] != 4 or m["node_count"] != nodes_size // 16:
            raise ValueError("Invalid model dimensions")
        if not 2 <= m["node_count"] < 2**31:
            raise ValueError("Invalid trie size")
        chars, positions, labels = m["characters"], m["pos"], m["labels"]
        if not chars or len(set(chars)) != len(chars):
            raise ValueError("Invalid character map")
        if any(type(c) is not int or not 0 <= c <= 0x10FFFF or 0xD800 <= c <= 0xDFFF for c in chars):
            raise ValueError("Invalid Unicode character")
        if m["context_offset"] != len(chars) + 1 or len(chars) + len(positions) >= 2**30:
            raise ValueError("Invalid symbol range")
        if not positions or positions[0] != "" or len(set(positions)) != len(positions):
            raise ValueError("Invalid POS table")
        for pos in positions[1:]:
            if (not isinstance(pos, str) or len(pos.split(",")) != 4 or
                    any(c in pos for c in "\n\r\t\0")):
                raise ValueError("Invalid POS entry")
            pos.encode("utf-8")
        for label in labels:
            if len(label) != 8:
                raise ValueError("Invalid label")
            shift, byte_shift, pos, strip, byte_strip, suffix, reading, kind = label
            if any(type(x) is not int for x in label[:5]):
                raise ValueError("Invalid numeric label")
            if not (0 <= shift <= byte_shift < 2**31 and 0 <= strip <= shift and
                    strip <= byte_strip <= byte_shift and 0 <= pos < len(positions)):
                raise ValueError("Invalid label indices")
            if type(kind) is not int or not 0 <= kind <= 5:
                raise ValueError("Invalid character-kind label")
            for value in (suffix, reading):
                if not isinstance(value, str) or any(c in value for c in "\n\r\t\0"):
                    raise ValueError("Invalid output field")
                value.encode("utf-8")
        if len(m["unknown_labels"]) != 4:
            raise ValueError("Invalid fallback table")
        for value in m["unknown_labels"]:
            if type(value) is not int or not 0 <= value < len(labels) or labels[value][0] != 0:
                raise ValueError("Invalid fallback label")
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Invalid dagger model metadata") from exc

