"""Malformed model files must raise, never enter unchecked UTF-8 pointer scans."""
import hashlib
import json
from pathlib import Path

import pytest

from dagger import Tagger
from dagger.model import HEADER, MAGIC, compile_model


def make_model(tmp_path):
    path = tmp_path / "test.dgr"
    compile_model([
        ("食べた", None, 3, "動詞,*,*,*", 1, "る", "タベタ"),
        ("𠮷", None, 1, "名詞,*,*,*", 0, "", "ヨシ"),
    ], path)
    return path


def mutate(path, change):
    raw = path.read_bytes()
    _, size, node_size, _ = HEADER.unpack_from(raw)
    metadata = json.loads(raw[64:64 + size])
    nodes = raw[64 + size + (-size) % 64:]
    change(metadata)
    head = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    body = head + bytes((-len(head)) % 64) + nodes
    path.write_bytes(HEADER.pack(MAGIC, len(head), node_size,
                                 hashlib.sha256(body).digest()) + body)


def test_roundtrip_and_metadata_are_detached(tmp_path):
    tagger = Tagger(make_model(tmp_path))
    metadata = tagger.metadata
    metadata["pos"][1] = "altered"
    assert tagger.analyze("食べた")[0].lemma == "食べる"
    assert tagger.metadata != metadata


def test_truncation_and_checksum_are_detected(tmp_path):
    path = make_model(tmp_path)
    original = path.read_bytes()
    path.write_bytes(original[:20])
    with pytest.raises(ValueError):
        Tagger(path)
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        Tagger(path)


@pytest.mark.parametrize("field,value", [(1, 2), (4, 1), (0, 2)])
def test_rehashed_invalid_utf8_boundaries_are_rejected(tmp_path, field, value):
    path = make_model(tmp_path)
    def change(m):
        label = next(row for row in m["labels"] if row[0] == 3)
        label[field] = value
    mutate(path, change)
    with pytest.raises(ValueError):
        Tagger(path)


@pytest.mark.parametrize("codepoint", [-1, 0x110000, 0xD800])
def test_rehashed_invalid_character_map_is_rejected(tmp_path, codepoint):
    path = make_model(tmp_path)
    mutate(path, lambda m: m["characters"].__setitem__(0, codepoint))
    with pytest.raises(ValueError):
        Tagger(path)


def test_rehashed_output_control_character_is_rejected(tmp_path):
    path = make_model(tmp_path)
    mutate(path, lambda m: m["pos"].__setitem__(1, "名詞\nEOS,*,*,*"))
    with pytest.raises(ValueError):
        Tagger(path)


def test_duplicate_pattern_conflicts_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Conflicting"):
        compile_model([
            ("甲", None, 1, "N,*,*,*", 0, "", "A"),
            ("甲", None, 1, "V,*,*,*", 0, "", "B"),
        ], tmp_path / "invalid.dgr")


@pytest.mark.parametrize("entries", [
    [((2,), 0), ((1,), 1)],
    [((1, 2), 0), ((1,), 1)],
    [((1,), 0), ((1,), 1)],
    [((0,), 0)],
    [((1,), -1)],
])
def test_low_level_trie_builder_rejects_unsafe_keys(entries):
    from dagger._core import build_double_array
    with pytest.raises(ValueError):
        build_double_array(entries, 3)


def test_control_characters_only_rejected_by_serialized_output_api(tmp_path):
    tagger = Tagger(make_model(tmp_path))
    for text in ("a\nb", "a\tb", "a\0b", "a\rb"):
        assert "".join(tagger.tokenize(text)) == text
        with pytest.raises(ValueError):
            tagger.parse(text)
    with pytest.raises(ValueError):
        tagger.parse_batch_bytes(b"a\tb\n")

