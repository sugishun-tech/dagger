"""Python-facing API. Use batch bytes when throughput matters."""
from __future__ import annotations

from dataclasses import dataclass
from .model import Model


@dataclass(frozen=True, slots=True)
class Token:
    surface: str
    pos: str
    lemma: str
    reading: str
    start: int
    end: int
    is_unknown: bool = False


class Tagger:
    """Immutable model; each call owns its work buffers and releases the GIL.

    Offsets returned by analyze() count Unicode code points, not UTF-8 bytes.
    Spaces are kept. Text is never normalized implicitly.
    """

    def __init__(self, model_path):
        self._model = Model(model_path)
        self._engine = self._model.engine

    @property
    def metadata(self):
        # Return a detached copy: mutation cannot alter a loaded engine.
        from copy import deepcopy
        return deepcopy(self._model.manifest)

    def analyze(self, text: str) -> list[Token]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        data = text.encode("utf-8")
        labels = self._model.manifest["labels"]
        positions = self._model.manifest["pos"]
        output = []
        offset = 0
        for begin, end, index, unknown in self._engine.spans(data):
            surface = data[begin:end].decode("utf-8")
            shift, byte_shift, pos, strip, byte_strip, suffix, reading, kind = labels[index]
            lemma = "*" if unknown else (surface[:-strip] if strip else surface) + suffix
            output.append(Token(surface, positions[pos], lemma,
                                "*" if unknown else reading,
                                offset, offset + len(surface), unknown))
            offset += len(surface)
        return output

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        data = text.encode("utf-8")
        return [data[begin:end].decode("utf-8")
                for begin, end, _, _ in self._engine.spans(data)]

    def parse(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if any(c in text for c in "\n\r\t\0"):
            raise ValueError("parse() accepts one sentence without TAB, CR, LF or NUL; use batch bytes for lines")
        return self._engine.parse(text.encode("utf-8")).decode("utf-8")

    def parse_batch_bytes(self, data: bytes, wakati: bool = False) -> bytes:
        """Analyze newline-separated UTF-8 sentences, without per-token Python objects."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if b"\t" in data or b"\0" in data:
            raise ValueError("MeCab output cannot represent TAB or NUL in a surface")
        if b"\r" in data and b"\r" in data.replace(b"\r\n", b"").removesuffix(b"\r"):
            raise ValueError("CR is only supported at a line ending")
        return self._engine.parse(data, wakati=wakati, batch=True)

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        return self._engine.count(text.encode("utf-8"))

    def count_batch_bytes(self, data: bytes) -> int:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        return self._engine.count(data, batch=True)

