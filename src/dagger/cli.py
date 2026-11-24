"""Small command line front end; sentence batches stay inside Cython."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":
        return train_main(argv[1:])
    parser = argparse.ArgumentParser(prog="dagger", description="Jagger-style Cython morphological analyzer")
    parser.add_argument("--model", "-m", required=True, help="Compiled .dgr model")
    parser.add_argument("--wakati", "-w", action="store_true", help="Space-separated surfaces")
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024,
                        help="Approximate byte batch size (default 1 MiB)")
    args = parser.parse_args(argv)
    if args.chunk_bytes < 1:
        parser.error("--chunk-bytes must be positive")
    from . import Tagger
    try:
        tagger = Tagger(args.model)
        source = sys.stdin.buffer
        destination = sys.stdout.buffer
        pending = b""
        while True:
            chunk = source.read(args.chunk_bytes)
            if not chunk:
                break
            pending += chunk
            last = pending.rfind(b"\n")
            if last >= 0:
                destination.write(tagger.parse_batch_bytes(pending[:last + 1], args.wakati))
                pending = pending[last + 1:]
        if pending:
            destination.write(tagger.parse_batch_bytes(pending, args.wakati))
        destination.flush()
    except BrokenPipeError:
        # Avoid a second error from interpreter shutdown flushing stdout.
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(2, f"dagger: {exc}\n")
    return 0


def train_main(argv):
    parser = argparse.ArgumentParser(prog="dagger train")
    parser.add_argument("corpus", help="UTF-8 MeCab-JUMAN annotated corpus")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--dictionary", "-d")
    parser.add_argument("--max-pattern-chars", type=int)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--no-prune", action="store_true")
    args = parser.parse_args(argv)
    from .training import train
    try:
        stats = train(args.corpus, args.model, dictionary_path=args.dictionary,
                      max_pattern_chars=args.max_pattern_chars,
                      prune=not args.no_prune, min_count=args.min_count)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"dagger train: {exc}\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0

