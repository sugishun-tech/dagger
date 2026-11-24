#!/usr/bin/env python3
"""Reproducible CLI and warmed Python API benchmark, with honest work labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import shlex
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dagger_evaluate", ROOT / "scripts" / "evaluate.py")
_evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluation)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def describe_path(path: Path) -> dict:
    """Fingerprint model files without executing or deserializing them."""
    if path.is_file():
        return {"path": str(path), "bytes": path.stat().st_size,
                "sha256": _evaluation.sha256(path)}
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            files.append({"name": str(item.relative_to(path)), "bytes": item.stat().st_size,
                          "sha256": _evaluation.sha256(item)})
    return {"path": str(path), "bytes": sum(item["bytes"] for item in files), "files": files}


def environment() -> dict:
    cpu_name = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    result = {
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu": cpu_name, "logical_cpus": os.cpu_count(),
        "python": sys.version, "python_executable": sys.executable,
        "python_compiler": platform.python_compiler(),
        "python_config_cc": sysconfig.get_config_var("CC"),
        "python_config_cflags": sysconfig.get_config_var("CFLAGS"),
        "perf_counter_resolution_seconds": time.get_clock_info("perf_counter").resolution,
    }
    try:
        result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    except AttributeError:
        pass
    for name in ("Cython", "dagger"):
        try:
            result[name + "_version"] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name + "_version"] = None
    return result


def summarize(seconds: list[float], nbytes: int, sentences: int, tokens: int) -> dict:
    middle = statistics.median(seconds)
    return {
        "raw_seconds": seconds, "median_seconds": middle,
        "min_seconds": min(seconds), "max_seconds": max(seconds),
        "input_bytes": nbytes, "sentences": sentences, "tokens": tokens,
        "sentences_per_second": sentences / middle,
        "tokens_per_second": tokens / middle,
        "input_mib_per_second": nbytes / (1024 * 1024 * middle),
    }


def parse_output(data: bytes, wakati: bool) -> list:
    if wakati:
        # Whitespace is the delimiter in the public wakati format. Full analysis
        # provides the authoritative span/label comparison when both are run.
        lines = data.decode("utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return [line.split() for line in lines]
    return list(_evaluation.read_mecab_lines(data.decode("utf-8").splitlines()))


def inspect_output(data: bytes, wakati: bool, sentence_count: int) -> dict:
    parsed = parse_output(data, wakati)
    if len(parsed) != sentence_count:
        raise RuntimeError(f"expected {sentence_count} output sentences; got {len(parsed)}")
    return {"sha256": digest_bytes(data), "bytes": len(data),
            "sentences": len(parsed), "tokens": sum(map(len, parsed))}


def compare_outputs(a: bytes, b: bytes, wakati: bool) -> dict:
    pa, pb = parse_output(a, wakati), parse_output(b, wakati)
    result = {"bytes_identical": a == b, "sentence_counts_identical": len(pa) == len(pb),
              "token_sequences_identical": pa == pb,
              "differing_sentences": sum(x != y for x, y in zip(pa, pb)) + abs(len(pa) - len(pb))}
    if not wakati:
        result["token_sequences_identical"] = (
            [[t[0] for t in sentence] for sentence in pa] ==
            [[t[0] for t in sentence] for sentence in pb])
        try:
            result["agreement_with_jagger_not_gold_accuracy"] = _evaluation.evaluate_corpora(pb, pa)
        except ValueError as exc:
            result["comparison_error"] = str(exc)
    return result


def capture(command: list[str], data: bytes, timeout: float) -> bytes:
    # Match the timed runs' regular-file stdin. Some optimized external CLIs
    # assume regular-file read behavior and mishandle short reads from a pipe.
    with tempfile.TemporaryFile() as stream:
        stream.write(data)
        stream.seek(0)
        result = subprocess.run(command, stdin=stream, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {shlex.join(command)}\n"
                           f"{result.stderr.decode('utf-8', errors='replace')[-4000:]}")
    return result.stdout


def cli_runs(commands: dict[str, list[str]], input_path: Path, runs: int,
             timeout: float) -> tuple[dict[str, list[float]], list[list[str]]]:
    times: dict[str, list[float]] = {key: [] for key in commands}
    orders = []
    rng = random.Random(0)
    # One complete untimed process per implementation warms the OS file cache.
    for command in commands.values():
        with input_path.open("rb") as stream:
            result = subprocess.run(command, stdin=stream, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-4000:])
    for _ in range(runs):
        names = list(commands)
        rng.shuffle(names)
        orders.append(names)
        for name in names:
            # The timer includes process creation, model loading, input reading,
            # decoding and formatting, stdout writes, and process termination.
            with input_path.open("rb") as stream:
                started = time.perf_counter()
                result = subprocess.run(commands[name], stdin=stream, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, timeout=timeout)
                elapsed = time.perf_counter() - started
            if result.returncode:
                raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-4000:])
            times[name].append(elapsed)
    return times, orders


def api_runs(tagger, data: bytes, runs: int, wakati: bool, count_only: bool) -> list[float]:
    function = tagger.count_batch_bytes if count_only else lambda b: tagger.parse_batch_bytes(b, wakati=wakati)
    value = function(data)
    del value
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        value = function(data)
        times.append(time.perf_counter() - started)
        # Releasing the previously returned output is outside the next run.
        del value
    return times


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="held-out UTF-8 text, one sentence per line")
    parser.add_argument("--jagger", type=Path, help="separately built upstream jagger executable")
    parser.add_argument("--jagger-model", type=Path)
    parser.add_argument("--jagger-version", default="unspecified", help="record exact upstream release/commit")
    parser.add_argument("--dagger-command", help="command prefix, e.g. 'python -m dagger'; no shell is used")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=100, help="number of corpus copies in each CLI run")
    parser.add_argument("--api-repeat", type=int, default=10, help="smaller batches limit materialized output memory")
    parser.add_argument("--mode", choices=("full", "wakati", "both"), default="both")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--skip-cli", action="store_true")
    parser.add_argument("--cpu", type=int, help="pin this process and its children to this logical CPU on Linux")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--training-corpus", type=Path, help="fingerprint the actual training file")
    parser.add_argument("--dictionary", type=Path, help="fingerprint the actual dictionary file")
    parser.add_argument("--build-note", action="append", default=[],
                        help="record verified extension/upstream compiler and flags; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if min(args.runs, args.repeat, args.api_repeat) < 1:
        parser.error("runs and repeat counts must be positive")
    if bool(args.jagger) != bool(args.jagger_model):
        parser.error("--jagger and --jagger-model must be supplied together")
    if args.skip_api and args.skip_cli:
        parser.error("cannot skip both API and CLI")
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu is supported only where sched_setaffinity is available")
        os.sched_setaffinity(0, {args.cpu})
    raw = args.input.read_bytes()
    raw.decode("utf-8")
    # Make corpus repetition preserve the last sentence boundary. CRLF is
    # normalized outside all timed regions and both analyzers receive the same bytes.
    data = raw.replace(b"\r\n", b"\n")
    if not data:
        parser.error("input corpus is empty")
    if not data.endswith(b"\n"):
        data += b"\n"
    sentences = data.count(b"\n")
    prefix = shlex.split(args.dagger_command) if args.dagger_command else [sys.executable, "-m", "dagger"]
    result = {
        "schema_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment(), "build_notes": args.build_note,
        "dagger_model": describe_path(args.model),
        "input": {"path": str(args.input), "original_sha256": digest_bytes(raw),
                  "normalized_sha256": digest_bytes(data), "bytes": len(data), "sentences": sentences},
        "method": {"runs": args.runs, "cli_repeat": args.repeat, "api_repeat": args.api_repeat,
                   "cli": "wall time, startup/model load/read/format/write/exit included; stdout to DEVNULL",
                   "stdin": "regular file for both validation captures and timed CLI runs",
                   "api": "model loaded once; warm run; UTF-8 input and output materialization included",
                   "count_only": "no output strings; not comparable to full/wakati CLI",
                   "run_order": "deterministic shuffled interleaving, seed 0",
                   "warmup": "one untimed complete run per mode and implementation",
                   "rss": "not measured", "test_repetition_is_not_training": True},
        "modes": {},
    }
    if args.jagger:
        result["jagger"] = {"version": args.jagger_version, "executable": describe_path(args.jagger),
                             "model": describe_path(args.jagger_model)}
    if args.training_corpus:
        result["training_corpus"] = describe_path(args.training_corpus)
    if args.dictionary:
        result["dictionary"] = describe_path(args.dictionary)
    tagger = None
    api_data = None
    if not args.skip_api:
        from dagger import Tagger
        tagger = Tagger(args.model)
        api_data = data * args.api_repeat
    modes = (False, True) if args.mode == "both" else (args.mode == "wakati",)
    with tempfile.TemporaryDirectory(prefix="dagger-benchmark-") as temp:
        repeated = Path(temp) / "input.txt"
        with repeated.open("wb") as stream:
            for _ in range(args.repeat):
                stream.write(data)
        for wakati in modes:
            mode = "wakati" if wakati else "full"
            commands = {"dagger": prefix + ["--model", str(args.model)] + (["--wakati"] if wakati else [])}
            if args.jagger:
                commands["jagger"] = [str(args.jagger.resolve()), "-m", str(args.jagger_model)] + (["-w"] if wakati else [])
            outputs = {name: capture(command, data, args.timeout) for name, command in commands.items()}
            inspected = {name: inspect_output(value, wakati, sentences) for name, value in outputs.items()}
            entry = {"commands": commands, "output_for_one_corpus": inspected}
            if args.jagger:
                entry["comparison"] = compare_outputs(outputs["dagger"], outputs["jagger"], wakati)
            if not args.skip_cli:
                times, order = cli_runs(commands, repeated, args.runs, args.timeout)
                entry["cli_run_order"] = order
                entry["cli"] = {name: summarize(values, len(data) * args.repeat,
                                                sentences * args.repeat, inspected[name]["tokens"] * args.repeat)
                                for name, values in times.items()}
                if args.jagger:
                    entry["dagger_cli_speed_relative_to_jagger"] = (
                        entry["cli"]["jagger"]["median_seconds"] / entry["cli"]["dagger"]["median_seconds"])
            if tagger is not None:
                api_output = tagger.parse_batch_bytes(data, wakati=wakati)
                if api_output != outputs["dagger"]:
                    raise RuntimeError(f"dagger API/CLI outputs differ in {mode} mode")
                token_count = inspected["dagger"]["tokens"] * args.api_repeat
                values = api_runs(tagger, api_data, args.runs, wakati, False)
                entry["dagger_api_materialized_output"] = summarize(values, len(api_data),
                                                                      sentences * args.api_repeat, token_count)
            result["modes"][mode] = entry
        if tagger is not None:
            expected_count = next(iter(result["modes"].values()))["output_for_one_corpus"]["dagger"]["tokens"]
            measured_count = tagger.count_batch_bytes(data)
            if measured_count != expected_count:
                raise RuntimeError(f"count API returned {measured_count}; materialized output has {expected_count}")
            result["dagger_api_count_only"] = summarize(
                api_runs(tagger, api_data, args.runs, False, True), len(api_data),
                sentences * args.api_repeat, expected_count * args.api_repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
