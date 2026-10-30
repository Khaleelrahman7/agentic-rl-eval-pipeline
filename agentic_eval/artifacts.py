"""Strict JSON reading, canonical hashing, and same-directory atomic artifact writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _reject_constant(token: str) -> Any:
    raise ValueError(f"{token} is not a JSON value")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"object repeats the key {key!r}")
        result[key] = value
    return result


def strict_loads(text: str) -> Any:
    """Parse RFC 8259 JSON, rejecting NaN/Infinity tokens and repeated object keys."""
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    except RecursionError as exc:
        raise ValueError("JSON nesting is too deep") from exc


def read_utf8(path: Path) -> str:
    """Read a file as strict UTF-8 without translating or splitting any character."""
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read {path.name} as UTF-8: {exc}") from exc


def load_json(path: Path) -> Any:
    return strict_loads(read_utf8(path))


def canonical_json(value: Any) -> str:
    # ASCII escapes keep every code point, including unpaired surrogates, representable.
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(problems: list[dict[str, Any]], settings: dict[str, Any]) -> str:
    payload = canonical_json({"problems": problems, "settings": settings}).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n")


def atomic_write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(canonical_json(value) + "\n" for value in values))
