"""Chunked, memory-bounded readers for Fannie Mae acquisition/performance files.

Everything is generator-based: a 20 GB performance file streams row by row in
constant memory. Files may be plain text or gzip (detected by extension).

Two error policies:
- strict=True (default): the first malformed row raises SchemaError naming the
  file, line number, and problem. Use for anything feeding a model.
- strict=False: malformed rows are counted and skipped; pass a LoadStats to
  collect the tally. Use for exploratory passes over dirty data.
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Optional, TextIO, TypeVar

from .schema import (
    ACQUISITION_SCHEMA,
    PERFORMANCE_SCHEMA,
    AcquisitionRecord,
    PerformanceRecord,
    SchemaError,
)

T = TypeVar("T")

_MAX_RECORDED_ERRORS = 20


@dataclass
class LoadStats:
    rows_read: int = 0
    rows_ok: int = 0
    rows_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def record_error(self, msg: str) -> None:
        self.rows_skipped += 1
        if len(self.errors) < _MAX_RECORDED_ERRORS:
            self.errors.append(msg)


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def _iter_records(
    path: Path,
    schema: tuple,
    record_cls: type,
    strict: bool,
    stats: Optional[LoadStats],
) -> Iterator:
    expected = len(schema)
    with _open_text(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            if stats is not None:
                stats.rows_read += 1
            tokens = line.split("|")
            if len(tokens) != expected:
                msg = (
                    f"{path.name}:{lineno}: expected {expected} fields, "
                    f"got {len(tokens)}"
                )
                if strict:
                    raise SchemaError(msg)
                if stats is not None:
                    stats.record_error(msg)
                continue
            try:
                values = {name: conv(tok) for (name, conv), tok in zip(schema, tokens)}
            except (ValueError, TypeError) as e:
                msg = f"{path.name}:{lineno}: {e}"
                if strict:
                    raise SchemaError(msg) from e
                if stats is not None:
                    stats.record_error(msg)
                continue
            if stats is not None:
                stats.rows_ok += 1
            yield record_cls(**values)


def iter_acquisitions(
    path: str | Path, *, strict: bool = True, stats: Optional[LoadStats] = None
) -> Iterator[AcquisitionRecord]:
    """Stream AcquisitionRecords from a classic-layout acquisition file."""
    return _iter_records(Path(path), ACQUISITION_SCHEMA, AcquisitionRecord, strict, stats)


def iter_performance(
    path: str | Path, *, strict: bool = True, stats: Optional[LoadStats] = None
) -> Iterator[PerformanceRecord]:
    """Stream PerformanceRecords from a classic-layout performance file."""
    return _iter_records(Path(path), PERFORMANCE_SCHEMA, PerformanceRecord, strict, stats)


def chunked(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Batch any iterable into lists of up to `size` items, preserving order.

    Downstream stages (labeling, feature building) consume performance streams
    in bounded batches with this — the file itself is never materialized.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch
