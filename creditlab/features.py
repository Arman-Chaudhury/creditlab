"""Origination-time feature engineering with a structural leakage audit.

The cardinal sin in PD modeling is leakage: letting anything the lender could
not have known at origination into the features. creditlab enforces that
structurally rather than by convention:

- Every feature is declared in FEATURE_SPEC with the acquisition-record fields
  it reads (`sources`). The builder function receives ONLY the
  AcquisitionRecord — performance data is used for the label and nothing else.
- `audit_leakage()` verifies every declared source is a field of
  AcquisitionRecord (the origination-time allowlist). A feature claiming a
  performance field fails loudly at import/test time, not silently in a
  backtest. The audit runs in CI as a test.

Missing-value policy (explicit, per feature):

- **Required** features (fico, oltv, dti, orig_rate, orig_upb): a loan missing
  any of them is excluded from the dataset and counted per-field in
  FeatureStats — never silently imputed.
- **Optional** features carry documented fallbacks: ocltv falls back to oltv,
  mi_pct to 0 (no MI), num_borrowers to 1. Flags (first-time buyer,
  co-borrower) are 0/1 with missing treated as 0.

`rate_spread` is the note rate minus an annual 30-year fixed benchmark
(approximate Freddie Mac PMMS annual averages, bundled below). It is a proxy —
monthly benchmarks would be tighter — but it is computed strictly from
origination-time information; pass a custom `benchmark` mapping to override.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from .labels import LoanLabel, label_file
from .loader import iter_acquisitions
from .schema import AcquisitionRecord


class LeakageError(ValueError):
    """A feature declares a source outside the origination-time allowlist."""


# Approximate annual averages of the Freddie Mac PMMS 30-year fixed rate.
# Good enough for a spread-at-origination proxy; override via `benchmark=`.
BENCHMARK_RATE_30Y: dict[int, float] = {
    1999: 7.44, 2000: 8.05, 2001: 6.97, 2002: 6.54, 2003: 5.83,
    2004: 5.84, 2005: 5.87, 2006: 6.41, 2007: 6.34, 2008: 6.03,
    2009: 5.04, 2010: 4.69, 2011: 4.45, 2012: 3.66, 2013: 3.98,
    2014: 4.17, 2015: 3.85, 2016: 3.65, 2017: 3.99, 2018: 4.54,
    2019: 3.94, 2020: 3.10, 2021: 2.96, 2022: 5.34, 2023: 6.81,
    2024: 6.70,
}


def benchmark_rate(year: int, benchmark: Mapping[int, float] | None = None) -> float:
    table = benchmark if benchmark is not None else BENCHMARK_RATE_30Y
    if year in table:
        return table[year]
    nearest = min(table, key=lambda y: abs(y - year))
    return table[nearest]


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "numeric" | "categorical"
    sources: tuple[str, ...]  # AcquisitionRecord fields this feature reads
    fn: Callable[[AcquisitionRecord], object]
    required: bool = False  # missing value -> exclude the loan
    vocab: tuple[str, ...] = ()  # fixed categories (categorical only)


def _flag(v: Optional[str]) -> float:
    return 1.0 if v == "Y" else 0.0


FEATURE_SPEC: tuple[Feature, ...] = (
    Feature("fico", "numeric", ("fico",), lambda a: a.fico, required=True),
    Feature("oltv", "numeric", ("oltv",), lambda a: a.oltv, required=True),
    Feature("dti", "numeric", ("dti",), lambda a: a.dti, required=True),
    Feature("orig_rate", "numeric", ("orig_rate",), lambda a: a.orig_rate, required=True),
    Feature(
        "log_upb", "numeric", ("orig_upb",),
        lambda a: math.log(a.orig_upb) if a.orig_upb else None, required=True,
    ),
    Feature(
        "ocltv", "numeric", ("ocltv", "oltv"),
        lambda a: a.ocltv if a.ocltv is not None else a.oltv,
    ),
    Feature(
        "rate_spread", "numeric", ("orig_rate", "orig_date"),
        lambda a: (
            a.orig_rate - benchmark_rate(a.orig_date.year)
            if a.orig_rate is not None and a.orig_date is not None
            else None
        ),
    ),
    Feature("mi_pct", "numeric", ("mi_pct",), lambda a: a.mi_pct or 0.0),
    Feature("num_borrowers", "numeric", ("num_borrowers",), lambda a: a.num_borrowers or 1),
    Feature("has_coborrower", "numeric", ("co_fico",), lambda a: 1.0 if a.co_fico else 0.0),
    Feature("first_time_buyer", "numeric", ("first_time_buyer",), lambda a: _flag(a.first_time_buyer)),
    Feature("purpose", "categorical", ("purpose",), lambda a: a.purpose, vocab=("P", "C", "R", "U")),
    Feature("occupancy", "categorical", ("occupancy",), lambda a: a.occupancy, vocab=("P", "S", "I")),
    Feature(
        "property_type", "categorical", ("property_type",),
        lambda a: a.property_type, vocab=("SF", "CO", "PU", "MH", "CP"),
    ),
    Feature("channel", "categorical", ("channel",), lambda a: a.channel, vocab=("R", "C", "B")),
)

_UNK = "UNK"


def audit_leakage(spec: Iterable[Feature] = FEATURE_SPEC) -> None:
    """Raise LeakageError unless every feature reads only origination fields.

    The allowlist is AcquisitionRecord's own fields: by construction nothing
    observed after origination (delinquency, balances, zero-balance codes)
    exists there. Run as a test so a leaky feature can never ship.
    """
    allowed = set(AcquisitionRecord.__dataclass_fields__)
    for f in spec:
        bad = set(f.sources) - allowed
        if bad:
            raise LeakageError(
                f"feature {f.name!r} reads non-origination fields: {sorted(bad)}"
            )
        if not f.sources:
            raise LeakageError(f"feature {f.name!r} declares no sources")


@dataclass(frozen=True, slots=True)
class FeatureRow:
    loan_id: str
    vintage: int
    y: bool
    numeric: dict[str, float]
    categorical: dict[str, str]


@dataclass
class FeatureStats:
    n_acquisitions: int = 0
    n_labels: int = 0
    n_joined: int = 0
    n_rows: int = 0
    n_censored_excluded: int = 0
    n_missing_required: int = 0
    missing_by_field: dict[str, int] = field(default_factory=dict)
    n_label_without_acq: int = 0
    n_acq_without_label: int = 0


def build_row(
    acq: AcquisitionRecord,
    label: LoanLabel,
    *,
    target: str = "d90",
    stats: Optional[FeatureStats] = None,
) -> Optional[FeatureRow]:
    """Join one acquisition record with its label. None if unusable
    (censored label or missing required feature), counted in stats."""
    y = label.d90_within_horizon if target == "d90" else label.d180_within_horizon
    if y is None:
        if stats is not None:
            stats.n_censored_excluded += 1
        return None

    numeric: dict[str, float] = {}
    categorical: dict[str, str] = {}
    for f in FEATURE_SPEC:
        value = f.fn(acq)
        if f.kind == "numeric":
            if value is None:
                if f.required:
                    if stats is not None:
                        stats.n_missing_required += 1
                        stats.missing_by_field[f.name] = (
                            stats.missing_by_field.get(f.name, 0) + 1
                        )
                    return None
                value = 0.0
            numeric[f.name] = float(value)
        else:
            v = value if value in f.vocab else _UNK
            categorical[f.name] = v

    return FeatureRow(
        loan_id=acq.loan_id,
        vintage=label.vintage,
        y=bool(y),
        numeric=numeric,
        categorical=categorical,
    )


def build_dataset(
    acq_path: str | Path,
    perf_path: str | Path,
    *,
    horizon_months: int = 36,
    target: str = "d90",
) -> tuple[list[FeatureRow], FeatureStats]:
    """Stream both files and return model-ready rows plus an accounting of
    every loan that did not become one (censored, missing fields, unjoined)."""
    stats = FeatureStats()
    labels: dict[str, LoanLabel] = {}
    for lab in label_file(perf_path, horizon_months=horizon_months):
        labels[lab.loan_id] = lab
    stats.n_labels = len(labels)

    rows: list[FeatureRow] = []
    joined_ids: set[str] = set()
    for acq in iter_acquisitions(acq_path):
        stats.n_acquisitions += 1
        lab = labels.get(acq.loan_id)
        if lab is None:
            stats.n_acq_without_label += 1
            continue
        stats.n_joined += 1
        joined_ids.add(acq.loan_id)
        row = build_row(acq, lab, target=target, stats=stats)
        if row is not None:
            rows.append(row)
    stats.n_label_without_acq = len(labels) - len(joined_ids)
    stats.n_rows = len(rows)
    return rows, stats


@dataclass(frozen=True)
class Matrix:
    X: list[list[float]]
    y: list[int]
    feature_names: list[str]
    loan_ids: list[str]
    vintages: list[int]


def matrix_columns(spec: Iterable[Feature] = FEATURE_SPEC) -> list[str]:
    """Deterministic design-matrix column order: numerics in spec order, then
    one-hot columns per categorical vocab (plus an UNK bucket each)."""
    cols: list[str] = []
    for f in spec:
        if f.kind == "numeric":
            cols.append(f.name)
    for f in spec:
        if f.kind == "categorical":
            cols.extend(f"{f.name}={v}" for v in (*f.vocab, _UNK))
    return cols


def to_matrix(rows: Iterable[FeatureRow]) -> Matrix:
    """Encode rows into a plain-lists design matrix (numpy arrives with the
    modeling milestones; nothing here needs it)."""
    cols = matrix_columns()
    index = {c: i for i, c in enumerate(cols)}
    X: list[list[float]] = []
    y: list[int] = []
    loan_ids: list[str] = []
    vintages: list[int] = []
    for row in rows:
        vec = [0.0] * len(cols)
        for name, value in row.numeric.items():
            vec[index[name]] = value
        for name, value in row.categorical.items():
            vec[index[f"{name}={value}"]] = 1.0
        X.append(vec)
        y.append(int(row.y))
        loan_ids.append(row.loan_id)
        vintages.append(row.vintage)
    return Matrix(X=X, y=y, feature_names=cols, loan_ids=loan_ids, vintages=vintages)
