"""Discrimination and calibration metrics, hand-rolled and exactly testable.

A PD model's job is honest probabilities, not just ranking — so calibration
metrics are first-class here, not an afterthought:

- `auc` — rank-based (Mann-Whitney) with proper tie handling.
- `brier_score` — mean squared error of the probabilities; compare against
  `brier_score` of a constant climatology predictor to measure *skill*.
- `reliability_curve` — predicted vs realized default rate per probability
  bin. Quantile binning by default: PD distributions are heavily skewed, and
  equal-width bins on [0, 1] would put almost every loan in the first bin.
- `decile_table` — the classic model-risk report: sort by predicted PD,
  cut into ten groups, compare expected vs realized rate per group.

Pure stdlib on purpose: these numbers get quoted in the README, so their
implementations are small enough to audit line by line and are pinned by
exact-arithmetic tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


def _validate(y_true: Sequence[int], probs: Sequence[float]) -> None:
    if len(y_true) != len(probs):
        raise ValueError(f"length mismatch: {len(y_true)} labels vs {len(probs)} probs")
    if not y_true:
        raise ValueError("empty inputs")


def auc(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Ties receive average ranks (the correct treatment). Returns None when the
    labels are all one class — AUC is undefined there, and None beats a
    misleading 0.5.
    """
    _validate(y_true, scores)
    n_pos = sum(1 for y in y_true if y)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank across the tie run
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    rank_sum_pos = sum(r for r, y in zip(ranks, y_true) if y)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def brier_score(y_true: Sequence[int], probs: Sequence[float]) -> float:
    """Mean squared error of predicted probabilities. Lower is better."""
    _validate(y_true, probs)
    return sum((p - y) ** 2 for y, p in zip(y_true, probs)) / len(y_true)


def calibration_in_the_large(
    y_true: Sequence[int], probs: Sequence[float]
) -> tuple[float, float]:
    """(mean predicted PD, realized default rate) — the first thing a model
    validator checks."""
    _validate(y_true, probs)
    return sum(probs) / len(probs), sum(y_true) / len(y_true)


def _rank_slices(n: int, n_bins: int) -> list[tuple[int, int]]:
    """Split n sorted items into n_bins contiguous slices of near-equal size."""
    n_bins = min(n_bins, n)
    base, extra = divmod(n, n_bins)
    slices, start = [], 0
    for b in range(n_bins):
        size = base + (1 if b < extra else 0)
        slices.append((start, start + size))
        start += size
    return slices


@dataclass(frozen=True)
class ReliabilityBin:
    n: int
    mean_predicted: float
    realized: float
    p_lo: float
    p_hi: float


def reliability_curve(
    y_true: Sequence[int],
    probs: Sequence[float],
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> list[ReliabilityBin]:
    """Predicted vs realized rate per bin. strategy='quantile' (default) uses
    equal-count bins over the sorted probabilities; 'uniform' uses equal-width
    bins on [0, 1] (sparse bins are dropped)."""
    _validate(y_true, probs)
    pairs = sorted(zip(probs, y_true))
    bins: list[ReliabilityBin] = []
    if strategy == "quantile":
        for lo, hi in _rank_slices(len(pairs), n_bins):
            chunk = pairs[lo:hi]
            ps = [p for p, _ in chunk]
            ys = [y for _, y in chunk]
            bins.append(
                ReliabilityBin(
                    n=len(chunk),
                    mean_predicted=sum(ps) / len(ps),
                    realized=sum(ys) / len(ys),
                    p_lo=ps[0],
                    p_hi=ps[-1],
                )
            )
    elif strategy == "uniform":
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            chunk = [
                (p, y) for p, y in pairs if lo <= p < hi or (b == n_bins - 1 and p == hi)
            ]
            if not chunk:
                continue
            ps = [p for p, _ in chunk]
            ys = [y for _, y in chunk]
            bins.append(
                ReliabilityBin(
                    n=len(chunk),
                    mean_predicted=sum(ps) / len(ps),
                    realized=sum(ys) / len(ys),
                    p_lo=lo,
                    p_hi=hi,
                )
            )
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return bins


@dataclass(frozen=True)
class DecileRow:
    decile: int  # 1 = highest predicted risk
    n: int
    mean_predicted: float
    realized: float
    lift: Optional[float]  # realized rate / overall rate


def decile_table(
    y_true: Sequence[int], probs: Sequence[float], *, n_deciles: int = 10
) -> list[DecileRow]:
    """Expected-vs-realized default rate by predicted-risk decile."""
    _validate(y_true, probs)
    overall = sum(y_true) / len(y_true)
    pairs = sorted(zip(probs, y_true), key=lambda t: -t[0])  # riskiest first
    rows = []
    for d, (lo, hi) in enumerate(_rank_slices(len(pairs), n_deciles), start=1):
        chunk = pairs[lo:hi]
        ps = [p for p, _ in chunk]
        ys = [y for _, y in chunk]
        realized = sum(ys) / len(ys)
        rows.append(
            DecileRow(
                decile=d,
                n=len(chunk),
                mean_predicted=sum(ps) / len(ps),
                realized=realized,
                lift=(realized / overall) if overall > 0 else None,
            )
        )
    return rows
