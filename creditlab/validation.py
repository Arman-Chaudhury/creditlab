"""Crisis out-of-time validation — the experiment creditlab exists to run.

A random train/test split scatters every vintage across both sides, so the
model is tested on the same macro environment it trained on. That flatters
every metric and is exactly how pre-2008 mortgage models were "validated."
The honest protocol is out-of-time (OOT) by origination vintage: train on
loans originated through `train_max_vintage`, validate on later vintages the
model has never seen — for the bundled experiment, train through 2006 and
meet the 2007-08 crisis cohorts cold.

`crisis_validation` runs BOTH protocols with the same trainer and puts them
side by side, so the headline table reads: "the naive split said the model
was fine; the honest split shows how calibration degrades." Alongside it,
PSI (population stability index) quantifies which inputs actually drifted —
the standard model-monitoring companion, with the usual reading:
< 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .baseline import EvalReport, evaluate, random_split, train_logistic
from .features import Matrix

_PSI_FLOOR = 1e-4  # zero-proportion smoothing; keeps PSI finite on empty bins


def psi(
    expected: Sequence[float], actual: Sequence[float], *, n_bins: int = 10
) -> float:
    """Population Stability Index of `actual` against `expected`.

    Bins are quantile cuts of the *expected* (training) distribution — the
    convention in credit monitoring: the training population defines the
    buckets, and PSI measures how the new population redistributes across
    them. Proportions are floored at 1e-4 so an emptied bin contributes a
    large-but-finite penalty instead of infinity.
    """
    if not expected or not actual:
        raise ValueError("empty inputs")
    exp_sorted = sorted(expected)
    n = len(exp_sorted)
    n_bins = max(1, min(n_bins, n))
    # interior cut points at quantile ranks; dedupe for low-cardinality data
    edges: list[float] = []
    for b in range(1, n_bins):
        cut = exp_sorted[(b * n) // n_bins - 1] if (b * n) % n_bins == 0 else (
            exp_sorted[(b * n) // n_bins]
        )
        if not edges or cut > edges[-1]:
            edges.append(cut)

    def proportions(values: Sequence[float]) -> list[float]:
        counts = [0] * (len(edges) + 1)
        for v in values:
            counts[bisect_left(edges, v)] += 1  # bin b holds values <= edges[b]
        return [max(c / len(values), _PSI_FLOOR) for c in counts]

    p_exp = proportions(expected)
    p_act = proportions(actual)
    return sum(
        (a - e) * math.log(a / e) for e, a in zip(p_exp, p_act)
    )


def split_by_vintage(matrix: Matrix, *, train_max_vintage: int) -> tuple[Matrix, Matrix]:
    """Out-of-time split: vintages <= train_max_vintage train, later validate."""
    def take(train_side: bool) -> Matrix:
        sel = [
            i for i, v in enumerate(matrix.vintages)
            if (v <= train_max_vintage) == train_side
        ]
        return Matrix(
            X=[matrix.X[i] for i in sel],
            y=[matrix.y[i] for i in sel],
            feature_names=matrix.feature_names,
            loan_ids=[matrix.loan_ids[i] for i in sel],
            vintages=[matrix.vintages[i] for i in sel],
        )

    train, test = take(True), take(False)
    if not train.X or not test.X:
        vs = sorted(set(matrix.vintages))
        raise ValueError(
            f"vintage split at {train_max_vintage} leaves an empty side "
            f"(vintages present: {vs})"
        )
    return train, test


def feature_psi(train: Matrix, test: Matrix, *, n_bins: int = 10) -> dict[str, float]:
    """Per-feature PSI of the OOT population against the training population."""
    out: dict[str, float] = {}
    for j, name in enumerate(train.feature_names):
        exp = [row[j] for row in train.X]
        act = [row[j] for row in test.X]
        out[name] = psi(exp, act, n_bins=n_bins)
    return out


def _ratio(report: EvalReport) -> Optional[float]:
    if report.mean_predicted <= 0.0:
        return None
    return report.realized_rate / report.mean_predicted


@dataclass(frozen=True)
class CrisisReport:
    train_max_vintage: int
    test_vintages: tuple[int, ...]
    in_sample: EvalReport        # OOT-trained model on its own training vintages
    oot: EvalReport              # OOT-trained model on the unseen later vintages
    naive: EvalReport            # random-split benchmark: same trainer, shuffled
    score_psi: float             # drift of predicted PDs, train -> OOT population
    feature_psi: dict[str, float]

    @property
    def oot_underprediction(self) -> Optional[float]:
        """Realized / predicted default rate on the OOT vintages.
        1.0 = perfectly calibrated in the large; 1.5 = the model saw only
        two-thirds of the risk coming."""
        return _ratio(self.oot)

    @property
    def naive_underprediction(self) -> Optional[float]:
        return _ratio(self.naive)

    def top_drifted_features(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.feature_psi.items(), key=lambda kv: -kv[1])[:n]

    def headline_table(self) -> str:
        """The README table: naive random split vs honest OOT, side by side."""
        def fmt(v: Optional[float], spec: str = ".4f") -> str:
            return format(v, spec) if v is not None else "n/a"

        rows = [
            ("loans evaluated", str(self.naive.n), str(self.oot.n)),
            ("realized default rate", fmt(self.naive.realized_rate), fmt(self.oot.realized_rate)),
            ("mean predicted PD", fmt(self.naive.mean_predicted), fmt(self.oot.mean_predicted)),
            ("realized / predicted", fmt(self.naive_underprediction, ".2f"),
             fmt(self.oot_underprediction, ".2f")),
            ("AUC", fmt(self.naive.auc), fmt(self.oot.auc)),
            ("Brier score", fmt(self.naive.brier, ".5f"), fmt(self.oot.brier, ".5f")),
            ("Brier skill vs climatology", fmt(self.naive.brier_skill, ".3f"),
             fmt(self.oot.brier_skill, ".3f")),
        ]
        head = (
            f"| metric | random split (naive) | OOT {min(self.test_vintages)}-"
            f"{max(self.test_vintages)} vintages (honest) |"
        )
        sep = "|---|---|---|"
        body = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
        return "\n".join([head, sep, body])

    def to_dict(self) -> dict:
        return {
            "train_max_vintage": self.train_max_vintage,
            "test_vintages": list(self.test_vintages),
            "in_sample": self.in_sample.to_dict(),
            "oot": self.oot.to_dict(),
            "naive": self.naive.to_dict(),
            "oot_underprediction": self.oot_underprediction,
            "naive_underprediction": self.naive_underprediction,
            "score_psi": self.score_psi,
            "feature_psi": dict(self.feature_psi),
        }


Trainer = Callable[[Matrix], object]  # returns anything with predict_proba


def crisis_validation(
    matrix: Matrix,
    *,
    train_max_vintage: int = 2006,
    trainer: Trainer = train_logistic,
    naive_test_frac: float = 0.25,
    naive_seed: int = 7,
) -> CrisisReport:
    """Run the honest protocol and the naive benchmark with the same trainer."""
    oot_train, oot_test = split_by_vintage(matrix, train_max_vintage=train_max_vintage)
    oot_model = trainer(oot_train)

    naive_train, naive_test = random_split(
        matrix, test_frac=naive_test_frac, seed=naive_seed
    )
    naive_model = trainer(naive_train)

    train_scores = oot_model.predict_proba(oot_train.X)
    test_scores = oot_model.predict_proba(oot_test.X)

    return CrisisReport(
        train_max_vintage=train_max_vintage,
        test_vintages=tuple(sorted(set(oot_test.vintages))),
        in_sample=evaluate(oot_model, oot_train),
        oot=evaluate(oot_model, oot_test),
        naive=evaluate(naive_model, naive_test),
        score_psi=psi(train_scores, test_scores),
        feature_psi=feature_psi(oot_train, oot_test),
    )
