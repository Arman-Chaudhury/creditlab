"""Regularized logistic regression baseline with a self-contained artifact.

scikit-learn does the fitting (L2-regularized, lbfgs); the *trained model* is
extracted into a plain LogisticModel of feature names, standardization
parameters, and coefficients. Consequences, all deliberate:

- Inference needs no scikit-learn: predict_proba is ten lines of arithmetic,
  and the artifact round-trips through a JSON-safe dict (CLI persistence in
  milestone 8).
- Coefficients live in standardized space, keyed by feature name — which is
  exactly the per-loan contribution decomposition milestone 7's
  adverse-action reason codes need. Interpretability is why a logistic
  baseline exists; keeping the weights inspectable is the point.
- Evaluation (EvalReport) bundles discrimination AND calibration: AUC,
  Brier vs a climatology reference, reliability curve, decile table, and
  calibration-in-the-large. Milestones 5 (GBM challenger) and 6 (crisis
  validation) reuse the same report so comparisons are apples-to-apples.

The random split here is deliberately naive (seeded shuffle): it leaks the
macro environment across train/test, which is the standard mistake. Milestone
6 introduces the vintage-based out-of-time split and quantifies exactly what
the naive split hides.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from random import Random
from typing import Optional, Sequence

from .features import Matrix
from .metrics import (
    DecileRow,
    ReliabilityBin,
    auc,
    brier_score,
    calibration_in_the_large,
    decile_table,
    reliability_curve,
)


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    coefs: tuple[float, ...]  # standardized space, aligned with feature_names
    intercept: float

    def coefficients(self) -> dict[str, float]:
        """{feature_name: standardized coefficient} — the interpretability view."""
        return dict(zip(self.feature_names, self.coefs))

    def _z(self, x: Sequence[float]) -> float:
        z = self.intercept
        for xi, mu, sd, w in zip(x, self.means, self.stds, self.coefs):
            z += w * (xi - mu) / sd
        return z

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        if X and len(X[0]) != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {len(X[0])}"
            )
        return [1.0 / (1.0 + math.exp(-self._z(x))) for x in X]

    def to_dict(self) -> dict:
        return {
            "kind": "logistic",
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "stds": list(self.stds),
            "coefs": list(self.coefs),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticModel":
        if d.get("kind") != "logistic":
            raise ValueError(f"not a logistic model artifact: kind={d.get('kind')!r}")
        return cls(
            feature_names=tuple(d["feature_names"]),
            means=tuple(d["means"]),
            stds=tuple(d["stds"]),
            coefs=tuple(d["coefs"]),
            intercept=float(d["intercept"]),
        )


def train_logistic(
    matrix: Matrix, *, c: float = 1.0, max_iter: int = 1000
) -> LogisticModel:
    """Fit L2-regularized logistic regression on a design matrix."""
    if not matrix.X:
        raise ValueError("empty training matrix")
    if len(set(matrix.y)) < 2:
        raise ValueError("training data has a single class; nothing to fit")

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(matrix.X, dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0.0] = 1.0  # constant columns (e.g. an unused one-hot) pass through
    Xs = (X - means) / stds

    clf = LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs")
    clf.fit(Xs, np.asarray(matrix.y))

    return LogisticModel(
        feature_names=tuple(matrix.feature_names),
        means=tuple(float(v) for v in means),
        stds=tuple(float(v) for v in stds),
        coefs=tuple(float(v) for v in clf.coef_[0]),
        intercept=float(clf.intercept_[0]),
    )


@dataclass(frozen=True)
class EvalReport:
    n: int
    n_default: int
    realized_rate: float
    mean_predicted: float
    auc: Optional[float]
    brier: float
    brier_climatology: float  # constant predictor at this set's realized rate
    reliability: list[ReliabilityBin] = field(default_factory=list)
    deciles: list[DecileRow] = field(default_factory=list)

    @property
    def brier_skill(self) -> Optional[float]:
        """1 - brier/brier_climatology; > 0 means the model beats a constant."""
        if self.brier_climatology == 0.0:
            return None
        return 1.0 - self.brier / self.brier_climatology

    def to_dict(self) -> dict:
        d = asdict(self)
        d["brier_skill"] = self.brier_skill
        return d


def evaluate(model: LogisticModel, matrix: Matrix, *, n_bins: int = 10) -> EvalReport:
    probs = model.predict_proba(matrix.X)
    y = matrix.y
    realized = sum(y) / len(y)
    mean_pred, _ = calibration_in_the_large(y, probs)
    climatology = [realized] * len(y)
    return EvalReport(
        n=len(y),
        n_default=sum(y),
        realized_rate=realized,
        mean_predicted=mean_pred,
        auc=auc(y, probs),
        brier=brier_score(y, probs),
        brier_climatology=brier_score(y, climatology),
        reliability=reliability_curve(y, probs, n_bins=n_bins),
        deciles=decile_table(y, probs, n_deciles=n_bins),
    )


def random_split(
    matrix: Matrix, *, test_frac: float = 0.25, seed: int = 7
) -> tuple[Matrix, Matrix]:
    """Seeded shuffle split. Ignores time on purpose — the naive benchmark
    that milestone 6's vintage-based out-of-time split is measured against."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    idx = list(range(len(matrix.X)))
    Random(seed).shuffle(idx)
    n_test = max(1, int(len(idx) * test_frac))
    test_idx = set(idx[:n_test])

    def take(keep_test: bool) -> Matrix:
        sel = [i for i in range(len(matrix.X)) if (i in test_idx) == keep_test]
        return Matrix(
            X=[matrix.X[i] for i in sel],
            y=[matrix.y[i] for i in sel],
            feature_names=matrix.feature_names,
            loan_ids=[matrix.loan_ids[i] for i in sel],
            vintages=[matrix.vintages[i] for i in sel],
        )

    return take(False), take(True)
