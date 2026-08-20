"""Gradient boosting challenger, benchmarked against the logistic baseline.

Why a challenger at all: gradient-boosted trees are the industry's strongest
learner on tabular credit data — but the baseline's interpretability (named
coefficients, exact per-loan contributions) is what regulators and model-risk
teams require. creditlab quantifies that trade instead of asserting it:
`compare_models` puts both models' EvalReports side by side on identical
splits, so the README can state exactly how much AUC/Brier the interpretable
model gives up (on the synthetic fixtures: little to nothing, because the
fixture hazard is genuinely logistic; on real data the gap is an empirical
question — which is the point).

Two backends, selected automatically:

- **lightgbm** (optional extra `creditlab[gbm]`): the reference challenger.
  Deterministic single-thread training, gain-based importances, and a
  JSON-safe artifact via LightGBM's native model string.
- **sklearn-hist** (always available): scikit-learn's
  HistGradientBoostingClassifier — the same histogram-GBM family LightGBM
  popularized — used when lightgbm isn't importable (its macOS wheel needs
  a system libomp). Importances fall back to permutation importance, and
  the artifact is not serializable (train-in-session only); both
  differences are explicit in the model object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .baseline import EvalReport
from .features import Matrix


def has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:  # ImportError, or OSError when libomp is missing on macOS
        return False


@dataclass(frozen=True)
class GBMModel:
    backend: str  # "lightgbm" | "sklearn-hist"
    feature_names: tuple[str, ...]
    importances: dict[str, float]  # normalized to sum 1
    importance_type: str  # "gain" (lightgbm) | "permutation" (sklearn-hist)
    _predictor: object

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        if X and len(X[0]) != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {len(X[0])}"
            )
        import numpy as np

        arr = np.asarray(X, dtype=float)
        if self.backend == "lightgbm":
            return [float(p) for p in self._predictor.predict(arr)]
        return [float(p) for p in self._predictor.predict_proba(arr)[:, 1]]

    def top_features(self, n: int = 5) -> list[str]:
        return [
            name for name, _ in
            sorted(self.importances.items(), key=lambda kv: -kv[1])[:n]
        ]

    def to_dict(self) -> dict:
        if self.backend != "lightgbm":
            raise NotImplementedError(
                "only the lightgbm backend serializes (native model string); "
                f"backend {self.backend!r} is train-in-session only"
            )
        return {
            "kind": "gbm",
            "backend": "lightgbm",
            "feature_names": list(self.feature_names),
            "importances": dict(self.importances),
            "importance_type": self.importance_type,
            "booster": self._predictor.model_to_string(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GBMModel":
        if d.get("kind") != "gbm" or d.get("backend") != "lightgbm":
            raise ValueError("not a lightgbm GBM artifact")
        import lightgbm as lgb

        return cls(
            backend="lightgbm",
            feature_names=tuple(d["feature_names"]),
            importances=dict(d["importances"]),
            importance_type=d.get("importance_type", "gain"),
            _predictor=lgb.Booster(model_str=d["booster"]),
        )


def _normalize(names: Sequence[str], raw: Sequence[float]) -> dict[str, float]:
    total = float(sum(raw))
    if total <= 0.0:
        return {n: 0.0 for n in names}
    return {n: float(v) / total for n, v in zip(names, raw)}


def train_gbm(
    matrix: Matrix,
    *,
    backend: str = "auto",
    seed: int = 7,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_leaves: int = 15,
    min_leaf_samples: int = 20,
    early_stopping_rounds: int = 20,
    validation_fraction: float = 0.15,
) -> GBMModel:
    """Fit the challenger. backend='auto' prefers lightgbm, falls back to
    sklearn's HistGradientBoostingClassifier.

    Both backends train with early stopping against a held-out slice of the
    training data (n_estimators is a cap, not a target): boosted trees left
    to run unchecked memorize small datasets and produce confidently
    miscalibrated probabilities — for a PD model that failure mode is worse
    than a few AUC points."""
    if not matrix.X:
        raise ValueError("empty training matrix")
    if len(set(matrix.y)) < 2:
        raise ValueError("training data has a single class; nothing to fit")
    if backend == "auto":
        backend = "lightgbm" if has_lightgbm() else "sklearn-hist"

    import numpy as np

    X = np.asarray(matrix.X, dtype=float)
    y = np.asarray(matrix.y)
    names = tuple(matrix.feature_names)

    if backend == "lightgbm":
        if not has_lightgbm():
            raise RuntimeError(
                "lightgbm requested but not importable — install creditlab[gbm] "
                "(on macOS also: brew install libomp), or use backend='sklearn-hist'"
            )
        import lightgbm as lgb

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * validation_fraction))
        val_idx, fit_idx = idx[:n_val], idx[n_val:]

        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=max_leaves,
            min_child_samples=min_leaf_samples,
            random_state=seed,
            deterministic=True,
            num_threads=1,
            force_row_wise=True,
            verbose=-1,
        )
        clf.fit(
            X[fit_idx], y[fit_idx],
            eval_X=X[val_idx], eval_y=y[val_idx],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        # Truncate the artifact to the best iteration and reload, so the
        # in-session predictor and the serialized model are the same trees.
        best = clf.best_iteration_ or -1
        booster = lgb.Booster(
            model_str=clf.booster_.model_to_string(num_iteration=best)
        )
        importances = _normalize(
            names, booster.feature_importance(importance_type="gain")
        )
        return GBMModel(
            backend="lightgbm",
            feature_names=names,
            importances=importances,
            importance_type="gain",
            _predictor=booster,
        )

    if backend == "sklearn-hist":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.inspection import permutation_importance

        clf = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaves,
            min_samples_leaf=min_leaf_samples,
            early_stopping=True,
            validation_fraction=validation_fraction,
            n_iter_no_change=early_stopping_rounds,
            random_state=seed,
        )
        clf.fit(X, y)
        perm = permutation_importance(
            clf, X, y, n_repeats=3, random_state=seed, scoring="neg_brier_score"
        )
        importances = _normalize(names, np.clip(perm.importances_mean, 0.0, None))
        return GBMModel(
            backend="sklearn-hist",
            feature_names=names,
            importances=importances,
            importance_type="permutation",
            _predictor=clf,
        )

    raise ValueError(f"unknown backend {backend!r}")


@dataclass(frozen=True)
class ModelComparison:
    baseline: EvalReport
    challenger: EvalReport
    challenger_backend: str

    @property
    def auc_delta(self) -> Optional[float]:
        if self.baseline.auc is None or self.challenger.auc is None:
            return None
        return self.challenger.auc - self.baseline.auc

    @property
    def brier_delta(self) -> float:
        """challenger - baseline; negative means the challenger is better."""
        return self.challenger.brier - self.baseline.brier

    def summary(self) -> str:
        auc_d = self.auc_delta
        lines = [
            f"challenger ({self.challenger_backend}) vs logistic baseline:",
            f"  AUC   {self.baseline.auc:.4f} -> {self.challenger.auc:.4f}"
            f" ({auc_d:+.4f})" if auc_d is not None else "  AUC   undefined",
            f"  Brier {self.baseline.brier:.5f} -> {self.challenger.brier:.5f}"
            f" ({self.brier_delta:+.5f})",
            "  trade: the baseline keeps named coefficients and exact per-loan",
            "  contributions (adverse-action reason codes); the challenger buys",
            "  whatever delta is above at the cost of that transparency.",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "challenger": self.challenger.to_dict(),
            "challenger_backend": self.challenger_backend,
            "auc_delta": self.auc_delta,
            "brier_delta": self.brier_delta,
        }


def compare_models(
    baseline_report: EvalReport, challenger_report: EvalReport, *, backend: str
) -> ModelComparison:
    return ModelComparison(
        baseline=baseline_report,
        challenger=challenger_report,
        challenger_backend=backend,
    )
