"""Adverse-action reason codes and a proxy-fairness audit.

**Reason codes.** ECOA (via Regulation B) requires lenders to tell declined
applicants the principal reasons for the decision. For the logistic baseline
those reasons are exact, not approximated: the model's log-odds are an
additive sum of per-feature contributions

    z = intercept + sum_i  coef_i * (x_i - mean_i) / std_i

so "top reasons" = the largest positive (risk-increasing) terms of that sum,
by construction summing — with the intercept — to the loan's exact score.
This is the payoff of keeping an interpretable baseline: no SHAP sampling, no
approximation error, an auditor can recompute every number by hand. (The GBM
challenger deliberately has no reason-code path — that asymmetry IS the
accuracy-vs-interpretability trade from milestone 5.)

**Proxy-fairness audit.** The dataset carries no protected attributes, and
creditlab's feature spec deliberately excludes geography (state, zip3) — the
classic redlining proxy. The audit checks the two ways bias could still leak:

- *Outcome parity by group*: per-group mean predicted PD vs realized default
  rate. A group the model systematically over-predicts (predicted >> realized)
  is being over-priced relative to its actual risk.
- *Proxy power screening*: for each model feature, the maximum PSI between the
  feature's overall distribution and its distribution within any single group.
  A feature whose distribution differs sharply by group can reconstruct group
  membership — a proxy — even though the group field itself is not a feature.

This is an audit, not a certificate: it can flag disparities, it cannot prove
their absence. It runs on any grouping attribute you pass (state by default,
via `attach_groups`).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .baseline import LogisticModel
from .features import Matrix
from .loader import iter_acquisitions
from .validation import psi

# Human-readable ECOA-style phrases for the numeric features. One-hot columns
# fall back to a generic template.
REASON_PHRASES: dict[str, str] = {
    "fico": "Credit score below the typical approved profile",
    "oltv": "Loan-to-value ratio higher than typical",
    "ocltv": "Combined loan-to-value ratio higher than typical",
    "dti": "Debt-to-income ratio higher than typical",
    "orig_rate": "Note rate above the typical approved profile",
    "rate_spread": "Rate spread at origination above typical",
    "log_upb": "Loan amount outside the typical approved range",
    "mi_pct": "Mortgage-insurance profile indicates elevated risk",
    "num_borrowers": "Number of borrowers on the application",
    "has_coborrower": "Co-borrower status",
    "first_time_buyer": "First-time homebuyer status",
}


def _phrase(feature: str) -> str:
    if feature in REASON_PHRASES:
        return REASON_PHRASES[feature]
    if "=" in feature:
        field, value = feature.split("=", 1)
        return f"Loan characteristic: {field} is {value}"
    return f"Loan characteristic: {feature}"


@dataclass(frozen=True)
class ReasonCode:
    feature: str
    contribution: float  # log-odds added by this feature for this loan (> 0)
    value: float         # the loan's raw feature value
    phrase: str


@dataclass(frozen=True)
class LoanExplanation:
    loan_id: str
    predicted_pd: float
    log_odds: float
    intercept: float
    reasons: list[ReasonCode]  # top risk-increasing contributions


def contributions(model: LogisticModel, x: Sequence[float]) -> dict[str, float]:
    """Exact per-feature log-odds contributions for one loan. Sums (with the
    intercept) to the loan's log-odds — asserted in the tests."""
    if len(x) != len(model.feature_names):
        raise ValueError(
            f"expected {len(model.feature_names)} features, got {len(x)}"
        )
    return {
        name: w * (xi - mu) / sd
        for name, xi, mu, sd, w in zip(
            model.feature_names, x, model.means, model.stds, model.coefs
        )
    }


def reason_codes(
    model: LogisticModel, x: Sequence[float], *, top_n: int = 4
) -> list[ReasonCode]:
    """Top risk-increasing contributions for one loan, largest first.
    Returns fewer than top_n when fewer contributions are positive — a loan
    with nothing pushing its risk up has nothing to cite."""
    contrib = contributions(model, x)
    values = dict(zip(model.feature_names, x))
    positive = sorted(
        ((name, c) for name, c in contrib.items() if c > 0.0),
        key=lambda kv: -kv[1],
    )
    return [
        ReasonCode(
            feature=name,
            contribution=c,
            value=values[name],
            phrase=_phrase(name),
        )
        for name, c in positive[:top_n]
    ]


def explain_loan(
    model: LogisticModel, matrix: Matrix, loan_id: str, *, top_n: int = 4
) -> LoanExplanation:
    """Reason codes for a specific loan in a design matrix."""
    try:
        i = matrix.loan_ids.index(loan_id)
    except ValueError:
        raise KeyError(f"loan {loan_id!r} not in matrix") from None
    x = matrix.X[i]
    z = model.intercept + sum(contributions(model, x).values())
    return LoanExplanation(
        loan_id=loan_id,
        predicted_pd=model.predict_proba([x])[0],
        log_odds=z,
        intercept=model.intercept,
        reasons=reason_codes(model, x, top_n=top_n),
    )


# --- proxy-fairness audit ----------------------------------------------------

OTHER_GROUP = "OTHER"


def attach_groups(
    acq_path: str | Path, matrix: Matrix, *, field: str = "state"
) -> list[str]:
    """Pull a grouping attribute (default: state) from the acquisition file,
    aligned with matrix.loan_ids. Loans without the attribute get 'UNKNOWN'."""
    lookup: dict[str, str] = {}
    for rec in iter_acquisitions(acq_path):
        value = getattr(rec, field)
        lookup[rec.loan_id] = str(value) if value is not None else "UNKNOWN"
    return [lookup.get(lid, "UNKNOWN") for lid in matrix.loan_ids]


@dataclass(frozen=True)
class GroupStats:
    group: str
    n: int
    mean_predicted: float
    realized: float

    @property
    def gap(self) -> float:
        """predicted - realized; positive = the model over-prices this group's
        risk relative to its actual outcomes."""
        return self.mean_predicted - self.realized


@dataclass(frozen=True)
class ProxyReport:
    group_field: str
    groups: list[GroupStats]
    feature_proxy_power: dict[str, float]  # max PSI of feature dist within any group

    @property
    def worst_overpredicted(self) -> Optional[GroupStats]:
        return max(self.groups, key=lambda g: g.gap, default=None)

    def top_proxy_features(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(
            self.feature_proxy_power.items(), key=lambda kv: -kv[1]
        )[:n]

    def summary(self, *, psi_flag: float = 0.25) -> str:
        lines = [f"proxy-fairness audit by {self.group_field}:"]
        for g in sorted(self.groups, key=lambda g: -g.gap):
            lines.append(
                f"  {g.group:>8s}  n={g.n:<5d} predicted={g.mean_predicted:.4f} "
                f"realized={g.realized:.4f} gap={g.gap:+.4f}"
            )
        flagged = [
            f"{name} (PSI {v:.2f})"
            for name, v in self.top_proxy_features()
            if v >= psi_flag
        ]
        lines.append(
            "  potential proxy features: " + (", ".join(flagged) if flagged else "none "
            f"(all feature PSIs < {psi_flag})")
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "group_field": self.group_field,
            "groups": [
                {
                    "group": g.group,
                    "n": g.n,
                    "mean_predicted": g.mean_predicted,
                    "realized": g.realized,
                    "gap": g.gap,
                }
                for g in self.groups
            ],
            "feature_proxy_power": dict(self.feature_proxy_power),
        }


def proxy_audit(
    model,
    matrix: Matrix,
    groups: Sequence[str],
    *,
    group_field: str = "state",
    min_group_size: int = 30,
) -> ProxyReport:
    """Audit model outputs across a grouping attribute that is NOT a feature.

    Groups smaller than min_group_size are pooled into OTHER: per-group rates
    on a handful of loans are noise, and reporting noise as disparity is its
    own kind of dishonesty. Works with any model exposing predict_proba.
    """
    if len(groups) != len(matrix.X):
        raise ValueError(
            f"{len(groups)} group labels for {len(matrix.X)} loans"
        )
    counts = Counter(groups)
    canonical = [
        g if counts[g] >= min_group_size else OTHER_GROUP for g in groups
    ]
    probs = model.predict_proba(matrix.X)

    by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(canonical):
        by_group[g].append(i)

    stats = [
        GroupStats(
            group=g,
            n=len(idx),
            mean_predicted=sum(probs[i] for i in idx) / len(idx),
            realized=sum(matrix.y[i] for i in idx) / len(idx),
        )
        for g, idx in sorted(by_group.items())
    ]

    proxy_power: dict[str, float] = {}
    for j, name in enumerate(matrix.feature_names):
        overall = [row[j] for row in matrix.X]
        worst = 0.0
        for g, idx in by_group.items():
            if g == OTHER_GROUP or len(idx) < min_group_size:
                continue
            # PSI inflates on small samples (~(bins-1)/2n bias plus variance):
            # scale bin count to group size so noise doesn't read as drift.
            n_bins = min(10, max(2, len(idx) // 20))
            worst = max(
                worst, psi(overall, [matrix.X[i][j] for i in idx], n_bins=n_bins)
            )
        proxy_power[name] = worst

    return ProxyReport(
        group_field=group_field,
        groups=stats,
        feature_proxy_power=proxy_power,
    )
