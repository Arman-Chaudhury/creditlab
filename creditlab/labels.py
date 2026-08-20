"""Default labels and vintage cohorts from monthly performance records.

The unit of prediction is the loan. From each loan's monthly history we derive:

- **D90+ / D180 within horizon** — did the loan ever reach 90 (180) days
  delinquent, or terminate in a credit-event zero-balance code, within the
  first `horizon_months` of loan age? This is the modeling target.
- **Censoring** — a loan whose observation window ends before the horizon
  with no event and no termination has an *unknown* label (None), not a
  negative one. Treating censored loans as non-defaults is the classic way
  to silently bias a PD model optimistic; creditlab makes the third state
  explicit and downstream milestones exclude it from training.
- **Competing risk convention** — a loan that prepays inside the horizon is
  a known non-default (False): its outcome is fully observed, there is
  simply no default to find. This is the standard cohort-PD convention;
  survival-style treatments are out of scope.

Streaming: labeling consumes the performance iterator loan by loan in
constant memory. Classic Fannie Mae performance files group all rows of a
loan contiguously; `iter_loan_histories` relies on that and (by default)
verifies it, raising rather than silently double-counting a loan whose rows
are scattered.

Vintage = origination year, derived from the performance stream itself
(`period` minus `loan_age` months), so labeling needs no acquisition join.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .loader import iter_performance
from .schema import DEFAULT_ZB_CODES, PREPAY_ZB_CODES, PerformanceRecord, parse_dlq

D90_MONTHS = 3   # delinquency status >= 3 months past due
D180_MONTHS = 6


class LabelError(ValueError):
    """Performance history unusable for labeling (missing fields, unsorted file)."""


@dataclass(frozen=True, slots=True)
class LoanLabel:
    loan_id: str
    vintage: int                    # origination year
    orig_date: date                 # derived: first period minus loan_age months
    observed_months: int            # last observed loan age
    d90_within_horizon: Optional[bool]   # None = censored (unknown)
    d180_within_horizon: Optional[bool]
    defaulted: bool                 # credit-event ZB code anywhere in observation
    default_month: Optional[int]    # loan age at first D90+ or credit event
    prepaid: bool
    prepay_month: Optional[int]


@dataclass
class CohortStats:
    vintage: int
    n_loans: int = 0
    n_d90: int = 0
    n_d180: int = 0
    n_prepaid: int = 0
    n_censored: int = 0

    @property
    def n_determinable(self) -> int:
        return self.n_loans - self.n_censored

    @property
    def d90_rate(self) -> Optional[float]:
        return self.n_d90 / self.n_determinable if self.n_determinable else None


def _sub_months(d: date, n: int) -> date:
    total = d.year * 12 + (d.month - 1) - n
    y, m = divmod(total, 12)
    return date(y, m + 1, 1)


def iter_loan_histories(
    records: Iterable[PerformanceRecord], *, check_sorted: bool = True
) -> Iterator[list[PerformanceRecord]]:
    """Group a performance stream into per-loan histories, in constant memory.

    Requires all rows of a loan to be contiguous (true for classic-layout
    files). With check_sorted=True (default), a loan_id that reappears after
    another loan raises LabelError instead of yielding a corrupt second
    history; the guard costs one set of loan_ids.
    """
    seen: set[str] = set()
    current: list[PerformanceRecord] = []
    for rec in records:
        if rec.loan_id is None:
            raise LabelError("performance row with missing loan_id")
        if current and rec.loan_id == current[-1].loan_id:
            current.append(rec)
            continue
        if current:
            yield current
        if check_sorted:
            if rec.loan_id in seen:
                raise LabelError(
                    f"loan {rec.loan_id} reappears non-contiguously; "
                    "sort the performance file by loan before labeling"
                )
            seen.add(rec.loan_id)
        current = [rec]
    if current:
        yield current


def label_history(
    history: list[PerformanceRecord], *, horizon_months: int = 36
) -> LoanLabel:
    """Label one loan's monthly history. Rows may arrive in any order."""
    if not history:
        raise LabelError("empty history")
    rows = sorted(history, key=lambda r: (r.loan_age is None, r.loan_age))
    first = rows[0]
    if first.loan_age is None or first.period is None:
        raise LabelError(f"loan {first.loan_id}: missing loan_age/period")

    orig = _sub_months(first.period, first.loan_age)
    observed = 0
    first_d90: Optional[int] = None
    first_d180: Optional[int] = None
    default_event: Optional[int] = None   # loan age of credit-event ZB
    prepay_month: Optional[int] = None

    for r in rows:
        if r.loan_age is None:
            raise LabelError(f"loan {r.loan_id}: missing loan_age")
        observed = max(observed, r.loan_age)
        dlq = parse_dlq(r.dlq_status)
        if dlq is not None:
            if dlq >= D90_MONTHS and first_d90 is None:
                first_d90 = r.loan_age
            if dlq >= D180_MONTHS and first_d180 is None:
                first_d180 = r.loan_age
        if r.zb_code in DEFAULT_ZB_CODES and default_event is None:
            default_event = r.loan_age
        elif r.zb_code in PREPAY_ZB_CODES and prepay_month is None:
            prepay_month = r.loan_age

    defaulted = default_event is not None
    # first month the loan is *known bad*: D90+ observation or the credit event
    bad_month = min(
        (m for m in (first_d90, default_event) if m is not None), default=None
    )
    bad180_month = min(
        (m for m in (first_d180, default_event) if m is not None), default=None
    )

    def within(event_month: Optional[int]) -> Optional[bool]:
        if event_month is not None and event_month <= horizon_months:
            return True
        # no event inside horizon — is the non-event actually observed?
        outcome_known = (
            observed >= horizon_months
            or (prepay_month is not None and prepay_month <= horizon_months)
            or (default_event is not None and default_event <= horizon_months)
        )
        return False if outcome_known else None

    return LoanLabel(
        loan_id=first.loan_id,
        vintage=orig.year,
        orig_date=orig,
        observed_months=observed,
        d90_within_horizon=within(bad_month),
        d180_within_horizon=within(bad180_month),
        defaulted=defaulted,
        default_month=bad_month,
        prepaid=prepay_month is not None,
        prepay_month=prepay_month,
    )


def label_file(
    perf_path: str | Path, *, horizon_months: int = 36, check_sorted: bool = True
) -> Iterator[LoanLabel]:
    """Stream LoanLabels straight from a performance file."""
    histories = iter_loan_histories(
        iter_performance(perf_path), check_sorted=check_sorted
    )
    for history in histories:
        yield label_history(history, horizon_months=horizon_months)


def vintage_cohorts(labels: Iterable[LoanLabel]) -> dict[int, CohortStats]:
    """Aggregate labels into per-origination-year cohorts, sorted by vintage."""
    cohorts: dict[int, CohortStats] = {}
    for lab in labels:
        c = cohorts.setdefault(lab.vintage, CohortStats(vintage=lab.vintage))
        c.n_loans += 1
        if lab.d90_within_horizon is None:
            c.n_censored += 1
        elif lab.d90_within_horizon:
            c.n_d90 += 1
        if lab.d180_within_horizon:
            c.n_d180 += 1
        if lab.prepaid:
            c.n_prepaid += 1
    return dict(sorted(cohorts.items()))
