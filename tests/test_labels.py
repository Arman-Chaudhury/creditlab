from datetime import date

import pytest

from creditlab.fixtures import FixtureConfig, generate
from creditlab.labels import (
    LabelError,
    iter_loan_histories,
    label_file,
    label_history,
    vintage_cohorts,
)
from creditlab.loader import iter_performance
from creditlab.schema import DEFAULT_ZB_CODES, PerformanceRecord

ORIG = date(2005, 6, 1)


def _add_months(d, n):
    total = d.year * 12 + (d.month - 1) + n
    y, m = divmod(total, 12)
    return date(y, m + 1, 1)


def perf(loan_id="L1", age=1, dlq="0", zb=None, orig=ORIG):
    """Minimal PerformanceRecord for label tests; only labeling fields filled."""
    fields = dict.fromkeys(PerformanceRecord.__dataclass_fields__, None)
    fields.update(
        loan_id=loan_id,
        period=_add_months(orig, age),
        loan_age=age,
        dlq_status=dlq,
        zb_code=zb,
    )
    return PerformanceRecord(**fields)


def history(*specs, loan_id="L1"):
    """specs: (age, dlq) or (age, dlq, zb) tuples."""
    return [perf(loan_id, *s) for s in specs]


class TestLabelHistory:
    def test_clean_loan_observed_past_horizon_is_false(self):
        h = history(*[(t, "0") for t in range(1, 41)])
        lab = label_history(h, horizon_months=36)
        assert lab.d90_within_horizon is False
        assert lab.d180_within_horizon is False
        assert not lab.defaulted and not lab.prepaid
        assert lab.vintage == 2005 and lab.orig_date == ORIG

    def test_d90_event_inside_horizon(self):
        h = history((1, "0"), (2, "1"), (3, "2"), (4, "3"), (5, "4"))
        lab = label_history(h, horizon_months=36)
        assert lab.d90_within_horizon is True
        assert lab.default_month == 4  # first month dlq >= 3

    def test_self_curing_blip_is_not_default(self):
        h = history(*[(t, "1" if t == 5 else "0") for t in range(1, 40)])
        lab = label_history(h, horizon_months=36)
        assert lab.d90_within_horizon is False

    def test_credit_event_zb_counts_even_without_d180_observation(self):
        # dlq maxes at 4, then REO disposition: both labels must be True
        h = history((1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, None, "09"))
        lab = label_history(h, horizon_months=36)
        assert lab.d90_within_horizon is True
        assert lab.d180_within_horizon is True  # via the credit event
        assert lab.defaulted
        assert lab.default_month == 3

    def test_prepay_inside_horizon_is_known_negative(self):
        h = history((1, "0"), (2, "0"), (3, "0", "01"))
        lab = label_history(h, horizon_months=36)
        assert lab.d90_within_horizon is False  # outcome known, not censored
        assert lab.prepaid and lab.prepay_month == 3

    def test_short_observation_is_censored(self):
        h = history(*[(t, "0") for t in range(1, 11)])  # 10 months, horizon 36
        lab = label_history(h, horizon_months=36)
        assert lab.d90_within_horizon is None
        assert lab.d180_within_horizon is None

    def test_event_after_horizon_is_false_within_but_defaulted(self):
        h = history(*[(t, "0") for t in range(1, 40)], (40, "3"), (41, None, "09"))
        lab = label_history(h, horizon_months=36)
        assert lab.d90_within_horizon is False
        assert lab.defaulted is True
        assert lab.default_month == 40

    def test_rows_in_any_order(self):
        h = history((4, "3"), (1, "0"), (3, "2"), (2, "1"))
        lab = label_history(h, horizon_months=36)
        assert lab.d90_within_horizon is True
        assert lab.default_month == 4

    def test_missing_loan_age_raises(self):
        bad = perf("L1", 1)
        object.__setattr__  # frozen; build directly instead
        fields = dict.fromkeys(PerformanceRecord.__dataclass_fields__, None)
        fields.update(loan_id="L1", period=date(2005, 7, 1), dlq_status="0")
        with pytest.raises(LabelError, match="loan_age"):
            label_history([PerformanceRecord(**fields)])
        assert bad.loan_age == 1  # sanity: helper unaffected

    def test_empty_history_raises(self):
        with pytest.raises(LabelError):
            label_history([])


class TestGroupingAndCohorts:
    def test_iter_loan_histories_groups_contiguously(self):
        recs = history((1, "0"), (2, "0"), loan_id="A") + history(
            (1, "0"), loan_id="B"
        )
        groups = list(iter_loan_histories(recs))
        assert [g[0].loan_id for g in groups] == ["A", "B"]
        assert [len(g) for g in groups] == [2, 1]

    def test_non_contiguous_loan_raises(self):
        recs = (
            history((1, "0"), loan_id="A")
            + history((1, "0"), loan_id="B")
            + history((2, "0"), loan_id="A")
        )
        with pytest.raises(LabelError, match="reappears"):
            list(iter_loan_histories(recs))

    def test_vintage_cohorts_partition_labels(self):
        labs = [
            label_history(history(*[(t, "0") for t in range(1, 40)], loan_id=f"L{i}"))
            for i in range(3)
        ]
        labs.append(label_history(history((1, "0"), (2, "3"), loan_id="BAD")))
        cohorts = vintage_cohorts(labs)
        assert set(cohorts) == {2005}
        c = cohorts[2005]
        assert c.n_loans == 4
        # the D90 loan has only 2 observed months: event known despite censoring risk
        assert c.n_d90 == 1
        assert c.n_censored == 0
        assert c.d90_rate == pytest.approx(0.25)


@pytest.fixture(scope="module")
def labels(tmp_path_factory):
    out = tmp_path_factory.mktemp("m2")
    _, perf_path = generate(FixtureConfig(n_loans=400, seed=99), out)
    return list(label_file(perf_path, horizon_months=36)), perf_path


class TestFixtureIntegration:
    def test_every_loan_labeled_once(self, labels):
        labs, _ = labels
        assert len(labs) == 400
        assert len({l.loan_id for l in labs}) == 400

    def test_credit_zb_loans_are_d90_true(self, labels):
        labs, perf_path = labels
        zb_loans = {
            r.loan_id
            for r in iter_performance(perf_path)
            if r.zb_code in DEFAULT_ZB_CODES
        }
        assert zb_loans  # fixture must contain defaults
        for lab in labs:
            if lab.loan_id in zb_loans and lab.default_month <= 36:
                assert lab.d90_within_horizon is True

    def test_crisis_vintages_have_higher_d90_rate(self, labels):
        labs, _ = labels
        cohorts = vintage_cohorts(labs)
        assert set(cohorts) == set(range(2004, 2009))
        calm = cohorts[2004]
        stressed_rate = max(cohorts[2006].d90_rate, cohorts[2007].d90_rate)
        assert stressed_rate > calm.d90_rate, (
            f"crisis vintages not riskier: {stressed_rate} vs {calm.d90_rate}"
        )

    def test_labels_are_deterministic(self, labels, tmp_path):
        labs, _ = labels
        _, perf2 = generate(FixtureConfig(n_loans=400, seed=99), tmp_path / "again")
        labs2 = list(label_file(perf2, horizon_months=36))
        assert labs == labs2
