import math
from datetime import date

import pytest

from creditlab.features import (
    FEATURE_SPEC,
    Feature,
    FeatureStats,
    LeakageError,
    audit_leakage,
    benchmark_rate,
    build_dataset,
    build_row,
    matrix_columns,
    to_matrix,
)
from creditlab.fixtures import FixtureConfig, generate
from creditlab.labels import LoanLabel
from creditlab.schema import AcquisitionRecord


def acq(**overrides):
    fields = dict.fromkeys(AcquisitionRecord.__dataclass_fields__, None)
    fields.update(
        loan_id="L1",
        orig_rate=6.5,
        orig_upb=300_000.0,
        orig_date=date(2006, 3, 1),
        oltv=80.0,
        dti=35.0,
        fico=700,
        purpose="P",
        occupancy="P",
        property_type="SF",
        channel="R",
    )
    fields.update(overrides)
    return AcquisitionRecord(**fields)


def label(y=False, vintage=2006, censored=False):
    return LoanLabel(
        loan_id="L1",
        vintage=vintage,
        orig_date=date(vintage, 3, 1),
        observed_months=40,
        d90_within_horizon=None if censored else y,
        d180_within_horizon=None if censored else y,
        defaulted=y,
        default_month=10 if y else None,
        prepaid=False,
        prepay_month=None,
    )


class TestLeakageAudit:
    def test_shipped_spec_is_clean(self):
        audit_leakage()  # must not raise

    def test_performance_field_is_rejected(self):
        leaky = Feature("cur_dlq", "numeric", ("dlq_status",), lambda a: 0.0)
        with pytest.raises(LeakageError, match="cur_dlq.*dlq_status"):
            audit_leakage([*FEATURE_SPEC, leaky])

    def test_sourceless_feature_is_rejected(self):
        vague = Feature("mystery", "numeric", (), lambda a: 0.0)
        with pytest.raises(LeakageError, match="mystery"):
            audit_leakage([vague])


class TestBuildRow:
    def test_basic_values(self):
        row = build_row(acq(), label())
        assert row is not None
        assert row.numeric["fico"] == 700.0
        assert row.numeric["log_upb"] == pytest.approx(math.log(300_000))
        # 2006 benchmark is 6.41 -> spread 0.09
        assert row.numeric["rate_spread"] == pytest.approx(6.5 - 6.41)
        assert row.categorical == {
            "purpose": "P", "occupancy": "P", "property_type": "SF", "channel": "R",
        }
        assert row.y is False and row.vintage == 2006

    def test_ocltv_falls_back_to_oltv(self):
        row = build_row(acq(ocltv=None, oltv=85.0), label())
        assert row.numeric["ocltv"] == 85.0
        row2 = build_row(acq(ocltv=95.0, oltv=85.0), label())
        assert row2.numeric["ocltv"] == 95.0

    def test_optional_fallbacks(self):
        row = build_row(acq(mi_pct=None, num_borrowers=None, co_fico=None), label())
        assert row.numeric["mi_pct"] == 0.0
        assert row.numeric["num_borrowers"] == 1.0
        assert row.numeric["has_coborrower"] == 0.0

    def test_missing_required_excludes_and_counts(self):
        stats = FeatureStats()
        assert build_row(acq(fico=None), label(), stats=stats) is None
        assert stats.n_missing_required == 1
        assert stats.missing_by_field == {"fico": 1}

    def test_censored_label_excluded(self):
        stats = FeatureStats()
        assert build_row(acq(), label(censored=True), stats=stats) is None
        assert stats.n_censored_excluded == 1

    def test_unknown_category_bucketed(self):
        row = build_row(acq(purpose="Z"), label())
        assert row.categorical["purpose"] == "UNK"

    def test_d180_target(self):
        row = build_row(acq(), label(y=True), target="d180")
        assert row.y is True


class TestBenchmark:
    def test_known_year(self):
        assert benchmark_rate(2006) == 6.41

    def test_out_of_table_year_uses_nearest(self):
        assert benchmark_rate(1990) == benchmark_rate(1999)
        assert benchmark_rate(2030) == benchmark_rate(2024)

    def test_custom_table(self):
        assert benchmark_rate(2006, {2006: 5.0}) == 5.0


class TestMatrix:
    def test_column_order_deterministic(self):
        assert matrix_columns() == matrix_columns()
        cols = matrix_columns()
        assert cols.index("fico") < cols.index("purpose=P")
        assert "occupancy=UNK" in cols

    def test_one_hot_encoding(self):
        rows = [build_row(acq(), label(y=True))]
        m = to_matrix(rows)
        assert len(m.X) == 1 and m.y == [1]
        vec = dict(zip(m.feature_names, m.X[0]))
        assert vec["purpose=P"] == 1.0
        assert vec["purpose=C"] == 0.0
        assert vec["fico"] == 700.0
        assert sum(vec[c] for c in m.feature_names if c.startswith("purpose=")) == 1.0


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("m3")
    acq_path, perf_path = generate(FixtureConfig(n_loans=400, seed=99), out)
    return build_dataset(acq_path, perf_path, horizon_months=36)


class TestFixtureIntegration:
    def test_accounting_adds_up(self, dataset):
        rows, stats = dataset
        assert stats.n_acquisitions == 400
        assert stats.n_labels == 400
        assert stats.n_joined == 400
        assert stats.n_acq_without_label == 0
        assert stats.n_label_without_acq == 0
        assert (
            stats.n_rows + stats.n_censored_excluded + stats.n_missing_required
            == stats.n_joined
        )
        assert stats.n_rows == len(rows)
        assert stats.n_missing_required == 0  # fixtures always have required fields

    def test_default_rate_is_learnable(self, dataset):
        rows, _ = dataset
        rate = sum(r.y for r in rows) / len(rows)
        assert 0.01 < rate < 0.6, f"degenerate default rate {rate}"

    def test_matrix_shape_and_vintages(self, dataset):
        rows, _ = dataset
        m = to_matrix(rows)
        assert len(m.X) == len(rows) == len(m.y) == len(m.vintages)
        assert all(len(v) == len(m.feature_names) for v in m.X)
        assert set(m.vintages) <= set(range(2004, 2009))

    def test_deterministic(self, dataset, tmp_path):
        rows, _ = dataset
        a2, p2 = generate(FixtureConfig(n_loans=400, seed=99), tmp_path)
        rows2, _ = build_dataset(a2, p2, horizon_months=36)
        assert rows == rows2
