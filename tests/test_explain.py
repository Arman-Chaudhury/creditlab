import math

import pytest

from creditlab.baseline import LogisticModel, random_split, train_logistic
from creditlab.explain import (
    LoanExplanation,
    attach_groups,
    contributions,
    explain_loan,
    proxy_audit,
    reason_codes,
)
from creditlab.features import Matrix, build_dataset, to_matrix
from creditlab.fixtures import FixtureConfig, generate


def toy_model():
    """Standardization is identity (mean 0, std 1) so contributions are
    coef * value — hand-checkable."""
    return LogisticModel(
        feature_names=("fico", "oltv", "dti"),
        means=(0.0, 0.0, 0.0),
        stds=(1.0, 1.0, 1.0),
        coefs=(-1.0, 0.5, 0.25),
        intercept=-2.0,
    )


class TestContributions:
    def test_exact_decomposition(self):
        m = toy_model()
        x = [-1.5, 2.0, 1.0]  # fico 1.5 sd below mean, oltv 2 sd above, ...
        c = contributions(m, x)
        assert c == {"fico": 1.5, "oltv": 1.0, "dti": 0.25}
        # contributions + intercept reproduce the exact log-odds / PD
        z = m.intercept + sum(c.values())
        assert m.predict_proba([x])[0] == pytest.approx(1 / (1 + math.exp(-z)))

    def test_wrong_width_raises(self):
        with pytest.raises(ValueError, match="expected 3"):
            contributions(toy_model(), [1.0])


class TestReasonCodes:
    def test_ordered_risk_increasing_only(self):
        m = toy_model()
        # fico +1.5, oltv +1.0, dti +0.25 — all positive
        codes = reason_codes(m, [-1.5, 2.0, 1.0], top_n=2)
        assert [c.feature for c in codes] == ["fico", "oltv"]  # top_n respected
        assert codes[0].contribution == pytest.approx(1.5)
        assert codes[0].phrase.startswith("Credit score")
        assert codes[0].value == -1.5

    def test_protective_factors_never_cited(self):
        m = toy_model()
        # high fico (protective, contribution -2), low oltv (protective),
        # only dti pushes risk up
        codes = reason_codes(m, [2.0, -1.0, 0.8])
        assert [c.feature for c in codes] == ["dti"]

    def test_no_positive_contributions_yields_empty(self):
        m = toy_model()
        codes = reason_codes(m, [2.0, -1.0, -0.5])
        assert codes == []

    def test_one_hot_fallback_phrase(self):
        m = LogisticModel(
            feature_names=("purpose=C",), means=(0.0,), stds=(1.0,),
            coefs=(0.7,), intercept=-3.0,
        )
        codes = reason_codes(m, [1.0])
        assert codes[0].phrase == "Loan characteristic: purpose is C"


@pytest.fixture(scope="module")
def fixture_setup(tmp_path_factory):
    out = tmp_path_factory.mktemp("m7")
    acq_path, perf_path = generate(FixtureConfig(n_loans=800, seed=99), out)
    rows, _ = build_dataset(acq_path, perf_path, horizon_months=36)
    matrix = to_matrix(rows)
    train, _ = random_split(matrix, test_frac=0.25, seed=7)
    model = train_logistic(train)
    return acq_path, matrix, model


class TestExplainLoan:
    def test_explains_real_loan_exactly(self, fixture_setup):
        _, matrix, model = fixture_setup
        loan_id = matrix.loan_ids[0]
        exp = explain_loan(model, matrix, loan_id)
        assert isinstance(exp, LoanExplanation)
        assert exp.loan_id == loan_id
        # the additive decomposition reproduces the model's own PD exactly
        assert exp.predicted_pd == pytest.approx(
            1 / (1 + math.exp(-exp.log_odds))
        )
        assert len(exp.reasons) <= 4
        assert all(r.contribution > 0 for r in exp.reasons)

    def test_riskiest_loan_cites_core_drivers(self, fixture_setup):
        _, matrix, model = fixture_setup
        probs = model.predict_proba(matrix.X)
        worst = matrix.loan_ids[probs.index(max(probs))]
        exp = explain_loan(model, matrix, worst, top_n=5)
        cited = {r.feature for r in exp.reasons}
        # the riskiest fixture loan must cite at least one core hazard driver
        assert cited & {"fico", "oltv", "ocltv", "dti", "mi_pct", "orig_rate"}

    def test_missing_loan_raises(self, fixture_setup):
        _, matrix, model = fixture_setup
        with pytest.raises(KeyError, match="nope"):
            explain_loan(model, matrix, "nope")


class TestProxyAudit:
    def test_groups_partition_and_gap_arithmetic(self, fixture_setup):
        acq_path, matrix, model = fixture_setup
        states = attach_groups(acq_path, matrix)
        report = proxy_audit(model, matrix, states)
        assert sum(g.n for g in report.groups) == len(matrix.X)
        for g in report.groups:
            assert g.gap == pytest.approx(g.mean_predicted - g.realized)

    def test_fixture_features_are_not_geography_proxies(self, fixture_setup):
        """Fixture features are sampled independently of state, so no feature
        should reconstruct geography (PSI < 0.25 across all groups)."""
        acq_path, matrix, model = fixture_setup
        states = attach_groups(acq_path, matrix)
        report = proxy_audit(model, matrix, states)
        worst = max(report.feature_proxy_power.values())
        assert worst < 0.25, (
            f"unexpected proxy: {report.top_proxy_features(3)}"
        )

    def test_planted_proxy_is_detected(self, fixture_setup):
        """Plant a feature that encodes group membership; the audit must
        flag it with high PSI."""
        acq_path, matrix, model = fixture_setup
        states = attach_groups(acq_path, matrix)
        planted = Matrix(
            X=[row + [100.0 if s == "CA" else 0.0] for row, s in zip(matrix.X, states)],
            y=matrix.y,
            feature_names=[*matrix.feature_names, "planted_proxy"],
            loan_ids=matrix.loan_ids,
            vintages=matrix.vintages,
        )

        class DummyModel:  # audit only needs predict_proba
            def predict_proba(self, X):
                return [0.1] * len(X)

        report = proxy_audit(DummyModel(), planted, states)
        assert report.feature_proxy_power["planted_proxy"] > 0.25
        assert report.top_proxy_features(1)[0][0] == "planted_proxy"

    def test_small_groups_pooled_into_other(self, fixture_setup):
        acq_path, matrix, model = fixture_setup
        states = attach_groups(acq_path, matrix)
        report = proxy_audit(model, matrix, states, min_group_size=10_000)
        assert [g.group for g in report.groups] == ["OTHER"]
        assert report.groups[0].n == len(matrix.X)

    def test_summary_and_dict(self, fixture_setup):
        acq_path, matrix, model = fixture_setup
        states = attach_groups(acq_path, matrix)
        report = proxy_audit(model, matrix, states)
        text = report.summary()
        assert "proxy-fairness audit by state" in text
        assert "gap=" in text
        import json

        blob = json.loads(json.dumps(report.to_dict()))
        assert blob["group_field"] == "state"

    def test_mismatched_lengths_raise(self, fixture_setup):
        _, matrix, model = fixture_setup
        with pytest.raises(ValueError, match="group labels"):
            proxy_audit(model, matrix, ["CA"])
