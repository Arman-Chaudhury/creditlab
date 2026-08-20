import pytest

from creditlab.baseline import (
    LogisticModel,
    evaluate,
    random_split,
    train_logistic,
)
from creditlab.features import Matrix, build_dataset, to_matrix
from creditlab.fixtures import FixtureConfig, generate


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    out = tmp_path_factory.mktemp("m4")
    acq, perf = generate(FixtureConfig(n_loans=800, seed=99), out)
    rows, _ = build_dataset(acq, perf, horizon_months=36)
    return to_matrix(rows)


@pytest.fixture(scope="module")
def split(matrix):
    return random_split(matrix, test_frac=0.25, seed=7)


@pytest.fixture(scope="module")
def model(split):
    train, _ = split
    return train_logistic(train)


class TestSplit:
    def test_disjoint_and_complete(self, matrix, split):
        train, test = split
        assert len(train.X) + len(test.X) == len(matrix.X)
        assert set(train.loan_ids).isdisjoint(test.loan_ids)
        assert set(train.loan_ids) | set(test.loan_ids) == set(matrix.loan_ids)

    def test_deterministic(self, matrix):
        a = random_split(matrix, seed=7)
        b = random_split(matrix, seed=7)
        assert a[1].loan_ids == b[1].loan_ids

    def test_bad_frac_raises(self, matrix):
        with pytest.raises(ValueError):
            random_split(matrix, test_frac=0.0)


class TestTraining:
    def test_model_learns_the_right_physics(self, model):
        """The fixture hazard rises with LTV/DTI and falls with FICO.

        FICO and DTI signs are individually identified. LTV is not: oltv,
        ocltv, and mi_pct are strongly collinear (ocltv = oltv + a spread;
        MI only exists above 80 LTV), so L2 splits the leverage weight
        arbitrarily among them — assert the *group* direction, which is what
        regularized regression actually identifies."""
        coefs = model.coefficients()
        assert coefs["fico"] < 0, f"fico coef {coefs['fico']} should be negative"
        assert coefs["dti"] > 0, f"dti coef {coefs['dti']} should be positive"
        ltv_block = coefs["oltv"] + coefs["ocltv"] + coefs["mi_pct"]
        assert ltv_block > 0, f"LTV block coef {ltv_block} should be positive"

    def test_discrimination_out_of_sample(self, model, split):
        _, test = split
        report = evaluate(model, test)
        assert report.auc is not None and report.auc > 0.60, (
            f"held-out AUC {report.auc} — model found no signal"
        )

    def test_beats_climatology_on_brier(self, model, split):
        train, test = split
        for m in (train, test):
            report = evaluate(model, m)
            assert report.brier_skill is not None and report.brier_skill > 0.0

    def test_calibrated_in_the_large_on_train(self, model, split):
        train, _ = split
        report = evaluate(model, train)
        # logistic regression with an intercept matches the base rate on train
        assert abs(report.mean_predicted - report.realized_rate) < 0.01

    def test_reasonably_calibrated_on_test(self, model, split):
        _, test = split
        report = evaluate(model, test)
        assert abs(report.mean_predicted - report.realized_rate) < 0.06

    def test_decile_monotonicity_ends(self, model, split):
        _, test = split
        report = evaluate(model, test)
        assert report.deciles[0].realized >= report.deciles[-1].realized

    def test_training_is_deterministic(self, split):
        train, _ = split
        m1 = train_logistic(train)
        m2 = train_logistic(train)
        assert m1.coefs == m2.coefs and m1.intercept == m2.intercept

    def test_single_class_raises(self, matrix):
        degenerate = Matrix(
            X=matrix.X[:5],
            y=[0, 0, 0, 0, 0],
            feature_names=matrix.feature_names,
            loan_ids=matrix.loan_ids[:5],
            vintages=matrix.vintages[:5],
        )
        with pytest.raises(ValueError, match="single class"):
            train_logistic(degenerate)

    def test_empty_matrix_raises(self, matrix):
        empty = Matrix([], [], matrix.feature_names, [], [])
        with pytest.raises(ValueError, match="empty"):
            train_logistic(empty)


class TestArtifact:
    def test_dict_roundtrip_preserves_predictions(self, model, split):
        _, test = split
        clone = LogisticModel.from_dict(model.to_dict())
        assert clone.predict_proba(test.X) == model.predict_proba(test.X)

    def test_artifact_is_json_safe(self, model):
        import json

        blob = json.dumps(model.to_dict())
        clone = LogisticModel.from_dict(json.loads(blob))
        assert clone == model

    def test_wrong_kind_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            LogisticModel.from_dict({"kind": "gbm"})

    def test_wrong_width_rejected(self, model):
        with pytest.raises(ValueError, match="expected"):
            model.predict_proba([[1.0, 2.0]])

    def test_probabilities_in_unit_interval(self, model, split):
        _, test = split
        assert all(0.0 < p < 1.0 for p in model.predict_proba(test.X))
