import pytest

from creditlab.baseline import evaluate, random_split, train_logistic
from creditlab.challenger import (
    GBMModel,
    compare_models,
    has_lightgbm,
    train_gbm,
)
from creditlab.features import Matrix, build_dataset, to_matrix
from creditlab.fixtures import FixtureConfig, generate

needs_lightgbm = pytest.mark.skipif(not has_lightgbm(), reason="lightgbm not importable")


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    out = tmp_path_factory.mktemp("m5")
    acq, perf = generate(FixtureConfig(n_loans=800, seed=99), out)
    rows, _ = build_dataset(acq, perf, horizon_months=36)
    return random_split(to_matrix(rows), test_frac=0.25, seed=7)


@pytest.fixture(scope="module")
def gbm(split):
    train, _ = split
    return train_gbm(train)  # auto backend


@pytest.fixture(scope="module")
def baseline_report(split):
    train, test = split
    return evaluate(train_logistic(train), test)


class TestTraining:
    def test_challenger_is_competitive(self, gbm, split, baseline_report):
        """On the fixtures the *baseline should win*: the synthetic hazard is
        exactly logistic in fico/oltv/dti, so the correctly-specified
        parametric model is the true model, and a tree ensemble on a few
        hundred rows approximates it with a small AUC deficit. The test pins
        'same league, not broken' (observed gap ~0.05); which model wins on
        real data is milestone 6's empirical question."""
        _, test = split
        report = evaluate(gbm, test)
        assert report.auc is not None and report.auc > 0.55
        assert report.auc >= baseline_report.auc - 0.08, (
            f"challenger AUC {report.auc:.4f} far below baseline "
            f"{baseline_report.auc:.4f}"
        )

    def test_beats_climatology(self, gbm, split):
        _, test = split
        report = evaluate(gbm, test)
        assert report.brier_skill is not None and report.brier_skill > 0.0

    def test_probabilities_valid(self, gbm, split):
        _, test = split
        assert all(0.0 <= p <= 1.0 for p in gbm.predict_proba(test.X))

    def test_deterministic(self, split):
        train, test = split
        p1 = train_gbm(train, n_estimators=80).predict_proba(test.X)
        p2 = train_gbm(train, n_estimators=80).predict_proba(test.X)
        assert p1 == p2

    def test_importances_normalized_and_sane(self, gbm):
        assert set(gbm.importances) == set(gbm.feature_names)
        assert sum(gbm.importances.values()) == pytest.approx(1.0)
        assert all(v >= 0.0 for v in gbm.importances.values())
        # fico is the strongest lever in the fixture hazard; the challenger
        # must notice it
        assert "fico" in gbm.top_features(5), (
            f"fico missing from top features: {gbm.top_features(5)}"
        )

    def test_wrong_width_rejected(self, gbm):
        with pytest.raises(ValueError, match="expected"):
            gbm.predict_proba([[1.0, 2.0]])

    def test_single_class_raises(self, split):
        train, _ = split
        degenerate = Matrix(
            X=train.X[:5], y=[0] * 5,
            feature_names=train.feature_names,
            loan_ids=train.loan_ids[:5], vintages=train.vintages[:5],
        )
        with pytest.raises(ValueError, match="single class"):
            train_gbm(degenerate)

    def test_unknown_backend_raises(self, split):
        train, _ = split
        with pytest.raises(ValueError, match="backend"):
            train_gbm(train, backend="bogus")


class TestSklearnBackend:
    def test_forced_sklearn_backend_works_everywhere(self, split):
        train, test = split
        model = train_gbm(train, backend="sklearn-hist", n_estimators=50)
        assert model.backend == "sklearn-hist"
        assert model.importance_type == "permutation"
        report = evaluate(model, test)
        assert report.auc is not None and report.auc > 0.55

    def test_sklearn_artifact_not_serializable(self, split):
        train, _ = split
        model = train_gbm(train, backend="sklearn-hist", n_estimators=50)
        with pytest.raises(NotImplementedError, match="train-in-session"):
            model.to_dict()


@needs_lightgbm
class TestLightgbmBackend:
    def test_artifact_roundtrip(self, split):
        train, test = split
        model = train_gbm(train, backend="lightgbm")
        assert model.backend == "lightgbm"
        assert model.importance_type == "gain"
        clone = GBMModel.from_dict(model.to_dict())
        assert clone.predict_proba(test.X) == pytest.approx(
            model.predict_proba(test.X)
        )

    def test_artifact_is_json_safe(self, split):
        import json

        train, _ = split
        model = train_gbm(train, backend="lightgbm")
        blob = json.dumps(model.to_dict())
        assert GBMModel.from_dict(json.loads(blob)).feature_names == model.feature_names


def test_lightgbm_requested_but_missing_raises(split, monkeypatch):
    import creditlab.challenger as ch

    monkeypatch.setattr(ch, "has_lightgbm", lambda: False)
    train, _ = split
    with pytest.raises(RuntimeError, match="lightgbm requested"):
        ch.train_gbm(train, backend="lightgbm")


class TestComparison:
    def test_deltas_and_summary(self, gbm, split, baseline_report):
        _, test = split
        challenger_report = evaluate(gbm, test)
        comp = compare_models(baseline_report, challenger_report, backend=gbm.backend)
        assert comp.auc_delta == pytest.approx(
            challenger_report.auc - baseline_report.auc
        )
        assert comp.brier_delta == pytest.approx(
            challenger_report.brier - baseline_report.brier
        )
        text = comp.summary()
        assert "AUC" in text and "Brier" in text and gbm.backend in text
        d = comp.to_dict()
        assert d["challenger_backend"] == gbm.backend
        assert d["baseline"]["n"] == baseline_report.n
