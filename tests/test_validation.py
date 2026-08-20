import pytest

from creditlab.challenger import train_gbm
from creditlab.features import build_dataset, to_matrix
from creditlab.fixtures import FixtureConfig, generate
from creditlab.validation import (
    crisis_validation,
    feature_psi,
    psi,
    split_by_vintage,
)


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    out = tmp_path_factory.mktemp("m6")
    acq, perf = generate(FixtureConfig(n_loans=800, seed=99), out)
    rows, _ = build_dataset(acq, perf, horizon_months=36)
    return to_matrix(rows)


@pytest.fixture(scope="module")
def report(matrix):
    return crisis_validation(matrix)  # train <=2006, test 2007-08, logistic


class TestPSI:
    def test_identical_distributions_are_zero(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        assert psi(vals, list(vals)) == pytest.approx(0.0, abs=1e-9)

    def test_hand_computed_two_bin_case(self):
        # expected [.5,.5] vs actual [.25,.75]:
        # (.25-.5)ln(.25/.5) + (.75-.5)ln(.75/.5) = 0.27465
        value = psi([1.0, 1.0, 2.0, 2.0], [1.0, 2.0, 2.0, 2.0], n_bins=2)
        assert value == pytest.approx(0.274653, abs=1e-5)

    def test_shift_is_positive_and_grows(self):
        base = [float(i) for i in range(100)]
        small = [v + 5.0 for v in base]
        large = [v + 50.0 for v in base]
        assert 0.0 < psi(base, small) < psi(base, large)

    def test_constant_feature_handled(self):
        assert psi([1.0] * 50, [1.0] * 50) == pytest.approx(0.0)
        # a constant that moved is a major shift, finite not infinite
        moved = psi([1.0] * 50, [2.0] * 50)
        assert 0.25 < moved < float("inf")

    def test_symmetric_floor_keeps_finite(self):
        # actual entirely outside expected's range -> emptied bins, finite PSI
        assert psi([1.0, 2.0, 3.0], [10.0, 11.0, 12.0]) < float("inf")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            psi([], [1.0])


class TestVintageSplit:
    def test_boundary_year_trains(self, matrix):
        train, test = split_by_vintage(matrix, train_max_vintage=2006)
        assert set(train.vintages) == {2004, 2005, 2006}
        assert set(test.vintages) == {2007, 2008}
        assert len(train.X) + len(test.X) == len(matrix.X)
        assert set(train.loan_ids).isdisjoint(test.loan_ids)

    def test_empty_side_raises(self, matrix):
        with pytest.raises(ValueError, match="empty side"):
            split_by_vintage(matrix, train_max_vintage=1990)
        with pytest.raises(ValueError, match="empty side"):
            split_by_vintage(matrix, train_max_vintage=2030)


class TestCrisisValidation:
    def test_in_sample_is_calibrated(self, report):
        # logistic with intercept matches its own training base rate
        assert abs(report.in_sample.mean_predicted - report.in_sample.realized_rate) < 0.01

    def test_crisis_vintages_are_riskier(self, report):
        assert report.oot.realized_rate > report.in_sample.realized_rate

    def test_headline_underprediction(self, report):
        """The finding the project exists to demonstrate: on unseen crisis
        vintages the model underpredicts (measured 1.31 on the fixtures),
        while the naive random split shows no such warning (measured 0.80)."""
        assert report.oot_underprediction > 1.15
        assert report.naive_underprediction < 1.10
        assert report.oot_underprediction > report.naive_underprediction + 0.2

    def test_discrimination_survives_while_calibration_breaks(self, report):
        # the classic pattern: ranking power holds up far better than levels
        assert report.oot.auc is not None and report.oot.auc > 0.55
        assert report.oot.auc <= report.naive.auc + 0.02

    def test_drift_shows_in_rate_variables_not_borrower_quality(self, report):
        # fixture vintages differ by base rate, not by borrower mix
        assert report.score_psi > 0.02
        assert report.feature_psi["fico"] < 0.10  # stable population
        assert report.feature_psi["orig_rate"] > report.feature_psi["fico"]
        top = [name for name, _ in report.top_drifted_features(3)]
        assert "orig_rate" in top or "rate_spread" in top

    def test_headline_table_renders(self, report):
        table = report.headline_table()
        assert "OOT 2007-2008" in table
        assert "realized / predicted" in table
        assert "AUC" in table
        assert table.count("\n") >= 8  # header + separator + 7 metric rows

    def test_to_dict_json_safe(self, report):
        import json

        blob = json.loads(json.dumps(report.to_dict()))
        assert blob["train_max_vintage"] == 2006
        assert blob["test_vintages"] == [2007, 2008]
        assert blob["oot_underprediction"] == pytest.approx(
            report.oot_underprediction
        )

    def test_deterministic(self, matrix, report):
        again = crisis_validation(matrix)
        assert again.to_dict() == report.to_dict()

    def test_works_with_gbm_trainer(self, matrix):
        gbm_report = crisis_validation(matrix, trainer=train_gbm)
        assert gbm_report.oot.n > 0
        assert gbm_report.oot.auc is not None and gbm_report.oot.auc > 0.55
        # the ensemble is no more clairvoyant about unseen vintages
        assert gbm_report.oot_underprediction > 1.0


def test_feature_psi_keys_match(matrix):
    train, test = split_by_vintage(matrix, train_max_vintage=2006)
    fp = feature_psi(train, test)
    assert set(fp) == set(matrix.feature_names)
    assert all(v >= 0.0 for v in fp.values())
