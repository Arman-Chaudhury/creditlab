import pytest

from creditlab.metrics import (
    auc,
    brier_score,
    calibration_in_the_large,
    decile_table,
    reliability_curve,
)


class TestAUC:
    def test_perfect_ranking(self):
        assert auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0

    def test_perfectly_wrong(self):
        assert auc([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]) == 0.0

    def test_known_hand_computed_value(self):
        # positive-negative pairs: (0.9,0.8)=1, (0.9,0.1)=1, (0.7,0.8)=0, (0.7,0.1)=1
        assert auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.75)

    def test_all_tied_scores_give_half(self):
        assert auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_partial_ties_use_average_ranks(self):
        # pairs: (0.5 vs 0.5)=0.5, (0.5 vs 0.2)=1 -> 1.5/2
        assert auc([1, 0, 0], [0.5, 0.5, 0.2]) == pytest.approx(0.75)

    def test_single_class_is_none(self):
        assert auc([1, 1], [0.2, 0.9]) is None
        assert auc([0, 0], [0.2, 0.9]) is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            auc([1, 0], [0.5])


class TestBrier:
    def test_exact_value(self):
        assert brier_score([1, 0], [0.8, 0.3]) == pytest.approx(0.065)

    def test_perfect_is_zero(self):
        assert brier_score([1, 0], [1.0, 0.0]) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            brier_score([], [])


def test_calibration_in_the_large():
    mean_pred, realized = calibration_in_the_large([1, 0, 0, 0], [0.4, 0.2, 0.2, 0.2])
    assert mean_pred == pytest.approx(0.25)
    assert realized == pytest.approx(0.25)


class TestReliability:
    def test_quantile_bins_have_near_equal_counts(self):
        y = [0, 1, 0, 1, 0, 1, 0]
        p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        bins = reliability_curve(y, p, n_bins=3)
        assert [b.n for b in bins] == [3, 2, 2]
        assert sum(b.n for b in bins) == 7
        # bins ordered by probability
        assert bins[0].mean_predicted < bins[1].mean_predicted < bins[2].mean_predicted

    def test_quantile_bin_values(self):
        bins = reliability_curve([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], n_bins=2)
        assert bins[0].mean_predicted == pytest.approx(0.15)
        assert bins[0].realized == 0.0
        assert bins[1].mean_predicted == pytest.approx(0.8)
        assert bins[1].realized == 1.0

    def test_more_bins_than_points_collapses(self):
        bins = reliability_curve([0, 1], [0.2, 0.8], n_bins=10)
        assert len(bins) == 2

    def test_uniform_strategy_drops_empty_bins(self):
        bins = reliability_curve([0, 1], [0.05, 0.95], n_bins=10, strategy="uniform")
        assert len(bins) == 2
        assert bins[0].p_lo == 0.0 and bins[0].p_hi == pytest.approx(0.1)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            reliability_curve([0, 1], [0.1, 0.9], strategy="bogus")


class TestDecileTable:
    def test_riskiest_decile_first_and_lift(self):
        # 10 loans, top-scored one defaults, rest don't
        y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        p = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        rows = decile_table(y, p)
        assert len(rows) == 10
        assert rows[0].decile == 1
        assert rows[0].realized == 1.0
        assert rows[0].lift == pytest.approx(10.0)  # 1.0 / 0.1 overall
        assert all(r.realized == 0.0 for r in rows[1:])

    def test_counts_partition_input(self):
        y = [0] * 23 + [1] * 2
        p = [i / 25 for i in range(25)]
        rows = decile_table(y, p)
        assert sum(r.n for r in rows) == 25
