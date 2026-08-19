from creditlab.fixtures import FixtureConfig, generate
from creditlab.loader import iter_performance
from creditlab.schema import DEFAULT_ZB_CODES, PREPAY_ZB_CODES, parse_dlq

CONFIG = FixtureConfig(n_loans=200, seed=13)


def test_deterministic_byte_identical(tmp_path):
    a1, p1 = generate(CONFIG, tmp_path / "run1")
    a2, p2 = generate(CONFIG, tmp_path / "run2")
    assert a1.read_bytes() == a2.read_bytes()
    assert p1.read_bytes() == p2.read_bytes()


def test_different_seed_differs(tmp_path):
    a1, _ = generate(CONFIG, tmp_path / "s13")
    a2, _ = generate(FixtureConfig(n_loans=200, seed=14), tmp_path / "s14")
    assert a1.read_bytes() != a2.read_bytes()


def test_outcome_mix_and_default_progression(tmp_path):
    _, perf_path = generate(CONFIG, tmp_path)
    histories: dict[str, list] = {}
    for rec in iter_performance(perf_path):
        histories.setdefault(rec.loan_id, []).append(rec)

    defaults = prepays = 0
    for loan_id, recs in histories.items():
        terminal = [r for r in recs if r.zb_code]
        assert len(terminal) <= 1, f"{loan_id}: multiple zero-balance rows"
        if not terminal:
            continue
        zb = terminal[0]
        assert zb is recs[-1], f"{loan_id}: rows after zero-balance"
        if zb.zb_code in DEFAULT_ZB_CODES:
            defaults += 1
            # progression must reach 90+ days (dlq >= 3) before the credit event
            dlqs = [parse_dlq(r.dlq_status) for r in recs[:-1]]
            assert any(d is not None and d >= 3 for d in dlqs), (
                f"{loan_id}: credit-event ZB without a D90+ progression"
            )
            assert zb.zb_date is not None
            assert zb.disposition_date is not None
        elif zb.zb_code in PREPAY_ZB_CODES:
            prepays += 1
            assert zb.cur_upb == 0.0

    # the hazard model must produce a meaningful mix, or downstream
    # milestones (labels, models) have nothing to learn from
    assert defaults >= 5, f"only {defaults} defaults generated"
    assert prepays >= 10, f"only {prepays} prepays generated"


def test_self_curing_blips_exist(tmp_path):
    """30-day delinquency blips that cure must exist — they are the false
    positives the D90+ labeler (milestone 2) must not count as defaults."""
    _, perf_path = generate(CONFIG, tmp_path)
    histories: dict[str, list] = {}
    for rec in iter_performance(perf_path):
        histories.setdefault(rec.loan_id, []).append(rec)

    blips = 0
    for recs in histories.values():
        dlqs = [parse_dlq(r.dlq_status) for r in recs]
        for i in range(len(dlqs) - 1):
            if dlqs[i] == 1 and dlqs[i + 1] == 0:
                blips += 1
    assert blips >= 3, f"only {blips} self-curing blips generated"


def test_crisis_years_raise_default_rate(tmp_path):
    """The embedded crisis multiplier must be visible in aggregate — this is
    the signal milestone 6's out-of-time validation exists to detect."""
    calm = FixtureConfig(n_loans=400, seed=99, crisis_years=())
    stressed = FixtureConfig(n_loans=400, seed=99)

    def default_count(cfg, out):
        _, perf = generate(cfg, out)
        return sum(1 for r in iter_performance(perf) if r.zb_code in DEFAULT_ZB_CODES)

    d_calm = default_count(calm, tmp_path / "calm")
    d_stressed = default_count(stressed, tmp_path / "stressed")
    assert d_stressed > d_calm * 1.5, (
        f"crisis multiplier not visible: {d_stressed} vs {d_calm}"
    )
