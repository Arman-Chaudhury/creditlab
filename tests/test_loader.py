import gzip

import pytest

from creditlab.fixtures import FixtureConfig, generate
from creditlab.loader import LoadStats, chunked, iter_acquisitions, iter_performance
from creditlab.schema import SchemaError

CONFIG = FixtureConfig(n_loans=120, seed=7)


@pytest.fixture(scope="module")
def fixture_files(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixtures")
    return generate(CONFIG, out)


def test_acquisitions_roundtrip(fixture_files):
    acq_path, _ = fixture_files
    records = list(iter_acquisitions(acq_path))
    assert len(records) == CONFIG.n_loans
    for r in records:
        assert r.loan_id and r.loan_id.startswith("CLB")
        assert 580 <= r.fico <= 820
        assert 40.0 <= r.oltv <= 97.0
        assert r.orig_term == 360
        assert CONFIG.first_vintage <= r.orig_date.year <= CONFIG.last_vintage


def test_performance_roundtrip_and_referential_integrity(fixture_files):
    acq_path, perf_path = fixture_files
    loan_ids = {r.loan_id for r in iter_acquisitions(acq_path)}
    n = 0
    seen = set()
    for rec in iter_performance(perf_path):
        n += 1
        seen.add(rec.loan_id)
        assert rec.loan_id in loan_ids
        assert rec.period is not None
        assert rec.loan_age >= 1
    assert n > CONFIG.n_loans  # multiple monthly rows per loan
    assert seen == loan_ids  # every loan has a performance history


def test_gzip_transparent(fixture_files, tmp_path):
    acq_path, _ = fixture_files
    gz = tmp_path / "acquisition.txt.gz"
    gz.write_bytes(gzip.compress(acq_path.read_bytes()))
    assert len(list(iter_acquisitions(gz))) == CONFIG.n_loans


def test_strict_mode_raises_with_location(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("A|B|C\n", encoding="utf-8")
    with pytest.raises(SchemaError, match=r"bad\.txt:1.*expected 25.*got 3"):
        list(iter_acquisitions(bad))


def test_strict_mode_raises_on_bad_type(tmp_path, fixture_files):
    acq_path, _ = fixture_files
    first = acq_path.read_text(encoding="utf-8").splitlines()[0].split("|")
    first[12] = "not-a-fico"
    bad = tmp_path / "badtype.txt"
    bad.write_text("|".join(first) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="badtype.txt:1"):
        list(iter_acquisitions(bad))


def test_lenient_mode_skips_and_counts(tmp_path, fixture_files):
    acq_path, _ = fixture_files
    lines = acq_path.read_text(encoding="utf-8").splitlines()
    lines.insert(1, "way|too|few")
    mixed = tmp_path / "mixed.txt"
    mixed.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats = LoadStats()
    records = list(iter_acquisitions(mixed, strict=False, stats=stats))
    assert len(records) == CONFIG.n_loans
    assert stats.rows_read == CONFIG.n_loans + 1
    assert stats.rows_ok == CONFIG.n_loans
    assert stats.rows_skipped == 1
    assert "expected 25" in stats.errors[0]


def test_chunked_batches_preserve_order():
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(chunked([], 3)) == []
    with pytest.raises(ValueError):
        list(chunked([1], 0))


def test_loader_is_lazy(fixture_files):
    _, perf_path = fixture_files
    it = iter_performance(perf_path)
    first = next(it)  # consuming one row must not require reading the file fully
    assert first.loan_id
    it.close()
