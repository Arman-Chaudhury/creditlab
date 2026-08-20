from datetime import date

from creditlab.schema import (
    ACQUISITION_SCHEMA,
    PERFORMANCE_SCHEMA,
    parse_dlq,
)


def test_column_counts_match_classic_layout():
    assert len(ACQUISITION_SCHEMA) == 25
    assert len(PERFORMANCE_SCHEMA) == 31


def test_no_duplicate_field_names():
    for schema in (ACQUISITION_SCHEMA, PERFORMANCE_SCHEMA):
        names = [name for name, _ in schema]
        assert len(names) == len(set(names))


def test_parse_dlq():
    assert parse_dlq("0") == 0
    assert parse_dlq("3") == 3
    assert parse_dlq("12") == 12
    assert parse_dlq("X") is None
    assert parse_dlq("x") is None
    assert parse_dlq("") is None
    assert parse_dlq(None) is None


def test_date_converters():
    conv = dict(ACQUISITION_SCHEMA)
    assert conv["orig_date"]("02/2006") == date(2006, 2, 1)
    assert conv["orig_date"]("") is None
    pconv = dict(PERFORMANCE_SCHEMA)
    assert pconv["period"]("03/01/2007") == date(2007, 3, 1)
