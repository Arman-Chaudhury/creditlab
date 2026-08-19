"""Fannie Mae Single-Family Loan Performance file schemas.

Implements the classic two-file layout (quarterly, pipe-delimited, no header):

- Acquisition file: one row per loan, 25 columns of origination attributes.
- Performance file: one row per loan per monthly reporting period, 31 columns.

Column order follows the published Fannie Mae glossary for the classic layout.
Fannie Mae's newer combined single-file format is out of scope for now; the
loader validates column counts, so a combined file fails fast with a clear
error instead of silently mis-parsing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional


class SchemaError(ValueError):
    """A row failed schema validation (wrong arity or unparseable field)."""


# --- field converters --------------------------------------------------------
# Every converter maps a raw pipe-delimited token to a typed value, with the
# empty string meaning "missing" (None) — the convention used in these files.

def _str(v: str) -> Optional[str]:
    v = v.strip()
    return v or None


def _float(v: str) -> Optional[float]:
    v = v.strip()
    return float(v) if v else None


def _int(v: str) -> Optional[int]:
    v = v.strip()
    return int(v) if v else None


def _date_mmyyyy(v: str) -> Optional[date]:
    v = v.strip()
    if not v:
        return None
    mm, yyyy = v.split("/")
    return date(int(yyyy), int(mm), 1)


def _date_mmddyyyy(v: str) -> Optional[date]:
    v = v.strip()
    if not v:
        return None
    mm, dd, yyyy = v.split("/")
    return date(int(yyyy), int(mm), int(dd))


def parse_dlq(raw: Optional[str]) -> Optional[int]:
    """Delinquency status -> months delinquent. 'X' (unknown) and missing -> None."""
    if raw is None:
        return None
    raw = raw.strip()
    if not raw or raw.upper() == "X":
        return None
    return int(raw)


# --- record types ------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    loan_id: Optional[str]
    channel: Optional[str]
    seller: Optional[str]
    orig_rate: Optional[float]
    orig_upb: Optional[float]
    orig_term: Optional[int]
    orig_date: Optional[date]
    first_pay_date: Optional[date]
    oltv: Optional[float]
    ocltv: Optional[float]
    num_borrowers: Optional[int]
    dti: Optional[float]
    fico: Optional[int]
    first_time_buyer: Optional[str]
    purpose: Optional[str]
    property_type: Optional[str]
    num_units: Optional[int]
    occupancy: Optional[str]
    state: Optional[str]
    zip3: Optional[str]
    mi_pct: Optional[float]
    product: Optional[str]
    co_fico: Optional[int]
    mi_type: Optional[str]
    relocation_flag: Optional[str]


@dataclass(frozen=True, slots=True)
class PerformanceRecord:
    loan_id: Optional[str]
    period: Optional[date]
    servicer: Optional[str]
    cur_rate: Optional[float]
    cur_upb: Optional[float]
    loan_age: Optional[int]
    months_to_maturity: Optional[int]
    adj_months_to_maturity: Optional[int]
    maturity_date: Optional[date]
    msa: Optional[str]
    dlq_status: Optional[str]  # raw token; use schema.parse_dlq() for the int view
    modification_flag: Optional[str]
    zb_code: Optional[str]  # 01 prepaid, 02 third-party sale, 03 short sale, 09 REO, ...
    zb_date: Optional[date]
    lpi_date: Optional[date]
    foreclosure_date: Optional[date]
    disposition_date: Optional[date]
    foreclosure_costs: Optional[float]
    preservation_costs: Optional[float]
    asset_recovery_costs: Optional[float]
    misc_holding_expenses: Optional[float]
    holding_taxes: Optional[float]
    net_sale_proceeds: Optional[float]
    credit_enhancement_proceeds: Optional[float]
    repurchase_make_whole_proceeds: Optional[float]
    other_foreclosure_proceeds: Optional[float]
    non_interest_bearing_upb: Optional[float]
    principal_forgiveness: Optional[float]
    rmw_proceeds_flag: Optional[str]
    foreclosure_writeoff: Optional[float]
    servicing_indicator: Optional[str]


Converter = Callable[[str], object]

ACQUISITION_SCHEMA: tuple[tuple[str, Converter], ...] = (
    ("loan_id", _str),
    ("channel", _str),
    ("seller", _str),
    ("orig_rate", _float),
    ("orig_upb", _float),
    ("orig_term", _int),
    ("orig_date", _date_mmyyyy),
    ("first_pay_date", _date_mmyyyy),
    ("oltv", _float),
    ("ocltv", _float),
    ("num_borrowers", _int),
    ("dti", _float),
    ("fico", _int),
    ("first_time_buyer", _str),
    ("purpose", _str),
    ("property_type", _str),
    ("num_units", _int),
    ("occupancy", _str),
    ("state", _str),
    ("zip3", _str),
    ("mi_pct", _float),
    ("product", _str),
    ("co_fico", _int),
    ("mi_type", _str),
    ("relocation_flag", _str),
)

PERFORMANCE_SCHEMA: tuple[tuple[str, Converter], ...] = (
    ("loan_id", _str),
    ("period", _date_mmddyyyy),
    ("servicer", _str),
    ("cur_rate", _float),
    ("cur_upb", _float),
    ("loan_age", _int),
    ("months_to_maturity", _int),
    ("adj_months_to_maturity", _int),
    ("maturity_date", _date_mmyyyy),
    ("msa", _str),
    ("dlq_status", _str),
    ("modification_flag", _str),
    ("zb_code", _str),
    ("zb_date", _date_mmyyyy),
    ("lpi_date", _date_mmddyyyy),
    ("foreclosure_date", _date_mmddyyyy),
    ("disposition_date", _date_mmddyyyy),
    ("foreclosure_costs", _float),
    ("preservation_costs", _float),
    ("asset_recovery_costs", _float),
    ("misc_holding_expenses", _float),
    ("holding_taxes", _float),
    ("net_sale_proceeds", _float),
    ("credit_enhancement_proceeds", _float),
    ("repurchase_make_whole_proceeds", _float),
    ("other_foreclosure_proceeds", _float),
    ("non_interest_bearing_upb", _float),
    ("principal_forgiveness", _float),
    ("rmw_proceeds_flag", _str),
    ("foreclosure_writeoff", _float),
    ("servicing_indicator", _str),
)

# Zero-balance codes that mean the loan terminated in a credit event.
DEFAULT_ZB_CODES = frozenset({"02", "03", "09", "15"})
PREPAY_ZB_CODES = frozenset({"01"})
