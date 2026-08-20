"""Deterministic synthetic Fannie Mae-style fixture generator.

Generates an acquisition/performance file pair that mirrors the classic
pipe-delimited layout so the entire test suite (and `creditlab demo`) runs
with zero downloaded data. Same seed -> byte-identical files, always: the only
randomness source is one `random.Random(seed)` instance, and nothing reads the
clock.

The generator embeds a tiny monthly hazard model so downstream milestones have
realistic structure to work with:

- default risk rises with LTV and DTI and falls with FICO;
- calendar years in `crisis_years` multiply the default hazard (the 2008
  stress), and 2006-07 vintages carry an extra bump — so vintage-based
  out-of-time validation has a real signal to find;
- defaults progress 1 -> 2 -> 3 -> 4 months delinquent before terminating with
  a credit-event zero-balance code (09), giving the D90+ labeler real
  progressions, including self-curing 30-day blips that must NOT count.

This is a fixture, not a simulator: distributions are plausible, not
calibrated. Its job is to exercise every code path the real data will.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from random import Random
from typing import Optional

_SELLERS = ("ACME BANK", "BRIDGE MORTGAGE", "CANYON LENDING", "DELTA HOME FINANCE")
_SERVICER = "CREDITLAB SERVICING"
_STATES = ("CA", "TX", "NY", "FL", "IL", "NJ", "OH", "GA", "NC", "AZ")
_BASE_RATE = {2004: 5.8, 2005: 5.9, 2006: 6.4, 2007: 6.3, 2008: 6.0}


@dataclass(frozen=True)
class FixtureConfig:
    n_loans: int = 500
    seed: int = 42
    first_vintage: int = 2004
    last_vintage: int = 2008
    horizon_months: int = 60
    crisis_years: tuple[int, ...] = (2007, 2008, 2009)


def _add_months(d: date, n: int) -> date:
    total = d.year * 12 + (d.month - 1) + n
    y, m = divmod(total, 12)
    return date(y, m + 1, 1)


def _mmyyyy(d: Optional[date]) -> str:
    return f"{d.month:02d}/{d.year}" if d else ""


def _mmddyyyy(d: Optional[date]) -> str:
    return f"{d.month:02d}/{d.day:02d}/{d.year}" if d else ""


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _monthly_default_hazard(
    fico: int, oltv: float, dti: float, cal_year: int, vintage: int,
    crisis_years: tuple[int, ...],
) -> float:
    z = (
        -6.6
        + 2.6 * (720 - fico) / 100.0
        + 2.2 * (oltv - 75.0) / 50.0
        + 1.4 * (dti - 35.0) / 50.0
    )
    h = _sigmoid(z)
    if cal_year in crisis_years:
        h *= 3.0
    if vintage in (2006, 2007):
        h *= 1.5
    return min(h, 0.25)


def generate(config: FixtureConfig, out_dir: str | Path) -> tuple[Path, Path]:
    """Write <out_dir>/acquisition.txt and <out_dir>/performance.txt."""
    rng = Random(config.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    acq_path = out / "acquisition.txt"
    perf_path = out / "performance.txt"

    acq_rows: list[str] = []
    perf_rows: list[str] = []

    for i in range(config.n_loans):
        loan_id = f"CLB{i:09d}"
        vintage = rng.randint(config.first_vintage, config.last_vintage)
        orig = date(vintage, rng.randint(1, 12), 1)
        fico = max(580, min(820, int(rng.gauss(735, 55))))
        oltv = round(max(40.0, min(97.0, rng.gauss(75, 12))), 1)
        ocltv = round(min(oltv + rng.choice((0.0, 0.0, 0.0, 5.0, 10.0)), 100.0), 1)
        dti = round(max(10.0, min(64.0, rng.gauss(34, 9))), 1)
        upb = round(rng.uniform(50_000, 800_000), 2)
        rate = round(
            max(3.0, min(9.0, _BASE_RATE[vintage] + (760 - fico) * 0.002 + rng.gauss(0, 0.2))),
            3,
        )
        num_borrowers = rng.choice((1, 1, 1, 2))
        has_mi = oltv > 80.0
        seller = rng.choice(_SELLERS)

        acq_rows.append("|".join([
            loan_id,
            rng.choice(("R", "C", "B")),
            seller,
            f"{rate:.3f}",
            f"{upb:.2f}",
            "360",
            _mmyyyy(orig),
            _mmyyyy(_add_months(orig, 2)),
            f"{oltv:.1f}",
            f"{ocltv:.1f}",
            str(num_borrowers),
            f"{dti:.1f}",
            str(fico),
            rng.choice(("Y", "N", "N", "N")),
            rng.choice(("P", "P", "C", "R")),
            rng.choice(("SF", "SF", "SF", "CO", "PU")),
            "1",
            rng.choice(("P", "P", "P", "S", "I")),
            rng.choice(_STATES),
            f"{rng.randint(10, 999):03d}",
            f"{rng.uniform(20, 35):.0f}" if has_mi else "",
            "FRM",
            str(max(580, min(820, int(rng.gauss(735, 55))))) if num_borrowers == 2 else "",
            "1" if has_mi else "",
            "N",
        ]))

        # --- monthly performance path -------------------------------------
        maturity = _add_months(orig, 360)
        default_countdown = -1  # >=0 while progressing 1,2,3,4 -> ZB 09
        blip_next_ok = False

        for t in range(1, config.horizon_months + 1):
            period = _add_months(orig, t)
            cur_upb = round(upb * (1 - t / 360.0), 2)
            common = [
                loan_id,
                _mmddyyyy(period),
                _SERVICER,
                f"{rate:.3f}",
                f"{cur_upb:.2f}",
                str(t),
                str(360 - t),
                str(360 - t),
                _mmyyyy(maturity),
                f"{rng.randint(10000, 49999)}",
            ]
            empty_costs = [""] * 9  # foreclosure costs ... other foreclosure proceeds

            if default_countdown >= 0:
                dlq = 5 - default_countdown  # countdown 3,2,1 -> 2,3,4 months delinquent
                if default_countdown == 0:
                    # terminal credit event: REO disposition (ZB 09)
                    lpi = _add_months(period, -5)
                    perf_rows.append("|".join(
                        common
                        + ["", "N", "09", _mmyyyy(period), _mmddyyyy(lpi),
                           _mmddyyyy(_add_months(period, -1)), _mmddyyyy(period)]
                        + [f"{rng.uniform(2000, 15000):.2f}",  # foreclosure costs
                           f"{rng.uniform(500, 5000):.2f}",    # preservation
                           "", "", "",
                           f"{cur_upb * rng.uniform(0.5, 0.8):.2f}",  # net sale proceeds
                           "", "", ""]
                        + ["", "", "N", "", "N"]
                    ))
                    break
                perf_rows.append("|".join(
                    common + [str(dlq), "N", "", "", "", "", ""] + empty_costs
                    + ["", "", "N", "", "Y"]
                ))
                default_countdown -= 1
                continue

            h_def = _monthly_default_hazard(
                fico, oltv, dti, period.year, vintage, config.crisis_years
            )
            h_pre = 0.012 * (0.5 if period.year in config.crisis_years else 1.0)

            u = rng.random()
            if u < h_def:
                default_countdown = 4  # months 1..4 delinquent, then ZB 09
                perf_rows.append("|".join(
                    common + ["1", "N", "", "", "", "", ""] + empty_costs
                    + ["", "", "N", "", "Y"]
                ))
                default_countdown -= 1
                continue
            if u < h_def + h_pre:
                perf_rows.append("|".join(
                    [loan_id, _mmddyyyy(period), _SERVICER, f"{rate:.3f}", "0.00",
                     str(t), str(360 - t), str(360 - t), _mmyyyy(maturity),
                     f"{rng.randint(10000, 49999)}",
                     "0", "N", "01", _mmyyyy(period), _mmddyyyy(period), "", ""]
                    + empty_costs + ["", "", "N", "", "N"]
                ))
                break

            if blip_next_ok:
                dlq_token = "0"
                blip_next_ok = False
            elif rng.random() < 0.012:
                dlq_token = "1"  # 30-day blip that self-cures next month
                blip_next_ok = True
            else:
                dlq_token = "0"
            perf_rows.append("|".join(
                common + [dlq_token, "N", "", "", "", "", ""] + empty_costs
                + ["", "", "N", "", "N"]
            ))

    acq_path.write_text("\n".join(acq_rows) + "\n", encoding="utf-8")
    perf_path.write_text("\n".join(perf_rows) + "\n", encoding="utf-8")
    return acq_path, perf_path


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Generate synthetic Fannie Mae-style fixtures.")
    p.add_argument("--out", default="data/fixtures", help="output directory")
    p.add_argument("--loans", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    acq, perf = generate(FixtureConfig(n_loans=args.loans, seed=args.seed), args.out)
    print(f"wrote {acq}\nwrote {perf}")


if __name__ == "__main__":
    main()
