"""creditlab CLI: ingest / train / validate / explain / tearsheet / demo.

Artifacts live in a working directory (default ./creditlab_work):

    dataset.json.gz    ingested design matrix + accounting + group labels
    model.json         trained model artifact (logistic: self-contained)
    validation.json    crisis out-of-time report
    tearsheet.html     self-contained one-page report

`creditlab demo` runs the whole pipeline on the bundled synthetic fixtures
(800 loans, seed 99 — the exact configuration behind the README table) and
verifies the recomputed headline numbers against the README reference values,
exiting non-zero if they drift. That check is also a unit test.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from . import fixtures as fixtures_mod
from .baseline import LogisticModel, evaluate, train_logistic
from .challenger import train_gbm
from .explain import attach_groups, explain_loan, proxy_audit
from .features import FeatureStats, Matrix, build_dataset, to_matrix
from .tearsheet import render_tearsheet
from .validation import crisis_validation

DEFAULT_WORKDIR = "creditlab_work"

# The README headline numbers (synthetic fixtures, 800 loans, seed 99).
# `creditlab demo` recomputes them and fails loudly if they drift.
DEMO_REFERENCE = {
    "naive_auc": (0.8742, 0.01),
    "oot_auc": (0.8255, 0.01),
    "naive_underprediction": (0.80, 0.05),
    "oot_underprediction": (1.31, 0.05),
}


# --- artifact persistence ----------------------------------------------------

def _save_dataset(
    workdir: Path,
    matrix: Matrix,
    stats: FeatureStats,
    groups: list[str],
    horizon_months: int,
) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "dataset.json.gz"
    payload = {
        "feature_names": matrix.feature_names,
        "X": matrix.X,
        "y": matrix.y,
        "loan_ids": matrix.loan_ids,
        "vintages": matrix.vintages,
        "groups": groups,
        "stats": asdict(stats),
        "horizon_months": horizon_months,
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def _load_dataset(workdir: Path) -> tuple[Matrix, FeatureStats, list[str], int]:
    path = workdir / "dataset.json.gz"
    if not path.exists():
        raise SystemExit(f"no dataset at {path} — run `creditlab ingest` first")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    matrix = Matrix(
        X=d["X"], y=d["y"], feature_names=d["feature_names"],
        loan_ids=d["loan_ids"], vintages=d["vintages"],
    )
    return matrix, FeatureStats(**d["stats"]), d["groups"], d["horizon_months"]


def _print_eval(tag: str, r) -> None:
    print(f"{tag}: n={r.n} defaults={r.n_default} realized={r.realized_rate:.4f} "
          f"predicted={r.mean_predicted:.4f} AUC={r.auc:.4f} "
          f"brier={r.brier:.5f} skill={r.brier_skill:.3f}")


# --- subcommands -------------------------------------------------------------

def cmd_fixtures(args: argparse.Namespace) -> int:
    acq, perf = fixtures_mod.generate(
        fixtures_mod.FixtureConfig(n_loans=args.loans, seed=args.seed), args.out
    )
    print(f"wrote {acq}\nwrote {perf}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    rows, stats = build_dataset(args.acq, args.perf, horizon_months=args.horizon)
    matrix = to_matrix(rows)
    groups = attach_groups(args.acq, matrix)
    path = _save_dataset(Path(args.workdir), matrix, stats, groups, args.horizon)
    print(f"ingested {stats.n_rows} model-ready loans "
          f"({stats.n_censored_excluded} censored, "
          f"{stats.n_missing_required} missing-required excluded) -> {path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    matrix, _, _, _ = _load_dataset(Path(args.workdir))
    if args.model == "logit":
        model = train_logistic(matrix)
        artifact = model.to_dict()
    else:
        model = train_gbm(matrix)
        try:
            artifact = model.to_dict()
        except NotImplementedError:
            print(f"note: {model.backend} backend artifact is not serializable; "
                  "model.json not written (train-in-session only)")
            artifact = None
    _print_eval(f"{args.model} in-sample", evaluate(model, matrix))
    if artifact is not None:
        path = Path(args.workdir) / "model.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        print(f"model artifact -> {path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    matrix, _, _, _ = _load_dataset(Path(args.workdir))
    report = crisis_validation(matrix, train_max_vintage=args.train_max_vintage)
    print(report.headline_table())
    print(f"\nscore PSI (train -> OOT): {report.score_psi:.4f}")
    drift = ", ".join(f"{n}={v:.2f}" for n, v in report.top_drifted_features(3))
    print(f"top drifted features: {drift}")
    path = Path(args.workdir) / "validation.json"
    path.write_text(json.dumps(report.to_dict()), encoding="utf-8")
    print(f"validation report -> {path}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    matrix, _, _, _ = _load_dataset(workdir)
    model_path = workdir / "model.json"
    if not model_path.exists():
        raise SystemExit(f"no model at {model_path} — run `creditlab train` first")
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    if artifact.get("kind") != "logistic":
        raise SystemExit(
            "reason codes require the logistic model (exact decomposition); "
            "re-run `creditlab train --model logit`"
        )
    model = LogisticModel.from_dict(artifact)
    try:
        exp = explain_loan(model, matrix, args.loan_id, top_n=args.top)
    except KeyError as e:
        raise SystemExit(str(e)) from None
    print(f"loan {exp.loan_id}: PD={exp.predicted_pd:.4f} "
          f"(log-odds {exp.log_odds:+.3f}, intercept {exp.intercept:+.3f})")
    if not exp.reasons:
        print("no risk-increasing factors — nothing to cite")
    for i, r in enumerate(exp.reasons, 1):
        print(f"  {i}. {r.phrase} [{r.feature}={r.value:g}, +{r.contribution:.3f} log-odds]")
    return 0


def cmd_tearsheet(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    matrix, stats, groups, horizon = _load_dataset(workdir)
    crisis = crisis_validation(matrix, train_max_vintage=args.train_max_vintage)
    production_model = train_logistic(matrix)
    fairness = proxy_audit(production_model, matrix, groups)
    html = render_tearsheet(
        title=args.title,
        data_note=args.data_note,
        stats=stats,
        horizon_months=horizon,
        model_eval=crisis.naive,  # held-out random-split performance
        crisis=crisis,
        fairness=fairness,
    )
    out = Path(args.out) if args.out else workdir / "tearsheet.html"
    out.write_text(html, encoding="utf-8")
    print(f"tearsheet -> {out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """End-to-end on synthetic fixtures; verifies the README numbers."""
    workdir = Path(args.out)
    fix_dir = workdir / "fixtures"
    acq, perf = fixtures_mod.generate(
        fixtures_mod.FixtureConfig(n_loans=800, seed=99), fix_dir
    )
    print(f"synthetic fixtures (800 loans, seed 99) -> {fix_dir}")

    rows, stats = build_dataset(acq, perf, horizon_months=36)
    matrix = to_matrix(rows)
    groups = attach_groups(acq, matrix)
    _save_dataset(workdir, matrix, stats, groups, 36)
    print(f"ingested {stats.n_rows} model-ready loans")

    model = train_logistic(matrix)
    (workdir / "model.json").write_text(json.dumps(model.to_dict()), encoding="utf-8")

    report = crisis_validation(matrix)
    print()
    print(report.headline_table())
    print()

    measured = {
        "naive_auc": report.naive.auc,
        "oot_auc": report.oot.auc,
        "naive_underprediction": report.naive_underprediction,
        "oot_underprediction": report.oot_underprediction,
    }
    ok = True
    for key, (ref, tol) in DEMO_REFERENCE.items():
        got = measured[key]
        status = "ok" if abs(got - ref) <= tol else "DRIFTED"
        if status == "DRIFTED":
            ok = False
        print(f"  {key}: {got:.4f} vs README {ref} (±{tol}) {status}")

    fairness = proxy_audit(model, matrix, groups)
    html = render_tearsheet(
        title="creditlab demo tearsheet",
        data_note="SYNTHETIC FIXTURE DATA (800 loans, seed 99) — not real "
        "mortgages. Download Fannie Mae data and run ingest/validate/tearsheet "
        "for the real-data version.",
        stats=stats, horizon_months=36,
        model_eval=report.naive, crisis=report, fairness=fairness,
    )
    (workdir / "tearsheet.html").write_text(html, encoding="utf-8")
    print(f"\ntearsheet -> {workdir / 'tearsheet.html'}")

    if ok:
        print("demo reproduces the README reference numbers ✓")
        return 0
    print("demo numbers drifted from the README reference — investigate before "
          "trusting either", file=sys.stderr)
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="creditlab",
        description="Loan-level mortgage PD modeling, stress-tested through 2008.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fixtures", help="generate synthetic fixture files")
    p.add_argument("--out", default="data/fixtures")
    p.add_argument("--loans", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_fixtures)

    p = sub.add_parser("ingest", help="parse + label + featurize a file pair")
    p.add_argument("acq", help="acquisition file (.txt or .txt.gz)")
    p.add_argument("perf", help="performance file (.txt or .txt.gz)")
    p.add_argument("--horizon", type=int, default=36)
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("train", help="fit a model on the ingested dataset")
    p.add_argument("--model", choices=("logit", "gbm"), default="logit")
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("validate", help="crisis out-of-time validation")
    p.add_argument("--train-max-vintage", type=int, default=2006)
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("explain", help="adverse-action reason codes for a loan")
    p.add_argument("loan_id")
    p.add_argument("--top", type=int, default=4)
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("tearsheet", help="render the self-contained HTML report")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default="creditlab tearsheet")
    p.add_argument("--data-note", default="Data source and caveats not specified "
                   "— pass --data-note to describe the input data.")
    p.add_argument("--train-max-vintage", type=int, default=2006)
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.set_defaults(func=cmd_tearsheet)

    p = sub.add_parser("demo", help="end-to-end on synthetic fixtures; verifies "
                       "the README numbers")
    p.add_argument("--out", default="creditlab_demo")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
