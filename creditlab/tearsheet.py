"""Self-contained HTML tearsheet — the one-page model-risk report.

Everything inline (CSS included, no external assets, no scripts), so the file
can be attached to an email or opened offline years later and still render.
Sections: dataset accounting, model performance (discrimination AND
calibration), the crisis out-of-time headline, drift attribution, and the
proxy-fairness audit. Every number comes from the same dataclasses the tests
pin — the tearsheet renders results, it never recomputes them.
"""
from __future__ import annotations

from html import escape
from typing import Optional

from .baseline import EvalReport
from .explain import ProxyReport
from .features import FeatureStats
from .validation import CrisisReport

_CSS = """
body { font: 14px/1.5 -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       color: #1a1f26; max-width: 880px; margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.5rem; border-bottom: 2px solid #1a1f26; padding-bottom: .3rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; }
table { border-collapse: collapse; margin: .75rem 0; width: 100%; }
th, td { border: 1px solid #c7ccd4; padding: .35rem .6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #eef1f5; }
.note { background: #fff8e6; border: 1px solid #e6d9a8; padding: .6rem .8rem;
        border-radius: 6px; margin: 1rem 0; }
.headline { background: #f0f4ff; border: 1px solid #b8c6e8; padding: .6rem .8rem;
            border-radius: 6px; }
.small { color: #5a6472; font-size: .85rem; }
"""


def _fmt(v: Optional[float], spec: str = ".4f") -> str:
    return format(v, spec) if v is not None else "n/a"


def _eval_table(r: EvalReport) -> str:
    rows = [
        ("loans", str(r.n)),
        ("defaults", str(r.n_default)),
        ("realized default rate", _fmt(r.realized_rate)),
        ("mean predicted PD", _fmt(r.mean_predicted)),
        ("AUC", _fmt(r.auc)),
        ("Brier score", _fmt(r.brier, ".5f")),
        ("Brier skill vs climatology", _fmt(r.brier_skill, ".3f")),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"<table><tbody>{body}</tbody></table>"


def _decile_table(r: EvalReport) -> str:
    head = ("<tr><th>decile</th><th>n</th><th>mean predicted</th>"
            "<th>realized</th><th>lift</th></tr>")
    body = "".join(
        f"<tr><td>{d.decile}</td><td>{d.n}</td><td>{_fmt(d.mean_predicted)}</td>"
        f"<td>{_fmt(d.realized)}</td><td>{_fmt(d.lift, '.2f')}</td></tr>"
        for d in r.deciles
    )
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


def _crisis_section(c: CrisisReport) -> str:
    rows = [
        ("loans evaluated", str(c.naive.n), str(c.oot.n)),
        ("realized default rate", _fmt(c.naive.realized_rate), _fmt(c.oot.realized_rate)),
        ("mean predicted PD", _fmt(c.naive.mean_predicted), _fmt(c.oot.mean_predicted)),
        ("realized / predicted", _fmt(c.naive_underprediction, ".2f"),
         _fmt(c.oot_underprediction, ".2f")),
        ("AUC", _fmt(c.naive.auc), _fmt(c.oot.auc)),
        ("Brier score", _fmt(c.naive.brier, ".5f"), _fmt(c.oot.brier, ".5f")),
    ]
    vmin, vmax = min(c.test_vintages), max(c.test_vintages)
    head = (f"<tr><th>metric</th><th>random split (naive)</th>"
            f"<th>OOT {vmin}&ndash;{vmax} vintages (honest)</th></tr>")
    body = "".join(
        f"<tr><td>{m}</td><td>{a}</td><td>{b}</td></tr>" for m, a, b in rows
    )
    drift = ", ".join(
        f"{escape(name)} {v:.2f}" for name, v in c.top_drifted_features(3)
    )
    return (
        f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"
        f"<p class='headline'><strong>Reading:</strong> discrimination survives "
        f"while calibration breaks &mdash; on unseen vintages realized defaults run "
        f"<strong>{_fmt(c.oot_underprediction, '.2f')}&times;</strong> the model's "
        f"prediction, a warning the naive split never raises "
        f"({_fmt(c.naive_underprediction, '.2f')}&times;). "
        f"Score PSI {c.score_psi:.3f}; top drifted features: {drift}.</p>"
    )


def _fairness_section(p: ProxyReport, psi_flag: float = 0.25) -> str:
    head = ("<tr><th>group</th><th>n</th><th>mean predicted</th>"
            "<th>realized</th><th>gap</th></tr>")
    body = "".join(
        f"<tr><td>{escape(g.group)}</td><td>{g.n}</td>"
        f"<td>{_fmt(g.mean_predicted)}</td><td>{_fmt(g.realized)}</td>"
        f"<td>{g.gap:+.4f}</td></tr>"
        for g in sorted(p.groups, key=lambda g: -g.gap)
    )
    flagged = [
        f"{escape(n)} (PSI {v:.2f})"
        for n, v in p.top_proxy_features()
        if v >= psi_flag
    ]
    proxies = ", ".join(flagged) if flagged else f"none (all feature PSIs &lt; {psi_flag})"
    return (
        f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"
        f"<p>Potential proxy features for {escape(p.group_field)}: {proxies}.</p>"
        f"<p class='small'>Positive gap = the model over-prices the group's risk "
        f"relative to realized outcomes. {escape(p.group_field)} is not a model "
        f"feature; this audit checks the model cannot reconstruct it.</p>"
    )


def _stats_table(s: FeatureStats, horizon_months: int) -> str:
    rows = [
        ("acquisition rows", s.n_acquisitions),
        ("labeled loans", s.n_labels),
        ("joined", s.n_joined),
        ("model-ready rows", s.n_rows),
        (f"censored before {horizon_months}m horizon (excluded)", s.n_censored_excluded),
        ("missing required features (excluded)", s.n_missing_required),
        ("acquisitions without performance history", s.n_acq_without_label),
        ("performance histories without acquisition row", s.n_label_without_acq),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"<table><tbody>{body}</tbody></table>"


def render_tearsheet(
    *,
    title: str,
    data_note: str,
    stats: FeatureStats,
    horizon_months: int,
    model_eval: EvalReport,
    crisis: CrisisReport,
    fairness: ProxyReport,
) -> str:
    """Assemble the full self-contained HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{escape(title)}</h1>
<p class="note">{escape(data_note)}</p>

<h2>Dataset accounting</h2>
{_stats_table(stats, horizon_months)}
<p class="small">Every loan is accounted for: model-ready rows + exclusions equal
the joined population. Censored loans are excluded, never counted as
non-defaults.</p>

<h2>Logistic baseline &mdash; performance</h2>
{_eval_table(model_eval)}
<h2>Expected vs realized by risk decile</h2>
{_decile_table(model_eval)}

<h2>Crisis out-of-time validation (train &le; {crisis.train_max_vintage})</h2>
{_crisis_section(crisis)}

<h2>Proxy-fairness audit</h2>
{_fairness_section(fairness)}

<p class="small">Generated by creditlab &mdash; loan-level mortgage PD modeling,
stress-tested through the 2008 crisis. All numbers computed by the audited,
unit-tested metrics in <code>creditlab.metrics</code> /
<code>creditlab.validation</code>; this page renders them, it does not
recompute them.</p>
</body>
</html>
"""
