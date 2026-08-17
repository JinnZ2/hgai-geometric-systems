#!/usr/bin/env python3
"""
Claim verifier — run the repository's published claims against the live code.

Every number this project publishes in README.md and docs/ is a claim about
what the code does. Claims drift: the code moves, the prose does not. This
script re-derives each claim from a live run and reports CONFIRMED,
FALSIFIED, or UNREACHABLE.

    CONFIRMED    the run reproduces the published number
    FALSIFIED    the run contradicts it
    UNREACHABLE  the claim describes a state the code cannot enter
    OPEN         a known defect, already written up in the experiment log
    SKIPPED      an optional dependency is missing

Usage:
    python verify_claims.py           # run all checks
    python verify_claims.py -v        # show expected vs observed for each
    python verify_claims.py --json    # machine-readable

Exit code is 0 when nothing is FALSIFIED or UNREACHABLE, 1 otherwise, so
this can gate a commit. OPEN findings do not fail the gate: they are
already recorded, and the gate exists to catch *new* drift between the
prose and the code. When an OPEN finding is fixed, its check flips to
CONFIRMED on its own.

Run this before editing any published number. The history of what it caught
is in docs/EXPERIMENT-LOG.md.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

CONFIRMED = "CONFIRMED"
FALSIFIED = "FALSIFIED"
UNREACHABLE = "UNREACHABLE"
OPEN = "OPEN"
SKIPPED = "SKIPPED"

#: Statuses that mean the prose and the code have drifted apart, and that
#: therefore fail the gate.
GATE_FAILING = (FALSIFIED, UNREACHABLE)


@dataclass
class Result:
    """Outcome of checking one published claim.

    Parameters
    ----------
    claim : str
        The claim as published, in the words of the document that makes it.
    source : str
        Where the claim is published.
    status : str
        CONFIRMED, FALSIFIED, UNREACHABLE, OPEN, or SKIPPED.
    log_entry : str
        Experiment log entry this check corresponds to, if any.
    expected : Any
        What the document says.
    observed : Any
        What the run produced.
    note : str
        Context for a non-confirmed result.
    """

    claim: str
    source: str
    status: str
    expected: Any = None
    observed: Any = None
    note: str = ""
    log_entry: str = ""


CHECKS: List[Callable[[], Result]] = []


def check(fn: Callable[[], Result]) -> Callable[[], Result]:
    """Register a claim check."""
    CHECKS.append(fn)
    return fn


# ---------------------------------------------------------------------------
# M(S) core
# ---------------------------------------------------------------------------

@check
def check_m_s_ceiling() -> Result:
    """The reachable ceiling of M(S) under the derived input domains."""
    from framework.core.m_s_calculator import MSCalculator, SystemMetrics

    best = MSCalculator.calculate(SystemMetrics(1.0, 1.0, 1.0, 1.0, 0.0))
    return Result(
        claim="M(S) = (R_e x A x D x C) - L with all inputs in [0,1] has ceiling 1.0",
        source="docs/02-M-S-equation.md",
        status=CONFIRMED if abs(best - 1.0) < 1e-9 else FALSIFIED,
        expected=1.0,
        observed=best,
        note="Product of four factors each <= 1, minus non-negative L.",
    )


@check
def check_interpretation_bands_reachable() -> Result:
    """Every interpretation band must be reachable within the derived range."""
    from framework.core.m_s_calculator import MSCalculator

    # Sample the reachable range densely and collect which labels appear.
    labels = set()
    n = 4001
    for i in range(n):
        score = -1.0 + 2.0 * i / (n - 1)  # [-1, +1]
        labels.add(MSCalculator.interpret(score))

    all_labels = {
        "Highly coherent and sustainable",
        "Strong coherence, good viability",
        "Moderate coherence, stable",
        "Weak coherence, stressed",
        "Low coherence, at risk",
        "Negative coherence, declining",
        "Severe negative coherence, collapse imminent",
    }
    missing = sorted(all_labels - labels)
    return Result(
        claim="Every M(S) interpretation band is reachable in [-1, +1]",
        source="framework/core/m_s_calculator.py",
        status=CONFIRMED if not missing else UNREACHABLE,
        expected="all 7 bands reachable",
        observed=f"{len(labels & all_labels)}/7 reachable",
        note="" if not missing else f"unreachable: {missing}",
    )


# ---------------------------------------------------------------------------
# HGAI engine
# ---------------------------------------------------------------------------

@check
def check_health_statuses_reachable() -> Result:
    """README advertises five health statuses. Can the engine emit them?"""
    from hgai import _assess_health_from_signals

    advertised = {"THRIVING", "HEALTHY", "STRESSED", "WARNING", "CRITICAL"}
    seen = set()

    # Sweep the geometry inputs the engine derives, rather than hunting for
    # text that happens to hit each band.
    for agency_mag in (0.0, 0.7, 1.42):
        for valence_ratio in (-1.0, 0.0, 1.0):
            for temporal_mag in (0.0, 0.2, 1.2):
                for presence_mag in (0.0, 0.09, 1.0):
                    for friction in (0, 3, 8):
                        geometry = {
                            "agency": {"magnitude": agency_mag, "ratio": 1.0},
                            "valence": {"magnitude": 1.4, "ratio": valence_ratio},
                            "temporal": {"magnitude": temporal_mag, "change_intensity": 0.0},
                            "presence": {"magnitude": presence_mag, "balance_signal": 0.0},
                        }
                        status = _assess_health_from_signals(geometry, friction, 0)[2]
                        seen.add(status)

    missing = sorted(advertised - seen)
    return Result(
        claim="Health status is one of THRIVING / HEALTHY / STRESSED / WARNING / CRITICAL",
        source="README.md",
        status=CONFIRMED if not missing else UNREACHABLE,
        expected=sorted(advertised),
        observed=sorted(seen),
        note="" if not missing else f"never emitted: {missing}",
    )


@check
def check_octant_scales_commensurable() -> Result:
    """The four octant magnitudes are mapped onto M(S) as if comparable."""
    from hgai import _encode_text_geometry

    texts = [
        "We openly published the raw data, methodology, and uncertainty ranges.",
        "The agency revised the methodology. Outliers were removed.",
        "Then after the event changed, because conditions shifted, therefore we adapted.",
        "There is balance present. Relationships are healthy, mutual and connected.",
        "The temperature will be 72 degrees.",
    ]
    peaks: Dict[str, float] = {}
    for text in texts:
        geometry = _encode_text_geometry(text)
        for octant in ("agency", "valence", "temporal", "presence"):
            magnitude = geometry[octant]["magnitude"]
            peaks[octant] = max(peaks.get(octant, 0.0), magnitude)

    spread = max(peaks.values()) / max(min(peaks.values()), 1e-9)
    ok = spread < 3.0
    return Result(
        claim="Agency, valence, temporal and presence magnitudes share a scale",
        source="hgai.py::_assess_health_from_signals",
        status=CONFIRMED if ok else OPEN,
        expected="peak magnitudes within ~3x of each other",
        observed={k: round(v, 4) for k, v in peaks.items()},
        note=(
            "Scales now commensurable; entry E4 can be closed."
            if ok
            else f"peak spread is {spread:.0f}x. A (temporal) and C (presence) "
            "are structurally starved relative to R_e (agency) and D "
            "(valence), so the multiplicative core is dominated by two of "
            "its four factors. Known and unfixed."
        ),
        log_entry="E4",
    )


# ---------------------------------------------------------------------------
# Resilience detectors
# ---------------------------------------------------------------------------

@check
def check_template_count() -> Result:
    """README states a template count and a category count."""
    from resilience.detectors import ALL_TEMPLATES, Category

    n_templates = len(ALL_TEMPLATES)
    n_categories = len({t.category for t in ALL_TEMPLATES})
    ok = n_templates == 24 and n_categories == 7
    return Result(
        claim="24 regex templates across 7 categories",
        source="README.md",
        status=CONFIRMED if ok else FALSIFIED,
        expected={"templates": 24, "categories": 7},
        observed={"templates": n_templates, "categories": n_categories},
        note="" if ok else "README template count is stale; update it.",
    )


# ---------------------------------------------------------------------------
# Auditor worked examples
# ---------------------------------------------------------------------------

def _audit_example(text: str, expected: Dict[str, Any], source: str) -> Result:
    from audit import Auditor

    result = Auditor().audit(text)
    observed = {
        "trust_score": result.trust_score,
        "friction": len(result.friction_alerts),
        "gaps": len(result.gaps),
    }
    ok = all(observed[k] == v for k, v in expected.items())
    return Result(
        claim=f"audit({text[:44]!r}...) -> {expected}",
        source=source,
        status=CONFIRMED if ok else FALSIFIED,
        expected=expected,
        observed=observed,
    )


@check
def check_audit_friction_example() -> Result:
    """The friction-heavy worked example on the README front page."""
    return _audit_example(
        "The agency revised the methodology. This was an isolated incident.",
        {"trust_score": 18, "friction": 2, "gaps": 5},
        "README.md",
    )


@check
def check_audit_forecast_example() -> Result:
    """The good-faith forecast worked example on the README front page."""
    return _audit_example(
        "Ensemble models show 70% chance of 2-4 inches. Confidence moderate.",
        {"trust_score": 48, "friction": 0, "gaps": 2},
        "README.md",
    )


# ---------------------------------------------------------------------------
# Defect field experiment
# ---------------------------------------------------------------------------

@check
def check_defect_field_improvement() -> Result:
    """README claims 24-28% convergence improvement, 100% defect survival."""
    from defect_field import sweep_defect_configs

    sweep = sweep_defect_configs(N=40, steps=200, seed=42)
    improvements = {
        name: round((1.0 - res.convergence_ratio) * 100)
        for name, res in sweep.items()
        if not name.startswith("no_defects")
    }
    survival = {
        name: round(res.defect_survival_rate * 100)
        for name, res in sweep.items()
        if not name.startswith("no_defects")
    }
    in_band = all(24 <= v <= 28 for v in improvements.values())
    all_survive = all(v == 100 for v in survival.values())
    return Result(
        claim="Defect configurations improve convergence 24-28%, 100% survival",
        source="README.md",
        status=CONFIRMED if (in_band and all_survive) else FALSIFIED,
        expected="all configurations in 24-28%, survival 100%",
        observed={"improvement_pct": improvements, "survival_pct": survival},
    )


# ---------------------------------------------------------------------------
# Defect weather model
# ---------------------------------------------------------------------------

@check
def check_defect_weather_margin() -> Result:
    """The revised claim: defect-aware wins are directional but immaterial."""
    from defect_weather_model import MATERIAL_THRESHOLD_PCT, run_scenario_comparison

    results = run_scenario_comparison()
    margins = {name: round(r.improvement, 4) for name, r in results.items()}
    directional = all(m > 0 for m in margins.values())
    immaterial = all(abs(m) < MATERIAL_THRESHOLD_PCT for m in margins.values())

    # The revised claim, from experiment log entry E3: the defect-aware model
    # is better in every scenario by sign, and by an amount too small for this
    # synthetic setup to resolve. If a margin ever clears the threshold, this
    # check fails and the claim gets upgraded -- that is the point.
    ok = directional and immaterial
    return Result(
        claim="Defect-aware model wins directionally in 3/3 scenarios, all margins immaterial",
        source="README.md (revised; original claimed a substantive 3/3 win)",
        status=CONFIRMED if ok else FALSIFIED,
        expected=f"all margins > 0 and < {MATERIAL_THRESHOLD_PCT:.0f}%",
        observed=margins,
        note=(
            "Sign is consistent, magnitude is not evidence. Needs real "
            "observational data to decide. Original '3/3 scenarios, "
            "consistently improves forecasts' was falsified."
            if ok
            else "A margin moved out of the tie band. Re-read the run and "
            "update the published claim."
        ),
        log_entry="E3",
    )


# ---------------------------------------------------------------------------
# Phi field theory
# ---------------------------------------------------------------------------

@check
def check_phi_vacuum_suppression() -> Result:
    """README claims 42 modes, 9 surviving, 65% suppression, small Lambda."""
    from phi_field_theory import PhiFieldTheory

    res = PhiFieldTheory().compute()
    suppression_pct = round((1.0 - res.suppression_ratio) * 100)
    observed = {
        "total": res.n_total,
        "surviving": res.n_surviving,
        "suppression_pct": suppression_pct,
        "lambda": float(f"{res.cosmological_constant:.4g}"),
    }
    ok = (
        res.n_total == 42
        and res.n_surviving == 9
        and suppression_pct == 65
        and 1.0e-4 < res.cosmological_constant < 2.0e-4
    )
    return Result(
        claim="42 total modes, 9 survive (21%), vacuum energy suppressed 65%, Lambda ~1.5e-04",
        source="README.md",
        status=CONFIRMED if ok else FALSIFIED,
        expected={"total": 42, "surviving": 9, "suppression_pct": 65, "lambda": 1.5e-4},
        observed=observed,
    )


# ---------------------------------------------------------------------------
# Dependency surface
# ---------------------------------------------------------------------------

@check
def check_numpy_only_core() -> Result:
    """README claims the core runs on numpy alone."""
    import importlib

    core_modules = ["hgai", "audit", "resilience.detectors", "resilience.notices"]
    failures = []
    for name in core_modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:  # pragma: no cover - environment dependent
            failures.append(f"{name}: {exc}")
    return Result(
        claim="Core tools require numpy only",
        source="README.md",
        status=CONFIRMED if not failures else FALSIFIED,
        expected="hgai, audit, resilience import with numpy alone",
        observed="all import" if not failures else failures,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> List[Result]:
    """Execute every registered check, isolating failures."""
    results = []
    for fn in CHECKS:
        try:
            results.append(fn())
        except Exception as exc:  # a check that crashes is itself a finding
            results.append(
                Result(
                    claim=fn.__doc__.strip().splitlines()[0] if fn.__doc__ else fn.__name__,
                    source=fn.__name__,
                    status=FALSIFIED,
                    note=f"check raised {type(exc).__name__}: {exc}",
                )
            )
    return results


def render(results: List[Result], verbose: bool = False) -> str:
    """Format results as a readable report."""
    width = 72
    lines = ["=" * width, "  CLAIM VERIFICATION", "=" * width, ""]

    for res in results:
        lines.append(f"  [{res.status:^11s}] {res.claim}")
        entry = f"  (log {res.log_entry})" if res.log_entry else ""
        lines.append(f"                 source: {res.source}{entry}")
        if verbose or res.status not in (CONFIRMED, SKIPPED):
            if res.expected is not None:
                lines.append(f"                 expected: {res.expected}")
            if res.observed is not None:
                lines.append(f"                 observed: {res.observed}")
        if res.note:
            lines.append(f"                 note: {res.note}")
        lines.append("")

    tally = {s: sum(1 for r in results if r.status == s) for s in
             (CONFIRMED, FALSIFIED, UNREACHABLE, OPEN, SKIPPED)}
    lines.append("-" * width)
    lines.append(
        f"  {tally[CONFIRMED]} confirmed | {tally[FALSIFIED]} falsified | "
        f"{tally[UNREACHABLE]} unreachable | {tally[OPEN]} open | "
        f"{tally[SKIPPED]} skipped"
    )
    if tally[FALSIFIED] or tally[UNREACHABLE]:
        lines.append("")
        lines.append("  Published claims do not match the code.")
        lines.append("  Edit the claim, or fix the code. Log it in")
        lines.append("  docs/EXPERIMENT-LOG.md either way.")
    if tally[OPEN]:
        lines.append("")
        lines.append(f"  {tally[OPEN]} known defect(s) still open. These do not fail")
        lines.append("  the gate -- they are already written up. See the log.")
    lines.append("=" * width)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify published claims against live runs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show expected vs observed for every check")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    results = run_all()

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    else:
        print(render(results, verbose=args.verbose))

    bad = sum(1 for r in results if r.status in GATE_FAILING)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
