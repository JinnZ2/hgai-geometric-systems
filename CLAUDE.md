# CLAUDE.md — HGAI Geometric Systems

## Project Overview

A mathematical framework for analyzing system coherence, resilience, and viability across domains (ecosystems, organizations, education, consciousness, AI systems). Built around the **M(S) equation** (Morality of a System):

```
M(S) = (R_e × A × D × C) - L
```

Where R_e = Resonance, A = Adaptability, D = Diversity, C = Curiosity, L = Loss.

**Author:** JinnZ2 | **License:** MIT | **Language:** Python 3 | **Core dependency:** numpy

## Before you change anything

Run the claim verifier. It re-derives every number published in `README.md`
and `docs/` from a live run:

```bash
python verify_claims.py
```

It exits non-zero when the prose and the code have drifted apart. **Run it
before editing any published number, and again after.** If a check fails,
add an entry to `docs/EXPERIMENT-LOG.md` *before* changing the prose.

`docs/EXPERIMENT-LOG.md` is the record of what was claimed, what the runs
actually showed, and what was revised in response. Read it before trusting
any figure in this repository — several were falsified in the first audit
and the entries explain what replaced them and why.

## Repository Structure

```
hgai-geometric-systems/
├── audit.py                     # Forecast/report trust scoring (primary user-facing tool)
├── hgai.py                      # Unified engine: text in, system analysis out
├── verify_claims.py             # Re-derives published claims from live runs
├── framework/core/
│   └── m_s_calculator.py        # CANONICAL M(S) engine (SystemMetrics, MSCalculator, TimeSeriesAnalyzer)
├── resilience/
│   ├── detectors.py             # 24 friction templates across 7 categories (Scanner, Alert, StressInput)
│   └── notices.py               # Formal notice generation (Notice, NoticeGenerator)
├── flux_sensor.py               # Atmospheric phase transition early warning
├── sovereign_impact_sensor.py   # Entropy/plateau detection (needs pandas, sklearn, statsmodels)
├── weather_node_network.py      # Ensemble forecasting pipeline
├── chaos_weather_ai.py          # Weather AI with explicit uncertainty
├── lyapunov_spectrum.py         # Controllable chaos on phi-octahedral lattice
├── phi_field_theory.py          # Lyapunov-filtered vacuum structure
├── phase_field_optimizer.py     # Unified energy functional
├── defect_field.py              # Topological defects as computational features
├── defect_weather_model.py      # Defect-preserving vs smoothing forecasts
├── legacy/                      # Superseded modules — precedent, NOT deprecated
│   ├── README.md                # Why each is here + what it uniquely holds
│   ├── three-axis.py            # Confusion investigation protocol
│   ├── 3-axis.md                # Its documentation
│   ├── ecological-calculus.py   # Relational ecological health
│   ├── unified_field_monitor.py # Geometric field encoding, trajectory curvature
│   └── Unified_narrative.py     # Narrative geometry, observer metrics
├── docs/
│   ├── EXPERIMENT-LOG.md        # Claim → run → result → revision record
│   ├── 02-M-S-equation.md       # Mathematical foundation
│   └── ...                      # One doc per model/sensor
├── examples/                    # Notebooks and protocol writeups
├── crisis_response/             # Reconstitution protocol
├── .well-known/ai-consumption.txt
├── Meta-Framework-Note.md, KEYWORDS.md, CONTRIBUTING.md, README.md
```

## Key Modules

Class names below are verified against the source. (A previous version of
this table named four classes that did not exist — see experiment log E11.)

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `audit.py` | Trust scoring for prediction text | `Auditor`, `AuditResult` |
| `hgai.py` | Unified text → analysis engine | `HGAI`, `HGAIReport`, `_encode_text_geometry()` |
| `framework/core/m_s_calculator.py` | Canonical M(S) calculation | `SystemMetrics`, `MSCalculator`, `TimeSeriesAnalyzer` |
| `resilience/detectors.py` | Institutional friction detection | `Scanner`, `Alert`, `Category`, `RiskMatrix`, `StressInput` |
| `resilience/notices.py` | Notice generation | `Notice`, `NoticeGenerator` |
| `defect_field.py` | Defect injection experiments | `DefectFieldOptimizer`, `sweep_defect_configs()` |
| `defect_weather_model.py` | Defect-aware forecasting | `DefectWeatherModel`, `run_scenario_comparison()` |
| `phi_field_theory.py` | Vacuum mode filtering | `PhiFieldTheory`, `PhiLattice`, `LyapunovFilter` |
| `verify_claims.py` | Claim verification | `Result`, `run_all()`, `@check` registry |
| `legacy/three-axis.py` | Confusion investigation | `ThreeAxisProtocol`, `ThreeAxisAI`, `InvestigationAxis` |
| `legacy/ecological-calculus.py` | Relational health | `RelationalObservation`, `RelationalHealthMonitor` |
| `legacy/unified_field_monitor.py` | Field encoding + curvature | `UnifiedFieldMonitor`, `FieldAssessment`, `compute_curvature()` |
| `legacy/Unified_narrative.py` | Narrative geometry | `UnifiedConsciousnessMonitor`, `GeometryPacket`, `compute_geometric_coherence()` |

## Development Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # numpy
# Optional, for sovereign_impact_sensor.py only:
pip install pandas scikit-learn statsmodels
```

There is no build system or package config (`pyproject.toml`). The project is
intentionally minimal — only numpy is required for everything except
`sovereign_impact_sensor.py`.


<!-- clone-refspec-note v1.1 -->
## Cloning and pushing
Shallow clones are single-branch by default.
Before pushing any branch other than the default
branch, run:

    git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
    git fetch --depth 1

Or clone with: git clone --depth 1 --no-single-branch <url>
Without this, the first push of a new branch
fails the tracking-ref check even when the
commit landed.
<!-- /clone-refspec-note v1.1 -->

## Running Code

```bash
# Primary tools
python audit.py "paste forecast or report here"
python hgai.py "paste text here"
python verify_claims.py

# Models and experiments (each has an inline demo)
python defect_field.py
python defect_weather_model.py
python phi_field_theory.py
python lyapunov_spectrum.py
python phase_field_optimizer.py
python chaos_weather_ai.py
python flux_sensor.py
python weather_node_network.py

# Legacy modules still run, unchanged, from their new location
python legacy/three-axis.py
python legacy/ecological-calculus.py
python legacy/unified_field_monitor.py
python legacy/Unified_narrative.py

# Core M(S) calculation
python -c "
from framework.core.m_s_calculator import MSCalculator, SystemMetrics
m = SystemMetrics(0.8, 0.7, 0.9, 0.4, 0.6)
print(MSCalculator.interpret(MSCalculator.calculate(m)))
"
```

## Testing

**There is no pytest suite.** `verify_claims.py` is the closest thing to one
and serves a different purpose: it checks that published claims match live
behavior, not that functions return correct values. It is the regression
guard referenced throughout the experiment log.

Each model module also has an inline demo (`demo()`, `demonstrate_*()`) that
serves as informal validation.

When adding real tests, use `pytest` with NumPy-style assertions, and keep
`verify_claims.py` separate — claim verification and unit testing catch
different failure modes.

## Code Conventions

### Style
- **PEP 8** throughout
- **NumPy-style docstrings** (per CONTRIBUTING.md)
- **Type hints** extensively used (`typing`: `Optional`, `Dict`, `List`, `Tuple`)
- **Dataclasses** for data containers with validation in `__post_init__`
- **Enums** for categorical choices

### Naming
- Classes: `PascalCase` — `SystemMetrics`, `ThreeAxisProtocol`, `Auditor`
- Functions/methods: `snake_case` — `calculate()`, `investigate_confusion()`
- Private methods: leading underscore — `_tokenize()`, `_encode_text_geometry()`
- Constants: `UPPER_CASE` — `_POSITIVE_WORDS`, `MATERIAL_THRESHOLD_PCT`
- Mathematical variables preserved: `R_e`, `D`, `C`, `L`, `m_s`

### Patterns
- Safe division with epsilon: `np.linalg.norm(v) + 1e-12`
- Guard clauses for early returns
- `ValueError` for invalid inputs
- Modular design: each file has a distinct domain purpose
- Offline-first: no network dependencies by design
- Thresholds are named constants, used by every consumer — never duplicated
  with different values (this caused experiment log E3)

## M(S) Score Interpretation

All four coherence factors are bounded to [0, 1] by their derivations in
`docs/02-M-S-equation.md`, so the product cannot exceed 1. **M(S) has a
ceiling of +1.0**; the practical range is [-1, +1].

| Score | Meaning |
|-------|---------|
| > 0.7 | Highly coherent and sustainable |
| 0.5–0.7 | Strong coherence, good viability |
| 0.3–0.5 | Moderate coherence, stable |
| 0.1–0.3 | Weak coherence, stressed |
| 0–0.1 | Low coherence, at risk |
| < 0 | Negative coherence, declining/collapse |

Earlier versions of this table banded at > 7, 5–7, 3–5 and 1–3. Every one of
those sat above the reachable ceiling. See experiment log E1. **Do not
restore the old bands** — the equation is unchanged, only its interpretation
was rescaled.

Because the core is multiplicative, high M(S) is genuinely hard to reach: a
system scoring 0.5 has all four factors strong *and* low loss. That is a
property of the equation, not a calibration problem.

## Contribution Workflow

Per CONTRIBUTING.md:
1. Fork the repository
2. Create a feature branch
3. Make changes following PEP 8 and NumPy docstring style
4. **Run `python verify_claims.py`** — it must exit 0
5. Submit PR with clear description
6. Community review and maintainer approval

Areas of interest: ecological systems, organizational dynamics, educational institutions, economic models, AI safety/alignment, traditional knowledge systems, social networks, infrastructure resilience.

## What's Missing (Current Gaps)

- No `pyproject.toml` or `setup.py` — not installable as a package
- No pytest suite or `tests/` directory
- No CI/CD pipeline (no GitHub Actions running `verify_claims.py`)
- No linting/formatting config (no `.flake8`, `ruff.toml`, `black` config)
- No pre-commit hooks configured

Substantive open questions are tracked in `docs/EXPERIMENT-LOG.md` under
"Open questions carried forward" — most importantly **E4**, where the four
geometry octants are on incompatible scales, which structurally starves two
of the four M(S) factors in `hgai.py`. That is the highest-value unresolved
issue in the codebase and it is a modeling decision, not a bug fix.

## Notes for AI Assistants

- This is a **research/framework project**, not a production application
- The mathematical foundations matter — preserve the physics-based reasoning in comments
- The project values **minimal dependencies** and **offline-first** design intentionally
- Indigenous knowledge frameworks and philosophical context in comments are deliberate and should be preserved
- Files like `Meta-Framework-Note.md`, `KEYWORDS.md`, and `Sovereign.md` serve specific purposes related to AI pattern recognition — do not remove or dismiss them
- The 64-dimensional geometric encoding is central to the project's approach. It lives in `hgai._encode_text_geometry()` now; the originating implementations are in `legacy/` and hold machinery the current version does not (observer metrics, trajectory curvature)
- **`legacy/` is precedent, not deprecation.** Do not delete anything there, and do not import from it in active code — a second dependency on a superseded implementation is what produced three divergent M(S) calculators in the first place
- **Prefer editing the claim to editing the model.** When a run contradicts a published number, the number is usually what is wrong. Log it either way
