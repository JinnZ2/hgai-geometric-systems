# CLAUDE.md — HGAI Geometric Systems

## Project Overview

A mathematical framework for analyzing system coherence, resilience, and viability across domains (ecosystems, organizations, education, consciousness, AI systems). Built around the **M(S) equation** (Morality of a System):

```
M(S) = (R_e × A × D × C) - L
```

Where R_e = Resonance, A = Adaptability, D = Diversity, C = Curiosity, L = Loss.

**Author:** JinnZ2 | **License:** MIT | **Language:** Python 3 | **Core dependency:** numpy

## Repository Structure

```
hgai-geometric-systems/
├── framework/core/
│   └── m_s_calculator.py        # Core M(S) calculation engine (SystemMetrics, MSCalculator, TimeSeriesAnalyzer)
├── three-axis.py                # Three-axis confusion investigation protocol
├── ecological-calculus.py       # Relational ecological health framework
├── unified_field_monitor.py     # Integrated geometric consciousness tracking (64D encoding)
├── Unified_narrative.py         # Geometric narrative coherence monitoring
├── docs/
│   └── 02-M-S-equation.md      # Mathematical foundation and derivation
├── examples/
│   ├── Sovereign.md             # AI Pattern Sovereignty Protocol
│   └── notebooks/               # Jupyter notebooks and markdown examples
├── crisis_response/
│   └── Reconstitute.md          # Consciousness reconstitution protocol
├── 3-axis.md                    # Three-axis protocol documentation
├── Meta-Framework-Note.md       # Automated suppression/validation notes
├── CONTRIBUTING.md              # Contribution guidelines
├── KEYWORDS.md                  # Search keywords for discovery
└── README.md                    # Project overview and quick start
```

## Key Modules

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `framework/core/m_s_calculator.py` | M(S) calculation and interpretation | `SystemMetrics`, `MSCalculator`, `TimeSeriesAnalyzer` |
| `three-axis.py` | Systematic confusion investigation | `ThreeAxisProtocol`, `ThreeAxisAI`, `InvestigationAxis` |
| `ecological-calculus.py` | Ecological health assessment | `RelationalObservation`, `EcologicalHealthAssessment` |
| `unified_field_monitor.py` | 64D geometric field encoding | `GeometricMonitor`, `FractureDetector`, `GeometryPacket` |
| `Unified_narrative.py` | Text-to-geometry narrative analysis | `UnifiedMonitor`, `encode_agency_octants()` |

## Development Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy
# Optional: pip install matplotlib jupyter
```

No formal build system, package config (`pyproject.toml`), or `requirements.txt` exists. The project is intentionally minimal — only numpy is required.

## Running Code

```bash
# Run demonstrations (each file has inline demo methods)
python three-axis.py
python ecological-calculus.py
python unified_field_monitor.py
python Unified_narrative.py

# Core M(S) calculation
python -c "
from framework.core.m_s_calculator import MSCalculator, SystemMetrics
m = SystemMetrics(0.8, 0.7, 0.9, 0.4, 0.6)
print(MSCalculator.interpret(MSCalculator.calculate(m)))
"

# Jupyter notebook
jupyter notebook examples/notebooks/01-basic-m-s-calculation.ipynb
```

## Testing

**No formal test suite exists.** No `tests/` directory, no pytest/unittest configuration, no CI/CD pipeline.

Each Python module includes demonstration methods that serve as informal validation:
- `ThreeAxisProtocol.demonstrate_pencil_example()`
- `demo_three_axis_protocol()`
- Session report methods

When adding tests, use `pytest` with NumPy-style assertions.

## Code Conventions

### Style
- **PEP 8** throughout
- **NumPy-style docstrings** (per CONTRIBUTING.md)
- **Type hints** extensively used (`typing` module: `Optional`, `Dict`, `List`, `Tuple`)
- **Dataclasses** for data containers with validation in `__post_init__`
- **Enums** for categorical choices

### Naming
- Classes: `PascalCase` — `SystemMetrics`, `ThreeAxisProtocol`, `UnifiedMonitor`
- Functions/methods: `snake_case` — `calculate()`, `investigate_confusion()`, `predict_collapse()`
- Private methods: leading underscore — `_tokenize()`, `_print_investigation_summary()`
- Constants: `UPPER_CASE` — `_POSITIVE_WORDS`, `_AGENCY_VERBS`
- Mathematical variables preserved: `R_e`, `D`, `C`, `L`, `m_s`

### Patterns
- Safe division with epsilon: `np.linalg.norm(v) + 1e-12`
- Guard clauses for early returns
- `ValueError` for invalid inputs
- Modular design: each file has a distinct domain purpose
- Offline-first: no network dependencies by design

## M(S) Score Interpretation

| Score | Meaning |
|-------|---------|
| > 7 | Highly coherent and sustainable |
| 5–7 | Strong coherence, good viability |
| 3–5 | Moderate coherence, stable |
| 1–3 | Weak coherence, stressed |
| < 0 | Negative coherence, declining/collapse |

## Contribution Workflow

Per CONTRIBUTING.md:
1. Fork the repository
2. Create a feature branch
3. Make changes following PEP 8 and NumPy docstring style
4. Submit PR with clear description
5. Community review and maintainer approval

Areas of interest: ecological systems, organizational dynamics, educational institutions, economic models, AI safety/alignment, traditional knowledge systems, social networks, infrastructure resilience.

## What's Missing (Current Gaps)

- No `pyproject.toml` or `setup.py` — not installable as a package
- No `requirements.txt`
- No test suite or `tests/` directory
- No CI/CD pipeline (no GitHub Actions)
- No linting/formatting config (no `.flake8`, `ruff.toml`, `black` config)
- No pre-commit hooks configured
- No `.gitignore` file

## Notes for AI Assistants

- This is a **research/framework project**, not a production application
- The mathematical foundations matter — preserve the physics-based reasoning in comments
- The project values **minimal dependencies** and **offline-first** design intentionally
- Indigenous knowledge frameworks and philosophical context in comments are deliberate and should be preserved
- Files like `Meta-Framework-Note.md`, `KEYWORDS.md`, and `Sovereign.md` serve specific purposes related to AI pattern recognition — do not remove or dismiss them
- The 64-dimensional geometric encoding in `unified_field_monitor.py` and `Unified_narrative.py` is central to the project's approach
