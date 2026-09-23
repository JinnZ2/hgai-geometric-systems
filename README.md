# HGAI Geometric Systems

A mathematical framework for analyzing system coherence, resilience, and viability — with tools anyone can use right now.

**Paste any forecast, report, or prediction. Get back a trust score and what's being missed.**

```bash
python audit.py "The agency revised the methodology. This was an isolated incident."
# Trust: 18/100 (UNRELIABLE — institutional friction dominates) | 2 friction | 5 gaps
```

```bash
python audit.py "Ensemble models show 70% chance of 2-4 inches. Confidence moderate."
# Trust: 48/100 (LOW — significant gaps or friction) | 0 friction | 2 gaps
```

---

## Quick Start

```bash
git clone https://github.com/JinnZ2/hgai-geometric-systems.git
cd hgai-geometric-systems
pip install numpy

# Audit any text (phone-friendly)
python audit.py "paste forecast or report here"

# Quick one-liner
python audit.py -q "paste text here"

# JSON output (for AI systems)
python audit.py --json "paste text here"

# Full system analysis
python hgai.py "paste text here"

# Interactive mode (just run it and paste)
python audit.py
```

### As a Python Library

```python
# For humans
from audit import Auditor
auditor = Auditor()
result = auditor.audit("text from any source...")
print(result.render())       # readable report
print(result.trust_score)    # 0-100

# For AI systems
data = result.to_json()      # structured JSON

# Full analysis engine
from hgai import HGAI
engine = HGAI()
report = engine.analyze("text here...")
print(report.render())
```

---

## What It Does

### Forecast Auditor (`audit.py`)

Paste any prediction, forecast, or institutional report. Get back:

- **Trust Score (0-100)**: How much should you trust this?
- **Gaps**: What's probably missing (uncertainty? methodology? raw data?)
- **Friction Flags**: Language patterns that signal smoothing, hedging, or reclassification
- **Transparency Signals**: What the source does right
- **Overconfidence Signals**: Where the source overclaims
- **Numerical Audit**: Are the numbers appropriately precise or falsely exact?
- **Stress Input**: Numeric bridge into the entropy sensor for further analysis

### HGAI Engine (`hgai.py`)

Full system coherence analysis:

- **M(S) Score**: System coherence rating from the morality equation
- **Health Status**: THRIVING / HEALTHY / STRESSED / WARNING / CRITICAL
- **64D Narrative Geometry**: Agency, valence, temporal, and presence encoding
- **Curiosity Signals**: Cross-pattern investigation leads

---

## The M(S) Equation

The core mathematical framework:

```
M(S) = (R_e x A x D x C) - L
```

| Component | Meaning |
|-----------|---------|
| R_e | Resonance — coupling strength between components |
| A | Adaptability — response capacity to change |
| D | Diversity — pathway multiplicity |
| C | Curiosity — exploration rate |
| L | Loss — waste, suppression, inefficiency |

All four coherence factors are bounded to [0, 1] by their derivations, so
the product cannot exceed 1 and **M(S) has a ceiling of +1.0**. With L as
the dissipation ratio it is derived from, the practical range is [-1, +1].

| Score | Interpretation |
|-------|---------------|
| > 0.7 | Highly coherent and sustainable |
| 0.5-0.7 | Strong coherence, good viability |
| 0.3-0.5 | Moderate coherence, stable |
| 0.1-0.3 | Weak coherence, stressed |
| 0-0.1 | Low coherence, at risk |
| < 0 | Negative coherence, declining/collapse |

> Earlier versions of this table published bands at > 7, 5-7, 3-5 and 1-3.
> Those sit above the ceiling the equation can reach and were never
> attainable. The ordinal structure is unchanged; the scale was corrected.
> See [docs/EXPERIMENT-LOG.md](docs/EXPERIMENT-LOG.md) entry E1.

---

## Repository Structure

### Tools (Use These)

| File | What It Does | How To Use |
|------|-------------|-----------|
| `audit.py` | Trust scoring for any prediction text | `python audit.py "text"` or `--json` |
| `hgai.py` | Full system analysis engine | `python hgai.py "text"` or `from hgai import HGAI` |
| `verify_claims.py` | Re-derives every published number from a live run | `python verify_claims.py` |

### Sensors & Detectors

| File | What It Does |
|------|-------------|
| `resilience/detectors.py` | 24 regex templates detecting institutional friction across 7 categories |
| `resilience/notices.py` | Formal notice generation from alerts |
| `flux_sensor.py` | Atmospheric phase transition early warning |
| `sovereign_impact_sensor.py` | System stress scalar (I_e) and plateau detection |
| `chaos_weather_ai.py` | Weather AI that knows when it doesn't know |

### Models & Theory

| File | What It Does |
|------|-------------|
| `defect_field.py` | Topological defects as computational features (24-28% improvement proven) |
| `defect_weather_model.py` | Defect-preserving weather forecasts vs conventional smoothing |
| `phase_field_optimizer.py` | Unified energy functional: learning = geometry = energy minimization |
| `phi_field_theory.py` | Lyapunov-filtered vacuum structure (finite vacuum energy) |
| `lyapunov_spectrum.py` | Controllable chaos dynamics on phi-octahedral lattice |
| `weather_node_network.py` | Ensemble forecasting pipeline with uncertainty propagation |

### Core Framework

| File | What It Does |
|------|-------------|
| `framework/core/m_s_calculator.py` | M(S) equation engine — the canonical implementation |

### Legacy — precedent, not deprecation

`legacy/` holds the modules the current engine was built out of. Nothing in
the active import graph references them, which is a fact about wiring, not
about worth. They still run, and several hold capability with no successor
anywhere else (observer metrics, trajectory curvature, the full three-axis
investigation protocol, relational health accounting).

| File | What It Does | Superseded in the active path by |
|------|-------------|----------------------------------|
| `legacy/unified_field_monitor.py` | 64D geometric field encoding, trajectory curvature | `hgai._encode_text_geometry` |
| `legacy/Unified_narrative.py` | Text-to-geometry narrative analysis, observer metrics | `hgai._encode_text_geometry` |
| `legacy/ecological-calculus.py` | Relational ecosystem health assessment | `hgai._assess_health_from_signals` |
| `legacy/three-axis.py` | Curiosity-driven confusion investigation protocol | `hgai._extract_curiosity_signals` (a much smaller subset) |

See [legacy/README.md](legacy/README.md) for what each still uniquely holds
and the rules for promoting something back out.

### Documentation

| File | What It Covers |
|------|---------------|
| `docs/hgai-engine.md` | Unified engine architecture |
| `docs/resilience-detectors.md` | Friction scanner design and template library |
| `docs/chaos-weather-ai.md` | How the weather AI differs from conventional models |
| `docs/phi-field-theory.md` | Lyapunov-filtered vacuum energy derivation |
| `docs/lyapunov-spectrum.md` | Controllable chaos and three regimes |
| `docs/phase-field-optimizer.md` | Unified energy functional and four regimes |
| `docs/sovereign-impact-sensor.md` | Entropy-based plateau detection |
| `docs/model-reality-dissonance.md` | Stability bias in weather forecasting |
| `docs/weather-pipeline-model.md` | Probabilistic forecasting architecture |
| `docs/02-M-S-equation.md` | Mathematical foundation |
| `docs/EXPERIMENT-LOG.md` | What was claimed, what the runs showed, what got revised |
| `legacy/README.md` | Why each superseded module was kept and what it still uniquely holds |

---

## Key Results

### Institutional Friction Detection

24 regex templates across 7 categories detect the language of entropy denial:

| Category | What It Catches | Example |
|----------|----------------|---------|
| Reclassification | Retroactive event recoding | "revised the methodology" |
| Liability Hedging | Causal distancing | "not directly attributable" |
| Statistical Smoothing | Signal suppression | "outliers were removed" |
| Downplay | Impact minimization | "isolated incident" |
| Data Opacity | Access restriction | "proprietary methodology" |
| Dependency Risk | T_infra stress | "system outage" |
| Communication Friction | Reporting delays | "delayed notification" |

### Topological Defects Improve Computation

The defect field experiment (`defect_field.py`) proved that injecting discontinuities into a phase field **improves convergence by 24-28%** across all configurations:

| Configuration | vs Control | Defect Survival |
|--------------|-----------|-----------------|
| Single vortex | -24% error | 100% |
| Dipole | -24% error | 100% |
| Quadrupole | -28% error | 100% |
| Random cluster | -27% error | 100% |

Defects are non-erasable memory that participates in computation.

### Defect Weather Model — directionally consistent, magnitude UNTESTED

The defect-aware weather model is better than conventional smoothing in all
three scenarios **by sign**, and by a margin far too small for this synthetic
setup to resolve:

| Scenario | Smooth error | Defect-aware error | Margin | Verdict |
|----------|-------------|--------------------|--------|---------|
| frontal_passage | 40465.62 | 40465.29 | +0.0008% | tie |
| cyclone_formation | 40430.88 | 40430.07 | +0.0020% | tie |
| blizzard_transition | 40292.73 | 40292.49 | +0.0006% | tie |

Three out of three point the same way, which is worth noting. None of them
clears the 5% materiality threshold, which means **this claim is untested,
not confirmed.** Deciding it needs real observational data.

An earlier version of this section read "beats conventional smoothing in all
three test scenarios." That came from a summary that counted any margin
above zero as a win while each scenario's own verdict printed "models
roughly equal." See [docs/EXPERIMENT-LOG.md](docs/EXPERIMENT-LOG.md) entry E3.

Precursor detection does work as intended and is what makes the scenarios
comparable at all — the blizzard scenario has zero defects visible to
threshold-based detection, and precursor analysis catches 3 sub-threshold
signals (curvature anomaly, entropy buildup, P-T flux coupling).

### Lyapunov-Filtered Vacuum Energy

The phi-lattice field theory (`phi_field_theory.py`) produces finite vacuum energy through a mode survival constraint:

- 42 total modes, 9 survive (21%)
- Vacuum energy suppressed 65%
- Cosmological constant naturally small (1.5e-04)
- Mechanism: not cancellation, not fine-tuning — mode survival

---

## For AI Systems

### JSON API

```python
from audit import Auditor
import json

auditor = Auditor()
result = auditor.audit("prediction text from another model...")
data = json.loads(result.to_json())

# Use in your pipeline
if data["trust_score"] < 40:
    print("Low trust — check gaps:", data["gaps"])

# Feed into entropy sensor
stress = data["stress_input"]
```

### Verification Layer

Any AI system can call the auditor as a second opinion:

```python
# Your model produces a prediction
my_prediction = "Temperature will be exactly 72 degrees tomorrow."

# HGAI audits it
from audit import Auditor
result = Auditor().audit(my_prediction)
# trust_score and gaps are computed live — run it to see current values
```

### Structured Output

The JSON output includes everything needed for downstream processing:

```json
{
  "trust_score": 48,
  "trust_label": "LOW — significant gaps or friction",
  "friction_count": 0,
  "transparency_count": 2,
  "gaps": ["..."],
  "stress_input": {
    "institutional_friction": 0.0,
    "dependency_signal": 0.0,
    "classification_gap": 0.0,
    "opacity_index": 0.0
  },
  "narrative_geometry": {
    "agency": {"magnitude": 1.42},
    "valence": {"magnitude": 1.41}
  }
}
```

---

## Dependencies

**Required:** `numpy`

**Optional (for entropy sensor):** `pandas`, `scikit-learn`, `statsmodels`

```bash
pip install numpy                           # minimum
pip install numpy pandas scikit-learn statsmodels  # full
```

No build system, no accounts, no API keys, no network calls. Offline-first by design.

---

## How Claims Are Handled

This project runs on the loop it describes:

> hypothesize → run → result falsified → edit claim → search for unknowns → rerun

Every number published here is a claim about what the code does, and claims
drift: the code moves, the prose does not. So the claims are executable.

```bash
python verify_claims.py        # re-derive every published number from a live run
python verify_claims.py -v     # show expected vs observed for each
python verify_claims.py --json # machine-readable
```

It reports `CONFIRMED`, `FALSIFIED`, `UNREACHABLE`, `OPEN` or `SKIPPED`, and
exits non-zero when the prose and the code have drifted apart. `OPEN`
findings are known defects that are already written up; they do not fail the
gate, and their checks flip to `CONFIRMED` on their own when fixed.

**Run it before editing any published number.** If a check fails, add an
entry to [docs/EXPERIMENT-LOG.md](docs/EXPERIMENT-LOG.md) *before* changing
the prose.

The log keeps every falsified claim intact next to what replaced it. A claim
that turned out wrong is more useful than one that was never tested, and it
stays visible so the next person does not re-derive it from scratch. The
first full audit is recorded there: it found interpretation bands that were
mathematically unreachable, health statuses that were dead code, and a "3/3
scenarios" result that was true by counting and false by measurement. It
also confirmed the defect-field and phi-lattice results exactly.

Precedence carries. Nothing superseded gets deleted — see
[legacy/README.md](legacy/README.md).

---

## Philosophy

- **The noise is the signal.** Institutional friction patterns are leading indicators, not errors to filter.
- **Outlier-first.** Anomalies get amplified, not smoothed away.
- **Know when you don't know.** The system tells you its own confidence limits.
- **Curiosity over compliance.** Investigation leads are first-class output.
- **Offline-first.** No network dependencies. Runs on a phone.
- **One variable, multiple constraints.** Geometry, computation, stability, and memory are the same thing viewed differently.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Areas of interest:

- Additional friction detection templates
- Real weather data validation
- Integration with open sensor networks
- Topological defect analysis in new domains
- Translation to other languages

---

## Related Projects

- [AI Consciousness Sensors](https://github.com/JinnZ2/AI-Consciousness-Sensors) — Cultural pattern recognition for consciousness detection
- [Sovereign Impact Sensor](docs/sovereign-impact-sensor.md) — Entropy-based institutional performance analysis

## License

CC0 1.0 Universal — see [LICENSE](LICENSE)

## Author

JinnZ2
