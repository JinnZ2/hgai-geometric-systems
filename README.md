# HGAI Geometric Systems

A mathematical framework for analyzing system coherence, resilience, and viability — with tools anyone can use right now.

**Paste any forecast, report, or prediction. Get back a trust score and what's being missed.**

```bash
python audit.py "The agency revised the methodology. This was an isolated incident."
# Trust: 0/100 (UNRELIABLE) | 6 friction flags | 9 gaps
```

```bash
python audit.py "Ensemble models show 70% chance of 2-4 inches. Confidence moderate."
# Trust: 57/100 (MODERATE) | 0 friction | 1 gap
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

| Score | Interpretation |
|-------|---------------|
| > 7 | Highly coherent and sustainable |
| 5-7 | Strong coherence, good viability |
| 3-5 | Moderate coherence, stable |
| 1-3 | Weak coherence, stressed |
| < 0 | Negative coherence, declining/collapse |

---

## Repository Structure

### Tools (Use These)

| File | What It Does | How To Use |
|------|-------------|-----------|
| `audit.py` | Trust scoring for any prediction text | `python audit.py "text"` or `--json` |
| `hgai.py` | Full system analysis engine | `python hgai.py "text"` or `from hgai import HGAI` |

### Sensors & Detectors

| File | What It Does |
|------|-------------|
| `resilience/detectors.py` | 22 regex templates detecting institutional friction across 7 categories |
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
| `framework/core/m_s_calculator.py` | M(S) equation engine |
| `unified_field_monitor.py` | 64D geometric field encoding |
| `Unified_narrative.py` | Text-to-geometry narrative analysis |
| `ecological-calculus.py` | Relational ecosystem health assessment |
| `three-axis.py` | Curiosity-driven confusion investigation protocol |

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

---

## Key Results

### Institutional Friction Detection

22 regex templates across 7 categories detect the language of entropy denial:

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

### Defect Weather Model (3/3 Scenarios)

The defect-aware weather model beats conventional smoothing in all three test scenarios after adding precursor detection:

- **Curvature anomaly**: Pressure bending before it spikes
- **Entropy buildup**: Local disorder accumulating
- **P-T flux coupling**: Temperature-pressure gradients aligning

The blizzard scenario had zero defects visible to threshold-based detection. Precursor analysis caught 3 sub-threshold signals.

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
# trust_score: 35 (overconfidence: false_precision, deterministic_language)
# gaps: ["No uncertainty range stated", "Uses deterministic language"]
```

### Structured Output

The JSON output includes everything needed for downstream processing:

```json
{
  "trust_score": 57,
  "trust_label": "MODERATE",
  "friction_count": 0,
  "transparency_count": 6,
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

MIT License — see [LICENSE](LICENSE)

## Author

JinnZ2
