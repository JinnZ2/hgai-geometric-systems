# HGAI — Holistic Geometric AI Engine

## What It Does

HGAI is a unified analysis engine. Feed it any text — a news article, an institutional filing, a field observation, a conversation — and it returns:

- **M(S) Score**: System coherence rating (how healthy/viable is the system described)
- **Health Status**: THRIVING / HEALTHY / STRESSED / WARNING / CRITICAL
- **Friction Alerts**: Institutional friction patterns detected (reclassification, hedging, smoothing, downplay, opacity, dependency risk, communication delays)
- **Stress Input**: Numeric bridge directly into the EntropySensor for further analysis
- **Narrative Geometry**: 64D encoding of agency, valence, temporal, and presence signals
- **Curiosity Signals**: Cross-pattern investigation leads (the "unknown axis")

---

## Quick Start

### From the Command Line (Phone-Friendly)

```bash
# Paste text as an argument
python hgai.py "The agency revised the methodology and outliers were removed."

# One-line quick summary
python hgai.py -q "paste any text here"

# Verbose mode (full alert details)
python hgai.py -v "paste any text here"

# Interactive mode (just run it, then paste)
python hgai.py
```

### As a Python Library

```python
from hgai import HGAI

engine = HGAI()

# Full analysis
report = engine.analyze("text from anywhere...")
print(report.render())

# One-line summary
print(engine.quick("text here"))

# Friction scan only (fastest)
print(engine.scan_only("text here"))

# Just the 64D geometry
geometry = engine.geometry_only("text here")
```

### With Formal Notices

```python
from hgai import HGAI

engine = HGAI()
report = engine.analyze(
    "The agency revised the methodology...",
    target_entities=["Regional Health Authority", "Analytics Contractor"]
)

# Print formal notices
for notice in report.notices:
    print(notice.render())
```

---

## How It Works

```
               INPUT TEXT
                  |
     +------------+------------+
     |            |            |
     v            v            v
  Friction     Narrative    Curiosity
  Scanner      Geometry     Extractor
     |         (64D)           |
     v            |            v
  RiskMatrix     |       Investigation
  StressInput    |          Leads
     |            |            |
     +-----+------+-----+-----+
           |             |
           v             v
      M(S) Score    Warnings &
      + Health      Signals
           |
           v
       HGAIReport
```

### Branch 1: Institutional Friction Scanner
Applies 24 regex templates across 7 categories to detect language patterns that signal institutional friction. Outputs alerts, a risk matrix, and a stress input bridge to the EntropySensor.

### Branch 2: Narrative Geometry Encoder
Encodes text into a 64-dimensional vector across four octants:
- **Agency** (16D): Active vs passive voice, first-person presence, engagement level
- **Valence** (16D): Positive vs negative sentiment, expansion/contraction signals
- **Temporal** (16D): Causal reasoning, change intensity, tense markers
- **Presence** (16D): Balance indicators, questioning, text density

### Branch 3: Curiosity Signal Extractor
Identifies cross-pattern signals worth investigating — the "unknown axis" from the three-axis protocol. These are combinations of friction categories that reveal systemic failure modes:
- Reclassification + Downplay = Active Classification Gap
- Data Opacity + Statistical Smoothing = Signal Suppression Pipeline
- Dependency Risk + Liability Hedging = Causal Chain Being Legally Severed
- High Agency + High Friction = Managed Disclosure / Narrative Control

### Consolidation: M(S) Score
Maps all signals to the M(S) equation components:
- **R_e (Resonance)**: Agency magnitude + positive valence
- **A (Adaptability)**: Temporal change handling + agency-based flexibility
- **D (Diversity)**: Valence range (penalizes one-sidedness) — note that
  M(S) defines D as *pathway multiplicity*, not sentiment mixture, so this
  proxy is inverted with respect to the equation: uniformly positive text,
  including a transparent good-faith source, scores as minimally diverse.
  Recorded as E4b in [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md)
- **C (Curiosity)**: Presence signals (questions, balance, engagement)
- **L (Loss)**: Friction count + severity + negative valence

---

## M(S) Score Interpretation

M(S) is bounded above at +1.0 (all four coherence factors are in [0, 1], so
their product cannot exceed 1). The health gates track that range:

| Score | Health | Meaning |
|-------|--------|---------|
| > 0.5 | THRIVING | High coherence, actively healthy |
| 0.3 - 0.5 | HEALTHY | Good coherence, stable |
| 0.1 - 0.3 | STRESSED | Some friction or imbalance |
| 0 - 0.1 | WARNING | Approaching critical threshold |
| < 0 | CRITICAL | Negative coherence, declining or collapsing |

> These gates previously sat at 5 / 3 / 1 against a score that cannot exceed
> 1.0, which made THRIVING, HEALTHY and STRESSED unreachable — the engine
> had two possible outputs and advertised five. See
> [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) entry E2.
>
> **Known limitation (E4).** Rescaling the gates removed an arithmetic
> impossibility, but a second defect still constrains what the engine
> reports in practice: the four octant magnitudes are on incompatible scales
> (agency and valence peak near 1.42, temporal near 0.20, presence near
> 0.10), so A and C are structurally starved in a multiplicative core. Two
> of the four factors effectively decide the score. Unresolved.

---

## Connection to Other Modules

HGAI is the front door. This table previously listed ten modules as
"connected" without distinguishing what the engine actually loads from what
it aspires to. Three categories, checked against the imports:

### Imported and executed

These are `import` statements in `hgai.py`. Removing any one breaks the engine.

| Module | Connection |
|--------|-----------|
| `framework/core/m_s_calculator.py` | Core M(S) equation used for scoring |
| `resilience/detectors.py` | Scanner + templates for friction detection |
| `resilience/notices.py` | Formal notice generation from alerts |

### Methodology reimplemented, module not imported

The *approach* was carried forward into `_encode_text_geometry` and
`_extract_curiosity_signals`; the originating code is not loaded. Those
modules still run standalone under `legacy/` and hold capability the
reimplementation does not cover — see [../legacy/README.md](../legacy/README.md).

| Module | What HGAI took from it |
|--------|------------------------|
| `legacy/unified_field_monitor.py` | 64D encoding methodology |
| `legacy/Unified_narrative.py` | Narrative encoding approach |
| `legacy/ecological-calculus.py` | Health status framework |
| `legacy/three-axis.py` | Curiosity signal / investigation lead methodology |

### Designed bridges, not yet wired

`StressInput` is built and returned by the engine, and nothing currently
consumes it. These are the intended consumers.

| Module | Intended connection | State |
|--------|--------------------|-------|
| `sovereign_impact_sensor.py` | StressInput → `EntropySensor.calibrate_signals()` | not called; also requires pandas / scikit-learn / statsmodels, so it will not import under the documented minimal install |
| `flux_sensor.py` | Flux data augments the temporal geometry octant | not called |
| `weather_node_network.py` | Risk triggers feed the ensemble pipeline | not called |

See [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) entry E9 for how this table was
checked.

---

## Dependencies

- `numpy` (required)
- `pandas`, `scikit-learn`, `statsmodels` (for EntropySensor bridge, optional for basic use)

---

## Design Philosophy

- **Text in, report out.** No setup required.
- **Phone-friendly.** Works from a terminal, Termux, or Pythonista.
- **Offline-first.** No network calls, no API keys, no accounts.
- **The noise is the signal.** Institutional friction patterns are leading indicators, not errors to filter.
- **Curiosity over compliance.** Investigation leads are first-class output.
