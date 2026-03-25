# Resilience Detectors — Text-to-Entropy Bridge

## Overview

The `resilience` package scans institutional text (filings, news articles, reports) for language patterns that signal institutional friction, risk reclassification, and statistical smoothing. It converts these qualitative signals into structured risk inputs that feed directly into the Sovereign-Impact-Sensor (EntropySensor) pipeline.

This fills the gap between where institutional friction first appears (in language) and where it eventually shows up (in numeric data). By the time the numbers change, the entropy event has already happened.

---

## Architecture

```
Raw Text (filings, articles, reports)
        |
        v
    Scanner (regex template matching)
        |
        v
    ScanReport (list of Alerts)
        |
        +---> NoticeGenerator (formal output per entity)
        |
        +---> RiskMatrix (aggregated category scores)
                    |
                    v
              StressInput (EntropySensor bridge)
                    |
                    v
              EntropySensor.calibrate_signals()
```

---

## Quick Start

```python
from resilience.detectors import Scanner, ALL_TEMPLATES, NoticeGenerator

scanner = Scanner()
scanner.add_templates(ALL_TEMPLATES)
report = scanner.scan("text from a filing or news article...")

for alert in report.alerts:
    print(alert.summary())

# Generate formal notices from alerts
gen = NoticeGenerator()
notices = gen.generate_batch(report.alerts, target_entities=["Company X"])

# Feed directly into the stress model pipeline
matrix = scanner.to_risk_matrix("text...")
stress_input = matrix.to_stress_input()
```

---

## Template Categories

The scanner ships with 22 templates across 7 friction categories:

| Category | What It Detects | Example Pattern |
|----------|----------------|-----------------|
| **Reclassification** | Retroactive event recoding | "revised the methodology", "reclassified as" |
| **Liability Hedging** | Causal distancing language | "not directly attributable", "no evidence of" |
| **Statistical Smoothing** | Signal suppression methods | "outliers were removed", "rolling average" |
| **Downplay** | Impact minimization | "isolated incident", "minimal impact" |
| **Data Opacity** | Access restriction | "data not yet available", "proprietary methodology" |
| **Dependency Risk** | T_infra stress signals | "system outage", "single point of failure" |
| **Communication Friction** | Reporting delays/conflicts | "delayed notification", "conflicting reports" |

---

## Stress Input Bridge

The `RiskMatrix.to_stress_input()` method maps category scores to four dimensions compatible with the EntropySensor:

| Stress Dimension | Source Categories |
|-----------------|-------------------|
| `institutional_friction` | Liability Hedging + Communication Friction |
| `dependency_signal` | Dependency Risk |
| `classification_gap` | Reclassification + Statistical Smoothing |
| `opacity_index` | Data Opacity + Downplay |

Scores are normalized per 1000 characters for cross-document comparison.

---

## Notice Generator

The `NoticeGenerator` converts alerts into formal, timestamped notices with:

- Category-specific framing and recommendations
- Evidence chain (matched text + context + signal description)
- Batch generation: groups alerts by category per target entity
- Summary reports with severity breakdown and stress input values

---

## Design Philosophy

- **Regex-based**: No NLP/ML dependencies. Patterns are transparent and auditable.
- **Offline-first**: No network calls. Runs anywhere Python runs.
- **Minimal dependencies**: Only `re`, `dataclasses`, `numpy`.
- **The noise is the signal**: Templates detect the language of entropy denial before it shows up in the numbers.
