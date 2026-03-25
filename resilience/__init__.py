"""
Resilience — Text-to-Entropy Bridge

Scans institutional text (filings, news articles, reports) for patterns
of institutional friction, risk reclassification, and statistical smoothing.
Converts qualitative signals into structured risk inputs that feed directly
into the Sovereign-Impact-Sensor (EntropySensor) pipeline.

Modules:
    detectors: Scanner, Template, Alert, RiskMatrix, ALL_TEMPLATES
    notices: NoticeGenerator for formal alert output
"""

from resilience.detectors import (
    Alert,
    RiskMatrix,
    Scanner,
    Template,
    ALL_TEMPLATES,
)
from resilience.notices import NoticeGenerator

__all__ = [
    "Alert",
    "NoticeGenerator",
    "RiskMatrix",
    "Scanner",
    "Template",
    "ALL_TEMPLATES",
]
