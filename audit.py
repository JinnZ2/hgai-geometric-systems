#!/usr/bin/env python3
"""
Forecast Auditor — Trust Scoring for Predictions

Paste any forecast, prediction, or institutional report. Get back:
- Trust score (0-100): How much should you trust this?
- What's missing: Gaps the source is probably not telling you
- Friction flags: Language patterns that signal smoothing or hedging
- Precursor signals: Hidden signals the source may be filtering out
- Confidence audit: Is the stated confidence justified?

Two modes:
    Human:  python audit.py "paste forecast text here"
    AI/JSON: python audit.py --json "text here"
    API:     from audit import Auditor; result = Auditor().audit("text")

Connects all HGAI modules:
    - resilience/detectors.py: Institutional friction scanning
    - hgai.py: M(S) coherence scoring + narrative geometry
    - chaos_weather_ai.py: Regime detection philosophy
    - defect_weather_model.py: Precursor detection concepts

Dependencies:
    - numpy (required)
    - json (stdlib, for JSON mode)
"""

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from resilience.detectors import (
    ALL_TEMPLATES,
    Alert,
    Category,
    RiskMatrix,
    Scanner,
    Severity,
    StressInput,
)
from hgai import HGAI, HGAIReport, _encode_text_geometry


# ---------------------------------------------------------------------------
# Trust Score Calculation
# ---------------------------------------------------------------------------

# Patterns that INCREASE trust (transparency signals)
_TRANSPARENCY_PATTERNS = [
    (r"uncertainty\s+(?:range|band|estimate|interval)", 3, "reports_uncertainty"),
    (r"confidence\s+(?:interval|level|range)", 3, "states_confidence"),
    (r"(?:probability|chance|likelihood)\s+of", 2, "uses_probability"),
    (r"(?:ensemble|multiple\s+model)", 2, "uses_ensemble"),
    (r"(?:range|between)\s+\d+\s*(?:and|to|-)\s*\d+", 2, "gives_range"),
    (r"(?:could|may|might)\s+(?:change|shift|vary)", 1, "acknowledges_variability"),
    (r"(?:monitor|watch|track|update)", 1, "promises_updates"),
    (r"(?:raw\s+data|open\s+(?:data|source|access))", 3, "data_transparency"),
    (r"(?:methodology|method)\s+(?:available|published|described|based)", 2, "methodology_open"),
    (r"(?:based\s+on|derived\s+from|using)\s+(?:the\s+)?(?:\w+\s+){0,3}(?:model|ensemble|data)", 2, "methodology_cited"),
    (r"(?:limitations?|caveats?|assumptions?)", 2, "states_limitations"),
]

# Patterns that DECREASE trust (overconfidence signals)
_OVERCONFIDENCE_PATTERNS = [
    (r"(?:will|shall)\s+(?:be|reach|hit|exceed)", -3, "deterministic_language"),
    (r"(?:guaranteed|certain|definite|assured)", -4, "certainty_claims"),
    (r"(?:no\s+(?:risk|chance|possibility))", -3, "denies_risk"),
    (r"(?:exactly|precisely)\s+\d+", -2, "false_precision"),
    (r"(?:unprecedented|never\s+before)", -2, "unprecedented_framing"),
    (r"(?:nothing\s+to\s+worry|no\s+cause\s+for)", -3, "dismissive"),
    (r"(?:experts?\s+(?:agree|confirm|say))", -1, "authority_appeal"),
    (r"(?:the\s+science\s+(?:is|says))", -1, "monolithic_science"),
]

# Number extraction for quantitative analysis
_NUMBER_PATTERN = re.compile(
    r'(-?\d+\.?\d*)\s*(?:to|-|–)\s*(-?\d+\.?\d*)|'
    r'(?:between)\s+(-?\d+\.?\d*)\s+(?:and)\s+(-?\d+\.?\d*)|'
    r'(?:about|approximately|around|roughly|~)\s*(-?\d+\.?\d*)|'
    r'(-?\d+\.?\d*)%'
)


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    """Complete audit of a forecast or prediction.

    Parameters
    ----------
    trust_score : int
        0-100 trust rating.
    trust_label : str
        Human-readable trust level.
    friction_count : int
        Institutional friction detections.
    transparency_count : int
        Transparency signals detected.
    overconfidence_count : int
        Overconfidence signals detected.
    friction_alerts : list
        Detailed friction alerts.
    transparency_signals : list
        What the source does RIGHT.
    overconfidence_signals : list
        Where the source overclaims.
    gaps : list
        What's probably missing from this forecast.
    m_s_score : float
        System coherence score.
    health_status : str
        Overall system health.
    stress_input : dict or None
        EntropySensor bridge values.
    narrative_geometry : dict
        64D encoding summary.
    numbers_found : list
        Extracted numerical claims.
    precision_audit : str
        Are the numbers appropriately precise?
    """
    trust_score: int
    trust_label: str
    friction_count: int
    transparency_count: int
    overconfidence_count: int
    friction_alerts: List[Dict[str, str]]
    transparency_signals: List[str]
    overconfidence_signals: List[str]
    gaps: List[str]
    m_s_score: float
    health_status: str
    stress_input: Optional[Dict[str, float]]
    narrative_geometry: Dict[str, Any]
    numbers_found: List[Dict[str, Any]]
    precision_audit: str

    def render(self) -> str:
        """Human-readable audit report."""
        lines = []
        sep = "=" * 55

        # Header with trust score
        lines.append(sep)
        lines.append(f"  FORECAST AUDIT — Trust: {self.trust_score}/100 ({self.trust_label})")
        lines.append(sep)

        # Trust bar
        filled = self.trust_score // 2
        bar = "#" * filled + "." * (50 - filled)
        lines.append(f"  [{bar}]")

        # Quick stats
        lines.append(f"\n  Friction flags:     {self.friction_count}")
        lines.append(f"  Transparency:       {self.transparency_count}")
        lines.append(f"  Overconfidence:     {self.overconfidence_count}")
        lines.append(f"  M(S) coherence:     {self.m_s_score:.2f} ({self.health_status})")

        # What's missing
        if self.gaps:
            lines.append(f"\n--- What's Probably Missing ---")
            for gap in self.gaps:
                lines.append(f"  ? {gap}")

        # Friction
        if self.friction_alerts:
            lines.append(f"\n--- Friction Flags ({self.friction_count}) ---")
            for alert in self.friction_alerts[:8]:
                lines.append(
                    f"  ! [{alert['severity']}] {alert['category']}: "
                    f"{alert['match']}"
                )

        # Overconfidence
        if self.overconfidence_signals:
            lines.append(f"\n--- Overconfidence Signals ---")
            for sig in self.overconfidence_signals:
                lines.append(f"  ^ {sig}")

        # Transparency (what they did right)
        if self.transparency_signals:
            lines.append(f"\n--- Transparency (Good) ---")
            for sig in self.transparency_signals:
                lines.append(f"  + {sig}")

        # Number precision
        if self.numbers_found:
            lines.append(f"\n--- Numerical Claims ---")
            lines.append(f"  {self.precision_audit}")
            for num in self.numbers_found[:5]:
                lines.append(f"  {num['text']}: {num['type']}")

        # Stress bridge
        if self.stress_input:
            lines.append(f"\n--- Stress Input (for EntropySensor) ---")
            for k, v in self.stress_input.items():
                if v > 0:
                    bar = "#" * min(int(v * 3), 30)
                    lines.append(f"  {k:30s}: {v:5.2f} {bar}")

        lines.append(f"\n{sep}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """JSON output for AI-to-AI verification."""
        return json.dumps({
            "trust_score": self.trust_score,
            "trust_label": self.trust_label,
            "friction_count": self.friction_count,
            "transparency_count": self.transparency_count,
            "overconfidence_count": self.overconfidence_count,
            "gaps": self.gaps,
            "friction_alerts": self.friction_alerts,
            "transparency_signals": self.transparency_signals,
            "overconfidence_signals": self.overconfidence_signals,
            "m_s_score": round(self.m_s_score, 4),
            "health_status": self.health_status,
            "stress_input": self.stress_input,
            "numbers_found": self.numbers_found,
            "precision_audit": self.precision_audit,
            "narrative_geometry": {
                k: {
                    "magnitude": round(v.get("magnitude", 0), 4)
                    if isinstance(v, dict) else v
                }
                for k, v in self.narrative_geometry.items()
            },
        }, indent=2)


# ---------------------------------------------------------------------------
# Number Extraction & Precision Audit
# ---------------------------------------------------------------------------

def extract_numbers(text: str) -> List[Dict[str, Any]]:
    """Extract numerical claims from text.

    Classifies each as: range, approximate, percentage, or point estimate.
    """
    numbers = []
    for match in _NUMBER_PATTERN.finditer(text):
        context = text[max(0, match.start()-30):match.end()+30].strip()

        if match.group(1) and match.group(2):
            numbers.append({
                "text": context,
                "type": "range",
                "low": float(match.group(1)),
                "high": float(match.group(2)),
            })
        elif match.group(3) and match.group(4):
            numbers.append({
                "text": context,
                "type": "range",
                "low": float(match.group(3)),
                "high": float(match.group(4)),
            })
        elif match.group(5):
            numbers.append({
                "text": context,
                "type": "approximate",
                "value": float(match.group(5)),
            })
        elif match.group(6):
            numbers.append({
                "text": context,
                "type": "percentage",
                "value": float(match.group(6)),
            })

    return numbers


def audit_precision(numbers: List[Dict], text: str) -> str:
    """Judge whether the numerical precision is appropriate."""
    if not numbers:
        return "No numerical claims found. Qualitative-only forecast."

    ranges = sum(1 for n in numbers if n["type"] == "range")
    approx = sum(1 for n in numbers if n["type"] == "approximate")
    point = sum(1 for n in numbers if n["type"] not in ("range", "approximate"))

    if ranges > point:
        return "Good: mostly ranges/intervals. Acknowledges uncertainty."
    elif approx > 0:
        return "Mixed: some approximations used. Moderate precision claims."
    else:
        return "Warning: point estimates dominate. May overstate precision."


# ---------------------------------------------------------------------------
# Gap Detection
# ---------------------------------------------------------------------------

def detect_gaps(
    text: str,
    friction_alerts: List[Alert],
    transparency_signals: List[str],
    numbers: List[Dict],
    geometry: Dict[str, Any],
) -> List[str]:
    """Identify what's probably missing from this forecast.

    Uses friction patterns, absence of transparency signals, and
    narrative geometry to infer gaps.
    """
    gaps = []
    text_lower = text.lower()

    # --- Missing uncertainty quantification ---
    has_uncertainty = any("uncertainty" in s or "confidence" in s
                         for s in transparency_signals)
    if not has_uncertainty:
        gaps.append(
            "No uncertainty range or confidence interval stated. "
            "How sure are they, really?"
        )

    # --- Missing methodology ---
    has_method = any("methodology" in s for s in transparency_signals)
    if not has_method and len(text) > 200:
        gaps.append(
            "No methodology described. How was this forecast produced?"
        )

    # --- Deterministic framing ---
    if "will" in text_lower and "could" not in text_lower and "may" not in text_lower:
        gaps.append(
            "Uses deterministic language ('will') without hedging. "
            "Where's the probability?"
        )

    # --- Friction-inferred gaps ---
    friction_cats = {a.category for a in friction_alerts}

    if Category.RECLASSIFICATION in friction_cats:
        gaps.append(
            "Reclassification language detected. The original data "
            "may tell a different story than what's reported."
        )

    if Category.STATISTICAL_SMOOTHING in friction_cats:
        gaps.append(
            "Statistical smoothing applied. Outliers may have been "
            "removed — those 'outliers' could be the most important signal."
        )

    if Category.DATA_OPACITY in friction_cats:
        gaps.append(
            "Data access restricted or proprietary. Independent "
            "verification is not possible."
        )

    if Category.DOWNPLAY in friction_cats:
        gaps.append(
            "Impact minimization language detected. The actual impact "
            "may be higher than stated."
        )

    # --- Cross-pattern gaps ---
    if (Category.RECLASSIFICATION in friction_cats
            and Category.DOWNPLAY in friction_cats):
        gaps.append(
            "CLASSIFICATION GAP: Reclassification + downplay co-occurring. "
            "Compare raw vs. revised figures if available."
        )

    if (Category.DATA_OPACITY in friction_cats
            and Category.STATISTICAL_SMOOTHING in friction_cats):
        gaps.append(
            "SIGNAL SUPPRESSION: Data opacity + smoothing = can't verify "
            "what was filtered out. Seek alternative data sources."
        )

    # --- Numerical gaps ---
    if not numbers and len(text) > 100:
        gaps.append(
            "No numerical claims found. Qualitative-only forecasts "
            "are harder to verify."
        )

    ranges = sum(1 for n in numbers if n["type"] == "range")
    if numbers and ranges == 0:
        gaps.append(
            "All numbers are point estimates, not ranges. Real forecasts "
            "have uncertainty — where is it?"
        )

    # --- Low agency ---
    agency = geometry.get("agency", {})
    if agency.get("ratio", 0.5) < 0.3:
        gaps.append(
            "Passive voice dominant. Who is responsible for this "
            "forecast? Accountability is unclear."
        )

    return gaps


# ---------------------------------------------------------------------------
# Trust Score Calculation
# ---------------------------------------------------------------------------

def calculate_trust(
    friction_count: int,
    max_severity: int,
    transparency_count: int,
    transparency_score: int,
    overconfidence_count: int,
    overconfidence_score: int,
    gap_count: int,
    m_s_score: float,
) -> int:
    """Calculate the 0-100 trust score.

    Starts at 50 (neutral) and adjusts based on signals.
    """
    score = 50.0

    # Transparency lifts trust
    score += transparency_score

    # Overconfidence lowers trust
    score += overconfidence_score  # already negative

    # Friction lowers trust
    score -= friction_count * 2
    score -= max_severity * 3

    # Gaps lower trust
    score -= gap_count * 3

    # M(S) adjusts: positive = trustworthy system, negative = declining
    if m_s_score > 0:
        score += min(m_s_score * 5, 10)
    else:
        score += max(m_s_score * 5, -15)

    return max(0, min(100, int(score)))


def trust_label(score: int) -> str:
    """Human-readable trust label."""
    if score >= 80:
        return "HIGH — transparent, uncertainty-aware"
    elif score >= 60:
        return "MODERATE — some gaps but reasonable"
    elif score >= 40:
        return "LOW — significant gaps or friction"
    elif score >= 20:
        return "VERY LOW — high friction, low transparency"
    else:
        return "UNRELIABLE — institutional friction dominates"


# ---------------------------------------------------------------------------
# Main Auditor
# ---------------------------------------------------------------------------

class Auditor:
    """Forecast audit engine.

    Combines all HGAI modules into a single trust-scoring pipeline.
    Works for weather forecasts, institutional reports, news articles,
    or any prediction text.

    Parameters
    ----------
    context_window : int
        Characters of context around friction matches. Default 50.

    Examples
    --------
    >>> auditor = Auditor()
    >>> result = auditor.audit("The forecast calls for 3-6 inches...")
    >>> print(result.render())        # human mode
    >>> print(result.to_json())       # AI/JSON mode
    >>> print(result.trust_score)     # just the number
    """

    def __init__(self, context_window: int = 50):
        self.hgai = HGAI(context_window=context_window)
        self.scanner = self.hgai.scanner

    def audit(self, text: str) -> AuditResult:
        """Run full audit on any text.

        Parameters
        ----------
        text : str
            Forecast, prediction, report, or article text.

        Returns
        -------
        AuditResult
            Complete audit with trust score, gaps, and signals.
        """
        if not text or not text.strip():
            return AuditResult(
                trust_score=0,
                trust_label="NO INPUT",
                friction_count=0,
                transparency_count=0,
                overconfidence_count=0,
                friction_alerts=[],
                transparency_signals=[],
                overconfidence_signals=[],
                gaps=["No text provided."],
                m_s_score=0.0,
                health_status="UNKNOWN",
                stress_input=None,
                narrative_geometry={},
                numbers_found=[],
                precision_audit="",
            )

        # --- Run HGAI analysis ---
        hgai_report = self.hgai.analyze(text)

        # --- Transparency signals ---
        transparency_signals = []
        transparency_score = 0
        for pattern, score, name in _TRANSPARENCY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                transparency_signals.append(name)
                transparency_score += score

        # --- Overconfidence signals ---
        overconfidence_signals = []
        overconfidence_score = 0
        for pattern, score, name in _OVERCONFIDENCE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                overconfidence_signals.append(name)
                overconfidence_score += score

        # --- Number extraction ---
        numbers = extract_numbers(text)
        precision = audit_precision(numbers, text)

        # --- Friction alerts (formatted) ---
        friction_dicts = []
        for alert in hgai_report.friction_alerts:
            friction_dicts.append({
                "severity": alert.severity.name,
                "category": alert.category.value,
                "match": alert.matched_text,
                "description": alert.description,
            })

        # --- Gap detection ---
        geometry = _encode_text_geometry(text)
        gaps = detect_gaps(
            text,
            hgai_report.friction_alerts,
            transparency_signals,
            numbers,
            geometry,
        )

        # --- Trust score ---
        max_sev = 0
        if hgai_report.friction_alerts:
            max_sev = max(a.severity.value for a in hgai_report.friction_alerts)

        score = calculate_trust(
            friction_count=len(hgai_report.friction_alerts),
            max_severity=max_sev,
            transparency_count=len(transparency_signals),
            transparency_score=transparency_score,
            overconfidence_count=len(overconfidence_signals),
            overconfidence_score=overconfidence_score,
            gap_count=len(gaps),
            m_s_score=hgai_report.m_s_score,
        )

        # --- Stress input ---
        stress_dict = None
        if hgai_report.stress_input:
            stress_dict = hgai_report.stress_input.to_dict()

        return AuditResult(
            trust_score=score,
            trust_label=trust_label(score),
            friction_count=len(hgai_report.friction_alerts),
            transparency_count=len(transparency_signals),
            overconfidence_count=len(overconfidence_signals),
            friction_alerts=friction_dicts,
            transparency_signals=transparency_signals,
            overconfidence_signals=overconfidence_signals,
            gaps=gaps,
            m_s_score=hgai_report.m_s_score,
            health_status=hgai_report.health_status,
            stress_input=stress_dict,
            narrative_geometry=hgai_report.narrative_geometry,
            numbers_found=numbers,
            precision_audit=precision,
        )

    def quick(self, text: str) -> str:
        """One-line trust score for fast mobile use."""
        result = self.audit(text)
        return (
            f"Trust: {result.trust_score}/100 ({result.trust_label}) | "
            f"{result.friction_count} friction | "
            f"{len(result.gaps)} gaps"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    json_mode = "--json" in sys.argv or "-j" in sys.argv
    quick_mode = "--quick" in sys.argv or "-q" in sys.argv

    text_args = [
        a for a in sys.argv[1:]
        if a not in ("--json", "-j", "--quick", "-q")
    ]

    auditor = Auditor()

    if text_args:
        text = " ".join(text_args)
        result = auditor.audit(text)

        if json_mode:
            print(result.to_json())
        elif quick_mode:
            print(auditor.quick(text))
        else:
            print(result.render())
    else:
        # Interactive mode
        print("=" * 50)
        print("  Forecast Auditor")
        print("  Paste text, press Enter twice to audit.")
        print("  Commands: quit, json, quick")
        print("=" * 50)

        while True:
            print("\n> ", end="", flush=True)
            try:
                lines = []
                while True:
                    line = input()
                    if line == "":
                        if lines:
                            break
                        continue
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\nDone.")
                break

            text = "\n".join(lines).strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                break
            if text.lower() == "json":
                json_mode = True
                print("JSON mode ON.")
                continue
            if text.lower() == "quick":
                print(auditor.quick(text))
                continue

            result = auditor.audit(text)
            if json_mode:
                print(result.to_json())
            else:
                print(result.render())


if __name__ == "__main__":
    main()
