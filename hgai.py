#!/usr/bin/env python3
"""
HGAI — Holistic Geometric AI

A unified engine for analyzing system coherence, institutional friction,
and resilience from text input. Connects all modules in the geometric
systems framework into a single, phone-friendly interface.

Usage:
    # Command line (paste text as argument)
    python hgai.py "The agency revised the methodology..."

    # Interactive mode (just run it and paste)
    python hgai.py

    # As a library
    from hgai import HGAI
    engine = HGAI()
    report = engine.analyze("text here...")
    print(report)

Modules connected:
    - M(S) Calculator: System coherence scoring
    - Resilience Scanner: Institutional friction detection
    - Narrative Monitor: 64D geometric text encoding
    - Ecological Monitor: Relational health assessment
    - Three-Axis Protocol: Confusion investigation
    - Entropy Sensor: System stress calibration (when data available)
    - Flux Sensor: Phase transition detection (when data available)
"""

import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# --- Internal imports: all modules in the framework ---
from framework.core.m_s_calculator import (
    MSCalculator,
    SystemMetrics as CoreMetrics,
)
from resilience.detectors import (
    ALL_TEMPLATES,
    Alert,
    Category,
    RiskMatrix,
    Scanner,
    StressInput,
)
from resilience.notices import Notice, NoticeGenerator


# ---------------------------------------------------------------------------
# Unified Report
# ---------------------------------------------------------------------------

@dataclass
class HGAIReport:
    """Consolidated output from all HGAI analysis branches.

    Parameters
    ----------
    text_analyzed : str
        The input text (truncated for display).
    m_s_score : float
        Consolidated M(S) coherence score.
    m_s_interpretation : str
        Human-readable interpretation of the M(S) score.
    health_status : str
        Overall system health assessment.
    friction_alerts : list of Alert
        Institutional friction detections.
    risk_matrix : RiskMatrix or None
        Aggregated risk by category.
    stress_input : StressInput or None
        EntropySensor-ready bridge values.
    notices : list of Notice
        Formal notices generated.
    narrative_geometry : dict
        64D encoding breakdown (agency, valence, temporal, presence).
    warnings : list of str
        Specific warnings and insights.
    curiosity_signals : list of str
        Patterns worth investigating further (three-axis leads).
    """
    text_analyzed: str = ""
    m_s_score: float = 0.0
    m_s_interpretation: str = ""
    health_status: str = "UNKNOWN"
    friction_alerts: List[Alert] = field(default_factory=list)
    risk_matrix: Optional[RiskMatrix] = None
    stress_input: Optional[StressInput] = None
    notices: List[Notice] = field(default_factory=list)
    narrative_geometry: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    curiosity_signals: List[str] = field(default_factory=list)

    def render(self, verbose: bool = False) -> str:
        """Render the report as formatted text.

        Parameters
        ----------
        verbose : bool
            If True, include full alert details and geometry vectors.
        """
        lines = []
        sep = "=" * 60

        # Header
        lines.append(sep)
        lines.append("  HGAI ANALYSIS REPORT")
        lines.append(sep)

        # Input preview
        preview = self.text_analyzed[:120].replace("\n", " ").strip()
        if len(self.text_analyzed) > 120:
            preview += "..."
        lines.append(f"\nInput: \"{preview}\"")
        lines.append(f"Length: {len(self.text_analyzed)} chars")

        # M(S) Score
        lines.append(f"\n--- System Coherence (M(S)) ---")
        lines.append(f"Score: {self.m_s_score:.2f}")
        lines.append(f"Status: {self.m_s_interpretation}")
        lines.append(f"Health: {self.health_status}")

        # Friction Summary
        if self.friction_alerts:
            lines.append(f"\n--- Institutional Friction ({len(self.friction_alerts)} detections) ---")
            if self.risk_matrix:
                lines.append(f"Alert density: {self.risk_matrix.density:.1f} per 1000 chars")
                lines.append(f"Total risk: {self.risk_matrix.total_risk:.2f}")

            # Category breakdown
            cat_counts: Dict[str, int] = {}
            for alert in self.friction_alerts:
                name = alert.category.value
                cat_counts[name] = cat_counts.get(name, 0) + 1
            for cat_name, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {cat_name:30s}: {count}")

            if verbose:
                lines.append("")
                for alert in self.friction_alerts:
                    lines.append(f"  {alert.summary()}")
        else:
            lines.append(f"\n--- Institutional Friction ---")
            lines.append("No friction patterns detected.")

        # Stress Bridge
        if self.stress_input:
            lines.append(f"\n--- Stress Input (EntropySensor Bridge) ---")
            for key, val in self.stress_input.to_dict().items():
                bar = "#" * min(int(val * 3), 40)
                lines.append(f"  {key:30s}: {val:6.2f} {bar}")

        # Narrative Geometry
        if self.narrative_geometry:
            lines.append(f"\n--- Narrative Geometry (64D Encoding) ---")
            for octant, data in self.narrative_geometry.items():
                if isinstance(data, dict):
                    mag = data.get("magnitude", 0.0)
                    bar = "#" * min(int(mag * 20), 30)
                    lines.append(f"  {octant:20s}: mag={mag:.3f} {bar}")
                    if verbose and "top_signals" in data:
                        for sig in data["top_signals"]:
                            lines.append(f"    - {sig}")

        # Warnings
        if self.warnings:
            lines.append(f"\n--- Warnings ---")
            for w in self.warnings:
                lines.append(f"  ! {w}")

        # Curiosity signals
        if self.curiosity_signals:
            lines.append(f"\n--- Worth Investigating ---")
            for c in self.curiosity_signals:
                lines.append(f"  ? {c}")

        # Notices count
        if self.notices:
            lines.append(f"\n--- Formal Notices: {len(self.notices)} generated ---")
            if verbose:
                for notice in self.notices:
                    lines.append(notice.render())

        lines.append(f"\n{sep}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Narrative Geometry Encoder (inline, avoids import issues)
# ---------------------------------------------------------------------------

# Word lists for lightweight text-to-geometry encoding
_AGENCY_VERBS = {
    "create", "creating", "created", "build", "building", "built",
    "decide", "decided", "choose", "chose", "lead", "leading",
    "initiate", "initiated", "design", "designed", "designing",
    "establish", "established", "establishing",
    "implement", "implemented", "implementing",
    "develop", "developed", "developing",
    "launch", "launched", "propose", "proposed",
    "drive", "driving", "drove",
    "manage", "managed", "managing",
    "control", "direct", "organize", "organized",
    "execute", "transform", "transforming",
    "invent", "innovate", "innovating",
    "pioneer", "forge", "shape", "shaping", "craft",
    "adapt", "adapting", "adapted", "monitor", "monitoring",
    "share", "sharing", "respond", "responding",
}
_PASSIVE_MARKERS = {
    "was", "were", "been", "being", "affected", "impacted", "subjected",
    "forced", "required", "mandated", "compelled", "constrained",
    "restricted", "limited", "prevented", "denied", "excluded",
}
_POSITIVE_WORDS = {
    "good", "great", "excellent", "healthy", "strong", "resilient",
    "thriving", "growing", "adapting", "connected", "balanced",
    "sustainable", "vibrant", "diverse", "coherent", "stable",
    "improving", "recovering", "flourishing", "harmonious",
}
_NEGATIVE_WORDS = {
    "bad", "poor", "failing", "weak", "declining", "collapsing",
    "stressed", "broken", "fragmented", "toxic", "depleted",
    "unsustainable", "degraded", "eroding", "deteriorating",
    "fractured", "isolated", "rigid", "stagnant", "dying",
}
_CAUSAL_CONNECTORS = {
    "because", "therefore", "consequently", "thus", "hence", "since",
    "due", "caused", "resulted", "led", "implies", "means",
}
_CHANGE_WORDS = {
    "sudden", "rapid", "abrupt", "spike", "surge", "collapse",
    "crash", "shift", "transition", "transform", "plunge", "jump",
    "explode", "plummet", "skyrocket", "accelerate", "decelerate",
}
_BALANCE_WORDS = {
    "balance", "equilibrium", "stable", "steady", "maintained",
    "proportion", "symmetry", "homeostasis", "regulated", "calibrated",
}


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    return re.findall(r'[a-z]+', text.lower())


def _encode_text_geometry(text: str) -> Dict[str, Any]:
    """Encode text into a 64D geometric representation.

    Returns a dictionary with four 16D octants and their metadata:
    - agency: Active vs passive voice, first-person presence
    - valence: Positive vs negative sentiment, expansion/contraction
    - temporal: Causal reasoning, change signals, tense markers
    - presence: Balance indicators, relationship health, degradation
    """
    tokens = _tokenize(text)
    token_set = set(tokens)
    n_tokens = max(len(tokens), 1)

    # --- Octant 1: Agency (16D) ---
    agency_count = len(token_set & _AGENCY_VERBS)
    passive_count = len(token_set & _PASSIVE_MARKERS)
    first_person = sum(1 for t in tokens if t in ("i", "we", "my", "our"))
    third_person = sum(1 for t in tokens if t in ("they", "them", "their", "it"))

    agency_ratio = agency_count / (agency_count + passive_count + 1e-6)
    agency_vec = np.zeros(16)
    agency_vec[0] = agency_ratio
    agency_vec[1] = passive_count / n_tokens
    agency_vec[2] = first_person / n_tokens
    agency_vec[3] = third_person / n_tokens
    agency_vec[4] = agency_count / n_tokens
    agency_vec[5] = 1.0 if agency_ratio > 0.6 else 0.0
    agency_vec[6] = 1.0 if passive_count > agency_count else 0.0

    # --- Octant 2: Valence (16D) ---
    pos_count = len(token_set & _POSITIVE_WORDS)
    neg_count = len(token_set & _NEGATIVE_WORDS)
    valence_ratio = (pos_count - neg_count) / (pos_count + neg_count + 1e-6)

    valence_vec = np.zeros(16)
    valence_vec[0] = (valence_ratio + 1) / 2  # normalize to [0, 1]
    valence_vec[1] = pos_count / n_tokens
    valence_vec[2] = neg_count / n_tokens
    valence_vec[3] = 1.0 if valence_ratio > 0.3 else 0.0
    valence_vec[4] = 1.0 if valence_ratio < -0.3 else 0.0

    # --- Octant 3: Temporal (16D) ---
    causal_count = len(token_set & _CAUSAL_CONNECTORS)
    change_count = len(token_set & _CHANGE_WORDS)
    past_markers = sum(1 for t in tokens if t in ("was", "were", "had", "did", "previously"))
    future_markers = sum(1 for t in tokens if t in ("will", "shall", "would", "could", "might"))

    temporal_vec = np.zeros(16)
    temporal_vec[0] = causal_count / n_tokens
    temporal_vec[1] = change_count / n_tokens
    temporal_vec[2] = past_markers / n_tokens
    temporal_vec[3] = future_markers / n_tokens
    temporal_vec[4] = 1.0 if change_count > 2 else 0.0

    # --- Octant 4: Presence (16D) ---
    balance_count = len(token_set & _BALANCE_WORDS)
    question_marks = text.count("?")
    exclamation = text.count("!")

    presence_vec = np.zeros(16)
    presence_vec[0] = balance_count / n_tokens
    presence_vec[1] = question_marks / n_tokens
    presence_vec[2] = exclamation / n_tokens
    presence_vec[3] = len(tokens) / 500  # text density (normalized)

    # Compose full 64D vector
    full_vector = np.concatenate([agency_vec, valence_vec, temporal_vec, presence_vec])

    return {
        "agency": {
            "magnitude": float(np.linalg.norm(agency_vec)),
            "ratio": float(agency_ratio),
            "top_signals": (
                ["high_agency"] if agency_ratio > 0.6
                else ["passive_voice"] if passive_count > agency_count
                else ["balanced_voice"]
            ),
        },
        "valence": {
            "magnitude": float(np.linalg.norm(valence_vec)),
            "ratio": float(valence_ratio),
            "top_signals": (
                ["positive_dominant"] if valence_ratio > 0.3
                else ["negative_dominant"] if valence_ratio < -0.3
                else ["neutral_valence"]
            ),
        },
        "temporal": {
            "magnitude": float(np.linalg.norm(temporal_vec)),
            "change_intensity": float(change_count / n_tokens),
            "top_signals": (
                ["high_change"] if change_count > 2
                else ["causal_reasoning"] if causal_count > 1
                else ["low_temporal"]
            ),
        },
        "presence": {
            "magnitude": float(np.linalg.norm(presence_vec)),
            "balance_signal": float(balance_count / n_tokens),
            "top_signals": (
                ["balanced_system"] if balance_count > 1
                else ["questioning"] if question_marks > 1
                else ["standard_presence"]
            ),
        },
        "vector_64d": full_vector,
    }


# ---------------------------------------------------------------------------
# Health Assessment from Text Signals
# ---------------------------------------------------------------------------

def _assess_health_from_signals(
    geometry: Dict[str, Any],
    friction_count: int,
    max_severity_value: int,
) -> tuple:
    """Derive M(S) score and health status from combined signals.

    Uses the M(S) equation: M(S) = (R_e x A x D x C) - L

    Maps text signals to M(S) components:
    - R_e (Resonance): Agency magnitude (active engagement)
    - A (Adaptability): Temporal change handling
    - D (Diversity): Valence range (not one-sided)
    - C (Curiosity): Presence signals (questions, exploration)
    - L (Loss): Friction density + negative dominance

    Returns (m_s_score, interpretation, health_status, warnings).
    """
    agency = geometry.get("agency", {})
    valence = geometry.get("valence", {})
    temporal = geometry.get("temporal", {})
    presence = geometry.get("presence", {})

    # Map to M(S) components [0-1]
    r_e = min(agency.get("magnitude", 0.0) / 1.5, 1.0)
    adaptability = min(temporal.get("magnitude", 0.0) / 1.2, 1.0)

    # Diversity: higher when valence is mixed, not one-sided
    # Floor at 0.3 -- even strongly positive text has some diversity
    valence_ratio = valence.get("ratio", 0.0)
    diversity = max(1.0 - abs(valence_ratio), 0.3)

    curiosity = min(presence.get("magnitude", 0.0) / 1.0, 1.0)

    # Valence magnitude contributes to resonance and adaptability
    # (strong positive signal = the system is actively resonating)
    valence_mag = valence.get("magnitude", 0.0)
    if valence_ratio > 0.2:
        r_e = max(r_e, min(valence_mag / 1.5, 1.0))
        adaptability = max(adaptability, 0.5)
        curiosity = max(curiosity, 0.4)

    # Agency boosts adaptability (active systems are adaptive)
    agency_ratio_val = agency.get("ratio", 0.0)
    if agency_ratio_val > 0.5:
        adaptability = max(adaptability, 0.6)
        curiosity = max(curiosity, 0.4)

    # Combined positive: high agency + positive valence = coherent system
    if agency_ratio_val > 0.4 and valence_ratio > 0.2:
        r_e = max(r_e, 0.8)
        diversity = max(diversity, 0.5)

    # Loss: institutional friction + negative signals
    friction_loss = min(friction_count * 0.05, 0.5)
    severity_loss = min(max_severity_value * 0.1, 0.4)
    negative_loss = max(-valence_ratio, 0.0) * 0.3
    loss = friction_loss + severity_loss + negative_loss

    # Calculate M(S)
    metrics = CoreMetrics(
        resonance=max(r_e, 0.01),
        adaptability=max(adaptability, 0.01),
        diversity=max(diversity, 0.01),
        curiosity=max(curiosity, 0.01),
        loss=loss,
    )
    m_s_score = MSCalculator.calculate(metrics)
    interpretation = MSCalculator.interpret(m_s_score)

    # Derive health status
    warnings = []
    if m_s_score > 5:
        health = "THRIVING"
    elif m_s_score > 3:
        health = "HEALTHY"
    elif m_s_score > 1:
        health = "STRESSED"
        if friction_count > 5:
            warnings.append(
                f"High institutional friction ({friction_count} detections) "
                "degrading system coherence."
            )
    elif m_s_score > 0:
        health = "WARNING"
        warnings.append("System coherence approaching critical threshold.")
        if max_severity_value >= 4:
            warnings.append("CRITICAL severity friction detected in text.")
    else:
        health = "CRITICAL"
        warnings.append("Negative coherence. System in decline or collapse.")

    # Additional signal-based warnings
    if agency.get("ratio", 0.5) < 0.3:
        warnings.append(
            "Low agency detected: passive voice dominant. "
            "System may lack self-directed response capacity."
        )
    if temporal.get("change_intensity", 0.0) > 0.05:
        warnings.append(
            "High change intensity in text: rapid transitions "
            "or disruptions being described."
        )
    if valence_ratio < -0.5:
        warnings.append(
            "Strongly negative valence: text describes deteriorating "
            "conditions with little positive signal."
        )

    return m_s_score, interpretation, health, warnings


# ---------------------------------------------------------------------------
# Curiosity Signal Extraction
# ---------------------------------------------------------------------------

def _extract_curiosity_signals(
    geometry: Dict[str, Any],
    friction_alerts: List[Alert],
) -> List[str]:
    """Identify patterns worth investigating further (three-axis leads).

    These are the "unknown axis" signals -- things that don't fit
    neatly into existing categories but might reveal new principles.
    """
    signals = []

    # High agency + high friction = possible institutional capture
    agency_ratio = geometry.get("agency", {}).get("ratio", 0.5)
    if agency_ratio > 0.6 and len(friction_alerts) > 3:
        signals.append(
            "High agency language combined with institutional friction: "
            "possible narrative control or managed disclosure."
        )

    # Reclassification + downplay = Classification Gap widening
    cats = {a.category for a in friction_alerts}
    if Category.RECLASSIFICATION in cats and Category.DOWNPLAY in cats:
        signals.append(
            "Reclassification + downplay co-occurring: "
            "active Classification Gap. Compare raw vs. revised figures."
        )

    if Category.DATA_OPACITY in cats and Category.STATISTICAL_SMOOTHING in cats:
        signals.append(
            "Data opacity + statistical smoothing: "
            "signal suppression pipeline. Seek alternative data sources."
        )

    if Category.DEPENDENCY_RISK in cats and Category.LIABILITY_HEDGING in cats:
        signals.append(
            "Dependency risk + liability hedging: "
            "T_infra failure being legally distanced from impact. "
            "Track the causal chain independently."
        )

    # High temporal change with no friction = genuine event reporting
    change = geometry.get("temporal", {}).get("change_intensity", 0.0)
    if change > 0.03 and len(friction_alerts) == 0:
        signals.append(
            "High change intensity with no friction signals: "
            "likely genuine event reporting. Good data source."
        )

    # Low agency + high negative = system under external stress
    if agency_ratio < 0.3 and geometry.get("valence", {}).get("ratio", 0) < -0.3:
        signals.append(
            "Low agency + negative valence: system being acted upon "
            "without self-directed response. Monitor for cascade."
        )

    return signals


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class HGAI:
    """Holistic Geometric AI — unified analysis engine.

    Connects all framework modules into a single interface.
    Feed it text, get back a consolidated report covering system
    coherence, institutional friction, narrative geometry, and
    investigation leads.

    Parameters
    ----------
    felt_threshold : float
        FELT-Sensor threshold for model/reality dissonance.
    context_window : int
        Characters of context around friction matches.
    verbose : bool
        If True, reports include full details.

    Examples
    --------
    >>> engine = HGAI()
    >>> report = engine.analyze("The agency revised the methodology...")
    >>> print(report.render())
    """

    def __init__(
        self,
        felt_threshold: float = 1.5,
        context_window: int = 60,
        verbose: bool = False,
    ):
        self.verbose = verbose

        # Initialize scanner with all templates
        self.scanner = Scanner(context_window=context_window)
        self.scanner.add_templates(ALL_TEMPLATES)

        # Notice generator
        self.notice_gen = NoticeGenerator(id_prefix="HGAI")

        # FELT threshold for entropy sensor bridge
        self.felt_threshold = felt_threshold

    def analyze(
        self,
        text: str,
        target_entities: Optional[List[str]] = None,
    ) -> HGAIReport:
        """Run full analysis on input text.

        Parameters
        ----------
        text : str
            Any text: news article, institutional filing, field
            observation, report, conversation fragment.
        target_entities : list of str, optional
            Entities to address in formal notices.

        Returns
        -------
        HGAIReport
            Consolidated analysis across all branches.
        """
        if not text or not text.strip():
            return HGAIReport(
                text_analyzed="",
                health_status="NO INPUT",
                warnings=["No text provided for analysis."],
            )

        # Branch 1: Institutional friction scan
        scan_report = self.scanner.scan(text)
        risk_matrix = self.scanner.to_risk_matrix(text)
        stress_input = risk_matrix.to_stress_input()

        # Branch 2: Narrative geometry encoding
        geometry = _encode_text_geometry(text)

        # Branch 3: Generate notices if entities specified
        notices = []
        if target_entities and scan_report.alerts:
            notices = self.notice_gen.generate_batch(
                scan_report.alerts,
                target_entities=target_entities,
            )

        # Consolidate: M(S) score from combined signals
        max_sev = 0
        if scan_report.alerts:
            max_sev = max(a.severity.value for a in scan_report.alerts)

        m_s_score, interpretation, health, warnings = _assess_health_from_signals(
            geometry=geometry,
            friction_count=len(scan_report.alerts),
            max_severity_value=max_sev,
        )

        # Extract curiosity signals (three-axis leads)
        curiosity = _extract_curiosity_signals(geometry, scan_report.alerts)

        # Strip the raw 64D vector from display geometry
        display_geometry = {
            k: v for k, v in geometry.items() if k != "vector_64d"
        }

        return HGAIReport(
            text_analyzed=text,
            m_s_score=m_s_score,
            m_s_interpretation=interpretation,
            health_status=health,
            friction_alerts=scan_report.alerts,
            risk_matrix=risk_matrix,
            stress_input=stress_input,
            notices=notices,
            narrative_geometry=display_geometry,
            warnings=warnings,
            curiosity_signals=curiosity,
        )

    def quick(self, text: str) -> str:
        """One-line summary for fast mobile use.

        Parameters
        ----------
        text : str
            Text to analyze.

        Returns
        -------
        str
            Single-line status: M(S) score, health, alert count.
        """
        report = self.analyze(text)
        return (
            f"M(S)={report.m_s_score:.2f} | "
            f"{report.health_status} | "
            f"{len(report.friction_alerts)} friction alerts | "
            f"{len(report.curiosity_signals)} investigation leads"
        )

    def scan_only(self, text: str) -> str:
        """Run only the friction scanner (fastest mode).

        Parameters
        ----------
        text : str
            Text to scan.

        Returns
        -------
        str
            Alert summaries.
        """
        report = self.scanner.scan(text)
        if not report.alerts:
            return "No institutional friction detected."
        lines = [f"{report.alert_count} friction patterns detected:"]
        for alert in report.alerts:
            lines.append(f"  {alert.summary()}")
        return "\n".join(lines)

    def geometry_only(self, text: str) -> Dict[str, Any]:
        """Run only the narrative geometry encoder.

        Parameters
        ----------
        text : str
            Text to encode.

        Returns
        -------
        dict
            64D geometry with octant metadata.
        """
        return _encode_text_geometry(text)


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def _interactive_mode(engine: HGAI, verbose: bool = False):
    """Interactive REPL for phone-friendly use."""
    print("=" * 50)
    print("  HGAI — Holistic Geometric AI")
    print("  Paste text and press Enter twice to analyze.")
    print("  Commands: 'quit', 'quick', 'scan', 'help'")
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
            print("Done.")
            break
        if text.lower() == "help":
            print(textwrap.dedent("""
                Commands:
                  (paste text)  Full analysis
                  quick         Switch to one-line output mode
                  scan          Switch to friction-scan-only mode
                  full          Switch to full report mode
                  verbose       Toggle verbose details
                  quit          Exit
            """))
            continue
        if text.lower() == "quick":
            print("Switched to quick mode. Paste text:")
            continue
        if text.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose: {'ON' if verbose else 'OFF'}")
            continue

        # Analyze
        report = engine.analyze(text)
        print(report.render(verbose=verbose))


def main():
    """CLI entry point."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    quick_mode = "--quick" in sys.argv or "-q" in sys.argv

    # Clean args
    text_args = [
        a for a in sys.argv[1:]
        if a not in ("--verbose", "-v", "--quick", "-q")
    ]

    engine = HGAI(verbose=verbose)

    if text_args:
        # Direct mode: text passed as argument
        text = " ".join(text_args)
        if quick_mode:
            print(engine.quick(text))
        else:
            report = engine.analyze(text)
            print(report.render(verbose=verbose))
    else:
        # Interactive mode
        _interactive_mode(engine, verbose=verbose)


if __name__ == "__main__":
    main()
