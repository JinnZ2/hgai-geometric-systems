"""
Resilience Detectors — Pattern-Based Institutional Friction Scanner

Scans text for language patterns that signal institutional friction,
risk reclassification, statistical smoothing, and liability hedging.
These are the leading indicators of the Classification Gap: the
discrepancy between reported and actual system stress.

Design Philosophy:
    - Regex-based template matching (no NLP dependencies)
    - Offline-first, minimal dependencies (re, dataclasses, numpy)
    - Templates detect the *language* of entropy denial before it
      shows up in the numeric data
    - Output bridges directly into EntropySensor.calibrate_signals()

Classes:
    Template: A named regex pattern with severity and category metadata.
    Alert: A single detection event from scanning text.
    ScanReport: Collection of alerts from a full scan pass.
    RiskMatrix: Aggregated risk scores by category, bridgeable to stress input.
    Scanner: The main scanning engine that applies templates to text.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(Enum):
    """Alert severity levels mapped to entropy magnitude."""
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


class Category(Enum):
    """Institutional friction categories.

    Each maps to a distinct failure mode in the Classification Gap.
    """
    RECLASSIFICATION = "reclassification"
    LIABILITY_HEDGING = "liability_hedging"
    STATISTICAL_SMOOTHING = "statistical_smoothing"
    DOWNPLAY = "downplay"
    DATA_OPACITY = "data_opacity"
    DEPENDENCY_RISK = "dependency_risk"
    COMMUNICATION_FRICTION = "communication_friction"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Template:
    """A named regex pattern for detecting institutional friction.

    Parameters
    ----------
    name : str
        Human-readable name for the pattern.
    pattern : str
        Regex pattern (case-insensitive matching applied at scan time).
    category : Category
        Which friction category this template detects.
    severity : Severity
        Default severity when this pattern matches.
    description : str
        Explanation of what this pattern signals.
    """
    name: str
    pattern: str
    category: Category
    severity: Severity
    description: str

    def __post_init__(self):
        # Pre-compile for performance
        self._compiled = re.compile(self.pattern, re.IGNORECASE)

    @property
    def compiled(self) -> re.Pattern:
        return self._compiled


@dataclass
class Alert:
    """A single detection event from scanning text.

    Parameters
    ----------
    template_name : str
        Name of the template that triggered.
    category : Category
        Friction category.
    severity : Severity
        Severity level.
    matched_text : str
        The actual text fragment that matched.
    context : str
        Surrounding text for human review.
    position : int
        Character offset in the source text.
    description : str
        What this detection signals.
    """
    template_name: str
    category: Category
    severity: Severity
    matched_text: str
    context: str
    position: int
    description: str

    def summary(self) -> str:
        """One-line summary for console output."""
        return (
            f"[{self.severity.name}] {self.category.value}: "
            f"{self.template_name} -- \"{self.matched_text[:60]}\" "
            f"(pos {self.position})"
        )


@dataclass
class ScanReport:
    """Collection of alerts from a full scan pass.

    Parameters
    ----------
    source_length : int
        Length of the scanned text.
    alerts : list of Alert
        All detections found.
    """
    source_length: int
    alerts: List[Alert] = field(default_factory=list)

    @property
    def alert_count(self) -> int:
        return len(self.alerts)

    @property
    def max_severity(self) -> Optional[Severity]:
        """Highest severity detected, or None if no alerts."""
        if not self.alerts:
            return None
        return max(self.alerts, key=lambda a: a.severity.value).severity

    def by_category(self) -> Dict[Category, List[Alert]]:
        """Group alerts by friction category."""
        grouped: Dict[Category, List[Alert]] = {}
        for alert in self.alerts:
            grouped.setdefault(alert.category, []).append(alert)
        return grouped

    def by_severity(self) -> Dict[Severity, List[Alert]]:
        """Group alerts by severity level."""
        grouped: Dict[Severity, List[Alert]] = {}
        for alert in self.alerts:
            grouped.setdefault(alert.severity, []).append(alert)
        return grouped

    def summary(self) -> str:
        """Multi-line summary of the scan."""
        lines = [
            f"Scan Report: {self.alert_count} alerts "
            f"across {self.source_length} chars"
        ]
        if self.max_severity:
            lines.append(f"Max severity: {self.max_severity.name}")
        for cat, alerts in self.by_category().items():
            lines.append(f"  {cat.value}: {len(alerts)} alerts")
        return "\n".join(lines)


@dataclass
class StressInput:
    """Bridge format for feeding into EntropySensor.calibrate_signals().

    Maps scanner output to the numeric vectors the SIS model expects.

    Parameters
    ----------
    institutional_friction : float
        Aggregate friction score (maps to communication/reporting noise).
    dependency_signal : float
        T_infra dependency risk detected in text.
    classification_gap : float
        Reclassification and smoothing signal strength.
    opacity_index : float
        Data opacity / access restriction signal.
    """
    institutional_friction: float
    dependency_signal: float
    classification_gap: float
    opacity_index: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "institutional_friction": self.institutional_friction,
            "dependency_signal": self.dependency_signal,
            "classification_gap": self.classification_gap,
            "opacity_index": self.opacity_index,
        }


@dataclass
class RiskMatrix:
    """Aggregated risk scores by category.

    Converts qualitative alert patterns into quantitative risk vectors
    suitable for the entropy sensor pipeline.

    Parameters
    ----------
    category_scores : dict
        Mapping of Category -> aggregate severity score.
    total_alerts : int
        Total number of alerts contributing.
    text_length : int
        Length of source text (for density normalization).
    """
    category_scores: Dict[Category, float]
    total_alerts: int
    text_length: int

    @property
    def density(self) -> float:
        """Alert density: alerts per 1000 characters."""
        if self.text_length == 0:
            return 0.0
        return (self.total_alerts / self.text_length) * 1000

    @property
    def total_risk(self) -> float:
        """Sum of all category scores."""
        return sum(self.category_scores.values())

    def to_vector(self) -> np.ndarray:
        """Convert to ordered numpy vector (one element per category)."""
        return np.array([
            self.category_scores.get(cat, 0.0)
            for cat in Category
        ])

    def to_stress_input(self) -> StressInput:
        """Bridge into EntropySensor-compatible stress input.

        Maps category scores to the four stress dimensions:
        - institutional_friction: hedging + communication friction
        - dependency_signal: dependency risk score
        - classification_gap: reclassification + smoothing
        - opacity_index: data opacity score
        """
        scores = self.category_scores

        institutional_friction = (
            scores.get(Category.LIABILITY_HEDGING, 0.0)
            + scores.get(Category.COMMUNICATION_FRICTION, 0.0)
        )
        dependency_signal = scores.get(Category.DEPENDENCY_RISK, 0.0)
        classification_gap = (
            scores.get(Category.RECLASSIFICATION, 0.0)
            + scores.get(Category.STATISTICAL_SMOOTHING, 0.0)
        )
        opacity_index = (
            scores.get(Category.DATA_OPACITY, 0.0)
            + scores.get(Category.DOWNPLAY, 0.0)
        )

        return StressInput(
            institutional_friction=institutional_friction,
            dependency_signal=dependency_signal,
            classification_gap=classification_gap,
            opacity_index=opacity_index,
        )


# ---------------------------------------------------------------------------
# Template Library
# ---------------------------------------------------------------------------

# --- Reclassification patterns ---
_RECLASSIFICATION_TEMPLATES = [
    Template(
        name="methodology_revision",
        pattern=r"revised\s+(?:the\s+)?methodology",
        category=Category.RECLASSIFICATION,
        severity=Severity.HIGH,
        description=(
            "Methodology revision signals retroactive reclassification "
            "of events to reduce perceived severity."
        ),
    ),
    Template(
        name="reclassified_cause",
        pattern=r"(?:reclassified|recoded|recategorized)\s+(?:as|to|from)",
        category=Category.RECLASSIFICATION,
        severity=Severity.HIGH,
        description=(
            "Explicit reclassification of cause of death or event type. "
            "Core Classification Gap indicator."
        ),
    ),
    Template(
        name="updated_criteria",
        pattern=r"updated\s+(?:the\s+)?(?:criteria|definition|threshold|classification)",
        category=Category.RECLASSIFICATION,
        severity=Severity.MODERATE,
        description="Criteria changes that shift what counts as a reportable event.",
    ),
    Template(
        name="retroactive_adjustment",
        pattern=r"retroactiv(?:e|ely)\s+(?:adjust|revis|correct|modif)",
        category=Category.RECLASSIFICATION,
        severity=Severity.CRITICAL,
        description=(
            "Retroactive data adjustment -- rewriting history to reduce "
            "apparent impact."
        ),
    ),
]

# --- Liability hedging patterns ---
_LIABILITY_TEMPLATES = [
    Template(
        name="not_directly_attributable",
        pattern=r"not\s+(?:directly\s+)?(?:attributable|linked|related)\s+to",
        category=Category.LIABILITY_HEDGING,
        severity=Severity.HIGH,
        description=(
            "Causal distancing language. Breaks the link between event "
            "and impact to reduce institutional liability."
        ),
    ),
    Template(
        name="cannot_be_determined",
        pattern=r"(?:cannot|could\s+not|unable\s+to)\s+(?:be\s+)?(?:determined|established|confirmed)",
        category=Category.LIABILITY_HEDGING,
        severity=Severity.MODERATE,
        description="Epistemic hedging to avoid causal attribution.",
    ),
    Template(
        name="within_normal_range",
        pattern=r"(?:within|consistent\s+with)\s+(?:normal|expected|typical)\s+(?:range|variation|limits)",
        category=Category.LIABILITY_HEDGING,
        severity=Severity.MODERATE,
        description=(
            "Normalizing language that frames anomalies as expected "
            "variation."
        ),
    ),
    Template(
        name="no_evidence_of",
        pattern=r"no\s+(?:clear\s+|direct\s+|definitive\s+)?evidence\s+(?:of|that|linking)",
        category=Category.LIABILITY_HEDGING,
        severity=Severity.HIGH,
        description=(
            "Absence-of-evidence framing used to deny connection "
            "rather than investigate."
        ),
    ),
]

# --- Statistical smoothing patterns ---
_SMOOTHING_TEMPLATES = [
    Template(
        name="seasonal_adjustment",
        pattern=r"(?:adjusted|corrected)\s+for\s+(?:seasonal|cyclical|periodic)\s+(?:variation|effects|factors)",
        category=Category.STATISTICAL_SMOOTHING,
        severity=Severity.MODERATE,
        description=(
            "Seasonal adjustment that may smooth away real anomalies."
        ),
    ),
    Template(
        name="rolling_average",
        pattern=r"(?:rolling|moving|smoothed)\s+(?:\d+[- ](?:day|week|month|year)\s+)?average",
        category=Category.STATISTICAL_SMOOTHING,
        severity=Severity.LOW,
        description="Rolling averages that dampen spike visibility.",
    ),
    Template(
        name="outlier_removal",
        pattern=r"(?:outliers?\s+(?:were\s+)?(?:removed|excluded|filtered))|(?:(?:removed|excluded|filtered)\s+outliers?)",
        category=Category.STATISTICAL_SMOOTHING,
        severity=Severity.HIGH,
        description=(
            "Outlier removal -- the 'noise' being filtered is often "
            "the most critical signal."
        ),
    ),
    Template(
        name="baseline_normalization",
        pattern=r"(?:normalized|adjusted)\s+(?:to|against|relative\s+to)\s+(?:the\s+)?(?:baseline|historical|reference)",
        category=Category.STATISTICAL_SMOOTHING,
        severity=Severity.MODERATE,
        description=(
            "Baseline normalization that anchors to potentially "
            "outdated reference periods."
        ),
    ),
]

# --- Downplay patterns ---
_DOWNPLAY_TEMPLATES = [
    Template(
        name="minimal_impact",
        pattern=r"(?:minimal|negligible|limited|marginal)\s+(?:impact|effect|consequence|risk)",
        category=Category.DOWNPLAY,
        severity=Severity.MODERATE,
        description="Minimization language that understates real impact.",
    ),
    Template(
        name="isolated_incident",
        pattern=r"(?:isolated|rare|unusual|atypical|one-off)\s+(?:incident|event|occurrence|case)",
        category=Category.DOWNPLAY,
        severity=Severity.HIGH,
        description=(
            "Framing systemic events as isolated to prevent pattern "
            "recognition."
        ),
    ),
    Template(
        name="no_cause_for_concern",
        pattern=r"no\s+(?:immediate\s+)?(?:cause\s+for\s+)?(?:concern|alarm|worry)",
        category=Category.DOWNPLAY,
        severity=Severity.MODERATE,
        description="Reassurance language that discourages further inquiry.",
    ),
]

# --- Data opacity patterns ---
_OPACITY_TEMPLATES = [
    Template(
        name="data_unavailable",
        pattern=r"data\s+(?:is\s+)?(?:not\s+(?:yet\s+)?available|unavailable|pending|under\s+review)",
        category=Category.DATA_OPACITY,
        severity=Severity.HIGH,
        description="Data access restriction or delayed release.",
    ),
    Template(
        name="proprietary_methodology",
        pattern=r"(?:proprietary|confidential|restricted)\s+(?:methodology|data|model|algorithm)",
        category=Category.DATA_OPACITY,
        severity=Severity.CRITICAL,
        description=(
            "Proprietary claims that prevent independent verification "
            "and reproduce institutional silos."
        ),
    ),
    Template(
        name="aggregated_reporting",
        pattern=r"(?:reported|presented|published)\s+(?:in\s+)?(?:aggregate|summary|consolidated)\s+form",
        category=Category.DATA_OPACITY,
        severity=Severity.MODERATE,
        description=(
            "Aggregation that hides granular patterns and prevents "
            "localized analysis."
        ),
    ),
]

# --- Dependency risk patterns ---
_DEPENDENCY_TEMPLATES = [
    Template(
        name="system_outage",
        pattern=r"(?:system|service|infrastructure|network)\s+(?:outage|failure|disruption|downtime)",
        category=Category.DEPENDENCY_RISK,
        severity=Severity.HIGH,
        description="Infrastructure dependency failure -- T_infra stress event.",
    ),
    Template(
        name="single_point_failure",
        pattern=r"(?:single\s+point\s+of\s+failure|critical\s+dependency|sole\s+(?:source|provider))",
        category=Category.DEPENDENCY_RISK,
        severity=Severity.CRITICAL,
        description="Single point of failure in critical infrastructure.",
    ),
    Template(
        name="vendor_dependency",
        pattern=r"(?:vendor|supplier|third[- ]party)\s+(?:dependency|reliance|lock[- ]?in)",
        category=Category.DEPENDENCY_RISK,
        severity=Severity.MODERATE,
        description="Vendor lock-in or supply chain dependency risk.",
    ),
]

# --- Communication friction patterns ---
_COMMUNICATION_TEMPLATES = [
    Template(
        name="delayed_notification",
        pattern=r"(?:delayed|late|untimely)\s+(?:notification|disclosure|reporting|announcement)",
        category=Category.COMMUNICATION_FRICTION,
        severity=Severity.HIGH,
        description="Communication delay that prevents timely response.",
    ),
    Template(
        name="conflicting_reports",
        pattern=r"(?:conflicting|contradictory|inconsistent)\s+(?:reports?|data|information|findings)",
        category=Category.COMMUNICATION_FRICTION,
        severity=Severity.HIGH,
        description=(
            "Conflicting information across sources -- institutional "
            "silo friction."
        ),
    ),
    Template(
        name="preliminary_estimate",
        pattern=r"(?:preliminary|provisional|tentative|draft)\s+(?:estimate|figure|assessment|report)",
        category=Category.COMMUNICATION_FRICTION,
        severity=Severity.LOW,
        description=(
            "Preliminary framing that delays definitive information "
            "and allows later revision."
        ),
    ),
]


# All templates combined
ALL_TEMPLATES: List[Template] = (
    _RECLASSIFICATION_TEMPLATES
    + _LIABILITY_TEMPLATES
    + _SMOOTHING_TEMPLATES
    + _DOWNPLAY_TEMPLATES
    + _OPACITY_TEMPLATES
    + _DEPENDENCY_TEMPLATES
    + _COMMUNICATION_TEMPLATES
)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class Scanner:
    """Pattern-based institutional friction scanner.

    Applies regex templates to text to detect language patterns that
    signal institutional friction, risk reclassification, and the
    Classification Gap.

    Parameters
    ----------
    context_window : int
        Number of characters to include before and after a match
        for human-readable context. Default 80.
    """

    def __init__(self, context_window: int = 80):
        self.templates: List[Template] = []
        self.context_window = context_window

    def add_templates(self, templates: List[Template]):
        """Add detection templates to the scanner.

        Parameters
        ----------
        templates : list of Template
            Templates to register for scanning.
        """
        self.templates.extend(templates)

    def scan(self, text: str) -> ScanReport:
        """Scan text for institutional friction patterns.

        Parameters
        ----------
        text : str
            Raw text from a filing, news article, or report.

        Returns
        -------
        ScanReport
            All detected alerts with matched text and context.
        """
        alerts: List[Alert] = []

        for template in self.templates:
            for match in template.compiled.finditer(text):
                start = max(0, match.start() - self.context_window)
                end = min(len(text), match.end() + self.context_window)
                context = text[start:end].replace("\n", " ").strip()

                alerts.append(Alert(
                    template_name=template.name,
                    category=template.category,
                    severity=template.severity,
                    matched_text=match.group(),
                    context=context,
                    position=match.start(),
                    description=template.description,
                ))

        # Sort by position in text
        alerts.sort(key=lambda a: a.position)

        return ScanReport(source_length=len(text), alerts=alerts)

    def to_risk_matrix(self, text: str) -> RiskMatrix:
        """Scan text and aggregate into a risk matrix.

        Each category's score is the sum of severity values for all
        alerts in that category, normalized by text length (per 1000
        characters) to allow cross-document comparison.

        Parameters
        ----------
        text : str
            Raw text to scan.

        Returns
        -------
        RiskMatrix
            Aggregated risk scores ready for stress input conversion.
        """
        report = self.scan(text)
        text_len = max(len(text), 1)

        category_scores: Dict[Category, float] = {}
        for alert in report.alerts:
            score = alert.severity.value
            category_scores[alert.category] = (
                category_scores.get(alert.category, 0.0) + score
            )

        # Normalize per 1000 chars for cross-document comparison
        for cat in category_scores:
            category_scores[cat] = (category_scores[cat] / text_len) * 1000

        return RiskMatrix(
            category_scores=category_scores,
            total_alerts=report.alert_count,
            text_length=len(text),
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_scanner():
    """Demonstrate the scanner on a synthetic institutional filing."""
    print("=" * 60)
    print("Resilience Scanner -- Institutional Friction Detection")
    print("=" * 60)

    # Synthetic text mimicking an institutional report
    sample_text = """
    Following the extreme weather event of January 15th, the regional
    health authority revised the methodology for classifying weather-
    related fatalities. Deaths previously attributed to hypothermia
    were reclassified as cardiac events, consistent with the updated
    criteria for cause-of-death reporting.

    The agency noted that the mortality figures were adjusted for
    seasonal variation and that outliers were removed from the final
    dataset. A rolling 7-day average was applied to smooth reporting
    fluctuations. The resulting figures show minimal impact on the
    regional mortality baseline.

    Officials stated there is no direct evidence linking the
    infrastructure outage to the reported fatalities, and that the
    system outage was an isolated incident not reflective of broader
    dependency risks. Data is not yet available for independent
    review, as the proprietary methodology used by the contractor
    remains under evaluation.

    A preliminary estimate of excess deaths was issued with delayed
    notification to affected communities. Conflicting reports from
    emergency services and the health authority have not yet been
    reconciled. The findings were reported in aggregate form, and
    officials emphasized there is no cause for concern at this time.
    """

    scanner = Scanner(context_window=60)
    scanner.add_templates(ALL_TEMPLATES)

    # Full scan
    report = scanner.scan(sample_text)
    print(f"\n{report.summary()}\n")

    print("--- Alerts ---")
    for alert in report.alerts:
        print(f"  {alert.summary()}")

    # Risk matrix
    matrix = scanner.to_risk_matrix(sample_text)
    print(f"\n--- Risk Matrix ---")
    print(f"Alert density: {matrix.density:.2f} per 1000 chars")
    print(f"Total risk score: {matrix.total_risk:.2f}")
    for cat in Category:
        score = matrix.category_scores.get(cat, 0.0)
        if score > 0:
            bar = "#" * int(score * 5)
            print(f"  {cat.value:30s}: {score:.3f} {bar}")

    # Bridge to stress input
    stress = matrix.to_stress_input()
    print(f"\n--- Stress Input (EntropySensor Bridge) ---")
    for key, val in stress.to_dict().items():
        print(f"  {key:30s}: {val:.4f}")

    print(f"\n  Risk vector: {matrix.to_vector()}")


if __name__ == "__main__":
    demo_scanner()
