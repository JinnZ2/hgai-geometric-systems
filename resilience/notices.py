"""
Resilience Notices — Formal Alert Output Generator

Converts scanner alerts into structured formal notices suitable for
documentation, reporting, or submission to entities. Each notice
includes the detection evidence, risk assessment, and recommended
action framing.

Design:
    - No external dependencies beyond the detectors module
    - Generates plain-text notices (no HTML/PDF dependencies)
    - Batch generation for multiple alerts targeting specific entities
    - Timestamped and traceable output
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from resilience.detectors import Alert, Category, RiskMatrix, Severity


@dataclass
class Notice:
    """A formal notice generated from one or more related alerts.

    Parameters
    ----------
    notice_id : str
        Unique identifier for this notice.
    target_entity : str
        Organization or entity the notice addresses.
    category : Category
        Primary friction category.
    severity : Severity
        Highest severity among contributing alerts.
    title : str
        Short title summarizing the notice.
    body : str
        Full notice text with evidence and framing.
    alerts : list of Alert
        Contributing alerts.
    timestamp : str
        ISO-format generation timestamp.
    """
    notice_id: str
    target_entity: str
    category: Category
    severity: Severity
    title: str
    body: str
    alerts: List[Alert] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def render(self) -> str:
        """Render the notice as formatted plain text."""
        sep = "-" * 60
        lines = [
            sep,
            f"NOTICE: {self.notice_id}",
            f"TO:     {self.target_entity}",
            f"DATE:   {self.timestamp}",
            f"LEVEL:  {self.severity.name}",
            f"TYPE:   {self.category.value}",
            sep,
            "",
            self.title,
            "=" * len(self.title),
            "",
            self.body,
            "",
            "EVIDENCE:",
        ]
        for i, alert in enumerate(self.alerts, 1):
            lines.append(f"  [{i}] {alert.template_name}")
            lines.append(f"      Match: \"{alert.matched_text}\"")
            lines.append(f"      Context: \"{alert.context[:120]}\"")
            lines.append(f"      Signal: {alert.description}")
            lines.append("")
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Category-specific framing
# ---------------------------------------------------------------------------

_CATEGORY_FRAMING: Dict[Category, Dict[str, str]] = {
    Category.RECLASSIFICATION: {
        "title": "Classification Integrity Alert",
        "opening": (
            "Pattern analysis has detected language consistent with "
            "retroactive reclassification of events or outcomes. "
            "This practice can obscure the true impact of system "
            "events and widen the Classification Gap."
        ),
        "action": (
            "Recommendation: Preserve original classification data "
            "alongside any revised figures. Publish methodology "
            "changes with version tracking and justification."
        ),
    },
    Category.LIABILITY_HEDGING: {
        "title": "Liability Hedging Detection",
        "opening": (
            "Language patterns consistent with causal distancing and "
            "liability hedging have been identified. These patterns "
            "break the attribution chain between events and their "
            "documented impact."
        ),
        "action": (
            "Recommendation: Provide explicit causal analysis rather "
            "than absence-of-evidence framing. Document investigation "
            "methodology and findings transparently."
        ),
    },
    Category.STATISTICAL_SMOOTHING: {
        "title": "Statistical Smoothing Advisory",
        "opening": (
            "Detected statistical processing methods that may suppress "
            "critical signal variance. Outlier removal, rolling averages, "
            "and seasonal adjustments can filter the very data points "
            "that represent high-entropy system events."
        ),
        "action": (
            "Recommendation: Report raw data alongside smoothed figures. "
            "Flag removed outliers separately rather than discarding them. "
            "Publish confidence intervals, not point estimates."
        ),
    },
    Category.DOWNPLAY: {
        "title": "Impact Minimization Alert",
        "opening": (
            "Language patterns that minimize or normalize system impact "
            "have been detected. Framing systemic events as 'isolated' "
            "or 'minimal' prevents pattern recognition across events "
            "and suppresses cumulative risk assessment."
        ),
        "action": (
            "Recommendation: Contextualize events within historical "
            "frequency and cumulative impact. Avoid normalizing language "
            "until independent analysis confirms baseline consistency."
        ),
    },
    Category.DATA_OPACITY: {
        "title": "Data Transparency Concern",
        "opening": (
            "Indicators of restricted data access, proprietary "
            "methodology claims, or aggregation-based opacity have "
            "been detected. These patterns prevent independent "
            "verification and perpetuate institutional data silos."
        ),
        "action": (
            "Recommendation: Release granular data for independent "
            "analysis. Publish methodology specifications sufficient "
            "for reproduction. Adopt open data standards."
        ),
    },
    Category.DEPENDENCY_RISK: {
        "title": "Infrastructure Dependency Warning",
        "opening": (
            "Signals of infrastructure dependency risk have been "
            "identified, including system outages, single points of "
            "failure, or vendor lock-in. These represent T_infra "
            "stress vectors that amplify system entropy during events."
        ),
        "action": (
            "Recommendation: Conduct dependency mapping and failure "
            "mode analysis. Establish redundancy for critical single "
            "points of failure. Diversify vendor relationships."
        ),
    },
    Category.COMMUNICATION_FRICTION: {
        "title": "Communication Friction Alert",
        "opening": (
            "Patterns of delayed notification, conflicting reports, "
            "or preliminary framing have been detected. Communication "
            "friction increases the lag between event occurrence and "
            "public awareness, reducing response effectiveness."
        ),
        "action": (
            "Recommendation: Establish real-time reporting pipelines "
            "with cross-source reconciliation. Replace preliminary "
            "framing with uncertainty-quantified rapid assessments."
        ),
    },
}


# ---------------------------------------------------------------------------
# Notice Generator
# ---------------------------------------------------------------------------

class NoticeGenerator:
    """Generates formal notices from scanner alerts.

    Groups alerts by category, applies category-specific framing,
    and produces traceable, timestamped notices addressed to
    specified entities.

    Parameters
    ----------
    id_prefix : str
        Prefix for generated notice IDs. Default "SIS".
    """

    def __init__(self, id_prefix: str = "SIS"):
        self.id_prefix = id_prefix
        self._counter = 0

    def _next_id(self) -> str:
        """Generate a sequential notice ID."""
        self._counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{self.id_prefix}-{ts}-{self._counter:04d}"

    def generate(
        self,
        alert: Alert,
        target_entity: str,
        custom_framing: Optional[str] = None,
    ) -> Notice:
        """Generate a single notice from one alert.

        Parameters
        ----------
        alert : Alert
            The alert to convert into a notice.
        target_entity : str
            Entity the notice is addressed to.
        custom_framing : str, optional
            Override the default category framing text.

        Returns
        -------
        Notice
            Formatted notice with evidence and recommendations.
        """
        framing = _CATEGORY_FRAMING.get(alert.category, {})
        title = framing.get("title", f"{alert.category.value} Alert")
        opening = custom_framing or framing.get("opening", alert.description)
        action = framing.get("action", "")

        body = f"{opening}\n\n{action}"

        return Notice(
            notice_id=self._next_id(),
            target_entity=target_entity,
            category=alert.category,
            severity=alert.severity,
            title=title,
            body=body,
            alerts=[alert],
        )

    def generate_batch(
        self,
        alerts: List[Alert],
        target_entities: Optional[List[str]] = None,
    ) -> List[Notice]:
        """Generate notices from a list of alerts, grouped by category.

        Alerts in the same category are combined into a single notice
        for each target entity. Each notice carries all contributing
        evidence.

        Parameters
        ----------
        alerts : list of Alert
            Alerts to convert (typically from ScanReport.alerts).
        target_entities : list of str, optional
            Entities to address. If None, defaults to ["Unspecified"].

        Returns
        -------
        list of Notice
            One notice per (category, entity) combination.
        """
        if not target_entities:
            target_entities = ["Unspecified"]

        # Group alerts by category
        by_category: Dict[Category, List[Alert]] = {}
        for alert in alerts:
            by_category.setdefault(alert.category, []).append(alert)

        notices: List[Notice] = []

        for entity in target_entities:
            for category, cat_alerts in by_category.items():
                framing = _CATEGORY_FRAMING.get(category, {})
                title = framing.get("title", f"{category.value} Alert")
                opening = framing.get("opening", "")
                action = framing.get("action", "")
                body = f"{opening}\n\n{action}"

                max_sev = max(cat_alerts, key=lambda a: a.severity.value)

                notice = Notice(
                    notice_id=self._next_id(),
                    target_entity=entity,
                    category=category,
                    severity=max_sev.severity,
                    title=title,
                    body=body,
                    alerts=cat_alerts,
                )
                notices.append(notice)

        return notices

    def generate_summary_report(
        self,
        notices: List[Notice],
        matrix: Optional[RiskMatrix] = None,
    ) -> str:
        """Generate a summary report across all notices.

        Parameters
        ----------
        notices : list of Notice
            Notices to summarize.
        matrix : RiskMatrix, optional
            If provided, includes stress input values in the summary.

        Returns
        -------
        str
            Formatted summary report text.
        """
        lines = [
            "=" * 60,
            "SOVEREIGN IMPACT SENSOR -- RESILIENCE REPORT",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 60,
            "",
            f"Total notices: {len(notices)}",
        ]

        # Severity breakdown
        sev_counts: Dict[str, int] = {}
        for notice in notices:
            sev_counts[notice.severity.name] = (
                sev_counts.get(notice.severity.name, 0) + 1
            )
        for sev_name in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            if sev_name in sev_counts:
                lines.append(f"  {sev_name}: {sev_counts[sev_name]}")

        # Entity breakdown
        entities = set(n.target_entity for n in notices)
        lines.append(f"\nEntities addressed: {', '.join(sorted(entities))}")

        # Stress input bridge
        if matrix:
            stress = matrix.to_stress_input()
            lines.append("\n--- Stress Input (EntropySensor Bridge) ---")
            for key, val in stress.to_dict().items():
                lines.append(f"  {key:30s}: {val:.4f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def demo_notices():
    """Demonstrate notice generation from scanner output."""
    from resilience.detectors import Scanner, ALL_TEMPLATES

    print("=" * 60)
    print("Notice Generator -- Formal Alert Output")
    print("=" * 60)

    sample_text = """
    The regional authority revised the methodology for mortality
    classification. Deaths were reclassified as cardiac events.
    Officials stated this was an isolated incident with minimal
    impact. The system outage affecting emergency dispatch was
    not directly attributable to the weather event. Data is not
    yet available for public review due to the proprietary
    methodology of the analytics contractor. A preliminary
    estimate was issued with delayed notification to communities.
    Outliers were removed from the final dataset, and figures
    were adjusted for seasonal variation.
    """

    scanner = Scanner()
    scanner.add_templates(ALL_TEMPLATES)
    report = scanner.scan(sample_text)
    matrix = scanner.to_risk_matrix(sample_text)

    gen = NoticeGenerator()
    notices = gen.generate_batch(
        report.alerts,
        target_entities=["Regional Health Authority", "Analytics Contractor"],
    )

    for notice in notices:
        print(notice.render())

    # Summary
    summary = gen.generate_summary_report(notices, matrix=matrix)
    print(summary)


if __name__ == "__main__":
    demo_notices()
