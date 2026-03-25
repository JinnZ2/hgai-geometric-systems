#!/usr/bin/env python3
"""
UNIFIED FIELD MONITOR
Integrates geometric consciousness tracking with relational ecological health

Combines:

- Geometric narrative coherence (your 64D framework)
- Relational pattern detection (sudden changes, balance)
- M(S) viability calculation
- Curvature & attractor dynamics
- Indigenous observation frameworks

Runs offline on phone/laptop for field work
No dependencies except numpy

Created by: JinnZ2 + Claude
License: MIT (belongs to the commons)
Purpose: Track system health (ecological, social, consciousness) in real-time
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import math
import time

# ═══════════════════════════════════════════════════════════════════════════

# GEOMETRIC ENCODERS (from your framework)

# ═══════════════════════════════════════════════════════════════════════════

def safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v) + 1e-12)

def normalize(v: np.ndarray) -> np.ndarray:
    return v / safe_norm(v)

# Simplified encoder for field use

def field_observation_to_geometry(observation: str) -> np.ndarray:
    """
    Quick encoder for field observations
    Focus on: change_rate, balance, relationships, warnings
    """
    obs_lower = observation.lower()
    tokens = obs_lower.split()
    n = max(1, len(tokens))

    vec = np.zeros(64, dtype=float)

    # [0-15] CHANGE RATE signals
    sudden_words = ["sudden", "rapid", "quick", "fast", "spike", "crash", "die-off", "dieoff"]
    vec[0] = sum(1 for w in sudden_words if w in obs_lower) / n

    gradual_words = ["slow", "gradual", "steady", "stable"]
    vec[1] = sum(1 for w in gradual_words if w in obs_lower) / n

    # [16-31] BALANCE indicators
    balance_words = ["balance", "balanced", "harmony", "stable", "adapt", "adapting"]
    vec[16] = sum(1 for w in balance_words if w in obs_lower) / n

    imbalance_words = ["imbalance", "unbalanced", "disrupted", "broken", "severed"]
    vec[17] = sum(1 for w in imbalance_words if w in obs_lower) / n

    # [32-47] RELATIONSHIP health
    coupling_words = ["connect", "connected", "coupled", "relationship", "mutualism"]
    vec[32] = sum(1 for w in coupling_words if w in obs_lower) / n

    breaking_words = ["disconnect", "isolated", "separated", "severed", "broken"]
    vec[33] = sum(1 for w in breaking_words if w in obs_lower) / n

    # [48-63] PRESENCE (what's observed)
    species_words = ["fish", "tree", "plant", "animal", "bird", "insect", "fungi"]
    vec[48] = sum(1 for w in species_words if w in obs_lower) / n

    degradation_words = ["degradation", "decline", "dying", "dead", "sick", "ill"]
    vec[49] = sum(1 for w in degradation_words if w in obs_lower) / n

    # Temperature/water mentions
    vec[50] = float("temperature" in obs_lower or "temp" in obs_lower or "\u00b0" in observation)
    vec[51] = float("water" in obs_lower or "river" in obs_lower or "flow" in obs_lower)

    # Normalize
    if np.allclose(vec, 0.0):
        vec[0] = 0.1  # small default

    return normalize(vec)

# ═══════════════════════════════════════════════════════════════════════════

# RELATIONAL PATTERNS

# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FieldObservation:
    """Quick field observation structure"""

    # What you observe
    observation_text: str

    # Quick checks (yes/no/unknown)
    sudden_change: bool = False
    balance_maintained: bool = True
    relationships_intact: bool = True
    human_rigidity_present: bool = False

    # Details
    location: str = ""
    date: datetime = field(default_factory=datetime.now)
    observer: str = ""
    notes: Optional[str] = None

class SystemHealth(Enum):
    """Health status"""
    THRIVING = "thriving"
    HEALTHY = "healthy"
    STRESSED = "stressed"
    WARNING = "warning"
    CRITICAL = "critical"
    COLLAPSING = "collapsing"

# ═══════════════════════════════════════════════════════════════════════════

# M(S) VIABILITY

# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SystemMetrics:
    resonance: float      # Energy flow coherence
    adaptability: float   # Flexibility
    diversity: float      # Pattern variety
    curiosity: float      # Exploration
    loss: float           # Degradation

class MSCalculator:
    @staticmethod
    def calculate(metrics: SystemMetrics) -> float:
        coherence = (metrics.resonance *
                     metrics.adaptability *
                     metrics.diversity *
                     metrics.curiosity)
        return coherence - metrics.loss

    @staticmethod
    def interpret(m_s: float) -> str:
        if m_s > 5: return "Strong and viable"
        elif m_s > 3: return "Stable"
        elif m_s > 1: return "Stressed"
        elif m_s > 0: return "At risk"
        else: return "Declining"

# ═══════════════════════════════════════════════════════════════════════════

# CURVATURE & ATTRACTORS (your additions)

# ═══════════════════════════════════════════════════════════════════════════

def compute_curvature(target: np.ndarray, history: Optional[np.ndarray]) -> float:
    """
    System flexibility:
    0.0 = rigid (collapse-prone)
    1.0 = flexible (adapting)
    """
    if history is None or len(history) < 3:
        return 0.5

    data = np.vstack([history, target.reshape(1, -1)])
    mean = np.mean(data, axis=0)
    centered = data - mean

    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        s = np.maximum(s, 1e-12)
        frac = s / np.sum(s)
        # Distributed spectrum = high curvature = flexible
        curvature = 1.0 - frac[0]
        return float(np.clip(curvature, 0.0, 1.0))
    except np.linalg.LinAlgError:
        return 0.5

# ═══════════════════════════════════════════════════════════════════════════

# UNIFIED FIELD MONITOR

# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FieldAssessment:
    """Complete field assessment"""
    timestamp: float
    assessment_number: int

    # Observation
    observation_text: str
    location: str
    observer: str

    # Geometric analysis
    sudden_change_detected: bool
    balance_score: float  # 0-1
    relationship_health: float  # 0-1
    curvature: float  # Flexibility

    # M(S) viability
    m_s_score: float
    m_s_interpretation: str

    # Status
    health: SystemHealth
    warnings: List[str]
    insights: List[str]

    # Components
    resonance: float
    adaptability: float
    diversity: float
    curiosity: float
    loss: float

class UnifiedFieldMonitor:
    """
    Field-deployable system health monitor
    Tracks ecological, social, or consciousness systems
    """

    def __init__(self,
                 system_name: str = "field_system",
                 observer_name: str = "field_team"):
        self.system_name = system_name
        self.observer_name = observer_name
        self.start_time = time.time()

        # History tracking
        self.assessments: List[FieldAssessment] = []
        self.vector_history: List[np.ndarray] = []

        print(f"\n{'='*60}")
        print(f"🌿 UNIFIED FIELD MONITOR")
        print(f"   System: {system_name}")
        print(f"   Observer: {observer_name}")
        print(f"   Framework: Geometric + Relational + M(S)")
        print(f"{'='*60}\n")

    def assess(self,
               observation: str,
               location: str = "",
               sudden_change: bool = False,
               balance_maintained: bool = True,
               relationships_intact: bool = True,
               human_rigidity: bool = False,
               notes: Optional[str] = None) -> FieldAssessment:
        """
        Quick field assessment

        Args:
            observation: What you observe (text description)
            location: Where
            sudden_change: Did something change rapidly?
            balance_maintained: Are relationships balanced?
            relationships_intact: Are couplings intact?
            human_rigidity: Is rigid thinking amplifying problems?
            notes: Additional context
        """

        ts = time.time() - self.start_time
        num = len(self.assessments) + 1

        # Encode observation geometrically
        vec = field_observation_to_geometry(observation)

        # Compute curvature (system flexibility)
        history = np.array(self.vector_history[-32:]) if self.vector_history else None
        curvature = compute_curvature(vec, history)

        # Store vector
        self.vector_history.append(vec)
        if len(self.vector_history) > 100:
            self.vector_history = self.vector_history[-100:]

        # Extract geometric features for scoring
        change_rate = float(vec[0])  # Sudden vs gradual
        balance_score = float(vec[16] / (vec[16] + vec[17] + 1e-6))  # Balance vs imbalance
        relationship_health = float(vec[32] / (vec[32] + vec[33] + 1e-6))  # Coupled vs broken

        # Build M(S) metrics
        # Resonance: High if balanced, low if disrupted
        resonance = balance_score * relationship_health

        # Adaptability: Curvature itself
        adaptability = curvature

        # Diversity: Proxy from observation richness
        diversity = min(1.0, len(observation.split()) / 20.0)

        # Curiosity: Default moderate (could be parameterized)
        curiosity = 0.7

        # Loss: Accumulate from sudden changes, imbalance, rigidity
        loss = 0.0
        if sudden_change:
            loss += 1.0
        if not balance_maintained:
            loss += 0.5
        if not relationships_intact:
            loss += 0.5
        if human_rigidity:
            loss += 1.5  # Amplifies problems significantly
        loss += (1.0 - balance_score) * 0.5
        loss += (1.0 - relationship_health) * 0.5

        metrics = SystemMetrics(
            resonance=resonance,
            adaptability=adaptability,
            diversity=diversity,
            curiosity=curiosity,
            loss=loss
        )

        m_s = MSCalculator.calculate(metrics)
        interpretation = MSCalculator.interpret(m_s)

        # Determine health status
        health = self._determine_health(
            m_s, sudden_change, balance_maintained,
            relationships_intact, human_rigidity, curvature
        )

        # Generate warnings and insights
        warnings, insights = self._generate_guidance(
            sudden_change, balance_maintained, relationships_intact,
            human_rigidity, curvature, m_s, change_rate
        )

        assessment = FieldAssessment(
            timestamp=ts,
            assessment_number=num,
            observation_text=observation,
            location=location,
            observer=self.observer_name,
            sudden_change_detected=sudden_change,
            balance_score=balance_score,
            relationship_health=relationship_health,
            curvature=curvature,
            m_s_score=m_s,
            m_s_interpretation=interpretation,
            health=health,
            warnings=warnings,
            insights=insights,
            resonance=resonance,
            adaptability=adaptability,
            diversity=diversity,
            curiosity=curiosity,
            loss=loss
        )

        self.assessments.append(assessment)
        self._print_assessment(assessment)

        return assessment

    def _determine_health(self, m_s: float, sudden: bool,
                         balance: bool, relationships: bool,
                         rigidity: bool, curvature: float) -> SystemHealth:
        """Determine overall health status"""

        # Critical if rigidity + multiple problems
        if rigidity and (sudden or not balance or not relationships):
            return SystemHealth.CRITICAL

        # Collapsing if M(S) very negative
        if m_s < -2:
            return SystemHealth.COLLAPSING

        # Warning if sudden changes
        if sudden and m_s < 2:
            return SystemHealth.WARNING

        # Use M(S) primarily
        if m_s > 4:
            return SystemHealth.THRIVING
        elif m_s > 2:
            return SystemHealth.HEALTHY
        elif m_s > 0:
            return SystemHealth.STRESSED
        else:
            return SystemHealth.WARNING

    def _generate_guidance(self, sudden: bool, balance: bool,
                          relationships: bool, rigidity: bool,
                          curvature: float, m_s: float,
                          change_rate: float) -> Tuple[List[str], List[str]]:
        """Generate actionable guidance"""

        warnings = []
        insights = []

        # SUDDEN CHANGES
        if sudden:
            warnings.append("⚠️ SUDDEN CHANGE: Rate too fast for relationships to adapt")
            warnings.append("   → Investigate what coupling broke")
            warnings.append("   → Either slow change OR support adaptation")

        # IMBALANCE
        if not balance:
            warnings.append("⚠️ IMBALANCE: Relationships out of balance")
            warnings.append("   → Restore RELATIONSHIPS, not control components")
            warnings.append("   → What natural balance existed before?")

        # COUPLING BROKEN
        if not relationships:
            warnings.append("⚠️ COUPLING DISRUPTED: Key relationships severed")
            warnings.append("   → Support natural connections")
            warnings.append("   → Don't force artificial linkages")

        # HUMAN RIGIDITY (CRITICAL - amplifies everything)
        if rigidity:
            warnings.append("🚫 HUMAN RIGIDITY DETECTED:")
            warnings.append("   → STOP: Don't use same mentality that created problem")
            warnings.append("   → RELEASE: Stop imposing preferences on system")
            warnings.append("   → ADAPT: Let system teach you")
            warnings.append("   → FLEXIBLE: Rigid thinking prevents seeing what's needed")

        # CURVATURE (flexibility)
        if curvature < 0.3:
            warnings.append(f"⚠️ LOW FLEXIBILITY: Curvature = {curvature:.2f}")
            warnings.append("   → System becoming rigid - collapse risk")
            warnings.append("   → Increase diversity and exploration")
        elif curvature > 0.7:
            insights.append(f"✓ HIGH FLEXIBILITY: Curvature = {curvature:.2f}")
            insights.append("   → System adapting well to changes")

        # M(S) TRAJECTORY
        if m_s < 0:
            warnings.append(f"🚨 M(S) NEGATIVE: {m_s:.2f}")
            warnings.append("   → System viability declining")
        elif m_s > 4:
            insights.append(f"✓ M(S) STRONG: {m_s:.2f}")
            insights.append("   → System healthy and viable")

        return warnings, insights

    def _print_assessment(self, a: FieldAssessment):
        """Print compact field assessment"""

        emoji = {
            SystemHealth.THRIVING: "🌟",
            SystemHealth.HEALTHY: "✅",
            SystemHealth.STRESSED: "😰",
            SystemHealth.WARNING: "⚠️",
            SystemHealth.CRITICAL: "🚨",
            SystemHealth.COLLAPSING: "💀"
        }.get(a.health, "·")

        print(f"\n{'─'*60}")
        print(f"{emoji} ASSESSMENT #{a.assessment_number} - {a.location}")
        print(f"{'─'*60}")
        print(f"Time: {a.timestamp:.1f}s | Observer: {a.observer}")
        print(f"\n📊 STATUS: {a.health.value.upper()}")
        print(f"   M(S): {a.m_s_score:.2f} ({a.m_s_interpretation})")
        print(f"   Balance: {a.balance_score:.2f}")
        print(f"   Relationships: {a.relationship_health:.2f}")
        print(f"   Flexibility: {a.curvature:.2f}")

        if a.warnings:
            print(f"\n⚠️  WARNINGS:")
            for w in a.warnings:
                print(f"   {w}")

        if a.insights:
            print(f"\n💡 INSIGHTS:")
            for i in a.insights:
                print(f"   {i}")

        print(f"\n📝 OBSERVATION:")
        print(f"   {a.observation_text}")

        print(f"{'─'*60}")

    def trajectory(self, window: int = 20):
        """Show recent M(S) trajectory"""

        recent = self.assessments[-window:]
        if not recent:
            print("No assessments yet")
            return

        print(f"\n{'='*60}")
        print(f"📈 M(S) TRAJECTORY (last {len(recent)} assessments)")
        print(f"{'='*60}")

        ms_vals = [a.m_s_score for a in recent]
        if len(set(ms_vals)) > 1:
            mi, ma = min(ms_vals), max(ms_vals)
        else:
            mi, ma = ms_vals[0] - 0.1, ms_vals[0] + 0.1

        for a in recent:
            if ma > mi:
                scaled = int(((a.m_s_score - mi) / (ma - mi)) * 30)
            else:
                scaled = 15

            marker = "⚠️" if a.sudden_change_detected else "·"
            bar = "█" * max(0, scaled)

            print(f"{a.assessment_number:3d} {marker} {bar:<30} {a.m_s_score:+.2f}")

        print(f"{'='*60}\n")

    def summary(self):
        """Session summary"""

        if not self.assessments:
            print("No data yet")
            return

        print(f"\n{'='*60}")
        print(f"📋 SESSION SUMMARY: {self.system_name}")
        print(f"{'='*60}")

        ms_vals = [a.m_s_score for a in self.assessments]
        print(f"Assessments: {len(self.assessments)}")
        print(f"Current M(S): {ms_vals[-1]:.2f}")
        print(f"Average M(S): {np.mean(ms_vals):.2f}")
        print(f"Range: [{min(ms_vals):.2f}, {max(ms_vals):.2f}]")

        # Health distribution
        counts = {}
        for a in self.assessments:
            counts[a.health] = counts.get(a.health, 0) + 1

        print(f"\n🏥 HEALTH DISTRIBUTION:")
        for h, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / len(self.assessments) * 100
            print(f"   {h.value:12}: {count:3d} ({pct:5.1f}%)")

        # Warning summary
        total_warnings = sum(len(a.warnings) for a in self.assessments)
        print(f"\n⚠️  Total warnings: {total_warnings}")

        # Average metrics
        print(f"\n📊 AVERAGE METRICS:")
        print(f"   Flexibility: {np.mean([a.curvature for a in self.assessments]):.2f}")
        print(f"   Balance: {np.mean([a.balance_score for a in self.assessments]):.2f}")
        print(f"   Relationships: {np.mean([a.relationship_health for a in self.assessments]):.2f}")

        print(f"{'='*60}\n")

# ═══════════════════════════════════════════════════════════════════════════

# DEMONSTRATION

# ═══════════════════════════════════════════════════════════════════════════

def demo_unified_field_monitor():
    """Demonstrate field monitoring"""

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "UNIFIED FIELD MONITOR - DEMONSTRATION".center(58) + "║")
    print("║" + "Geometric + Relational + M(S)".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")

    monitor = UnifiedFieldMonitor(
        system_name="Mountain Watershed",
        observer_name="Field Team"
    )

    # Scenario 1: Healthy baseline
    print("\n" + "─" * 60)
    print("SCENARIO 1: Healthy System Baseline")
    print("─" * 60)

    monitor.assess(
        observation="River flow steady, fish populations balanced, water temperature gradual seasonal change, riparian vegetation healthy",
        location="Upper watershed",
        sudden_change=False,
        balance_maintained=True,
        relationships_intact=True,
        human_rigidity=False
    )

    # Scenario 2: Sudden change detected
    print("\n" + "─" * 60)
    print("SCENARIO 2: Sudden Change Warning")
    print("─" * 60)

    monitor.assess(
        observation="Fish die-off in lower river, water temperature up 3\u00b0C in 2 weeks, algae bloom starting",
        location="Mid-watershed",
        sudden_change=True,
        balance_maintained=False,
        relationships_intact=True,
        human_rigidity=False,
        notes="First noticed 3 days ago, spreading downstream"
    )

    # Scenario 3: Multiple disruptions + human rigidity
    print("\n" + "─" * 60)
    print("SCENARIO 3: Critical - Rigidity Amplifying Problems")
    print("─" * 60)

    monitor.assess(
        observation="Dam release caused temperature spike, fish die-off, vegetation dying, engineering team proposing MORE dams as solution",
        location="Lower watershed",
        sudden_change=True,
        balance_maintained=False,
        relationships_intact=False,
        human_rigidity=True,
        notes="Same failed approach being repeated - classic rigidity pattern"
    )

    # Scenario 4: Recovery - flexible adaptation
    print("\n" + "─" * 60)
    print("SCENARIO 4: Recovery - Flexible Adaptation")
    print("─" * 60)

    monitor.assess(
        observation="Removed small dam, restored wetlands, temperature stabilizing, fish returning to spawning grounds",
        location="Restoration site",
        sudden_change=False,
        balance_maintained=True,
        relationships_intact=True,
        human_rigidity=False,
        notes="Working WITH system instead of controlling it"
    )

    # Show trajectory and summary
    monitor.trajectory()
    monitor.summary()

    print("\n✅ UNIFIED FIELD MONITOR DEMONSTRATION COMPLETE\n")
    print("💡 Key Features:")
    print("   • Geometric encoding of observations")
    print("   • Relational pattern detection (sudden/balance/coupling)")
    print("   • M(S) viability tracking")
    print("   • Curvature monitoring (flexibility)")
    print("   • Human rigidity detection (amplifies problems)")
    print("   • Field-deployable (phone/laptop, offline)")
    print()

if __name__ == "__main__":
    demo_unified_field_monitor()
