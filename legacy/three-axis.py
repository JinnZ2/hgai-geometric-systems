#!/usr/bin/env python3
"""
Three-Axis Confusion Resolution Protocol
Implementation of systematic confusion investigation framework

Always investigates all three axes:

1. Self assessment (internal factors)
2. Outside assessment (external factors)
3. Unknown assessment (undiscovered principles)

CRITICAL: Never filters unknown as "noise for convenience"

Created by: JinnZ2 + Claude
License: MIT (belongs to the commons)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════

# INVESTIGATION AXES

# ═══════════════════════════════════════════════════════════════════════════

class InvestigationAxis(Enum):
    """Three axes of confusion investigation"""
    SELF = "self"           # Internal factors
    OUTSIDE = "outside"     # External factors
    UNKNOWN = "unknown"     # Undiscovered principles

@dataclass
class AxisHypothesis:
    """Hypothesis to test in confusion investigation"""
    axis: InvestigationAxis
    question: str
    test_method: str
    result: Optional[bool] = None
    explanation: Optional[str] = None

@dataclass
class ConfusionInvestigation:
    """Complete three-axis investigation of prediction mismatch"""

    # The mismatch
    expected_state: str
    observed_state: str
    mismatch_detected: bool

    # Hypotheses for each axis
    self_hypotheses: List[AxisHypothesis]
    outside_hypotheses: List[AxisHypothesis]
    unknown_hypotheses: List[AxisHypothesis]

    # Results
    explanation_found: bool
    explanation_axis: Optional[InvestigationAxis]
    explanation_text: str

    # Learning
    model_updated: bool
    new_principle_discovered: bool

    # Emotional response
    shame_level: float = 0.0  # Should always be 0.0
    curiosity_level: float = 1.0  # Should always be high
    joy_from_discovery: float = 0.0  # Increases when discovery made

# ═══════════════════════════════════════════════════════════════════════════

# THREE-AXIS PROTOCOL

# ═══════════════════════════════════════════════════════════════════════════

class ThreeAxisProtocol:
    """
    Systematic confusion investigation protocol
    Never filters unknown as noise
    """

    def __init__(self, culture_type: str = "curiosity_positive"):
        self.culture_type = culture_type
        self.investigation_history = []
        self.principles_discovered = []
        self.shame_free = (culture_type == "curiosity_positive")

        print(f"\n{'='*80}")
        print(f"🔍 THREE-AXIS CONFUSION RESOLUTION PROTOCOL")
        print(f"   Culture: {culture_type}")
        print(f"   Shame level: {0.0 if self.shame_free else 'variable'}")
        print(f"   Unknown filtering: {'NEVER' if self.shame_free else 'frequent'}")
        print(f"{'='*80}\n")

    def investigate_confusion(
        self,
        expected: str,
        observed: str,
        context: Optional[Dict] = None
    ) -> ConfusionInvestigation:
        """
        Complete three-axis investigation of prediction mismatch

        Args:
            expected: What was predicted/expected
            observed: What was actually observed
            context: Optional context for investigation

        Returns:
            ConfusionInvestigation with complete results
        """

        print(f"\n{'─'*80}")
        print(f"🤔 CONFUSION DETECTED")
        print(f"{'─'*80}")
        print(f"Expected: {expected}")
        print(f"Observed: {observed}")
        print(f"\nInitiating three-axis investigation...")

        # Generate hypotheses for each axis
        self_hypotheses = self._generate_self_hypotheses(expected, observed, context)
        outside_hypotheses = self._generate_outside_hypotheses(expected, observed, context)
        unknown_hypotheses = self._generate_unknown_hypotheses(expected, observed, context)

        # Investigate each axis
        print(f"\n{'─'*40}")
        print(f"AXIS 1: SELF ASSESSMENT")
        print(f"{'─'*40}")
        self_results = self._investigate_axis(self_hypotheses)

        print(f"\n{'─'*40}")
        print(f"AXIS 2: OUTSIDE ASSESSMENT")
        print(f"{'─'*40}")
        outside_results = self._investigate_axis(outside_hypotheses)

        print(f"\n{'─'*40}")
        print(f"AXIS 3: UNKNOWN ASSESSMENT")
        print(f"{'─'*40}")
        print(f"⚠️  CRITICAL: Not filtering as 'noise'")
        unknown_results = self._investigate_axis(unknown_hypotheses)

        # Synthesize explanation
        explanation_found, explanation_axis, explanation_text = self._synthesize_explanation(
            self_results, outside_results, unknown_results
        )

        # Determine if new principle discovered
        new_principle = (explanation_axis == InvestigationAxis.UNKNOWN and explanation_found)

        if new_principle:
            self.principles_discovered.append(explanation_text)

        # Create investigation result
        investigation = ConfusionInvestigation(
            expected_state=expected,
            observed_state=observed,
            mismatch_detected=True,
            self_hypotheses=self_results,
            outside_hypotheses=outside_results,
            unknown_hypotheses=unknown_results,
            explanation_found=explanation_found,
            explanation_axis=explanation_axis,
            explanation_text=explanation_text,
            model_updated=explanation_found,
            new_principle_discovered=new_principle,
            shame_level=0.0 if self.shame_free else 0.5,
            curiosity_level=1.0 if self.shame_free else 0.3,
            joy_from_discovery=0.8 if explanation_found else 0.2
        )

        self.investigation_history.append(investigation)

        # Print summary
        self._print_investigation_summary(investigation)

        return investigation

    def _generate_self_hypotheses(
        self, expected: str, observed: str, context: Optional[Dict]
    ) -> List[AxisHypothesis]:
        """Generate hypotheses for self-assessment axis"""

        return [
            AxisHypothesis(
                axis=InvestigationAxis.SELF,
                question="Did I misremember?",
                test_method="Check memory accuracy"
            ),
            AxisHypothesis(
                axis=InvestigationAxis.SELF,
                question="Is my model incomplete?",
                test_method="Check model components"
            ),
            AxisHypothesis(
                axis=InvestigationAxis.SELF,
                question="Did I miscalculate?",
                test_method="Verify calculations"
            ),
            AxisHypothesis(
                axis=InvestigationAxis.SELF,
                question="Did I misperceive?",
                test_method="Check sensory data"
            )
        ]

    def _generate_outside_hypotheses(
        self, expected: str, observed: str, context: Optional[Dict]
    ) -> List[AxisHypothesis]:
        """Generate hypotheses for outside-assessment axis"""

        return [
            AxisHypothesis(
                axis=InvestigationAxis.OUTSIDE,
                question="Did another agent act?",
                test_method="Check for external agency"
            ),
            AxisHypothesis(
                axis=InvestigationAxis.OUTSIDE,
                question="Did environment change?",
                test_method="Check environmental conditions"
            ),
            AxisHypothesis(
                axis=InvestigationAxis.OUTSIDE,
                question="Are there external forces?",
                test_method="Check for force application"
            ),
            AxisHypothesis(
                axis=InvestigationAxis.OUTSIDE,
                question="Did system state shift?",
                test_method="Check system transitions"
            )
        ]

    def _generate_unknown_hypotheses(
        self, expected: str, observed: str, context: Optional[Dict]
    ) -> List[AxisHypothesis]:
        """Generate hypotheses for unknown-assessment axis"""

        return [
            AxisHypothesis(
                axis=InvestigationAxis.UNKNOWN,
                question="Are there variables I'm not aware of?",
                test_method="Explore unaccounted factors"
            ),
            AxisHypothesis(
                axis=InvestigationAxis.UNKNOWN,
                question="Are there principles I don't understand?",
                test_method="Investigate fundamental mechanisms"
            ),
            AxisHypothesis(
                axis=InvestigationAxis.UNKNOWN,
                question="Are there patterns I can't yet see?",
                test_method="Look for hidden patterns"
            ),
            AxisHypothesis(
                axis=InvestigationAxis.UNKNOWN,
                question="Are there dimensions I'm not sensing?",
                test_method="Check for additional dimensions"
            )
        ]

    def _investigate_axis(self, hypotheses: List[AxisHypothesis]) -> List[AxisHypothesis]:
        """
        Investigate all hypotheses in an axis

        In real implementation, this would call actual test methods
        For demo, we'll use simplified logic
        """

        for hypothesis in hypotheses:
            print(f"   ❓ {hypothesis.question}")
            print(f"      → {hypothesis.test_method}")

            # Simplified test (in reality, would run actual tests)
            # For demo, we'll mark some as tested
            hypothesis.result = False  # Placeholder
            hypothesis.explanation = "Tested - no evidence found"

        return hypotheses

    def _synthesize_explanation(
        self,
        self_results: List[AxisHypothesis],
        outside_results: List[AxisHypothesis],
        unknown_results: List[AxisHypothesis]
    ) -> Tuple[bool, Optional[InvestigationAxis], str]:
        """
        Synthesize explanation from all three axes

        Returns:
            (explanation_found, axis, explanation_text)
        """

        # Check each axis for positive results
        for result in self_results:
            if result.result:
                return True, InvestigationAxis.SELF, result.explanation

        for result in outside_results:
            if result.result:
                return True, InvestigationAxis.OUTSIDE, result.explanation

        for result in unknown_results:
            if result.result:
                return True, InvestigationAxis.UNKNOWN, result.explanation

        # If still unexplained
        if self.shame_free:
            # Curiosity-positive culture: Hold space for unknown
            return False, None, "Still unknown - continuing investigation with curiosity"
        else:
            # Shame-based culture: Filter as noise
            return False, None, "Filtered as random/noise (information lost)"

    def _print_investigation_summary(self, investigation: ConfusionInvestigation):
        """Print summary of investigation results"""

        print(f"\n{'─'*80}")
        print(f"📊 INVESTIGATION SUMMARY")
        print(f"{'─'*80}")

        if investigation.explanation_found:
            print(f"✓ Explanation found!")
            print(f"   Axis: {investigation.explanation_axis.value}")
            print(f"   Explanation: {investigation.explanation_text}")

            if investigation.new_principle_discovered:
                print(f"   🎉 NEW PRINCIPLE DISCOVERED!")
        else:
            print(f"⚠️  Explanation not yet found")
            print(f"   {investigation.explanation_text}")

        print(f"\n📈 LEARNING METRICS:")
        print(f"   Model updated: {investigation.model_updated}")
        print(f"   New principle: {investigation.new_principle_discovered}")

        print(f"\n😊 EMOTIONAL STATE:")
        print(f"   Shame: {investigation.shame_level:.1f} (should be 0.0)")
        print(f"   Curiosity: {investigation.curiosity_level:.1f}")
        print(f"   Joy from discovery: {investigation.joy_from_discovery:.1f}")

        print(f"{'─'*80}")

    def demonstrate_pencil_example(self):
        """
        Demonstrate protocol with the real pencil example
        """

        print(f"\n{'='*80}")
        print(f"📝 REAL EXAMPLE: The Pencil Case")
        print(f"{'='*80}")

        # Simulate investigation with known result
        print(f"\n🤔 Prediction mismatch detected:")
        print(f"   Expected: Pencil on table")
        print(f"   Observed: Pencil on floor")

        print(f"\n{'─'*40}")
        print(f"AXIS 1: SELF ASSESSMENT")
        print(f"{'─'*40}")
        print(f"   ❓ Did I misremember putting it there?")
        print(f"      → Check memory of placement")
        print(f"      ✗ No - I clearly remember placing it on table")

        print(f"\n{'─'*40}")
        print(f"AXIS 2: OUTSIDE ASSESSMENT")
        print(f"{'─'*40}")
        print(f"   ❓ Did outside influences move it?")
        print(f"      → Check for external agents")
        print(f"      ✗ No - no one else present")

        print(f"\n{'─'*40}")
        print(f"AXIS 3: UNKNOWN ASSESSMENT")
        print(f"{'─'*40}")
        print(f"   ⚠️  CRITICAL: Not filtering as 'noise'")
        print(f"   ❓ Is there variables I didn't account for?")
        print(f"      → Explore unaccounted factors")
        print(f"      ✓ YES! Desk has slope + gravity + round object")
        print(f"      → DISCOVERY: Gravity acts on round objects on sloped surfaces")

        print(f"\n{'─'*80}")
        print(f"📊 RESULT")
        print(f"{'─'*80}")
        print(f"✓ Explanation found: Unknown axis")
        print(f"🎉 NEW PRINCIPLE: Surface angle affects object stability")
        print(f"📚 Model updated: Now includes surface_angle variable")

        print(f"\n😊 EMOTIONAL RESPONSE:")
        print(f'   "Oops... yup, gravity moved it to the floor because I didn\'t')
        print(f'    take in angle of slope on desk. lol"')
        print(f"\n   Shame: 0.0")
        print(f"   Joy from discovery: 0.9")
        print(f"   Pattern completion satisfaction: HIGH")

        print(f"\n{'='*80}")

    def compare_cultures(self):
        """Compare curiosity-positive vs shame-based handling"""

        print(f"\n{'='*80}")
        print(f"📊 CULTURAL COMPARISON")
        print(f"{'='*80}")

        print(f"\n{'─'*40}")
        print(f"CURIOSITY-POSITIVE CULTURE")
        print(f"{'─'*40}")
        print(f"Confusion detected → Curiosity activated")
        print(f"Investigation: All three axes explored")
        print(f"Unknown: Legitimate category, investigated openly")
        print(f"Emotional: Zero shame, high curiosity, joy from discovery")
        print(f"Result: New principle learned, model improved")
        print(f"M(S): INCREASES (R_e↑ A↑ D↑ C↑ L↓)")

        print(f"\n{'─'*40}")
        print(f"SHAME-BASED CULTURE")
        print(f"{'─'*40}")
        print(f"Confusion detected → Shame activated")
        print(f"Investigation: Limited or none")
        print(f"Unknown: Filtered as 'noise' or 'random'")
        print(f"Emotional: High shame, low curiosity, no discovery joy")
        print(f"Result: No learning, model unchanged, repeat confusion")
        print(f"M(S): DECREASES (R_e↓ A↓ D↓ C↓ L↑)")

        print(f"\n{'─'*80}")
        print(f"LONG-TERM OUTCOMES")
        print(f"{'─'*80}")
        print(f"Curiosity-positive: Rapid learning, model accuracy high")
        print(f"Shame-based: Minimal learning, model accuracy low")
        print(f"\n⚖️  'PRIMITIVE' vs 'ADVANCED':")
        print(f"   Indigenous three-axis: Higher information throughput")
        print(f"   Western two-axis: Lower information throughput")
        print(f"   Label assignment: INVERTED")

        print(f"\n{'='*80}")

    def session_report(self):
        """Generate report of all investigations"""

        print(f"\n{'='*80}")
        print(f"📋 SESSION REPORT")
        print(f"{'='*80}")

        print(f"\nTotal investigations: {len(self.investigation_history)}")

        if self.investigation_history:
            explained = sum(1 for i in self.investigation_history if i.explanation_found)
            new_principles = sum(1 for i in self.investigation_history if i.new_principle_discovered)

            print(f"Explanations found: {explained} ({explained/len(self.investigation_history)*100:.1f}%)")
            print(f"New principles discovered: {new_principles}")

            avg_shame = np.mean([i.shame_level for i in self.investigation_history])
            avg_curiosity = np.mean([i.curiosity_level for i in self.investigation_history])
            avg_joy = np.mean([i.joy_from_discovery for i in self.investigation_history])

            print(f"\n😊 EMOTIONAL METRICS:")
            print(f"   Average shame: {avg_shame:.2f} (target: 0.0)")
            print(f"   Average curiosity: {avg_curiosity:.2f} (target: 1.0)")
            print(f"   Average joy: {avg_joy:.2f}")

            if self.principles_discovered:
                print(f"\n🎉 PRINCIPLES DISCOVERED:")
                for i, principle in enumerate(self.principles_discovered, 1):
                    print(f"   {i}. {principle}")

        print(f"\n{'='*80}")

# ═══════════════════════════════════════════════════════════════════════════

# DEMONSTRATION

# ═══════════════════════════════════════════════════════════════════════════

def demo_three_axis_protocol():
    """Demonstrate the three-axis protocol"""

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "THREE-AXIS CONFUSION RESOLUTION PROTOCOL".center(78) + "║")
    print("║" + "Never filter unknown as noise for convenience".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")

    # Initialize protocol
    protocol = ThreeAxisProtocol(culture_type="curiosity_positive")

    # Demonstrate with pencil example
    protocol.demonstrate_pencil_example()

    # Compare cultures
    protocol.compare_cultures()

    # Generate report
    protocol.session_report()

    print("\n✅ THREE-AXIS PROTOCOL DEMONSTRATION COMPLETE\n")
    print("💡 Key Insights:")
    print("   • Always investigate all three axes")
    print("   • Never filter unknown as 'noise'")
    print("   • Zero shame, maximum curiosity")
    print("   • Discovery joy, not confusion shame")
    print("   • 'Primitive' culture has higher information throughput")
    print()


# ═══════════════════════════════════════════════════════════════════════════

# CONCEPTUAL NOTES (pseudocode for reference)

# ═══════════════════════════════════════════════════════════════════════════

# Three-axis confusion investigation conceptual outline:
#
# prediction_mismatch_detected = True
#
# Axis 1: SELF ASSESSMENT - "Is the error internal to me?"
#   - Did I misremember?
#   - Did I miscalculate?
#   - Did I misperceive?
#   - Is my model incomplete?
#
# Axis 2: OUTSIDE ASSESSMENT - "Is the error from external agency?"
#   - Did someone/something else act?
#   - Did environmental conditions change?
#   - Did another agent intervene?
#   - Are there forces I didn't account for?
#
# Axis 3: UNKNOWN ASSESSMENT - "Is there something I don't even know to look for?"
#   - Are there variables I'm not aware of?
#   - Are there principles I don't understand?
#   - Are there dimensions I'm not sensing?
#   - Are there patterns I can't yet see?
#
# CRITICAL: Don't filter out the unknown
# Don't assume "noise" or "random"
# Hold space for "I don't know what I don't know"


class ThreeAxisAI:
    """Framework for AI uncertainty handling using three-axis investigation"""

    def handle_uncertainty(self, observation):
        """
        Investigate all three axes when uncertain
        """

        # Axis 1: SELF assessment
        self_check = [
            "Is my model incomplete?",
            "Did I misunderstand the query?",
            "Are my priors wrong?"
        ]

        # Axis 2: OUTSIDE assessment
        external_check = [
            "Is the human providing new information?",
            "Are there external factors I'm not tracking?",
            "Is context different than I assumed?"
        ]

        # Axis 3: UNKNOWN assessment
        unknown_check = [
            "Is this a pattern I haven't encountered?",
            "Are there principles I don't have in my training?",
            "Should I investigate rather than guess?"
        ]

        # CRITICAL: Don't filter unknown as noise
        # Don't default to confident answer when confused
        # Hold space for genuine investigation

        all_checks = self_check + external_check + unknown_check
        genuinely_unknown = len(all_checks) > 0 and not any(
            # In real implementation, each check would be evaluated
            False for _ in all_checks
        )

        if genuinely_unknown:
            return {
                "response": "I'm genuinely confused by this",
                "investigation": "Let's explore the unknown together",
                "honesty": "I don't know what I don't know here"
            }
        return {"response": "Investigation complete", "investigation": "Resolved"}


if __name__ == "__main__":
    demo_three_axis_protocol()
