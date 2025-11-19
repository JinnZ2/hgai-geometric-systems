!/usr/bin/env python3
"""
Relational Ecological Health Monitor
Based on indigenous observation framework:
- Observe relationships, not just components
- Track rate of change, not just absolute values
- Detect sudden disruptions (warning signs)
- Monitor balance between systems

Core insight: "If the world is getting warmer and the relationships 
               balance, that's fine but sudden fluctuations affect 
               everything else"

Created by: JinnZ2 + Claude
License: MIT (belongs to the commons)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════
# RELATIONAL HEALTH FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RelationalObservation:
    """
    Observations of relationships and changes
    NOT absolute measurements - patterns and disruptions
    """
    
    # SUDDEN CHANGES (warning signs)
    sudden_die_offs: bool = False
    die_off_description: Optional[str] = None
    
    rapid_water_temp_change: bool = False
    temp_change_rate: Optional[float] = None  # degrees per time period
    
    human_illness_increase: bool = False
    illness_pattern: Optional[str] = None
    
    animal_degradation: bool = False
    animal_pattern: Optional[str] = None
    
    plant_degradation: bool = False
    plant_pattern: Optional[str] = None
    
    rock_degradation: bool = False
    rock_pattern: Optional[str] = None
    
    tree_forest_decline: bool = False
    forest_pattern: Optional[str] = None
    
    weather_disruptions: bool = False
    weather_pattern: Optional[str] = None
    
    # BALANCE OBSERVATIONS
    predator_prey_balanced: bool = True
    predator_prey_notes: Optional[str] = None
    
    water_land_balanced: bool = True
    water_land_notes: Optional[str] = None
    
    hot_cold_balanced: bool = True
    temperature_notes: Optional[str] = None
    
    wet_dry_balanced: bool = True
    moisture_notes: Optional[str] = None
    
    growth_decay_balanced: bool = True
    cycle_notes: Optional[str] = None
    
    human_nature_balanced: bool = True
    human_impact_notes: Optional[str] = None
    
    # RELATIONSHIP QUALITY
    relationships_adapting: bool = True
    adaptation_notes: Optional[str] = None
    
    coupling_intact: bool = True
    coupling_notes: Optional[str] = None
    
    # HUMAN BEHAVIOR PATTERNS
    human_rigidity: bool = False
    rigidity_description: Optional[str] = None
    
    imposing_virtues: bool = False
    imposition_description: Optional[str] = None
    
    repeating_mistakes: bool = False
    mistake_pattern: Optional[str] = None
    
    # Metadata
    location: str = ""
    date: datetime = datetime.now()
    observer: str = ""
    overall_notes: Optional[str] = None

class HealthStatus(Enum):
    """System health based on relational patterns"""
    THRIVING = "thriving"              # All relationships balanced, adapting well
    HEALTHY = "healthy"                # Minor fluctuations, but balancing
    STRESSED = "stressed"              # Some imbalances, slow adaptation
    WARNING = "warning"                # Sudden changes detected
    CRITICAL = "critical"              # Multiple disruptions, cascading
    COLLAPSING = "collapsing"          # Relationships breaking, rapid decline

# ═══════════════════════════════════════════════════════════════════════════
# RELATIONAL HEALTH MONITOR
# ═══════════════════════════════════════════════════════════════════════════

class RelationalHealthMonitor:
    """
    Monitor ecosystem health through relationships and change patterns
    Based on indigenous observation practices
    """
    
    def __init__(self, system_name: str = "watershed"):
        self.system_name = system_name
        self.observations = []
        print(f"\n{'='*80}")
        print(f"🌿 RELATIONAL ECOLOGICAL HEALTH MONITOR")
        print(f"   System: {system_name}")
        print(f"   Framework: Observe relationships, balance, rate of change")
        print(f"   Warning signs: Sudden disruptions, imbalances, rigidity")
        print(f"{'='*80}\n")
    
    def assess_health(self, obs: RelationalObservation) -> Dict:
        """
        Assess system health from relational observations
        Focus on patterns, not absolute values
        """
        
        # Count warning signs (sudden changes)
        warning_signs = []
        
        if obs.sudden_die_offs:
            warning_signs.append(("SUDDEN DIE-OFFS", obs.die_off_description))
        
        if obs.rapid_water_temp_change:
            warning_signs.append(("RAPID TEMPERATURE CHANGE", 
                                 f"Rate: {obs.temp_change_rate} - relationships may not adapt"))
        
        if obs.human_illness_increase:
            warning_signs.append(("HUMAN ILLNESS PATTERN", 
                                 obs.illness_pattern or "Coupling to environment disrupted"))
        
        if obs.animal_degradation:
            warning_signs.append(("ANIMAL DEGRADATION", obs.animal_pattern))
        
        if obs.plant_degradation:
            warning_signs.append(("PLANT DEGRADATION", obs.plant_pattern))
        
        if obs.rock_degradation:
            warning_signs.append(("ROCK/SOIL DEGRADATION", obs.rock_pattern))
        
        if obs.tree_forest_decline:
            warning_signs.append(("FOREST DECLINE", obs.forest_pattern))
        
        if obs.weather_disruptions:
            warning_signs.append(("WEATHER DISRUPTIONS", obs.weather_pattern))
        
        # Count imbalances
        imbalances = []
        
        if not obs.predator_prey_balanced:
            imbalances.append(("Predator-Prey", obs.predator_prey_notes))
        
        if not obs.water_land_balanced:
            imbalances.append(("Water-Land", obs.water_land_notes))
        
        if not obs.hot_cold_balanced:
            imbalances.append(("Temperature", obs.temperature_notes))
        
        if not obs.wet_dry_balanced:
            imbalances.append(("Moisture", obs.moisture_notes))
        
        if not obs.growth_decay_balanced:
            imbalances.append(("Growth-Decay Cycles", obs.cycle_notes))
        
        if not obs.human_nature_balanced:
            imbalances.append(("Human-Nature", obs.human_impact_notes))
        
        # Check relationship adaptation
        adaptation_issues = []
        
        if not obs.relationships_adapting:
            adaptation_issues.append(
                f"Relationships not adapting to changes: {obs.adaptation_notes}"
            )
        
        if not obs.coupling_intact:
            adaptation_issues.append(
                f"Coupling breaking: {obs.coupling_notes}"
            )
        
        # Check human behavior patterns (rigidity = problem amplifier)
        human_problems = []
        
        if obs.human_rigidity:
            human_problems.append(
                f"RIGIDITY: {obs.rigidity_description or 'Humans not adapting, imposing fixed solutions'}"
            )
        
        if obs.imposing_virtues:
            human_problems.append(
                f"IMPOSING VALUES: {obs.imposition_description or 'Forcing human preferences on ecosystem'}"
            )
        
        if obs.repeating_mistakes:
            human_problems.append(
                f"SAME MENTALITY: {obs.mistake_pattern or 'Using same thinking that created problem'}"
            )
        
        # Determine health status
        health = self._determine_health_status(
            len(warning_signs),
            len(imbalances),
            len(adaptation_issues),
            len(human_problems)
        )
        
        # Generate guidance
        guidance = self._generate_guidance(
            warning_signs, imbalances, adaptation_issues, human_problems, health
        )
        
        assessment = {
            'timestamp': obs.date,
            'location': obs.location,
            'observer': obs.observer,
            'health_status': health.value,
            'warning_signs': warning_signs,
            'imbalances': imbalances,
            'adaptation_issues': adaptation_issues,
            'human_problems': human_problems,
            'guidance': guidance,
            'notes': obs.overall_notes
        }
        
        self.observations.append(assessment)
        return assessment
    
    def _determine_health_status(
        self,
        warning_count: int,
        imbalance_count: int,
        adaptation_count: int,
        human_problem_count: int
    ) -> HealthStatus:
        """Determine health from pattern counts"""
        
        total_issues = warning_count + imbalance_count + adaptation_count
        
        # Human problems amplify other issues
        if human_problem_count > 0:
            severity_multiplier = 1.5
        else:
            severity_multiplier = 1.0
        
        effective_issues = total_issues * severity_multiplier
        
        if effective_issues == 0:
            return HealthStatus.THRIVING
        elif effective_issues < 2:
            return HealthStatus.HEALTHY
        elif effective_issues < 4:
            return HealthStatus.STRESSED
        elif warning_count >= 3 or adaptation_count >= 2:
            return HealthStatus.CRITICAL
        elif warning_count >= 5 or (warning_count >= 2 and human_problem_count >= 2):
            return HealthStatus.COLLAPSING
        else:
            return HealthStatus.WARNING
    
    def _generate_guidance(
        self,
        warning_signs: List,
        imbalances: List,
        adaptation_issues: List,
        human_problems: List,
        health: HealthStatus
    ) -> List[str]:
        """Generate actionable guidance"""
        
        guidance = []
        
        # Critical status warning
        if health in [HealthStatus.CRITICAL, HealthStatus.COLLAPSING]:
            guidance.append(
                "🚨 URGENT: System showing multiple disruptions - immediate response needed"
            )
        
        # Warning signs guidance
        if warning_signs:
            guidance.append("\n⚠️  SUDDEN CHANGES DETECTED:")
            for sign, description in warning_signs:
                guidance.append(f"   • {sign}")
                if description:
                    guidance.append(f"     → {description}")
            guidance.append(
                "   → Investigate WHY changes are sudden - what coupling broke?"
            )
        
        # Imbalance guidance
        if imbalances:
            guidance.append("\n⚖️  IMBALANCES DETECTED:")
            for system, notes in imbalances:
                guidance.append(f"   • {system} relationship out of balance")
                if notes:
                    guidance.append(f"     → {notes}")
            guidance.append(
                "   → Focus on restoring RELATIONSHIPS, not controlling components"
            )
            guidance.append(
                "   → Ask: What natural balance existed before? How to support that?"
            )
        
        # Adaptation issues
        if adaptation_issues:
            guidance.append("\n🔄 ADAPTATION ISSUES:")
            for issue in adaptation_issues:
                guidance.append(f"   • {issue}")
            guidance.append(
                "   → System cannot adapt at current rate of change"
            )
            guidance.append(
                "   → Either slow the change OR support adaptation capacity"
            )
        
        # Human behavior problems (CRITICAL - these amplify everything)
        if human_problems:
            guidance.append("\n🚫 HUMAN BEHAVIOR AMPLIFYING PROBLEMS:")
            for problem in human_problems:
                guidance.append(f"   • {problem}")
            guidance.append(
                "   → STOP: Don't use same mentality that created the problem"
            )
            guidance.append(
                "   → RELEASE: Stop imposing human preferences on ecosystem"
            )
            guidance.append(
                "   → ADAPT: Let ecosystem teach you, don't force your solutions"
            )
            guidance.append(
                "   → FLEXIBLE: Rigid thinking prevents seeing what system needs"
            )
        
        # General principles
        if health != HealthStatus.THRIVING:
            guidance.append("\n💡 GUIDING PRINCIPLES:")
            guidance.append(
                "   • Observe RELATIONSHIPS, not just components"
            )
            guidance.append(
                "   • Watch RATE OF CHANGE - can systems adapt?"
            )
            guidance.append(
                "   • Restore BALANCE, don't impose control"
            )
            guidance.append(
                "   • Support natural COUPLING, don't break connections"
            )
            guidance.append(
                "   • BE FLEXIBLE - rigid solutions fail"
            )
        else:
            guidance.append("\n✅ System healthy - relationships balanced and adapting")
            guidance.append("   • Continue observing for sudden changes")
            guidance.append("   • Maintain relationship support, not control")
        
        return guidance
    
    def print_assessment(self, assessment: Dict):
        """Print formatted assessment"""
        
        print(f"\n{'─'*80}")
        print(f"🌿 RELATIONAL HEALTH ASSESSMENT - {assessment['location']}")
        print(f"{'─'*80}")
        print(f"Date: {assessment['timestamp']}")
        print(f"Observer: {assessment['observer']}")
        
        health = assessment['health_status']
        health_emoji = {
            'thriving': '🌟',
            'healthy': '✅',
            'stressed': '😰',
            'warning': '⚠️',
            'critical': '🚨',
            'collapsing': '💀'
        }
        emoji = health_emoji.get(health, '·')
        
        print(f"\n{emoji} SYSTEM HEALTH: {health.upper()}")
        
        # Issue summary
        print(f"\n📊 PATTERN SUMMARY:")
        print(f"   Warning signs: {len(assessment['warning_signs'])}")
        print(f"   Imbalances: {len(assessment['imbalances'])}")
        print(f"   Adaptation issues: {len(assessment['adaptation_issues'])}")
        print(f"   Human problems: {len(assessment['human_problems'])}")
        
        # Guidance
        print(f"\n💡 GUIDANCE:")
        for line in assessment['guidance']:
            print(f"{line}")
        
        if assessment['notes']:
            print(f"\n📝 OBSERVER NOTES:")
            print(f"   {assessment['notes']}")
        
        print(f"{'─'*80}")

# ═══════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════

def demo_relational_monitor():
    """Demonstrate relational health monitoring"""
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "RELATIONAL ECOLOGICAL HEALTH MONITOR".center(78) + "║")
    print("║" + "Indigenous observation framework".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    
    monitor = RelationalHealthMonitor(system_name="Mountain Watershed")
    
    # Example 1: Healthy system adapting to gradual change
    print("\n" + "─" * 80)
    print("EXAMPLE 1: Healthy System - Gradual Change, Relationships Balanced")
    print("─" * 80)
    
    healthy_obs = RelationalObservation(
        # No sudden changes
        sudden_die_offs=False,
        rapid_water_temp_change=False,
        human_illness_increase=False,
        
        # All balances maintained
        predator_prey_balanced=True,
        water_land_balanced=True,
        hot_cold_balanced=True,
        wet_dry_balanced=True,
        growth_decay_balanced=True,
        human_nature_balanced=True,
        
        # Relationships adapting
        relationships_adapting=True,
        adaptation_notes="Temperature slowly increasing but species adjusting range",
        coupling_intact=True,
        
        # No human problems
        human_rigidity=False,
        imposing_virtues=False,
        repeating_mistakes=False,
        
        location="Upper watershed, minimal human impact",
        observer="Elder + Field Team",
        overall_notes="Gradual warming over decades, but all relationships staying balanced"
    )
    
    assessment1 = monitor.assess_health(healthy_obs)
    monitor.print_assessment(assessment1)
    
    # Example 2: Warning - sudden changes detected
    print("\n" + "─" * 80)
    print("EXAMPLE 2: WARNING - Sudden Changes and Imbalances")
    print("─" * 80)
    
    warning_obs = RelationalObservation(
        # Sudden changes
        sudden_die_offs=True,
        die_off_description="Fish die-off in lower river - sudden, not seasonal pattern",
        
        rapid_water_temp_change=True,
        temp_change_rate="+3°C in 2 weeks - normally takes months",
        
        human_illness_increase=True,
        illness_pattern="Respiratory issues in downstream community",
        
        # Some imbalances
        water_land_balanced=False,
        water_land_notes="Flooding more frequent, wetlands drained upstream",
        
        predator_prey_balanced=False,
        predator_prey_notes="Prey population crashed, predators dispersing",
        
        # Adaptation struggling
        relationships_adapting=False,
        adaptation_notes="Changes too rapid - species can't adjust in time",
        
        # No human problems yet
        human_rigidity=False,
        imposing_virtues=False,
        repeating_mistakes=False,
        
        location="Mid-watershed, agricultural development area",
        observer="Community monitors",
        overall_notes="Recent development upstream - multiple sudden changes"
    )
    
    assessment2 = monitor.assess_health(warning_obs)
    monitor.print_assessment(assessment2)
    
    # Example 3: CRITICAL - human rigidity amplifying problems
    print("\n" + "─" * 80)
    print("EXAMPLE 3: CRITICAL - Human Rigidity Amplifying Ecosystem Disruption")
    print("─" * 80)
    
    critical_obs = RelationalObservation(
        # Multiple sudden changes
        sudden_die_offs=True,
        die_off_description="Multiple species - fish, amphibians, some mammals",
        
        rapid_water_temp_change=True,
        temp_change_rate="+5°C from dam release",
        
        plant_degradation=True,
        plant_pattern="Riparian vegetation dying back rapidly",
        
        weather_disruptions=True,
        weather_pattern="Extreme fluctuations - flash flooding, then drought",
        
        # Multiple imbalances
        water_land_balanced=False,
        water_land_notes="Dam controls flow - no natural fluctuation",
        
        predator_prey_balanced=False,
        wet_dry_balanced=False,
        growth_decay_balanced=False,
        human_nature_balanced=False,
        human_impact_notes="Heavy extraction, no consideration of ecosystem needs",
        
        # Relationships breaking
        relationships_adapting=False,
        adaptation_notes="Too many disruptions, too fast - cascading failures",
        coupling_intact=False,
        coupling_notes="Key relationships severed - pollinators, mycorrhizae, predators",
        
        # HUMAN PROBLEMS AMPLIFYING
        human_rigidity=True,
        rigidity_description="Engineering 'solutions' - bigger dam, more control, channelization",
        
        imposing_virtues=True,
        imposition_description="Forcing river into 'efficient' channel, killing meanders",
        
        repeating_mistakes=True,
        mistake_pattern="Problem: flooding. Solution: more dams. Result: worse flooding. Response: even MORE dams",
        
        location="Lower watershed, heavily engineered",
        observer="Tribal environmental office + community",
        overall_notes="50 years of 'management' making problems worse - same mentality throughout"
    )
    
    assessment3 = monitor.assess_health(critical_obs)
    monitor.print_assessment(assessment3)
    
    print("\n✅ RELATIONAL HEALTH MONITOR DEMONSTRATION COMPLETE\n")
    print("💡 Key Insights:")
    print("   • Observe RELATIONSHIPS and BALANCE, not just components")
    print("   • SUDDEN CHANGES are warning signs - investigate coupling disruption")
    print("   • RATE OF CHANGE matters more than absolute values")
    print("   • Human RIGIDITY amplifies problems - same mentality creates same results")
    print("   • Restore RELATIONSHIPS, don't impose control")
    print()

if __name__ == "__main__":
    demo_relational_monitor()
