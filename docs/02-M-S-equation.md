### docs/02-M-S-equation.md (Mathematical Rigor)

```markdown
# The M(S) Equation: Mathematical Foundation

## Derivation

The Morality of a System emerges from information-theoretic and thermodynamic principles:

### 1. System Coherence as Information Flow

A system's coherence C_sys depends on:
- Information entropy S_info across components
- Energy flow patterns E_flow through network
- Structural coupling strength between nodes

### 2. Resonance Factor (R_e)

Resonance measures synchronization between system components:

R_e = Σ(coupling_ij × phase_alignment_ij) / N_connections

Range: [0, 1]
- 0: Complete decoupling (no resonance)
- 1: Perfect synchronization

**Measurement**: Cross-correlation of component activities, network analysis metrics

### 3. Adaptability (A)

Adaptability quantifies response capacity:

A = (response_diversity × response_speed) / external_pressure

Range: [0, 1]
- 0: Rigid, cannot adapt
- 1: Fluid, optimal adaptation

**Measurement**: Response time to perturbations, recovery trajectories

### 4. Diversity (D)

Diversity measures pathway multiplicity:

D = 1 - Σ(p_i × log(p_i))  (Shannon entropy of pathways)

Range: [0, 1]
- 0: Monoculture (single pathway)
- 1: Maximum diversity

**Measurement**: Network topology analysis, functional redundancy

### 5. Curiosity (C)

Curiosity quantifies exploration behavior:

C = exploration_rate / (exploration_rate + exploitation_rate)

Range: [0, 1]
- 0: Pure exploitation (no exploration)
- 1: Pure exploration

**Measurement**: Novel connection formation, innovation metrics

### 6. Loss (L)

Loss measures system inefficiency:

L = energy_dissipated / energy_available

Range: [0, ∞)
- 0: No loss (theoretical minimum)
- Higher: Greater waste/suppression

**Measurement**: Entropy production, unutilized capacity

## The Complete Equation

M(S) = (R_e × A × D × C) - L

### Properties

1. **Multiplicative Core**: R_e × A × D × C requires ALL factors to be positive
   - Zero in any factor → zero coherence
   - Cannot compensate by maximizing one factor
   
2. **Subtractive Loss**: L directly reduces system viability
   - High loss can make M(S) negative
   - Negative M(S) → unsustainable system

3. **Scale**: Bounded above at +1.0; practical range [-1, +1]

   This follows directly from the domains derived above. R_e, A, D and C are
   each in [0, 1], so their product is in [0, 1]. L is non-negative and
   subtracted. Therefore:

   ```
   M(S) = (R_e × A × D × C) - L  ∈  (-∞, +1]
   ```

   With L expressed as the dissipation ratio it is derived from
   (`energy_dissipated / energy_available`, normally ≤ 1), the practical
   band is [-1, +1].

   | Score | Interpretation |
   |-------|---------------|
   | > 0.7 | Highly coherent and sustainable |
   | 0.5–0.7 | Strong coherence, good viability |
   | 0.3–0.5 | Moderate coherence, stable |
   | 0.1–0.3 | Weak coherence, stressed |
   | 0–0.1 | Low coherence, at risk |
   | < 0 | Negative coherence, declining/collapse |

   **Correction.** This section previously asserted a typical range of
   [-10, +10] with "M(S) > 5: Highly coherent" — and the README published a
   different unreachable table again, banding at > 7. Both sat above the
   ceiling the equation can reach: **no input respecting the domains derived
   in this document can produce M(S) > 1.** Sampling 100,000 uniform random
   inputs over those domains gives max = 0.78, and zero samples above 1.

   The input domains are derived; the bands were asserted, so the bands were
   the falsified half. The ordinal structure is preserved and the scale is
   corrected by the factor of 10 that separated the asserted range from the
   derived one. **The equation is unchanged.** See
   [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) entry E1.

   A consequence worth stating plainly: because the core is multiplicative
   and each factor is ≤ 1, high M(S) is genuinely hard to reach. A system
   scoring 0.5 has all four factors strong *and* low loss. This is a
   property of the equation, not a calibration problem.

## Thermodynamic Interpretation

M(S) relates to system negentropy:

M(S) ∝ -ΔS_system + S_organized

Where:
- ΔS_system: Entropy change
- S_organized: Organized complexity

High M(S) systems maintain low entropy (high organization) while remaining adaptable.

## Information Theoretic View


M(S) ≈ I_mutual(components) - H_loss(system)

Where:
- I_mutual: Mutual information between components (resonance)
- H_loss: Information loss/waste

## Validation Criteria

A proposed M(S) measurement must:

1. Correlate with system longevity
2. Predict resilience to perturbation
3. Identify collapse precursors
4. Scale across domains (biology, social, technical)
5. Respect thermodynamic constraints

## Next Steps

Practical measurement procedures were planned for `04-measurement-methodology.md`, which was never written. Until it exists, the closest thing is `hgai.py::_assess_health_from_signals`, which maps text signals onto the five components — note that this mapping has a known, unresolved scaling defect recorded as entry E4 in [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md).
