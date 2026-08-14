# legacy/

**Precedence carries.** Nothing here was deleted, deprecated, or judged
wrong. These are the modules the current engine was built out of. They still
run, they still hold ideas that have no successor, and they are the record
of where the framework's reasoning came from.

A file is in this folder for exactly one reason: **nothing in the active
import graph references it.** That is a fact about wiring, not about worth.

```bash
# every module here still runs standalone, unchanged
python legacy/three-axis.py
python legacy/ecological-calculus.py
python legacy/unified_field_monitor.py
python legacy/Unified_narrative.py
```

---

## Why these four

`hgai.py` is the front door of the current framework. Its module docstring
claimed to connect seven modules. It imports three:

```
framework.core.m_s_calculator
resilience.detectors
resilience.notices
```

The narrative, field, ecological and three-axis modules were not imported.
A subset of their *methodology* was reimplemented inline as
`_encode_text_geometry` and `_extract_curiosity_signals`. That gap — between
what the engine documented and what it actually loaded — is what defines
this folder. See `docs/EXPERIMENT-LOG.md` entry E9.

| Module | Generation | Superseded in the active path by | Still unique to it |
|--------|-----------|----------------------------------|--------------------|
| `Unified_narrative.py` | Nov 2025 | `hgai._encode_text_geometry` (64D octant encoding) | observer metrics, geometric coherence between two parties, octahedral projection |
| `unified_field_monitor.py` | Nov 2025 | `hgai._encode_text_geometry` + `resilience.detectors` | trajectory curvature, field-observation encoding, attractor tracking |
| `three-axis.py` | Nov 2025 | `hgai._extract_curiosity_signals` (a much smaller subset) | the full investigation protocol, hypothesis structure, `ThreeAxisAI` |
| `ecological-calculus.py` | Nov 2025 | `hgai._assess_health_from_signals` (health framing only) | relational observation model, reciprocity accounting |

`3-axis.md` moved with `three-axis.py` — it documents that protocol and
belongs beside it.

---

## What is genuinely not reproduced anywhere else

If any of the following is needed, it is here and only here. Re-promoting a
module is one `git mv`.

**`Unified_narrative.py`**
- `build_observer_metric_from_profile()` / `default_observer_metric()` — a
  metric tensor for the 64D space, so distance can be measured from a
  specified observer's frame rather than assumed Euclidean.
- `compute_geometric_coherence()` — coherence between self, other and field
  vectors. The current engine encodes one text at a time and has no
  two-party comparison at all.
- `octahedral_projection_summary()` / `ascii_octahedral_line()` — the
  octahedral projection and its terminal rendering.
- `angle_between()` with an optional metric argument.

**`unified_field_monitor.py`**
- `compute_curvature()` — curvature of a trajectory through the geometric
  space, i.e. how sharply a system's state is turning. `hgai` is stateless
  across calls and computes nothing over time.
- `field_observation_to_geometry()` — encodes structured field observations
  rather than free text.
- `FieldAssessment` / `UnifiedFieldMonitor` — attractor and history tracking.

**`three-axis.py`**
- `ThreeAxisProtocol` with `AxisHypothesis` and `ConfusionInvestigation` —
  the full hypothesize/test/revise structure across three investigation
  axes. `hgai._extract_curiosity_signals` emits investigation *leads*; it
  does not run investigations.
- `ThreeAxisAI` — the agent-facing wrapper.
- `demonstrate_pencil_example()` — the worked example the protocol was
  derived from.

**`ecological-calculus.py`**
- `RelationalObservation` / `RelationalHealthMonitor` — health as a property
  of relationships between entities rather than of an entity. This framing
  does not exist in the current engine.

---

## The M(S) divergence that made the demotion unambiguous

Three separate implementations of the M(S) equation existed simultaneously,
and they disagreed. Same inputs `(R_e=0.8, A=0.7, D=0.9, C=0.4, L=0.6)`:

| Implementation | M(S) | Interpretation | Ceiling |
|---------------|------|----------------|---------|
| `framework/core/m_s_calculator.py` | **−0.3984** | Severe negative coherence, collapse imminent | 1.0 |
| `legacy/Unified_narrative.py` | **+0.6096** | Low coherence, at risk | 3.375 |
| `legacy/unified_field_monitor.py` | **−0.3984** | Declining | 1.0 |

`Unified_narrative.py` uses a different formula —
`R_e × (0.5+A) × (0.5+D) × (0.5+C) − L` — with 0.5 offsets that break the
multiplicative core's defining property: that a zero in any factor zeroes
the whole product. Under the offsets, a system with zero adaptability,
zero diversity and zero curiosity still scores positive. `unified_field_monitor.py`
uses the correct formula but a third set of interpretation bands.

**`framework/core/m_s_calculator.py` is canonical.** It implements the
equation as derived in `docs/02-M-S-equation.md`. The other two are
retained as precedent — they show which variants were tried — but must not
be imported for scoring.

The canonical implementation had its own falsified claim, unrelated to this
divergence: its interpretation bands were unreachable. See experiment log
entry E1. The bands were corrected; the equation was not touched.

---

## Considered and deliberately left in the active tree

Recorded so the reasoning does not have to be reconstructed.

| Module | Unreferenced? | Kept active because |
|--------|--------------|---------------------|
| `sovereign_impact_sensor.py` | yes | Named as the `StressInput` consumer. The bridge is designed and documented; only the call is missing. Demoting it would break a documented integration path rather than record a superseded one. |
| `flux_sensor.py` | imported by `chaos_weather_ai.py` | In the active graph. |
| `weather_node_network.py` | yes | Same as the impact sensor — an intended endpoint of the current pipeline, not a predecessor of it. |
| `phase_field_optimizer.py` | yes | Standalone theory result with no successor. Not superseded by anything. |
| `phi_field_theory.py` | yes | Same. Its published figures verify exactly (log E6). |

The distinction being applied: **legacy means superseded, not merely
unwired.** A module with a successor in the active path belongs here. A
module that is simply not called yet does not.

---

## Rules for this folder

1. **Do not delete anything here.** The record is the point.
2. **Do not import from `legacy/` in active code.** If something here is
   needed, promote it back out deliberately — do not create a second
   dependency on a superseded implementation. That is how the three
   divergent M(S) calculators happened.
3. **These modules must keep running.** They are executable precedent. If a
   change breaks one, that is a finding worth logging.
4. **Moving something here requires a log entry** in
   `docs/EXPERIMENT-LOG.md` saying what superseded it and what capability
   it still uniquely holds.
