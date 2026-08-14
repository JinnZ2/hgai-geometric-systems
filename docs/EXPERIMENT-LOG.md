# Experiment Log

The method this project runs on:

> hypothesize → run → result falsified → edit claim → search for unknowns → rerun

This file is the record of that loop. Every entry keeps the original claim
intact, states what the run actually produced, and says what was changed in
response. **Nothing is quietly rewritten.** A claim that turned out wrong is
more useful than a claim that was never tested, and it stays visible so the
next person does not re-derive it from scratch.

Precedence carries. A superseded claim is not a deleted claim.

## How to use this log

- Before editing any published number, run `python verify_claims.py`.
- If a check fails, add an entry here *before* changing the prose.
- If you fix an `OPEN` finding, do not delete its entry — mark it CLOSED and
  say what the fix was.

Status vocabulary matches `verify_claims.py`:

| Status | Meaning |
|--------|---------|
| CONFIRMED | The run reproduces the published number |
| FALSIFIED | The run contradicts it; the claim has been revised |
| UNREACHABLE | The claim describes a state the code cannot enter |
| OPEN | A known defect, written up, not yet fixed |

---

## Round 1 — 2026-08-14

**Hypothesis under test:** the numbers published in `README.md`, `CLAUDE.md`
and `docs/` describe what the code does.

**Method:** install numpy, execute every standalone module, compare each
published figure against its live run.

**Setup:** Python 3, numpy 2.4.6, no optional dependencies.

**Headline result:** of the claims checked, four were falsified and two of
those were unreachable — describing states the code could not enter under
any input. The rest reproduced exactly.

---

### E1 — M(S) interpretation bands were mathematically unreachable

**Status:** FALSIFIED → claim revised.

**Claim as published** (`README.md`, `CLAUDE.md`, `docs/02-M-S-equation.md`):

| Score | Meaning |
|-------|---------|
| > 7 | Highly coherent and sustainable |
| 5–7 | Strong coherence, good viability |
| 3–5 | Moderate coherence, stable |
| 1–3 | Weak coherence, stressed |
| < 0 | Negative coherence, declining/collapse |

`docs/02-M-S-equation.md` additionally asserted **"Scale: Typical range
[-10, +10]"**.

**The run:**

```python
from framework.core.m_s_calculator import MSCalculator, SystemMetrics
MSCalculator.calculate(SystemMetrics(1.0, 1.0, 1.0, 1.0, 0.0))
# -> 1.0
```

**Why:** the same document derives every input domain from first principles
and bounds all four coherence factors to [0, 1] — R_e from normalized
cross-correlation, A from a ratio, D from Shannon entropy, C from
exploration/(exploration+exploitation). A product of four numbers each ≤ 1
cannot exceed 1. L is subtracted and non-negative. So

```
M(S) ∈ (-∞, +1]
```

Every band above 1 — four of the six published bands, including all three
"healthy" ones — sat above a ceiling the equation cannot reach. Sampling
100,000 uniform random inputs over the documented domains:

```
n=100000  min=-0.9997  max=0.7798  mean=-0.4388
fraction scoring > 1: 0.0000%
```

Not rare. Impossible.

**Corroborating signal:** the repository already published *two mutually
inconsistent* band tables — `docs/02-M-S-equation.md` said "> 5: highly
coherent," README and CLAUDE.md said "> 7." Two different unreachable
tables is what an untested assertion looks like.

**Which side was wrong.** The input domains are derived; the bands were
asserted. So the bands are the falsified half.

**Claim edited to:** the ordinal structure of the published bands was kept
and the scale corrected by the factor of 10 separating the asserted
"typical range [-10, +10]" from the derived range [-1, +1]:

| Score | Meaning |
|-------|---------|
| > 0.7 | Highly coherent and sustainable |
| 0.5–0.7 | Strong coherence, good viability |
| 0.3–0.5 | Moderate coherence, stable |
| 0.1–0.3 | Weak coherence, stressed |
| 0–0.1 | Low coherence, at risk |
| < 0 | Negative coherence, declining/collapse |

**The equation itself was not touched.** `M(S) = (R_e × A × D × C) - L`
stands exactly as derived. Only the interpretation of its output moved.

**Changed:** `framework/core/m_s_calculator.py` (bands + `M_S_CEILING`),
`README.md`, `CLAUDE.md`, `docs/02-M-S-equation.md`.

**Regression guard:** `check_m_s_ceiling`, `check_interpretation_bands_reachable`.

---

### E2 — Three of five HGAI health statuses were dead code

**Status:** UNREACHABLE → gates rescaled.

**Claim as published** (`README.md`): "Health Status: THRIVING / HEALTHY /
STRESSED / WARNING / CRITICAL".

**The run:** `hgai.py` gated `THRIVING` at `m_s_score > 5`, `HEALTHY` at
`> 3`, `STRESSED` at `> 1` — against a score whose ceiling is 1.0 (E1).

```
maximally healthy text  M(S)=  0.113  health=WARNING
neutral text            M(S)=  0.000  health=WARNING
friction-heavy text     M(S)= -0.899  health=CRITICAL

health statuses ever observed: ['CRITICAL', 'WARNING']
```

A source doing everything right — publishing raw data, stating uncertainty,
inviting replication — was reported as `WARNING`. The engine had two
reachable outputs and advertised five.

**Claim edited to:** gates rescaled to 0.5 / 0.3 / 0.1 to track the
corrected bands from E1. All five statuses are now reachable.

**Changed:** `hgai.py::_assess_health_from_signals`.

**Regression guard:** `check_health_statuses_reachable`, which sweeps the
geometry inputs directly rather than hunting for text that happens to land
in each band.

**Caveat — this fix is necessary but not sufficient.** It removed an
arithmetic impossibility. A second, independent defect (E4) still caps what
the engine produces in practice.

---

### E3 — "3/3 scenarios" was true by counting and false by measurement

**Status:** FALSIFIED → claim revised, threshold unified.

**Claim as published** (`README.md`): "The defect-aware weather model beats
conventional smoothing in all three test scenarios."

**The run:**

```
              scenario      smooth      defect    improve  verdict
     frontal_passage    40465.62    40465.29   +0.0008%      TIE
   cyclone_formation    40430.88    40430.07   +0.0020%      TIE
 blizzard_transition    40292.73    40292.49   +0.0006%      TIE
```

**Why the claim passed:** the code applied two different thresholds to the
same numbers. The per-scenario verdict used a ±5% materiality band and
correctly printed *"Models roughly equal. Defects neither helped nor hurt
significantly."* The summary counted a win at `improvement > 0` — any
positive float. So the same run simultaneously reported "roughly equal"
three times and "3/3 scenarios, consistently improves forecasts."

The reported improvement was `+0.0%` in the original output. Reading `+0.0%`
as a win is the tell. The formatter rounded the margin to nothing and the
counter still scored it.

**Claim edited to:** the defect-aware model is directionally better in 3/3
scenarios by a margin far below what this synthetic setup can resolve. The
sign is consistent; the magnitude is not evidence. **The claim that defect
preservation improves forecasts is UNTESTED here, not confirmed.** Deciding
it requires real observational data.

**Changed:** `defect_weather_model.py` — single `MATERIAL_THRESHOLD_PCT`
constant now used by both the verdict and the summary, margins printed to
4 decimal places so a rounding artifact cannot hide again, and a
`NOT SUPPORTED` conclusion branch for the all-ties case. Extracted
`run_scenario_comparison()` so the verifier measures the same run the demo
prints. `README.md` claim rewritten.

**Note on what was *not* touched:** the model's physics is unchanged. Only
the reporting of its results moved. The underlying `defect_field.py` result
(E5) is a genuine effect; it simply does not transfer to this weather setup
at a resolvable magnitude.

**Regression guard:** `check_defect_weather_margin` — fails if any margin
ever leaves the tie band, in either direction. If defect preservation starts
genuinely winning, this check breaks and the claim gets upgraded. That is
the point.

---

### E4 — The four geometry octants are not on a common scale

**Status:** OPEN. Reproduced, characterized, not fixed.

**This is the most consequential unknown found. It is a design question, not
a typo, and it belongs to the author.**

`hgai.py::_assess_health_from_signals` maps the four octant magnitudes onto
the four M(S) coherence factors:

| M(S) factor | Read from | Divisor |
|-------------|-----------|---------|
| R_e Resonance | agency magnitude | 1.5 |
| A Adaptability | temporal magnitude | 1.2 |
| D Diversity | valence ratio | — |
| C Curiosity | presence magnitude | 1.0 |

The divisors treat the four magnitudes as commensurable. They are not:

```
octant          min      max
agency       0.0000   1.4196
valence      0.5000   1.4240
temporal     0.0000   0.2000
presence     0.0100   0.0867
```

A 14× spread between the peaks. The cause is in `_encode_text_geometry`: the
agency and valence octants include binary 0/1 indicator dimensions
(`agency_vec[5]`, `agency_vec[6]`, `valence_vec[3]`, `valence_vec[4]`), which
push their vector norms to ~1.41. The temporal and presence octants are
built almost entirely from `count / n_tokens` fractions, which stay near
zero.

**Consequence:** in a *multiplicative* core, a factor pinned near zero
dominates the product. A and C are structurally starved, so `M(S)` is
effectively decided by R_e and D, and only ever rises above the floor
because of hardcoded rescue clauses (`adaptability = max(adaptability, 0.6)`,
`curiosity = max(curiosity, 0.4)`). Those clauses are load-bearing. Without
them the score would be ~0 for all input.

Worked example — text that is transparent, curious, and adaptive:

```
agency   magnitude=1.4174 ratio=1.00
valence  magnitude=1.4144 ratio=1.00
temporal magnitude=0.0000            <- text says "adapt", "changes"
presence magnitude=0.0900
raw mapping: R_e=0.945 A=0.000 D=0.300 C=0.090  product=0.0000
```

**A second, separable problem in the same mapping.** `D` (Diversity) is
computed as `max(1.0 - abs(valence_ratio), 0.3)`. Diversity in M(S) is
*pathway multiplicity*. Valence spread is *sentiment mixture*. Using one as
a proxy for the other means uniformly positive text — the transparent,
good-faith source — is scored as minimally diverse and floored at 0.3, while
text with mixed sentiment scores higher. The proxy is inverted with respect
to what the equation means by D.

**Why this was not fixed here:** rescaling the octants or re-deriving the
D proxy changes every score the engine has ever produced. E1 and E2 were
arithmetic impossibilities with one correct answer. This one has a modeling
decision inside it, and the framework's author should make it.

**Candidate directions, untested:**

1. Normalize each octant to a common range before mapping (per-octant
   min/max over a reference corpus, or L2-normalize each 16D vector).
2. Give the temporal and presence octants the same indicator dimensions the
   agency and valence octants have, so all four are built the same way.
3. Derive D from pathway/lexical diversity — type-token ratio, distinct
   causal chains, number of distinct actors — rather than from valence.
4. Remove the `max(..., 0.6)` / `max(..., 0.4)` rescue clauses once the
   scales are commensurable, and see what the engine reports without them.

**Regression guard:** `check_octant_scales_commensurable`. It reports OPEN
and does not fail the gate. When the scales are brought within 3× it flips
to CONFIRMED on its own and this entry can be marked CLOSED.

---

### E5 — Defect field improvement: CONFIRMED

**Status:** CONFIRMED. Reproduced exactly.

**Claim as published:** injecting topological defects improves convergence
by 24–28%, with 100% defect survival.

**The run** (`sweep_defect_configs(N=40, steps=200, seed=42)`):

| Configuration | loss | ratio | published | measured | survival |
|--------------|------|-------|-----------|----------|----------|
| control | 1589.67 | 1.0000 | — | — | 0% |
| single_vortex | 1211.81 | 0.7623 | −24% | −23.8% | 100% |
| dipole | 1213.27 | 0.7632 | −24% | −23.7% | 100% |
| quadrupole | 1136.74 | 0.7151 | −28% | −28.5% | 100% |
| random_cluster | 1164.37 | 0.7325 | −27% | −26.8% | 100% |

Every figure reproduces at the published rounding. This is the project's
strongest empirical result and it survived the audit intact.

**Unknown, worth recording:** the effect is established on a synthetic phase
field with a fixed seed. Seed sensitivity has not been characterized. Before
this is cited as a general result, it should be run across a seed sweep.

**Regression guard:** `check_defect_field_improvement`.

---

### E6 — Phi-lattice vacuum suppression: CONFIRMED

**Status:** CONFIRMED. Reproduced exactly.

**Claim as published:** 42 total modes, 9 survive (21%), vacuum energy
suppressed 65%, cosmological constant 1.5e-04.

**The run** (`PhiFieldTheory().compute()`):

```
Total modes: 42
Surviving modes (lambda ~ 0): 9          -> 21.4%
Vacuum energy (all modes):   10.7260
Vacuum energy (filtered):     3.7359     -> ratio 0.3483, suppression 65.2%
Cosmological constant:        1.543574e-04
```

All four figures confirmed.

---

### E7 — Detector template count was stale

**Status:** FALSIFIED → claim revised.

**Claim as published** (`README.md`, twice): "22 regex templates across 7
categories."

**The run:** `len(ALL_TEMPLATES)` → **24**. Categories → 7, correct.

Two templates were added without the count being updated. Small, but it is
the same failure mode as E1 and E3: a number in prose that no longer tracks
the code. This is why `verify_claims.py` exists.

**Claim edited to:** 24 templates across 7 categories.

**Regression guard:** `check_template_count`.

---

### E8 — README worked examples did not reproduce

**Status:** FALSIFIED → claim revised.

The two examples on the README front page — the first thing any reader runs
— produced different numbers than advertised.

| Input | Published | Actual |
|-------|-----------|--------|
| "The agency revised the methodology. This was an isolated incident." | Trust 0/100, 6 friction, 9 gaps | **Trust 18/100, 2 friction, 5 gaps** |
| "Ensemble models show 70% chance of 2-4 inches. Confidence moderate." | Trust 57/100 (MODERATE), 0 friction, 1 gap | **Trust 48/100 (LOW), 0 friction, 2 gaps** |

The README's JSON sample also showed `"transparency_count": 6` for the second
example; the actual value is 2.

**Why:** the README was rewritten (commit `0e8cbae`) *after* `audit.py` was
written (`ebc29e6`), and the example outputs were transcribed rather than
executed.

**Claim edited to:** all figures replaced with executed output.

**Regression guard:** `check_audit_friction_example`,
`check_audit_forecast_example`. These pin the README's front-page examples
to real runs, so the first thing a reader tries is the first thing that
breaks if it drifts.

---

### E9 — `hgai.py` documented five integrations it does not have

**Status:** FALSIFIED → claim revised.

**Claim as published** (`hgai.py` module docstring, and the "Connection to
Other Modules" table in `docs/hgai-engine.md`): HGAI connects the M(S)
Calculator, Resilience Scanner, Narrative Monitor, Ecological Monitor,
Three-Axis Protocol, Entropy Sensor, and Flux Sensor.

**The run** — actual imports in `hgai.py`:

```
framework.core.m_s_calculator
resilience.detectors
resilience.notices
```

Three of seven. The narrative, ecological and three-axis modules are not
imported; a subset of their *methodology* was reimplemented inline as
`_encode_text_geometry` and `_extract_curiosity_signals`. The entropy and
flux sensors are not called at all — `StressInput` is built and returned,
but nothing consumes it.

**Claim edited to:** the docstring and the doc table now separate three
categories — imported and executed, methodology reimplemented (originating
module in `legacy/`), and not wired in at all.

**Consequence for repository structure:** this is what made the `legacy/`
folder the right shape. See `legacy/README.md`.

---

### E10 — The AI consumption declaration was at an unreachable path

**Status:** FALSIFIED → fixed.

The file intended as `/.well-known/ai-consumption.txt` was committed to a
directory literally named `". well-known"` — leading dot, **space**, then
`well-known`. Verbatim from `git ls-files | cat -A`:

```
. well-known/ai-consumption.txt$
```

No crawler, agent, or convention-following tool would ever find it. A
declaration of AI training permissions that cannot be fetched is a
declaration that does not exist. Fixed via `git mv` to `.well-known/`.

**Unknown:** whether anything ever cited the broken path.

---

### E11 — Four of twelve documented class names did not exist

**Status:** FALSIFIED → claim revised.

`CLAUDE.md` published a "Key Modules" table naming the classes an AI
assistant should expect in each file. Checked against `grep -rln "class X"`:

| Documented class | Exists |
|-----------------|--------|
| `FractureDetector` | **no** |
| `GeometricMonitor` | **no** |
| `UnifiedMonitor` | **no** |
| `EcologicalHealthAssessment` | **no** |
| `GeometryPacket` | yes |
| `ThreeAxisProtocol`, `ThreeAxisAI`, `InvestigationAxis` | yes |
| `RelationalObservation` | yes |
| `SystemMetrics`, `MSCalculator`, `TimeSeriesAnalyzer` | yes (three copies — see `legacy/README.md`) |

The real names are `UnifiedFieldMonitor`, `UnifiedConsciousnessMonitor` and
`RelationalHealthMonitor`. `FractureDetector` does not correspond to
anything in the repository at all.

This matters more than an ordinary typo: `CLAUDE.md` exists specifically to
orient AI assistants, and it was sending them to look for symbols that were
never there.

**Claim edited to:** table corrected against `grep`, and rewritten to
reflect the post-reorganization layout.

---

## Open questions carried forward

Recorded so they are not rediscovered from zero.

1. **E4 — octant scale normalization.** The blocking issue for HGAI scoring.
   Four candidate directions listed above, none tested.
2. **E4b — the Diversity proxy.** `D` from valence spread is inverted with
   respect to pathway multiplicity. Needs a real derivation.
3. **E5 seed sensitivity.** The 24–28% defect result is one seed. Sweep it.
4. **E3 real data.** The defect weather claim cannot be settled on synthetic
   scenarios at this resolution. Needs observational input.
5. **Unconsumed bridges.** `sovereign_impact_sensor.py`, `flux_sensor.py`
   and `weather_node_network.py` are documented as connected and are not
   called. `StressInput` is the intended handoff and is currently a
   dead-end output. Either wire them or say plainly that they are standalone.
6. **`sovereign_impact_sensor.py` cannot run** in the documented minimal
   install — it imports pandas, scikit-learn and statsmodels at module
   level, so it fails immediately under "pip install numpy". Either guard
   the imports or state the requirement at the top of the file.
7. **The `legacy/` modules carry capability that has no successor.**
   `FractureDetector`, `compute_curvature`, the observer-metric machinery
   and the full three-axis protocol exist nowhere else. Listed in
   `legacy/README.md`. They were demoted for being unreferenced, not for
   being wrong.
8. **`3-axis.md` references `three_axis_protocol.py` "(to be built)"** —
   but `three-axis.py` exists and implements it. Either the doc predates
   the implementation or they diverged.
9. **M(S) is reimplemented inline in all three example notebooks.** On top
   of the three module-level copies (see `legacy/README.md`), each of
   `examples/notebooks/*.md` defines its own `calculate_m_s()` and
   `interpret_m_s()`. Their bands were rescaled with everything else in E1,
   but they will drift again — they are copies, not imports. The notebooks
   should import from `framework.core.m_s_calculator`.

---

## Method notes

Things this round taught, worth keeping:

- **A claim that cannot fail is not a result.** E3 passed for months because
  its success condition was `> 0` on a float. Any threshold that a rounding
  artifact can satisfy will eventually be satisfied by one.
- **Check reachability, not just correctness.** E1 and E2 were not wrong
  arithmetic — the arithmetic was fine. The bands described states the
  system could not enter. Testing "does it compute the right answer" would
  never have caught it; testing "can this branch ever be taken" caught both
  immediately.
- **Internal contradiction is the cheapest signal available.** In E3 the
  code printed "roughly equal" and "3/3 wins" in the same run. In E1 the
  repository published two different unreachable band tables. Nobody had to
  run anything to notice either — they just had to be read side by side.
- **Transcribed output drifts; executed output does not.** E7 and E8 are both
  numbers that were correct when typed and were never re-run.
- **Prefer editing the claim to editing the model.** E1 rescaled the
  interpretation and left the equation alone. E3 fixed the reporting and
  left the physics alone. When a run falsifies a claim, the claim is usually
  what is wrong.
