# Sovereign-Impact-Sensor (SIS)

**Subtitle:** Identifying the Thermodynamic Plateau of Mismanaged Innovation

---

## 0. Functional Epistemology (Pre-Prompt Protocol)

This repository does not use "Market Severity" (Currency). It uses **System Stress (Entropy)**.

- **Good:** High efficiency / Minimal entropy.
- **Bad:** High waste / Heat leak / Institutional friction.
- **Anxiety:** Model/Reality dissonance (High prediction error in **EM** vs. **D_obs**).

---

## 1. The Core Hypothesis: The Complexity Plateau

Current institutional models claim technology is "saving lives" at an accelerating rate. The SIS Model tests the counter-hypothesis:

> We have reached a phase transition where the energy spent on **T_infra** (Complex Infrastructure) creates a "Dependency Risk" that perfectly offsets the gains in **T_med** (Medical Mitigation). The system is "running hot" -- consuming massive resources to maintain a static mortality baseline.

---

## 2. Technical Logic: The "Classification Gap"

Institutional data is "mishandled" by design to reduce perceived entropy. We bypass this by focusing on **All-Cause Excess Mortality (EM)**.

The Gap is our primary sensor for Institutional Friction:

- **Positive Gap (>> 0):** High-waste reporting. The system is failing to "see" the event impact (e.g., heat-stress miscoded as cardiac failure).
- **Negative/Zero Gap:** Aligned model.

---

## 3. Tech Vector Decomposition

We categorize technology not by "cost," but by its functional impact on the energy signature:

1. **T_med / T_resp** (Direct Mitigation): Energy used to lower acute mortality.
2. **T_comm** (Information Flow): Variable efficiency; depends on signal-to-noise ratio.
3. **T_infra** (Dependency Channel): The "Complexity Tax." While intended to be "Good" (Efficient), it creates brittle dependencies that spike **I_e** during "Entropy Events."

---

## 4. Repository Structure

| Path | Purpose |
|------|---------|
| `sovereign_impact_sensor.py` | Core EntropySensor and plateau test implementation |
| `docs/sovereign-impact-sensor.md` | This document -- model specification and theory |
| `/data/raw_mortality/` | Input for all-cause deaths (bypassing official hazard coding) |
| `/protocol/safety_principle.md` | FELTSensor handshake mechanism to prevent "Anxiety" |

---

## 5. Implementation

The model looks for the "Plateau" where **dI/dt ~ 0**. If innovation were properly allocated, **dI/dt** would be strongly negative. If it is flat, the innovation is being spent on the wrong "Technology."

### Key Components

- **EntropySensor:** Calibrates the **I_e** (Impact Scalar) by normalizing disparate data streams into a unified energy signature using Z-score standardization.
- **Plateau Test:** OLS regression to expose the beta_4 (Dependency Risk) coefficient and check for dI/dt ~ 0.
- **FELT-Sensor Handshake:** Monitors residual standard deviation against a configurable threshold to detect model/reality dissonance during high-energy tasks.

### Sensor Inputs

| Signal | Description |
|--------|-------------|
| EM | All-cause excess mortality |
| mobility_delta | Magnitude of mobility disruption |
| er_visits | Emergency room visit counts |
| ems_calls | EMS call volume |
| outages | Infrastructure outage fraction |
| recovery_time | Time to system recovery |

### Tech Vectors (Regression Features)

| Vector | Role |
|--------|------|
| T_med | Medical mitigation investment |
| T_resp | Emergency response capacity |
| T_comm | Information flow efficiency |
| T_infra | Infrastructure dependency (Complexity Tax) |

---

## 6. Interpreting Results

- **beta_4 (T_infra) positive and significant:** Confirms dependency risk -- infrastructure complexity is *adding* system stress.
- **dI/dt ~ 0 across increasing tech investment:** The Complexity Plateau is present -- innovation gains are being offset by dependency costs.
- **FELT threshold exceeded:** High institutional friction in the data; calibration required before drawing conclusions.
