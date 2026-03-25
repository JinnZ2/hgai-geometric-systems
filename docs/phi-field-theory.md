# Phi-Lattice Field Theory — Lyapunov-Filtered Vacuum Structure

## The Core Claim

The vacuum energy problem is resolved not by cancellation or fine-tuning, but by a **mode survival constraint**: only modes with Lyapunov exponent lambda ~ 0 physically persist.

```
E_vac = sum(all modes) hbar*omega  -->  sum(lambda~0) hbar*omega
```

Result: finite, dominated by a narrow critical band.

---

## I. Structured Substrate

Standard QFT places fields on a continuum:
```
phi(x,t),  x in R^3
```

This system uses a phi-octahedral lattice:
```
psi_i(t),  i in phi-scaled lattice
r_n = r0 * phi^n
```

The metric **emerges** from node spacing rather than being imposed.

---

## II. Effective Action

Hamiltonian:
```
H_ij = J_ij + V_i * delta_ij
```

Hopping term (encodes geometry + phase):
```
J_ij = J0 * exp(-d_ij / xi) * exp(i * (phi_i - phi_j))
```

On-site potential (confining):
```
V_i = V0 * (r_i / r_max)^2
```

---

## III. Continuum Limit

```
H  -->  -div(D(r) grad) + V(r) + i A(r) . grad
```

Where:
- D(r) ~ exp(-r/xi) : diffusion from coupling decay
- A(r) : effective gauge field from phase gradients
- Geometry encoded in spatial variation

---

## IV-V. Mode Decomposition + Lyapunov Filtering

Eigenspectrum:
```
H psi_n = omega_n psi_n
```

Dynamical evolution:
```
psi_{t+1} = K psi_t
psi_n(t) ~ exp(lambda_n * t)
lambda_n = ln|mu_n|
```

**Key redefinition:** Only modes with lambda_n ~ 0 persist.

---

## VI. Effective Density of States

Standard:
```
g(omega) ~ omega^2
```

Filtered:
```
g_eff(omega) = g(omega) * Theta(lambda(omega))

Theta(lambda) = 1 if |lambda| < epsilon, 0 otherwise
```

Result: `E_vac_eff = integral g_eff(omega) * hbar * omega d_omega` is **finite**.

---

## VII. Geometric Origin of Filtering

From phi-scaling:
```
lambda_n ~ -c * phi^n
```

- Large-scale modes (small n): lambda ~ 0 (survive)
- Small-scale modes (large n): lambda << 0 (filtered out)

High-frequency vacuum modes exist mathematically but cannot sustain amplitude physically.

---

## VIII. Topological Constraint

Edge modes in topological photonics have omega_topo with lambda ~ 0.

```
E_vac ~ sum(topological) hbar * omega
```

Vacuum energy = **topological spectral residue**.

---

## IX. Emergent Spacetime Metric

From coupling structure:
```
g_ij^{-1} ~ J_ij

ds^2 ~ exp(+r/xi) dr^2
```

Space is non-uniformly stretched. Outer regions are effectively "farther apart", which geometrically suppresses high-frequency modes.

---

## X. Cosmological Constant

```
Lambda_eff ~ <T_mu_nu>_filtered / volume
```

Since only lambda ~ 0 modes contribute:
```
Lambda_eff << Lambda_QFT
```

**Mechanism:** Not cancellation. Not fine-tuning. Mode survival constraint.

---

## Computational Results (7 shells, xi=2.0, epsilon=0.3)

```
Total modes:                42
Surviving modes:            9  (21.4%)

Vacuum energy (all modes):  10.726
Vacuum energy (filtered):   3.736
Suppression:                65%

Cosmological constant:      1.54e-04
```

Surviving modes are concentrated on **shells 0-2** (innermost), exactly where the organism model predicts the critical computation layer.

### Coherence Sweep

| xi  | E_full | E_filtered | Surviving | Suppression |
|-----|--------|-----------|-----------|-------------|
| 0.5 | 2.66   | 0.00      | 0/42      | 100%        |
| 1.0 | 5.94   | 0.00      | 0/42      | 100%        |
| 2.0 | 10.73  | 3.74      | 9/42      | 65%         |
| 4.0 | 16.54  | 6.24      | 14/42     | 62%         |
| 8.0 | 22.48  | 9.63      | 21/42     | 57%         |

---

## Usage

```python
from phi_field_theory import PhiFieldTheory, analytic_geff

# Full calculation
theory = PhiFieldTheory(n_shells=7, xi=2.0, epsilon=0.3)
results = theory.compute()
print(results.summary())

# Access individual quantities
print(f"Vacuum energy (filtered): {results.vacuum_energy_filtered}")
print(f"Cosmological constant: {results.cosmological_constant}")
print(f"Surviving modes: {results.n_surviving}/{results.n_total}")

# Analytic g_eff
import numpy as np
omega = np.linspace(0.01, 2.0, 100)
geff = analytic_geff(omega, n_shells=7)

# Parameter sweep
sweep = theory.parameter_sweep()
for xi, res in sweep:
    print(f"xi={xi}: {res.n_surviving} surviving, Lambda={res.cosmological_constant:.2e}")
```

---

## Next Directions

1. **Explicit g_eff(omega)**: Derive closed-form from phi-scaling (partially done via `analytic_geff()`)
2. **Curvature coupling**: Relate J_ij to Ricci scalar (implemented via `estimate_ricci_scalar()`)
3. **Dynamical spacetime**: Let lattice evolve -> emergent gravity analog
4. **Compare to known frameworks**: Holography, condensed matter analog gravity

---

## Physical Picture

Standard vacuum: infinite oscillators all active.

This system: infinite oscillators exist, but only a thin critical band is alive.

```
Vacuum = self-consistent attractor of the field
```

The vacuum energy problem is reframed as: **counting modes that cannot physically persist**.
