# Lyapunov Spectrum — Controllable Chaos Dynamics

## Core Equation

Perturbation growth:

```
delta(t) ~ delta_0 * exp(lambda * t)
```

- lambda > 0: exponential divergence (chaos)
- lambda = 0: neutral stability (edge of chaos)
- lambda < 0: convergence (stable system)

---

## 1. Lorenz System (Weather Reference)

The canonical chaotic system representing atmospheric dynamics:

```
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z
```

Standard parameters: sigma=10, rho=28, beta=8/3

Produces:
- Strange attractor (butterfly pattern)
- Positive Lyapunov exponent (~0.9)
- Predictability horizon of ~2 weeks for weather

**Key limitation:** Fixed physics, uncontrollable lambda.

---

## 2. Phi-Octahedral Lattice

Nodes on concentric shells at golden-ratio scaled radii:

```
r_n = r0 * phi^n
```

Coupling matrix:

```
K[i,j] = kappa0 * exp(-d_ij / xi) * exp(i * (phi_i - phi_j))
```

Lyapunov exponents from eigenvalues:

```
lambda_i = ln|mu_i|
```

**Key advantage:** Tunable lambda via three control knobs.

---

## 3. Control Knobs

### Coherence Length (xi)
- Higher xi: more modes near lambda ~ 0 (computation)
- Lower xi: stronger decay, more storage (memory)

### Phase Field (phi_i)
- Controls interference patterns
- Tunes eigenvalue distribution
- Direct control over which modes go chaotic

### Geometry (r0, shells)
- Controls spectral spacing
- Determines hierarchical depth
- Golden ratio spacing creates natural scale separation

### Topology (edge states)
- Injects guaranteed lambda ~ 0 modes
- Topological protection against perturbation

---

## 4. Three Regimes

| Regime | Lambda | Function | Analog |
|--------|--------|----------|--------|
| Stable | lambda < 0 | Memory storage, error correction | Crystalline archive |
| Critical | lambda ~ 0 | Max information transfer, optimal computation | Edge of chaos |
| Chaotic | lambda > 0 | Exploration, pattern generation, adaptation | Adaptive sensing |

---

## 5. Spatial Mode Mapping

The key insight (direction #2): mapping Lyapunov modes back onto physical space reveals where chaos vs memory physically live in the lattice.

```
Organism Structure:
  Core (inner shells, lambda < 0):   Stable memory crystal
  Shell (middle shells, lambda ~ 0): Computation layer
  Outer field (outer shells, lambda > 0): Exploratory sensing / adaptation
```

This is structurally identical to:
- **Brain:** stable + critical dynamics
- **Atmosphere:** structured chaos
- **Living systems:** core stability with adaptive periphery

---

## 6. Weather-Lattice Correspondence

| Weather System | Photonic Lattice |
|---------------|-----------------|
| State vector X | Optical field psi |
| Perturbation delta_X | Phase/amplitude perturbation |
| Jacobian J | Coupling matrix K |
| Lyapunov exponent lambda | Mode growth rate |
| Fixed physics | Tunable parameters |

---

## 7. System Identity (Compressed)

```
Lyapunov Spectrum = Geometry + Phase + Topology
```

```
System = Energy Flow + Nonlinearity + Constraint => lambda spectrum
```

---

## 8. Connection to HGAI Framework

The Lyapunov spectrum connects to the broader framework:

- **M(S) Score:** System at lambda ~ 0 (critical) has maximum coherence
- **Entropy Sensor:** lambda > 0 regions produce entropy; lambda < 0 absorb it
- **Flux Sensor:** Phase transitions correspond to eigenvalue crossings through zero
- **Resilience Scanner:** Institutional systems stuck at lambda < 0 (rigid) or lambda > 0 (chaotic) without the critical computation layer

---

## Usage

```python
from lyapunov_spectrum import (
    LorenzSystem,
    PhiOctahedralLattice,
    LyapunovAnalyzer,
)

# Lorenz reference
lorenz = LorenzSystem()
trajectory = lorenz.simulate(steps=10000)
exponents = lorenz.compute_lyapunov()

# Lattice analysis
lattice = PhiOctahedralLattice(shells=5, xi=2.0)
spectrum = lattice.compute_spectrum()
print(spectrum.summary())

# Spatial mode mapping (where does chaos live?)
spatial = lattice.compute_spatial_map()
print(spatial.summary())

# Tune the control knobs
lattice.tune_coherence(xi=4.0)   # more computation modes
lattice.tune_coherence(xi=0.5)   # more memory modes

# Coherence sweep
analyzer = LyapunovAnalyzer()
sweep = analyzer.sweep_coherence()
for xi, spec in sweep:
    print(f"xi={xi:.1f}: lambda_max={spec.max_exponent:.4f}")
```
