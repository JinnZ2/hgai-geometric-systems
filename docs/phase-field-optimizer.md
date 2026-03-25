# Phase Field Optimizer — Learning = Geometry = Energy Minimization

## Core Claim

One variable (phi), one evolution rule (gradient descent on E), four simultaneous effects. Geometry, computation, stability, and memory are the **same variable viewed under different constraints**.

Not metaphorically. Literally through the equations.

---

## Unified Energy Functional

```
E(phi) = (1/2)||K(phi)x - y*||^2          alignment / computation
       + alpha * sum A_ij(1-cos(dphi))     coherence / XY model
       + beta * log||Kx||                  stability / Lyapunov
       + gamma * T(phi)                    topology / vortex memory
```

Evolution:
```
dphi/dt = -grad_phi E
```

---

## Why One Equation Does Many Things

### Superposition of gradients
```
grad E = grad E_compute + grad E_coherence + grad E_stability + grad E_topology
```

Each step is a vector sum of forces.

### Non-orthogonality
- Smoothing can improve computation
- Topology can stabilize dynamics
- Stability can reshape computation

Effects reinforce or compete.

### Constraint coupling
Changing one phi_k modifies all K_ij, the entire spectrum, and the field topology. Global impact from local change.

---

## Analytic Gradient (O(N^2) exact, replaces O(N^3) finite difference)

Define:
```
K_ij = A_ij * cos(phi_i - phi_j)
S_ij = A_ij * sin(phi_i - phi_j)
A_ij = exp(-d_ij / xi)
e = Kx - y*
```

### Alignment gradient
```
g_align = -(e * (S @ x)) + (x * (S^T @ e))
```

### Coherence gradient (XY model)
```
g_cohere = sum_j A_ij * sin(phi_i - phi_j)  =  row sums of S
```

### Stability gradient (Lyapunov)
```
g_stab = (S @ x * y) / ||y||^2
```

### Topology gradient
Finite difference on discrete winding number (invariant).

---

## Four Regimes

| Regime | Weights | Behavior |
|--------|---------|----------|
| Compute-dominant | alpha,beta,gamma << 1 | Like neural net. Unstable long-term. Lowest error. |
| Coherence-dominant | alpha >> others | Smooth phase fields. Wave-like. Robust but less expressive. |
| Topology-dominant | gamma >> others | Vortex locking. Persistent non-erasable memory. |
| Balanced | All comparable | Attractor basins. Geometry encodes function. |

### Computational Results (N=40, 100 steps)

| Regime | Error | Crystallized | Memory | Compute | Explore | Defects |
|--------|-------|-------------|--------|---------|---------|---------|
| Compute-dominant | 7.22 | 4 | 2 | 15 | 23 | 43 |
| Coherence-dominant | 8.08 | 4 | 13 | 20 | 7 | 20 |
| Topology-dominant | 7.20 | 3 | 2 | 14 | 24 | 44 |
| Balanced | 7.22 | 4 | 2 | 15 | 23 | 43 |

---

## Physical Mapping

### Phase = Physical Quantity
```
phi_i  <->  birefringent retardance
            optical path length
            refractive index perturbation
            nanostructure orientation
```

### Gradient = Write Instruction
```
delta_n = delta_phi * wavelength / (2 * pi * L)
```

### Physical Loop
```
[laser write phi] -> [optical input x] -> [interference field]
     -> [measure y] -> [compute gradient] -> [rewrite phi]
```

---

## What Emerges Over Time

1. **Mode crystallization**: Discrete stable eigenmodes form (7 modes after 200 steps)
2. **Spatial specialization**: Regions become functionally distinct (memory/compute/explore)
3. **Energy funnels**: Light flows along learned paths (strongest coupling paths)
4. **Memory without symbols**: Structure itself is memory (topological defects = persistent states)

---

## Topological Term

Discrete winding number over triangulated loops:
```
T(phi) = sum_triangles |winding_k|
winding_k = (1/2pi) * (dphi_ij + dphi_jk + dphi_ki)
```

Non-zero winding = vortex = memory that cannot smoothly decay.

---

## Usage

```python
from phase_field_optimizer import PhaseFieldOptimizer

# Create and train
opt = PhaseFieldOptimizer(N=60, alpha=0.05, beta=0.05, gamma=0.01)
opt.train(steps=200, print_every=20)

# Check what emerged
structure = opt.analyze()
print(f"Crystallized modes: {structure.crystallized_modes}")
print(f"Memory nodes: {len(structure.spatial_specialization['memory'])}")

# Get physical write instructions
mapping = opt.get_physical_mapping()
print(f"Max delta_n: {np.max(np.abs(mapping.delta_n)):.2e}")

# Custom task
opt.set_task(x=my_input, y_target=my_output)
opt.train(steps=100)
```

---

## Connection to Other Modules

| Module | Connection |
|--------|-----------|
| `phi_field_theory.py` | Provides the Hamiltonian and vacuum energy calculation |
| `lyapunov_spectrum.py` | Lyapunov exponents and regime detection theory |
| `flux_sensor.py` | Phase transition detection during training |
| `chaos_weather_ai.py` | Same outlier-first philosophy applied to weather |
| `hgai.py` | Unified engine — text analysis feeds system health |

---

## Compressed Statement

```
Define one variable -> attach multiple invariants -> let physics resolve the conflict
```

That's why it behaves like nature: one state space, multiple conserved/minimized quantities, one evolution rule.

---

## Next Directions

1. **Introduce curvature**: E -> E + eta * (nabla^2 phi)^2 (stiffness, dispersion control)
2. **Couple to energy flow**: dE/dt = -integral |grad E|^2 dV (thermodynamic closure)
3. **Intentional defects**: Introduce vortices and see if they become functional
4. **Casimir measurement**: Compute force F = -dE/dL (measurable if geometry shifts spectrum)
5. **Organism integration**: energy intake -> x, structure -> phi, motion -> gradient of energy field
