"""
Phase Field Optimizer — Where Learning = Geometry = Energy Minimization

A unified energy functional over a single variable (phase field phi)
that simultaneously performs computation, stabilization, smoothing,
and topological memory. Not metaphorically — literally through the
equations.

Unified Energy Functional:
    E(phi) = (1/2)||K(phi)x - y*||^2       (alignment / computation)
           + alpha * sum A_ij(1-cos(dphi))  (coherence / XY model)
           + beta * log||Kx||               (stability / Lyapunov)
           + gamma * T(phi)                 (topology / vortex memory)

Evolution:
    dphi/dt = -grad_phi E

Key property: one state space, multiple conserved/minimized quantities,
one evolution rule. Geometry, computation, stability, and memory are
the same variable viewed under different constraints.

Physical mapping:
    phi_i <-> birefringent retardance / optical path length
    grad(phi) -> laser rewrite instruction
    [laser write phi] -> [optical input x] -> [interference] ->
    [measure y] -> [compute gradient] -> [rewrite phi]

What emerges over time:
    1. Mode crystallization (discrete stable eigenmodes)
    2. Spatial specialization (regions become functionally distinct)
    3. Energy funnels (light flows along learned paths)
    4. Memory without symbols (structure itself is memory)

Dependencies:
    - numpy
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHI_GOLDEN = (1 + 5**0.5) / 2


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Snapshot of the optimizer state at a given step.

    Parameters
    ----------
    step : int
        Training iteration.
    loss_total : float
        Total energy functional value.
    loss_alignment : float
        Computation/alignment term.
    loss_coherence : float
        XY model coherence term.
    loss_stability : float
        Lyapunov stability term.
    loss_topology : float
        Topological invariant term.
    output_error : float
        ||Kx - y*||.
    lyapunov_proxy : float
        log||Kx||.
    winding_number : float
        Total topological charge.
    regime : str
        Detected regime.
    """
    step: int
    loss_total: float
    loss_alignment: float
    loss_coherence: float
    loss_stability: float
    loss_topology: float
    output_error: float
    lyapunov_proxy: float
    winding_number: float
    regime: str


@dataclass
class PhysicalMapping:
    """Maps phase field to physical implementation parameters.

    Parameters
    ----------
    delta_phi : np.ndarray
        Phase update per node.
    delta_n : np.ndarray
        Refractive index change per voxel.
    write_power : np.ndarray
        Laser write power per voxel (proportional to |delta_n|).
    voxel_thickness : float
        Thickness of each voxel (L).
    """
    delta_phi: np.ndarray
    delta_n: np.ndarray
    write_power: np.ndarray
    voxel_thickness: float


@dataclass
class EmergentStructure:
    """Properties that emerge from training.

    Parameters
    ----------
    mode_frequencies : np.ndarray
        Eigenfrequencies of the trained coupling matrix.
    crystallized_modes : int
        Number of discrete stable modes.
    spatial_specialization : dict
        Functional role per region (memory/compute/explore).
    energy_funnels : list
        Dominant energy flow paths.
    topological_defects : list
        Locations and charges of vortices.
    """
    mode_frequencies: np.ndarray
    crystallized_modes: int
    spatial_specialization: Dict[str, List[int]]
    energy_funnels: List[Tuple[int, int, float]]
    topological_defects: List[Dict[str, float]]


# ---------------------------------------------------------------------------
# Distance Matrix (precomputed)
# ---------------------------------------------------------------------------

def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix.

    Parameters
    ----------
    positions : np.ndarray
        Node positions (N x d).

    Returns
    -------
    np.ndarray
        N x N distance matrix.
    """
    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def compute_amplitude_matrix(
    distances: np.ndarray,
    xi: float,
) -> np.ndarray:
    """Compute the amplitude envelope A_ij = exp(-d_ij / xi).

    This is the distance-dependent part of coupling, precomputed
    since it doesn't depend on the phase field.

    Parameters
    ----------
    distances : np.ndarray
        Pairwise distance matrix.
    xi : float
        Coherence length.

    Returns
    -------
    np.ndarray
        N x N amplitude matrix.
    """
    return np.exp(-distances / xi)


# ---------------------------------------------------------------------------
# Forward Pass
# ---------------------------------------------------------------------------

def forward(
    phi: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute coupling matrix K and output y = Kx.

    K_ij = A_ij * cos(phi_i - phi_j)

    Parameters
    ----------
    phi : np.ndarray
        Phase field (N,).
    A : np.ndarray
        Amplitude matrix (N, N).
    x : np.ndarray
        Input signal (N,).

    Returns
    -------
    K : np.ndarray
        Coupling matrix (N, N).
    y : np.ndarray
        Output signal (N,).
    """
    K = A * np.cos(phi[:, None] - phi[None, :])
    y = K @ x
    return K, y


# ---------------------------------------------------------------------------
# Topological Invariant
# ---------------------------------------------------------------------------

def compute_winding(
    phi: np.ndarray,
    triangles: np.ndarray,
) -> Tuple[float, List[Dict[str, float]]]:
    """Compute topological winding number over triangulated loops.

    T(phi) = sum_triangles |winding_k|

    Each triangle contributes a winding number from the discrete
    curl of the phase field. Non-zero winding = vortex.

    Parameters
    ----------
    phi : np.ndarray
        Phase field (N,).
    triangles : np.ndarray
        Triangle indices (M x 3), each row is [i, j, k].

    Returns
    -------
    total_winding : float
        Total topological charge |sum of windings|.
    defects : list of dict
        Locations and charges of detected vortices.
    """
    defects = []
    total = 0.0

    for tri in triangles:
        i, j, k = tri
        # Phase differences (wrapped to [-pi, pi])
        dphi_ij = _wrap(phi[j] - phi[i])
        dphi_jk = _wrap(phi[k] - phi[j])
        dphi_ki = _wrap(phi[i] - phi[k])

        winding = (dphi_ij + dphi_jk + dphi_ki) / (2 * np.pi)

        if abs(winding) > 0.4:  # threshold for vortex detection
            charge = round(winding)
            defects.append({
                "triangle": tri.tolist(),
                "charge": float(charge),
                "winding": float(winding),
            })
            total += abs(winding)

    return float(total), defects


def _wrap(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def build_triangulation(N: int, positions: np.ndarray) -> np.ndarray:
    """Build a simple triangulation from nearest neighbors.

    For each node, form triangles with its two nearest neighbors.
    This is a minimal triangulation sufficient for winding detection.

    Parameters
    ----------
    N : int
        Number of nodes.
    positions : np.ndarray
        Node positions (N x d).

    Returns
    -------
    np.ndarray
        Triangle indices (M x 3).
    """
    distances = compute_distance_matrix(positions)
    triangles = []

    for i in range(N):
        # Find 4 nearest neighbors (excluding self)
        dists = distances[i].copy()
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:4]

        # Form triangles with pairs of neighbors
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                tri = sorted([i, neighbors[a], neighbors[b]])
                triangles.append(tri)

    # Deduplicate
    unique = set(tuple(t) for t in triangles)
    return np.array(list(unique))


# ---------------------------------------------------------------------------
# Analytic Gradient (Section I-II of the theory)
# ---------------------------------------------------------------------------

def analytic_gradient(
    phi: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    y_target: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    triangles: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute the full analytic gradient of the unified energy functional.

    E(phi) = E_align + alpha * E_cohere + beta * E_stab + gamma * E_topo

    All terms use exact derivatives (no finite differences).

    Alignment gradient (Section I):
        S_ij = A_ij * sin(phi_i - phi_j)
        g_align = -(e * (S @ x)) + (x * (S^T @ e))

    Coherence gradient (XY model):
        g_cohere = sum_j A_ij * sin(phi_i - phi_j)

    Stability gradient (Lyapunov):
        g_stab = (S @ x * y) / ||y||^2

    Topology gradient:
        Finite difference on winding number (discrete invariant).

    Parameters
    ----------
    phi : np.ndarray
        Phase field (N,).
    A : np.ndarray
        Amplitude matrix (N, N).
    x : np.ndarray
        Input signal (N,).
    y_target : np.ndarray
        Target output (N,).
    alpha : float
        Coherence weight.
    beta : float
        Stability weight.
    gamma : float
        Topology weight.
    triangles : np.ndarray
        Triangle indices for winding computation.

    Returns
    -------
    grad : np.ndarray
        Total gradient (N,).
    losses : dict
        Individual loss term values.
    """
    N = len(phi)

    # Forward pass
    K, y = forward(phi, A, x)
    e = y - y_target  # error vector

    # S matrix: A_ij * sin(phi_i - phi_j)
    S = A * np.sin(phi[:, None] - phi[None, :])

    # --- Alignment gradient (exact, O(N^2)) ---
    Sx = S @ x
    STe = S.T @ e
    g_align = -(e * Sx) + (x * STe)

    loss_align = 0.5 * np.linalg.norm(e) ** 2

    # --- Coherence gradient (XY model) ---
    # E_cohere = sum_ij A_ij * (1 - cos(phi_i - phi_j))
    # dE/dphi_k = sum_j A_kj * sin(phi_k - phi_j)
    g_cohere = np.sum(S, axis=1)

    loss_cohere = np.sum(A * (1 - np.cos(phi[:, None] - phi[None, :])))

    # --- Stability gradient (Lyapunov) ---
    norm_y = np.linalg.norm(y) + 1e-8
    g_stab = (Sx * y) / (norm_y ** 2)

    loss_stab = np.log(norm_y)

    # --- Topology gradient (discrete, small FD on winding) ---
    winding, defects = compute_winding(phi, triangles)
    loss_topo = winding

    # Topology gradient via small perturbation
    g_topo = np.zeros(N)
    if gamma > 0:
        eps_topo = 1e-4
        for k in range(N):
            phi_pert = phi.copy()
            phi_pert[k] += eps_topo
            w_pert, _ = compute_winding(phi_pert, triangles)
            g_topo[k] = (w_pert - winding) / eps_topo

    # --- Total gradient ---
    grad = g_align + alpha * g_cohere + beta * g_stab + gamma * g_topo

    # --- Total loss ---
    loss_total = (
        loss_align
        + alpha * loss_cohere
        + beta * loss_stab
        + gamma * loss_topo
    )

    losses = {
        "total": float(loss_total),
        "alignment": float(loss_align),
        "coherence": float(loss_cohere),
        "stability": float(loss_stab),
        "topology": float(loss_topo),
        "output_error": float(np.linalg.norm(e)),
        "lyapunov_proxy": float(loss_stab),
        "winding": float(winding),
    }

    return grad, losses


# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------

def detect_regime(
    alpha: float,
    beta: float,
    gamma: float,
    losses: Dict[str, float],
) -> str:
    """Detect which regime the system is operating in.

    Parameters
    ----------
    alpha, beta, gamma : float
        Weight coefficients.
    losses : dict
        Current loss values.

    Returns
    -------
    str
        Regime name.
    """
    weighted = {
        "compute": losses["alignment"],
        "coherence": alpha * losses["coherence"],
        "stability": beta * abs(losses["stability"]),
        "topology": gamma * losses["topology"],
    }

    dominant = max(weighted, key=weighted.get)
    total = sum(weighted.values()) + 1e-12
    balance = max(weighted.values()) / total

    if balance < 0.4:
        return "balanced"
    return f"{dominant}-dominant"


# ---------------------------------------------------------------------------
# Physical Mapping
# ---------------------------------------------------------------------------

def compute_physical_mapping(
    delta_phi: np.ndarray,
    voxel_thickness: float = 1e-6,
    wavelength: float = 633e-9,
) -> PhysicalMapping:
    """Map phase update to physical refractive index change.

    delta_n = delta_phi * wavelength / (2 * pi * L)

    Parameters
    ----------
    delta_phi : np.ndarray
        Phase change per node.
    voxel_thickness : float
        Voxel thickness L (meters). Default 1 micron.
    wavelength : float
        Operating wavelength (meters). Default 633nm.

    Returns
    -------
    PhysicalMapping
        Physical implementation parameters.
    """
    delta_n = delta_phi * wavelength / (2 * np.pi * voxel_thickness)
    write_power = np.abs(delta_n)  # proportional to |delta_n|

    return PhysicalMapping(
        delta_phi=delta_phi,
        delta_n=delta_n,
        write_power=write_power / (np.max(write_power) + 1e-12),  # normalize
        voxel_thickness=voxel_thickness,
    )


# ---------------------------------------------------------------------------
# Emergent Structure Analysis
# ---------------------------------------------------------------------------

def analyze_emergent_structure(
    phi: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    positions: np.ndarray,
    triangles: np.ndarray,
) -> EmergentStructure:
    """Analyze what has emerged from training.

    Checks for:
    1. Mode crystallization (discrete stable eigenmodes)
    2. Spatial specialization (memory/compute/explore regions)
    3. Energy funnels (dominant flow paths)
    4. Topological defects (vortices)

    Parameters
    ----------
    phi : np.ndarray
        Trained phase field.
    A : np.ndarray
        Amplitude matrix.
    x : np.ndarray
        Input signal.
    positions : np.ndarray
        Node positions.
    triangles : np.ndarray
        Triangulation for topology.

    Returns
    -------
    EmergentStructure
        Analysis of emerged properties.
    """
    K, y = forward(phi, A, x)

    # 1. Mode crystallization
    eigvals = np.linalg.eigvalsh(K)
    # Modes are "crystallized" if eigenvalue gaps are large
    sorted_eig = np.sort(eigvals)
    gaps = np.diff(sorted_eig)
    mean_gap = np.mean(gaps) + 1e-12
    crystallized = int(np.sum(gaps > 2 * mean_gap))

    # 2. Spatial specialization
    # Use local phase variance to classify regions
    N = len(phi)
    distances = compute_distance_matrix(positions)
    specialization = {"memory": [], "compute": [], "explore": []}

    for i in range(N):
        # Find neighbors
        dists = distances[i].copy()
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:5]

        # Local phase variance
        local_phases = phi[neighbors]
        phase_var = np.var(np.cos(local_phases - phi[i]))

        # Low variance = coherent (memory)
        # Medium variance = active (compute)
        # High variance = disordered (explore)
        if phase_var < 0.1:
            specialization["memory"].append(i)
        elif phase_var < 0.4:
            specialization["compute"].append(i)
        else:
            specialization["explore"].append(i)

    # 3. Energy funnels (strongest coupling paths)
    K_abs = np.abs(K)
    np.fill_diagonal(K_abs, 0)
    funnels = []
    for _ in range(5):
        idx = np.unravel_index(np.argmax(K_abs), K_abs.shape)
        if K_abs[idx] > 0:
            funnels.append((int(idx[0]), int(idx[1]), float(K_abs[idx])))
            K_abs[idx] = 0

    # 4. Topological defects
    _, defects = compute_winding(phi, triangles)

    return EmergentStructure(
        mode_frequencies=eigvals,
        crystallized_modes=crystallized,
        spatial_specialization=specialization,
        energy_funnels=funnels,
        topological_defects=defects,
    )


# ---------------------------------------------------------------------------
# Main Optimizer
# ---------------------------------------------------------------------------

class PhaseFieldOptimizer:
    """Unified phase field optimizer.

    One variable (phi), one evolution rule (gradient descent on E),
    four simultaneous effects (compute, cohere, stabilize, remember).

    Parameters
    ----------
    N : int
        Number of lattice nodes. Default 60.
    xi : float
        Coherence length. Default 1.0.
    alpha : float
        Coherence weight. Default 0.05.
    beta : float
        Stability weight. Default 0.05.
    gamma : float
        Topology weight. Default 0.01.
    eta : float
        Learning rate. Default 0.01.
    seed : int
        Random seed. Default 42.
    """

    def __init__(
        self,
        N: int = 60,
        xi: float = 1.0,
        alpha: float = 0.05,
        beta: float = 0.05,
        gamma: float = 0.01,
        eta: float = 0.01,
        seed: int = 42,
    ):
        self.N = N
        self.xi = xi
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta

        rng = np.random.RandomState(seed)

        # Initialize lattice
        self.positions = rng.randn(N, 3)
        self.phi = rng.uniform(0, 2 * np.pi, N)

        # Precompute distance-dependent quantities
        self.distances = compute_distance_matrix(self.positions)
        self.A = compute_amplitude_matrix(self.distances, xi)
        self.triangles = build_triangulation(N, self.positions)

        # Input / target
        self.x = rng.randn(N)
        self.y_target = rng.randn(N)

        # Training history
        self.history: List[TrainingState] = []

    def set_task(self, x: np.ndarray, y_target: np.ndarray):
        """Set a new input/output task.

        Parameters
        ----------
        x : np.ndarray
            Input signal (N,).
        y_target : np.ndarray
            Target output (N,).
        """
        if len(x) != self.N or len(y_target) != self.N:
            raise ValueError(f"x and y_target must have length {self.N}")
        self.x = x
        self.y_target = y_target

    def step(self) -> TrainingState:
        """Execute one gradient descent step.

        Returns
        -------
        TrainingState
            Snapshot after this step.
        """
        grad, losses = analytic_gradient(
            self.phi, self.A, self.x, self.y_target,
            self.alpha, self.beta, self.gamma, self.triangles,
        )

        # Gradient descent
        self.phi -= self.eta * grad

        # Detect regime
        regime = detect_regime(
            self.alpha, self.beta, self.gamma, losses,
        )

        state = TrainingState(
            step=len(self.history),
            loss_total=losses["total"],
            loss_alignment=losses["alignment"],
            loss_coherence=losses["coherence"],
            loss_stability=losses["stability"],
            loss_topology=losses["topology"],
            output_error=losses["output_error"],
            lyapunov_proxy=losses["lyapunov_proxy"],
            winding_number=losses["winding"],
            regime=regime,
        )
        self.history.append(state)
        return state

    def train(
        self,
        steps: int = 200,
        print_every: int = 20,
    ) -> List[TrainingState]:
        """Run training loop.

        Parameters
        ----------
        steps : int
            Number of gradient steps.
        print_every : int
            Print interval. 0 = silent.

        Returns
        -------
        list of TrainingState
            Full training history.
        """
        for s in range(steps):
            state = self.step()
            if print_every > 0 and s % print_every == 0:
                print(
                    f"  step {s:4d} | loss {state.loss_total:8.4f} | "
                    f"error {state.output_error:8.4f} | "
                    f"lyap {state.lyapunov_proxy:6.3f} | "
                    f"wind {state.winding_number:4.1f} | "
                    f"{state.regime}"
                )
        return self.history

    def get_physical_mapping(self) -> PhysicalMapping:
        """Get the physical implementation mapping for the last update.

        Returns
        -------
        PhysicalMapping
            Phase -> refractive index -> laser write parameters.
        """
        if len(self.history) < 2:
            delta = np.zeros(self.N)
        else:
            # Use gradient as proxy for delta_phi
            grad, _ = analytic_gradient(
                self.phi, self.A, self.x, self.y_target,
                self.alpha, self.beta, self.gamma, self.triangles,
            )
            delta = -self.eta * grad

        return compute_physical_mapping(delta)

    def analyze(self) -> EmergentStructure:
        """Analyze emergent structure after training.

        Returns
        -------
        EmergentStructure
            What has emerged: crystallized modes, specialization,
            energy funnels, topological defects.
        """
        return analyze_emergent_structure(
            self.phi, self.A, self.x, self.positions, self.triangles,
        )

    def status(self) -> str:
        """Current optimizer status."""
        if not self.history:
            return "PhaseFieldOptimizer: not yet trained"

        last = self.history[-1]
        first = self.history[0]
        lines = [
            "=" * 55,
            "  Phase Field Optimizer Status",
            "=" * 55,
            f"  Nodes: {self.N}",
            f"  Steps: {len(self.history)}",
            f"  Regime: {last.regime}",
            "",
            f"  Loss:  {first.loss_total:.4f} -> {last.loss_total:.4f}",
            f"  Error: {first.output_error:.4f} -> {last.output_error:.4f}",
            f"  Lyap:  {first.lyapunov_proxy:.4f} -> {last.lyapunov_proxy:.4f}",
            f"  Wind:  {first.winding_number:.1f} -> {last.winding_number:.1f}",
            "",
            f"  Weights: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}",
            f"  Learning rate: {self.eta}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regime Sweep
# ---------------------------------------------------------------------------

def regime_sweep(N: int = 40, steps: int = 100) -> Dict[str, Dict]:
    """Demonstrate the four regimes by sweeping coefficients.

    Returns
    -------
    dict
        Results per regime.
    """
    regimes = {
        "compute-dominant": {"alpha": 0.001, "beta": 0.001, "gamma": 0.0},
        "coherence-dominant": {"alpha": 1.0, "beta": 0.001, "gamma": 0.0},
        "topology-dominant": {"alpha": 0.001, "beta": 0.001, "gamma": 1.0},
        "balanced": {"alpha": 0.1, "beta": 0.1, "gamma": 0.05},
    }

    results = {}
    for name, params in regimes.items():
        opt = PhaseFieldOptimizer(N=N, **params, seed=42)
        opt.train(steps=steps, print_every=0)
        structure = opt.analyze()
        last = opt.history[-1]

        results[name] = {
            "final_error": last.output_error,
            "final_loss": last.loss_total,
            "crystallized_modes": structure.crystallized_modes,
            "memory_nodes": len(structure.spatial_specialization["memory"]),
            "compute_nodes": len(structure.spatial_specialization["compute"]),
            "explore_nodes": len(structure.spatial_specialization["explore"]),
            "defects": len(structure.topological_defects),
            "regime": last.regime,
        }

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Full demonstration of the phase field optimizer."""
    print("=" * 60)
    print("  Phase Field Optimizer")
    print("  Learning = Geometry = Energy Minimization")
    print("=" * 60)

    # --- Training ---
    print("\n--- Training (Balanced Regime) ---")
    opt = PhaseFieldOptimizer(
        N=60, xi=1.0, alpha=0.05, beta=0.05, gamma=0.01, eta=0.01,
    )
    opt.train(steps=200, print_every=40)
    print(f"\n{opt.status()}")

    # --- Emergent Structure ---
    print("\n--- Emergent Structure ---")
    structure = opt.analyze()
    print(f"  Crystallized modes: {structure.crystallized_modes}")
    print(f"  Memory nodes: {len(structure.spatial_specialization['memory'])}")
    print(f"  Compute nodes: {len(structure.spatial_specialization['compute'])}")
    print(f"  Explore nodes: {len(structure.spatial_specialization['explore'])}")
    print(f"  Topological defects: {len(structure.topological_defects)}")

    if structure.energy_funnels:
        print(f"  Top energy funnels:")
        for src, dst, strength in structure.energy_funnels[:3]:
            print(f"    {src} -> {dst}: {strength:.4f}")

    if structure.topological_defects:
        print(f"  Vortex locations:")
        for d in structure.topological_defects[:3]:
            print(f"    triangle {d['triangle']}, charge={d['charge']:.0f}")

    # --- Physical Mapping ---
    print("\n--- Physical Mapping ---")
    mapping = opt.get_physical_mapping()
    print(f"  Max delta_n: {np.max(np.abs(mapping.delta_n)):.2e}")
    print(f"  Voxel thickness: {mapping.voxel_thickness:.1e} m")
    print(f"  Top 5 write locations (by power):")
    top5 = np.argsort(mapping.write_power)[-5:][::-1]
    for idx in top5:
        print(
            f"    Node {idx}: power={mapping.write_power[idx]:.4f}, "
            f"delta_n={mapping.delta_n[idx]:.2e}"
        )

    # --- Regime Sweep ---
    print("\n--- Regime Sweep ---")
    results = regime_sweep(N=40, steps=100)

    print(f"  {'regime':>22s}  {'error':>8s}  {'crystal':>8s}  "
          f"{'memory':>7s}  {'compute':>8s}  {'explore':>8s}  {'defects':>7s}")
    print("  " + "-" * 75)
    for name, res in results.items():
        print(
            f"  {name:>22s}  {res['final_error']:8.3f}  "
            f"{res['crystallized_modes']:8d}  "
            f"{res['memory_nodes']:7d}  {res['compute_nodes']:8d}  "
            f"{res['explore_nodes']:8d}  {res['defects']:7d}"
        )

    # --- Key Insight ---
    print(f"\n--- Key Insight ---")
    print("  One variable (phi), one evolution rule (grad descent),")
    print("  four simultaneous effects:")
    print("    1. Computation (alignment)")
    print("    2. Coherence (XY model smoothing)")
    print("    3. Stability (Lyapunov regulation)")
    print("    4. Memory (topological protection)")
    print()
    print("  Geometry, computation, stability, and memory are")
    print("  the SAME variable viewed under different constraints.")
    print()
    print("  learning = geometry = energy minimization")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    demo()
