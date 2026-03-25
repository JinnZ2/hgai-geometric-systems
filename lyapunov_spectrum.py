"""
Lyapunov Spectrum Analyzer — Controllable Chaos Dynamics

Implements Lyapunov exponent analysis for three interconnected systems:

1. Lorenz System: Canonical chaotic weather model (reference baseline)
2. Phi-Octahedral Lattice: Photonic coupling matrix with tunable chaos
3. Spatial Mode Mapper: Maps Lyapunov modes back onto physical space

Core Insight:
    Unlike weather (fixed physics, uncontrollable lambda), the lattice
    provides a controllable Lyapunov field where stability, computation,
    and chaos are spatially separated and tunable.

Three regimes:
    - Stable (lambda < 0): Memory storage, error correction
    - Critical (lambda ~ 0): Maximum information transfer, edge of chaos
    - Chaotic (lambda > 0): Exploration, pattern generation, adaptation

System Identity:
    Lyapunov Spectrum = Geometry + Phase + Topology

Dependencies:
    - numpy
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHI = (1 + 5**0.5) / 2  # Golden ratio

# Octahedral direction vectors
OCTAHEDRAL_DIRS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=float)


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class LyapunovSpectrum:
    """Container for a computed Lyapunov spectrum.

    Parameters
    ----------
    exponents : np.ndarray
        Sorted Lyapunov exponents (descending).
    eigenvalues : np.ndarray
        Raw eigenvalues of the coupling/Jacobian matrix.
    regime_counts : dict
        Count of modes in each regime (stable, critical, chaotic).
    max_exponent : float
        Largest Lyapunov exponent (lambda_max).
    """
    exponents: np.ndarray
    eigenvalues: np.ndarray
    regime_counts: Dict[str, int]
    max_exponent: float

    def summary(self) -> str:
        """Human-readable spectrum summary."""
        lines = [
            f"Lyapunov Spectrum: {len(self.exponents)} modes",
            f"  lambda_max = {self.max_exponent:.4f}",
            f"  Chaotic (lambda > 0): {self.regime_counts['chaotic']} modes",
            f"  Critical (lambda ~ 0): {self.regime_counts['critical']} modes",
            f"  Stable (lambda < 0): {self.regime_counts['stable']} modes",
        ]
        if self.max_exponent > 0:
            lines.append("  Status: CHAOTIC — positive Lyapunov exponent")
        elif abs(self.max_exponent) < 0.1:
            lines.append("  Status: CRITICAL — edge of chaos")
        else:
            lines.append("  Status: STABLE — convergent dynamics")
        return "\n".join(lines)


@dataclass
class SpatialMode:
    """A Lyapunov mode mapped back to physical space.

    Parameters
    ----------
    mode_index : int
        Index in the sorted spectrum.
    exponent : float
        Lyapunov exponent value.
    regime : str
        'stable', 'critical', or 'chaotic'.
    shell_participation : dict
        How much each shell contributes to this mode.
    eigenvector : np.ndarray
        The full eigenvector for spatial visualization.
    """
    mode_index: int
    exponent: float
    regime: str
    shell_participation: Dict[int, float]
    eigenvector: np.ndarray

    def dominant_shell(self) -> int:
        """Shell with highest participation in this mode."""
        return max(self.shell_participation, key=self.shell_participation.get)


@dataclass
class SpatialMap:
    """Complete spatial mapping of Lyapunov modes onto the lattice.

    Parameters
    ----------
    modes : list of SpatialMode
        All modes with spatial attribution.
    shell_roles : dict
        Primary role of each shell (memory, computation, exploration).
    organism_structure : dict
        Biological analog: core/shell/outer field mapping.
    """
    modes: List[SpatialMode]
    shell_roles: Dict[int, str]
    organism_structure: Dict[str, List[int]]

    def summary(self) -> str:
        """Human-readable spatial map."""
        lines = ["Spatial Mode Map:"]
        for shell, role in sorted(self.shell_roles.items()):
            lines.append(f"  Shell {shell}: {role}")
        lines.append("\nOrganism Structure:")
        for region, shells in self.organism_structure.items():
            lines.append(f"  {region}: shells {shells}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Lorenz System (Reference Chaotic Dynamics)
# ---------------------------------------------------------------------------

class LorenzSystem:
    """Lorenz attractor — canonical chaotic weather model.

    Equations:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Parameters
    ----------
    sigma : float
        Prandtl number. Default 10.
    rho : float
        Rayleigh number. Default 28.
    beta : float
        Geometric factor. Default 8/3.
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
    ):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute dx/dt, dy/dt, dz/dt.

        Parameters
        ----------
        state : np.ndarray
            Current state [x, y, z].

        Returns
        -------
        np.ndarray
            Derivatives [dx/dt, dy/dt, dz/dt].
        """
        x, y, z = state
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        ])

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix at a given state.

        Parameters
        ----------
        state : np.ndarray
            Current state [x, y, z].

        Returns
        -------
        np.ndarray
            3x3 Jacobian matrix.
        """
        x, y, z = state
        return np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, -self.beta],
        ])

    def simulate(
        self,
        initial_state: Optional[np.ndarray] = None,
        dt: float = 0.01,
        steps: int = 10000,
    ) -> np.ndarray:
        """Integrate the Lorenz system using RK4.

        Parameters
        ----------
        initial_state : np.ndarray, optional
            Starting [x, y, z]. Default [1, 1, 1].
        dt : float
            Time step. Default 0.01.
        steps : int
            Number of integration steps. Default 10000.

        Returns
        -------
        np.ndarray
            Trajectory array of shape (steps+1, 3).
        """
        if initial_state is None:
            initial_state = np.array([1.0, 1.0, 1.0])

        trajectory = np.zeros((steps + 1, 3))
        trajectory[0] = initial_state

        for i in range(steps):
            state = trajectory[i]
            k1 = self.derivatives(state)
            k2 = self.derivatives(state + 0.5 * dt * k1)
            k3 = self.derivatives(state + 0.5 * dt * k2)
            k4 = self.derivatives(state + dt * k3)
            trajectory[i + 1] = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return trajectory

    def compute_lyapunov(
        self,
        dt: float = 0.01,
        steps: int = 50000,
        transient: int = 1000,
    ) -> np.ndarray:
        """Estimate the full Lyapunov spectrum via QR decomposition.

        Parameters
        ----------
        dt : float
            Time step.
        steps : int
            Number of integration steps.
        transient : int
            Steps to discard for transient dynamics.

        Returns
        -------
        np.ndarray
            Three Lyapunov exponents, sorted descending.
        """
        state = np.array([1.0, 1.0, 1.0])
        Q = np.eye(3)
        lyap_sum = np.zeros(3)

        # Skip transient
        for _ in range(transient):
            k1 = self.derivatives(state)
            k2 = self.derivatives(state + 0.5 * dt * k1)
            k3 = self.derivatives(state + 0.5 * dt * k2)
            k4 = self.derivatives(state + dt * k3)
            state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        # Compute spectrum
        count = 0
        for _ in range(steps):
            # Evolve state
            k1 = self.derivatives(state)
            k2 = self.derivatives(state + 0.5 * dt * k1)
            k3 = self.derivatives(state + 0.5 * dt * k2)
            k4 = self.derivatives(state + dt * k3)
            state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

            # Evolve perturbation basis
            J = self.jacobian(state)
            Q = Q + dt * (J @ Q)

            # QR decomposition to orthonormalize
            Q, R = np.linalg.qr(Q)
            lyap_sum += np.log(np.abs(np.diag(R)) + 1e-12)
            count += 1

        exponents = lyap_sum / (count * dt)
        return np.sort(exponents)[::-1]


# ---------------------------------------------------------------------------
# 2. Phi-Octahedral Lattice
# ---------------------------------------------------------------------------

class PhiOctahedralLattice:
    """Phi-scaled octahedral lattice with tunable coupling dynamics.

    Builds a photonic lattice where nodes are arranged on concentric
    shells at golden-ratio scaled radii. The coupling matrix K determines
    the Lyapunov spectrum, which can be tuned via three control knobs:

    1. Coherence length (xi): Controls decay rate of coupling
    2. Phase field (phi_i): Controls interference patterns
    3. Geometry (r0, shells): Controls spectral spacing

    Parameters
    ----------
    r0 : float
        Base radius. Default 1.0.
    shells : int
        Number of concentric shells. Default 5.
    xi : float
        Coherence length — coupling decay rate. Default 2.0.
    kappa0 : float
        Base coupling strength. Default 1.0.
    seed : int or None
        Random seed for phase field. Default 0.
    """

    def __init__(
        self,
        r0: float = 1.0,
        shells: int = 5,
        xi: float = 2.0,
        kappa0: float = 1.0,
        seed: Optional[int] = 0,
    ):
        self.r0 = r0
        self.shells = shells
        self.xi = xi
        self.kappa0 = kappa0
        self.seed = seed

        # Build lattice
        self.positions, self.shell_indices = self._build_lattice()
        self.n_nodes = len(self.positions)

        # Generate phase field
        rng = np.random.RandomState(seed)
        self.phase_field = rng.uniform(0, 2 * np.pi, self.n_nodes)

        # Build coupling matrix
        self.K = self._build_coupling_matrix()

    def _build_lattice(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build node positions on phi-scaled octahedral shells."""
        positions = []
        shell_indices = []
        for n in range(self.shells):
            r = self.r0 * (PHI ** n)
            for d in OCTAHEDRAL_DIRS:
                positions.append(r * d)
                shell_indices.append(n)
        return np.array(positions), np.array(shell_indices)

    def _build_coupling_matrix(self) -> np.ndarray:
        """Build the complex coupling matrix K.

        K[i,j] = kappa0 * exp(-d_ij / xi) * exp(i * (phi_i - phi_j))
        """
        N = self.n_nodes
        K = np.zeros((N, N), dtype=complex)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                coupling = self.kappa0 * np.exp(-d / self.xi)
                phase = np.exp(1j * (self.phase_field[i] - self.phase_field[j]))
                K[i, j] = coupling * phase

        return K

    def compute_spectrum(self) -> LyapunovSpectrum:
        """Compute the Lyapunov spectrum from the coupling matrix.

        Lyapunov exponents are derived from eigenvalues:
            lambda_i = ln|mu_i|
        where mu_i are eigenvalues of K.

        Returns
        -------
        LyapunovSpectrum
            Full spectrum with regime classification.
        """
        eigvals = np.linalg.eigvals(self.K)
        exponents = np.log(np.abs(eigvals) + 1e-12)
        sorted_exp = np.sort(exponents.real)[::-1]

        # Classify regimes
        critical_threshold = 0.1
        regime_counts = {
            "chaotic": int(np.sum(sorted_exp > critical_threshold)),
            "critical": int(np.sum(np.abs(sorted_exp) <= critical_threshold)),
            "stable": int(np.sum(sorted_exp < -critical_threshold)),
        }

        return LyapunovSpectrum(
            exponents=sorted_exp,
            eigenvalues=eigvals,
            regime_counts=regime_counts,
            max_exponent=float(sorted_exp[0]),
        )

    def compute_spatial_map(self) -> SpatialMap:
        """Map Lyapunov modes back onto physical space.

        This is direction #2: where in the lattice do chaos vs memory
        physically live?

        Returns
        -------
        SpatialMap
            Modes with shell participation and organism structure.
        """
        eigvals, eigvecs = np.linalg.eig(self.K)
        exponents = np.log(np.abs(eigvals) + 1e-12).real

        # Sort by exponent (descending)
        order = np.argsort(exponents)[::-1]

        critical_threshold = 0.1
        modes = []

        for idx, mode_idx in enumerate(order):
            exp = exponents[mode_idx]
            vec = eigvecs[:, mode_idx]

            # Compute shell participation (how much amplitude per shell)
            participation = {}
            for shell in range(self.shells):
                mask = self.shell_indices == shell
                shell_amp = np.sum(np.abs(vec[mask]) ** 2)
                participation[shell] = float(shell_amp)

            # Normalize
            total = sum(participation.values()) + 1e-12
            participation = {k: v / total for k, v in participation.items()}

            # Classify regime
            if exp > critical_threshold:
                regime = "chaotic"
            elif exp < -critical_threshold:
                regime = "stable"
            else:
                regime = "critical"

            modes.append(SpatialMode(
                mode_index=idx,
                exponent=float(exp),
                regime=regime,
                shell_participation=participation,
                eigenvector=vec,
            ))

        # Determine shell roles from mode distribution
        shell_chaotic = {s: 0.0 for s in range(self.shells)}
        shell_critical = {s: 0.0 for s in range(self.shells)}
        shell_stable = {s: 0.0 for s in range(self.shells)}

        for mode in modes:
            for shell, part in mode.shell_participation.items():
                if mode.regime == "chaotic":
                    shell_chaotic[shell] += part
                elif mode.regime == "critical":
                    shell_critical[shell] += part
                else:
                    shell_stable[shell] += part

        shell_roles = {}
        for s in range(self.shells):
            scores = {
                "memory (stable)": shell_stable[s],
                "computation (critical)": shell_critical[s],
                "exploration (chaotic)": shell_chaotic[s],
            }
            shell_roles[s] = max(scores, key=scores.get)

        # Map to organism structure
        organism = {"core": [], "shell": [], "outer_field": []}
        for s, role in shell_roles.items():
            if "stable" in role:
                organism["core"].append(s)
            elif "critical" in role:
                organism["shell"].append(s)
            else:
                organism["outer_field"].append(s)

        return SpatialMap(
            modes=modes,
            shell_roles=shell_roles,
            organism_structure=organism,
        )

    def tune_coherence(self, xi: float):
        """Tune coherence length and rebuild coupling matrix.

        Parameters
        ----------
        xi : float
            New coherence length.
            Higher xi = more modes near lambda ~ 0
            Lower xi = stronger decay, more storage
        """
        self.xi = xi
        self.K = self._build_coupling_matrix()

    def tune_phase(self, phase_field: np.ndarray):
        """Set a new phase field and rebuild coupling matrix.

        Parameters
        ----------
        phase_field : np.ndarray
            Phase values for each node.
        """
        if len(phase_field) != self.n_nodes:
            raise ValueError(
                f"Phase field must have {self.n_nodes} elements."
            )
        self.phase_field = phase_field
        self.K = self._build_coupling_matrix()

    def tune_geometry(self, r0: float, shells: int):
        """Rebuild lattice with new geometry parameters.

        Parameters
        ----------
        r0 : float
            New base radius.
        shells : int
            New number of shells.
        """
        self.r0 = r0
        self.shells = shells
        self.positions, self.shell_indices = self._build_lattice()
        self.n_nodes = len(self.positions)
        rng = np.random.RandomState(self.seed)
        self.phase_field = rng.uniform(0, 2 * np.pi, self.n_nodes)
        self.K = self._build_coupling_matrix()


# ---------------------------------------------------------------------------
# 3. Comparative Analyzer
# ---------------------------------------------------------------------------

class LyapunovAnalyzer:
    """Compares Lyapunov dynamics across weather and lattice systems.

    Provides the bridge between atmospheric chaos (uncontrollable)
    and lattice chaos (tunable), demonstrating how geometry + phase +
    topology = a programmable chaos engine.
    """

    def __init__(self):
        self.lorenz = LorenzSystem()
        self.lattice = PhiOctahedralLattice()

    def compare_spectra(self) -> Dict[str, LyapunovSpectrum]:
        """Compute and compare Lorenz vs lattice spectra.

        Returns
        -------
        dict
            'lorenz' and 'lattice' LyapunovSpectrum objects.
        """
        # Lorenz spectrum (fast estimate)
        lorenz_exp = self.lorenz.compute_lyapunov(steps=10000)
        lorenz_spectrum = LyapunovSpectrum(
            exponents=lorenz_exp,
            eigenvalues=lorenz_exp,  # simplified
            regime_counts={
                "chaotic": int(np.sum(lorenz_exp > 0.1)),
                "critical": int(np.sum(np.abs(lorenz_exp) <= 0.1)),
                "stable": int(np.sum(lorenz_exp < -0.1)),
            },
            max_exponent=float(lorenz_exp[0]),
        )

        # Lattice spectrum
        lattice_spectrum = self.lattice.compute_spectrum()

        return {
            "lorenz": lorenz_spectrum,
            "lattice": lattice_spectrum,
        }

    def sweep_coherence(
        self,
        xi_values: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, LyapunovSpectrum]]:
        """Sweep coherence length and track spectrum changes.

        Demonstrates the primary control knob: how xi tunes the
        balance between memory, computation, and exploration.

        Parameters
        ----------
        xi_values : np.ndarray, optional
            Coherence lengths to test. Default [0.5, 1, 2, 4, 8].

        Returns
        -------
        list of (xi, LyapunovSpectrum)
            Spectrum at each coherence length.
        """
        if xi_values is None:
            xi_values = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

        results = []
        for xi in xi_values:
            self.lattice.tune_coherence(xi)
            spectrum = self.lattice.compute_spectrum()
            results.append((float(xi), spectrum))

        # Restore default
        self.lattice.tune_coherence(2.0)
        return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_lyapunov():
    """Full demonstration of Lyapunov spectrum analysis."""
    print("=" * 60)
    print("  Lyapunov Spectrum Analyzer")
    print("  Controllable Chaos Dynamics")
    print("=" * 60)

    # --- Lorenz System ---
    print("\n--- 1. Lorenz System (Weather Reference) ---")
    lorenz = LorenzSystem()
    print("Computing Lorenz Lyapunov spectrum...")
    lorenz_exp = lorenz.compute_lyapunov(steps=10000)
    print(f"  Exponents: {lorenz_exp}")
    print(f"  lambda_max = {lorenz_exp[0]:.4f}")
    if lorenz_exp[0] > 0:
        print("  -> Chaotic: positive Lyapunov exponent confirmed")

    # --- Phi-Octahedral Lattice ---
    print("\n--- 2. Phi-Octahedral Lattice ---")
    lattice = PhiOctahedralLattice(shells=5, xi=2.0)
    spectrum = lattice.compute_spectrum()
    print(spectrum.summary())

    # --- Spatial Mode Mapping ---
    print("\n--- 3. Spatial Mode Map (Where Chaos Lives) ---")
    spatial = lattice.compute_spatial_map()
    print(spatial.summary())

    # Show top chaotic and stable modes
    chaotic_modes = [m for m in spatial.modes if m.regime == "chaotic"]
    stable_modes = [m for m in spatial.modes if m.regime == "stable"]
    critical_modes = [m for m in spatial.modes if m.regime == "critical"]

    if chaotic_modes:
        top = chaotic_modes[0]
        print(f"\n  Top chaotic mode (lambda={top.exponent:.4f}):")
        print(f"    Dominant shell: {top.dominant_shell()}")
        for s, p in sorted(top.shell_participation.items()):
            bar = "#" * int(p * 40)
            print(f"    Shell {s}: {p:.3f} {bar}")

    if stable_modes:
        bot = stable_modes[-1]
        print(f"\n  Most stable mode (lambda={bot.exponent:.4f}):")
        print(f"    Dominant shell: {bot.dominant_shell()}")
        for s, p in sorted(bot.shell_participation.items()):
            bar = "#" * int(p * 40)
            print(f"    Shell {s}: {p:.3f} {bar}")

    # --- Coherence Sweep ---
    print("\n--- 4. Coherence Sweep (Control Knob) ---")
    analyzer = LyapunovAnalyzer()
    sweep = analyzer.sweep_coherence()

    print(f"  {'xi':>6s}  {'lambda_max':>10s}  {'chaotic':>8s}  {'critical':>8s}  {'stable':>8s}")
    print("  " + "-" * 48)
    for xi, spec in sweep:
        rc = spec.regime_counts
        print(
            f"  {xi:6.1f}  {spec.max_exponent:10.4f}  "
            f"{rc['chaotic']:8d}  {rc['critical']:8d}  {rc['stable']:8d}"
        )

    # --- System Comparison ---
    print("\n--- 5. Weather vs Lattice ---")
    print("  Weather (Lorenz):")
    print(f"    lambda ~ 0.5-1.0 per day (FIXED physics)")
    print(f"    Computed: {lorenz_exp[0]:.4f}")
    print("  Lattice (Phi-Octahedral):")
    print(f"    lambda = TUNABLE via xi, phase, geometry")
    print(f"    Current: {spectrum.max_exponent:.4f}")
    print("\n  Key difference: Weather is uncontrollable chaos.")
    print("  The lattice is a programmable chaos engine.")

    # --- Organism Structure ---
    print("\n--- 6. Organism Structure ---")
    print("  Core (lambda < 0):  Stable memory crystal")
    print("  Shell (lambda ~ 0): Computation layer")
    print("  Outer (lambda > 0): Exploratory sensing / adaptation")
    print(f"\n  Mapped shells:")
    for region, shells in spatial.organism_structure.items():
        if shells:
            print(f"    {region}: {shells}")

    print("\n" + "=" * 60)
    print("  System = Energy Flow + Nonlinearity + Constraint")
    print("         => Lyapunov Spectrum")
    print("=" * 60)


if __name__ == "__main__":
    demo_lyapunov()
