"""
Defect Field — Intentional Topological Defects in Continuous Phase Fields

The key experiment: inject vortex defects into a 2D phase field and see
if they become functional (not errors). If a discontinuity that cannot
be removed by smooth transformation becomes useful, you've found:

    Non-erasable memory that also participates in computation.

This module extends the phase field optimizer to continuous 2D fields
with explicit vortex injection, defect tracking, and analysis of
whether defects help or hurt computation.

Observations to track:
    1. Defect persistence: Do vortices survive training?
    2. Energy localization: Do gradients cluster near defects?
    3. Functional behavior: Do defects speed or slow convergence?
    4. Emergent behavior: Dipoles, information anchors, phase domains?

Dependencies:
    - numpy
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class Defect:
    """A topological defect (vortex) in the phase field.

    Parameters
    ----------
    x : float
        X position.
    y : float
        Y position.
    charge : int
        Topological charge (+1 or -1).
    """
    x: float
    y: float
    charge: int


@dataclass
class FieldSnapshot:
    """Snapshot of the field state at a training step.

    Parameters
    ----------
    step : int
        Training iteration.
    loss : float
        Total loss value.
    loss_compute : float
        Computation alignment loss.
    n_defects : int
        Number of detected defects.
    defect_positions : list
        (x, y, charge) for each defect.
    energy_near_defects : float
        Gradient energy within defect neighborhoods.
    energy_far : float
        Gradient energy away from defects.
    phase_domain_count : int
        Number of distinct phase domains.
    max_gradient : float
        Maximum gradient magnitude.
    """
    step: int
    loss: float
    loss_compute: float
    n_defects: int
    defect_positions: List[Tuple[float, float, int]]
    energy_near_defects: float
    energy_far: float
    phase_domain_count: int
    max_gradient: float


@dataclass
class DefectExperimentResults:
    """Comparison of training with and without defects.

    Parameters
    ----------
    with_defects : list of FieldSnapshot
        Training history with injected defects.
    without_defects : list of FieldSnapshot
        Training history without defects (control).
    defects_helped : bool
        Whether defects improved convergence.
    convergence_ratio : float
        Final loss(with) / final loss(without).
    defect_survival_rate : float
        Fraction of injected defects that survived training.
    energy_localization : float
        Ratio of gradient energy near defects vs. far.
    verdict : str
        Summary judgment.
    """
    with_defects: List[FieldSnapshot]
    without_defects: List[FieldSnapshot]
    defects_helped: bool
    convergence_ratio: float
    defect_survival_rate: float
    energy_localization: float
    verdict: str


# ---------------------------------------------------------------------------
# 2D Field Operations
# ---------------------------------------------------------------------------

def laplacian_2d(f: np.ndarray) -> np.ndarray:
    """Discrete Laplacian on a 2D grid (periodic boundary).

    Parameters
    ----------
    f : np.ndarray
        2D field (N x N).

    Returns
    -------
    np.ndarray
        Laplacian of f.
    """
    return (
        np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0)
        + np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1)
        - 4 * f
    )


def gradient_magnitude(f: np.ndarray) -> np.ndarray:
    """Compute |grad(f)| on a 2D grid.

    Parameters
    ----------
    f : np.ndarray
        2D field.

    Returns
    -------
    np.ndarray
        Gradient magnitude at each point.
    """
    dx = 0.5 * (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))
    dy = 0.5 * (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))
    return np.sqrt(dx**2 + dy**2)


def add_vortex(
    phi: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    x0: float,
    y0: float,
    charge: int = 1,
) -> np.ndarray:
    """Inject a topological vortex into the phase field.

    phi += charge * arctan2(Y - y0, X - x0)

    A vortex is a discontinuity that cannot be removed by smooth
    transformation. Its winding integral = 2*pi*charge.

    Parameters
    ----------
    phi : np.ndarray
        Phase field (N x N).
    X, Y : np.ndarray
        Coordinate grids.
    x0, y0 : float
        Vortex center position.
    charge : int
        Topological charge (+1 or -1).

    Returns
    -------
    np.ndarray
        Updated phase field.
    """
    return phi + charge * np.arctan2(Y - y0, X - x0)


# ---------------------------------------------------------------------------
# Defect Detection
# ---------------------------------------------------------------------------

def detect_defects(
    phi: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    threshold: float = 0.8,
) -> List[Tuple[float, float, int]]:
    """Detect topological defects via discrete winding number.

    For each 2x2 plaquette, compute the winding number from the
    phase differences around the loop. Non-zero winding = defect.

    Parameters
    ----------
    phi : np.ndarray
        Phase field (N x N).
    X, Y : np.ndarray
        Coordinate grids.
    threshold : float
        Minimum |winding| to count as a defect.

    Returns
    -------
    list of (x, y, charge)
        Detected defect positions and charges.
    """
    N = phi.shape[0]
    defects = []

    for i in range(N - 1):
        for j in range(N - 1):
            # Plaquette corners: (i,j), (i,j+1), (i+1,j+1), (i+1,j)
            d1 = _wrap_angle(phi[i, j+1] - phi[i, j])
            d2 = _wrap_angle(phi[i+1, j+1] - phi[i, j+1])
            d3 = _wrap_angle(phi[i+1, j] - phi[i+1, j+1])
            d4 = _wrap_angle(phi[i, j] - phi[i+1, j])

            winding = (d1 + d2 + d3 + d4) / (2 * np.pi)

            if abs(winding) > threshold:
                cx = 0.5 * (X[i, j] + X[i+1, j+1])
                cy = 0.5 * (Y[i, j] + Y[i+1, j+1])
                charge = int(round(winding))
                defects.append((float(cx), float(cy), charge))

    return defects


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Energy Analysis
# ---------------------------------------------------------------------------

def analyze_energy_localization(
    phi: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    defects: List[Tuple[float, float, int]],
    radius: float = 0.15,
) -> Tuple[float, float]:
    """Measure gradient energy near vs. far from defects.

    If energy localizes near defects, they're acting as
    information anchors.

    Parameters
    ----------
    phi : np.ndarray
        Phase field.
    X, Y : np.ndarray
        Coordinate grids.
    defects : list
        Detected defect positions.
    radius : float
        Neighborhood radius around each defect.

    Returns
    -------
    energy_near : float
        Average gradient energy near defects.
    energy_far : float
        Average gradient energy far from defects.
    """
    grad_mag = gradient_magnitude(phi)

    # Create mask for defect neighborhoods
    near_mask = np.zeros_like(phi, dtype=bool)
    for dx, dy, _ in defects:
        dist = np.sqrt((X - dx)**2 + (Y - dy)**2)
        near_mask |= (dist < radius)

    far_mask = ~near_mask

    near_count = np.sum(near_mask)
    far_count = np.sum(far_mask)

    energy_near = np.sum(grad_mag[near_mask]) / max(near_count, 1)
    energy_far = np.sum(grad_mag[far_mask]) / max(far_count, 1)

    # Cap ratio to avoid overflow when defects cover the field
    if energy_far < 1e-6:
        energy_far = max(energy_far, energy_near * 1e-3)

    return float(energy_near), float(energy_far)


def count_phase_domains(phi: np.ndarray, n_bins: int = 6) -> int:
    """Count distinct phase domains via histogram clustering.

    Parameters
    ----------
    phi : np.ndarray
        Phase field.
    n_bins : int
        Number of phase bins.

    Returns
    -------
    int
        Number of occupied phase bins (proxy for domain count).
    """
    wrapped = phi % (2 * np.pi)
    hist, _ = np.histogram(wrapped, bins=n_bins, range=(0, 2 * np.pi))
    return int(np.sum(hist > phi.size * 0.02))  # >2% occupancy threshold


# ---------------------------------------------------------------------------
# Continuous Field Optimizer
# ---------------------------------------------------------------------------

class DefectFieldOptimizer:
    """2D continuous phase field with intentional vortex defects.

    Extends the phase field optimizer to a continuous 2D grid where
    vortices can be injected, tracked, and tested for functionality.

    Parameters
    ----------
    N : int
        Grid size (N x N). Default 40.
    alpha : float
        Coherence (smoothness) weight. Default 0.2.
    beta : float
        Stability weight. Default 0.05.
    eta : float
        Learning rate. Default 0.05.
    seed : int
        Random seed. Default 42.
    """

    def __init__(
        self,
        N: int = 40,
        alpha: float = 0.2,
        beta: float = 0.05,
        eta: float = 0.05,
        seed: int = 42,
    ):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

        # Coordinate grids
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        self.X, self.Y = np.meshgrid(x, y)

        # Phase field (starts flat)
        self.phi = np.zeros((N, N))

        # Input / target
        rng = np.random.RandomState(seed)
        self.inp = rng.randn(N, N)
        self.target = rng.randn(N, N)

        # Injected defects (for tracking)
        self.injected_defects: List[Defect] = []

        # Training history
        self.history: List[FieldSnapshot] = []

    def inject_vortex(self, x0: float, y0: float, charge: int = 1):
        """Inject a topological vortex into the phase field.

        Parameters
        ----------
        x0, y0 : float
            Vortex center (in [-1, 1] coordinates).
        charge : int
            Topological charge (+1 or -1).
        """
        self.phi = add_vortex(self.phi, self.X, self.Y, x0, y0, charge)
        self.injected_defects.append(Defect(x=x0, y=y0, charge=charge))

    def forward(self) -> np.ndarray:
        """Compute output: y = cos(phi) * input."""
        return np.cos(self.phi) * self.inp

    def loss(self) -> float:
        """Compute alignment loss."""
        out = self.forward()
        return 0.5 * np.linalg.norm(out - self.target) ** 2

    def gradient(self) -> np.ndarray:
        """Compute analytic gradient of the unified energy.

        Returns
        -------
        np.ndarray
            Gradient field (N x N).
        """
        out = self.forward()
        e = out - self.target

        # Computation gradient
        g_compute = -np.sin(self.phi) * self.inp * e

        # Coherence (smoothness via Laplacian)
        g_smooth = laplacian_2d(self.phi)

        # Stability (Lyapunov)
        norm = np.linalg.norm(out) + 1e-8
        g_stab = (out * (-np.sin(self.phi) * self.inp)) / norm

        return g_compute + self.alpha * g_smooth + self.beta * g_stab

    def step(self) -> FieldSnapshot:
        """Execute one gradient descent step with full analysis.

        Returns
        -------
        FieldSnapshot
            Current state including defect analysis.
        """
        # Gradient step
        g = self.gradient()
        self.phi -= self.eta * g

        # Compute losses
        out = self.forward()
        loss_total = 0.5 * np.linalg.norm(out - self.target) ** 2
        loss_compute = loss_total  # primary loss

        # Defect detection
        defects = detect_defects(self.phi, self.X, self.Y)

        # Energy localization
        energy_near, energy_far = analyze_energy_localization(
            self.phi, self.X, self.Y, defects,
        )

        # Phase domains
        domains = count_phase_domains(self.phi)

        # Max gradient
        grad_mag = gradient_magnitude(self.phi)
        max_grad = float(np.max(grad_mag))

        snapshot = FieldSnapshot(
            step=len(self.history),
            loss=float(loss_total),
            loss_compute=float(loss_compute),
            n_defects=len(defects),
            defect_positions=defects,
            energy_near_defects=energy_near,
            energy_far=energy_far,
            phase_domain_count=domains,
            max_gradient=max_grad,
        )
        self.history.append(snapshot)
        return snapshot

    def train(
        self,
        steps: int = 200,
        print_every: int = 20,
    ) -> List[FieldSnapshot]:
        """Run training loop with defect tracking.

        Parameters
        ----------
        steps : int
            Number of gradient steps.
        print_every : int
            Print interval (0 = silent).

        Returns
        -------
        list of FieldSnapshot
        """
        for s in range(steps):
            snap = self.step()
            if print_every > 0 and s % print_every == 0:
                loc_ratio = snap.energy_near_defects / (snap.energy_far + 1e-8)
                print(
                    f"  step {s:4d} | loss {snap.loss:8.2f} | "
                    f"defects {snap.n_defects:3d} | "
                    f"domains {snap.phase_domain_count:2d} | "
                    f"E_near/E_far {loc_ratio:5.2f}"
                )
        return self.history

    def reset_phi(self):
        """Reset phase field to zero (for control experiment)."""
        self.phi = np.zeros((self.N, self.N))
        self.injected_defects = []
        self.history = []


# ---------------------------------------------------------------------------
# A/B Experiment: Defects Help or Hurt?
# ---------------------------------------------------------------------------

def run_defect_experiment(
    N: int = 40,
    steps: int = 200,
    defect_configs: Optional[List[Tuple[float, float, int]]] = None,
    seed: int = 42,
) -> DefectExperimentResults:
    """Run controlled experiment: training WITH vs WITHOUT defects.

    Parameters
    ----------
    N : int
        Grid size.
    steps : int
        Training steps.
    defect_configs : list of (x, y, charge), optional
        Defects to inject. Default: vortex-antivortex dipole.
    seed : int
        Random seed (same for both runs).

    Returns
    -------
    DefectExperimentResults
        Comparison with verdict.
    """
    if defect_configs is None:
        defect_configs = [
            (0.0, 0.0, 1),    # vortex
            (0.3, 0.3, -1),   # antivortex (dipole)
        ]

    # --- Run WITH defects ---
    opt_with = DefectFieldOptimizer(N=N, seed=seed)
    for x0, y0, charge in defect_configs:
        opt_with.inject_vortex(x0, y0, charge)
    history_with = opt_with.train(steps=steps, print_every=0)

    # --- Run WITHOUT defects (control) ---
    opt_without = DefectFieldOptimizer(N=N, seed=seed)
    history_without = opt_without.train(steps=steps, print_every=0)

    # --- Compare ---
    final_with = history_with[-1]
    final_without = history_without[-1]

    # Did defects survive?
    initial_defects = len(defect_configs)
    final_defects = final_with.n_defects
    survival_rate = min(final_defects / max(initial_defects, 1), 1.0)

    # Convergence comparison
    ratio = final_with.loss / (final_without.loss + 1e-12)
    helped = ratio < 0.98  # defects helped if loss is >2% lower

    # Energy localization
    loc = final_with.energy_near_defects / (final_with.energy_far + 1e-8)

    # Verdict
    if helped and survival_rate > 0.5:
        verdict = (
            "CASE A: Defects HELP. They persist and improve convergence. "
            "Non-erasable memory participating in computation."
        )
    elif helped and survival_rate < 0.5:
        verdict = (
            "MIXED: Defects helped initially but dissolved. "
            "Transient memory with computational benefit."
        )
    elif not helped and survival_rate > 0.5:
        verdict = (
            "CASE B (partial): Defects persist but don't help. "
            "Stable memory but not yet functional."
        )
    else:
        verdict = (
            "CASE B: Defects dissolved and didn't help. "
            "The field smoothed them out."
        )

    return DefectExperimentResults(
        with_defects=history_with,
        without_defects=history_without,
        defects_helped=helped,
        convergence_ratio=float(ratio),
        defect_survival_rate=float(survival_rate),
        energy_localization=float(loc),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Multi-Configuration Experiment
# ---------------------------------------------------------------------------

def sweep_defect_configs(
    N: int = 40,
    steps: int = 200,
    seed: int = 42,
) -> Dict[str, DefectExperimentResults]:
    """Test multiple defect configurations.

    Configs:
    1. Single vortex (monopole)
    2. Vortex-antivortex dipole
    3. Quadrupole (4 defects)
    4. Random cluster

    Returns
    -------
    dict of str -> DefectExperimentResults
    """
    rng = np.random.RandomState(seed)

    configs = {
        "no_defects (control)": [],
        "single_vortex": [(0.0, 0.0, 1)],
        "dipole": [(0.0, 0.0, 1), (0.3, 0.3, -1)],
        "quadrupole": [
            (-0.3, -0.3, 1), (0.3, -0.3, -1),
            (-0.3, 0.3, -1), (0.3, 0.3, 1),
        ],
        "random_cluster": [
            (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
             rng.choice([-1, 1]))
            for _ in range(6)
        ],
    }

    results = {}
    for name, defects in configs.items():
        if name == "no_defects (control)":
            # Just run the control
            opt = DefectFieldOptimizer(N=N, seed=seed)
            opt.train(steps=steps, print_every=0)
            results[name] = DefectExperimentResults(
                with_defects=opt.history,
                without_defects=opt.history,
                defects_helped=False,
                convergence_ratio=1.0,
                defect_survival_rate=0.0,
                energy_localization=1.0,
                verdict="Control (no defects).",
            )
        else:
            results[name] = run_defect_experiment(
                N=N, steps=steps, defect_configs=defects, seed=seed,
            )

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Full defect field experiment."""
    print("=" * 60)
    print("  Defect Field — Intentional Topological Defects")
    print("  Do discontinuities become functional?")
    print("=" * 60)

    # --- Single run with defects ---
    print("\n--- Training WITH Vortex Dipole ---")
    opt = DefectFieldOptimizer(N=40, alpha=0.2, beta=0.05, eta=0.05)
    opt.inject_vortex(0.0, 0.0, charge=1)
    opt.inject_vortex(0.3, 0.3, charge=-1)
    print(f"  Injected: vortex at (0,0), antivortex at (0.3,0.3)")
    opt.train(steps=200, print_every=40)

    first = opt.history[0]
    last = opt.history[-1]
    print(f"\n  Initial defects: {first.n_defects}")
    print(f"  Final defects:   {last.n_defects}")
    print(f"  Defect survival: {'YES' if last.n_defects > 0 else 'NO'}")
    print(f"  Phase domains:   {first.phase_domain_count} -> {last.phase_domain_count}")

    loc = last.energy_near_defects / (last.energy_far + 1e-8)
    print(f"  Energy localization (near/far): {loc:.2f}")
    if loc > 1.2:
        print("  -> Energy CLUSTERS near defects (information anchors)")
    elif loc < 0.8:
        print("  -> Energy avoids defects (defects are inert)")
    else:
        print("  -> Energy roughly uniform")

    # --- A/B Experiment ---
    print("\n--- A/B Experiment: Defects Help or Hurt? ---")
    result = run_defect_experiment(N=40, steps=200)

    print(f"  Final loss (with defects):    {result.with_defects[-1].loss:.2f}")
    print(f"  Final loss (without defects): {result.without_defects[-1].loss:.2f}")
    print(f"  Convergence ratio:  {result.convergence_ratio:.4f}")
    print(f"  Defect survival:    {result.defect_survival_rate:.0%}")
    print(f"  Energy localization: {result.energy_localization:.2f}")
    print(f"\n  VERDICT: {result.verdict}")

    # --- Multi-config sweep ---
    print("\n--- Defect Configuration Sweep ---")
    sweep = sweep_defect_configs(N=40, steps=200)

    print(f"  {'config':>22s}  {'loss':>8s}  {'ratio':>7s}  "
          f"{'survive':>8s}  {'E_loc':>6s}  {'defects':>7s}")
    print("  " + "-" * 65)
    for name, res in sweep.items():
        final = res.with_defects[-1]
        print(
            f"  {name:>22s}  {final.loss:8.2f}  "
            f"{res.convergence_ratio:7.4f}  "
            f"{res.defect_survival_rate:8.0%}  "
            f"{res.energy_localization:6.2f}  "
            f"{final.n_defects:7d}"
        )

    # --- Convergence curves ---
    print("\n--- Convergence Comparison (sampled) ---")
    with_hist = result.with_defects
    without_hist = result.without_defects
    print(f"  {'step':>6s}  {'with_defects':>14s}  {'without':>14s}  {'delta':>10s}")
    print("  " + "-" * 48)
    for i in range(0, len(with_hist), 40):
        w = with_hist[i].loss
        wo = without_hist[i].loss
        delta = w - wo
        marker = "<-" if delta < 0 else ""
        print(f"  {i:6d}  {w:14.2f}  {wo:14.2f}  {delta:10.2f} {marker}")

    # --- Defect tracking over time ---
    print("\n--- Defect Count Over Time ---")
    for i in range(0, len(with_hist), 40):
        snap = with_hist[i]
        bar = "#" * min(snap.n_defects, 50)
        print(f"  step {i:4d}: {snap.n_defects:3d} {bar}")

    # --- Key insight ---
    print(f"\n--- Key Insight ---")
    print("  A defect is a discontinuity that cannot be removed")
    print("  by smooth transformation.")
    print()
    if result.defects_helped:
        print("  In this system, defects BECAME FUNCTIONAL:")
        print("  non-erasable memory that participates in computation.")
        print("  This is where natural systems cross into true adaptability.")
    else:
        print("  In this configuration, defects were neutral or harmful.")
        print("  Try: different alpha/beta, more training, or different")
        print("  defect placements to find the functional regime.")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    demo()
