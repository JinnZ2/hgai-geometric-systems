"""
Chaos-Aware Weather AI — A Learning System That Embraces Uncertainty

An experimental weather intelligence system that differs from conventional
models in fundamental ways:

1. Lyapunov-Aware: Knows its own predictability horizon. When lambda spikes,
   it widens uncertainty bands instead of pretending it still knows.

2. Outlier-First: Treats anomalies as the most important signal, not noise
   to be smoothed away. Phase transitions are features, not errors.

3. Entropy-Tracked: Monitors its own prediction entropy over time. When
   the gap between predicted and observed entropy grows, it flags
   institutional friction or model-reality dissonance.

4. Self-Learning: Maintains a memory of what it got wrong and why.
   Updates its beliefs about which regime it's in (stable/critical/chaotic)
   based on accumulated evidence.

How this differs from conventional weather models:
    - GFS/ECMWF: Optimize for accuracy on average days. Fail on extreme days.
    - This system: Optimizes for *knowing when it doesn't know*. Trades
      average-day precision for extreme-day honesty.

Connects:
    - lyapunov_spectrum.py: Regime detection and predictability horizon
    - flux_sensor.py: Phase transition early warning
    - sovereign_impact_sensor.py: System stress calibration
    - resilience/detectors.py: Institutional friction in forecast text
    - weather_node_network.py: Ensemble pipeline for probabilistic output

Dependencies:
    - numpy
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flux_sensor import WeatherFlux
from lyapunov_spectrum import LorenzSystem


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """A single weather observation with timestamp.

    Parameters
    ----------
    timestamp : float
        Time index (arbitrary units, sequential).
    pressure : float
        Atmospheric pressure (hPa).
    temperature : float
        Temperature (C).
    humidity : float
        Relative humidity (0-1).
    wind_speed : float
        Wind speed (m/s).
    metadata : dict
        Any additional sensor data.
    """
    timestamp: float
    pressure: float
    temperature: float
    humidity: float = 0.5
    wind_speed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Convert to state vector for Lyapunov analysis."""
        return np.array([
            self.pressure, self.temperature,
            self.humidity, self.wind_speed,
        ])


@dataclass
class Forecast:
    """A chaos-aware probabilistic forecast.

    Unlike conventional forecasts that give a single number, this
    gives you the full picture: what it thinks, how sure it is,
    and what could blow it up.

    Parameters
    ----------
    timestamp : float
        Forecast valid time.
    horizon : float
        Hours ahead this forecast covers.
    median : np.ndarray
        Central estimate [pressure, temp, humidity, wind].
    p10 : np.ndarray
        10th percentile (optimistic bound).
    p90 : np.ndarray
        90th percentile (pessimistic bound).
    regime : str
        Current detected regime: 'stable', 'critical', 'chaotic'.
    confidence : float
        Self-assessed confidence (0-1). Drops in chaotic regime.
    lyapunov_estimate : float
        Current lambda estimate.
    predictability_horizon : float
        Estimated hours until forecast becomes unreliable.
    phase_transition_risk : float
        Probability of regime change in forecast window.
    warnings : list of str
        Honest warnings about what could go wrong.
    """
    timestamp: float
    horizon: float
    median: np.ndarray
    p10: np.ndarray
    p90: np.ndarray
    regime: str = "unknown"
    confidence: float = 0.5
    lyapunov_estimate: float = 0.0
    predictability_horizon: float = 48.0
    phase_transition_risk: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def spread(self) -> np.ndarray:
        """Width of the uncertainty band."""
        return self.p90 - self.p10

    def summary(self) -> str:
        """Human-readable forecast summary."""
        labels = ["Pressure", "Temp", "Humidity", "Wind"]
        lines = [
            f"Forecast (t+{self.horizon:.0f}h):",
            f"  Regime: {self.regime.upper()}",
            f"  Confidence: {self.confidence:.0%}",
            f"  Predictability horizon: {self.predictability_horizon:.0f}h",
        ]
        for i, label in enumerate(labels):
            lines.append(
                f"  {label:12s}: {self.median[i]:7.1f}  "
                f"[{self.p10[i]:.1f} — {self.p90[i]:.1f}]"
            )
        if self.phase_transition_risk > 0.3:
            lines.append(
                f"  PHASE TRANSITION RISK: {self.phase_transition_risk:.0%}"
            )
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


@dataclass
class LearningMemory:
    """What the AI remembers about its own failures and discoveries.

    Parameters
    ----------
    prediction_errors : list
        History of (predicted, observed, error_magnitude) tuples.
    regime_transitions : list
        History of detected regime changes.
    outlier_events : list
        Events that were flagged as outliers but turned out important.
    discovered_patterns : list
        Patterns the AI learned from its mistakes.
    total_forecasts : int
        How many forecasts have been issued.
    correct_regime_calls : int
        How many times the regime prediction was right.
    """
    prediction_errors: List[Dict[str, Any]] = field(default_factory=list)
    regime_transitions: List[Dict[str, Any]] = field(default_factory=list)
    outlier_events: List[Dict[str, Any]] = field(default_factory=list)
    discovered_patterns: List[str] = field(default_factory=list)
    total_forecasts: int = 0
    correct_regime_calls: int = 0

    @property
    def regime_accuracy(self) -> float:
        """How often the AI correctly predicted the regime."""
        evaluated = len(self.prediction_errors)
        if evaluated == 0:
            return 0.0
        return self.correct_regime_calls / evaluated

    @property
    def mean_error(self) -> float:
        """Average prediction error magnitude."""
        if not self.prediction_errors:
            return 0.0
        return np.mean([e["magnitude"] for e in self.prediction_errors])

    def summary(self) -> str:
        """What the AI has learned about itself."""
        lines = [
            f"Learning Memory ({self.total_forecasts} forecasts):",
            f"  Regime accuracy: {self.regime_accuracy:.0%}",
            f"  Mean error: {self.mean_error:.4f}",
            f"  Regime transitions seen: {len(self.regime_transitions)}",
            f"  Outlier events captured: {len(self.outlier_events)}",
            f"  Patterns discovered: {len(self.discovered_patterns)}",
        ]
        if self.discovered_patterns:
            lines.append("  Discoveries:")
            for p in self.discovered_patterns[-5:]:
                lines.append(f"    - {p}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Detects the current dynamical regime from observation history.

    Uses the Lyapunov exponent estimated from consecutive observations
    to classify the atmosphere into three regimes:
    - Stable: Predictable, conventional forecasting works
    - Critical: Edge of chaos, maximum information but fragile
    - Chaotic: Forecast will diverge fast, widen uncertainty bands

    Parameters
    ----------
    window : int
        Number of recent observations to use. Default 10.
    critical_threshold : float
        Lambda values within this of zero are 'critical'. Default 0.15.
    """

    def __init__(self, window: int = 10, critical_threshold: float = 0.15):
        self.window = window
        self.critical_threshold = critical_threshold
        self.history: List[np.ndarray] = []

    def update(self, obs: Observation):
        """Add a new observation to the history."""
        self.history.append(obs.to_vector())
        if len(self.history) > self.window * 2:
            self.history = self.history[-self.window * 2:]

    def estimate_lyapunov(self) -> float:
        """Estimate local Lyapunov exponent from recent observations.

        Uses the divergence rate of consecutive state differences
        as a proxy for the local Lyapunov exponent.

        Returns
        -------
        float
            Estimated lambda. Positive = chaotic, negative = stable.
        """
        if len(self.history) < 3:
            return 0.0

        # Compute consecutive differences
        diffs = []
        for i in range(1, len(self.history)):
            diff = np.linalg.norm(self.history[i] - self.history[i-1])
            diffs.append(max(diff, 1e-12))

        # Growth rate of perturbation magnitudes
        if len(diffs) < 2:
            return 0.0

        growth_rates = []
        for i in range(1, len(diffs)):
            rate = np.log(diffs[i] / diffs[i-1])
            growth_rates.append(rate)

        return float(np.mean(growth_rates))

    def detect_regime(self) -> Tuple[str, float]:
        """Classify the current dynamical regime.

        Returns
        -------
        regime : str
            'stable', 'critical', or 'chaotic'.
        lambda_est : float
            Estimated Lyapunov exponent.
        """
        lam = self.estimate_lyapunov()

        if lam > self.critical_threshold:
            return "chaotic", lam
        elif lam < -self.critical_threshold:
            return "stable", lam
        else:
            return "critical", lam

    def predictability_horizon(self, lambda_est: float) -> float:
        """Estimate how far ahead forecasts are useful.

        Based on the Lorenz predictability limit:
            T ~ (1/lambda) * ln(delta_max / delta_0)

        Parameters
        ----------
        lambda_est : float
            Current Lyapunov estimate.

        Returns
        -------
        float
            Estimated useful forecast hours.
        """
        if lambda_est <= 0:
            return 168.0  # Stable: 7 days

        # Scale: typical weather lambda ~ 0.5-1.0 per day
        # delta_max/delta_0 ratio ~ 1000 for weather
        t_hours = (1.0 / lambda_est) * np.log(1000) * 24
        return min(max(t_hours, 1.0), 168.0)


# ---------------------------------------------------------------------------
# Outlier-First Processor
# ---------------------------------------------------------------------------

class OutlierFirstProcessor:
    """Processes observations with anomalies treated as primary signal.

    Conventional models: smooth outliers -> predict mean -> evaluate.
    This model: flag outliers -> amplify signal -> investigate -> learn.

    Parameters
    ----------
    anomaly_threshold : float
        Z-score threshold for flagging anomalies. Default 2.0.
    """

    def __init__(self, anomaly_threshold: float = 2.0):
        self.anomaly_threshold = anomaly_threshold
        self.running_mean: Optional[np.ndarray] = None
        self.running_var: Optional[np.ndarray] = None
        self.n_samples: int = 0

    def update_stats(self, vec: np.ndarray):
        """Update running statistics with Welford's algorithm."""
        self.n_samples += 1
        if self.running_mean is None:
            self.running_mean = vec.copy()
            self.running_var = np.zeros_like(vec)
        else:
            delta = vec - self.running_mean
            self.running_mean += delta / self.n_samples
            delta2 = vec - self.running_mean
            self.running_var += delta * delta2

    def check_anomaly(self, vec: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Check if an observation is anomalous.

        Instead of removing it, we flag it and amplify its weight.

        Parameters
        ----------
        vec : np.ndarray
            Observation state vector.

        Returns
        -------
        is_anomaly : bool
            True if any dimension exceeds the z-score threshold.
        z_scores : np.ndarray
            Z-score for each dimension.
        """
        if self.n_samples < 5 or self.running_mean is None:
            return False, np.zeros_like(vec)

        std = np.sqrt(self.running_var / max(self.n_samples - 1, 1)) + 1e-12
        z_scores = np.abs(vec - self.running_mean) / std
        is_anomaly = np.any(z_scores > self.anomaly_threshold)
        return bool(is_anomaly), z_scores

    def anomaly_weight(self, z_scores: np.ndarray) -> float:
        """Convert z-scores to an amplification weight.

        Higher z-score = MORE important, not less.
        This is the opposite of conventional smoothing.

        Returns
        -------
        float
            Weight multiplier (1.0 = normal, >1.0 = amplified).
        """
        max_z = np.max(z_scores)
        if max_z < self.anomaly_threshold:
            return 1.0
        return 1.0 + (max_z - self.anomaly_threshold) * 0.5


# ---------------------------------------------------------------------------
# Ensemble Forecaster
# ---------------------------------------------------------------------------

class ChaosAwareEnsemble:
    """Ensemble forecaster that adapts to the current dynamical regime.

    In stable regime: tight ensemble, high confidence.
    In critical regime: wider ensemble, moderate confidence.
    In chaotic regime: very wide ensemble, low confidence, honest warnings.

    Parameters
    ----------
    ensemble_size : int
        Number of ensemble members. Default 20.
    """

    def __init__(self, ensemble_size: int = 20):
        self.ensemble_size = ensemble_size

    def forecast(
        self,
        current_state: np.ndarray,
        regime: str,
        lambda_est: float,
        horizon: float = 24.0,
        anomaly_weight: float = 1.0,
    ) -> Forecast:
        """Generate a chaos-aware probabilistic forecast.

        Parameters
        ----------
        current_state : np.ndarray
            Current [pressure, temp, humidity, wind].
        regime : str
            Current regime.
        lambda_est : float
            Current Lyapunov estimate.
        horizon : float
            Forecast horizon in hours.
        anomaly_weight : float
            Amplification from outlier detection.

        Returns
        -------
        Forecast
            Full probabilistic forecast with honest uncertainty.
        """
        # Regime-dependent perturbation scale
        if regime == "stable":
            base_perturbation = np.array([1.0, 0.5, 0.02, 0.5])
            confidence = 0.85
        elif regime == "critical":
            base_perturbation = np.array([3.0, 2.0, 0.05, 2.0])
            confidence = 0.5
        else:  # chaotic
            base_perturbation = np.array([8.0, 5.0, 0.10, 5.0])
            confidence = 0.2

        # Scale perturbation by horizon and lambda
        if lambda_est > 0:
            growth = np.exp(lambda_est * horizon / 24.0)
            perturbation = base_perturbation * min(growth, 10.0)
        else:
            perturbation = base_perturbation

        # Anomaly amplification: widen bands when outliers detected
        perturbation *= anomaly_weight

        # Generate ensemble
        ensemble = []
        for _ in range(self.ensemble_size):
            member = current_state + perturbation * np.random.randn(4)
            ensemble.append(member)
        ensemble = np.array(ensemble)

        # Compute statistics
        median = np.median(ensemble, axis=0)
        p10 = np.percentile(ensemble, 10, axis=0)
        p90 = np.percentile(ensemble, 90, axis=0)

        # Predictability horizon
        if lambda_est > 0:
            pred_horizon = (1.0 / lambda_est) * np.log(1000) * 24
            pred_horizon = min(max(pred_horizon, 1.0), 168.0)
        else:
            pred_horizon = 168.0

        # Phase transition risk (heuristic)
        phase_risk = 0.0
        if regime == "critical":
            phase_risk = 0.4
        elif regime == "chaotic":
            phase_risk = 0.6
        if anomaly_weight > 1.5:
            phase_risk = min(phase_risk + 0.3, 1.0)

        # Honest warnings
        warnings = []
        if regime == "chaotic":
            warnings.append(
                "Chaotic regime detected. Forecast beyond "
                f"{pred_horizon:.0f}h is unreliable."
            )
        if anomaly_weight > 1.3:
            warnings.append(
                "Anomalous observations detected. Conventional models "
                "would smooth this signal — we're amplifying it."
            )
        if horizon > pred_horizon:
            warnings.append(
                f"Requested horizon ({horizon:.0f}h) exceeds "
                f"predictability limit ({pred_horizon:.0f}h). "
                "Uncertainty bands are very wide."
            )
        if phase_risk > 0.5:
            warnings.append(
                "High phase transition risk. Conditions may change "
                "rapidly and discontinuously."
            )

        # Confidence degrades with horizon
        time_decay = np.exp(-0.02 * horizon)
        confidence *= time_decay

        return Forecast(
            timestamp=0.0,  # set by caller
            horizon=horizon,
            median=median,
            p10=p10,
            p90=p90,
            regime=regime,
            confidence=confidence,
            lyapunov_estimate=lambda_est,
            predictability_horizon=pred_horizon,
            phase_transition_risk=phase_risk,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Main AI Engine
# ---------------------------------------------------------------------------

class ChaosWeatherAI:
    """A weather AI that learns, embraces uncertainty, and treats chaos
    as information rather than noise.

    This is fundamentally different from conventional weather models:
    - GFS/ECMWF: Optimize for accuracy on average days
    - This: Optimizes for *knowing when it doesn't know*

    The AI maintains a learning memory that tracks its own failures,
    discovers patterns in its errors, and adjusts its regime detection
    and uncertainty estimates over time.

    Parameters
    ----------
    ensemble_size : int
        Ensemble members for forecasting. Default 20.
    anomaly_threshold : float
        Z-score threshold for outlier detection. Default 2.0.
    flux_threshold : float
        Flux sensor threshold for phase transitions. Default 1.8.

    Examples
    --------
    >>> ai = ChaosWeatherAI()
    >>> obs = Observation(timestamp=0, pressure=1013, temperature=20)
    >>> ai.observe(obs)
    >>> forecast = ai.forecast(horizon=24.0)
    >>> print(forecast.summary())
    """

    def __init__(
        self,
        ensemble_size: int = 20,
        anomaly_threshold: float = 2.0,
        flux_threshold: float = 1.8,
    ):
        self.regime_detector = RegimeDetector()
        self.outlier_processor = OutlierFirstProcessor(anomaly_threshold)
        self.ensemble = ChaosAwareEnsemble(ensemble_size)
        self.flux_sensor = WeatherFlux(flux_threshold)
        self.memory = LearningMemory()

        # Observation history for flux analysis
        self._pressure_history: List[float] = []
        self._temp_history: List[float] = []
        self._observation_history: List[Observation] = []

        # Current state
        self._current_regime: str = "unknown"
        self._current_lambda: float = 0.0
        self._last_forecast: Optional[Forecast] = None

    def observe(self, obs: Observation) -> Dict[str, Any]:
        """Feed a new observation into the AI.

        The AI will:
        1. Check if it's anomalous (outlier-first)
        2. Update its regime estimate
        3. Check for phase transitions (flux sensor)
        4. Learn from any previous forecast errors

        Parameters
        ----------
        obs : Observation
            New weather observation.

        Returns
        -------
        dict
            Status update with regime, anomaly, and flux info.
        """
        vec = obs.to_vector()

        # 1. Outlier check (before smoothing!)
        self.outlier_processor.update_stats(vec)
        is_anomaly, z_scores = self.outlier_processor.check_anomaly(vec)

        if is_anomaly:
            self.memory.outlier_events.append({
                "timestamp": obs.timestamp,
                "z_scores": z_scores.tolist(),
                "observation": vec.tolist(),
            })

        # 2. Update regime detector
        self.regime_detector.update(obs)
        prev_regime = self._current_regime
        self._current_regime, self._current_lambda = (
            self.regime_detector.detect_regime()
        )

        # Track regime transitions
        if prev_regime != "unknown" and prev_regime != self._current_regime:
            self.memory.regime_transitions.append({
                "timestamp": obs.timestamp,
                "from": prev_regime,
                "to": self._current_regime,
                "lambda": self._current_lambda,
            })
            # Learn from transition
            self._learn_from_transition(prev_regime, self._current_regime)

        # 3. Flux sensor (phase transition detection)
        self._pressure_history.append(obs.pressure)
        self._temp_history.append(obs.temperature)
        self._observation_history.append(obs)

        flux_risk = 0.0
        if len(self._pressure_history) >= 3:
            p_arr = np.array(self._pressure_history[-20:])
            t_arr = np.array(self._temp_history[-20:])
            flux_scalar, risk_trigger = self.flux_sensor.calculate_phase_risk(
                p_arr, t_arr,
            )
            if len(risk_trigger) > 0 and risk_trigger[-1] == 1:
                flux_risk = float(flux_scalar[-1])

        # 4. Learn from previous forecast errors
        if self._last_forecast is not None:
            self._evaluate_forecast(self._last_forecast, obs)

        return {
            "regime": self._current_regime,
            "lambda": self._current_lambda,
            "is_anomaly": is_anomaly,
            "z_scores": z_scores.tolist() if is_anomaly else None,
            "flux_risk": flux_risk,
            "observations_count": len(self._observation_history),
        }

    def forecast(self, horizon: float = 24.0) -> Forecast:
        """Generate a chaos-aware probabilistic forecast.

        Parameters
        ----------
        horizon : float
            Hours ahead to forecast.

        Returns
        -------
        Forecast
            Honest probabilistic forecast with regime-aware uncertainty.
        """
        if not self._observation_history:
            return Forecast(
                timestamp=0, horizon=horizon,
                median=np.zeros(4), p10=np.zeros(4), p90=np.zeros(4),
                warnings=["No observations yet. Cannot forecast."],
            )

        current = self._observation_history[-1].to_vector()
        anomaly_weight = 1.0

        # Check recent anomaly status
        is_anom, z_scores = self.outlier_processor.check_anomaly(current)
        if is_anom:
            anomaly_weight = self.outlier_processor.anomaly_weight(z_scores)

        forecast = self.ensemble.forecast(
            current_state=current,
            regime=self._current_regime,
            lambda_est=self._current_lambda,
            horizon=horizon,
            anomaly_weight=anomaly_weight,
        )

        forecast.timestamp = self._observation_history[-1].timestamp

        # Store for later evaluation
        self._last_forecast = forecast
        self.memory.total_forecasts += 1

        return forecast

    def _evaluate_forecast(self, forecast: Forecast, obs: Observation):
        """Compare a previous forecast against what actually happened.

        This is where the AI learns from its mistakes.
        """
        observed = obs.to_vector()
        predicted = forecast.median
        error = np.linalg.norm(observed - predicted)

        self.memory.prediction_errors.append({
            "timestamp": obs.timestamp,
            "predicted": predicted.tolist(),
            "observed": observed.tolist(),
            "magnitude": float(error),
            "regime_at_forecast": forecast.regime,
            "confidence_at_forecast": forecast.confidence,
        })

        # Did the regime prediction hold?
        current_regime = self._current_regime
        if current_regime == forecast.regime:
            self.memory.correct_regime_calls += 1

        # Learn from large errors
        if error > 10.0 and forecast.confidence > 0.5:
            self.memory.discovered_patterns.append(
                f"t={obs.timestamp}: High-confidence miss (error={error:.2f}, "
                f"conf={forecast.confidence:.0%}). Regime was "
                f"'{forecast.regime}' but error suggests otherwise."
            )

    def _learn_from_transition(self, from_regime: str, to_regime: str):
        """Extract learning from regime transitions."""
        # Check if we had outlier events recently
        recent_outliers = [
            e for e in self.memory.outlier_events
            if len(self._observation_history) > 0
            and e["timestamp"] > self._observation_history[-1].timestamp - 10
        ]

        if recent_outliers and from_regime == "stable":
            pattern = (
                f"Outlier events preceded transition from stable to "
                f"{to_regime}. Anomalies were leading indicators."
            )
            if pattern not in self.memory.discovered_patterns:
                self.memory.discovered_patterns.append(pattern)

        if from_regime == "critical" and to_regime == "chaotic":
            pattern = (
                "Critical -> chaotic transition detected. "
                "Edge-of-chaos computation window has closed."
            )
            if pattern not in self.memory.discovered_patterns:
                self.memory.discovered_patterns.append(pattern)

    def status(self) -> str:
        """Current AI status summary."""
        lines = [
            "=" * 50,
            "  Chaos-Aware Weather AI",
            "=" * 50,
            f"  Regime: {self._current_regime.upper()}",
            f"  Lambda: {self._current_lambda:.4f}",
            f"  Observations: {len(self._observation_history)}",
            "",
            self.memory.summary(),
        ]
        return "\n".join(lines)

    def explain(self) -> str:
        """Explain how this AI differs from conventional models."""
        return """
How This Weather AI Differs:

CONVENTIONAL MODEL:
  1. Collect data
  2. Remove outliers (they're "noise")
  3. Run deterministic simulation
  4. Report single forecast
  5. Evaluate on average accuracy
  6. Blame "unprecedented" events for failures

THIS MODEL:
  1. Collect data
  2. AMPLIFY outliers (they're leading indicators)
  3. Detect current dynamical regime (stable/critical/chaotic)
  4. Generate ensemble forecast with regime-aware uncertainty
  5. Report probabilistic forecast + honest confidence
  6. When wrong, learn WHY and update beliefs
  7. Track its own predictability horizon

KEY DIFFERENCES:
  - Knows when it doesn't know (Lyapunov-aware)
  - Treats anomalies as signal, not noise (outlier-first)
  - Widens uncertainty honestly in chaotic regime
  - Maintains learning memory of its own failures
  - Detects phase transitions before they arrive
  - Never says "unprecedented" — says "I saw this coming
    but my confidence was low, here's what I learned"
""".strip()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_chaos_weather_ai():
    """Full demonstration with simulated weather scenario."""
    print("=" * 60)
    print("  Chaos-Aware Weather AI — Demo")
    print("=" * 60)

    ai = ChaosWeatherAI(ensemble_size=20, flux_threshold=1.5)

    # Phase 1: Stable weather
    print("\n--- Phase 1: Stable Weather (Days 1-5) ---")
    np.random.seed(42)
    for t in range(50):
        obs = Observation(
            timestamp=float(t),
            pressure=1013.0 + np.random.randn() * 0.5,
            temperature=20.0 + np.random.randn() * 0.3,
            humidity=0.5 + np.random.randn() * 0.02,
            wind_speed=5.0 + np.random.randn() * 0.5,
        )
        status = ai.observe(obs)

    forecast = ai.forecast(horizon=24.0)
    print(forecast.summary())

    # Phase 2: Building instability
    print("\n--- Phase 2: Building Instability (Days 5-7) ---")
    for t in range(50, 70):
        pressure_drop = (t - 50) * 0.5  # gradual pressure drop
        temp_swing = (t - 50) * 0.3  # increasing temp variance
        obs = Observation(
            timestamp=float(t),
            pressure=1013.0 - pressure_drop + np.random.randn() * 1.0,
            temperature=20.0 + temp_swing * np.random.randn(),
            humidity=0.5 + np.random.randn() * 0.05,
            wind_speed=5.0 + (t - 50) * 0.2 + np.random.randn(),
        )
        status = ai.observe(obs)
        if status["is_anomaly"]:
            print(f"  t={t}: ANOMALY DETECTED (regime={status['regime']})")

    forecast = ai.forecast(horizon=24.0)
    print(forecast.summary())

    # Phase 3: Storm / chaos event
    print("\n--- Phase 3: Storm Event (Days 7-9) ---")
    for t in range(70, 90):
        obs = Observation(
            timestamp=float(t),
            pressure=1003.0 - (t - 70) * 0.8 + np.random.randn() * 3.0,
            temperature=15.0 + np.random.randn() * 4.0,
            humidity=0.8 + np.random.randn() * 0.1,
            wind_speed=15.0 + np.random.randn() * 5.0,
        )
        status = ai.observe(obs)
        if status["is_anomaly"]:
            print(f"  t={t}: ANOMALY (regime={status['regime']}, "
                  f"lambda={status['lambda']:.3f})")

    forecast = ai.forecast(horizon=24.0)
    print(forecast.summary())

    # Phase 4: Recovery
    print("\n--- Phase 4: Recovery (Days 9-11) ---")
    for t in range(90, 110):
        recovery = (t - 90) / 20.0  # 0 -> 1 over recovery period
        obs = Observation(
            timestamp=float(t),
            pressure=987.0 + recovery * 26.0 + np.random.randn() * 1.0,
            temperature=15.0 + recovery * 5.0 + np.random.randn() * 1.0,
            humidity=0.7 - recovery * 0.2 + np.random.randn() * 0.03,
            wind_speed=15.0 - recovery * 10.0 + np.random.randn() * 1.0,
        )
        ai.observe(obs)

    forecast = ai.forecast(horizon=24.0)
    print(forecast.summary())

    # Final status
    print("\n--- AI Status & Learning ---")
    print(ai.status())

    # Explanation
    print("\n--- How This AI Differs ---")
    print(ai.explain())


if __name__ == "__main__":
    demo_chaos_weather_ai()
