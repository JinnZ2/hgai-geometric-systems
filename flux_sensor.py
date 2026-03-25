"""
Weather-Flux Sensor — Phase Transition Detection

Sovereign sensor for identifying rapid atmospheric phase transitions.
Treats "noise" as a leading indicator of entropy events rather than
filtering it out as institutional models do.

Core Principle:
    If the temporal derivative of entropy spikes, the system is
    undergoing a phase transition. Stability models that smooth
    these signals are optimizing for a world that no longer exists.

Integrates with:
    - sovereign_impact_sensor.py (I_e system stress scalar)
    - weather_node_network.py (ensemble pipeline)
"""

from typing import Optional, Tuple

import numpy as np


class WeatherFlux:
    """Sovereign sensor for identifying rapid atmospheric phase transitions.

    Monitors the temporal derivative of entropy by computing coupled
    pressure-temperature flux. High flux values indicate imminent
    "stability breakdown" -- the phase transitions that conventional
    models dismiss as noise.

    Parameters
    ----------
    flux_threshold : float
        Threshold for the flux scalar. Values above this trigger a
        phase transition warning. Default 1.8.
    window_size : int
        Rolling window for smoothed flux detection. Helps distinguish
        sustained transitions from single-point spikes. Default 3.
    """

    def __init__(self, flux_threshold: float = 1.8, window_size: int = 3):
        self.flux_threshold = flux_threshold
        self.window_size = window_size

    def calculate_phase_risk(
        self,
        pressure_series: np.ndarray,
        temp_series: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect non-linear spikes where stability models fail.

        Computes the coupled energy flux between pressure and temperature
        derivatives. When this flux exceeds the threshold, the "noise"
        is actually a phase transition forming.

        Parameters
        ----------
        pressure_series : np.ndarray
            Time series of atmospheric pressure readings.
        temp_series : np.ndarray
            Time series of temperature readings (same length).

        Returns
        -------
        flux_scalar : np.ndarray
            Absolute coupled flux at each transition point (length n-1).
        risk_trigger : np.ndarray
            Binary array: 1 where flux exceeds threshold, 0 otherwise.

        Raises
        ------
        ValueError
            If series lengths don't match or are too short.
        """
        if len(pressure_series) != len(temp_series):
            raise ValueError(
                "Pressure and temperature series must have equal length."
            )
        if len(pressure_series) < 2:
            raise ValueError("Need at least 2 data points for flux calculation.")

        # Rate of change in pressure and temperature
        delta_p = np.diff(pressure_series)
        delta_t = np.diff(temp_series)

        # Flux Scalar: coupled energy derivative
        # High values = imminent stability breakdown
        flux_scalar = np.abs(delta_p * delta_t)

        # Leading Indicator: is the "noise" actually a phase transition?
        risk_trigger = np.where(flux_scalar > self.flux_threshold, 1, 0)

        return flux_scalar, risk_trigger

    def calculate_entropy_rate(
        self,
        pressure_series: np.ndarray,
        temp_series: np.ndarray,
        humidity_series: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the temporal derivative of system entropy.

        Uses pressure, temperature, and optionally humidity to estimate
        the rate of entropy change. Spikes in this derivative are the
        leading indicator that institutional models treat as noise.

        Parameters
        ----------
        pressure_series : np.ndarray
            Time series of atmospheric pressure.
        temp_series : np.ndarray
            Time series of temperature.
        humidity_series : np.ndarray, optional
            Time series of relative humidity. If provided, adds a
            moisture coupling term to the entropy estimate.

        Returns
        -------
        entropy_rate : np.ndarray
            Temporal derivative of estimated system entropy (length n-1).
        """
        if len(pressure_series) != len(temp_series):
            raise ValueError(
                "Pressure and temperature series must have equal length."
            )

        # Thermodynamic entropy proxy: S ~ Cv * ln(T) + R * ln(V)
        # Using P as inverse proxy for volume (ideal gas)
        temp_safe = np.maximum(temp_series, 1e-6)
        pressure_safe = np.maximum(pressure_series, 1e-6)

        entropy_proxy = np.log(temp_safe) - np.log(pressure_safe)

        if humidity_series is not None:
            if len(humidity_series) != len(temp_series):
                raise ValueError(
                    "Humidity series must match pressure/temp length."
                )
            # Moisture coupling: latent heat contribution
            humidity_safe = np.clip(humidity_series, 0.01, 1.0)
            entropy_proxy += 0.5 * np.log(humidity_safe)

        # Temporal derivative
        entropy_rate = np.diff(entropy_proxy)
        return entropy_rate

    def rolling_flux_detection(
        self,
        pressure_series: np.ndarray,
        temp_series: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect sustained phase transitions using a rolling window.

        Single-point spikes may be sensor noise. Sustained elevated
        flux over the rolling window indicates a true phase transition
        that stability models will miss.

        Parameters
        ----------
        pressure_series : np.ndarray
            Time series of atmospheric pressure.
        temp_series : np.ndarray
            Time series of temperature.

        Returns
        -------
        rolling_flux : np.ndarray
            Rolling mean of flux scalar.
        sustained_risk : np.ndarray
            Binary array: 1 where rolling flux exceeds threshold.
        """
        flux_scalar, _ = self.calculate_phase_risk(pressure_series, temp_series)

        if len(flux_scalar) < self.window_size:
            # Not enough data for rolling window; fall back to raw
            sustained_risk = np.where(flux_scalar > self.flux_threshold, 1, 0)
            return flux_scalar, sustained_risk

        # Rolling mean via convolution
        kernel = np.ones(self.window_size) / self.window_size
        rolling_flux = np.convolve(flux_scalar, kernel, mode="valid")
        sustained_risk = np.where(rolling_flux > self.flux_threshold, 1, 0)

        return rolling_flux, sustained_risk


def demo_flux_sensor():
    """Demonstrate the flux sensor detecting a phase transition.

    Simulates a scenario where a stable forecast masks an incoming
    blizzard. The flux sensor detects the transition before it arrives.
    """
    print("=" * 55)
    print("Weather-Flux Sensor -- Phase Transition Detection")
    print("=" * 55)

    # Simulated readings: stable period, then rapid transition
    np.random.seed(42)
    n_stable = 10
    n_transition = 5
    n_event = 5

    # Pressure: stable -> rapid drop (storm approaching)
    p_stable = 1013.0 + np.random.randn(n_stable) * 0.5
    p_transition = np.linspace(1013, 995, n_transition) + np.random.randn(n_transition) * 0.3
    p_event = 995.0 + np.random.randn(n_event) * 1.0
    pressure = np.concatenate([p_stable, p_transition, p_event])

    # Temperature: stable -> sharp drop (cold front)
    t_stable = 2.0 + np.random.randn(n_stable) * 0.3
    t_transition = np.linspace(2, -15, n_transition) + np.random.randn(n_transition) * 0.5
    t_event = -15.0 + np.random.randn(n_event) * 1.0
    temperature = np.concatenate([t_stable, t_transition, t_event])

    sensor = WeatherFlux(flux_threshold=1.8, window_size=3)

    # Point-wise flux detection
    flux_scalar, risk_trigger = sensor.calculate_phase_risk(pressure, temperature)
    print(f"\nFlux scalar range: [{flux_scalar.min():.2f}, {flux_scalar.max():.2f}]")
    print(f"Risk triggers: {risk_trigger.sum()} / {len(risk_trigger)} time steps")

    # Entropy rate
    entropy_rate = sensor.calculate_entropy_rate(pressure, temperature)
    print(f"Entropy rate range: [{entropy_rate.min():.4f}, {entropy_rate.max():.4f}]")

    # Rolling detection
    rolling_flux, sustained_risk = sensor.rolling_flux_detection(pressure, temperature)
    print(f"Sustained risk triggers: {sustained_risk.sum()} / {len(sustained_risk)} windows")

    # Identify transition onset
    if risk_trigger.sum() > 0:
        first_trigger = np.argmax(risk_trigger)
        print(f"\nFirst phase transition signal at time step {first_trigger}")
        print(
            f"Flux value: {flux_scalar[first_trigger]:.2f} "
            f"(threshold: {sensor.flux_threshold})"
        )
        print("Stability models would still report 'clear conditions' here.")
    else:
        print("\nNo phase transitions detected in this series.")


if __name__ == "__main__":
    demo_flux_sensor()
