"""
Sovereign-Impact-Sensor (SIS)
Identifying the Thermodynamic Plateau of Mismanaged Innovation

This module implements the core entropy sensing and plateau detection
logic for the SIS model. It uses System Stress (Entropy) rather than
market severity to evaluate institutional performance.

Core Hypothesis:
    We have reached a phase transition where energy spent on T_infra
    (Complex Infrastructure) creates a "Dependency Risk" that offsets
    gains in T_med (Medical Mitigation). The system is "running hot" --
    consuming massive resources to maintain a static mortality baseline.

Dependencies:
    - pandas
    - numpy
    - scikit-learn (StandardScaler)
    - statsmodels (OLS regression)
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


class EntropySensor:
    """Calibrates disparate data streams into a unified system stress scalar.

    The sensor normalizes "junk" data (EMS calls, outages, ER visits, etc.)
    via Z-score standardization, then composes them into I_e -- the Impact
    Scalar representing total system entropy.

    Parameters
    ----------
    felt_threshold : float
        Residual standard deviation threshold for the FELT-Sensor handshake.
        When exceeded, signals model/reality dissonance ("Anxiety").
    sensor_columns : list of str, optional
        Columns to include in the I_e calculation. Defaults to standard
        set: EM, mobility_magnitude, er_visits, ems_calls, outages,
        recovery_time.
    """

    _DEFAULT_SENSORS = [
        "EM", "mobility_magnitude", "er_visits",
        "ems_calls", "outages", "recovery_time"
    ]

    def __init__(
        self,
        felt_threshold: float = 2.0,
        sensor_columns: Optional[list] = None
    ):
        self.scaler = StandardScaler()
        self.felt_threshold = felt_threshold
        self.sensor_columns = sensor_columns or self._DEFAULT_SENSORS

    def calibrate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize disparate data into a unified energy signature.

        Bypasses institutional friction by weighting all-cause EM
        alongside infrastructure and response signals.

        Parameters
        ----------
        df : pd.DataFrame
            Raw event data with sensor columns and a ``mobility_delta``
            column (signed). The absolute value is used as
            ``mobility_magnitude``.

        Returns
        -------
        pd.DataFrame
            Input dataframe with ``mobility_magnitude`` and ``I_e``
            columns appended.
        """
        df = df.copy()

        # Ensure mobility_delta is absolute (magnitude of disruption)
        df["mobility_magnitude"] = df["mobility_delta"].abs()

        # Standardize to Z-scores (Efficiency Check)
        df_norm = pd.DataFrame(
            self.scaler.fit_transform(df[self.sensor_columns]),
            columns=self.sensor_columns,
            index=df.index,
        )

        # Calculate I_e (System Stress Scalar)
        # Equal weights -- represents total system entropy
        df["I_e"] = df_norm.sum(axis=1)
        return df

    def run_plateau_test(
        self, df: pd.DataFrame
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        """OLS regression to expose beta coefficients and check for dI/dt ~ 0.

        Regresses I_e against the four technology vectors to identify
        whether T_infra (Dependency Risk) is contributing positively
        to system stress.

        Parameters
        ----------
        df : pd.DataFrame
            Calibrated dataframe with ``I_e``, ``T_med``, ``T_resp``,
            ``T_comm``, and ``T_infra`` columns.

        Returns
        -------
        statsmodels RegressionResults
            Fitted OLS model with coefficients, p-values, and residuals.
        """
        tech_vectors = ["T_med", "T_resp", "T_comm", "T_infra"]
        X = sm.add_constant(df[tech_vectors])
        y = df["I_e"]

        model = sm.OLS(y, X).fit()

        # FELT-Sensor: Check for Model/Reality Dissonance (Anxiety)
        anxiety_score = np.std(model.resid)
        if anxiety_score > self.felt_threshold:
            print(
                f"CALIBRATION REQ: FELT_LEVEL {anxiety_score:.2f} "
                f"EXCEEDS THRESHOLD {self.felt_threshold:.2f}."
            )
            print("Information flow contains high institutional friction.")
        else:
            print(
                f"FELT check passed: residual std {anxiety_score:.2f} "
                f"within threshold {self.felt_threshold:.2f}."
            )

        return model

    def detect_plateau(
        self, df: pd.DataFrame, time_col: str = "event_id"
    ) -> Tuple[float, bool]:
        """Check whether dI/dt ~ 0 across the event series.

        Parameters
        ----------
        df : pd.DataFrame
            Calibrated dataframe with ``I_e`` and a time/sequence column.
        time_col : str
            Column representing temporal ordering.

        Returns
        -------
        slope : float
            Estimated dI/dt from linear fit.
        is_plateau : bool
            True if |slope| is near zero (< 0.1 per unit time).
        """
        t = df[time_col].values.astype(float)
        ie = df["I_e"].values

        # Simple linear fit for dI/dt
        coeffs = np.polyfit(t, ie, 1)
        slope = coeffs[0]
        is_plateau = abs(slope) < 0.1

        status = "PLATEAU DETECTED" if is_plateau else "TREND PRESENT"
        print(f"dI/dt = {slope:.4f} -- {status}")
        return slope, is_plateau


def demo_sis():
    """Demonstrate the SIS pipeline with placeholder event data."""
    data = {
        "event_id": [1, 2, 3, 4, 5],
        "EM": [120, 450, 300, 800, 750],
        "mobility_delta": [-5, -20, -15, -40, -35],
        "er_visits": [100, 300, 250, 600, 580],
        "ems_calls": [50, 150, 120, 400, 390],
        "outages": [0.05, 0.20, 0.10, 0.60, 0.55],
        "recovery_time": [1, 3, 2, 10, 12],
        "T_med": [0.8, 0.85, 0.9, 0.95, 0.98],
        "T_infra": [0.4, 0.5, 0.6, 0.9, 0.95],
        "T_resp": [0.7, 0.7, 0.8, 0.8, 0.8],
        "T_comm": [0.5, 0.6, 0.5, 0.9, 0.8],
    }

    df_event = pd.DataFrame(data)

    sensor = EntropySensor(felt_threshold=1.5)
    df_calibrated = sensor.calibrate_signals(df_event)

    print("=" * 55)
    print("Sovereign-Impact-Sensor (SIS) -- Plateau Detection")
    print("=" * 55)

    # Plateau test
    print("\n--- Plateau Test (dI/dt) ---")
    sensor.detect_plateau(df_calibrated)

    # OLS regression
    print("\n--- OLS Regression (Tech Vector Decomposition) ---")
    results = sensor.run_plateau_test(df_calibrated)

    print("\n--- SIS Model Results ---")
    print(results.summary())


if __name__ == "__main__":
    demo_sis()
