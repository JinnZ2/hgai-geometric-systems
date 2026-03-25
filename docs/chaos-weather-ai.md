# Chaos-Aware Weather AI

## What It Is

A weather intelligence system that fundamentally differs from conventional models. Instead of optimizing for accuracy on average days (and failing on extreme days), it optimizes for *knowing when it doesn't know*.

## How Conventional Models Work

```
1. Collect data
2. Remove outliers ("noise")
3. Run deterministic simulation
4. Report single forecast
5. Evaluate on average accuracy
6. Blame "unprecedented" events for failures
```

## How This Model Works

```
1. Collect data
2. AMPLIFY outliers (they're leading indicators)
3. Detect current dynamical regime (stable/critical/chaotic)
4. Generate ensemble forecast with regime-aware uncertainty
5. Report probabilistic forecast + honest confidence
6. When wrong, learn WHY and update beliefs
7. Track its own predictability horizon
```

---

## Architecture

```
Observation Stream
      |
      v
+------------------+     +------------------+
| Outlier-First    |     | Regime Detector  |
| Processor        |     | (Lyapunov est.)  |
| - anomaly check  |     | - stable/crit/   |
| - z-scores       |     |   chaotic        |
| - amplify signal |     | - predict horizon|
+------------------+     +------------------+
      |                         |
      v                         v
+------------------+     +------------------+
| Flux Sensor      |     | Chaos-Aware      |
| (phase trans.)   |---->| Ensemble         |
| - P*T coupling   |     | - regime-scaled  |
| - risk trigger   |     |   perturbations  |
+------------------+     | - honest bounds  |
                          +------------------+
                                |
                                v
                          +------------------+
                          | Forecast         |
                          | - median + p10/90|
                          | - confidence     |
                          | - warnings       |
                          | - pred. horizon  |
                          +------------------+
                                |
                                v
                          +------------------+
                          | Learning Memory  |
                          | - error tracking |
                          | - regime accuracy|
                          | - discoveries    |
                          +------------------+
```

---

## Components

### Regime Detector
Estimates the local Lyapunov exponent from consecutive observations. Classifies the atmosphere into three regimes:

| Regime | Lambda | Meaning | Forecast Strategy |
|--------|--------|---------|-------------------|
| Stable | lambda < 0 | Predictable | Tight ensemble, high confidence |
| Critical | lambda ~ 0 | Edge of chaos | Wider ensemble, moderate confidence |
| Chaotic | lambda > 0 | Diverging | Very wide ensemble, low confidence, honest warnings |

### Outlier-First Processor
The opposite of conventional smoothing:
- Conventional: Detect outlier -> Remove -> Predict on smoothed data
- This: Detect outlier -> Amplify weight -> Investigate -> Learn

Higher z-score = MORE important to the forecast, not less.

### Chaos-Aware Ensemble
Adapts perturbation scale to the current regime:
- Stable: base perturbation, 85% confidence
- Critical: 3x perturbation, 50% confidence
- Chaotic: 8x perturbation, 20% confidence + growth factor

When lambda > 0, uncertainty bands grow exponentially with forecast horizon.

### Flux Sensor Integration
Uses the `WeatherFlux` sensor from `flux_sensor.py` to detect rapid phase transitions in pressure-temperature coupling. When flux exceeds threshold, phase transition risk increases.

### Learning Memory
Tracks:
- **Prediction errors**: What it predicted vs what happened
- **Regime transitions**: When the atmosphere changed regime
- **Outlier events**: Anomalies that conventional models would have smoothed away
- **Discovered patterns**: What the AI learned from its mistakes

---

## Usage

```python
from chaos_weather_ai import ChaosWeatherAI, Observation

ai = ChaosWeatherAI()

# Feed observations
obs = Observation(timestamp=0, pressure=1013, temperature=20, humidity=0.5)
status = ai.observe(obs)
# status: {'regime': 'stable', 'lambda': -0.02, 'is_anomaly': False, ...}

# Get forecast
forecast = ai.forecast(horizon=24.0)
print(forecast.summary())
# Shows: regime, confidence, median + uncertainty bands, warnings

# Check what it's learned
print(ai.status())
print(ai.memory.summary())

# Understand the philosophy
print(ai.explain())
```

---

## Key Insight

The blizzard-on-a-clear-forecast problem happens because conventional models:
1. Smooth away the outlier signals that precede the transition
2. Don't know their own predictability horizon
3. Report confidence as if lambda is always negative

This model:
1. Amplifies the outlier signals
2. Estimates lambda in real time and adjusts
3. Says "I'm only 20% confident because we're in chaotic regime, and my predictability horizon is 6 hours, not 48"

That's the difference between a model that says "clear skies" and a model that says "I don't have enough confidence to promise you clear skies — here's why."

---

## Connection to the HGAI Framework

| Module | How It Connects |
|--------|----------------|
| `lyapunov_spectrum.py` | Regime detection theory, lambda estimation |
| `flux_sensor.py` | Phase transition detection (integrated directly) |
| `sovereign_impact_sensor.py` | System stress / entropy calibration |
| `resilience/detectors.py` | Scan forecast text for institutional friction |
| `weather_node_network.py` | Ensemble pipeline architecture |
| `hgai.py` | Feed forecast text through unified engine |
| `framework/core/m_s_calculator.py` | M(S) scoring of forecast system health |
