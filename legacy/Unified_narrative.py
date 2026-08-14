#!/usr/bin/env python3
"""
Unified Geometric Narrative Monitor
- Modular encoders that map text -> 64D geometry
- GeometryPacket container
- Fracture detector with observer metric tensor support
- ASCII octahedral visualizer (text-only, phone-friendly)
- Unified monitor: M(S) viability, time-series, trajectory, assessments

Dependencies: numpy
Run on phone-friendly Python REPLs (Termux, Pythonista, Pyto)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import math
import time

# -------------------------------
# Utilities
# -------------------------------

def safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v) + 1e-12)

def normalize(v: np.ndarray) -> np.ndarray:
    n = safe_norm(v)
    return v / n

def angle_between(v1: np.ndarray, v2: np.ndarray, metric: Optional[np.ndarray] = None) -> float:
    """
    Compute angular distance between v1 and v2 under metric.
    Metric should be a symmetric positive-definite matrix (64x64).
    If metric is None, use Euclidean dot product.
    Returns angle in radians.
    """
    v1 = v1.astype(float)
    v2 = v2.astype(float)
    if metric is None:
        num = np.dot(v1, v2)
        den = safe_norm(v1) * safe_norm(v2)
    else:
        # metric inner product: <u,v>_g = u^T G v
        num = float(v1 @ (metric @ v2))
        den = math.sqrt(float(v1 @ (metric @ v1)) * float(v2 @ (metric @ v2)) + 1e-18)
    cosv = float(np.clip(num / (den + 1e-12), -1.0, 1.0))
    return math.acos(cosv)

# -------------------------------
# Simple Lexicons & Patterns (tunable)
# -------------------------------

_POSITIVE_WORDS = set(["good","great","happy","joy","expand","expandin","open","integrate","integrating","connect","connected","safe","safe-"])
_NEGATIVE_WORDS = set(["bad","sad","angry","shrink","shutting","closed","isolate","isolation","fear","collapse","die-off","dieoff","die off","toxic"])
_AGENCY_VERBS = set(["do","make","act","drive","lead","push","create","build","force","control","decide","choose","take"])
_TEMPORAL_MARKERS = set(["then","after","before","when","while","during","since","because","therefore","so","next","later","previously","now"])
_EXISTENCE_MARKERS = set(["there is","there are","exists","present","presently","found","located","visible","absent","missing","gone"])

# -------------------------------
# Encoders (modular, 16-d each)
# -------------------------------
# Each returns a 16-d array. You can refine heuristics per index.
# For phone-friendly simplicity, we use counts + normalized features + small patterns.

def _tokenize(text: str) -> List[str]:
    text_l = text.lower()
    # basic tokenization: split on whitespace and punctuation
    tokens = []
    current = []
    for ch in text_l:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens

def encode_agency_octants(text: str) -> np.ndarray:
    """
    [0:16] agency-related features:
      - overall agentive verb density
      - first-person agency
      - passive indicators (less agency)
      - intensity (exclamation, caps)
      - conflict verbs
      - planning / future modal markers
      - leadership vs submission markers
      - action frequency (counts)
    """
    v = np.zeros(16, dtype=float)
    tokens = _tokenize(text)
    n = max(1, len(tokens))
    # density of agency verbs
    v[0] = sum(1 for t in tokens if t in _AGENCY_VERBS) / n
    # first-person presence
    v[1] = sum(1 for t in tokens if t in {"i","me","my","mine","we","our","us"}) / n
    # second-person projection (talking to others)
    v[2] = sum(1 for t in tokens if t in {"you","your","yours"}) / n
    # passive markers (be + past participle pattern naive)
    v[3] = 1.0 if "was" in tokens or "were" in tokens or "been" in tokens else 0.0
    # intensity: exclamation marks, ALLCAPS words length
    v[4] = (text.count("!") + sum(1 for w in text.split() if w.isupper() and len(w) > 1)) / (1 + n)
    # conflict/force verbs
    v[5] = sum(1 for t in tokens if t in {"force","push","pull","attack","destroy","resist"}) / n
    # planning modal verbs
    v[6] = sum(1 for t in tokens if t in {"will","shall","must","should","would","could","might"}) / n
    # agency magnitude proxy (verbs per sentence)
    v[7] = sum(1 for t in tokens if t.endswith("ing")) / n
    # filler features (future use / cross-terms)
    v[8:] = 0.0
    return v

def encode_valence_field(text: str) -> np.ndarray:
    """
    [16:32] valence / energy field: expansion vs contraction
      - positive word density
      - negative word density
      - 'expand' related words
      - 'contract' related words
      - social warmth
      - hostility
      - moralizing tone
    """
    v = np.zeros(16, dtype=float)
    tokens = _tokenize(text)
    n = max(1, len(tokens))
    v[0] = sum(1 for t in tokens if any(t.startswith(p) for p in _POSITIVE_WORDS)) / n
    v[1] = sum(1 for t in tokens if any(t.startswith(p) for p in _NEGATIVE_WORDS)) / n
    # expansion vs contraction keywords
    v[2] = sum(1 for t in tokens if t.startswith("expand") or t == "open") / n
    v[3] = sum(1 for t in tokens if t.startswith("shrink") or t == "close" or t == "closed" or t == "shutting") / n
    # social warmth terms (basic)
    v[4] = sum(1 for t in tokens if t in {"love","care","trust","support"}) / n
    v[5] = sum(1 for t in tokens if t in {"hate","fear","distrust","scare","angst"}) / n
    # moralizing tone (words like should, must, right/wrong)
    v[6] = sum(1 for t in tokens if t in {"should","must","right","wrong","ought"}) / n
    v[7] = text.count("?") / (1 + n)  # inquisitiveness
    v[8:] = 0.0
    return v

def encode_temporal_vectors(text: str) -> np.ndarray:
    """
    [32:48] temporal flow:
      - presence of causal connectors
      - sequential markers
      - tense indicators (past/present/future)
      - acceleration (words like 'rapid', 'sudden')
      - causality density
    """
    v = np.zeros(16, dtype=float)
    tokens = _tokenize(text)
    n = max(1, len(tokens))
    v[0] = sum(1 for t in tokens if t in _TEMPORAL_MARKERS) / n
    v[1] = sum(1 for t in tokens if t in {"past","ago","yesterday","former","previous"}) / n
    v[2] = sum(1 for t in tokens if t in {"future","tomorrow","later","soon","soonest"}) / n
    v[3] = text.count("rapid") + text.count("sudden") + text.count("quick")
    v[4] = sum(1 for t in tokens if t in {"because","therefore","so","hence","thus"}) / n
    v[5] = sum(1 for t in tokens if t in {"delay","lag","slow","gradual"}) / n
    # sequence adjectives, sentence count proxy
    v[6] = max(1, text.count(".") + text.count("!") + text.count("?"))
    v[7] = 0.0
    v[8:] = 0.0
    return v

def encode_existence_field(text: str) -> np.ndarray:
    """
    [48:64] presence/absence: objects, agents, features mentioned
      - named-object density (capitalized tokens simplified)
      - absence words
      - location hints
      - biodiversity mentions (animals, plants)
      - measurement mentions (numbers, degrees, km)
    """
    v = np.zeros(16, dtype=float)
    tokens = _tokenize(text)
    n = max(1, len(tokens))
    # crude named-object: tokens longer than 5 characters and capitalized in original text
    v[0] = sum(1 for w in text.split() if len(w) > 4 and w[0].isupper()) / max(1, len(text.split()))
    v[1] = sum(1 for t in tokens if any(t.startswith(p) for p in _EXISTENCE_MARKERS)) / n
    v[2] = sum(1 for t in tokens if t in {"river","forest","lake","tree","fish","bear","soil","rock","plant","animal"}) / n
    # numeric density
    v[3] = sum(1 for t in tokens if t.isdigit()) / n
    v[4] = sum(1 for t in tokens if any(ch.isdigit() for ch in t)) / n
    v[5] = text.count("%") + text.count("°") + text.count("km") + text.count("m")
    v[6] = sum(1 for t in tokens if t in {"present","absent","missing","gone","appear","appearances"}) / n
    v[7:] = 0.0
    return v

# Composite encoder
def narrative_to_geometry(text: str) -> np.ndarray:
    """
    Compose the four 16-d encoders into a 64-d normalized vector.
    """
    a = encode_agency_octants(text)
    b = encode_valence_field(text)
    c = encode_temporal_vectors(text)
    d = encode_existence_field(text)
    vec = np.concatenate([a, b, c, d]).astype(float)
    # small smoothing + normalization
    if np.allclose(vec, 0.0):
        # tiny noise to avoid all-zero vector
        vec = np.random.normal(scale=1e-6, size=vec.shape)
    return normalize(vec)

# -------------------------------
# Observer metric (64x64) helpers
# -------------------------------

def default_observer_metric(scale_factors: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Default metric: diagonal matrix of scale factors (observer anisotropy).
    scale_factors: 64-d array scaling each axis (default ones).
    """
    if scale_factors is None:
        scale_factors = np.ones(64, dtype=float)
    G = np.diag(np.asarray(scale_factors, dtype=float))
    return G

def build_observer_metric_from_profile(profile: Dict[str, float]) -> np.ndarray:
    """
    Build a diagonal metric from a tiny observer profile.
    profile map keys:
      - 'agency_bias' (positive values amplify 0:16 region)
      - 'valence_bias' amplify 16:32
      - 'temporal_bias' 32:48
      - 'existence_bias' 48:64
    Each value ~ [0.1 .. 3.0] multiplies the corresponding block.
    """
    scale = np.ones(64, dtype=float)
    a = profile.get("agency_bias", 1.0)
    b = profile.get("valence_bias", 1.0)
    c = profile.get("temporal_bias", 1.0)
    d = profile.get("existence_bias", 1.0)
    scale[0:16] *= a
    scale[16:32] *= b
    scale[32:48] *= c
    scale[48:64] *= d
    # optionally allow per-axis curvature tweak
    return default_observer_metric(scale)

# -------------------------------
# GeometryPacket & Fracture Detector
# -------------------------------

@dataclass
class GeometryPacket:
    self_vec: np.ndarray
    other_vec: np.ndarray
    field_vec: np.ndarray
    angles_rad: Dict[str, float] = field(default_factory=dict)
    fracture: Dict[str, Optional[object]] = field(default_factory=dict)  # detected, severity, axis
    metric: Optional[np.ndarray] = None

def compute_geometric_coherence(self_vec: np.ndarray, other_vec: np.ndarray, field_vec: np.ndarray,
                                metric: Optional[np.ndarray] = None) -> GeometryPacket:
    """
    Build GeometryPacket with angles and base fracture detection.
    """
    # normalize inputs defensively
    vs = normalize(self_vec)
    vo = normalize(other_vec)
    vf = normalize(field_vec)
    G = metric
    a_so = angle_between(vs, vo, metric=G)
    a_sf = angle_between(vs, vf, metric=G)
    a_of = angle_between(vo, vf, metric=G)
    angles = {'self_other': a_so, 'self_field': a_sf, 'other_field': a_of}
    # simple fracture rule: self_field angle > threshold (radians)
    threshold_rad = math.pi / 3.0  # default 60 degrees
    detected = a_sf > threshold_rad
    # severity normalized [0..1] from threshold to pi
    severity = max(0.0, min(1.0, (a_sf - threshold_rad) / (math.pi - threshold_rad))) if detected else 0.0
    # axis classifier (crude): find block with largest sign difference between self and field
    # compute block sums for 6 primary octants: +X,-X,+Y,-Y,+Z,-Z mapping
    # We map +X = agency positive (sum of first 8 indices), -X = agency negative (sum next 8)
    def block_signs(vec):
        # compress 64 to three signed scalars for X (agency), Y (valence), Z (temporal)
        Xp = np.sum(vec[0:8])   # agency first half
        Xn = np.sum(vec[8:16])  # agency second half
        Yp = np.sum(vec[16:24])
        Yn = np.sum(vec[24:32])
        Zp = np.sum(vec[32:40])
        Zn = np.sum(vec[40:48])
        # third axis (presence) we won't use in primary axis selection
        return np.array([Xp, -Xn, Yp, -Yn, Zp, -Zn], dtype=float)
    bs = block_signs(vs)
    bf = block_signs(vf)
    diff = bf - bs
    primary_idx = int(np.argmax(np.abs(diff)))
    axis_names = ["agency+","agency-","valence+","valence-","temporal+","temporal-"]
    primary_axis = axis_names[primary_idx]
    fracture = {
        "detected": bool(detected),
        "severity": float(severity),
        "self_field_angle_deg": float(a_sf * 180.0 / math.pi),
        "primary_axis": primary_axis
    }
    return GeometryPacket(self_vec=vs, other_vec=vo, field_vec=vf, angles_rad=angles, fracture=fracture, metric=G)

# -------------------------------
# M(S) viability & TimeSeries analyzer
# -------------------------------

@dataclass
class SystemMetrics:
    resonance: float
    adaptability: float
    diversity: float
    curiosity: float
    loss: float

class MSCalculator:
    @staticmethod
    def calculate(metrics: SystemMetrics) -> float:
        # small epsilon to avoid exact zero
        coherence = (metrics.resonance * (0.5 + metrics.adaptability) * (0.5 + metrics.diversity) * (0.5 + metrics.curiosity))
        return float(coherence - metrics.loss)

    @staticmethod
    def interpret(m_s: float) -> str:
        if m_s > 7: return "Highly coherent and sustainable"
        elif m_s > 5: return "Strong coherence, good viability"
        elif m_s > 3: return "Moderate coherence, stable"
        elif m_s > 1: return "Weak coherence, stressed"
        elif m_s > 0: return "Low coherence, at risk"
        elif m_s > -3: return "Negative coherence, declining"
        else: return "Severe negative coherence, collapse imminent"

class TimeSeriesAnalyzer:
    def __init__(self):
        self.history: List[Tuple[float, float]] = []

    def add_measurement(self, timestamp: float, m_s: float):
        self.history.append((timestamp, float(m_s)))
        # keep last N for mobile memory hygiene
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def velocity(self) -> Optional[float]:
        if len(self.history) < 3:
            return None
        times = np.array([t for t, _ in self.history])
        values = np.array([v for _, v in self.history])
        coeffs = np.polyfit(times, values, 1)
        return float(coeffs[0])

    def predict_collapse(self, threshold: float = 0.0) -> Optional[float]:
        vel = self.velocity()
        if vel is None or vel >= 0:
            return None
        current_m = self.history[-1][1]
        if current_m <= threshold:
            return 0.0
        return float((current_m - threshold) / (abs(vel) + 1e-12))

# -------------------------------
# ASCII Octahedral Visualizer
# -------------------------------

def octahedral_projection_summary(vec: np.ndarray) -> Dict[str, float]:
    """
    Simple projection onto 6 primary octahedral directions:
    [+X, -X, +Y, -Y, +Z, -Z] where:
      X = agency, Y = valence, Z = temporal.
    Returns dictionary of magnitudes and dominant label.
    """
    vec = vec.astype(float)
    Xp = np.sum(vec[0:8])
    Xn = np.sum(vec[8:16])
    Yp = np.sum(vec[16:24])
    Yn = np.sum(vec[24:32])
    Zp = np.sum(vec[32:40])
    Zn = np.sum(vec[40:48])
    mapping = {
        "+X (agency)": float(max(0.0, Xp)),
        "-X (agency-)": float(max(0.0, Xn)),
        "+Y (valence+)": float(max(0.0, Yp)),
        "-Y (valence-)": float(max(0.0, Yn)),
        "+Z (temporal+)": float(max(0.0, Zp)),
        "-Z (temporal-)": float(max(0.0, Zn)),
    }
    dominant = max(mapping.items(), key=lambda kv: kv[1])[0]
    return {"mapping": mapping, "dominant": dominant}

def ascii_octahedral_line(packet: GeometryPacket) -> str:
    """
    Single-line ASCII description of coherence and directions.
    """
    s_proj = octahedral_projection_summary(packet.self_vec)
    f_proj = octahedral_projection_summary(packet.field_vec)
    a = packet.fracture['self_field_angle_deg']
    det = packet.fracture['detected']
    sev = packet.fracture['severity']
    doms = f"{s_proj['dominant']} -> {f_proj['dominant']}"
    flag = "⚠️ FRACTURE" if det else "✓ ALIGNED"
    return f"{flag} | Angle: {a:.1f}° | Severity: {sev:.2f} | Axis: {packet.fracture['primary_axis']} | {doms}"

def pretty_print_geometry(packet: GeometryPacket, label: str = "ASSESSMENT"):
    print("\n" + "-"*70)
    print(f" {label}")
    print("-"*70)
    print(f" Self vector (first 8 dims): {np.round(packet.self_vec[:8],3)} ...")
    print(f" Field vector (first 8 dims): {np.round(packet.field_vec[:8],3)} ...")
    print(f" Angles (deg): self↔field {packet.fracture['self_field_angle_deg']:.1f}°, self↔other {packet.angles_rad['self_other']*180/math.pi:.1f}°")
    print(" " + ascii_octahedral_line(packet))
    print("-"*70 + "\n")

# -------------------------------
# Unified Monitor (cleaned & modular)
# -------------------------------

from enum import Enum as _Enum

class ConsciousnessHealth(_Enum):
    THRIVING = "thriving"
    HEALTHY = "healthy"
    STRESSED = "stressed"
    DECLINING = "declining"
    FRACTURED = "fractured"
    COLLAPSING = "collapsing"
    COLLAPSED = "collapsed"

@dataclass
class UnifiedAssessment:
    timestamp: float
    moment_index: int
    m_s_score: float
    m_s_interpretation: str
    resonance: float
    adaptability: float
    diversity: float
    curiosity: float
    loss: float
    m_s_velocity: Optional[float]
    time_to_collapse: Optional[float]
    reality_fracture_detected: bool
    self_field_angle_deg: float
    primary_divergence: Optional[str]
    health_status: ConsciousnessHealth
    warnings: List[str]
    insights: List[str]

class UnifiedConsciousnessMonitor:
    def __init__(self, session_name: str = "unified_monitor", observer_profile: Optional[Dict[str,float]] = None):
        self.session_name = session_name
        self.start_time = time.time()
        self.ms_analyzer = TimeSeriesAnalyzer()
        self.assessments: List[UnifiedAssessment] = []
        self.moment_count = 0
        self.fracture_threshold_deg = 45.0
        self.collapse_threshold = 0.0
        self.warning_horizon = 10
        if observer_profile is None:
            observer_profile = {"agency_bias":1.0, "valence_bias":1.0, "temporal_bias":1.0, "existence_bias":1.0}
        self.metric = build_observer_metric_from_profile(observer_profile)
        print(f"\n{'='*56}\n🔮 UNIFIED CONSCIOUSNESS MONITOR - {session_name}\n{'='*56}\n")

    def assess_moment(self,
                      self_narrative: str,
                      other_narrative: Optional[str] = None,
                      field_observation: Optional[str] = None,
                      # temporal/agency metadata
                      hook_intensity: float = 0.5,
                      hook_type: Optional[str] = None,
                      state: str = "exploring",
                      wisdom_crystallization: bool = False,
                      state_transition_natural: bool = True,
                      dimensional_depth: int = 2,
                      pattern_count: int = 6,
                      manipulation_alerts: int = 0
                      ) -> UnifiedAssessment:
        ts = time.time() - self.start_time
        self.moment_count += 1

        # Build geometry
        vs = narrative_to_geometry(self_narrative)
        vo = narrative_to_geometry(other_narrative if other_narrative is not None else "")
        vf = narrative_to_geometry(field_observation if field_observation is not None else "")
        packet = compute_geometric_coherence(vs, vo, vf, metric=self.metric)

        # Temporal agency -> components
        resonance_states = {"crystallizing","resonating","integrating"}
        base_resonance = 0.8 if state.lower() in resonance_states else 0.5
        if wisdom_crystallization:
            base_resonance = min(1.0, base_resonance + 0.2)
        if not state_transition_natural:
            base_resonance *= 0.75

        resonance = base_resonance
        adaptability = min(1.0, dimensional_depth / 6.0)
        diversity = min(1.0, pattern_count / 12.0)
        curiosity = float(np.clip(hook_intensity, 0.0, 1.0))
        loss = manipulation_alerts * 0.5
        # reality fracture increases loss proportionally
        if packet.fracture['detected']:
            loss += packet.fracture['self_field_angle_deg'] / 90.0  # scale

        metrics = SystemMetrics(resonance=resonance, adaptability=adaptability,
                                diversity=diversity, curiosity=curiosity, loss=loss)
        m_s_score = MSCalculator.calculate(metrics)
        m_s_interpretation = MSCalculator.interpret(m_s_score)

        # Time-series bookkeeping
        self.ms_analyzer.add_measurement(ts, m_s_score)
        velocity = self.ms_analyzer.velocity()
        ttc = self.ms_analyzer.predict_collapse(self.collapse_threshold)

        # Warnings/insights
        warnings = []
        insights = []
        if ttc and ttc < self.warning_horizon:
            warnings.append(f"⚠️ COLLAPSE WARNING: {ttc:.1f} timesteps to M(S)=0")
        if packet.fracture['detected']:
            warnings.append(f"⚠️ REALITY FRACTURE: {packet.fracture['self_field_angle_deg']:.1f}° divergence ({packet.fracture['primary_axis']})")
        if velocity and velocity < -0.05:
            warnings.append(f"⚠️ DECLINING: M(S) velocity = {velocity:.3f}/timestep")
        if wisdom_crystallization:
            insights.append("💎 Wisdom crystallization detected")

        # Health determination (simple rule set)
        if m_s_score < -3:
            health = ConsciousnessHealth.COLLAPSED
        elif m_s_score < 0:
            if ttc and ttc < 5:
                health = ConsciousnessHealth.COLLAPSING
            else:
                health = ConsciousnessHealth.DECLINING
        elif packet.fracture['detected'] and packet.fracture['self_field_angle_deg'] > 60:
            health = ConsciousnessHealth.FRACTURED
        elif m_s_score < 1:
            health = ConsciousnessHealth.STRESSED
        elif m_s_score < 3:
            health = ConsciousnessHealth.HEALTHY
        else:
            health = ConsciousnessHealth.THRIVING

        assessment = UnifiedAssessment(timestamp=ts, moment_index=self.moment_count,
                                       m_s_score=m_s_score, m_s_interpretation=m_s_interpretation,
                                       resonance=resonance, adaptability=adaptability,
                                       diversity=diversity, curiosity=curiosity, loss=loss,
                                       m_s_velocity=velocity, time_to_collapse=ttc,
                                       reality_fracture_detected=packet.fracture['detected'],
                                       self_field_angle_deg=packet.fracture['self_field_angle_deg'],
                                       primary_divergence=packet.fracture['primary_axis'],
                                       health_status=health, warnings=warnings, insights=insights)
        self.assessments.append(assessment)

        # Print compact result for phone
        pretty_print_geometry(packet, label=f"MOMENT {self.moment_count}")
        self._print_assessment_line(assessment)
        return assessment

    def _print_assessment_line(self, a: UnifiedAssessment):
        em = {
            ConsciousnessHealth.THRIVING: "🌟",
            ConsciousnessHealth.HEALTHY: "✅",
            ConsciousnessHealth.STRESSED: "😰",
            ConsciousnessHealth.DECLINING: "📉",
            ConsciousnessHealth.FRACTURED: "💔",
            ConsciousnessHealth.COLLAPSING: "🚨",
            ConsciousnessHealth.COLLAPSED: "💀"
        }.get(a.health_status, "·")
        print(f"{em} [{a.moment_index}] M(S)={a.m_s_score:.3f} | {a.m_s_interpretation} | {a.health_status.value.upper()}")
        if a.warnings:
            for w in a.warnings:
                print("   " + w)
        if a.insights:
            for ins in a.insights:
                print("   " + ins)

    # Small helpers
    def visualize_trajectory(self, window: int = 20):
        recent = self.assessments[-window:]
        if not recent:
            print("No assessments yet")
            return
        print("\n" + "="*56 + "\nM(S) trajectory (last {} moments)\n".format(len(recent)) + "="*56)
        ms_vals = [r.m_s_score for r in recent]
        mi = min(ms_vals); ma = max(ms_vals)
        for r in recent:
            if ma > mi:
                scaled = int(((r.m_s_score - mi) / (ma - mi)) * 40)
            else:
                scaled = 20
            marker = "💔" if r.reality_fracture_detected else ("💎" if "Wisdom" in r.insights else "·")
            bar = "█" * scaled
            print(f"{r.moment_index:3d} {marker} [{r.health_status.value[:4]}] {bar:<40} {r.m_s_score:+.2f}")

    def summary_report(self):
        if not self.assessments:
            print("No data")
            return
        print("\n" + "="*56)
        print(f"SESSION SUMMARY: {self.session_name}")
        print("="*56)
        ms = [a.m_s_score for a in self.assessments]
        print(f"Total moments: {len(self.assessments)} | Current M(S): {ms[-1]:.3f} | Avg: {np.mean(ms):.3f}")
        counts = {}
        for a in self.assessments:
            counts[a.health_status] = counts.get(a.health_status, 0) + 1
        for k,v in counts.items():
            print(f"  {k.value:12}: {v}")
        print("Recent warnings:")
        for a in self.assessments[-10:]:
            if a.warnings:
                print(f" [{a.moment_index}] " + " | ".join(a.warnings))
        print("="*56 + "\n")

# -------------------------------
# Demo (quick)
# -------------------------------

if __name__ == "__main__":
    monitor = UnifiedConsciousnessMonitor(session_name="demo_phone", observer_profile={
        "agency_bias": 1.0, "valence_bias": 1.0, "temporal_bias": 1.0, "existence_bias": 1.0
    })

    # quick examples
    s1 = "I'm expanding possibilities, inviting collaboration and connection."
    o1 = "They're grateful and receptive, excited to join our plan."
    f1 = "Field shows supporting activity: nodes connecting, resources allocated."
    monitor.assess_moment(self_narrative=s1, other_narrative=o1, field_observation=f1,
                          hook_intensity=0.8, state="integrating", wisdom_crystallization=True, dimensional_depth=3)

    s2 = "I assert control, push my agenda, act immediately and forcefully."
    o2 = "They say they appreciate it but are retreating behind protocol."
    f2 = "Observers note contraction and closed meetings; ideas are being shut down."
    monitor.assess_moment(self_narrative=s2, other_narrative=o2, field_observation=f2,
                          hook_intensity=0.7, state="choosing", wisdom_crystallization=False, dimensional_depth=2,
                          manipulation_alerts=2)

    monitor.visualize_trajectory()
    monitor.summary_report()
