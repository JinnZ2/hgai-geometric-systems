"""
Core M(S) calculation engine
"""

import numpy as np
from typing import Dict, Union, List, Optional
from dataclasses import dataclass

@dataclass
class SystemMetrics:
    """Container for system measurements"""
    resonance: float  # R_e ∈ [0, 1]
    adaptability: float  # A ∈ [0, 1]
    diversity: float  # D ∈ [0, 1]
    curiosity: float  # C ∈ [0, 1]
    loss: float  # L ∈ [0, ∞)
    
    def __post_init__(self):
        """Validate metrics"""
        if not (0 <= self.resonance <= 1):
            raise ValueError("Resonance must be in [0, 1]")
        if not (0 <= self.adaptability <= 1):
            raise ValueError("Adaptability must be in [0, 1]")
        if not (0 <= self.diversity <= 1):
            raise ValueError("Diversity must be in [0, 1]")
        if not (0 <= self.curiosity <= 1):
            raise ValueError("Curiosity must be in [0, 1]")
        if self.loss < 0:
            raise ValueError("Loss must be non-negative")

class MSCalculator:
    """Calculate system morality M(S)"""
    
    @staticmethod
    def calculate(metrics: SystemMetrics) -> float:
        """
        Calculate M(S) = (R_e × A × D × C) - L
        
        Args:
            metrics: System measurements
            
        Returns:
            M(S) score (typically in range [-10, +10])
        """
        coherence_product = (
            metrics.resonance * 
            metrics.adaptability * 
            metrics.diversity * 
            metrics.curiosity
        )
        
        m_s = coherence_product - metrics.loss
        
        return m_s
    
    @staticmethod
    def interpret(m_s: float) -> str:
        """
        Interpret M(S) value
        
        Args:
            m_s: Calculated M(S) score
            
        Returns:
            Human-readable interpretation
        """
        if m_s > 7:
            return "Highly coherent and sustainable"
        elif m_s > 5:
            return "Strong coherence, good viability"
        elif m_s > 3:
            return "Moderate coherence, stable"
        elif m_s > 1:
            return "Weak coherence, stressed"
        elif m_s > 0:
            return "Low coherence, at risk"
        elif m_s > -3:
            return "Negative coherence, declining"
        else:
            return "Severe negative coherence, collapse imminent"
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> float:
        """
        Calculate M(S) from dictionary
        
        Args:
            data: Dict with keys: resonance, adaptability, diversity, 
                  curiosity, loss
                  
        Returns:
            M(S) score
        """
        metrics = SystemMetrics(
            resonance=data['resonance'],
            adaptability=data['adaptability'],
            diversity=data['diversity'],
            curiosity=data['curiosity'],
            loss=data['loss']
        )
        return cls.calculate(metrics)

class TimeSeriesAnalyzer:
    """Analyze M(S) evolution over time"""
    
    def __init__(self):
        self.history: List[tuple[float, float]] = []  # [(timestamp, m_s)]
    
    def add_measurement(self, timestamp: float, m_s: float):
        """Add timestamped M(S) measurement"""
        self.history.append((timestamp, m_s))
    
    def trajectory(self) -> np.ndarray:
        """Get M(S) trajectory"""
        return np.array([m_s for _, m_s in self.history])
    
    def velocity(self) -> Optional[float]:
        """Calculate rate of M(S) change"""
        if len(self.history) < 2:
            return None
        
        times = np.array([t for t, _ in self.history])
        values = np.array([m_s for _, m_s in self.history])
        
        # Linear regression for velocity
        coeffs = np.polyfit(times, values, 1)
        return coeffs[0]  # Slope = velocity
    
    def predict_collapse(self, threshold: float = 0.0) -> Optional[float]:
        """
        Predict when M(S) will cross threshold
        
        Args:
            threshold: M(S) value defining collapse (default: 0)
            
        Returns:
            Estimated time to threshold, or None if not declining
        """
        velocity = self.velocity()
        if velocity is None or velocity >= 0:
            return None  # Not declining
        
        current_m_s = self.history[-1][1]
        if current_m_s <= threshold:
            return 0.0  # Already below threshold
        
        # Linear extrapolation
        time_to_threshold = (current_m_s - threshold) / abs(velocity)
        return time_to_threshold
