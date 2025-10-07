"""
Action Humanizer - Ajoute des variations humaines aux actions
"""

import random
import time
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ActionHumanizer:
    """Humanise les actions pour éviter la détection"""
    
    def __init__(self):
        self.action_count = 0
        self.last_action_time = time.time()
    
    def humanize_position(self, x: int, y: int, variance: int = 3) -> Tuple[int, int]:
        """Ajoute variation à une position"""
        return (
            x + random.randint(-variance, variance),
            y + random.randint(-variance, variance)
        )
    
    def humanize_duration(self, base_duration: float, variance_pct: float = 0.2) -> float:
        """Ajoute variation à une durée"""
        variance = base_duration * variance_pct
        return base_duration + random.uniform(-variance, variance)
    
    def humanize_pause(self, base_pause: float = 0.5) -> float:
        """Pause aléatoire entre actions"""
        return random.uniform(base_pause * 0.5, base_pause * 1.5)
    
    def should_rest(self, probability: float = 0.05) -> bool:
        """Devrait faire une pause?"""
        self.action_count += 1
        
        # Plus d'actions = plus de fatigue
        fatigue_factor = min(1.0, self.action_count / 1000)
        adjusted_prob = probability * (1 + fatigue_factor)
        
        return random.random() < adjusted_prob
    
    def get_rest_duration(self) -> float:
        """Durée de repos"""
        return random.uniform(2.0, 5.0)
    
    def reset_action_count(self):
        """Reset compteur actions"""
        self.action_count = 0
