"""
Safety Module - Sécurité et modes de test
ObservationMode, DryRun, Emergency Stop, Anti-detection
"""

from .observation_mode import (
    ObservationMode,
    ObservationLog,
    create_observation_mode
)

# Placeholder for future safety modules
def create_safety_manager(*args, **kwargs):
    """Placeholder for SafetyManager - to be implemented"""
    return None

__all__ = [
    "ObservationMode",
    "ObservationLog",
    "create_observation_mode",
    "create_safety_manager"
]