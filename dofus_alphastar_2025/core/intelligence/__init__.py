"""
Module Intelligence - Phase 1-2 Projet Augmenta
Intelligence Passive, Opportunites, Fatigue
"""

from .opportunity_manager import (
    OpportunityManager,
    Opportunity,
    OpportunityTracker
)

from .passive_intelligence import (
    PassiveIntelligence,
    OpportunityDetection
)

from .fatigue_simulation import (
    FatigueSimulator,
    FatigueState
)

__all__ = [
    'PassiveIntelligence',
    'OpportunityDetection',
    'OpportunityManager',
    'Opportunity',
    'OpportunityTracker',
    'FatigueSimulator',
    'FatigueState'
]