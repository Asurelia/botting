"""
AlphaStar Engine - Agent principal avec système de league
"""

from .alphastar_agent import (
    AlphaStarAgent,
    DofusAlphaStarAgent,
    create_alphastar_agent
)

from .league_system import (
    LeagueManager,
    AgentPool,
    MatchmakingSystem,
    create_league_system
)

from .training_orchestrator import (
    TrainingOrchestrator,
    SupervisedLearningPhase,
    ReinforcementLearningPhase,
    LeagueTrainingPhase
)

__all__ = [
    "AlphaStarAgent",
    "DofusAlphaStarAgent",
    "create_alphastar_agent",
    "LeagueManager",
    "AgentPool",
    "MatchmakingSystem",
    "create_league_system",
    "TrainingOrchestrator",
    "SupervisedLearningPhase",
    "ReinforcementLearningPhase",
    "LeagueTrainingPhase"
]