"""
Autonomous Learning Module - Apprentissage Autonome Incarné
Bot qui apprend et évolue comme un humain dans son environnement

Composants principaux:
- SelfAwarenessEngine: Conscience de soi
- ContinuousLearningEngine: Apprentissage continu
- AutobiographicalMemory: Mémoire de vie
- EmergentDecisionSystem: Décisions émergentes autonomes
"""

from .self_awareness import (
    SelfAwarenessEngine,
    create_self_awareness_engine,
    EmotionalState,
    PhysicalNeed,
    CognitiveNeed,
    SocialNeed,
    SelfState,
    WorldPerception
)

from .continuous_learning import (
    ContinuousLearningEngine,
    create_continuous_learning_engine,
    ExperienceType,
    LearningMode,
    Experience,
    LearningGoal
)

from .autobiographical_memory import (
    AutobiographicalMemory,
    create_autobiographical_memory,
    MemoryCategory,
    MemoryImportance,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory
)

from .emergent_decision_system import (
    EmergentDecisionSystem,
    create_emergent_decision_system,
    DecisionOrigin,
    Decision
)


__all__ = [
    # Self-Awareness
    "SelfAwarenessEngine",
    "create_self_awareness_engine",
    "EmotionalState",
    "PhysicalNeed",
    "CognitiveNeed",
    "SocialNeed",
    "SelfState",
    "WorldPerception",

    # Continuous Learning
    "ContinuousLearningEngine",
    "create_continuous_learning_engine",
    "ExperienceType",
    "LearningMode",
    "Experience",
    "LearningGoal",

    # Autobiographical Memory
    "AutobiographicalMemory",
    "create_autobiographical_memory",
    "MemoryCategory",
    "MemoryImportance",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",

    # Emergent Decision System
    "EmergentDecisionSystem",
    "create_emergent_decision_system",
    "DecisionOrigin",
    "Decision",
]
