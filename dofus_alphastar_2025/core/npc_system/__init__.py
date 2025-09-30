"""
NPC System - Système de reconnaissance et interaction PNJ DOFUS
Intelligence contextuelle pour interactions naturelles
"""

from .npc_recognition import (
    NPCRecognition,
    NPCData,
    NPCInteractionContext,
    create_npc_recognition
)

from .contextual_intelligence import (
    ContextualIntelligence,
    GameContext,
    ContextualDecision,
    create_contextual_intelligence
)

from .monster_database import (
    MonsterDatabase,
    MonsterData,
    CombatStrategy,
    create_monster_database
)

from .group_coordination import (
    GroupCoordination,
    GroupMember,
    GroupAction,
    create_group_coordination
)

__all__ = [
    "NPCRecognition",
    "NPCData",
    "NPCInteractionContext",
    "create_npc_recognition",
    "ContextualIntelligence",
    "GameContext",
    "ContextualDecision",
    "create_contextual_intelligence",
    "MonsterDatabase",
    "MonsterData",
    "CombatStrategy",
    "create_monster_database",
    "GroupCoordination",
    "GroupMember",
    "GroupAction",
    "create_group_coordination"
]