"""
Event bus stub
"""

from enum import Enum


class EventType(Enum):
    """Types d'événements"""
    COMBAT_START = "combat_start"
    COMBAT_END = "combat_end"
    LEVEL_UP = "level_up"
    DEATH = "death"
    LOOT_DETECTED = "loot_detected"
    MOB_DETECTED = "mob_detected"
    NPC_DETECTED = "npc_detected"


class EventPriority(Enum):
    """Priorités d'événements"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
