"""
Quest System - Système de gestion de quêtes intelligent pour DOFUS
Intégration avec HRM pour prise de décision contextuelle
"""

from .quest_manager import (
    QuestManager,
    Quest,
    QuestStep,
    QuestStatus,
    QuestType,
    create_quest_manager
)

from .quest_tracker import (
    QuestTracker,
    QuestProgress,
    QuestObjective,
    create_quest_tracker
)

from .dialogue_system import (
    DialogueSystem,
    DialogueChoice,
    NPCInteraction,
    create_dialogue_system
)

from .inventory_manager import (
    InventoryManager,
    InventoryItem,
    InventoryAction,
    create_inventory_manager
)

__all__ = [
    "QuestManager",
    "Quest",
    "QuestStep",
    "QuestStatus",
    "QuestType",
    "create_quest_manager",
    "QuestTracker",
    "QuestProgress",
    "QuestObjective",
    "create_quest_tracker",
    "DialogueSystem",
    "DialogueChoice",
    "NPCInteraction",
    "create_dialogue_system",
    "InventoryManager",
    "InventoryItem",
    "InventoryAction",
    "create_inventory_manager"
]