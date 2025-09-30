"""
DOFUS Knowledge Base - Systeme de Connaissance Integre
Module d'integration pour toutes les bases de donnees DOFUS Unity
Approche 100% vision - Reconnaissance automatique + apprentissage adaptatif
"""

from .spells_database import (
    DofusSpellsDatabase, DofusSpell, SpellCost, SpellRange, SpellEffect,
    DofusClass, SpellType, TargetType, get_spells_database
)

from .monsters_database import (
    DofusMonstersDatabase, DofusMonster, MonsterStats, MonsterResistances,
    MonsterSpell, MonsterBehavior, MonsterDrops, MonsterLocation,
    MonsterRank, MonsterElement, AIPattern, get_monsters_database
)

from .maps_database import (
    DofusMapsDatabase, DofusMap, ResourceSpawn, MonsterSpawn, MapTransition,
    MapType, ResourceType, CellType, get_maps_database
)

from .economy_tracker import (
    DofusEconomyTracker, PriceEntry, PriceTrend, MarketOpportunityAlert,
    CraftProfitAnalysis, PriceCategory, TrendDirection, MarketOpportunity,
    get_economy_tracker
)

from .dofus_data_extractor import (
    DofusDataExtractor, BundleInfo, ExtractedData, get_dofus_extractor
)

__version__ = "1.0.0"
__author__ = "Claude Code AI Assistant"

# Exports principaux
__all__ = [
    # Spells
    "DofusSpellsDatabase", "DofusSpell", "DofusClass", "get_spells_database",

    # Monsters
    "DofusMonstersDatabase", "DofusMonster", "MonsterRank", "get_monsters_database",

    # Maps
    "DofusMapsDatabase", "DofusMap", "MapType", "get_maps_database",

    # Economy
    "DofusEconomyTracker", "PriceEntry", "MarketOpportunity", "get_economy_tracker",

    # Data Extraction
    "DofusDataExtractor", "get_dofus_extractor",

    # Unified Access
    "DofusKnowledgeBase"
]