"""
External Data Module - Intégrations avec APIs externes
DofusDB, Ganymède, et autres sources de données
"""

from .dofusdb_client import (
    DofusDBClient,
    ItemData,
    SpellData,
    MonsterData,
    create_dofusdb_client
)

__all__ = [
    "DofusDBClient",
    "ItemData",
    "SpellData",
    "MonsterData",
    "create_dofusdb_client"
]