"""
Module Professions - Système complet de gestion des métiers pour le botting.

Ce module fournit:
- Classe de base abstraite pour tous les métiers
- Implémentations complètes pour Fermier, Bûcheron, Mineur et Alchimiste
- Gestionnaire global pour coordination multi-métiers
- Optimisation de routes et calculs de rentabilité
- Système de synergies entre métiers
- Patterns de reconnaissance et farming automatisé

Usage:
    from modules.professions import ProfessionManager, Farmer, Lumberjack, Miner, Alchemist
    
    # Utilisation complète
    manager = ProfessionManager()
    session = manager.optimize_global_session(4.0, OptimizationStrategy.PROFIT_FOCUSED)
    results = manager.execute_session(session)
    
    # Utilisation métier individuel
    farmer = Farmer()
    route = farmer.get_optimal_route((1, 30))  # Ressources niveau 1-30
    profitability = farmer.calculate_profitability('ble', 3600)
"""

from .base import BaseProfession, ResourceData, ResourceType, QualityLevel, ProfessionStats
from .farmer import Farmer
from .lumberjack import Lumberjack  
from .miner import Miner
from .alchemist import Alchemist
from .profession_manager import ProfessionManager, OptimizationStrategy, GlobalSession

__version__ = "1.0.0"
__author__ = "Claude Code"

__all__ = [
    # Classes de base
    'BaseProfession',
    'ResourceData', 
    'ResourceType',
    'QualityLevel',
    'ProfessionStats',
    
    # Métiers spécialisés
    'Farmer',
    'Lumberjack',
    'Miner', 
    'Alchemist',
    
    # Gestionnaire global
    'ProfessionManager',
    'OptimizationStrategy',
    'GlobalSession'
]