"""
Module d'automatisation avancée des métiers pour DOFUS.
Système complet avec IA, ML et optimisation multi-personnages.

Ce module fournit:
- Farming intelligent multi-zones avec prédiction respawn
- Craft automatique avec calcul rentabilité temps réel  
- Optimisation XP/Kamas selon marché actuel
- Prédiction respawn ressources avec ML
- Planification sessions métiers optimales

Fonctionnalités avancées:
- Prédiction ML des temps de respawn
- Routes dynamiques selon affluence zones
- Calcul rentabilité temps réel via scan HDV
- Changement automatique métier selon objectifs
- Gestion stocks inter-personnages
- Craft en masse avec optimisation matériaux
- Détection concurrence sur zones
- Mode furtif zones bondées
- Alertes ressources rares
- Export statistiques détaillées

Intégration complète avec le système existant.
"""

from .advanced_farming import (
    AdvancedFarmer,
    FarmingStrategy,
    ZoneStatus,
    ZoneData,
    HarvestSession
)

from .craft_automation import (
    CraftAutomation,
    CraftPriority,
    CraftComplexity,
    RecipeData,
    CraftSession,
    MarketData
)

from .profession_optimizer import (
    ProfessionOptimizer,
    OptimizationGoal,
    TimeHorizon,
    OptimizationConstraint,
    ActivityRecommendation,
    OptimizationResult,
    MarketOpportunity
)

from .resource_predictor import (
    ResourcePredictor,
    PredictionModel,
    PredictionConfidence,
    ResourceObservation,
    PredictionResult,
    ResourcePattern
)

from .profession_scheduler import (
    ProfessionScheduler,
    SchedulePriority,
    SessionType,
    CharacterStatus,
    Character,
    ScheduledTask,
    Schedule,
    SchedulingConstraint
)

# Version du module
__version__ = "1.0.0"

# Auteur et métadonnées
__author__ = "Claude Code & Anthropic"
__description__ = "Système d'automatisation avancée des métiers DOFUS avec IA et ML"
__status__ = "Stable"

# Classes principales exportées
__all__ = [
    # Advanced Farming
    "AdvancedFarmer",
    "FarmingStrategy", 
    "ZoneStatus",
    "ZoneData",
    "HarvestSession",
    
    # Craft Automation
    "CraftAutomation",
    "CraftPriority",
    "CraftComplexity", 
    "RecipeData",
    "CraftSession",
    "MarketData",
    
    # Profession Optimizer
    "ProfessionOptimizer",
    "OptimizationGoal",
    "TimeHorizon",
    "OptimizationConstraint",
    "ActivityRecommendation",
    "OptimizationResult",
    "MarketOpportunity",
    
    # Resource Predictor
    "ResourcePredictor",
    "PredictionModel",
    "PredictionConfidence",
    "ResourceObservation", 
    "PredictionResult",
    "ResourcePattern",
    
    # Profession Scheduler
    "ProfessionScheduler",
    "SchedulePriority",
    "SessionType",
    "CharacterStatus",
    "Character",
    "ScheduledTask",
    "Schedule",
    "SchedulingConstraint"
]


def get_module_info():
    """Retourne les informations du module"""
    return {
        "name": "professions_advanced",
        "version": __version__,
        "author": __author__, 
        "description": __description__,
        "status": __status__,
        "components": [
            {
                "name": "AdvancedFarmer",
                "description": "Farming intelligent multi-zones avec prédiction respawn et stratégies adaptatives",
                "features": [
                    "Prédiction ML des respawns",
                    "Routes dynamiques optimisées", 
                    "Détection et évitement de concurrence",
                    "Modes stealth et compétitif",
                    "Analyse performance temps réel"
                ]
            },
            {
                "name": "CraftAutomation", 
                "description": "Craft automatique avec analyse de marché et optimisation rentabilité",
                "features": [
                    "Calcul rentabilité temps réel",
                    "Intégration données HDV",
                    "Optimisation ordre de craft",
                    "Gestion intelligente stocks",
                    "Rapports de performance détaillés"
                ]
            },
            {
                "name": "ProfessionOptimizer",
                "description": "Optimiseur multi-objectifs XP/Kamas avec contraintes",
                "features": [
                    "Optimisation multi-objectifs",
                    "Analyse opportunités marché",
                    "Gestion contraintes complexes", 
                    "Recommandations adaptatives",
                    "Stratégies alternatives"
                ]
            },
            {
                "name": "ResourcePredictor",
                "description": "Prédicteur de respawn avec Machine Learning",
                "features": [
                    "Modèles ML avancés (RF, GB, NN)",
                    "Apprentissage adaptatif",
                    "Détection patterns temporels",
                    "Prédictions avec intervalles confiance",
                    "Validation et auto-amélioration"
                ]
            },
            {
                "name": "ProfessionScheduler",
                "description": "Planificateur optimal multi-personnages avec synergies",
                "features": [
                    "Planification multi-personnages",
                    "Détection synergies inter-métiers",
                    "Résolution conflits automatique",
                    "Optimisation temporelle globale",
                    "Gestion contraintes complexes"
                ]
            }
        ]
    }


def create_full_automation_system(base_farmer=None, market_analyzer=None, 
                                profession_manager=None, pathfinder=None,
                                detector=None, anti_detection=None):
    """
    Crée un système d'automatisation complet intégré.
    
    Args:
        base_farmer: Instance Farmer de base
        market_analyzer: Analyseur de marché
        profession_manager: Gestionnaire de métiers
        pathfinder: Système de navigation
        detector: Détecteur de ressources
        anti_detection: Système anti-détection
        
    Returns:
        Dict contenant tous les composants du système
    """
    
    # Création des composants principaux
    components = {}
    
    # Resource Predictor (indépendant)
    components['predictor'] = ResourcePredictor(PredictionModel.ENSEMBLE)
    
    # Profession Optimizer (nécessite market_analyzer)
    if profession_manager and market_analyzer:
        components['optimizer'] = ProfessionOptimizer(profession_manager, market_analyzer)
    
    # Advanced Farmer (nécessite plusieurs dépendances)
    if base_farmer and pathfinder and detector and anti_detection:
        components['farmer'] = AdvancedFarmer(base_farmer, pathfinder, detector, anti_detection)
    
    # Craft Automation (nécessite market_analyzer)
    if market_analyzer:
        # Utilise base_farmer comme profession si disponible, sinon None
        components['crafter'] = CraftAutomation(base_farmer, market_analyzer)
    
    # Profession Scheduler (système central)
    if 'optimizer' in components and 'predictor' in components:
        components['scheduler'] = ProfessionScheduler(
            components['optimizer'],
            components['predictor'],
            components.get('farmer'),
            components.get('crafter')
        )
    
    print(f"🏗️ Système d'automatisation créé avec {len(components)} composants")
    
    return components


# Fonctions utilitaires pour l'intégration
def validate_system_dependencies():
    """Valide que toutes les dépendances sont disponibles"""
    dependencies = {
        'numpy': True,
        'pandas': True, 
        'scikit-learn': False,  # Optionnel
        'asyncio': True
    }
    
    try:
        import numpy
        import pandas
        import asyncio
        dependencies['numpy'] = True
        dependencies['pandas'] = True
        dependencies['asyncio'] = True
    except ImportError as e:
        print(f"⚠️ Dépendance manquante: {e}")
        return False
    
    try:
        import sklearn
        dependencies['scikit-learn'] = True
        print("✅ scikit-learn disponible - Modèles ML activés")
    except ImportError:
        print("⚠️ scikit-learn non disponible - Utilisation modèles simplifiés")
    
    return True


# Initialisation du module
if __name__ != "__main__":
    print(f"📚 Module professions_advanced v{__version__} chargé")
    print("   Fonctionnalités disponibles:")
    print("   - Farming intelligent multi-zones")
    print("   - Craft automatique avec rentabilité")
    print("   - Optimisation XP/Kamas temps réel")
    print("   - Prédiction ML respawn ressources")
    print("   - Planification sessions optimales")
    print("   - Gestion multi-personnages")
    
    # Validation des dépendances
    if validate_system_dependencies():
        print("✅ Toutes les dépendances sont satisfaites")
    else:
        print("⚠️ Certaines dépendances manquent - Fonctionnalités limitées")
else:
    # Mode démonstration si exécuté directement
    import asyncio
    
    async def demo_complete_system():
        """Démonstration du système complet"""
        print("🚀 Démonstration système d'automatisation avancée DOFUS")
        
        # Créer système basique sans dépendances
        system = create_full_automation_system()
        
        if system:
            print(f"✅ Système créé avec {len(system)} composants")
            
            # Test rapide de chaque composant
            for name, component in system.items():
                print(f"   - {name}: {type(component).__name__} ✅")
        
        print("\n📖 Pour utilisation complète, intégrer avec:")
        print("   from modules.professions_advanced import *")
        print("   system = create_full_automation_system(...)")
    
    asyncio.run(demo_complete_system())