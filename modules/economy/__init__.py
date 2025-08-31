"""
Module économique pour Dofus Bot
Système d'analyse de marché, gestion d'inventaire et optimisation de craft
"""

from .market_analyzer import (
    MarketAnalyzer,
    MarketItem,
    ArbitrageOpportunity,
    PricePrediction,
    MLPricePredictor,
    ArbitrageDetector
)

from .inventory_manager import (
    InventoryManager,
    Item,
    ItemType,
    StorageLocation,
    ValueCalculator,
    InventoryOptimizer
)

from .crafting_optimizer import (
    CraftingOptimizer,
    Recipe,
    CraftJob,
    CraftAnalysis,
    OptimizationGoal,
    CraftType,
    ProfitabilityCalculator,
    CraftQueue,
    ResourceManager
)

__version__ = "1.0.0"
__author__ = "Claude Code Assistant"
__description__ = "Système économique intelligent pour bot Dofus"

# Configuration par défaut
DEFAULT_CONFIG = {
    "market_analyzer": {
        "scan_interval": 300,  # 5 minutes
        "min_profit_margin": 0.1,
        "min_roi": 0.05
    },
    "inventory_manager": {
        "auto_optimization": True,
        "optimization_interval": 300,
        "value_weights": {
            'market_value': 0.4,
            'craft_value': 0.3,
            'utility_score': 0.2,
            'rarity_bonus': 0.1
        }
    },
    "crafting_optimizer": {
        "default_goal": "balanced",
        "auto_crafting": False,
        "resource_buffer": 0.1  # 10% de buffer pour les ressources
    }
}

class EconomySystem:
    """Système économique principal intégrant tous les modules"""
    
    def __init__(self, config: dict = None):
        """Initialise le système économique"""
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Initialisation des modules
        self.market_analyzer = MarketAnalyzer()
        self.inventory_manager = InventoryManager()
        self.crafting_optimizer = CraftingOptimizer()
        
        # État du système
        self.running = False
    
    def start(self):
        """Démarre tous les systèmes économiques"""
        if self.running:
            return
        
        print("Démarrage du système économique...")
        
        # Démarrage des modules
        if self.config["market_analyzer"]["scan_interval"] > 0:
            self.market_analyzer.start_continuous_scan()
            print("✓ Analyseur de marché démarré")
        
        if self.config["inventory_manager"]["auto_optimization"]:
            self.inventory_manager.start_auto_optimization()
            print("✓ Optimiseur d'inventaire démarré")
        
        if self.config["crafting_optimizer"]["auto_crafting"]:
            self.crafting_optimizer.start_auto_crafting()
            print("✓ Optimiseur de craft démarré")
        
        self.running = True
        print("Système économique opérationnel!")
    
    def stop(self):
        """Arrête tous les systèmes économiques"""
        if not self.running:
            return
        
        print("Arrêt du système économique...")
        
        self.market_analyzer.stop_continuous_scan()
        self.inventory_manager.stop_auto_optimization()
        self.crafting_optimizer.stop_auto_crafting()
        
        self.running = False
        print("Système économique arrêté.")
    
    def get_system_status(self) -> dict:
        """Retourne l'état global du système"""
        return {
            "running": self.running,
            "market_analyzer": {
                "active": self.market_analyzer.running,
                "scan_interval": self.config["market_analyzer"]["scan_interval"]
            },
            "inventory_manager": {
                "auto_optimization": self.inventory_manager.auto_optimization,
                "total_items": sum(len(items) for items in self.inventory_manager.inventory.values())
            },
            "crafting_optimizer": {
                "auto_crafting": self.crafting_optimizer._running,
                "queued_jobs": len([j for j in self.crafting_optimizer.craft_queue.jobs if j.status == "queued"])
            }
        }

# Fonctions utilitaires
def create_economy_system(config_path: str = None) -> EconomySystem:
    """Crée et configure le système économique"""
    config = None
    
    if config_path:
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {e}")
    
    return EconomySystem(config)

# Export des principales classes et fonctions
__all__ = [
    # Classes principales
    'EconomySystem',
    'MarketAnalyzer',
    'InventoryManager', 
    'CraftingOptimizer',
    
    # Classes de données
    'MarketItem',
    'Item',
    'Recipe',
    'CraftJob',
    
    # Enums
    'ItemType',
    'StorageLocation',
    'OptimizationGoal',
    'CraftType',
    
    # Classes d'analyse
    'ArbitrageOpportunity',
    'PricePrediction',
    'CraftAnalysis',
    
    # Classes utilitaires
    'MLPricePredictor',
    'ArbitrageDetector',
    'ValueCalculator',
    'InventoryOptimizer',
    'ProfitabilityCalculator',
    'CraftQueue',
    'ResourceManager',
    
    # Fonctions utilitaires
    'create_economy_system',
    'DEFAULT_CONFIG'
]