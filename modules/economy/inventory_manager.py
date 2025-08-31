"""
Gestionnaire d'inventaire intelligent pour Dofus
Optimisation de stockage, gestion automatique des ressources, calculs de valeur
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
from collections import defaultdict, deque
import threading
import time

class ItemType(Enum):
    """Types d'items"""
    RESOURCE = "resource"
    EQUIPMENT = "equipment"
    CONSUMABLE = "consumable"
    QUEST = "quest"
    COSMETIC = "cosmetic"
    CURRENCY = "currency"

class StorageLocation(Enum):
    """Emplacements de stockage"""
    INVENTORY = "inventory"
    BANK = "bank"
    GUILD_BANK = "guild_bank"
    HOUSE = "house"
    SHOP = "shop"

@dataclass
class Item:
    """Représente un item dans l'inventaire"""
    item_id: int
    name: str
    quantity: int
    location: StorageLocation
    slot_position: Optional[int] = None
    item_type: ItemType = ItemType.RESOURCE
    level: int = 0
    weight: float = 1.0
    stack_size: int = 100
    market_value: float = 0.0
    craft_value: float = 0.0
    utility_score: float = 0.0
    last_used: Optional[datetime] = None
    acquisition_date: datetime = field(default_factory=datetime.now)
    is_locked: bool = False
    tags: Set[str] = field(default_factory=set)

@dataclass
class StorageSpace:
    """Représente un espace de stockage"""
    location: StorageLocation
    total_slots: int
    used_slots: int
    max_weight: float
    current_weight: float
    access_cost: float = 0.0  # Coût d'accès (ex: téléportation)
    access_time: float = 0.0  # Temps d'accès en secondes
    
    @property
    def free_slots(self) -> int:
        return self.total_slots - self.used_slots
    
    @property
    def weight_utilization(self) -> float:
        if self.max_weight == 0:
            return 0.0
        return self.current_weight / self.max_weight
    
    @property
    def slot_utilization(self) -> float:
        return self.used_slots / self.total_slots

@dataclass
class OptimizationRule:
    """Règle d'optimisation pour l'inventaire"""
    name: str
    condition: str  # Expression évaluable
    action: str     # Action à effectuer
    priority: int   # Priorité (plus élevé = plus prioritaire)
    enabled: bool = True

class InventoryDatabase:
    """Base de données pour l'historique d'inventaire"""
    
    def __init__(self, db_path: str = "inventory_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialise les tables de la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Historique des items
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS item_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                location TEXT NOT NULL,
                action TEXT NOT NULL,  -- added, removed, moved, used
                timestamp DATETIME NOT NULL,
                market_value REAL,
                character_name TEXT
            )
        ''')
        
        # Statistiques d'utilisation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_stats (
                item_id INTEGER PRIMARY KEY,
                total_acquired INTEGER DEFAULT 0,
                total_used INTEGER DEFAULT 0,
                total_sold INTEGER DEFAULT 0,
                last_used_date DATETIME,
                avg_hold_time REAL,  -- Temps moyen de détention
                usage_frequency REAL  -- Fréquence d'utilisation
            )
        ''')
        
        # Optimisations effectuées
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                optimization_type TEXT NOT NULL,
                items_moved INTEGER DEFAULT 0,
                space_saved INTEGER DEFAULT 0,
                value_optimized REAL DEFAULT 0,
                timestamp DATETIME NOT NULL,
                efficiency_gain REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_item_action(self, item: Item, action: str, character_name: str = ""):
        """Enregistre une action sur un item"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO item_history 
            (item_id, name, quantity, location, action, timestamp, market_value, character_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.item_id, item.name, item.quantity, item.location.value,
            action, datetime.now(), item.market_value, character_name
        ))
        
        conn.commit()
        conn.close()
    
    def update_usage_stats(self, item_id: int, action: str):
        """Met à jour les statistiques d'utilisation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if action == "acquired":
            cursor.execute('''
                INSERT OR IGNORE INTO usage_stats (item_id) VALUES (?)
            ''', (item_id,))
            
            cursor.execute('''
                UPDATE usage_stats SET total_acquired = total_acquired + 1
                WHERE item_id = ?
            ''', (item_id,))
        
        elif action == "used":
            cursor.execute('''
                UPDATE usage_stats 
                SET total_used = total_used + 1, last_used_date = ?
                WHERE item_id = ?
            ''', (datetime.now(), item_id))
        
        conn.commit()
        conn.close()

class ValueCalculator:
    """Calculateur de valeur d'items"""
    
    def __init__(self):
        self.market_prices = {}
        self.craft_costs = {}
        self.utility_weights = {
            'market_value': 0.4,
            'craft_value': 0.3,
            'utility_score': 0.2,
            'rarity_bonus': 0.1
        }
    
    def update_market_price(self, item_id: int, price: float):
        """Met à jour le prix de marché d'un item"""
        self.market_prices[item_id] = price
    
    def calculate_item_value(self, item: Item) -> float:
        """Calcule la valeur totale d'un item"""
        market_component = item.market_value * self.utility_weights['market_value']
        craft_component = item.craft_value * self.utility_weights['craft_value']
        utility_component = item.utility_score * self.utility_weights['utility_score']
        
        # Bonus de rareté basé sur le niveau
        rarity_bonus = (item.level / 200) * 100 * self.utility_weights['rarity_bonus']
        
        total_value = market_component + craft_component + utility_component + rarity_bonus
        return total_value * item.quantity
    
    def calculate_storage_efficiency(self, item: Item) -> float:
        """Calcule l'efficacité de stockage (valeur/poids)"""
        if item.weight == 0:
            return float('inf')
        
        total_value = self.calculate_item_value(item)
        return total_value / (item.weight * item.quantity)
    
    def calculate_turnover_rate(self, item: Item, usage_stats: Dict) -> float:
        """Calcule le taux de rotation d'un item"""
        stats = usage_stats.get(item.item_id, {})
        total_used = stats.get('total_used', 0)
        total_acquired = stats.get('total_acquired', 1)
        
        if total_acquired == 0:
            return 0.0
        
        return total_used / total_acquired

class InventoryOptimizer:
    """Optimiseur d'inventaire avec algorithmes avancés"""
    
    def __init__(self, value_calculator: ValueCalculator):
        self.value_calculator = value_calculator
        self.optimization_rules = []
        self.load_default_rules()
    
    def load_default_rules(self):
        """Charge les règles d'optimisation par défaut"""
        default_rules = [
            OptimizationRule(
                name="Déplacer items haute valeur vers banque",
                condition="item.market_value > 10000 and item.location == StorageLocation.INVENTORY",
                action="move_to_bank",
                priority=10
            ),
            OptimizationRule(
                name="Grouper ressources identiques",
                condition="item.item_type == ItemType.RESOURCE",
                action="stack_optimize",
                priority=8
            ),
            OptimizationRule(
                name="Vendre items faible valeur peu utilisés",
                condition="item.utility_score < 0.2 and item.market_value < 100",
                action="sell",
                priority=5
            ),
            OptimizationRule(
                name="Stocker équipements niveau élevé",
                condition="item.item_type == ItemType.EQUIPMENT and item.level > 100",
                action="move_to_house",
                priority=7
            )
        ]
        self.optimization_rules.extend(default_rules)
    
    def optimize_storage_allocation(self, inventory: Dict[StorageLocation, List[Item]], 
                                  storage_spaces: Dict[StorageLocation, StorageSpace]) -> List[Tuple[Item, StorageLocation, StorageLocation]]:
        """Optimise l'allocation des items dans les espaces de stockage"""
        moves = []
        
        # Calcul des scores d'efficacité pour chaque item
        item_scores = []
        for location, items in inventory.items():
            for item in items:
                efficiency = self.value_calculator.calculate_storage_efficiency(item)
                value = self.value_calculator.calculate_item_value(item)
                
                item_scores.append({
                    'item': item,
                    'current_location': location,
                    'efficiency': efficiency,
                    'value': value,
                    'weight': item.weight * item.quantity
                })
        
        # Tri par valeur décroissante
        item_scores.sort(key=lambda x: x['value'], reverse=True)
        
        # Optimisation par programmation dynamique simplifiée
        for item_data in item_scores:
            item = item_data['item']
            current_loc = item_data['current_location']
            
            # Trouve le meilleur emplacement pour cet item
            best_location = self._find_best_location(item, storage_spaces, inventory)
            
            if best_location != current_loc and best_location is not None:
                # Vérifier si le déplacement est possible
                target_space = storage_spaces[best_location]
                if (target_space.free_slots > 0 and 
                    target_space.current_weight + item_data['weight'] <= target_space.max_weight):
                    
                    moves.append((item, current_loc, best_location))
                    
                    # Mise à jour temporaire des espaces pour la simulation
                    storage_spaces[current_loc].used_slots -= 1
                    storage_spaces[current_loc].current_weight -= item_data['weight']
                    storage_spaces[best_location].used_slots += 1
                    storage_spaces[best_location].current_weight += item_data['weight']
        
        return moves
    
    def _find_best_location(self, item: Item, storage_spaces: Dict[StorageLocation, StorageSpace], 
                           inventory: Dict[StorageLocation, List[Item]]) -> Optional[StorageLocation]:
        """Trouve le meilleur emplacement pour un item"""
        scores = {}
        
        for location, space in storage_spaces.items():
            if space.free_slots <= 0:
                continue
            
            score = 0
            
            # Facteur de valeur (items précieux vers stockage sécurisé)
            value = self.value_calculator.calculate_item_value(item)
            if location in [StorageLocation.BANK, StorageLocation.HOUSE] and value > 5000:
                score += 10
            
            # Facteur d'accessibilité (items fréquents vers inventaire)
            if item.utility_score > 0.7 and location == StorageLocation.INVENTORY:
                score += 8
            
            # Facteur de coût d'accès
            score -= space.access_cost / 1000  # Normalisation
            score -= space.access_time / 60    # Normalisation par minute
            
            # Facteur d'utilisation de l'espace
            utilization_penalty = space.slot_utilization * 2
            score -= utilization_penalty
            
            # Facteur de regroupement (items similaires ensemble)
            similar_items = sum(1 for inv_item in inventory.get(location, []) 
                              if inv_item.item_type == item.item_type)
            if similar_items > 0:
                score += min(similar_items, 5) * 0.5
            
            scores[location] = score
        
        if not scores:
            return None
        
        return max(scores, key=scores.get)
    
    def optimize_stacks(self, inventory: Dict[StorageLocation, List[Item]]) -> List[Tuple[Item, Item]]:
        """Optimise l'empilement des items identiques"""
        merge_operations = []
        
        for location, items in inventory.items():
            # Grouper par item_id
            items_by_id = defaultdict(list)
            for item in items:
                if not item.is_locked:
                    items_by_id[item.item_id].append(item)
            
            # Optimiser chaque groupe
            for item_id, item_list in items_by_id.items():
                if len(item_list) <= 1:
                    continue
                
                # Trier par quantité décroissante
                item_list.sort(key=lambda x: x.quantity, reverse=True)
                
                base_item = item_list[0]
                for other_item in item_list[1:]:
                    if (base_item.quantity + other_item.quantity <= base_item.stack_size and
                        base_item.location == other_item.location):
                        
                        merge_operations.append((base_item, other_item))
        
        return merge_operations
    
    def suggest_items_to_sell(self, inventory: Dict[StorageLocation, List[Item]], 
                            usage_stats: Dict[int, Dict]) -> List[Tuple[Item, str]]:
        """Suggère les items à vendre avec justification"""
        suggestions = []
        
        all_items = []
        for items in inventory.values():
            all_items.extend(items)
        
        for item in all_items:
            if item.is_locked:
                continue
            
            reasons = []
            sell_score = 0
            
            # Critère: Faible valeur d'utilité
            if item.utility_score < 0.3:
                sell_score += 3
                reasons.append("Faible utilité")
            
            # Critère: Peu utilisé
            stats = usage_stats.get(item.item_id, {})
            usage_frequency = stats.get('usage_frequency', 0)
            if usage_frequency < 0.1:
                sell_score += 2
                reasons.append("Rarement utilisé")
            
            # Critère: Ancienneté
            days_since_acquisition = (datetime.now() - item.acquisition_date).days
            if days_since_acquisition > 30:
                sell_score += 1
                reasons.append("Stocké depuis longtemps")
            
            # Critère: Valeur marchande faible
            if item.market_value < 100:
                sell_score += 1
                reasons.append("Faible valeur marchande")
            
            # Critère: Surstockage
            if item.quantity > item.stack_size * 0.8:
                sell_score += 2
                reasons.append("Surstockage")
            
            # Seuil de recommandation
            if sell_score >= 4:
                suggestion_text = f"Score: {sell_score}/10 - Raisons: {', '.join(reasons)}"
                suggestions.append((item, suggestion_text))
        
        # Trier par score décroissant
        suggestions.sort(key=lambda x: len(x[1].split("Score: ")[1].split("/")[0]), reverse=True)
        
        return suggestions[:20]  # Top 20

class InventoryManager:
    """Gestionnaire d'inventaire principal"""
    
    def __init__(self, db_path: str = "inventory_data.db"):
        self.db = InventoryDatabase(db_path)
        self.value_calculator = ValueCalculator()
        self.optimizer = InventoryOptimizer(self.value_calculator)
        
        # État de l'inventaire
        self.inventory: Dict[StorageLocation, List[Item]] = defaultdict(list)
        self.storage_spaces: Dict[StorageLocation, StorageSpace] = {}
        self.usage_stats: Dict[int, Dict] = {}
        
        # Configuration
        self.auto_optimization = False
        self.optimization_interval = 300  # 5 minutes
        
        # Thread d'optimisation
        self._optimization_thread = None
        self._running = False
        
        self._init_default_storage_spaces()
    
    def _init_default_storage_spaces(self):
        """Initialise les espaces de stockage par défaut"""
        self.storage_spaces = {
            StorageLocation.INVENTORY: StorageSpace(
                location=StorageLocation.INVENTORY,
                total_slots=60,
                used_slots=0,
                max_weight=1000.0,
                current_weight=0.0,
                access_cost=0.0,
                access_time=0.0
            ),
            StorageLocation.BANK: StorageSpace(
                location=StorageLocation.BANK,
                total_slots=200,
                used_slots=0,
                max_weight=10000.0,
                current_weight=0.0,
                access_cost=0.0,
                access_time=30.0
            ),
            StorageLocation.GUILD_BANK: StorageSpace(
                location=StorageLocation.GUILD_BANK,
                total_slots=100,
                used_slots=0,
                max_weight=5000.0,
                current_weight=0.0,
                access_cost=0.0,
                access_time=60.0
            ),
            StorageLocation.HOUSE: StorageSpace(
                location=StorageLocation.HOUSE,
                total_slots=400,
                used_slots=0,
                max_weight=50000.0,
                current_weight=0.0,
                access_cost=100.0,  # Coût de téléportation
                access_time=120.0
            )
        }
    
    def add_item(self, item: Item, location: StorageLocation = StorageLocation.INVENTORY) -> bool:
        """Ajoute un item à l'inventaire"""
        # Vérification de l'espace disponible
        storage = self.storage_spaces[location]
        item_weight = item.weight * item.quantity
        
        if (storage.free_slots <= 0 or 
            storage.current_weight + item_weight > storage.max_weight):
            return False
        
        # Tentative de stack avec un item existant
        for existing_item in self.inventory[location]:
            if (existing_item.item_id == item.item_id and 
                existing_item.quantity + item.quantity <= existing_item.stack_size):
                
                existing_item.quantity += item.quantity
                storage.current_weight += item_weight
                
                self.db.log_item_action(item, "stacked", "")
                self.db.update_usage_stats(item.item_id, "acquired")
                return True
        
        # Ajout comme nouvel item
        item.location = location
        self.inventory[location].append(item)
        storage.used_slots += 1
        storage.current_weight += item_weight
        
        self.db.log_item_action(item, "added", "")
        self.db.update_usage_stats(item.item_id, "acquired")
        return True
    
    def remove_item(self, item_id: int, quantity: int = 1, 
                   location: StorageLocation = None) -> Optional[Item]:
        """Retire un item de l'inventaire"""
        locations_to_check = [location] if location else self.inventory.keys()
        
        for loc in locations_to_check:
            for item in self.inventory[loc]:
                if item.item_id == item_id:
                    if item.quantity >= quantity:
                        if item.quantity == quantity:
                            # Retirer complètement
                            self.inventory[loc].remove(item)
                            self.storage_spaces[loc].used_slots -= 1
                            self.storage_spaces[loc].current_weight -= item.weight * item.quantity
                            
                            self.db.log_item_action(item, "removed", "")
                            return item
                        else:
                            # Retirer partiellement
                            item.quantity -= quantity
                            removed_item = Item(
                                item_id=item.item_id,
                                name=item.name,
                                quantity=quantity,
                                location=item.location,
                                item_type=item.item_type,
                                level=item.level,
                                weight=item.weight,
                                market_value=item.market_value
                            )
                            
                            self.storage_spaces[loc].current_weight -= item.weight * quantity
                            self.db.log_item_action(removed_item, "removed", "")
                            return removed_item
        
        return None
    
    def move_item(self, item: Item, target_location: StorageLocation) -> bool:
        """Déplace un item vers un autre emplacement"""
        if item.location == target_location:
            return True
        
        # Vérifier l'espace disponible
        target_space = self.storage_spaces[target_location]
        item_weight = item.weight * item.quantity
        
        if (target_space.free_slots <= 0 or 
            target_space.current_weight + item_weight > target_space.max_weight):
            return False
        
        # Retirer de l'emplacement actuel
        current_location = item.location
        self.inventory[current_location].remove(item)
        self.storage_spaces[current_location].used_slots -= 1
        self.storage_spaces[current_location].current_weight -= item_weight
        
        # Ajouter au nouvel emplacement
        item.location = target_location
        self.inventory[target_location].append(item)
        self.storage_spaces[target_location].used_slots += 1
        self.storage_spaces[target_location].current_weight += item_weight
        
        self.db.log_item_action(item, f"moved_to_{target_location.value}", "")
        return True
    
    def optimize_inventory(self) -> Dict[str, Any]:
        """Lance l'optimisation complète de l'inventaire"""
        start_time = time.time()
        optimization_report = {
            "timestamp": datetime.now(),
            "moves_performed": [],
            "stacks_optimized": 0,
            "space_freed": 0,
            "value_relocated": 0.0,
            "execution_time": 0.0
        }
        
        # 1. Optimisation des stacks
        merge_operations = self.optimizer.optimize_stacks(self.inventory)
        for base_item, other_item in merge_operations:
            if base_item.quantity + other_item.quantity <= base_item.stack_size:
                base_item.quantity += other_item.quantity
                self.inventory[base_item.location].remove(other_item)
                self.storage_spaces[base_item.location].used_slots -= 1
                
                optimization_report["stacks_optimized"] += 1
                optimization_report["space_freed"] += 1
        
        # 2. Réallocation optimale
        moves = self.optimizer.optimize_storage_allocation(self.inventory, self.storage_spaces)
        for item, from_loc, to_loc in moves:
            if self.move_item(item, to_loc):
                optimization_report["moves_performed"].append({
                    "item_id": item.item_id,
                    "from": from_loc.value,
                    "to": to_loc.value,
                    "value": self.value_calculator.calculate_item_value(item)
                })
                optimization_report["value_relocated"] += self.value_calculator.calculate_item_value(item)
        
        # 3. Mise à jour du temps d'exécution
        optimization_report["execution_time"] = time.time() - start_time
        
        # Log en base de données
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO optimizations 
            (optimization_type, items_moved, space_saved, value_optimized, timestamp, efficiency_gain)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            "full_optimization",
            len(optimization_report["moves_performed"]),
            optimization_report["space_freed"],
            optimization_report["value_relocated"],
            datetime.now(),
            optimization_report["space_freed"] / max(1, optimization_report["execution_time"])
        ))
        conn.commit()
        conn.close()
        
        return optimization_report
    
    def get_inventory_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'inventaire"""
        summary = {
            "total_items": 0,
            "total_value": 0.0,
            "storage_utilization": {},
            "item_distribution": defaultdict(int),
            "top_valuable_items": []
        }
        
        all_items = []
        for location, items in self.inventory.items():
            summary["total_items"] += len(items)
            
            for item in items:
                item_value = self.value_calculator.calculate_item_value(item)
                summary["total_value"] += item_value
                summary["item_distribution"][item.item_type.value] += 1
                all_items.append((item, item_value))
            
            # Utilisation du stockage
            space = self.storage_spaces[location]
            summary["storage_utilization"][location.value] = {
                "slot_usage": f"{space.used_slots}/{space.total_slots}",
                "weight_usage": f"{space.current_weight:.1f}/{space.max_weight:.1f}",
                "slot_percentage": space.slot_utilization * 100,
                "weight_percentage": space.weight_utilization * 100
            }
        
        # Top 10 des items les plus précieux
        all_items.sort(key=lambda x: x[1], reverse=True)
        summary["top_valuable_items"] = [
            {
                "name": item.name,
                "quantity": item.quantity,
                "location": item.location.value,
                "value": value
            }
            for item, value in all_items[:10]
        ]
        
        return summary
    
    def get_selling_suggestions(self) -> List[Tuple[Item, str]]:
        """Obtient les suggestions de vente"""
        return self.optimizer.suggest_items_to_sell(self.inventory, self.usage_stats)
    
    def start_auto_optimization(self):
        """Démarre l'optimisation automatique"""
        if self.auto_optimization:
            return
        
        self.auto_optimization = True
        self._running = True
        self._optimization_thread = threading.Thread(target=self._auto_optimization_loop)
        self._optimization_thread.daemon = True
        self._optimization_thread.start()
    
    def stop_auto_optimization(self):
        """Arrête l'optimisation automatique"""
        self.auto_optimization = False
        self._running = False
        if self._optimization_thread:
            self._optimization_thread.join()
    
    def _auto_optimization_loop(self):
        """Boucle d'optimisation automatique"""
        while self._running:
            try:
                if self.auto_optimization:
                    self.optimize_inventory()
                
                time.sleep(self.optimization_interval)
            except Exception as e:
                print(f"Erreur lors de l'optimisation automatique: {e}")
                time.sleep(60)
    
    def export_inventory_report(self, filepath: str):
        """Exporte un rapport complet d'inventaire"""
        summary = self.get_inventory_summary()
        suggestions = self.get_selling_suggestions()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "selling_suggestions": [
                {
                    "item": {
                        "id": item.item_id,
                        "name": item.name,
                        "quantity": item.quantity,
                        "location": item.location.value,
                        "market_value": item.market_value
                    },
                    "reason": reason
                }
                for item, reason in suggestions[:10]
            ],
            "optimization_history": []  # TODO: Implémenter l'historique
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation du gestionnaire
    manager = InventoryManager()
    
    # Exemple d'ajout d'items
    items_example = [
        Item(
            item_id=1,
            name="Blé",
            quantity=50,
            location=StorageLocation.INVENTORY,
            item_type=ItemType.RESOURCE,
            weight=0.1,
            market_value=10.0
        ),
        Item(
            item_id=2,
            name="Épée Royale",
            quantity=1,
            location=StorageLocation.INVENTORY,
            item_type=ItemType.EQUIPMENT,
            level=150,
            weight=5.0,
            market_value=50000.0,
            utility_score=0.9
        )
    ]
    
    # Ajout des items
    for item in items_example:
        success = manager.add_item(item)
        print(f"Ajout de {item.name}: {'Succès' if success else 'Échec'}")
    
    # Résumé de l'inventaire
    summary = manager.get_inventory_summary()
    print(f"\nRésumé de l'inventaire:")
    print(f"Total items: {summary['total_items']}")
    print(f"Valeur totale: {summary['total_value']:.2f}")
    
    # Optimisation
    print("\nLancement de l'optimisation...")
    report = manager.optimize_inventory()
    print(f"Optimisation terminée en {report['execution_time']:.2f}s")
    print(f"Déplacements: {len(report['moves_performed'])}")
    
    # Suggestions de vente
    suggestions = manager.get_selling_suggestions()
    if suggestions:
        print(f"\nSuggestions de vente ({len(suggestions)}):")
        for item, reason in suggestions[:3]:
            print(f"- {item.name} (x{item.quantity}): {reason}")
    
    # Export du rapport
    manager.export_inventory_report("inventory_report.json")
    print("\nRapport exporté: inventory_report.json")