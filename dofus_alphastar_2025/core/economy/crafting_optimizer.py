"""
Optimiseur de craft intelligent pour Dofus
Calculs de rentabilité, gestion des queues, optimisation multi-critères
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from collections import defaultdict, deque
import heapq
import threading
import time
import math

class CraftType(Enum):
    """Types de craft"""
    EQUIPMENT = "equipment"
    CONSUMABLE = "consumable" 
    RESOURCE = "resource"
    RUNE = "rune"
    PET = "pet"
    MOUNT = "mount"

class OptimizationGoal(Enum):
    """Objectifs d'optimisation"""
    PROFIT = "profit"          # Maximiser les profits
    XP = "xp"                 # Maximiser l'XP
    TIME = "time"             # Minimiser le temps
    BALANCED = "balanced"      # Équilibré
    RESOURCES = "resources"    # Optimiser l'usage des ressources

@dataclass
class Recipe:
    """Représente une recette de craft"""
    recipe_id: int
    name: str
    profession: str
    level_required: int
    ingredients: Dict[int, int]  # item_id -> quantity needed
    result_item_id: int
    result_quantity: int = 1
    craft_time: float = 30.0  # secondes
    xp_gained: int = 0
    success_rate: float = 1.0
    craft_cost: float = 0.0  # Coûts fixes (ateliers, etc.)
    
    def __post_init__(self):
        """Calculs post-initialisation"""
        if self.xp_gained == 0:
            # Estimation basée sur le niveau requis
            self.xp_gained = max(1, int(self.level_required * 2.5))

@dataclass
class CraftJob:
    """Tâche de craft dans la queue"""
    job_id: str
    recipe: Recipe
    quantity_to_craft: int
    priority: float
    estimated_profit: float
    estimated_xp: int
    estimated_time: float
    required_materials: Dict[int, int]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "queued"  # queued, in_progress, completed, failed
    
    def __lt__(self, other):
        """Comparaison pour la priority queue (inversée car heapq est min-heap)"""
        return self.priority > other.priority

@dataclass
class CraftAnalysis:
    """Analyse de rentabilité d'une recette"""
    recipe_id: int
    profit_per_craft: float
    profit_margin: float
    roi_percentage: float
    xp_per_hour: float
    profit_per_hour: float
    break_even_quantity: int
    risk_score: float
    market_saturation: float
    resource_availability: float
    confidence_score: float

@dataclass
class ProfessionStatus:
    """État d'une profession"""
    profession_name: str
    current_level: int
    current_xp: int
    xp_to_next_level: int
    available_recipes: List[int]
    efficiency_bonus: float = 1.0
    workshop_bonus: float = 1.0

class CraftingDatabase:
    """Base de données pour les données de craft"""
    
    def __init__(self, db_path: str = "crafting_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialise les tables de la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Recettes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipes (
                recipe_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                profession TEXT NOT NULL,
                level_required INTEGER NOT NULL,
                result_item_id INTEGER NOT NULL,
                result_quantity INTEGER DEFAULT 1,
                craft_time REAL DEFAULT 30.0,
                xp_gained INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0,
                craft_cost REAL DEFAULT 0.0
            )
        ''')
        
        # Ingrédients des recettes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipe_ingredients (
                recipe_id INTEGER NOT NULL,
                item_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                FOREIGN KEY (recipe_id) REFERENCES recipes (recipe_id)
            )
        ''')
        
        # Historique des crafts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS craft_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id INTEGER NOT NULL,
                quantity_crafted INTEGER NOT NULL,
                actual_profit REAL,
                actual_time REAL,
                xp_gained INTEGER,
                timestamp DATETIME NOT NULL,
                success_rate REAL,
                materials_cost REAL
            )
        ''')
        
        # Analyses de rentabilité
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profitability_analysis (
                recipe_id INTEGER PRIMARY KEY,
                profit_per_craft REAL NOT NULL,
                profit_margin REAL NOT NULL,
                roi_percentage REAL NOT NULL,
                xp_per_hour REAL NOT NULL,
                profit_per_hour REAL NOT NULL,
                risk_score REAL NOT NULL,
                last_updated DATETIME NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_recipe(self, recipe: Recipe):
        """Stocke une recette en base"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insertion de la recette
        cursor.execute('''
            INSERT OR REPLACE INTO recipes 
            (recipe_id, name, profession, level_required, result_item_id, 
             result_quantity, craft_time, xp_gained, success_rate, craft_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            recipe.recipe_id, recipe.name, recipe.profession, recipe.level_required,
            recipe.result_item_id, recipe.result_quantity, recipe.craft_time,
            recipe.xp_gained, recipe.success_rate, recipe.craft_cost
        ))
        
        # Suppression des anciens ingrédients
        cursor.execute('DELETE FROM recipe_ingredients WHERE recipe_id = ?', (recipe.recipe_id,))
        
        # Insertion des ingrédients
        for item_id, quantity in recipe.ingredients.items():
            cursor.execute('''
                INSERT INTO recipe_ingredients (recipe_id, item_id, quantity)
                VALUES (?, ?, ?)
            ''', (recipe.recipe_id, item_id, quantity))
        
        conn.commit()
        conn.close()
    
    def get_recipe(self, recipe_id: int) -> Optional[Recipe]:
        """Récupère une recette"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Récupération de la recette
        cursor.execute('SELECT * FROM recipes WHERE recipe_id = ?', (recipe_id,))
        recipe_data = cursor.fetchone()
        
        if not recipe_data:
            conn.close()
            return None
        
        # Récupération des ingrédients
        cursor.execute('SELECT item_id, quantity FROM recipe_ingredients WHERE recipe_id = ?', (recipe_id,))
        ingredients_data = cursor.fetchall()
        
        conn.close()
        
        ingredients = {item_id: quantity for item_id, quantity in ingredients_data}
        
        return Recipe(
            recipe_id=recipe_data[0],
            name=recipe_data[1],
            profession=recipe_data[2],
            level_required=recipe_data[3],
            result_item_id=recipe_data[4],
            result_quantity=recipe_data[5],
            craft_time=recipe_data[6],
            xp_gained=recipe_data[7],
            success_rate=recipe_data[8],
            craft_cost=recipe_data[9],
            ingredients=ingredients
        )
    
    def log_craft_completion(self, job: CraftJob, actual_profit: float, actual_time: float):
        """Enregistre la completion d'un craft"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO craft_history 
            (recipe_id, quantity_crafted, actual_profit, actual_time, 
             xp_gained, timestamp, success_rate, materials_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job.recipe.recipe_id, job.quantity_to_craft, actual_profit,
            actual_time, job.estimated_xp, datetime.now(),
            job.recipe.success_rate, 0.0  # TODO: Calculer le coût réel des matériaux
        ))
        
        conn.commit()
        conn.close()

class ProfitabilityCalculator:
    """Calculateur de rentabilité des crafts"""
    
    def __init__(self):
        self.market_prices = {}  # item_id -> price
        self.resource_costs = {}  # item_id -> cost
        self.market_demand = {}  # item_id -> demand_score
        self.volatility_data = {}  # item_id -> volatility
    
    def update_market_data(self, item_prices: Dict[int, float], demand_data: Dict[int, float] = None):
        """Met à jour les données de marché"""
        self.market_prices.update(item_prices)
        if demand_data:
            self.market_demand.update(demand_data)
    
    def calculate_recipe_profitability(self, recipe: Recipe, 
                                     current_profession_level: int = None) -> CraftAnalysis:
        """Calcule la rentabilité d'une recette"""
        
        # Vérification du niveau requis
        if current_profession_level and current_profession_level < recipe.level_required:
            return CraftAnalysis(
                recipe_id=recipe.recipe_id,
                profit_per_craft=0,
                profit_margin=0,
                roi_percentage=0,
                xp_per_hour=0,
                profit_per_hour=0,
                break_even_quantity=float('inf'),
                risk_score=1.0,
                market_saturation=1.0,
                resource_availability=0.0,
                confidence_score=0.0
            )
        
        # Calcul du coût des matériaux
        materials_cost = self._calculate_materials_cost(recipe.ingredients)
        
        # Prix de vente du produit fini
        result_price = self.market_prices.get(recipe.result_item_id, 0)
        total_revenue = result_price * recipe.result_quantity
        
        # Calcul des profits
        total_cost = materials_cost + recipe.craft_cost
        profit_per_craft = (total_revenue - total_cost) * recipe.success_rate
        
        if total_cost > 0:
            profit_margin = profit_per_craft / total_revenue
            roi_percentage = (profit_per_craft / total_cost) * 100
        else:
            profit_margin = 0
            roi_percentage = 0
        
        # Calculs temporels
        effective_craft_time = recipe.craft_time / recipe.success_rate
        crafts_per_hour = 3600 / effective_craft_time
        profit_per_hour = profit_per_craft * crafts_per_hour
        xp_per_hour = recipe.xp_gained * crafts_per_hour
        
        # Point mort
        if profit_per_craft > 0:
            break_even_quantity = max(1, int(math.ceil(recipe.craft_cost / profit_per_craft)))
        else:
            break_even_quantity = float('inf')
        
        # Évaluation du risque
        risk_score = self._calculate_risk_score(recipe)
        
        # Saturation du marché
        market_saturation = self._calculate_market_saturation(recipe.result_item_id)
        
        # Disponibilité des ressources
        resource_availability = self._calculate_resource_availability(recipe.ingredients)
        
        # Score de confiance global
        confidence_score = self._calculate_confidence_score(
            recipe, risk_score, market_saturation, resource_availability
        )
        
        return CraftAnalysis(
            recipe_id=recipe.recipe_id,
            profit_per_craft=profit_per_craft,
            profit_margin=profit_margin,
            roi_percentage=roi_percentage,
            xp_per_hour=xp_per_hour,
            profit_per_hour=profit_per_hour,
            break_even_quantity=break_even_quantity,
            risk_score=risk_score,
            market_saturation=market_saturation,
            resource_availability=resource_availability,
            confidence_score=confidence_score
        )
    
    def _calculate_materials_cost(self, ingredients: Dict[int, int]) -> float:
        """Calcule le coût total des matériaux"""
        total_cost = 0
        for item_id, quantity in ingredients.items():
            item_price = self.market_prices.get(item_id, 0)
            total_cost += item_price * quantity
        return total_cost
    
    def _calculate_risk_score(self, recipe: Recipe) -> float:
        """Calcule le score de risque (0 = faible risque, 1 = risque élevé)"""
        risk = 0.0
        
        # Risque lié au taux de réussite
        risk += (1.0 - recipe.success_rate) * 0.3
        
        # Risque lié à la volatilité des prix
        result_volatility = self.volatility_data.get(recipe.result_item_id, 0.1)
        risk += result_volatility * 0.3
        
        # Risque lié aux matériaux
        materials_volatility = 0
        for item_id in recipe.ingredients:
            materials_volatility += self.volatility_data.get(item_id, 0.1)
        materials_volatility /= len(recipe.ingredients) if recipe.ingredients else 1
        risk += materials_volatility * 0.2
        
        # Risque lié au niveau requis (plus élevé = plus stable)
        level_risk = max(0, (50 - recipe.level_required) / 50) * 0.2
        risk += level_risk
        
        return min(1.0, risk)
    
    def _calculate_market_saturation(self, item_id: int) -> float:
        """Calcule la saturation du marché (0 = non saturé, 1 = saturé)"""
        # TODO: Implémenter avec des données réelles de marché
        # Pour l'instant, retourne une valeur par défaut basée sur la demande
        demand = self.market_demand.get(item_id, 0.5)
        return max(0.0, min(1.0, 1.0 - demand))
    
    def _calculate_resource_availability(self, ingredients: Dict[int, int]) -> float:
        """Calcule la disponibilité des ressources (0 = indisponible, 1 = très disponible)"""
        if not ingredients:
            return 1.0
        
        availability_scores = []
        for item_id, quantity in ingredients.items():
            # Score basé sur le prix (plus cher = moins disponible)
            price = self.market_prices.get(item_id, 1000)
            price_score = max(0.1, min(1.0, 100 / price))  # Normalisation
            
            # Score basé sur la quantité requise
            quantity_score = max(0.1, min(1.0, 10 / quantity))  # Plus on en a besoin, moins c'est disponible
            
            availability_scores.append(price_score * quantity_score)
        
        return sum(availability_scores) / len(availability_scores)
    
    def _calculate_confidence_score(self, recipe: Recipe, risk_score: float, 
                                  market_saturation: float, resource_availability: float) -> float:
        """Calcule le score de confiance global"""
        confidence = 1.0
        
        # Réduction basée sur le risque
        confidence *= (1.0 - risk_score)
        
        # Réduction basée sur la saturation
        confidence *= (1.0 - market_saturation)
        
        # Bonus pour la disponibilité des ressources
        confidence *= resource_availability
        
        # Bonus pour le niveau de la recette (expérience = confiance)
        level_confidence = min(1.0, recipe.level_required / 200)
        confidence = (confidence + level_confidence) / 2
        
        return max(0.0, min(1.0, confidence))

class CraftQueue:
    """Gestionnaire de queue de craft optimisée"""
    
    def __init__(self):
        self.jobs: List[CraftJob] = []
        self.priority_queue = []  # Heap pour l'optimisation
        self.job_counter = 0
        self.processing_job: Optional[CraftJob] = None
        self.lock = threading.Lock()
    
    def add_job(self, recipe: Recipe, quantity: int, goal: OptimizationGoal = OptimizationGoal.BALANCED) -> str:
        """Ajoute un job à la queue"""
        with self.lock:
            job_id = f"craft_{self.job_counter:06d}"
            self.job_counter += 1
            
            # Calcul de la priorité basée sur l'objectif
            priority = self._calculate_priority(recipe, quantity, goal)
            
            # Estimation des ressources nécessaires
            required_materials = {
                item_id: qty * quantity 
                for item_id, qty in recipe.ingredients.items()
            }
            
            job = CraftJob(
                job_id=job_id,
                recipe=recipe,
                quantity_to_craft=quantity,
                priority=priority,
                estimated_profit=0,  # TODO: Calculer avec ProfitabilityCalculator
                estimated_xp=recipe.xp_gained * quantity,
                estimated_time=recipe.craft_time * quantity,
                required_materials=required_materials
            )
            
            self.jobs.append(job)
            heapq.heappush(self.priority_queue, job)
            
            return job_id
    
    def get_next_job(self) -> Optional[CraftJob]:
        """Récupère le prochain job à traiter"""
        with self.lock:
            if self.priority_queue:
                job = heapq.heappop(self.priority_queue)
                job.status = "in_progress"
                job.started_at = datetime.now()
                self.processing_job = job
                return job
            return None
    
    def complete_job(self, job_id: str, success: bool = True):
        """Marque un job comme terminé"""
        with self.lock:
            for job in self.jobs:
                if job.job_id == job_id:
                    job.status = "completed" if success else "failed"
                    job.completed_at = datetime.now()
                    
                    if self.processing_job and self.processing_job.job_id == job_id:
                        self.processing_job = None
                    break
    
    def reorder_queue(self, new_goal: OptimizationGoal):
        """Réorganise la queue selon un nouvel objectif"""
        with self.lock:
            # Recalcule les priorités
            for job in self.jobs:
                if job.status == "queued":
                    job.priority = self._calculate_priority(job.recipe, job.quantity_to_craft, new_goal)
            
            # Reconstruit le heap
            queued_jobs = [job for job in self.jobs if job.status == "queued"]
            self.priority_queue = queued_jobs[:]
            heapq.heapify(self.priority_queue)
    
    def _calculate_priority(self, recipe: Recipe, quantity: int, goal: OptimizationGoal) -> float:
        """Calcule la priorité d'un job"""
        base_priority = 0.0
        
        if goal == OptimizationGoal.PROFIT:
            # Priorité basée sur le profit estimé par heure
            profit_per_hour = 100  # TODO: Calculer réellement
            base_priority = profit_per_hour
            
        elif goal == OptimizationGoal.XP:
            # Priorité basée sur l'XP par heure
            xp_per_hour = recipe.xp_gained / (recipe.craft_time / 3600)
            base_priority = xp_per_hour
            
        elif goal == OptimizationGoal.TIME:
            # Priorité inversée au temps (plus rapide = plus prioritaire)
            base_priority = 1000 / recipe.craft_time
            
        elif goal == OptimizationGoal.BALANCED:
            # Combinaison équilibrée
            profit_score = 100 * 0.4  # TODO: Calculer
            xp_score = (recipe.xp_gained / recipe.craft_time) * 0.3
            time_score = (1000 / recipe.craft_time) * 0.3
            base_priority = profit_score + xp_score + time_score
            
        elif goal == OptimizationGoal.RESOURCES:
            # Priorité basée sur l'efficacité d'utilisation des ressources
            resource_count = len(recipe.ingredients)
            base_priority = 1000 / max(1, resource_count)
        
        # Facteurs de modification
        # Bonus pour les petites quantités (finissent plus vite)
        quantity_factor = max(0.1, 100 / quantity)
        
        # Bonus pour les hauts niveaux (plus de valeur)
        level_factor = recipe.level_required / 200
        
        return base_priority * (1 + quantity_factor + level_factor)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Retourne l'état de la queue"""
        with self.lock:
            status = {
                "total_jobs": len(self.jobs),
                "queued": len([j for j in self.jobs if j.status == "queued"]),
                "in_progress": len([j for j in self.jobs if j.status == "in_progress"]),
                "completed": len([j for j in self.jobs if j.status == "completed"]),
                "failed": len([j for j in self.jobs if j.status == "failed"]),
                "current_job": None,
                "estimated_total_time": sum(j.estimated_time for j in self.jobs if j.status == "queued"),
                "estimated_total_xp": sum(j.estimated_xp for j in self.jobs if j.status == "queued")
            }
            
            if self.processing_job:
                status["current_job"] = {
                    "job_id": self.processing_job.job_id,
                    "recipe_name": self.processing_job.recipe.name,
                    "quantity": self.processing_job.quantity_to_craft,
                    "progress": 0,  # TODO: Calculer le progrès réel
                    "started_at": self.processing_job.started_at.isoformat() if self.processing_job.started_at else None
                }
            
            return status

class ResourceManager:
    """Gestionnaire de ressources pour le craft"""
    
    def __init__(self):
        self.available_resources = defaultdict(int)  # item_id -> quantity
        self.reserved_resources = defaultdict(int)    # item_id -> quantity réservée
        self.resource_sources = defaultdict(list)     # item_id -> list of sources
    
    def update_inventory(self, resources: Dict[int, int]):
        """Met à jour l'inventaire des ressources disponibles"""
        self.available_resources.update(resources)
    
    def check_availability(self, requirements: Dict[int, int]) -> Tuple[bool, Dict[int, int]]:
        """Vérifie la disponibilité des ressources et retourne les manquantes"""
        missing = {}
        available = True
        
        for item_id, needed in requirements.items():
            available_qty = self.available_resources[item_id] - self.reserved_resources[item_id]
            if available_qty < needed:
                missing[item_id] = needed - available_qty
                available = False
        
        return available, missing
    
    def reserve_resources(self, requirements: Dict[int, int], job_id: str) -> bool:
        """Réserve des ressources pour un job"""
        available, missing = self.check_availability(requirements)
        
        if not available:
            return False
        
        # Réservation
        for item_id, quantity in requirements.items():
            self.reserved_resources[item_id] += quantity
        
        return True
    
    def release_resources(self, requirements: Dict[int, int], job_id: str, consumed: bool = True):
        """Libère les ressources réservées"""
        for item_id, quantity in requirements.items():
            # Libère la réservation
            self.reserved_resources[item_id] = max(0, self.reserved_resources[item_id] - quantity)
            
            # Si consommées, retire de l'inventaire
            if consumed:
                self.available_resources[item_id] = max(0, self.available_resources[item_id] - quantity)
    
    def suggest_acquisition_strategy(self, missing_resources: Dict[int, int]) -> List[Dict[str, Any]]:
        """Suggère des stratégies pour acquérir les ressources manquantes"""
        strategies = []
        
        for item_id, needed_qty in missing_resources.items():
            # Différentes stratégies possibles
            item_strategies = {
                "item_id": item_id,
                "needed_quantity": needed_qty,
                "options": []
            }
            
            # Option 1: Achat au marché
            market_price = 100  # TODO: Récupérer le prix réel
            item_strategies["options"].append({
                "method": "market_buy",
                "cost": market_price * needed_qty,
                "time": 60,  # 1 minute pour acheter
                "reliability": 0.9
            })
            
            # Option 2: Farm/Récolte
            item_strategies["options"].append({
                "method": "farming",
                "cost": 0,
                "time": needed_qty * 30,  # 30 sec par ressource
                "reliability": 0.7
            })
            
            # Option 3: Craft si c'est possible
            item_strategies["options"].append({
                "method": "craft",
                "cost": 50 * needed_qty,  # Coût des matières premières
                "time": needed_qty * 60,  # 1 min par craft
                "reliability": 0.8
            })
            
            # Tri par efficacité (coût/temps)
            for option in item_strategies["options"]:
                if option["time"] > 0:
                    option["efficiency"] = option["reliability"] / (option["cost"] + option["time"])
                else:
                    option["efficiency"] = float('inf')
            
            item_strategies["options"].sort(key=lambda x: x["efficiency"], reverse=True)
            strategies.append(item_strategies)
        
        return strategies

class CraftingOptimizer:
    """Optimiseur de craft principal"""
    
    def __init__(self, db_path: str = "crafting_data.db"):
        self.db = CraftingDatabase(db_path)
        self.profitability_calc = ProfitabilityCalculator()
        self.craft_queue = CraftQueue()
        self.resource_manager = ResourceManager()
        
        # État des professions
        self.professions: Dict[str, ProfessionStatus] = {}
        
        # Configuration
        self.auto_optimization = False
        self.current_goal = OptimizationGoal.BALANCED
        
        # Thread de traitement
        self._processing_thread = None
        self._running = False
    
    def add_profession(self, profession_name: str, level: int, current_xp: int = 0):
        """Ajoute une profession"""
        # TODO: Calculer l'XP nécessaire au niveau suivant
        xp_to_next = max(0, (level + 1) * 1000 - current_xp)
        
        self.professions[profession_name] = ProfessionStatus(
            profession_name=profession_name,
            current_level=level,
            current_xp=current_xp,
            xp_to_next_level=xp_to_next,
            available_recipes=[]  # TODO: Charger depuis la base
        )
    
    def analyze_recipe(self, recipe_id: int) -> Optional[CraftAnalysis]:
        """Analyse une recette"""
        recipe = self.db.get_recipe(recipe_id)
        if not recipe:
            return None
        
        return self.profitability_calc.calculate_recipe_profitability(recipe)
    
    def find_optimal_recipes(self, goal: OptimizationGoal, 
                           profession: str = None, 
                           max_level: int = None,
                           min_profit: float = 0) -> List[Tuple[Recipe, CraftAnalysis]]:
        """Trouve les recettes optimales selon les critères"""
        optimal_recipes = []
        
        # TODO: Récupérer toutes les recettes depuis la base
        # Pour l'exemple, on utilise des données fictives
        
        return optimal_recipes
    
    def create_crafting_plan(self, target_items: Dict[int, int], 
                           goal: OptimizationGoal = OptimizationGoal.BALANCED) -> Dict[str, Any]:
        """Crée un plan de craft optimisé"""
        plan = {
            "target_items": target_items,
            "optimization_goal": goal.value,
            "crafting_jobs": [],
            "resource_requirements": defaultdict(int),
            "estimated_stats": {
                "total_time": 0,
                "total_cost": 0,
                "total_profit": 0,
                "total_xp": 0
            },
            "acquisition_strategy": []
        }
        
        for item_id, quantity in target_items.items():
            # TODO: Trouver les recettes pour cet item
            # TODO: Optimiser la quantité et les recettes
            # TODO: Ajouter les jobs à la queue
            pass
        
        return plan
    
    def start_auto_crafting(self):
        """Démarre le craft automatique"""
        if self._running:
            return
        
        self._running = True
        self._processing_thread = threading.Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
    
    def stop_auto_crafting(self):
        """Arrête le craft automatique"""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join()
    
    def _processing_loop(self):
        """Boucle de traitement des crafts"""
        while self._running:
            try:
                job = self.craft_queue.get_next_job()
                if job:
                    self._process_craft_job(job)
                else:
                    time.sleep(1)  # Attendre s'il n'y a pas de jobs
            except Exception as e:
                print(f"Erreur lors du traitement des crafts: {e}")
                time.sleep(5)
    
    def _process_craft_job(self, job: CraftJob):
        """Traite un job de craft"""
        print(f"Démarrage du craft: {job.recipe.name} x{job.quantity_to_craft}")
        
        # Vérification des ressources
        available, missing = self.resource_manager.check_availability(job.required_materials)
        
        if not available:
            print(f"Ressources manquantes: {missing}")
            job.status = "failed"
            return
        
        # Réservation des ressources
        if not self.resource_manager.reserve_resources(job.required_materials, job.job_id):
            print("Impossible de réserver les ressources")
            job.status = "failed"
            return
        
        try:
            # Simulation du craft (à remplacer par l'interface réelle)
            craft_time = job.estimated_time
            time.sleep(min(craft_time, 5))  # Maximum 5 secondes pour la démo
            
            # Craft terminé avec succès
            self.resource_manager.release_resources(job.required_materials, job.job_id, consumed=True)
            self.craft_queue.complete_job(job.job_id, success=True)
            
            # Log en base de données
            self.db.log_craft_completion(job, job.estimated_profit, craft_time)
            
            print(f"Craft terminé: {job.recipe.name} x{job.quantity_to_craft}")
            
        except Exception as e:
            # Échec du craft
            print(f"Échec du craft: {e}")
            self.resource_manager.release_resources(job.required_materials, job.job_id, consumed=False)
            self.craft_queue.complete_job(job.job_id, success=False)
    
    def get_optimizer_status(self) -> Dict[str, Any]:
        """Retourne l'état de l'optimiseur"""
        return {
            "auto_crafting_active": self._running,
            "current_goal": self.current_goal.value,
            "queue_status": self.craft_queue.get_queue_status(),
            "professions": {name: {
                "level": prof.current_level,
                "xp": prof.current_xp,
                "xp_to_next": prof.xp_to_next_level
            } for name, prof in self.professions.items()},
            "resource_summary": {
                "available_items": len(self.resource_manager.available_resources),
                "reserved_items": sum(self.resource_manager.reserved_resources.values())
            }
        }
    
    def export_crafting_report(self, filepath: str):
        """Exporte un rapport de craft"""
        status = self.get_optimizer_status()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "optimizer_status": status,
            "recent_crafts": [],  # TODO: Récupérer depuis la base
            "profitability_rankings": [],  # TODO: Ranking des recettes
            "resource_analysis": {},
            "recommendations": []
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation de l'optimiseur
    optimizer = CraftingOptimizer()
    
    # Ajout de professions
    optimizer.add_profession("Boulanger", 120, 250000)
    optimizer.add_profession("Forgeron", 95, 180000)
    
    # Exemple de recette
    bread_recipe = Recipe(
        recipe_id=1001,
        name="Pain Complet",
        profession="Boulanger",
        level_required=50,
        ingredients={101: 10, 102: 5},  # 10 Blé + 5 Eau
        result_item_id=201,
        result_quantity=3,
        craft_time=45.0,
        xp_gained=125,
        success_rate=0.95
    )
    
    # Stockage de la recette
    optimizer.db.store_recipe(bread_recipe)
    
    # Mise à jour des prix de marché
    optimizer.profitability_calc.update_market_data({
        101: 15.0,  # Blé: 15k
        102: 5.0,   # Eau: 5k
        201: 80.0   # Pain: 80k
    })
    
    # Mise à jour des ressources disponibles
    optimizer.resource_manager.update_inventory({
        101: 500,  # 500 Blé
        102: 200   # 200 Eau
    })
    
    # Analyse de la recette
    analysis = optimizer.analyze_recipe(1001)
    if analysis:
        print(f"Analyse de {bread_recipe.name}:")
        print(f"  Profit par craft: {analysis.profit_per_craft:.2f}")
        print(f"  ROI: {analysis.roi_percentage:.1f}%")
        print(f"  Profit/heure: {analysis.profit_per_hour:.2f}")
        print(f"  XP/heure: {analysis.xp_per_hour:.0f}")
        print(f"  Score de confiance: {analysis.confidence_score:.2f}")
    
    # Ajout d'un job à la queue
    job_id = optimizer.craft_queue.add_job(bread_recipe, 10, OptimizationGoal.PROFIT)
    print(f"\nJob ajouté: {job_id}")
    
    # État de la queue
    queue_status = optimizer.craft_queue.get_queue_status()
    print(f"Jobs en queue: {queue_status['queued']}")
    print(f"Temps estimé total: {queue_status['estimated_total_time']:.1f}s")
    
    # Démarrage du craft automatique (pour les tests)
    print(f"\nDémarrage du craft automatique...")
    optimizer.start_auto_crafting()
    
    try:
        time.sleep(10)  # Laisser tourner 10 secondes
        
        # État final
        final_status = optimizer.get_optimizer_status()
        print(f"\nÉtat final:")
        print(f"Jobs complétés: {final_status['queue_status']['completed']}")
        
        # Export du rapport
        optimizer.export_crafting_report("crafting_report.json")
        print("Rapport exporté: crafting_report.json")
        
    finally:
        optimizer.stop_auto_crafting()