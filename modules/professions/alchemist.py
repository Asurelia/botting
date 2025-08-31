"""
Module Alchimiste - Gestion complète du métier d'Alchimiste.
Inclut toutes les recettes, potions et calculs de rentabilité du crafting.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .base import BaseProfession, ResourceData, ResourceType, QualityLevel
import math
import random

@dataclass
class Recipe:
    """Données d'une recette d'alchimie"""
    id: str
    name: str
    level_required: int
    base_xp: int
    craft_time: float
    ingredients: Dict[str, int]  # ingredient_id: quantity
    result_item: str
    result_quantity: int
    success_rate: float
    critical_rate: float  # Chance de réussite critique (bonus quantité)
    market_value: int  # Valeur du produit fini

@dataclass
class Ingredient:
    """Ingrédient utilisé dans les recettes"""
    id: str
    name: str
    type: str  # 'harvested', 'bought', 'crafted'
    average_cost: int  # Coût moyen en kamas
    availability: float  # Disponibilité (0-1)

class Alchemist(BaseProfession):
    """Classe Alchimiste avec toutes les recettes et potions"""
    
    def __init__(self):
        super().__init__("Alchimiste", "alchemist")
        self.load_ingredients()
        self.load_resources()  # Ici ce seront les recettes
        self.crafting_patterns = {
            'efficient': self._efficient_crafting,
            'profitable': self._profitable_crafting,
            'leveling': self._leveling_crafting,
            'bulk': self._bulk_crafting
        }
        self.current_pattern = 'efficient'
        self.workshop_level = 1  # Niveau de l'atelier
        self.auto_ingredient_buy = True
        self.ingredient_stock = {}  # Stock actuel d'ingrédients
        
    def load_ingredients(self) -> None:
        """Charge tous les ingrédients utilisés en alchimie"""
        self.ingredients = {
            # Ingrédients de base - récoltés
            'ble': Ingredient('ble', 'Blé', 'harvested', 2, 0.95),
            'orge': Ingredient('orge', 'Orge', 'harvested', 3, 0.93),
            'avoine': Ingredient('avoine', 'Avoine', 'harvested', 5, 0.90),
            'menthe': Ingredient('menthe', 'Menthe', 'harvested', 25, 0.85),
            'orchidee': Ingredient('orchidee', 'Orchidée', 'harvested', 50, 0.75),
            'ginseng': Ingredient('ginseng', 'Ginseng', 'harvested', 80, 0.70),
            'belladone': Ingredient('belladone', 'Belladone', 'harvested', 120, 0.60),
            'mandragore': Ingredient('mandragore', 'Mandragore', 'harvested', 180, 0.55),
            
            # Liquides de base - achetés
            'eau_pure': Ingredient('eau_pure', 'Eau Pure', 'bought', 1, 1.0),
            'huile_olive': Ingredient('huile_olive', 'Huile d\'Olive', 'bought', 8, 0.95),
            'alcool': Ingredient('alcool', 'Alcool', 'bought', 15, 0.90),
            'essence_magique': Ingredient('essence_magique', 'Essence Magique', 'bought', 50, 0.80),
            
            # Poudres et cristaux - craftés ou achetés
            'poudre_os': Ingredient('poudre_os', 'Poudre d\'Os', 'bought', 20, 0.85),
            'cristal_ame': Ingredient('cristal_ame', 'Cristal d\'Âme', 'bought', 100, 0.70),
            'perle_sagesse': Ingredient('perle_sagesse', 'Perle de Sagesse', 'bought', 200, 0.60),
            'sang_dragon': Ingredient('sang_dragon', 'Sang de Dragon', 'bought', 500, 0.40),
            
            # Minerais pour potions
            'fer': Ingredient('fer', 'Fer', 'harvested', 5, 0.92),
            'or': Ingredient('or', 'Or', 'harvested', 45, 0.78),
            'mithril': Ingredient('mithril', 'Mithril', 'harvested', 150, 0.65),
        }
    
    def load_resources(self) -> None:
        """Charge toutes les recettes d'alchimie comme 'ressources'"""
        # Convertir les recettes en ResourceData pour compatibilité avec la classe de base
        recipes_data = self._get_all_recipes()
        
        self.resources = {}
        self.recipes = {}  # Garder les recettes originales
        
        for recipe in recipes_data:
            # Créer une ResourceData équivalente
            resource = ResourceData(
                id=recipe.id,
                name=recipe.name,
                type=ResourceType.POTION,
                level_required=recipe.level_required,
                base_xp=recipe.base_xp,
                base_time=recipe.craft_time,
                quality=self._get_quality_from_level(recipe.level_required),
                market_value=recipe.market_value,
                coordinates=(0, 0),  # Les recettes n'ont pas de coordonnées
                success_rate=recipe.success_rate
            )
            
            self.resources[recipe.id] = resource
            self.recipes[recipe.id] = recipe
    
    def _get_all_recipes(self) -> List[Recipe]:
        """Retourne toutes les recettes d'alchimie"""
        return [
            # Potions de soin niveau 1-20
            Recipe(
                id='potion_soin_mineure', name='Potion de Soin Mineure',
                level_required=1, base_xp=10, craft_time=8.0,
                ingredients={'eau_pure': 2, 'menthe': 1},
                result_item='potion_soin_mineure', result_quantity=1,
                success_rate=0.95, critical_rate=0.05, market_value=25
            ),
            Recipe(
                id='potion_soin_legere', name='Potion de Soin Légère',
                level_required=10, base_xp=18, craft_time=12.0,
                ingredients={'eau_pure': 3, 'menthe': 2, 'ble': 1},
                result_item='potion_soin_legere', result_quantity=1,
                success_rate=0.90, critical_rate=0.08, market_value=45
            ),
            Recipe(
                id='potion_soin_moyenne', name='Potion de Soin Moyenne',
                level_required=20, base_xp=30, craft_time=15.0,
                ingredients={'eau_pure': 4, 'orchidee': 1, 'huile_olive': 1},
                result_item='potion_soin_moyenne', result_quantity=1,
                success_rate=0.85, critical_rate=0.10, market_value=80
            ),
            
            # Potions de mana niveau 5-25
            Recipe(
                id='potion_mana_mineure', name='Potion de Mana Mineure',
                level_required=5, base_xp=12, craft_time=10.0,
                ingredients={'eau_pure': 2, 'orge': 2},
                result_item='potion_mana_mineure', result_quantity=1,
                success_rate=0.92, critical_rate=0.06, market_value=30
            ),
            Recipe(
                id='potion_mana_moyenne', name='Potion de Mana Moyenne',
                level_required=25, base_xp=35, craft_time=18.0,
                ingredients={'essence_magique': 1, 'ginseng': 1, 'cristal_ame': 1},
                result_item='potion_mana_moyenne', result_quantity=1,
                success_rate=0.80, critical_rate=0.12, market_value=150
            ),
            
            # Potions de force niveau 15-40
            Recipe(
                id='potion_force_mineure', name='Potion de Force Mineure',
                level_required=15, base_xp=22, craft_time=14.0,
                ingredients={'fer': 1, 'poudre_os': 1, 'alcool': 1},
                result_item='potion_force_mineure', result_quantity=1,
                success_rate=0.88, critical_rate=0.08, market_value=60
            ),
            Recipe(
                id='potion_force_majeure', name='Potion de Force Majeure',
                level_required=40, base_xp=55, craft_time=25.0,
                ingredients={'or': 1, 'belladone': 2, 'essence_magique': 2},
                result_item='potion_force_majeure', result_quantity=1,
                success_rate=0.75, critical_rate=0.15, market_value=250
            ),
            
            # Potions d'agilité niveau 18-45
            Recipe(
                id='potion_agilite_legere', name='Potion d\'Agilité Légère',
                level_required=18, base_xp=25, craft_time=16.0,
                ingredients={'avoine': 3, 'huile_olive': 2},
                result_item='potion_agilite_legere', result_quantity=1,
                success_rate=0.87, critical_rate=0.09, market_value=70
            ),
            Recipe(
                id='potion_agilite_superieure', name='Potion d\'Agilité Supérieure',
                level_required=45, base_xp=65, craft_time=28.0,
                ingredients={'mandragore': 1, 'cristal_ame': 2, 'alcool': 3},
                result_item='potion_agilite_superieure', result_quantity=1,
                success_rate=0.70, critical_rate=0.18, market_value=320
            ),
            
            # Potions de résistance niveau 30-60
            Recipe(
                id='potion_resistance_feu', name='Potion de Résistance au Feu',
                level_required=30, base_xp=40, craft_time=20.0,
                ingredients={'fer': 2, 'essence_magique': 1, 'poudre_os': 2},
                result_item='potion_resistance_feu', result_quantity=1,
                success_rate=0.82, critical_rate=0.12, market_value=120
            ),
            Recipe(
                id='potion_resistance_eau', name='Potion de Résistance à l\'Eau',
                level_required=32, base_xp=42, craft_time=20.0,
                ingredients={'orchidee': 2, 'eau_pure': 5, 'cristal_ame': 1},
                result_item='potion_resistance_eau', result_quantity=1,
                success_rate=0.82, critical_rate=0.12, market_value=125
            ),
            Recipe(
                id='potion_resistance_universelle', name='Potion de Résistance Universelle',
                level_required=60, base_xp=90, craft_time=35.0,
                ingredients={'mithril': 1, 'perle_sagesse': 1, 'essence_magique': 3},
                result_item='potion_resistance_universelle', result_quantity=1,
                success_rate=0.65, critical_rate=0.20, market_value=500
            ),
            
            # Potions rares et légendaires niveau 50-100
            Recipe(
                id='elixir_vie_eternelle', name='Élixir de Vie Éternelle',
                level_required=70, base_xp=120, craft_time=45.0,
                ingredients={'sang_dragon': 1, 'perle_sagesse': 2, 'mandragore': 3},
                result_item='elixir_vie_eternelle', result_quantity=1,
                success_rate=0.50, critical_rate=0.25, market_value=800
            ),
            Recipe(
                id='potion_transformation', name='Potion de Transformation',
                level_required=80, base_xp=150, craft_time=55.0,
                ingredients={'sang_dragon': 2, 'cristal_ame': 5, 'essence_magique': 10},
                result_item='potion_transformation', result_quantity=1,
                success_rate=0.40, critical_rate=0.30, market_value=1200
            ),
            Recipe(
                id='nectar_divin', name='Nectar Divin',
                level_required=90, base_xp=200, craft_time=75.0,
                ingredients={'sang_dragon': 3, 'perle_sagesse': 5, 'mithril': 2},
                result_item='nectar_divin', result_quantity=1,
                success_rate=0.30, critical_rate=0.35, market_value=2000
            ),
            Recipe(
                id='panacee_universelle', name='Panacée Universelle',
                level_required=100, base_xp=300, craft_time=120.0,
                ingredients={'sang_dragon': 5, 'perle_sagesse': 10, 'mithril': 3, 'essence_magique': 20},
                result_item='panacee_universelle', result_quantity=1,
                success_rate=0.20, critical_rate=0.40, market_value=5000
            ),
            
            # Potions utilitaires niveau 25-50
            Recipe(
                id='potion_rappel', name='Potion de Rappel',
                level_required=25, base_xp=35, craft_time=22.0,
                ingredients={'cristal_ame': 1, 'eau_pure': 8, 'alcool': 2},
                result_item='potion_rappel', result_quantity=1,
                success_rate=0.85, critical_rate=0.10, market_value=100
            ),
            Recipe(
                id='huile_arme', name='Huile d\'Arme',
                level_required=35, base_xp=45, craft_time=18.0,
                ingredients={'huile_olive': 5, 'fer': 2, 'poudre_os': 3},
                result_item='huile_arme', result_quantity=3,
                success_rate=0.88, critical_rate=0.15, market_value=80
            ),
            Recipe(
                id='potion_invisibilite', name='Potion d\'Invisibilité',
                level_required=50, base_xp=75, craft_time=40.0,
                ingredients={'belladone': 3, 'cristal_ame': 3, 'essence_magique': 5},
                result_item='potion_invisibilite', result_quantity=1,
                success_rate=0.68, critical_rate=0.22, market_value=450
            )
        ]
    
    def _get_quality_from_level(self, level: int) -> QualityLevel:
        """Détermine la qualité basée sur le niveau requis"""
        if level <= 20:
            return QualityLevel.COMMON
        elif level <= 40:
            return QualityLevel.UNCOMMON
        elif level <= 60:
            return QualityLevel.RARE
        elif level <= 80:
            return QualityLevel.EPIC
        else:
            return QualityLevel.LEGENDARY
    
    def get_optimal_route(self, level_range: Tuple[int, int] = None) -> List[ResourceData]:
        """Retourne les recettes optimales pour le crafting"""
        available_resources = self.get_available_resources(level_range)
        
        if not available_resources:
            return []
        
        # Calculer la rentabilité de chaque recette
        scored_recipes = []
        for resource in available_resources:
            recipe = self.recipes.get(resource.id)
            if not recipe:
                continue
            
            profitability = self.calculate_recipe_profitability(recipe.id)
            
            # Score composite
            score = (
                profitability.get('xp_per_hour', 0) * 0.3 +
                profitability.get('profit_per_hour', 0) * 0.4 +
                profitability.get('success_rate', 0) * 100 * 0.2 +
                profitability.get('ingredient_availability', 0) * 0.1
            )
            
            scored_recipes.append((resource, score))
        
        # Trier par score décroissant
        scored_recipes.sort(key=lambda x: x[1], reverse=True)
        
        return [r[0] for r in scored_recipes[:10]]
    
    def calculate_profitability(self, resource_id: str, duration: float = 3600) -> Dict[str, float]:
        """Interface commune - délègue au calcul de recette"""
        return self.calculate_recipe_profitability(resource_id, duration)
    
    def calculate_recipe_profitability(self, recipe_id: str, duration: float = 3600) -> Dict[str, float]:
        """Calcule la rentabilité d'une recette d'alchimie"""
        recipe = self.recipes.get(recipe_id)
        if not recipe:
            return {}
        
        # Coût des ingrédients
        ingredient_cost = self._calculate_ingredient_cost(recipe)
        
        # Valeur du produit fini avec chance critique
        expected_quantity = recipe.result_quantity * (1 + recipe.critical_rate * 0.5)  # Critique = +50% quantité
        expected_revenue = recipe.market_value * expected_quantity * recipe.success_rate
        
        # Profit brut par craft
        gross_profit = expected_revenue - ingredient_cost
        
        # Temps total incluant préparation des ingrédients
        ingredient_prep_time = self._calculate_ingredient_prep_time(recipe)
        total_time = recipe.craft_time + ingredient_prep_time
        
        # Calculs par heure
        crafts_per_hour = 3600 / total_time
        successful_crafts = crafts_per_hour * recipe.success_rate
        
        xp_per_hour = successful_crafts * recipe.base_xp
        revenue_per_hour = successful_crafts * expected_revenue
        cost_per_hour = crafts_per_hour * ingredient_cost  # Coût même si échec
        profit_per_hour = revenue_per_hour - cost_per_hour
        
        # Disponibilité des ingrédients
        ingredient_availability = self._calculate_ingredient_availability(recipe)
        
        return {
            'xp_per_hour': round(xp_per_hour, 2),
            'revenue_per_hour': round(revenue_per_hour, 2),
            'cost_per_hour': round(cost_per_hour, 2),
            'profit_per_hour': round(profit_per_hour, 2),
            'crafts_per_hour': round(successful_crafts, 2),
            'ingredient_cost': round(ingredient_cost, 2),
            'expected_revenue': round(expected_revenue, 2),
            'profit_per_craft': round(gross_profit, 2),
            'success_rate': recipe.success_rate,
            'critical_rate': recipe.critical_rate,
            'ingredient_availability': round(ingredient_availability, 2),
            'roi_percent': round((gross_profit / max(ingredient_cost, 1)) * 100, 1)
        }
    
    def _calculate_ingredient_cost(self, recipe: Recipe) -> float:
        """Calcule le coût total des ingrédients"""
        total_cost = 0
        for ingredient_id, quantity in recipe.ingredients.items():
            ingredient = self.ingredients.get(ingredient_id)
            if ingredient:
                # Coût ajusté selon la disponibilité
                adjusted_cost = ingredient.average_cost / max(ingredient.availability, 0.1)
                total_cost += adjusted_cost * quantity
        return total_cost
    
    def _calculate_ingredient_prep_time(self, recipe: Recipe) -> float:
        """Calcule le temps de préparation des ingrédients"""
        prep_time = 0
        for ingredient_id, quantity in recipe.ingredients.items():
            ingredient = self.ingredients.get(ingredient_id)
            if ingredient:
                if ingredient.type == 'harvested':
                    # Temps pour récolter
                    prep_time += quantity * 2.0  # 2 secondes par ingrédient récolté
                elif ingredient.type == 'bought':
                    # Temps pour acheter (ou sortir du stock)
                    prep_time += quantity * 0.5 if self.auto_ingredient_buy else quantity * 1.5
                # Les ingrédients 'crafted' nécessiteraient un calcul récursif
        return prep_time
    
    def _calculate_ingredient_availability(self, recipe: Recipe) -> float:
        """Calcule la disponibilité moyenne des ingrédients"""
        if not recipe.ingredients:
            return 1.0
        
        total_availability = 0
        for ingredient_id in recipe.ingredients.keys():
            ingredient = self.ingredients.get(ingredient_id)
            if ingredient:
                total_availability += ingredient.availability
        
        return total_availability / len(recipe.ingredients)
    
    def get_crafting_pattern(self, recipes: List[str], pattern_name: str = None) -> Dict[str, any]:
        """Applique un pattern de crafting spécifique"""
        pattern_name = pattern_name or self.current_pattern
        pattern_func = self.crafting_patterns.get(pattern_name, self._efficient_crafting)
        
        recipe_objects = [self.recipes[r_id] for r_id in recipes if r_id in self.recipes]
        return pattern_func(recipe_objects)
    
    def _efficient_crafting(self, recipes: List[Recipe]) -> Dict[str, any]:
        """Pattern efficace - maximise XP/temps"""
        # Trier par XP par minute
        sorted_recipes = sorted(recipes, 
                              key=lambda r: r.base_xp / (r.craft_time / 60), 
                              reverse=True)
        
        return {
            'pattern': 'efficient',
            'recipes': [r.id for r in sorted_recipes[:8]],
            'strategy': 'Maximiser l\'expérience par minute',
            'estimated_xp_bonus': 1.3,
            'focus': 'Leveling rapide'
        }
    
    def _profitable_crafting(self, recipes: List[Recipe]) -> Dict[str, any]:
        """Pattern profitable - maximise les bénéfices"""
        # Calculer et trier par profit par heure
        profitable_recipes = []
        for recipe in recipes:
            profit_data = self.calculate_recipe_profitability(recipe.id)
            profit_per_hour = profit_data.get('profit_per_hour', 0)
            if profit_per_hour > 0:
                profitable_recipes.append((recipe, profit_per_hour))
        
        profitable_recipes.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'pattern': 'profitable',
            'recipes': [r[0].id for r in profitable_recipes[:6]],
            'strategy': 'Maximiser les profits',
            'estimated_profit_bonus': 1.5,
            'expected_hourly_profit': sum(r[1] for r in profitable_recipes[:6]),
            'focus': 'Génération de kamas'
        }
    
    def _leveling_crafting(self, recipes: List[Recipe]) -> Dict[str, any]:
        """Pattern leveling - progression optimisée"""
        # Équilibrer XP et coût des ingrédients
        leveling_scores = []
        for recipe in recipes:
            profit_data = self.calculate_recipe_profitability(recipe.id)
            xp_per_hour = profit_data.get('xp_per_hour', 0)
            ingredient_cost = profit_data.get('ingredient_cost', 0)
            
            # Score favorisant XP élevée et coût faible
            if ingredient_cost > 0:
                efficiency_score = xp_per_hour / math.sqrt(ingredient_cost)
            else:
                efficiency_score = xp_per_hour
            
            leveling_scores.append((recipe, efficiency_score))
        
        leveling_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'pattern': 'leveling',
            'recipes': [r[0].id for r in leveling_scores[:10]],
            'strategy': 'Progression équilibrée XP/coût',
            'estimated_cost_efficiency': 1.2,
            'focus': 'Montée en niveau économique'
        }
    
    def _bulk_crafting(self, recipes: List[Recipe]) -> Dict[str, any]:
        """Pattern bulk - production en masse"""
        # Favoriser les recettes avec bonne disponibilité d'ingrédients
        bulk_suitable = []
        for recipe in recipes:
            profit_data = self.calculate_recipe_profitability(recipe.id)
            availability = profit_data.get('ingredient_availability', 0)
            success_rate = recipe.success_rate
            
            # Score pour production en masse
            bulk_score = (availability * 0.6 + success_rate * 0.4) * recipe.base_xp
            bulk_suitable.append((recipe, bulk_score))
        
        bulk_suitable.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'pattern': 'bulk',
            'recipes': [r[0].id for r in bulk_suitable[:5]],
            'strategy': 'Production en grande quantité',
            'estimated_volume_bonus': 1.4,
            'focus': 'Stockage et vente de masse',
            'recommended_batch_size': 50
        }
    
    def estimate_crafting_session(self, recipe_id: str, target_quantity: int = 10, 
                                duration_hours: float = 2.0) -> Dict[str, any]:
        """Estime une session de crafting"""
        recipe = self.recipes.get(recipe_id)
        if not recipe:
            return {}
        
        profitability = self.calculate_recipe_profitability(recipe_id)
        
        # Limitation par la durée
        max_crafts_by_time = duration_hours * profitability.get('crafts_per_hour', 0)
        
        # Limitation par les ingrédients disponibles
        max_crafts_by_ingredients = self._calculate_max_crafts_by_ingredients(recipe)
        
        # Prendre la limitation la plus restrictive
        max_possible_crafts = min(target_quantity, max_crafts_by_time, max_crafts_by_ingredients)
        
        # Calculs de la session
        total_time = max_possible_crafts / profitability.get('crafts_per_hour', 1)
        expected_successes = max_possible_crafts * recipe.success_rate
        expected_criticals = expected_successes * recipe.critical_rate
        
        total_xp = expected_successes * recipe.base_xp
        total_cost = max_possible_crafts * profitability.get('ingredient_cost', 0)
        total_revenue = expected_successes * recipe.market_value * (1 + recipe.critical_rate * 0.5)
        total_profit = total_revenue - total_cost
        
        return {
            'recipe_name': recipe.name,
            'target_quantity': target_quantity,
            'achievable_quantity': round(max_possible_crafts),
            'expected_successes': round(expected_successes),
            'expected_criticals': round(expected_criticals),
            'session_time_hours': round(total_time, 2),
            'total_xp_gain': round(total_xp),
            'total_cost': round(total_cost),
            'total_revenue': round(total_revenue),
            'total_profit': round(total_profit),
            'limiting_factor': 'temps' if max_crafts_by_time < max_crafts_by_ingredients else 'ingrédients',
            'profitability_rating': 'Excellent' if total_profit > total_cost * 0.3 else 'Bon' if total_profit > 0 else 'Déficitaire'
        }
    
    def _calculate_max_crafts_by_ingredients(self, recipe: Recipe) -> int:
        """Calcule le nombre maximum de crafts basé sur les ingrédients disponibles"""
        if not self.auto_ingredient_buy:
            # Si pas d'achat auto, limité par le stock actuel
            max_crafts = float('inf')
            for ingredient_id, quantity_needed in recipe.ingredients.items():
                current_stock = self.ingredient_stock.get(ingredient_id, 0)
                max_crafts_for_ingredient = current_stock // quantity_needed
                max_crafts = min(max_crafts, max_crafts_for_ingredient)
            return int(max_crafts) if max_crafts != float('inf') else 0
        else:
            # Si achat auto, limité par la disponibilité sur le marché
            # Simplification: on assume une disponibilité de 1000 unités par ingrédient
            return 1000
    
    def manage_ingredient_stock(self, recipe_ids: List[str], duration_hours: float = 4.0) -> Dict[str, any]:
        """Gère le stock d'ingrédients pour un ensemble de recettes"""
        ingredient_needs = {}
        
        # Calculer les besoins totaux
        for recipe_id in recipe_ids:
            recipe = self.recipes.get(recipe_id)
            if not recipe:
                continue
            
            profitability = self.calculate_recipe_profitability(recipe_id)
            crafts_planned = duration_hours * profitability.get('crafts_per_hour', 0)
            
            for ingredient_id, quantity_per_craft in recipe.ingredients.items():
                total_needed = crafts_planned * quantity_per_craft
                if ingredient_id in ingredient_needs:
                    ingredient_needs[ingredient_id] += total_needed
                else:
                    ingredient_needs[ingredient_id] = total_needed
        
        # Calculer ce qu'il faut acheter/récolter
        shopping_list = {}
        harvesting_list = {}
        total_cost = 0
        
        for ingredient_id, needed_quantity in ingredient_needs.items():
            ingredient = self.ingredients.get(ingredient_id)
            current_stock = self.ingredient_stock.get(ingredient_id, 0)
            
            if current_stock < needed_quantity:
                to_acquire = needed_quantity - current_stock
                cost = to_acquire * ingredient.average_cost
                total_cost += cost
                
                if ingredient.type == 'bought':
                    shopping_list[ingredient_id] = {
                        'name': ingredient.name,
                        'quantity': round(to_acquire),
                        'unit_cost': ingredient.average_cost,
                        'total_cost': round(cost),
                        'availability': ingredient.availability
                    }
                elif ingredient.type == 'harvested':
                    harvesting_list[ingredient_id] = {
                        'name': ingredient.name,
                        'quantity': round(to_acquire),
                        'estimated_time_hours': round(to_acquire * 0.05, 2)  # 3 min par ingrédient
                    }
        
        return {
            'total_recipes': len(recipe_ids),
            'planning_duration_hours': duration_hours,
            'ingredient_needs': {k: round(v) for k, v in ingredient_needs.items()},
            'shopping_list': shopping_list,
            'harvesting_list': harvesting_list,
            'total_shopping_cost': round(total_cost),
            'estimated_prep_time_hours': round(len(harvesting_list) * 0.5, 2),
            'ready_to_craft': len(shopping_list) == 0 and len(harvesting_list) == 0
        }
    
    def get_workshop_upgrades(self) -> Dict[str, any]:
        """Recommandations d'amélioration de l'atelier"""
        upgrade_benefits = {
            2: {'craft_speed': 1.1, 'critical_rate': 1.2, 'cost': 10000},
            3: {'craft_speed': 1.2, 'critical_rate': 1.4, 'cost': 25000},
            4: {'craft_speed': 1.3, 'critical_rate': 1.6, 'cost': 50000},
            5: {'craft_speed': 1.4, 'critical_rate': 1.8, 'cost': 100000}
        }
        
        if self.workshop_level >= 5:
            return {'message': 'Atelier déjà au niveau maximum'}
        
        next_level = self.workshop_level + 1
        benefits = upgrade_benefits.get(next_level, {})
        
        # Calculer le ROI de l'amélioration
        current_hourly_profit = 0
        for recipe in list(self.recipes.values())[:10]:  # Top 10 recipes
            if self.can_harvest(recipe.id):
                profit = self.calculate_recipe_profitability(recipe.id)
                current_hourly_profit += profit.get('profit_per_hour', 0) * 0.1  # 10% du temps sur chaque
        
        improved_profit = current_hourly_profit * benefits.get('craft_speed', 1.0)
        daily_improvement = (improved_profit - current_hourly_profit) * 8  # 8h/jour
        payback_days = benefits.get('cost', 0) / max(daily_improvement, 1)
        
        return {
            'current_level': self.workshop_level,
            'next_level': next_level,
            'upgrade_cost': benefits.get('cost', 0),
            'benefits': {
                'craft_speed_multiplier': benefits.get('craft_speed', 1.0),
                'critical_rate_multiplier': benefits.get('critical_rate', 1.0)
            },
            'roi_analysis': {
                'current_hourly_profit': round(current_hourly_profit, 2),
                'improved_hourly_profit': round(improved_profit, 2),
                'daily_improvement': round(daily_improvement, 2),
                'payback_period_days': round(payback_days, 1)
            },
            'recommendation': 'Recommandé' if payback_days < 30 else 'Considérer' if payback_days < 90 else 'Non prioritaire'
        }
    
    def __str__(self) -> str:
        available_recipes = len(self.get_available_resources())
        workshop_efficiency = 1.0 + (self.workshop_level - 1) * 0.1
        return f"Alchimiste (Niveau {self.stats.level}) - {available_recipes} recettes disponibles - Atelier niveau {self.workshop_level} ({workshop_efficiency:.1f}x)"