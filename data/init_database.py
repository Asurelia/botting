"""
Module d'initialisation des données DOFUS
========================================

Ce module contient toutes les données de base du jeu DOFUS nécessaires
pour le fonctionnement du bot. Il inclut les sorts, objets, cartes,
monstres et autres données de référence.

Créé le: 2025-08-31
Version: 1.0.0
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DofusDataInitializer:
    """Gestionnaire d'initialisation des données DOFUS"""
    
    def __init__(self, db_path: str = "data/databases/dofus_bot.db"):
        """
        Initialise le gestionnaire de données
        
        Args:
            db_path: Chemin vers la base de données
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Base de données non trouvée: {db_path}")
    
    def connect(self) -> sqlite3.Connection:
        """Crée une connexion à la base de données"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def initialize_all_data(self) -> bool:
        """
        Initialise toutes les données de référence
        
        Returns:
            True si l'initialisation réussit, False sinon
        """
        try:
            logger.info("Début de l'initialisation des données DOFUS...")
            
            with self.connect() as conn:
                # Initialiser dans l'ordre des dépendances
                self._init_classes_data(conn)
                self._init_spells_data(conn)
                self._init_items_data(conn)
                self._init_monsters_data(conn)
                self._init_maps_data(conn)
                self._init_resources_data(conn)
                self._init_recipes_data(conn)
                self._init_market_categories(conn)
                self._init_combat_strategies(conn)
                
                conn.commit()
            
            logger.info("Initialisation des données terminée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            return False
    
    def _init_classes_data(self, conn: sqlite3.Connection):
        """Initialise les données des classes"""
        logger.info("Initialisation des classes...")
        
        # Les 18 classes de DOFUS
        classes_data = [
            {"id": 1, "name": "Feca", "element": "Terre", "role": "Tank"},
            {"id": 2, "name": "Osamodas", "element": "Neutre", "role": "Invocateur"},
            {"id": 3, "name": "Enutrof", "element": "Terre", "role": "Support"},
            {"id": 4, "name": "Sram", "element": "Eau", "role": "DPS"},
            {"id": 5, "name": "Xelor", "element": "Air", "role": "Support"},
            {"id": 6, "name": "Ecaflip", "element": "Feu", "role": "DPS"},
            {"id": 7, "name": "Eniripsa", "element": "Eau", "role": "Soigneur"},
            {"id": 8, "name": "Iop", "element": "Feu", "role": "DPS"},
            {"id": 9, "name": "Cra", "element": "Air", "role": "DPS"},
            {"id": 10, "name": "Sadida", "element": "Terre", "role": "Support"},
            {"id": 11, "name": "Sacrieur", "element": "Feu", "role": "Tank"},
            {"id": 12, "name": "Pandawa", "element": "Eau", "role": "Tank"},
            {"id": 13, "name": "Roublard", "element": "Air", "role": "DPS"},
            {"id": 14, "name": "Zobal", "element": "Neutre", "role": "Tank"},
            {"id": 15, "name": "Steamer", "element": "Feu", "role": "DPS"},
            {"id": 16, "name": "Eliotrope", "element": "Neutre", "role": "DPS"},
            {"id": 17, "name": "Huppermage", "element": "Neutre", "role": "DPS"},
            {"id": 18, "name": "Ouginak", "element": "Terre", "role": "DPS"}
        ]
        
        # Stocker dans la table de configuration pour référence
        for class_data in classes_data:
            conn.execute("""
                INSERT OR REPLACE INTO config (key, value, category, description)
                VALUES (?, ?, 'classes', ?)
            """, (
                f"class_{class_data['id']}",
                json.dumps(class_data),
                f"Données de la classe {class_data['name']}"
            ))
    
    def _init_spells_data(self, conn: sqlite3.Connection):
        """Initialise les sorts de base des classes principales"""
        logger.info("Initialisation des sorts...")
        
        spells_data = [
            # Sorts Iop
            {
                "id": 1, "name": "Compulsion", "class": "Iop", "element": "Feu",
                "level_required": 1, "ap_cost": 3, "range_min": 1, "range_max": 1,
                "cast_per_turn": 2, "line_of_sight": False, "cooldown": 0,
                "effects": json.dumps({
                    "1": {"damage": "15-19", "critical": "19-23"},
                    "6": {"damage": "40-44", "critical": "50-54"}
                })
            },
            {
                "id": 2, "name": "Epée du Jugement", "class": "Iop", "element": "Air",
                "level_required": 3, "ap_cost": 4, "range_min": 1, "range_max": 6,
                "cast_per_turn": 1, "line_of_sight": True, "cooldown": 0,
                "effects": json.dumps({
                    "1": {"damage": "18-22", "critical": "23-27"},
                    "6": {"damage": "45-49", "critical": "56-60"}
                })
            },
            {
                "id": 3, "name": "Pression", "class": "Iop", "element": "Eau",
                "level_required": 6, "ap_cost": 3, "range_min": 1, "range_max": 3,
                "cast_per_turn": 2, "line_of_sight": False, "cooldown": 0,
                "effects": json.dumps({
                    "1": {"damage": "12-16", "critical": "15-19"},
                    "6": {"damage": "35-39", "critical": "44-48"}
                })
            },
            
            # Sorts Cra
            {
                "id": 10, "name": "Flèche Magique", "class": "Cra", "element": "Air",
                "level_required": 1, "ap_cost": 3, "range_min": 2, "range_max": 8,
                "cast_per_turn": -1, "line_of_sight": True, "cooldown": 0,
                "effects": json.dumps({
                    "1": {"damage": "8-12", "critical": "10-15"},
                    "6": {"damage": "28-32", "critical": "35-40"}
                })
            },
            {
                "id": 11, "name": "Flèche d'Expiation", "class": "Cra", "element": "Eau",
                "level_required": 3, "ap_cost": 3, "range_min": 1, "range_max": 6,
                "cast_per_turn": 2, "line_of_sight": True, "cooldown": 0,
                "effects": json.dumps({
                    "1": {"damage": "10-14", "critical": "13-17"},
                    "6": {"damage": "30-34", "critical": "38-42"}
                })
            },
            {
                "id": 12, "name": "Flèche Punitive", "class": "Cra", "element": "Feu",
                "level_required": 6, "ap_cost": 4, "range_min": 1, "range_max": 8,
                "cast_per_turn": 1, "line_of_sight": True, "cooldown": 0,
                "effects": json.dumps({
                    "1": {"damage": "15-19", "critical": "19-24"},
                    "6": {"damage": "40-44", "critical": "50-55"}
                })
            },
            
            # Sorts Eniripsa
            {
                "id": 20, "name": "Mot de Jouvence", "class": "Eniripsa", "element": "Eau",
                "level_required": 1, "ap_cost": 2, "range_min": 1, "range_max": 6,
                "cast_per_turn": -1, "line_of_sight": True, "cooldown": 0,
                "effects": json.dumps({
                    "1": {"heal": "8-12", "critical": "10-15"},
                    "6": {"heal": "28-32", "critical": "35-40"}
                })
            },
            {
                "id": 21, "name": "Mot Blessant", "class": "Eniripsa", "element": "Air",
                "level_required": 3, "ap_cost": 3, "range_min": 1, "range_max": 6,
                "cast_per_turn": 2, "line_of_sight": True, "cooldown": 0,
                "effects": json.dumps({
                    "1": {"damage": "10-14", "critical": "13-17"},
                    "6": {"damage": "30-34", "critical": "38-42"}
                })
            },
            
            # Sorts de mouvement communs
            {
                "id": 100, "name": "Poussée", "class": "Commun", "element": "Neutre",
                "level_required": 1, "ap_cost": 2, "range_min": 1, "range_max": 1,
                "cast_per_turn": 1, "line_of_sight": False, "cooldown": 0,
                "effects": json.dumps({
                    "1": {"push": 2, "damage": "1-2"},
                    "6": {"push": 2, "damage": "1-2"}
                })
            }
        ]
        
        for spell in spells_data:
            conn.execute("""
                INSERT OR REPLACE INTO spells 
                (id, name, class, element, level_required, ap_cost, range_min, 
                 range_max, cast_per_turn, cast_per_target, line_of_sight, 
                 cooldown, effects)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """, (
                spell["id"], spell["name"], spell["class"], spell["element"],
                spell["level_required"], spell["ap_cost"], spell["range_min"],
                spell["range_max"], spell["cast_per_turn"], spell["line_of_sight"],
                spell["cooldown"], spell["effects"]
            ))
    
    def _init_items_data(self, conn: sqlite3.Connection):
        """Initialise les objets de base"""
        logger.info("Initialisation des objets...")
        
        items_data = [
            # Armes de base
            {
                "id": 1, "name": "Épée de Boisaille", "type": "épée", "level": 1,
                "description": "Une épée en bois pour débuter",
                "effects": json.dumps({"dommages": "1-5", "PA": 4, "CC": 5}),
                "weight": 2, "rarity": "common"
            },
            {
                "id": 2, "name": "Arc de Boisaille", "type": "arc", "level": 1,
                "description": "Un arc en bois pour débuter",
                "effects": json.dumps({"dommages": "1-4", "PA": 3, "portée": "2-7"}),
                "weight": 1, "rarity": "common"
            },
            {
                "id": 3, "name": "Baguette du Novice", "type": "baguette", "level": 1,
                "description": "Une baguette magique pour débuter",
                "effects": json.dumps({"dommages": "1-3", "PA": 3, "sorts": "+1"}),
                "weight": 1, "rarity": "common"
            },
            
            # Équipements
            {
                "id": 10, "name": "Casque du Bouftou", "type": "casque", "level": 10,
                "description": "Casque obtenu en combattant les Bouftous",
                "effects": json.dumps({"vitalité": 20, "sagesse": 5}),
                "weight": 2, "rarity": "uncommon"
            },
            {
                "id": 11, "name": "Cape du Bouftou", "type": "cape", "level": 12,
                "description": "Cape en poil de Bouftou",
                "effects": json.dumps({"vitalité": 15, "agilité": 5}),
                "weight": 1, "rarity": "uncommon"
            },
            
            # Ressources de base
            {
                "id": 50, "name": "Blé", "type": "céréale", "level": 1,
                "description": "Céréale de base pour l'alchimie",
                "weight": 1, "stackable": True, "max_stack": 100,
                "market_category": "ressources", "rarity": "common"
            },
            {
                "id": 51, "name": "Avoine", "type": "céréale", "level": 10,
                "description": "Céréale plus résistante",
                "weight": 1, "stackable": True, "max_stack": 100,
                "market_category": "ressources", "rarity": "common"
            },
            {
                "id": 60, "name": "Frêne", "type": "bois", "level": 1,
                "description": "Bois de base pour la menuiserie",
                "weight": 2, "stackable": True, "max_stack": 100,
                "market_category": "ressources", "rarity": "common"
            },
            {
                "id": 70, "name": "Fer", "type": "minerai", "level": 1,
                "description": "Minerai de base pour la forge",
                "weight": 3, "stackable": True, "max_stack": 100,
                "market_category": "ressources", "rarity": "common"
            },
            
            # Consommables
            {
                "id": 100, "name": "Pain", "type": "nourriture", "level": 1,
                "description": "Restaure des points de vie",
                "effects": json.dumps({"heal": "20-30"}),
                "weight": 1, "stackable": True, "max_stack": 100,
                "rarity": "common"
            },
            {
                "id": 101, "name": "Potion de Soin Mineure", "type": "potion", "level": 1,
                "description": "Restaure 50 points de vie",
                "effects": json.dumps({"heal": 50}),
                "weight": 1, "stackable": True, "max_stack": 10,
                "rarity": "common"
            }
        ]
        
        for item in items_data:
            conn.execute("""
                INSERT OR REPLACE INTO items 
                (id, name, type, level, description, effects, weight, 
                 stackable, max_stack, market_category, rarity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item["id"], item["name"], item["type"], item.get("level", 1),
                item.get("description", ""), item.get("effects", "{}"),
                item.get("weight", 1), item.get("stackable", False),
                item.get("max_stack", 1), item.get("market_category", ""),
                item.get("rarity", "common")
            ))
    
    def _init_monsters_data(self, conn: sqlite3.Connection):
        """Initialise les données des monstres"""
        logger.info("Initialisation des monstres...")
        
        monsters_data = [
            # Monstres niveau 1-20
            {
                "id": 1, "name": "Larve Bleue", "level": 1, "race": "Larve",
                "element": "Eau", "health": 25, "ap": 6, "mp": 3,
                "resistances": json.dumps({"neutre": 0, "terre": 0, "feu": 0, "eau": 10, "air": 0}),
                "spells": json.dumps([{"name": "Piqûre", "damage": "1-3"}]),
                "drops": json.dumps([{"id": 200, "name": "Soie de Larve Bleue", "chance": 0.3}]),
                "experience": 8, "kamas_min": 1, "kamas_max": 3
            },
            {
                "id": 2, "name": "Bouftou", "level": 10, "race": "Bouftou",
                "element": "Terre", "health": 80, "ap": 6, "mp": 3,
                "resistances": json.dumps({"neutre": 5, "terre": 15, "feu": 0, "eau": 0, "air": 0}),
                "spells": json.dumps([{"name": "Charge", "damage": "8-12"}]),
                "drops": json.dumps([
                    {"id": 10, "name": "Casque du Bouftou", "chance": 0.02},
                    {"id": 11, "name": "Cape du Bouftou", "chance": 0.02}
                ]),
                "experience": 45, "kamas_min": 5, "kamas_max": 15
            },
            {
                "id": 3, "name": "Tofu", "level": 2, "race": "Tofu",
                "element": "Air", "health": 30, "ap": 6, "mp": 4,
                "resistances": json.dumps({"neutre": 0, "terre": 0, "feu": 0, "eau": 0, "air": 10}),
                "spells": json.dumps([{"name": "Coup de Bec", "damage": "2-4"}]),
                "drops": json.dumps([{"id": 201, "name": "Plume de Tofu", "chance": 0.4}]),
                "experience": 12, "kamas_min": 1, "kamas_max": 4
            },
            {
                "id": 4, "name": "Prespic", "level": 5, "race": "Prespic",
                "element": "Feu", "health": 50, "ap": 6, "mp": 3,
                "resistances": json.dumps({"neutre": 0, "terre": 0, "feu": 12, "eau": 0, "air": 0}),
                "spells": json.dumps([{"name": "Dard Enflammé", "damage": "4-8"}]),
                "drops": json.dumps([{"id": 202, "name": "Dard de Prespic", "chance": 0.25}]),
                "experience": 28, "kamas_min": 3, "kamas_max": 8
            },
            
            # Boss de donjon niveau moyen
            {
                "id": 10, "name": "Maître Bouftou", "level": 15, "race": "Bouftou",
                "element": "Terre", "health": 200, "ap": 8, "mp": 4,
                "resistances": json.dumps({"neutre": 10, "terre": 25, "feu": 5, "eau": 5, "air": 5}),
                "spells": json.dumps([
                    {"name": "Charge Puissante", "damage": "15-25"},
                    {"name": "Coup de Tête", "damage": "20-30"}
                ]),
                "drops": json.dumps([
                    {"id": 300, "name": "Épée du Maître Bouftou", "chance": 0.1},
                    {"id": 301, "name": "Amulette du Bouftou", "chance": 0.05}
                ]),
                "experience": 120, "kamas_min": 20, "kamas_max": 50
            }
        ]
        
        for monster in monsters_data:
            conn.execute("""
                INSERT OR REPLACE INTO monsters 
                (id, name, level, race, element, health, ap, mp, 
                 resistances, spells, drops, experience, kamas_min, kamas_max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                monster["id"], monster["name"], monster["level"], monster["race"],
                monster["element"], monster["health"], monster["ap"], monster["mp"],
                monster["resistances"], monster["spells"], monster["drops"],
                monster["experience"], monster["kamas_min"], monster["kamas_max"]
            ))
    
    def _init_maps_data(self, conn: sqlite3.Connection):
        """Initialise les cartes principales"""
        logger.info("Initialisation des cartes...")
        
        maps_data = [
            # Zone de départ - Incarnam
            {
                "id": 1, "x": 0, "y": 0, "area_id": 1, "sub_area_id": 1,
                "name": "Centre d'Incarnam", "indoor": False, "has_zaap": True,
                "monsters": json.dumps([]),
                "resources": json.dumps([]),
                "npcs": json.dumps([{"name": "Maître Yokai", "type": "guide"}])
            },
            {
                "id": 2, "x": 1, "y": 0, "area_id": 1, "sub_area_id": 1,
                "name": "Plaines d'Incarnam", "indoor": False, "has_zaap": False,
                "monsters": json.dumps([{"id": 1, "name": "Larve Bleue", "group_size": "1-3"}]),
                "resources": json.dumps([]),
                "npcs": json.dumps([])
            },
            
            # Astrub
            {
                "id": 10, "x": -1, "y": -1, "area_id": 2, "sub_area_id": 2,
                "name": "Centre d'Astrub", "indoor": False, "has_zaap": True,
                "monsters": json.dumps([]),
                "resources": json.dumps([]),
                "npcs": json.dumps([
                    {"name": "Forgeron d'Astrub", "type": "forgeron"},
                    {"name": "Zaap de Transport", "type": "zaap"}
                ])
            },
            {
                "id": 11, "x": 0, "y": -1, "area_id": 2, "sub_area_id": 3,
                "name": "Forêt d'Astrub", "indoor": False, "has_zaap": False,
                "monsters": json.dumps([
                    {"id": 3, "name": "Tofu", "group_size": "1-4"},
                    {"id": 2, "name": "Bouftou", "group_size": "1-2"}
                ]),
                "resources": json.dumps([
                    {"id": 60, "name": "Frêne", "respawn": 180}
                ]),
                "npcs": json.dumps([])
            },
            {
                "id": 12, "x": 1, "y": -1, "area_id": 2, "sub_area_id": 4,
                "name": "Champs d'Astrub", "indoor": False, "has_zaap": False,
                "monsters": json.dumps([
                    {"id": 4, "name": "Prespic", "group_size": "1-3"}
                ]),
                "resources": json.dumps([
                    {"id": 50, "name": "Blé", "respawn": 120},
                    {"id": 51, "name": "Avoine", "respawn": 240}
                ]),
                "npcs": json.dumps([])
            },
            
            # Mines d'Astrub
            {
                "id": 20, "x": -2, "y": -1, "area_id": 2, "sub_area_id": 5,
                "name": "Mines d'Astrub", "indoor": True, "has_zaap": False,
                "monsters": json.dumps([]),
                "resources": json.dumps([
                    {"id": 70, "name": "Fer", "respawn": 300}
                ]),
                "npcs": json.dumps([
                    {"name": "Mineur Astrubien", "type": "pnj"}
                ])
            },
            
            # Donjon du Bouftou
            {
                "id": 30, "x": 2, "y": -1, "area_id": 2, "sub_area_id": 6,
                "name": "Donjon du Bouftou - Entrée", "indoor": True, "has_zaap": False,
                "monsters": json.dumps([
                    {"id": 2, "name": "Bouftou", "group_size": "2-4"}
                ]),
                "resources": json.dumps([]),
                "npcs": json.dumps([])
            },
            {
                "id": 31, "x": 2, "y": -1, "area_id": 2, "sub_area_id": 6,
                "name": "Donjon du Bouftou - Salle du Boss", "indoor": True, "has_zaap": False,
                "monsters": json.dumps([
                    {"id": 10, "name": "Maître Bouftou", "group_size": "1", "boss": True}
                ]),
                "resources": json.dumps([]),
                "npcs": json.dumps([])
            }
        ]
        
        for map_data in maps_data:
            conn.execute("""
                INSERT OR REPLACE INTO maps 
                (id, x, y, area_id, sub_area_id, name, indoor, has_zaap,
                 monsters, resources, npcs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                map_data["id"], map_data["x"], map_data["y"],
                map_data["area_id"], map_data["sub_area_id"],
                map_data["name"], map_data["indoor"], map_data["has_zaap"],
                map_data["monsters"], map_data["resources"], map_data["npcs"]
            ))
    
    def _init_resources_data(self, conn: sqlite3.Connection):
        """Initialise les ressources et professions"""
        logger.info("Initialisation des ressources...")
        
        resources_data = [
            # Ressources de Paysan
            {
                "id": 50, "name": "Blé", "type": "cereals", "profession": "farmer",
                "level_required": 1, "respawn_time": 120,
                "locations": json.dumps([{"map_id": 12, "cells": [123, 145, 167]}]),
                "market_value": 2
            },
            {
                "id": 51, "name": "Avoine", "type": "cereals", "profession": "farmer",
                "level_required": 10, "respawn_time": 240,
                "locations": json.dumps([{"map_id": 12, "cells": [200, 223]}]),
                "market_value": 5
            },
            {
                "id": 52, "name": "Orge", "type": "cereals", "profession": "farmer",
                "level_required": 20, "respawn_time": 300,
                "locations": json.dumps([{"map_id": 25, "cells": [89, 112]}]),
                "market_value": 8
            },
            
            # Ressources de Bûcheron
            {
                "id": 60, "name": "Frêne", "type": "wood", "profession": "lumberjack",
                "level_required": 1, "respawn_time": 180,
                "locations": json.dumps([{"map_id": 11, "cells": [156, 178, 201]}]),
                "market_value": 3
            },
            {
                "id": 61, "name": "Chêne", "type": "wood", "profession": "lumberjack",
                "level_required": 10, "respawn_time": 300,
                "locations": json.dumps([{"map_id": 15, "cells": [134, 189]}]),
                "market_value": 6
            },
            
            # Ressources de Mineur
            {
                "id": 70, "name": "Fer", "type": "ore", "profession": "miner",
                "level_required": 1, "respawn_time": 300,
                "locations": json.dumps([{"map_id": 20, "cells": [45, 67, 89, 112]}]),
                "market_value": 4
            },
            {
                "id": 71, "name": "Cuivre", "type": "ore", "profession": "miner",
                "level_required": 10, "respawn_time": 360,
                "locations": json.dumps([{"map_id": 20, "cells": [134, 156]}]),
                "market_value": 7
            },
            
            # Ressources d'Alchimiste (drops de monstres)
            {
                "id": 200, "name": "Soie de Larve Bleue", "type": "monster_drop", "profession": "alchemist",
                "level_required": 1, "respawn_time": 0,
                "locations": json.dumps([{"monster_id": 1, "chance": 0.3}]),
                "market_value": 10
            },
            {
                "id": 201, "name": "Plume de Tofu", "type": "monster_drop", "profession": "alchemist",
                "level_required": 1, "respawn_time": 0,
                "locations": json.dumps([{"monster_id": 3, "chance": 0.4}]),
                "market_value": 8
            }
        ]
        
        for resource in resources_data:
            conn.execute("""
                INSERT OR REPLACE INTO resources 
                (id, name, type, profession, level_required, respawn_time, 
                 locations, market_value, last_price_update)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                resource["id"], resource["name"], resource["type"],
                resource["profession"], resource["level_required"],
                resource["respawn_time"], resource["locations"], resource["market_value"]
            ))
    
    def _init_recipes_data(self, conn: sqlite3.Connection):
        """Initialise les recettes de craft"""
        logger.info("Initialisation des recettes...")
        
        recipes_data = [
            # Recettes Boulanger
            {
                "item_id": 100, "profession": "baker", "level_required": 1,
                "ingredients": json.dumps([{"id": 50, "name": "Blé", "quantity": 1}]),
                "crafting_time": 5, "experience_gained": 5
            },
            {
                "item_id": 110, "profession": "baker", "level_required": 10,
                "ingredients": json.dumps([
                    {"id": 51, "name": "Avoine", "quantity": 1},
                    {"id": 50, "name": "Blé", "quantity": 1}
                ]),
                "crafting_time": 10, "experience_gained": 15
            },
            
            # Recettes Forgeron
            {
                "item_id": 1, "profession": "smith", "level_required": 1,
                "ingredients": json.dumps([
                    {"id": 70, "name": "Fer", "quantity": 2},
                    {"id": 60, "name": "Frêne", "quantity": 1}
                ]),
                "crafting_time": 30, "experience_gained": 20
            },
            
            # Recettes Alchimiste
            {
                "item_id": 101, "profession": "alchemist", "level_required": 1,
                "ingredients": json.dumps([
                    {"id": 200, "name": "Soie de Larve Bleue", "quantity": 1},
                    {"id": 50, "name": "Blé", "quantity": 2}
                ]),
                "crafting_time": 15, "experience_gained": 10
            }
        ]
        
        for recipe in recipes_data:
            conn.execute("""
                INSERT OR REPLACE INTO recipes 
                (item_id, profession, level_required, ingredients, 
                 crafting_time, experience_gained, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                recipe["item_id"], recipe["profession"], recipe["level_required"],
                recipe["ingredients"], recipe["crafting_time"], recipe["experience_gained"]
            ))
    
    def _init_market_categories(self, conn: sqlite3.Connection):
        """Initialise les catégories du marché"""
        logger.info("Initialisation des catégories marché...")
        
        categories = [
            {"key": "market_category_equipment", "value": "Équipements"},
            {"key": "market_category_weapons", "value": "Armes"},
            {"key": "market_category_resources", "value": "Ressources"},
            {"key": "market_category_consumables", "value": "Consommables"},
            {"key": "market_category_quest_items", "value": "Objets de Quête"},
            {"key": "market_category_runes", "value": "Runes"}
        ]
        
        for category in categories:
            conn.execute("""
                INSERT OR REPLACE INTO config (key, value, category, description)
                VALUES (?, ?, 'market', 'Catégorie du marché')
            """, (category["key"], category["value"]))
    
    def _init_combat_strategies(self, conn: sqlite3.Connection):
        """Initialise les stratégies de combat de base"""
        logger.info("Initialisation des stratégies de combat...")
        
        strategies = [
            {
                "name": "Iop Aggressif", "character_class": "Iop", "priority": 100,
                "conditions": json.dumps({
                    "enemy_count": {"min": 1, "max": 3},
                    "health_percent": {"min": 50}
                }),
                "actions": json.dumps([
                    {"type": "move", "target": "closest_enemy"},
                    {"type": "spell", "spell_id": 1, "target": "closest_enemy"},
                    {"type": "spell", "spell_id": 2, "target": "weakest_enemy"}
                ])
            },
            {
                "name": "Cra Distance", "character_class": "Cra", "priority": 100,
                "conditions": json.dumps({
                    "enemy_count": {"min": 1, "max": 4}
                }),
                "actions": json.dumps([
                    {"type": "move", "target": "safe_distance", "distance": 6},
                    {"type": "spell", "spell_id": 10, "target": "priority_enemy"},
                    {"type": "spell", "spell_id": 11, "target": "weakest_enemy"}
                ])
            },
            {
                "name": "Eniripsa Support", "character_class": "Eniripsa", "priority": 90,
                "conditions": json.dumps({
                    "ally_health": {"min": 1, "threshold": 60}
                }),
                "actions": json.dumps([
                    {"type": "spell", "spell_id": 20, "target": "lowest_health_ally"},
                    {"type": "spell", "spell_id": 21, "target": "closest_enemy"}
                ])
            },
            {
                "name": "Fuite d'Urgence", "character_class": "Tous", "priority": 200,
                "conditions": json.dumps({
                    "health_percent": {"max": 20}
                }),
                "actions": json.dumps([
                    {"type": "move", "target": "escape_route"},
                    {"type": "flee"}
                ])
            }
        ]
        
        for strategy in strategies:
            conn.execute("""
                INSERT OR REPLACE INTO combat_strategies 
                (name, character_class, priority, conditions, actions, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                strategy["name"], strategy["character_class"], strategy["priority"],
                strategy["conditions"], strategy["actions"]
            ))
    
    def get_initialization_stats(self) -> Dict[str, int]:
        """
        Retourne les statistiques d'initialisation
        
        Returns:
            Dictionnaire avec le nombre d'enregistrements par table
        """
        stats = {}
        
        try:
            with self.connect() as conn:
                tables = [
                    "spells", "items", "monsters", "maps", "resources", 
                    "recipes", "combat_strategies"
                ]
                
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cursor.fetchone()[0]
                    
                # Compter les configurations
                cursor = conn.execute("SELECT COUNT(*) FROM config")
                stats["config_entries"] = cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Erreur lors du calcul des stats: {e}")
            stats["error"] = str(e)
        
        return stats


# Fonction principale d'initialisation
def initialize_dofus_data(db_path: str = "data/databases/dofus_bot.db") -> bool:
    """
    Initialise toutes les données DOFUS dans la base de données
    
    Args:
        db_path: Chemin vers la base de données
        
    Returns:
        True si l'initialisation réussit, False sinon
    """
    try:
        # Vérifier que la base de données existe
        if not Path(db_path).exists():
            logger.error(f"Base de données non trouvée: {db_path}")
            logger.info("Exécutez d'abord config/database_setup.py")
            return False
        
        # Initialiser les données
        initializer = DofusDataInitializer(db_path)
        success = initializer.initialize_all_data()
        
        if success:
            # Afficher les statistiques
            stats = initializer.get_initialization_stats()
            
            print("[OK] Initialisation des données DOFUS terminée!")
            print("Statistiques d'initialisation:")
            for table, count in stats.items():
                if table != "error":
                    print(f"  - {table}: {count} entrées")
        
        return success
        
    except Exception as e:
        logger.error(f"Erreur critique lors de l'initialisation: {e}")
        return False


if __name__ == "__main__":
    # Initialiser les données
    success = initialize_dofus_data()
    exit(0 if success else 1)