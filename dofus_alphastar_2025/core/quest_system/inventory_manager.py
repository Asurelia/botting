#!/usr/bin/env python3
"""
InventoryManager - Gestionnaire intelligent d'inventaire DOFUS
Optimise automatiquement l'espace et gère les items selon les priorités de quête
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json

import torch
import numpy as np

from config import config
from core.hrm_reasoning import create_hrm_model, HRMOutput
from core.vision_engine_v2 import create_vision_engine, SAMSegment

logger = logging.getLogger(__name__)

class ItemType(Enum):
    """Types d'items"""
    WEAPON = "weapon"
    ARMOR = "armor"
    ACCESSORY = "accessory"
    CONSUMABLE = "consumable"
    RESOURCE = "resource"
    QUEST_ITEM = "quest_item"
    CURRENCY = "currency"
    TOOL = "tool"
    COSMETIC = "cosmetic"
    UNKNOWN = "unknown"

class ItemRarity(Enum):
    """Rareté des items"""
    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    EPIC = 4
    LEGENDARY = 5
    MYTHIC = 6

class InventoryAction(Enum):
    """Actions d'inventaire possibles"""
    KEEP = "keep"
    DROP = "drop"
    SELL = "sell"
    USE = "use"
    BANK = "bank"
    TRADE = "trade"
    DELETE = "delete"
    EQUIP = "equip"

@dataclass
class InventoryItem:
    """Item dans l'inventaire"""
    item_id: str
    name: str
    item_type: ItemType
    rarity: ItemRarity = ItemRarity.COMMON
    quantity: int = 1
    position: Optional[Tuple[int, int]] = None
    level_requirement: int = 1
    value_estimate: int = 0

    # Propriétés de quête
    is_quest_item: bool = False
    related_quests: List[str] = field(default_factory=list)
    quest_priority: int = 0

    # Propriétés de gameplay
    is_equipped: bool = False
    is_stackable: bool = True
    stack_size: int = 100
    last_used: float = 0.0

    # Métadonnées
    description: str = ""
    icon_hash: Optional[str] = None
    detected_confidence: float = 0.0

    @property
    def priority_score(self) -> float:
        """Score de priorité pour tri automatique"""
        score = 0.0

        # Bonus items de quête
        if self.is_quest_item:
            score += 100 + self.quest_priority * 10

        # Bonus équipement équipé
        if self.is_equipped:
            score += 50

        # Bonus par rareté
        score += self.rarity.value * 5

        # Bonus récent usage
        if time.time() - self.last_used < 3600:  # 1 heure
            score += 20

        # Bonus valeur
        score += min(20, self.value_estimate / 1000)

        return score

@dataclass
class InventorySlot:
    """Slot d'inventaire"""
    position: Tuple[int, int]
    item: Optional[InventoryItem] = None
    is_locked: bool = False

    @property
    def is_empty(self) -> bool:
        return self.item is None

    @property
    def is_full(self) -> bool:
        if not self.item:
            return False
        return self.item.quantity >= self.item.stack_size

class ItemDatabase:
    """Base de données des items DOFUS"""

    def __init__(self, data_file: str = "data/items/item_database.json"):
        self.data_file = data_file
        self.items: Dict[str, Dict[str, Any]] = {}
        self.name_to_id: Dict[str, str] = {}
        self.load_database()

    def load_database(self):
        """Charge la base de données des items"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.items = data.get("items", {})

                # Créer index nom -> id
                for item_id, item_data in self.items.items():
                    name = item_data.get("name", "").lower()
                    self.name_to_id[name] = item_id

                    # Ajouter variantes de nom
                    for alias in item_data.get("aliases", []):
                        self.name_to_id[alias.lower()] = item_id

            logger.info(f"Base de données items chargée: {len(self.items)} items")

        except Exception as e:
            logger.warning(f"Impossible de charger la base d'items: {e}")
            self._create_default_database()

    def _create_default_database(self):
        """Crée une base de données par défaut"""
        self.items = {
            "pain": {
                "name": "Pain",
                "type": "consumable",
                "rarity": 1,
                "value": 1,
                "stackable": True
            },
            "potion_de_vie": {
                "name": "Potion de Vie",
                "type": "consumable",
                "rarity": 2,
                "value": 50,
                "stackable": True
            }
        }

        for item_id, item_data in self.items.items():
            self.name_to_id[item_data["name"].lower()] = item_id

    def identify_item(self, item_name: str) -> Optional[Dict[str, Any]]:
        """Identifie un item par son nom"""
        clean_name = self._clean_item_name(item_name)
        item_id = self.name_to_id.get(clean_name)

        if item_id:
            return self.items[item_id]

        # Recherche fuzzy
        return self._fuzzy_search(clean_name)

    def _clean_item_name(self, name: str) -> str:
        """Nettoie le nom d'item pour la recherche"""
        # Supprimer caractères spéciaux et normaliser
        clean = re.sub(r'[^\w\s]', '', name.lower())
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def _fuzzy_search(self, name: str) -> Optional[Dict[str, Any]]:
        """Recherche approximative d'item"""
        best_match = None
        best_score = 0

        for known_name, item_id in self.name_to_id.items():
            # Score basé sur correspondance de mots
            name_words = set(name.split())
            known_words = set(known_name.split())

            if name_words & known_words:  # Intersection non vide
                score = len(name_words & known_words) / len(name_words | known_words)

                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = self.items[item_id]

        return best_match

class InventoryAnalyzer:
    """Analyseur d'inventaire via vision"""

    def __init__(self):
        self.vision_engine = create_vision_engine()
        self.item_database = ItemDatabase()

        # Patterns de reconnaissance d'items
        self.quantity_pattern = r'(\d+)x?\s*(.*)'
        self.item_patterns = {
            'equipped': r'\[E\]',
            'quest': r'\[Q\]',
            'stackable': r'x\d+',
        }

    def analyze_inventory_screen(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyse l'écran d'inventaire"""
        try:
            # Analyser avec SAM 2 pour détecter les slots
            vision_results = self.vision_engine.analyze_screenshot(screenshot)
            ui_elements = vision_results.get("ui_elements", [])
            text_detections = vision_results.get("text_detections", [])

            # Identifier la grille d'inventaire
            inventory_grid = self._detect_inventory_grid(ui_elements)

            # Analyser les items dans chaque slot
            items_detected = self._analyze_inventory_items(
                screenshot, inventory_grid, text_detections
            )

            # Calculer statistiques
            stats = self._calculate_inventory_stats(items_detected)

            return {
                "items": items_detected,
                "grid_size": inventory_grid.get("size", (0, 0)),
                "stats": stats,
                "analysis_time": time.time()
            }

        except Exception as e:
            logger.error(f"Erreur analyse inventaire: {e}")
            return {"error": str(e)}

    def _detect_inventory_grid(self, ui_elements: List) -> Dict[str, Any]:
        """Détecte la grille d'inventaire"""
        # Chercher éléments rectangulaires réguliers (slots)
        slot_candidates = []

        for element in ui_elements:
            if hasattr(element, 'segment'):
                segment = element.segment
                # Slots sont généralement carrés ou rectangulaires
                if 30 <= segment.width <= 80 and 30 <= segment.height <= 80:
                    slot_candidates.append({
                        "x": segment.x,
                        "y": segment.y,
                        "width": segment.width,
                        "height": segment.height
                    })

        if not slot_candidates:
            return {"size": (0, 0), "slots": []}

        # Organiser en grille
        slots_by_row = {}
        slot_height = slot_candidates[0]["height"]

        for slot in slot_candidates:
            row = slot["y"] // (slot_height + 5)  # 5px d'espacement
            if row not in slots_by_row:
                slots_by_row[row] = []
            slots_by_row[row].append(slot)

        # Trier chaque rangée par X
        for row in slots_by_row:
            slots_by_row[row].sort(key=lambda s: s["x"])

        rows = len(slots_by_row)
        cols = max(len(slots_by_row[row]) for row in slots_by_row) if slots_by_row else 0

        return {
            "size": (rows, cols),
            "slots": slots_by_row,
            "slot_size": (slot_candidates[0]["width"], slot_candidates[0]["height"])
        }

    def _analyze_inventory_items(self,
                                screenshot: np.ndarray,
                                grid_info: Dict[str, Any],
                                text_detections: List) -> List[InventoryItem]:
        """Analyse les items dans l'inventaire"""
        items = []
        slots = grid_info.get("slots", {})

        # Associer texte aux slots
        for row_idx, row_slots in slots.items():
            for col_idx, slot in enumerate(row_slots):
                # Chercher texte dans ce slot
                slot_items = self._extract_slot_items(
                    slot, text_detections, (row_idx, col_idx)
                )
                items.extend(slot_items)

        return items

    def _extract_slot_items(self,
                           slot: Dict[str, Any],
                           text_detections: List,
                           position: Tuple[int, int]) -> List[InventoryItem]:
        """Extrait les items d'un slot spécifique"""
        items = []

        # Zone du slot avec marge
        slot_x = slot["x"] - 10
        slot_y = slot["y"] - 10
        slot_w = slot["width"] + 20
        slot_h = slot["height"] + 20

        # Texte dans cette zone
        slot_texts = []
        for detection in text_detections:
            text_x, text_y = detection.bbox[:2]

            if (slot_x <= text_x <= slot_x + slot_w and
                slot_y <= text_y <= slot_y + slot_h):
                slot_texts.append(detection.text)

        if not slot_texts:
            return items

        # Analyser le texte combiné
        combined_text = " ".join(slot_texts)
        item_info = self._parse_item_text(combined_text)

        if item_info:
            # Identifier dans la base de données
            db_info = self.item_database.identify_item(item_info["name"])

            item = InventoryItem(
                item_id=item_info.get("id", f"unknown_{int(time.time())}"),
                name=item_info["name"],
                item_type=ItemType(db_info.get("type", "unknown")) if db_info else ItemType.UNKNOWN,
                rarity=ItemRarity(db_info.get("rarity", 1)) if db_info else ItemRarity.COMMON,
                quantity=item_info.get("quantity", 1),
                position=position,
                value_estimate=db_info.get("value", 0) if db_info else 0,
                is_quest_item=item_info.get("is_quest", False),
                is_equipped=item_info.get("is_equipped", False),
                detected_confidence=0.8
            )
            items.append(item)

        return items

    def _parse_item_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse le texte d'un item"""
        if not text.strip():
            return None

        item_info = {
            "name": text.strip(),
            "quantity": 1,
            "is_quest": False,
            "is_equipped": False
        }

        # Extraire quantité
        qty_match = re.match(self.quantity_pattern, text)
        if qty_match:
            quantity, name = qty_match.groups()
            item_info["quantity"] = int(quantity)
            item_info["name"] = name.strip()

        # Détecter marqueurs spéciaux
        for marker, pattern in self.item_patterns.items():
            if re.search(pattern, text):
                if marker == "equipped":
                    item_info["is_equipped"] = True
                elif marker == "quest":
                    item_info["is_quest"] = True

        return item_info

    def _calculate_inventory_stats(self, items: List[InventoryItem]) -> Dict[str, Any]:
        """Calcule les statistiques d'inventaire"""
        total_items = len(items)
        total_value = sum(item.value_estimate * item.quantity for item in items)

        # Répartition par type
        type_distribution = {}
        for item in items:
            item_type = item.item_type.value
            type_distribution[item_type] = type_distribution.get(item_type, 0) + 1

        # Items de quête
        quest_items = [item for item in items if item.is_quest_item]

        return {
            "total_items": total_items,
            "total_value": total_value,
            "quest_items": len(quest_items),
            "equipped_items": len([item for item in items if item.is_equipped]),
            "type_distribution": type_distribution,
            "average_value": total_value / max(total_items, 1)
        }

class InventoryOptimizer:
    """Optimiseur d'inventaire avec HRM"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hrm_model = create_hrm_model().to(self.device)

        # Règles d'optimisation
        self.optimization_rules = {
            "quest_priority": 10,  # Items de quête = priorité max
            "equipment_priority": 8,  # Équipement équipé
            "rarity_priority": 6,  # Items rares
            "value_threshold": 1000,  # Seuil de valeur pour garder
            "recent_use_hours": 24,  # Heures pour considérer usage récent
        }

    def optimize_inventory(self,
                          items: List[InventoryItem],
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimise l'inventaire selon les priorités"""

        if not items:
            return {"actions": [], "summary": "Inventaire vide"}

        # Analyser le contexte
        optimization_context = {
            "current_quests": context.get("current_quests", []) if context else [],
            "player_level": context.get("player_level", 1) if context else 1,
            "inventory_space": context.get("inventory_space", 100) if context else 100,
            "current_kamas": context.get("current_kamas", 0) if context else 0,
            "force_cleanup": context.get("force_cleanup", False) if context else False
        }

        # Trier par priorité
        sorted_items = sorted(items, key=lambda x: x.priority_score, reverse=True)

        # Générer actions d'optimisation
        actions = []

        for item in sorted_items:
            action = self._determine_item_action(item, optimization_context)
            if action["action"] != InventoryAction.KEEP:
                actions.append(action)

        # Résumé de l'optimisation
        summary = self._create_optimization_summary(actions, items)

        return {
            "actions": actions,
            "summary": summary,
            "optimization_context": optimization_context
        }

    def _determine_item_action(self,
                              item: InventoryItem,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Détermine l'action à prendre pour un item"""

        # Items de quête active = toujours garder
        if item.is_quest_item and any(quest in item.related_quests
                                     for quest in context["current_quests"]):
            return {
                "action": InventoryAction.KEEP,
                "item": item,
                "reason": "Item de quête active",
                "priority": 10
            }

        # Équipement équipé = garder
        if item.is_equipped:
            return {
                "action": InventoryAction.KEEP,
                "item": item,
                "reason": "Équipement équipé",
                "priority": 9
            }

        # Items de valeur élevée
        if item.value_estimate > self.optimization_rules["value_threshold"]:
            if item.item_type in [ItemType.WEAPON, ItemType.ARMOR]:
                return {
                    "action": InventoryAction.BANK,
                    "item": item,
                    "reason": "Item de valeur - à stocker",
                    "priority": 7
                }
            else:
                return {
                    "action": InventoryAction.SELL,
                    "item": item,
                    "reason": "Item de valeur - à vendre",
                    "priority": 6
                }

        # Items récemment utilisés
        recent_use_threshold = time.time() - (self.optimization_rules["recent_use_hours"] * 3600)
        if item.last_used > recent_use_threshold:
            return {
                "action": InventoryAction.KEEP,
                "item": item,
                "reason": "Utilisé récemment",
                "priority": 5
            }

        # Items de quête obsolètes
        if item.is_quest_item and not any(quest in item.related_quests
                                         for quest in context["current_quests"]):
            return {
                "action": InventoryAction.DROP,
                "item": item,
                "reason": "Quête obsolète",
                "priority": 2
            }

        # Items communs de faible valeur
        if (item.rarity == ItemRarity.COMMON and
            item.value_estimate < 100 and
            item.item_type == ItemType.RESOURCE):
            return {
                "action": InventoryAction.SELL,
                "item": item,
                "reason": "Ressource commune",
                "priority": 3
            }

        # Par défaut, garder
        return {
            "action": InventoryAction.KEEP,
            "item": item,
            "reason": "Garder par défaut",
            "priority": 4
        }

    def _create_optimization_summary(self,
                                   actions: List[Dict[str, Any]],
                                   all_items: List[InventoryItem]) -> Dict[str, Any]:
        """Crée un résumé de l'optimisation"""

        action_counts = {}
        total_value_freed = 0
        items_to_process = 0

        for action in actions:
            action_type = action["action"].value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

            if action["action"] in [InventoryAction.DROP, InventoryAction.SELL]:
                items_to_process += 1
                if action["action"] == InventoryAction.SELL:
                    total_value_freed += action["item"].value_estimate

        space_freed = len([a for a in actions
                          if a["action"] in [InventoryAction.DROP, InventoryAction.SELL, InventoryAction.BANK]])

        return {
            "total_actions": len(actions),
            "action_breakdown": action_counts,
            "space_freed": space_freed,
            "items_to_process": items_to_process,
            "estimated_kamas_gain": total_value_freed,
            "optimization_efficiency": (space_freed / max(len(all_items), 1)) * 100
        }

class InventoryManager:
    """Gestionnaire principal d'inventaire"""

    def __init__(self):
        self.analyzer = InventoryAnalyzer()
        self.optimizer = InventoryOptimizer()

        # État de l'inventaire
        self.current_items: List[InventoryItem] = []
        self.last_analysis: float = 0.0
        self.analysis_interval: float = 5.0  # Analyser toutes les 5 secondes

        # Contexte de jeu
        self.game_context = {
            "current_quests": [],
            "player_level": 1,
            "current_kamas": 0,
            "inventory_space": 100
        }

        # Statistiques
        self.total_optimizations = 0
        self.items_processed = 0
        self.kamas_gained = 0

        logger.info("InventoryManager initialisé avec succès")

    def update_context(self, **kwargs):
        """Met à jour le contexte de jeu"""
        self.game_context.update(kwargs)

    def analyze_and_optimize(self, screenshot: np.ndarray, force_analysis: bool = False) -> Dict[str, Any]:
        """Analyse et optimise l'inventaire"""
        current_time = time.time()

        # Vérifier s'il faut analyser
        if not force_analysis and (current_time - self.last_analysis) < self.analysis_interval:
            return {"status": "skipped", "reason": "Analysis too recent"}

        try:
            # Analyser l'inventaire
            analysis_result = self.analyzer.analyze_inventory_screen(screenshot)

            if "error" in analysis_result:
                return {"status": "error", "error": analysis_result["error"]}

            # Mettre à jour l'état
            self.current_items = analysis_result["items"]
            self.last_analysis = current_time

            # Optimiser si nécessaire
            optimization_result = self.optimizer.optimize_inventory(
                self.current_items, self.game_context
            )

            # Mettre à jour statistiques
            self.total_optimizations += 1

            return {
                "status": "success",
                "analysis": analysis_result,
                "optimization": optimization_result,
                "timestamp": current_time
            }

        except Exception as e:
            logger.error(f"Erreur analyse/optimisation inventaire: {e}")
            return {"status": "error", "error": str(e)}

    def execute_optimization_action(self, action: Dict[str, Any]) -> bool:
        """Exécute une action d'optimisation"""
        try:
            action_type = action["action"]
            item = action["item"]

            # Log de l'action
            logger.info(f"Exécution action {action_type.value} pour {item.name}")

            # Ici, l'intégration avec le système de contrôle du bot
            # pour exécuter réellement l'action (clic, drag&drop, etc.)

            # Simuler succès
            self.items_processed += 1
            if action_type == InventoryAction.SELL:
                self.kamas_gained += item.value_estimate

            return True

        except Exception as e:
            logger.error(f"Erreur exécution action inventaire: {e}")
            return False

    def get_inventory_stats(self) -> Dict[str, Any]:
        """Statistiques de l'inventaire"""
        current_stats = {}

        if self.current_items:
            total_value = sum(item.value_estimate * item.quantity for item in self.current_items)
            quest_items = [item for item in self.current_items if item.is_quest_item]

            current_stats = {
                "total_items": len(self.current_items),
                "total_value": total_value,
                "quest_items": len(quest_items),
                "equipped_items": len([item for item in self.current_items if item.is_equipped]),
                "space_used": len(self.current_items),
                "space_available": max(0, self.game_context["inventory_space"] - len(self.current_items))
            }

        return {
            "current": current_stats,
            "lifetime": {
                "total_optimizations": self.total_optimizations,
                "items_processed": self.items_processed,
                "kamas_gained": self.kamas_gained,
                "last_analysis": self.last_analysis
            }
        }

    def get_priority_items(self, item_type: Optional[ItemType] = None) -> List[InventoryItem]:
        """Récupère les items prioritaires"""
        filtered_items = self.current_items

        if item_type:
            filtered_items = [item for item in filtered_items if item.item_type == item_type]

        return sorted(filtered_items, key=lambda x: x.priority_score, reverse=True)

    def find_items_by_name(self, name_pattern: str) -> List[InventoryItem]:
        """Trouve des items par nom (regex supporté)"""
        pattern = re.compile(name_pattern, re.IGNORECASE)
        return [item for item in self.current_items if pattern.search(item.name)]

    def get_quest_items(self, quest_id: str = None) -> List[InventoryItem]:
        """Récupère les items de quête"""
        quest_items = [item for item in self.current_items if item.is_quest_item]

        if quest_id:
            quest_items = [item for item in quest_items if quest_id in item.related_quests]

        return quest_items

def create_inventory_manager() -> InventoryManager:
    """Factory function pour créer un InventoryManager"""
    return InventoryManager()