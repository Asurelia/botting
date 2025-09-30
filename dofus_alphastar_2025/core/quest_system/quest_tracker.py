#!/usr/bin/env python3
"""
QuestTracker - Suivi en temps réel des objectifs de quête DOFUS
Utilise TrOCR et analyse vision pour détecter automatiquement la progression
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

import torch
import numpy as np
import cv2
from PIL import Image

from config import config
from core.vision_engine_v2 import create_vision_engine, TextDetection

logger = logging.getLogger(__name__)

class ObjectiveType(Enum):
    """Types d'objectifs de quête"""
    KILL_COUNT = "kill_count"
    ITEM_COLLECTION = "item_collection"
    LOCATION_VISIT = "location_visit"
    NPC_INTERACTION = "npc_interaction"
    ITEM_DELIVERY = "item_delivery"
    EXPERIENCE_GAIN = "experience_gain"
    LEVEL_REACH = "level_reach"
    CUSTOM = "custom"

@dataclass
class QuestObjective:
    """Objectif de quête trackable"""
    objective_id: str
    objective_type: ObjectiveType
    description: str
    target: str
    current_count: int = 0
    target_count: int = 1
    last_update: float = 0.0

    # Patterns de détection TrOCR
    text_patterns: List[str] = field(default_factory=list)
    screen_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Métadonnées
    zone_specific: bool = False
    current_zone: Optional[str] = None
    priority: int = 1  # 1=basse, 5=haute

    @property
    def progress_percentage(self) -> float:
        """Pourcentage de progression"""
        if self.target_count == 0:
            return 100.0
        return min(100.0, (self.current_count / self.target_count) * 100.0)

    @property
    def is_completed(self) -> bool:
        """Vérifie si l'objectif est terminé"""
        return self.current_count >= self.target_count

    def update_progress(self, new_count: int, force: bool = False) -> bool:
        """Met à jour la progression"""
        if new_count > self.current_count or force:
            self.current_count = new_count
            self.last_update = time.time()
            return True
        return False

@dataclass
class QuestProgress:
    """Progression complète d'une quête"""
    quest_id: str
    quest_name: str
    objectives: List[QuestObjective] = field(default_factory=list)
    last_screen_check: float = 0.0
    total_checks: int = 0
    detection_accuracy: float = 0.0

    @property
    def overall_progress(self) -> float:
        """Progression globale de la quête"""
        if not self.objectives:
            return 0.0

        total_progress = sum(obj.progress_percentage for obj in self.objectives)
        return total_progress / len(self.objectives)

    @property
    def completed_objectives(self) -> int:
        """Nombre d'objectifs terminés"""
        return sum(1 for obj in self.objectives if obj.is_completed)

    def get_active_objectives(self) -> List[QuestObjective]:
        """Retourne les objectifs non terminés"""
        return [obj for obj in self.objectives if not obj.is_completed]

class TextPatternMatcher:
    """Gestionnaire de patterns pour reconnaissance de texte de quête"""

    def __init__(self):
        # Patterns prédéfinis pour DOFUS
        self.kill_patterns = [
            r"tuer (\d+) ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)",
            r"éliminer (\d+) ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)",
            r"vaincre (\d+) ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)",
            r"(\d+)/(\d+) ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+) tués?"
        ]

        self.collection_patterns = [
            r"collecter (\d+) ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)",
            r"récupérer (\d+) ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)",
            r"ramasser (\d+) ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)",
            r"(\d+)/(\d+) ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)"
        ]

        self.npc_patterns = [
            r"parler à ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)",
            r"voir ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)",
            r"rencontrer ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)"
        ]

        self.location_patterns = [
            r"aller à ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)",
            r"se rendre à ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)",
            r"visiter ([a-zA-Zàâäéèêëïîôùûüÿç\s-]+)"
        ]

    def extract_quest_info(self, text: str) -> Dict[str, Any]:
        """Extrait les informations de quête depuis le texte"""
        text_lower = text.lower().strip()

        # Essayer patterns de kill
        for pattern in self.kill_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if len(match.groups()) == 3:  # Pattern avec progression (x/y)
                    current, target, monster = match.groups()
                    return {
                        "type": ObjectiveType.KILL_COUNT,
                        "target": monster.strip(),
                        "current": int(current),
                        "target_count": int(target),
                        "raw_text": text
                    }
                else:  # Pattern simple (tuer X)
                    count, monster = match.groups()
                    return {
                        "type": ObjectiveType.KILL_COUNT,
                        "target": monster.strip(),
                        "target_count": int(count),
                        "raw_text": text
                    }

        # Essayer patterns de collection
        for pattern in self.collection_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if len(match.groups()) == 3:  # Pattern avec progression
                    current, target, item = match.groups()
                    return {
                        "type": ObjectiveType.ITEM_COLLECTION,
                        "target": item.strip(),
                        "current": int(current),
                        "target_count": int(target),
                        "raw_text": text
                    }
                else:  # Pattern simple
                    count, item = match.groups()
                    return {
                        "type": ObjectiveType.ITEM_COLLECTION,
                        "target": item.strip(),
                        "target_count": int(count),
                        "raw_text": text
                    }

        # Essayer patterns PNJ
        for pattern in self.npc_patterns:
            match = re.search(pattern, text_lower)
            if match:
                npc_name = match.group(1).strip()
                return {
                    "type": ObjectiveType.NPC_INTERACTION,
                    "target": npc_name,
                    "target_count": 1,
                    "raw_text": text
                }

        # Essayer patterns de localisation
        for pattern in self.location_patterns:
            match = re.search(pattern, text_lower)
            if match:
                location = match.group(1).strip()
                return {
                    "type": ObjectiveType.LOCATION_VISIT,
                    "target": location,
                    "target_count": 1,
                    "raw_text": text
                }

        return None

    def is_quest_text(self, text: str) -> bool:
        """Détermine si un texte est lié à une quête"""
        quest_keywords = [
            'quête', 'mission', 'tâche', 'objectif',
            'tuer', 'éliminer', 'vaincre', 'collecter', 'récupérer',
            'parler', 'voir', 'rencontrer', 'aller', 'visiter',
            'ramasser', 'livrer', 'apporter'
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in quest_keywords)

class QuestTracker:
    """Tracker principal pour suivi automatique des quêtes"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vision engine pour TrOCR
        self.vision_engine = create_vision_engine()

        # Pattern matcher
        self.pattern_matcher = TextPatternMatcher()

        # État du tracking
        self.tracked_quests: Dict[str, QuestProgress] = {}
        self.last_full_scan: float = 0.0
        self.scan_interval: float = 2.0  # Scan complet toutes les 2 secondes
        self.quick_scan_interval: float = 0.5  # Scan rapide toutes les 0.5s

        # Régions d'écran importantes pour DOFUS
        self.quest_regions = {
            "quest_log": (0.02, 0.02, 0.3, 0.8),  # Journal de quête (gauche)
            "chat_area": (0.02, 0.7, 0.6, 0.98),  # Zone de chat (bas)
            "notifications": (0.3, 0.02, 0.98, 0.3),  # Notifications (haut droite)
            "dialogue": (0.2, 0.4, 0.8, 0.8)  # Zone de dialogue (centre)
        }

        # Cache pour optimisation
        self.text_cache: Dict[str, List[TextDetection]] = {}
        self.cache_duration = 1.0  # Cache valide 1 seconde

        # Statistiques
        self.total_scans = 0
        self.successful_detections = 0
        self.last_detection_time = 0.0

        logger.info("QuestTracker initialisé avec succès")

    def track_quest(self, quest_id: str, quest_name: str, objectives: List[Dict[str, Any]]) -> bool:
        """Commence le tracking d'une quête"""
        try:
            quest_objectives = []

            for obj_data in objectives:
                objective = QuestObjective(
                    objective_id=obj_data["objective_id"],
                    objective_type=ObjectiveType(obj_data["type"]),
                    description=obj_data["description"],
                    target=obj_data["target"],
                    target_count=obj_data.get("target_count", 1),
                    text_patterns=obj_data.get("text_patterns", []),
                    screen_regions=obj_data.get("screen_regions", []),
                    zone_specific=obj_data.get("zone_specific", False),
                    priority=obj_data.get("priority", 1)
                )
                quest_objectives.append(objective)

            progress = QuestProgress(
                quest_id=quest_id,
                quest_name=quest_name,
                objectives=quest_objectives
            )

            self.tracked_quests[quest_id] = progress
            logger.info(f"Tracking démarré pour: {quest_name} ({len(quest_objectives)} objectifs)")
            return True

        except Exception as e:
            logger.error(f"Erreur démarrage tracking {quest_id}: {e}")
            return False

    def stop_tracking(self, quest_id: str):
        """Arrête le tracking d'une quête"""
        if quest_id in self.tracked_quests:
            del self.tracked_quests[quest_id]
            logger.info(f"Tracking arrêté pour: {quest_id}")

    def update_from_screenshot(self, screenshot: np.ndarray, force_full_scan: bool = False) -> Dict[str, Any]:
        """Met à jour le tracking depuis une capture d'écran"""
        current_time = time.time()

        # Déterminer type de scan
        should_full_scan = (
            force_full_scan or
            (current_time - self.last_full_scan) > self.scan_interval
        )

        if should_full_scan:
            return self._full_screen_scan(screenshot)
        else:
            return self._quick_scan(screenshot)

    def _full_screen_scan(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Scan complet de l'écran pour détecter toutes les quêtes"""
        current_time = time.time()
        self.last_full_scan = current_time
        self.total_scans += 1

        try:
            # Analyser toutes les régions importantes
            all_detections = []

            for region_name, (x_ratio, y_ratio, w_ratio, h_ratio) in self.quest_regions.items():
                region_detections = self._scan_screen_region(
                    screenshot, x_ratio, y_ratio, w_ratio, h_ratio, region_name
                )
                all_detections.extend(region_detections)

            # Analyser le texte détecté
            quest_updates = {}
            new_objectives = []

            for detection in all_detections:
                if self.pattern_matcher.is_quest_text(detection.text):
                    quest_info = self.pattern_matcher.extract_quest_info(detection.text)

                    if quest_info:
                        # Mettre à jour objectifs existants
                        updated = self._update_existing_objectives(quest_info, detection)

                        if updated:
                            quest_updates[updated] = quest_info
                        else:
                            # Nouvel objectif potentiel
                            new_objectives.append({
                                "quest_info": quest_info,
                                "detection": detection
                            })

            # Statistiques
            if quest_updates or new_objectives:
                self.successful_detections += 1
                self.last_detection_time = current_time

            return {
                "scan_type": "full",
                "timestamp": current_time,
                "quest_updates": quest_updates,
                "new_objectives": new_objectives,
                "total_detections": len(all_detections),
                "quest_detections": len(quest_updates) + len(new_objectives)
            }

        except Exception as e:
            logger.error(f"Erreur scan complet: {e}")
            return {"scan_type": "full", "error": str(e)}

    def _quick_scan(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Scan rapide focalisé sur les zones de notification"""
        current_time = time.time()

        try:
            # Scanner uniquement notifications et chat
            quick_regions = ["notifications", "chat_area"]
            detections = []

            for region_name in quick_regions:
                if region_name in self.quest_regions:
                    region_coords = self.quest_regions[region_name]
                    region_detections = self._scan_screen_region(
                        screenshot, *region_coords, region_name
                    )
                    detections.extend(region_detections)

            # Chercher mises à jour rapides
            updates = {}
            for detection in detections:
                if any(keyword in detection.text.lower() for keyword in ['tué', 'collecté', 'trouvé', 'terminé']):
                    quest_info = self.pattern_matcher.extract_quest_info(detection.text)
                    if quest_info:
                        quest_id = self._update_existing_objectives(quest_info, detection)
                        if quest_id:
                            updates[quest_id] = quest_info

            return {
                "scan_type": "quick",
                "timestamp": current_time,
                "updates": updates,
                "detections_count": len(detections)
            }

        except Exception as e:
            logger.error(f"Erreur scan rapide: {e}")
            return {"scan_type": "quick", "error": str(e)}

    def _scan_screen_region(self,
                           screenshot: np.ndarray,
                           x_ratio: float, y_ratio: float,
                           w_ratio: float, h_ratio: float,
                           region_name: str) -> List[TextDetection]:
        """Scanne une région spécifique de l'écran"""

        # Calculer coordonnées absolues
        h, w = screenshot.shape[:2]
        x1 = int(x_ratio * w)
        y1 = int(y_ratio * h)
        x2 = int(w_ratio * w)
        y2 = int(h_ratio * h)

        # Extraire la région
        region = screenshot[y1:y2, x1:x2]

        if region.size == 0:
            return []

        # Cache key
        region_hash = hash(region.tobytes())
        cache_key = f"{region_name}_{region_hash}"

        # Vérifier cache
        if cache_key in self.text_cache:
            cache_entry = self.text_cache[cache_key]
            if (time.time() - cache_entry["timestamp"]) < self.cache_duration:
                return cache_entry["detections"]

        try:
            # Convertir en image PIL
            region_image = Image.fromarray(region)

            # Analyser avec TrOCR
            vision_results = self.vision_engine.analyze_screenshot(region)
            detections = vision_results.get("text_detections", [])

            # Ajuster les coordonnées relatives à l'écran complet
            for detection in detections:
                bbox = detection.bbox
                detection.bbox = (
                    bbox[0] + x1,
                    bbox[1] + y1,
                    bbox[2],
                    bbox[3]
                )

            # Mettre en cache
            self.text_cache[cache_key] = {
                "detections": detections,
                "timestamp": time.time()
            }

            return detections

        except Exception as e:
            logger.error(f"Erreur scan région {region_name}: {e}")
            return []

    def _update_existing_objectives(self, quest_info: Dict[str, Any], detection: TextDetection) -> Optional[str]:
        """Met à jour les objectifs existants avec les nouvelles informations"""

        target = quest_info["target"]
        objective_type = quest_info["type"]

        for quest_id, progress in self.tracked_quests.items():
            for objective in progress.objectives:
                # Correspondance par type et cible
                if (objective.objective_type == objective_type and
                    objective.target.lower() == target.lower()):

                    # Mettre à jour la progression
                    new_count = quest_info.get("current", objective.current_count + 1)

                    if objective.update_progress(new_count):
                        logger.debug(f"Objectif mis à jour: {objective.description} ({new_count}/{objective.target_count})")
                        return quest_id

        return None

    def get_quest_progress(self, quest_id: str) -> Optional[QuestProgress]:
        """Récupère la progression d'une quête"""
        return self.tracked_quests.get(quest_id)

    def get_all_progress(self) -> Dict[str, QuestProgress]:
        """Récupère toutes les progressions"""
        return self.tracked_quests.copy()

    def get_completion_stats(self) -> Dict[str, Any]:
        """Statistiques de complétion"""
        total_objectives = 0
        completed_objectives = 0

        for progress in self.tracked_quests.values():
            total_objectives += len(progress.objectives)
            completed_objectives += progress.completed_objectives

        completion_rate = (completed_objectives / max(total_objectives, 1)) * 100

        return {
            "total_quests": len(self.tracked_quests),
            "total_objectives": total_objectives,
            "completed_objectives": completed_objectives,
            "completion_rate": completion_rate,
            "tracking_accuracy": (self.successful_detections / max(self.total_scans, 1)) * 100,
            "last_detection": self.last_detection_time,
            "total_scans": self.total_scans
        }

    def create_auto_objective(self, detection: TextDetection) -> Optional[QuestObjective]:
        """Crée automatiquement un objectif depuis une détection"""
        quest_info = self.pattern_matcher.extract_quest_info(detection.text)

        if not quest_info:
            return None

        objective_id = f"auto_{int(time.time())}_{hash(quest_info['target'])}"

        return QuestObjective(
            objective_id=objective_id,
            objective_type=quest_info["type"],
            description=quest_info["raw_text"],
            target=quest_info["target"],
            target_count=quest_info["target_count"],
            current_count=quest_info.get("current", 0),
            text_patterns=[quest_info["raw_text"]],
            screen_regions=[detection.bbox]
        )

    def cleanup_cache(self):
        """Nettoie le cache expiré"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.text_cache.items()
            if (current_time - entry["timestamp"]) > self.cache_duration * 2
        ]

        for key in expired_keys:
            del self.text_cache[key]

def create_quest_tracker() -> QuestTracker:
    """Factory function pour créer un QuestTracker"""
    return QuestTracker()