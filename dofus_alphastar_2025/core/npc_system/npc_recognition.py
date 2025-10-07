#!/usr/bin/env python3
"""
NPCRecognition - Système de reconnaissance et classification des PNJ DOFUS
Utilise vision + IA pour identifier et catégoriser automatiquement les PNJ
"""

import time
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from PIL import Image

from config import config
from core.vision_engine_v2 import create_vision_engine, SAMSegment, TextDetection
from core.hrm_reasoning import create_hrm_model, HRMOutput

logger = logging.getLogger(__name__)

class NPCType(Enum):
    """Types de PNJ"""
    QUEST_GIVER = "quest_giver"
    MERCHANT = "merchant"
    BANK = "bank"
    TRAINER = "trainer"
    GUARD = "guard"
    ZAAP_KEEPER = "zaap_keeper"
    GUILD_MASTER = "guild_master"
    CRAFT_MASTER = "craft_master"
    INN_KEEPER = "inn_keeper"
    TRANSPORT = "transport"
    VILLAGER = "villager"
    IMPORTANT = "important"
    UNKNOWN = "unknown"

class NPCState(Enum):
    """États d'interaction PNJ"""
    IDLE = "idle"
    BUSY = "busy"
    TALKING = "talking"
    QUEST_AVAILABLE = "quest_available"
    QUEST_COMPLETE = "quest_complete"
    SHOP_OPEN = "shop_open"
    UNAVAILABLE = "unavailable"

class InteractionType(Enum):
    """Types d'interaction possible"""
    TALK = "talk"
    TRADE = "trade"
    TRAIN = "train"
    BANK = "bank"
    CRAFT = "craft"
    QUEST_START = "quest_start"
    QUEST_COMPLETE = "quest_complete"
    TRANSPORT = "transport"
    NONE = "none"

@dataclass
class NPCAppearance:
    """Caractéristiques visuelles d'un PNJ"""
    color_signature: Optional[np.ndarray] = None
    size_range: Tuple[int, int] = (30, 100)
    shape_features: Dict[str, float] = field(default_factory=dict)
    clothing_colors: List[str] = field(default_factory=list)
    distinctive_features: List[str] = field(default_factory=list)
    confidence: float = 0.0

@dataclass
class NPCData:
    """Données complètes d'un PNJ"""
    npc_id: str
    name: str
    npc_type: NPCType
    location: Tuple[int, int]
    map_id: Optional[str] = None

    # Informations visuelles
    appearance: Optional[NPCAppearance] = None
    last_seen: float = 0.0

    # Interactions
    available_interactions: List[InteractionType] = field(default_factory=list)
    current_state: NPCState = NPCState.IDLE

    # Dialogue et quêtes
    known_dialogues: List[str] = field(default_factory=list)
    quests_available: List[str] = field(default_factory=list)
    quests_completed: List[str] = field(default_factory=list)

    # Commerce
    sells_items: List[str] = field(default_factory=list)
    buys_items: List[str] = field(default_factory=list)
    price_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Métadonnées
    level_requirement: int = 1
    faction: Optional[str] = None
    importance_score: float = 1.0
    notes: str = ""

@dataclass
class NPCInteractionContext:
    """Contexte d'interaction avec PNJ"""
    npc_data: NPCData
    player_context: Dict[str, Any]
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    current_objective: Optional[str] = None
    urgency_level: int = 1  # 1=basse, 5=haute

class NPCDatabase:
    """Base de données des PNJ connus"""

    def __init__(self, data_file: str = "data/npcs/npc_database.json"):
        self.data_file = Path(data_file)
        self.npcs: Dict[str, NPCData] = {}
        self.location_index: Dict[str, List[str]] = {}  # map_id -> npc_ids
        self.type_index: Dict[NPCType, List[str]] = {}  # type -> npc_ids
        self.name_index: Dict[str, str] = {}  # normalized_name -> npc_id

        self.load_database()

    def load_database(self):
        """Charge la base de données des PNJ"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._load_npcs_from_dict(data)
            except Exception as e:
                logger.warning(f"Erreur chargement base PNJ: {e}")

        # Base par défaut si vide
        if not self.npcs:
            self._create_default_npcs()

        logger.info(f"Base PNJ chargée: {len(self.npcs)} PNJ")

    def _load_npcs_from_dict(self, data: Dict[str, Any]):
        """Charge PNJ depuis dictionnaire"""
        for npc_id, npc_data in data.get("npcs", {}).items():
            try:
                npc = NPCData(
                    npc_id=npc_id,
                    name=npc_data["name"],
                    npc_type=NPCType(npc_data.get("type", "unknown")),
                    location=tuple(npc_data.get("location", [0, 0])),
                    map_id=npc_data.get("map_id"),
                    available_interactions=[InteractionType(t) for t in npc_data.get("interactions", [])],
                    quests_available=npc_data.get("quests_available", []),
                    sells_items=npc_data.get("sells_items", []),
                    buys_items=npc_data.get("buys_items", []),
                    level_requirement=npc_data.get("level_requirement", 1),
                    faction=npc_data.get("faction"),
                    importance_score=npc_data.get("importance", 1.0),
                    notes=npc_data.get("notes", "")
                )

                self._add_npc_to_indices(npc)

            except Exception as e:
                logger.warning(f"Erreur chargement PNJ {npc_id}: {e}")

    def _create_default_npcs(self):
        """Crée PNJ par défaut pour Ganymède"""
        default_npcs = [
            {
                "npc_id": "gardien_ganymede",
                "name": "Gardien de Ganymède",
                "type": NPCType.QUEST_GIVER,
                "location": (350, 250),
                "map_id": "ganymede_center",
                "interactions": [InteractionType.TALK, InteractionType.QUEST_START],
                "importance": 5.0
            },
            {
                "npc_id": "marchand_novice",
                "name": "Marchand Novice",
                "type": NPCType.MERCHANT,
                "location": (400, 350),
                "map_id": "ganymede_center",
                "interactions": [InteractionType.TRADE],
                "sells_items": ["Pain", "Potion de Vie", "Parchemin de Rappel"]
            },
            {
                "npc_id": "maitre_armes",
                "name": "Maître d'Armes",
                "type": NPCType.TRAINER,
                "location": (300, 200),
                "map_id": "ganymede_center",
                "interactions": [InteractionType.TRAIN],
                "importance": 3.0
            }
        ]

        for npc_data in default_npcs:
            npc = NPCData(
                npc_id=npc_data["npc_id"],
                name=npc_data["name"],
                npc_type=npc_data["type"],
                location=npc_data["location"],
                map_id=npc_data["map_id"],
                available_interactions=npc_data["interactions"],
                sells_items=npc_data.get("sells_items", []),
                importance_score=npc_data.get("importance", 1.0)
            )
            self._add_npc_to_indices(npc)

    def _add_npc_to_indices(self, npc: NPCData):
        """Ajoute PNJ aux indices"""
        self.npcs[npc.npc_id] = npc

        # Index par location
        if npc.map_id:
            if npc.map_id not in self.location_index:
                self.location_index[npc.map_id] = []
            self.location_index[npc.map_id].append(npc.npc_id)

        # Index par type
        if npc.npc_type not in self.type_index:
            self.type_index[npc.npc_type] = []
        self.type_index[npc.npc_type].append(npc.npc_id)

        # Index par nom
        normalized_name = self._normalize_name(npc.name)
        self.name_index[normalized_name] = npc.npc_id

    def _normalize_name(self, name: str) -> str:
        """Normalise nom pour recherche"""
        return re.sub(r'[^\w\s]', '', name.lower()).strip()

    def find_npc_by_name(self, name: str) -> Optional[NPCData]:
        """Trouve PNJ par nom"""
        normalized = self._normalize_name(name)
        npc_id = self.name_index.get(normalized)
        return self.npcs.get(npc_id) if npc_id else None

    def find_npcs_by_type(self, npc_type: NPCType) -> List[NPCData]:
        """Trouve PNJ par type"""
        npc_ids = self.type_index.get(npc_type, [])
        return [self.npcs[npc_id] for npc_id in npc_ids if npc_id in self.npcs]

    def find_npcs_in_map(self, map_id: str) -> List[NPCData]:
        """Trouve PNJ dans une carte"""
        npc_ids = self.location_index.get(map_id, [])
        return [self.npcs[npc_id] for npc_id in npc_ids if npc_id in self.npcs]

    def find_nearest_npc(self, position: Tuple[int, int], map_id: str = None) -> Optional[NPCData]:
        """Trouve PNJ le plus proche"""
        candidates = []

        if map_id:
            candidates = self.find_npcs_in_map(map_id)
        else:
            candidates = list(self.npcs.values())

        if not candidates:
            return None

        min_distance = float('inf')
        nearest_npc = None

        for npc in candidates:
            distance = np.sqrt((position[0] - npc.location[0])**2 +
                             (position[1] - npc.location[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_npc = npc

        return nearest_npc

class NPCVisualClassifier:
    """Classificateur visuel pour PNJ"""

    def __init__(self):
        self.vision_engine = create_vision_engine()

        # Caractéristiques visuelles par type de PNJ
        self.type_signatures = {
            NPCType.QUEST_GIVER: {
                "size_range": (40, 80),
                "typical_colors": ["yellow", "gold", "white"],
                "indicators": ["!", "?", "exclamation"]
            },
            NPCType.MERCHANT: {
                "size_range": (35, 70),
                "typical_colors": ["brown", "green", "blue"],
                "indicators": ["bag", "coin", "shop"]
            },
            NPCType.GUARD: {
                "size_range": (45, 85),
                "typical_colors": ["red", "metal", "armor"],
                "indicators": ["sword", "shield", "armor"]
            },
            NPCType.BANK: {
                "size_range": (40, 75),
                "typical_colors": ["gold", "purple", "rich"],
                "indicators": ["vault", "money", "bank"]
            }
        }

    def classify_npc_from_appearance(self,
                                   screenshot: np.ndarray,
                                   npc_region: Tuple[int, int, int, int]) -> Tuple[NPCType, float]:
        """Classifie type de PNJ depuis apparence"""

        x, y, w, h = npc_region
        npc_image = screenshot[y:y+h, x:x+w]

        if npc_image.size == 0:
            return NPCType.UNKNOWN, 0.0

        # Analyser couleurs dominantes
        color_analysis = self._analyze_npc_colors(npc_image)

        # Analyser taille
        npc_size = max(w, h)

        # Chercher indicateurs visuels
        visual_indicators = self._detect_visual_indicators(npc_image)

        # Scorer chaque type
        type_scores = {}

        for npc_type, signature in self.type_signatures.items():
            score = 0.0

            # Score de taille
            size_min, size_max = signature["size_range"]
            if size_min <= npc_size <= size_max:
                score += 0.3

            # Score de couleur
            typical_colors = signature["typical_colors"]
            color_match = any(color in color_analysis for color in typical_colors)
            if color_match:
                score += 0.4

            # Score d'indicateurs
            indicators = signature["indicators"]
            indicator_match = any(indicator in visual_indicators for indicator in indicators)
            if indicator_match:
                score += 0.3

            type_scores[npc_type] = score

        # Meilleur type
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            return best_type[0], best_type[1]

        return NPCType.UNKNOWN, 0.0

    def _analyze_npc_colors(self, npc_image: np.ndarray) -> List[str]:
        """Analyse couleurs dominantes du PNJ"""
        if npc_image.size == 0:
            return []

        # Convertir en HSV
        hsv = cv2.cvtColor(npc_image, cv2.COLOR_RGB2HSV)

        # Analyser teinte dominante
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hue_hist)

        # Mapper teinte vers couleur
        color_mapping = {
            (0, 20): "red",
            (20, 40): "yellow",
            (40, 80): "green",
            (80, 130): "blue",
            (130, 160): "purple",
            (160, 180): "red"
        }

        detected_colors = []
        for (hue_min, hue_max), color_name in color_mapping.items():
            if hue_min <= dominant_hue <= hue_max:
                detected_colors.append(color_name)

        # Analyser saturation pour détecter métallique/or
        sat_mean = np.mean(hsv[:, :, 1])
        val_mean = np.mean(hsv[:, :, 2])

        if val_mean > 180 and sat_mean > 100:
            detected_colors.append("gold")
        elif val_mean > 150 and sat_mean < 50:
            detected_colors.append("metal")

        return detected_colors

    def _detect_visual_indicators(self, npc_image: np.ndarray) -> List[str]:
        """Détecte indicateurs visuels spéciaux"""
        indicators = []

        # Analyse basique de formes
        gray = cv2.cvtColor(npc_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Détecter formes géométriques (très basique)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Minimum size
                # Approximation polygonale
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 3:
                    indicators.append("triangle")
                elif len(approx) == 4:
                    indicators.append("rectangle")
                elif len(approx) > 8:
                    indicators.append("circle")

        return indicators

class NPCRecognition:
    """Système principal de reconnaissance PNJ"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Composants
        self.vision_engine = create_vision_engine()
        self.npc_database = NPCDatabase()
        self.visual_classifier = NPCVisualClassifier()
        self.hrm_model = create_hrm_model().to(self.device)

        # État de reconnaissance
        self.detected_npcs: Dict[str, NPCData] = {}
        self.recognition_cache: Dict[str, Dict[str, Any]] = {}

        # Statistiques
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.cache_hits = 0

        logger.info("NPCRecognition initialisé avec succès")

    def analyze_npcs_in_screenshot(self, screenshot: np.ndarray, map_id: str = None) -> List[NPCData]:
        """Analyse et reconnaît tous les PNJ dans une capture"""

        self.total_recognitions += 1

        try:
            # Analyse vision générale
            vision_results = self.vision_engine.analyze_screenshot(screenshot)
            sam_segments = vision_results.get("sam_segments", [])
            text_detections = vision_results.get("text_detections", [])

            # Filtrer segments potentiels de PNJ
            npc_candidates = self._filter_npc_candidates(sam_segments)

            # Analyser chaque candidat
            recognized_npcs = []

            for candidate in npc_candidates:
                npc_data = self._analyze_npc_candidate(
                    screenshot, candidate, text_detections, map_id
                )

                if npc_data:
                    recognized_npcs.append(npc_data)
                    self.detected_npcs[npc_data.npc_id] = npc_data

            if recognized_npcs:
                self.successful_recognitions += 1

            return recognized_npcs

        except Exception as e:
            logger.error(f"Erreur reconnaissance PNJ: {e}")
            return []

    def _filter_npc_candidates(self, segments: List[SAMSegment]) -> List[SAMSegment]:
        """Filtre segments susceptibles d'être des PNJ"""
        npc_candidates = []

        for segment in segments:
            # Critères de taille (PNJ typiques)
            if 20 <= segment.width <= 150 and 20 <= segment.height <= 150:
                # Ratio d'aspect approximativement vertical
                aspect_ratio = segment.height / max(segment.width, 1)
                if 0.8 <= aspect_ratio <= 2.0:
                    npc_candidates.append(segment)

        return npc_candidates

    def _analyze_npc_candidate(self,
                             screenshot: np.ndarray,
                             segment: SAMSegment,
                             text_detections: List[TextDetection],
                             map_id: str = None) -> Optional[NPCData]:
        """Analyse candidat PNJ individuel"""

        # Position et taille
        npc_region = (segment.x, segment.y, segment.width, segment.height)
        npc_position = (segment.x + segment.width//2, segment.y + segment.height//2)

        # Chercher PNJ connu à proximité
        known_npc = self.npc_database.find_nearest_npc(npc_position, map_id)
        if known_npc:
            distance = np.sqrt((npc_position[0] - known_npc.location[0])**2 +
                             (npc_position[1] - known_npc.location[1])**2)
            if distance < 50:  # Tolerance de 50 pixels
                # Mettre à jour position et timestamp
                known_npc.location = npc_position
                known_npc.last_seen = time.time()
                return known_npc

        # Classification visuelle
        npc_type, confidence = self.visual_classifier.classify_npc_from_appearance(
            screenshot, npc_region
        )

        if confidence < 0.3:
            return None

        # Chercher nom dans le texte
        npc_name = self._find_npc_name_in_text(npc_position, text_detections)

        # Créer nouveau PNJ
        npc_id = f"npc_{int(time.time())}_{hash(str(npc_position))}"

        new_npc = NPCData(
            npc_id=npc_id,
            name=npc_name or f"PNJ {npc_type.value}",
            npc_type=npc_type,
            location=npc_position,
            map_id=map_id,
            last_seen=time.time()
        )

        # Inférer interactions possibles
        new_npc.available_interactions = self._infer_interactions(npc_type)

        return new_npc

    def _find_npc_name_in_text(self,
                             npc_position: Tuple[int, int],
                             text_detections: List[TextDetection]) -> Optional[str]:
        """Trouve nom de PNJ dans les détections de texte"""

        max_distance = 80  # Distance max pour associer texte à PNJ

        for detection in text_detections:
            text_center = (detection.bbox[0] + detection.bbox[2]//2,
                          detection.bbox[1] + detection.bbox[3]//2)

            distance = np.sqrt((npc_position[0] - text_center[0])**2 +
                             (npc_position[1] - text_center[1])**2)

            if distance <= max_distance:
                # Vérifier si le texte ressemble à un nom
                if self._is_likely_npc_name(detection.text):
                    return detection.text

        return None

    def _is_likely_npc_name(self, text: str) -> bool:
        """Vérifie si texte ressemble à un nom de PNJ"""
        # Noms commencent par majuscule
        if not text[0].isupper():
            return False

        # Longueur raisonnable
        if len(text) < 3 or len(text) > 30:
            return False

        # Pas que des chiffres
        if text.isdigit():
            return False

        # Mots français/anglais probables
        common_npc_words = [
            "gardien", "marchand", "maître", "apprenti", "chef",
            "garde", "capitaine", "sergent", "keeper", "master",
            "merchant", "trainer", "guard"
        ]

        text_lower = text.lower()
        if any(word in text_lower for word in common_npc_words):
            return True

        # Pattern de nom propre
        if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', text):
            return True

        return False

    def _infer_interactions(self, npc_type: NPCType) -> List[InteractionType]:
        """Infère interactions possibles selon type PNJ"""

        interaction_mapping = {
            NPCType.QUEST_GIVER: [InteractionType.TALK, InteractionType.QUEST_START, InteractionType.QUEST_COMPLETE],
            NPCType.MERCHANT: [InteractionType.TALK, InteractionType.TRADE],
            NPCType.BANK: [InteractionType.TALK, InteractionType.BANK],
            NPCType.TRAINER: [InteractionType.TALK, InteractionType.TRAIN],
            NPCType.GUARD: [InteractionType.TALK],
            NPCType.ZAAP_KEEPER: [InteractionType.TALK, InteractionType.TRANSPORT],
            NPCType.GUILD_MASTER: [InteractionType.TALK],
            NPCType.CRAFT_MASTER: [InteractionType.TALK, InteractionType.CRAFT],
            NPCType.TRANSPORT: [InteractionType.TALK, InteractionType.TRANSPORT],
            NPCType.VILLAGER: [InteractionType.TALK],
            NPCType.UNKNOWN: [InteractionType.TALK]
        }

        return interaction_mapping.get(npc_type, [InteractionType.TALK])

    def get_interaction_context(self, npc: NPCData, player_context: Dict[str, Any]) -> NPCInteractionContext:
        """Crée contexte d'interaction avec PNJ"""

        context = NPCInteractionContext(
            npc_data=npc,
            player_context=player_context
        )

        # Déterminer objectif probable
        context.current_objective = self._determine_interaction_objective(npc, player_context)

        # Niveau d'urgence
        context.urgency_level = self._calculate_urgency_level(npc, player_context)

        return context

    def _determine_interaction_objective(self, npc: NPCData, player_context: Dict[str, Any]) -> Optional[str]:
        """Détermine objectif probable d'interaction"""

        # Quête en cours
        current_quests = player_context.get("current_quests", [])
        for quest_id in current_quests:
            if quest_id in npc.quests_available:
                return f"complete_quest_{quest_id}"

        # Quête disponible
        if npc.quests_available and InteractionType.QUEST_START in npc.available_interactions:
            return "start_quest"

        # Commerce si marchand et inventaire plein
        if (npc.npc_type == NPCType.MERCHANT and
            player_context.get("inventory_full", False)):
            return "sell_items"

        # Banque si beaucoup de kamas
        if (npc.npc_type == NPCType.BANK and
            player_context.get("kamas", 0) > 10000):
            return "bank_kamas"

        return None

    def _calculate_urgency_level(self, npc: NPCData, player_context: Dict[str, Any]) -> int:
        """Calcule niveau d'urgence d'interaction"""

        urgency = 1

        # Quête urgente
        if npc.quests_available:
            urgency += 2

        # PNJ important
        if npc.importance_score > 3.0:
            urgency += 1

        # Inventaire plein
        if player_context.get("inventory_full", False) and npc.npc_type == NPCType.MERCHANT:
            urgency += 2

        return min(5, urgency)

    def get_best_interaction_npc(self,
                               npcs: List[NPCData],
                               player_context: Dict[str, Any]) -> Optional[NPCData]:
        """Trouve meilleur PNJ pour interaction selon contexte"""

        if not npcs:
            return None

        scored_npcs = []

        for npc in npcs:
            score = self._score_npc_interaction_value(npc, player_context)
            scored_npcs.append((npc, score))

        # Trier par score
        scored_npcs.sort(key=lambda x: x[1], reverse=True)

        return scored_npcs[0][0] if scored_npcs[0][1] > 0 else None

    def _score_npc_interaction_value(self, npc: NPCData, player_context: Dict[str, Any]) -> float:
        """Score valeur d'interaction avec PNJ"""

        score = 0.0

        # Score base par importance
        score += npc.importance_score

        # Bonus quêtes
        current_quests = player_context.get("current_quests", [])
        completed_quests = player_context.get("completed_quests", [])

        for quest_id in npc.quests_available:
            if quest_id in current_quests:
                score += 5.0  # Compléter quête
            elif quest_id not in completed_quests:
                score += 3.0  # Nouvelle quête

        # Bonus commerce si besoin
        if npc.npc_type == NPCType.MERCHANT:
            if player_context.get("inventory_full", False):
                score += 2.0
            if player_context.get("need_items", []):
                score += 1.0

        # Bonus services spéciaux
        if npc.npc_type == NPCType.BANK and player_context.get("kamas", 0) > 5000:
            score += 1.5

        if npc.npc_type == NPCType.TRAINER and player_context.get("level_up_available", False):
            score += 2.0

        return score

    def update_npc_state(self, npc_id: str, new_state: NPCState):
        """Met à jour état d'un PNJ"""
        if npc_id in self.detected_npcs:
            self.detected_npcs[npc_id].current_state = new_state

    def get_recognition_stats(self) -> Dict[str, Any]:
        """Statistiques de reconnaissance"""

        success_rate = (self.successful_recognitions / max(self.total_recognitions, 1)) * 100

        npc_types_count = {}
        for npc in self.detected_npcs.values():
            npc_type = npc.npc_type.value
            npc_types_count[npc_type] = npc_types_count.get(npc_type, 0) + 1

        return {
            "total_recognitions": self.total_recognitions,
            "successful_recognitions": self.successful_recognitions,
            "success_rate": success_rate,
            "cache_hits": self.cache_hits,
            "detected_npcs": len(self.detected_npcs),
            "known_npcs": len(self.npc_database.npcs),
            "npc_types_detected": npc_types_count
        }

def create_npc_recognition() -> NPCRecognition:
    """Factory function pour créer un NPCRecognition"""
    return NPCRecognition()