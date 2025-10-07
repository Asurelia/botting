#!/usr/bin/env python3
"""
DialogueSystem - Système de gestion des dialogues et interactions PNJ
Utilise HRM pour prendre des décisions contextuelles dans les dialogues
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum

import torch
import numpy as np

from config import config
from core.hrm_reasoning import create_hrm_model, HRMOutput
from core.vision_engine_v2 import create_vision_engine, TextDetection

logger = logging.getLogger(__name__)

class DialogueState(Enum):
    """États du dialogue"""
    CLOSED = "closed"
    OPENING = "opening"
    WAITING_RESPONSE = "waiting_response"
    CHOOSING_OPTION = "choosing_option"
    READING_TEXT = "reading_text"
    QUEST_DIALOGUE = "quest_dialogue"
    SHOP_DIALOGUE = "shop_dialogue"
    ERROR = "error"

class NPCType(Enum):
    """Types de PNJ"""
    QUEST_GIVER = "quest_giver"
    MERCHANT = "merchant"
    TRAINER = "trainer"
    GUARD = "guard"
    VILLAGER = "villager"
    BANK = "bank"
    GUILD_MASTER = "guild_master"
    ZAAP = "zaap"
    UNKNOWN = "unknown"

class DialogueActionType(Enum):
    """Types d'actions de dialogue"""
    SELECT_OPTION = "select_option"
    CONTINUE = "continue"
    CLOSE = "close"
    WAIT = "wait"
    ACCEPT_QUEST = "accept_quest"
    DECLINE_QUEST = "decline_quest"
    COMPLETE_QUEST = "complete_quest"
    BUY_ITEM = "buy_item"
    SELL_ITEM = "sell_item"

@dataclass
class DialogueChoice:
    """Choix de dialogue"""
    choice_id: int
    text: str
    action_type: DialogueActionType = DialogueActionType.SELECT_OPTION
    leads_to_quest: bool = False
    completes_quest: bool = False
    requires_items: List[str] = field(default_factory=list)
    cost: int = 0
    reward: Optional[str] = None
    priority: int = 0  # Plus élevé = plus prioritaire

@dataclass
class NPCInteraction:
    """Interaction avec un PNJ"""
    npc_name: str
    npc_type: NPCType
    location: Optional[Tuple[int, int]] = None
    dialogue_state: DialogueState = DialogueState.CLOSED
    available_choices: List[DialogueChoice] = field(default_factory=list)
    dialogue_history: List[str] = field(default_factory=list)
    last_interaction: float = 0.0

    # Contexte de quête
    related_quests: List[str] = field(default_factory=list)
    can_give_quest: bool = False
    can_complete_quest: bool = False

    # Contexte marchand
    can_trade: bool = False
    shop_category: Optional[str] = None

@dataclass
class DialogueContext:
    """Contexte pour prise de décision de dialogue"""
    current_quests: List[str] = field(default_factory=list)
    completed_quests: List[str] = field(default_factory=list)
    player_level: int = 1
    player_kamas: int = 0
    player_inventory: List[str] = field(default_factory=list)
    current_objective: Optional[str] = None
    urgency_level: int = 1  # 1=basse, 5=haute
    auto_mode: bool = True

class DialogueTextAnalyzer:
    """Analyseur de texte de dialogue DOFUS"""

    def __init__(self):
        # Patterns de reconnaissance de dialogue
        self.quest_patterns = {
            "quest_available": [
                r"j'ai une mission pour toi",
                r"peux-tu m'aider",
                r"j'aurais besoin de ton aide",
                r"acceptes-tu cette mission",
                r"veux-tu m'aider"
            ],
            "quest_complete": [
                r"bravo.*mission accomplie",
                r"parfait.*voici ta récompense",
                r"excellent travail",
                r"mission terminée",
                r"félicitations"
            ],
            "quest_in_progress": [
                r"as-tu.*déjà",
                r"où en es-tu",
                r"la mission avance",
                r"n'oublie pas de"
            ]
        }

        self.choice_patterns = [
            r"^(\d+)\.\s*(.*)",  # "1. Option de choix"
            r"^\[(\d+)\]\s*(.*)",  # "[1] Option de choix"
            r"^-\s*(\d+)\s*-\s*(.*)",  # "- 1 - Option"
        ]

        self.merchant_patterns = [
            r"que veux-tu acheter",
            r"voici mes marchandises",
            r"j'ai de belles choses",
            r"prix.*kamas",
            r"acheter.*vendre"
        ]

    def analyze_dialogue_text(self, text_detections: List[TextDetection]) -> Dict[str, Any]:
        """Analyse le texte de dialogue détecté"""
        dialogue_info = {
            "dialogue_type": "unknown",
            "npc_name": None,
            "dialogue_text": [],
            "choices": [],
            "quest_related": False,
            "merchant_related": False
        }

        # Trier par position verticale (haut vers bas)
        sorted_detections = sorted(text_detections, key=lambda d: d.bbox[1])

        for detection in sorted_detections:
            text = detection.text.strip()

            if not text:
                continue

            # Détecter nom de PNJ (généralement en haut, avec majuscule)
            if self._is_npc_name(text, detection):
                dialogue_info["npc_name"] = text
                continue

            # Détecter choix de dialogue
            choice = self._extract_dialogue_choice(text)
            if choice:
                dialogue_info["choices"].append(choice)
                continue

            # Ajouter au texte de dialogue
            dialogue_info["dialogue_text"].append(text)

            # Analyser le type de dialogue
            if self._is_quest_dialogue(text):
                dialogue_info["quest_related"] = True
                if any(pattern in text.lower() for patterns in self.quest_patterns.values() for pattern in patterns):
                    dialogue_info["dialogue_type"] = "quest"

            elif self._is_merchant_dialogue(text):
                dialogue_info["merchant_related"] = True
                dialogue_info["dialogue_type"] = "merchant"

        return dialogue_info

    def _is_npc_name(self, text: str, detection: TextDetection) -> bool:
        """Détermine si le texte est un nom de PNJ"""
        # Nom généralement en haut de l'écran, commence par majuscule
        bbox = detection.bbox
        is_top_area = bbox[1] < 200  # Haut de l'écran
        is_capitalized = text[0].isupper() if text else False
        is_short = len(text.split()) <= 3  # Nom court

        return is_top_area and is_capitalized and is_short

    def _extract_dialogue_choice(self, text: str) -> Optional[DialogueChoice]:
        """Extrait un choix de dialogue depuis le texte"""
        for pattern in self.choice_patterns:
            match = re.match(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    choice_id, choice_text = match.groups()

                    # Analyser le type d'action
                    action_type = DialogueActionType.SELECT_OPTION
                    priority = 0

                    choice_lower = choice_text.lower()

                    if any(word in choice_lower for word in ['accepter', 'oui', 'j\'accepte']):
                        action_type = DialogueActionType.ACCEPT_QUEST
                        priority = 5
                    elif any(word in choice_lower for word in ['refuser', 'non', 'plus tard']):
                        action_type = DialogueActionType.DECLINE_QUEST
                        priority = 1
                    elif any(word in choice_lower for word in ['terminer', 'fini', 'accompli']):
                        action_type = DialogueActionType.COMPLETE_QUEST
                        priority = 5
                    elif any(word in choice_lower for word in ['acheter', 'achat']):
                        action_type = DialogueActionType.BUY_ITEM
                        priority = 2
                    elif any(word in choice_lower for word in ['vendre', 'vente']):
                        action_type = DialogueActionType.SELL_ITEM
                        priority = 2

                    return DialogueChoice(
                        choice_id=int(choice_id),
                        text=choice_text.strip(),
                        action_type=action_type,
                        priority=priority
                    )

        return None

    def _is_quest_dialogue(self, text: str) -> bool:
        """Détermine si le texte est lié à une quête"""
        quest_keywords = [
            'mission', 'quête', 'tâche', 'aide', 'service',
            'chercher', 'trouver', 'apporter', 'tuer', 'éliminer',
            'récompense', 'merci', 'bravo', 'félicitations'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in quest_keywords)

    def _is_merchant_dialogue(self, text: str) -> bool:
        """Détermine si le texte est lié au commerce"""
        merchant_keywords = [
            'acheter', 'vendre', 'marchandises', 'prix', 'kamas',
            'boutique', 'magasin', 'commerce', 'article', 'stock'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in merchant_keywords)

class DialogueDecisionEngine:
    """Moteur de décision pour dialogues utilisant HRM"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hrm_model = create_hrm_model().to(self.device)

        # Cache de décisions
        self.decision_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = 30.0  # 30 secondes

    def make_dialogue_decision(self,
                             interaction: NPCInteraction,
                             context: DialogueContext,
                             dialogue_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prend une décision de dialogue basée sur le contexte"""

        # Créer clé de cache
        cache_key = self._create_cache_key(interaction, context, dialogue_info)

        # Vérifier cache
        if cache_key in self.decision_cache:
            cached = self.decision_cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_duration:
                return cached["decision"]

        try:
            # Analyser les choix disponibles
            choices = dialogue_info.get("choices", [])

            if not choices:
                # Pas de choix, continuer ou fermer
                return self._handle_no_choices(interaction, context, dialogue_info)

            # Évaluer chaque choix
            choice_scores = []
            for choice in choices:
                score = self._evaluate_choice(choice, interaction, context)
                choice_scores.append((choice, score))

            # Trier par score
            choice_scores.sort(key=lambda x: x[1], reverse=True)
            best_choice, best_score = choice_scores[0]

            # Créer la décision
            decision = {
                "action": "select_choice",
                "choice_id": best_choice.choice_id,
                "choice_text": best_choice.text,
                "action_type": best_choice.action_type.value,
                "score": best_score,
                "reasoning": self._explain_choice(best_choice, interaction, context),
                "confidence": min(1.0, best_score / 10.0)
            }

            # Mettre en cache
            self.decision_cache[cache_key] = {
                "decision": decision,
                "timestamp": time.time()
            }

            return decision

        except Exception as e:
            logger.error(f"Erreur décision dialogue: {e}")
            return {
                "action": "wait",
                "reason": f"Erreur: {e}",
                "confidence": 0.0
            }

    def _handle_no_choices(self,
                          interaction: NPCInteraction,
                          context: DialogueContext,
                          dialogue_info: Dict[str, Any]) -> Dict[str, Any]:
        """Gère les dialogues sans choix explicites"""

        dialogue_text = " ".join(dialogue_info.get("dialogue_text", []))

        # Si c'est un dialogue de quête et on a des quêtes en cours
        if dialogue_info.get("quest_related") and context.current_quests:
            return {
                "action": "continue",
                "reason": "Dialogue de quête en cours",
                "confidence": 0.8
            }

        # Si dialogue terminé naturellement
        if any(word in dialogue_text.lower() for word in ['merci', 'au revoir', 'à bientôt']):
            return {
                "action": "close",
                "reason": "Dialogue terminé",
                "confidence": 0.9
            }

        # Par défaut, continuer
        return {
            "action": "continue",
            "reason": "Continuer le dialogue",
            "confidence": 0.5
        }

    def _evaluate_choice(self,
                        choice: DialogueChoice,
                        interaction: NPCInteraction,
                        context: DialogueContext) -> float:
        """Évalue un choix de dialogue (score 0-10)"""

        score = choice.priority  # Score de base

        # Bonus pour actions de quête
        if choice.action_type == DialogueActionType.ACCEPT_QUEST:
            if context.auto_mode and not context.current_quests:
                score += 5  # Bonus si pas de quête active
            elif len(context.current_quests) >= 3:
                score -= 3  # Malus si trop de quêtes

        elif choice.action_type == DialogueActionType.COMPLETE_QUEST:
            if any(quest in interaction.related_quests for quest in context.current_quests):
                score += 8  # Gros bonus pour compléter quête active

        elif choice.action_type == DialogueActionType.DECLINE_QUEST:
            if len(context.current_quests) >= 5:
                score += 2  # Bonus si déjà surchargé
            else:
                score -= 2  # Malus sinon

        # Bonus pour commerce si on a des kamas
        elif choice.action_type == DialogueActionType.BUY_ITEM:
            if context.player_kamas > 1000:
                score += 1
            else:
                score -= 2

        # Contexte d'urgence
        score += context.urgency_level

        # Bonus pour objectif actuel
        if context.current_objective:
            choice_lower = choice.text.lower()
            objective_lower = context.current_objective.lower()

            # Correspondance de mots-clés
            objective_words = objective_lower.split()
            matching_words = sum(1 for word in objective_words if word in choice_lower)
            score += matching_words * 2

        return max(0.0, score)

    def _explain_choice(self,
                       choice: DialogueChoice,
                       interaction: NPCInteraction,
                       context: DialogueContext) -> str:
        """Explique pourquoi ce choix a été sélectionné"""

        reasons = []

        if choice.action_type == DialogueActionType.ACCEPT_QUEST:
            reasons.append("Accepter nouvelle quête")
            if not context.current_quests:
                reasons.append("Aucune quête active")

        elif choice.action_type == DialogueActionType.COMPLETE_QUEST:
            reasons.append("Compléter quête en cours")

        elif choice.priority > 3:
            reasons.append("Choix prioritaire")

        if context.current_objective:
            if any(word in choice.text.lower() for word in context.current_objective.lower().split()):
                reasons.append("Correspond à l'objectif actuel")

        if not reasons:
            reasons.append("Meilleur choix disponible")

        return " | ".join(reasons)

    def _create_cache_key(self,
                         interaction: NPCInteraction,
                         context: DialogueContext,
                         dialogue_info: Dict[str, Any]) -> str:
        """Crée une clé de cache pour la décision"""

        key_components = [
            interaction.npc_name,
            interaction.npc_type.value,
            str(len(dialogue_info.get("choices", []))),
            str(len(context.current_quests)),
            str(context.urgency_level),
            str(context.auto_mode)
        ]

        # Ajouter texte des choix (simplifié)
        choices_text = "|".join([
            f"{c.choice_id}:{c.action_type.value}"
            for c in dialogue_info.get("choices", [])
        ])
        key_components.append(choices_text)

        return "_".join(key_components)

class DialogueSystem:
    """Système principal de gestion des dialogues"""

    def __init__(self):
        self.vision_engine = create_vision_engine()
        self.text_analyzer = DialogueTextAnalyzer()
        self.decision_engine = DialogueDecisionEngine()

        # État du système
        self.current_interaction: Optional[NPCInteraction] = None
        self.dialogue_context = DialogueContext()

        # Cache d'interactions
        self.known_npcs: Dict[str, NPCInteraction] = {}

        # Statistiques
        self.total_interactions = 0
        self.successful_decisions = 0
        self.last_decision_time = 0.0

        logger.info("DialogueSystem initialisé avec succès")

    def update_context(self, **kwargs):
        """Met à jour le contexte de dialogue"""
        for key, value in kwargs.items():
            if hasattr(self.dialogue_context, key):
                setattr(self.dialogue_context, key, value)

    def analyze_dialogue_screen(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyse l'écran pour détecter les dialogues"""
        try:
            # Analyser avec le moteur de vision
            vision_results = self.vision_engine.analyze_screenshot(screenshot)
            text_detections = vision_results.get("text_detections", [])

            # Filtrer les détections dans la zone de dialogue
            h, w = screenshot.shape[:2]
            dialogue_detections = []

            for detection in text_detections:
                bbox = detection.bbox
                # Zone de dialogue généralement au centre
                if (0.2 * w < bbox[0] < 0.8 * w and
                    0.3 * h < bbox[1] < 0.8 * h):
                    dialogue_detections.append(detection)

            if not dialogue_detections:
                return {"dialogue_active": False}

            # Analyser le texte de dialogue
            dialogue_info = self.text_analyzer.analyze_dialogue_text(dialogue_detections)
            dialogue_info["dialogue_active"] = True

            return dialogue_info

        except Exception as e:
            logger.error(f"Erreur analyse dialogue: {e}")
            return {"dialogue_active": False, "error": str(e)}

    def handle_dialogue_interaction(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Gère une interaction de dialogue complète"""

        # Analyser l'écran
        dialogue_info = self.analyze_dialogue_screen(screenshot)

        if not dialogue_info.get("dialogue_active"):
            # Pas de dialogue actif
            self.current_interaction = None
            return {"status": "no_dialogue"}

        # Créer ou mettre à jour l'interaction
        npc_name = dialogue_info.get("npc_name", "Unknown")

        if npc_name not in self.known_npcs:
            # Nouveau PNJ
            self.known_npcs[npc_name] = NPCInteraction(
                npc_name=npc_name,
                npc_type=self._detect_npc_type(dialogue_info),
                dialogue_state=DialogueState.WAITING_RESPONSE
            )

        interaction = self.known_npcs[npc_name]
        self.current_interaction = interaction

        # Mettre à jour l'état
        interaction.last_interaction = time.time()
        interaction.dialogue_history.extend(dialogue_info.get("dialogue_text", []))

        # Créer les choix
        choices = []
        for choice_data in dialogue_info.get("choices", []):
            if isinstance(choice_data, dict):
                choice = DialogueChoice(
                    choice_id=choice_data["choice_id"],
                    text=choice_data["text"],
                    action_type=DialogueActionType(choice_data.get("action_type", "select_option"))
                )
            else:
                choice = choice_data
            choices.append(choice)

        interaction.available_choices = choices

        # Prendre une décision
        decision = self.decision_engine.make_dialogue_decision(
            interaction, self.dialogue_context, dialogue_info
        )

        self.total_interactions += 1
        if decision.get("confidence", 0) > 0.7:
            self.successful_decisions += 1
            self.last_decision_time = time.time()

        return {
            "status": "dialogue_active",
            "npc_name": npc_name,
            "npc_type": interaction.npc_type.value,
            "decision": decision,
            "dialogue_info": dialogue_info,
            "choices_count": len(choices)
        }

    def _detect_npc_type(self, dialogue_info: Dict[str, Any]) -> NPCType:
        """Détecte le type de PNJ basé sur le dialogue"""

        dialogue_text = " ".join(dialogue_info.get("dialogue_text", [])).lower()

        # Mots-clés pour différents types
        if any(word in dialogue_text for word in ['mission', 'quête', 'aide', 'service']):
            return NPCType.QUEST_GIVER
        elif any(word in dialogue_text for word in ['acheter', 'vendre', 'marchandises', 'boutique']):
            return NPCType.MERCHANT
        elif any(word in dialogue_text for word in ['entraîner', 'apprendre', 'sort', 'technique']):
            return NPCType.TRAINER
        elif any(word in dialogue_text for word in ['banque', 'coffre', 'dépôt']):
            return NPCType.BANK
        elif any(word in dialogue_text for word in ['guilde', 'alliance']):
            return NPCType.GUILD_MASTER
        elif any(word in dialogue_text for word in ['transport', 'voyage', 'zaap']):
            return NPCType.ZAAP
        elif any(word in dialogue_text for word in ['garde', 'sécurité', 'ordre']):
            return NPCType.GUARD
        else:
            return NPCType.VILLAGER

    def get_interaction_stats(self) -> Dict[str, Any]:
        """Statistiques des interactions"""
        success_rate = (self.successful_decisions / max(self.total_interactions, 1)) * 100

        return {
            "total_interactions": self.total_interactions,
            "successful_decisions": self.successful_decisions,
            "success_rate": success_rate,
            "known_npcs": len(self.known_npcs),
            "current_interaction": self.current_interaction.npc_name if self.current_interaction else None,
            "last_decision": self.last_decision_time
        }

    def get_npc_history(self, npc_name: str) -> Optional[NPCInteraction]:
        """Récupère l'historique d'interaction avec un PNJ"""
        return self.known_npcs.get(npc_name)

    def clear_npc_cache(self):
        """Vide le cache des PNJ connus"""
        self.known_npcs.clear()
        logger.info("Cache PNJ vidé")

def create_dialogue_system() -> DialogueSystem:
    """Factory function pour créer un DialogueSystem"""
    return DialogueSystem()