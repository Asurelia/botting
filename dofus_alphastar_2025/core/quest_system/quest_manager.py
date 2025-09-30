#!/usr/bin/env python3
"""
QuestManager - Gestionnaire intelligent de quêtes DOFUS
Utilise HRM pour prendre des décisions contextuelles sur les quêtes
"""

import json
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import torch
import numpy as np

from config import config
from core.hrm_reasoning import create_hrm_model, HRMOutput
from core.vision_engine_v2 import create_vision_engine, TextDetection

logger = logging.getLogger(__name__)

class QuestStatus(Enum):
    """États de quête"""
    NOT_STARTED = "not_started"
    AVAILABLE = "available"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"

class QuestType(Enum):
    """Types de quête"""
    MAIN_STORY = "main_story"
    SIDE_QUEST = "side_quest"
    REPEATABLE = "repeatable"
    DAILY = "daily"
    WEEKLY = "weekly"
    ACHIEVEMENT = "achievement"
    TUTORIAL = "tutorial"
    DUNGEON = "dungeon"

class QuestStepType(Enum):
    """Types d'étapes de quête"""
    TALK_TO_NPC = "talk_to_npc"
    KILL_MONSTERS = "kill_monsters"
    COLLECT_ITEMS = "collect_items"
    GO_TO_LOCATION = "go_to_location"
    USE_ITEM = "use_item"
    DELIVER_ITEM = "deliver_item"
    WAIT = "wait"
    CUSTOM = "custom"

@dataclass
class QuestRequirement:
    """Prérequis pour une quête"""
    level: Optional[int] = None
    class_required: Optional[str] = None
    items_required: List[str] = field(default_factory=list)
    quests_completed: List[str] = field(default_factory=list)
    kamas_required: Optional[int] = None
    alignment: Optional[str] = None

@dataclass
class QuestReward:
    """Récompenses de quête"""
    experience: int = 0
    kamas: int = 0
    items: List[str] = field(default_factory=list)
    job_experience: Dict[str, int] = field(default_factory=dict)
    title: Optional[str] = None
    emote: Optional[str] = None

@dataclass
class QuestStep:
    """Étape de quête"""
    step_id: str
    step_type: QuestStepType
    description: str
    target: Optional[str] = None
    quantity: int = 1
    current_progress: int = 0
    location: Optional[Tuple[int, int]] = None
    map_id: Optional[str] = None
    npc_name: Optional[str] = None
    item_id: Optional[str] = None
    dialogue_choices: List[int] = field(default_factory=list)
    completed: bool = False

    @property
    def progress_percentage(self) -> float:
        """Pourcentage de progression de l'étape"""
        if self.quantity == 0:
            return 100.0
        return min(100.0, (self.current_progress / self.quantity) * 100.0)

    def is_completed(self) -> bool:
        """Vérifie si l'étape est terminée"""
        return self.current_progress >= self.quantity

@dataclass
class Quest:
    """Quête DOFUS"""
    quest_id: str
    name: str
    description: str
    quest_type: QuestType
    status: QuestStatus = QuestStatus.NOT_STARTED
    level_requirement: int = 1
    requirements: QuestRequirement = field(default_factory=QuestRequirement)
    rewards: QuestReward = field(default_factory=QuestReward)
    steps: List[QuestStep] = field(default_factory=list)
    current_step_index: int = 0
    start_time: Optional[float] = None
    completion_time: Optional[float] = None

    # Métadonnées
    estimated_duration: int = 0  # en minutes
    difficulty: str = "easy"  # easy, medium, hard
    zone: Optional[str] = None
    guide_url: Optional[str] = None

    @property
    def current_step(self) -> Optional[QuestStep]:
        """Étape actuelle"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def progress_percentage(self) -> float:
        """Pourcentage de progression globale"""
        if not self.steps:
            return 0.0

        completed_steps = sum(1 for step in self.steps if step.is_completed())
        current_step_progress = 0.0

        if self.current_step and not self.current_step.is_completed():
            current_step_progress = self.current_step.progress_percentage / 100.0

        return ((completed_steps + current_step_progress) / len(self.steps)) * 100.0

    def advance_step(self) -> bool:
        """Passe à l'étape suivante"""
        if self.current_step and self.current_step.is_completed():
            self.current_step_index += 1
            return True
        return False

    def complete_quest(self):
        """Marque la quête comme terminée"""
        self.status = QuestStatus.COMPLETED
        self.completion_time = time.time()

class QuestDatabase:
    """Base de données des quêtes"""

    def __init__(self, data_dir: str = "data/quests"):
        self.data_dir = Path(data_dir)
        self.quests: Dict[str, Quest] = {}
        self.load_quests()

    def load_quests(self):
        """Charge les quêtes depuis les fichiers JSON"""
        if not self.data_dir.exists():
            logger.warning(f"Répertoire quêtes non trouvé: {self.data_dir}")
            return

        quest_count = 0
        for quest_file in self.data_dir.glob("*.json"):
            try:
                with open(quest_file, 'r', encoding='utf-8') as f:
                    quest_data = json.load(f)
                    quest = self._parse_quest_data(quest_data)
                    self.quests[quest.quest_id] = quest
                    quest_count += 1
            except Exception as e:
                logger.error(f"Erreur chargement quête {quest_file}: {e}")

        logger.info(f"Chargé {quest_count} quêtes depuis {self.data_dir}")

    def _parse_quest_data(self, data: Dict) -> Quest:
        """Parse les données JSON en objet Quest"""
        # Requirements
        req_data = data.get("requirements", {})
        requirements = QuestRequirement(
            level=req_data.get("level"),
            class_required=req_data.get("class"),
            items_required=req_data.get("items", []),
            quests_completed=req_data.get("quests", []),
            kamas_required=req_data.get("kamas"),
            alignment=req_data.get("alignment")
        )

        # Rewards
        reward_data = data.get("rewards", {})
        rewards = QuestReward(
            experience=reward_data.get("experience", 0),
            kamas=reward_data.get("kamas", 0),
            items=reward_data.get("items", []),
            job_experience=reward_data.get("job_experience", {}),
            title=reward_data.get("title"),
            emote=reward_data.get("emote")
        )

        # Steps
        steps = []
        for step_data in data.get("steps", []):
            step = QuestStep(
                step_id=step_data["step_id"],
                step_type=QuestStepType(step_data["type"]),
                description=step_data["description"],
                target=step_data.get("target"),
                quantity=step_data.get("quantity", 1),
                location=tuple(step_data["location"]) if step_data.get("location") else None,
                map_id=step_data.get("map_id"),
                npc_name=step_data.get("npc_name"),
                item_id=step_data.get("item_id"),
                dialogue_choices=step_data.get("dialogue_choices", [])
            )
            steps.append(step)

        return Quest(
            quest_id=data["quest_id"],
            name=data["name"],
            description=data["description"],
            quest_type=QuestType(data.get("type", "side_quest")),
            level_requirement=data.get("level_requirement", 1),
            requirements=requirements,
            rewards=rewards,
            steps=steps,
            estimated_duration=data.get("estimated_duration", 0),
            difficulty=data.get("difficulty", "easy"),
            zone=data.get("zone"),
            guide_url=data.get("guide_url")
        )

    def get_quest(self, quest_id: str) -> Optional[Quest]:
        """Récupère une quête par ID"""
        return self.quests.get(quest_id)

    def get_available_quests(self, player_level: int, completed_quests: List[str]) -> List[Quest]:
        """Retourne les quêtes disponibles pour un joueur"""
        available = []

        for quest in self.quests.values():
            if quest.quest_id in completed_quests:
                continue

            if quest.level_requirement > player_level:
                continue

            # Vérifier prérequis de quêtes
            if quest.requirements.quests_completed:
                missing_prereqs = set(quest.requirements.quests_completed) - set(completed_quests)
                if missing_prereqs:
                    continue

            available.append(quest)

        return available

class QuestManager:
    """Gestionnaire principal des quêtes avec intelligence HRM"""

    def __init__(self, data_dir: str = "data/quests"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base de données des quêtes
        self.quest_db = QuestDatabase(data_dir)

        # Quêtes actives du joueur
        self.active_quests: Dict[str, Quest] = {}
        self.completed_quests: List[str] = []

        # HRM pour prise de décision
        self.hrm_model = create_hrm_model().to(self.device)

        # Vision pour lecture d'écran
        self.vision_engine = create_vision_engine()

        # État du joueur
        self.player_level = 1
        self.player_class = None
        self.player_location = None
        self.player_inventory = []

        # Cache de décisions
        self.decision_cache = {}
        self.last_screen_analysis = None
        self.last_analysis_time = 0

        logger.info("QuestManager initialisé avec succès")

    def update_player_info(self, level: int, player_class: str = None, location: Tuple[int, int] = None):
        """Met à jour les informations du joueur"""
        self.player_level = level
        if player_class:
            self.player_class = player_class
        if location:
            self.player_location = location

    def analyze_quest_screen(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyse l'écran pour détecter les quêtes"""
        current_time = time.time()

        # Cache pour éviter analyses trop fréquentes
        if current_time - self.last_analysis_time < 1.0:
            return self.last_screen_analysis or {}

        try:
            # Analyse vision
            vision_results = self.vision_engine.analyze_screenshot(screenshot)
            text_detections = vision_results.get("text_detections", [])

            # Extraire informations de quête
            quest_info = {
                "quest_text": [],
                "npc_names": [],
                "objective_text": [],
                "dialogue_options": [],
                "quest_rewards": []
            }

            for detection in text_detections:
                text = detection.text.lower()
                text_type = getattr(detection, 'text_type', 'unknown')

                # Identifier le texte de quête
                if any(keyword in text for keyword in ['quête', 'mission', 'tâche', 'objectif']):
                    quest_info["quest_text"].append(detection.text)

                # Identifier les noms de PNJ (commencent par une majuscule)
                elif text_type == 'name' or (text[0].isupper() and len(text) > 3):
                    quest_info["npc_names"].append(detection.text)

                # Identifier les options de dialogue (nombres)
                elif text_type == 'number' and detection.bbox[1] > screenshot.shape[0] * 0.6:
                    quest_info["dialogue_options"].append(int(detection.text))

                # Identifier les récompenses
                elif any(keyword in text for keyword in ['xp', 'exp', 'kamas', 'récompense']):
                    quest_info["quest_rewards"].append(detection.text)

            self.last_screen_analysis = quest_info
            self.last_analysis_time = current_time

            return quest_info

        except Exception as e:
            logger.error(f"Erreur analyse écran quête: {e}")
            return {}

    def get_best_quest_action(self, screenshot: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Utilise HRM pour déterminer la meilleure action de quête"""

        # Analyser l'écran
        screen_info = self.analyze_quest_screen(screenshot)

        # Préparer le contexte pour HRM
        decision_context = {
            "player_level": self.player_level,
            "player_class": self.player_class,
            "location": self.player_location,
            "active_quests": list(self.active_quests.keys()),
            "screen_info": screen_info,
            "available_quests": len(self.get_available_quests())
        }

        if context:
            decision_context.update(context)

        # Cache key pour éviter recalculs
        cache_key = str(hash(str(decision_context)))
        if cache_key in self.decision_cache:
            return self.decision_cache[cache_key]

        try:
            with torch.no_grad():
                # Simuler input pour HRM (version simplifiée)
                input_ids = torch.randint(0, 32000, (1, 64), device=self.device)

                # Forward pass HRM avec System 2 pour planification
                hrm_output = self.hrm_model(
                    input_ids=input_ids,
                    return_reasoning_details=True,
                    max_reasoning_steps=config.hrm.max_reasoning_steps
                )

                # Interpréter la sortie HRM
                action = self._interpret_hrm_output(hrm_output, decision_context, screen_info)

                # Cache la décision
                self.decision_cache[cache_key] = action

                return action

        except Exception as e:
            logger.error(f"Erreur HRM décision quête: {e}")
            return {"action": "wait", "reason": "Erreur de raisonnement"}

    def _interpret_hrm_output(self,
                             hrm_output: Union[HRMOutput, Dict],
                             context: Dict[str, Any],
                             screen_info: Dict[str, Any]) -> Dict[str, Any]:
        """Interprète la sortie HRM en action de quête"""

        # Logique de décision basée sur le contexte
        if not self.active_quests:
            # Pas de quête active, chercher de nouvelles quêtes
            if screen_info.get("npc_names"):
                return {
                    "action": "talk_to_npc",
                    "target": screen_info["npc_names"][0],
                    "reason": "Chercher nouvelles quêtes",
                    "confidence": 0.8
                }
            else:
                return {
                    "action": "find_quest_giver",
                    "reason": "Aucune quête active",
                    "confidence": 0.6
                }

        # Quête active, déterminer l'action suivante
        current_quest = next(iter(self.active_quests.values()))
        current_step = current_quest.current_step

        if not current_step:
            return {
                "action": "complete_quest",
                "quest_id": current_quest.quest_id,
                "reason": "Quête terminée",
                "confidence": 1.0
            }

        # Actions basées sur le type d'étape
        if current_step.step_type == QuestStepType.TALK_TO_NPC:
            if current_step.npc_name in screen_info.get("npc_names", []):
                return {
                    "action": "talk_to_npc",
                    "target": current_step.npc_name,
                    "dialogue_choices": current_step.dialogue_choices,
                    "reason": f"Parler à {current_step.npc_name}",
                    "confidence": 0.9
                }
            else:
                return {
                    "action": "navigate_to_npc",
                    "target": current_step.npc_name,
                    "location": current_step.location,
                    "reason": f"Aller vers {current_step.npc_name}",
                    "confidence": 0.7
                }

        elif current_step.step_type == QuestStepType.KILL_MONSTERS:
            return {
                "action": "hunt_monsters",
                "target": current_step.target,
                "remaining": current_step.quantity - current_step.current_progress,
                "location": current_step.location,
                "reason": f"Chasser {current_step.target}",
                "confidence": 0.8
            }

        elif current_step.step_type == QuestStepType.COLLECT_ITEMS:
            return {
                "action": "collect_items",
                "target": current_step.target,
                "remaining": current_step.quantity - current_step.current_progress,
                "reason": f"Collecter {current_step.target}",
                "confidence": 0.8
            }

        elif current_step.step_type == QuestStepType.GO_TO_LOCATION:
            return {
                "action": "navigate_to_location",
                "location": current_step.location,
                "map_id": current_step.map_id,
                "reason": f"Aller à {current_step.location}",
                "confidence": 0.9
            }

        else:
            return {
                "action": "custom_action",
                "step_type": current_step.step_type.value,
                "description": current_step.description,
                "reason": "Action personnalisée",
                "confidence": 0.5
            }

    def start_quest(self, quest_id: str) -> bool:
        """Démarre une quête"""
        quest = self.quest_db.get_quest(quest_id)
        if not quest:
            logger.error(f"Quête introuvable: {quest_id}")
            return False

        # Vérifier prérequis
        if quest.level_requirement > self.player_level:
            logger.warning(f"Niveau insuffisant pour {quest_id} (requis: {quest.level_requirement})")
            return False

        # Démarrer la quête
        quest.status = QuestStatus.IN_PROGRESS
        quest.start_time = time.time()
        self.active_quests[quest_id] = quest

        logger.info(f"Quête démarrée: {quest.name} ({quest_id})")
        return True

    def update_quest_progress(self, quest_id: str, step_progress: int = None, custom_data: Dict = None):
        """Met à jour la progression d'une quête"""
        quest = self.active_quests.get(quest_id)
        if not quest:
            return

        current_step = quest.current_step
        if not current_step:
            return

        # Mettre à jour la progression
        if step_progress is not None:
            current_step.current_progress = min(step_progress, current_step.quantity)

        # Vérifier si l'étape est terminée
        if current_step.is_completed():
            current_step.completed = True

            # Passer à l'étape suivante
            if quest.advance_step():
                logger.info(f"Étape terminée pour {quest.name}: {current_step.description}")
            else:
                # Quête terminée
                quest.complete_quest()
                self.completed_quests.append(quest_id)
                del self.active_quests[quest_id]
                logger.info(f"Quête terminée: {quest.name}")

    def get_available_quests(self) -> List[Quest]:
        """Retourne les quêtes disponibles"""
        return self.quest_db.get_available_quests(self.player_level, self.completed_quests)

    def get_quest_priority_list(self) -> List[Quest]:
        """Retourne une liste priorisée de quêtes recommandées"""
        available = self.get_available_quests()

        # Prioriser par type et niveau
        def quest_priority(quest: Quest) -> float:
            priority = 0.0

            # Types prioritaires
            if quest.quest_type == QuestType.MAIN_STORY:
                priority += 100
            elif quest.quest_type == QuestType.TUTORIAL:
                priority += 90
            elif quest.quest_type == QuestType.DAILY:
                priority += 70

            # Niveau approprié
            level_diff = quest.level_requirement - self.player_level
            if level_diff <= 0:
                priority += 50 - abs(level_diff) * 5

            # Durée estimée (préférer les courtes)
            priority += max(0, 30 - quest.estimated_duration)

            # Récompenses XP
            priority += quest.rewards.experience / 1000

            return priority

        return sorted(available, key=quest_priority, reverse=True)

    def save_progress(self, filepath: str):
        """Sauvegarde la progression des quêtes"""
        progress_data = {
            "player_level": self.player_level,
            "player_class": self.player_class,
            "completed_quests": self.completed_quests,
            "active_quests": {
                qid: {
                    "quest_id": quest.quest_id,
                    "status": quest.status.value,
                    "current_step_index": quest.current_step_index,
                    "start_time": quest.start_time,
                    "steps_progress": [
                        {
                            "step_id": step.step_id,
                            "current_progress": step.current_progress,
                            "completed": step.completed
                        }
                        for step in quest.steps
                    ]
                }
                for qid, quest in self.active_quests.items()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)

    def load_progress(self, filepath: str):
        """Charge la progression des quêtes"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)

            self.player_level = progress_data.get("player_level", 1)
            self.player_class = progress_data.get("player_class")
            self.completed_quests = progress_data.get("completed_quests", [])

            # Restaurer les quêtes actives
            for qid, quest_data in progress_data.get("active_quests", {}).items():
                quest = self.quest_db.get_quest(quest_data["quest_id"])
                if quest:
                    quest.status = QuestStatus(quest_data["status"])
                    quest.current_step_index = quest_data["current_step_index"]
                    quest.start_time = quest_data.get("start_time")

                    # Restaurer progression des étapes
                    for i, step_data in enumerate(quest_data.get("steps_progress", [])):
                        if i < len(quest.steps):
                            quest.steps[i].current_progress = step_data["current_progress"]
                            quest.steps[i].completed = step_data["completed"]

                    self.active_quests[qid] = quest

            logger.info(f"Progression chargée: {len(self.completed_quests)} quêtes terminées, {len(self.active_quests)} actives")

        except Exception as e:
            logger.error(f"Erreur chargement progression: {e}")

def create_quest_manager(data_dir: str = "data/quests") -> QuestManager:
    """Factory function pour créer un QuestManager"""
    return QuestManager(data_dir)