"""
Ganymede Integration pour TacticalBot
Intégration avec les guides de quêtes Ganymede
Parsing et exécution intelligente des guides optimisés

Fonctionnalités:
- Parsing des guides Ganymede (format JSON/HTML)
- Suivi de progression des quêtes
- Navigation intelligente entre étapes
- Adaptation aux échecs et blocages
- Optimisation de l'ordre d'exécution
"""

import json
import logging
import requests
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
from enum import Enum

from ...engine.module_interface import IModule, ModuleStatus
from ...state.realtime_state import GameState


class QuestStepType(Enum):
    """Types d'étapes de quête"""
    TALK_TO_NPC = "talk_to_npc"
    KILL_MONSTER = "kill_monster"
    COLLECT_ITEM = "collect_item"
    GO_TO_LOCATION = "go_to_location"
    USE_ITEM = "use_item"
    WAIT = "wait"
    CRAFT_ITEM = "craft_item"
    CHOICE = "choice"


class QuestDifficulty(Enum):
    """Difficulté des quêtes"""
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5


@dataclass
class QuestStep:
    """Étape d'une quête"""
    step_id: int
    step_type: QuestStepType
    description: str
    
    # Détails de l'étape
    target: Optional[str] = None  # NPC, monstre, item
    location: Optional[Tuple[int, int]] = None
    map_name: Optional[str] = None
    quantity: int = 1
    
    # Conditions
    prerequisites: List[int] = field(default_factory=list)
    level_required: int = 1
    items_required: Dict[str, int] = field(default_factory=dict)
    
    # Métadonnées
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    difficulty: QuestDifficulty = QuestDifficulty.MEDIUM
    hints: List[str] = field(default_factory=list)
    
    # État
    completed: bool = False
    attempts: int = 0
    last_attempt: Optional[datetime] = None


@dataclass
class Quest:
    """Quête complète avec toutes ses étapes"""
    quest_id: str
    name: str
    level_required: int
    
    # Étapes
    steps: List[QuestStep] = field(default_factory=list)
    current_step: int = 0
    
    # Récompenses
    rewards: Dict[str, Any] = field(default_factory=dict)
    
    # Métadonnées
    category: str = "main"
    repeatable: bool = False
    estimated_total_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    difficulty: QuestDifficulty = QuestDifficulty.MEDIUM
    
    # Progression
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    abandoned: bool = False
    abandon_reason: Optional[str] = None
    
    def progress_percentage(self) -> float:
        """Calcule le pourcentage de progression"""
        if not self.steps:
            return 0.0
        completed_steps = sum(1 for step in self.steps if step.completed)
        return (completed_steps / len(self.steps)) * 100
    
    def is_completed(self) -> bool:
        """Vérifie si la quête est terminée"""
        return all(step.completed for step in self.steps)
    
    def get_current_step(self) -> Optional[QuestStep]:
        """Obtient l'étape courante"""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None


class GanymedeParser:
    """Parser pour les guides Ganymede"""
    
    def __init__(self, cache_dir: str = "data/ganymede_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.GanymedeParser")
    
    def fetch_quest_guide(self, quest_name: str) -> Optional[Dict]:
        """Récupère un guide de quête depuis Ganymede"""
        try:
            # Vérification cache local
            cached_guide = self._load_from_cache(quest_name)
            if cached_guide:
                return cached_guide
            
            # Recherche sur Ganymede (API ou scraping)
            guide_data = self._fetch_from_ganymede(quest_name)
            
            if guide_data:
                self._save_to_cache(quest_name, guide_data)
                return guide_data
            
            return None
        
        except Exception as e:
            self.logger.error(f"Erreur récupération guide {quest_name}: {e}")
            return None
    
    def parse_guide_to_quest(self, guide_data: Dict) -> Quest:
        """Parse un guide Ganymede en objet Quest"""
        quest = Quest(
            quest_id=guide_data.get("id", "unknown"),
            name=guide_data.get("name", "Unknown Quest"),
            level_required=guide_data.get("level", 1),
            category=guide_data.get("category", "main"),
            rewards=guide_data.get("rewards", {})
        )
        
        # Parsing des étapes
        steps_data = guide_data.get("steps", [])
        for i, step_data in enumerate(steps_data):
            step = self._parse_step(i, step_data)
            quest.steps.append(step)
        
        # Calcul durée totale
        total_duration = sum(
            (step.estimated_duration for step in quest.steps),
            timedelta()
        )
        quest.estimated_total_duration = total_duration
        
        return quest
    
    def _parse_step(self, step_id: int, step_data: Dict) -> QuestStep:
        """Parse une étape de quête"""
        # Détection du type d'étape
        step_type = self._detect_step_type(step_data)
        
        step = QuestStep(
            step_id=step_id,
            step_type=step_type,
            description=step_data.get("description", ""),
            target=step_data.get("target"),
            location=self._parse_location(step_data.get("location")),
            map_name=step_data.get("map"),
            quantity=step_data.get("quantity", 1),
            level_required=step_data.get("level", 1),
            hints=step_data.get("hints", [])
        )
        
        return step
    
    def _detect_step_type(self, step_data: Dict) -> QuestStepType:
        """Détecte le type d'étape depuis les données"""
        description = step_data.get("description", "").lower()
        
        if "talk to" in description or "speak with" in description:
            return QuestStepType.TALK_TO_NPC
        elif "kill" in description or "defeat" in description:
            return QuestStepType.KILL_MONSTER
        elif "collect" in description or "gather" in description:
            return QuestStepType.COLLECT_ITEM
        elif "go to" in description or "travel to" in description:
            return QuestStepType.GO_TO_LOCATION
        elif "use" in description:
            return QuestStepType.USE_ITEM
        elif "craft" in description:
            return QuestStepType.CRAFT_ITEM
        else:
            return QuestStepType.TALK_TO_NPC  # Par défaut
    
    def _parse_location(self, location_str: Optional[str]) -> Optional[Tuple[int, int]]:
        """Parse une chaîne de location en coordonnées"""
        if not location_str:
            return None
        
        # Format attendu: "[x, y]" ou "x,y"
        match = re.search(r'\[?(-?\d+),\s*(-?\d+)\]?', location_str)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        
        return None
    
    def _fetch_from_ganymede(self, quest_name: str) -> Optional[Dict]:
        """Récupère depuis l'API/site Ganymede"""
        try:
            # URL de l'API Ganymede (à adapter)
            url = f"https://dofus-map.com/api/quests/{quest_name}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            
            self.logger.warning(f"Guide non trouvé pour {quest_name}")
            return None
        
        except Exception as e:
            self.logger.error(f"Erreur fetch Ganymede: {e}")
            return None
    
    def _load_from_cache(self, quest_name: str) -> Optional[Dict]:
        """Charge depuis le cache local"""
        cache_file = self.cache_dir / f"{quest_name}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Vérification fraîcheur (7 jours)
                cached_time = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
                if datetime.now() - cached_time < timedelta(days=7):
                    return data.get("guide_data")
            
            except Exception as e:
                self.logger.warning(f"Erreur lecture cache: {e}")
        
        return None
    
    def _save_to_cache(self, quest_name: str, guide_data: Dict):
        """Sauvegarde dans le cache local"""
        cache_file = self.cache_dir / f"{quest_name}.json"
        
        try:
            cache_data = {
                "cached_at": datetime.now().isoformat(),
                "guide_data": guide_data
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde cache: {e}")


class QuestExecutor:
    """Exécuteur intelligent de quêtes"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QuestExecutor")
        self.execution_strategies = self._load_execution_strategies()
    
    def execute_step(self, step: QuestStep, game_state: GameState) -> Dict[str, Any]:
        """Exécute une étape de quête"""
        self.logger.info(f"Exécution étape: {step.description}")
        
        # Sélection de la stratégie d'exécution
        strategy = self.execution_strategies.get(step.step_type)
        
        if not strategy:
            return {"success": False, "reason": "No strategy for step type"}
        
        # Exécution
        result = strategy(step, game_state)
        
        # Mise à jour de l'étape
        step.attempts += 1
        step.last_attempt = datetime.now()
        
        if result.get("success"):
            step.completed = True
        
        return result
    
    def _execute_talk_to_npc(self, step: QuestStep, game_state: GameState) -> Dict:
        """Exécute une étape de dialogue avec NPC"""
        # 1. Vérifier si on est à la bonne position
        if step.location and game_state.character.position:
            current_pos = (game_state.character.position.x, game_state.character.position.y)
            distance = self._calculate_distance(current_pos, step.location)
            
            if distance > 5:
                return {
                    "success": False,
                    "reason": "Too far from NPC",
                    "action_needed": "navigate",
                    "target_location": step.location
                }
        
        # 2. Chercher le NPC
        npc_found = self._find_npc(step.target, game_state)
        
        if not npc_found:
            return {
                "success": False,
                "reason": "NPC not found",
                "action_needed": "search_npc",
                "target": step.target
            }
        
        # 3. Interagir avec le NPC
        return {
            "success": True,
            "action_needed": "interact_npc",
            "target": step.target
        }
    
    def _execute_kill_monster(self, step: QuestStep, game_state: GameState) -> Dict:
        """Exécute une étape de combat"""
        # Vérification niveau
        if game_state.character.level < step.level_required:
            return {
                "success": False,
                "reason": "Level too low",
                "recommended_action": "level_up"
            }
        
        # Vérification HP
        if game_state.character.hp_percentage() < 30:
            return {
                "success": False,
                "reason": "Low HP",
                "action_needed": "heal"
            }
        
        # Recherche du monstre
        monster_found = self._find_monster(step.target, game_state)
        
        if not monster_found:
            return {
                "success": False,
                "reason": "Monster not found",
                "action_needed": "search_monster",
                "target": step.target,
                "location": step.location
            }
        
        return {
            "success": True,
            "action_needed": "engage_combat",
            "target": step.target,
            "quantity": step.quantity
        }
    
    def _execute_collect_item(self, step: QuestStep, game_state: GameState) -> Dict:
        """Exécute une étape de collecte"""
        # Vérification inventaire
        current_quantity = self._check_inventory(step.target, game_state)
        
        if current_quantity >= step.quantity:
            return {"success": True, "reason": "Already have item"}
        
        remaining = step.quantity - current_quantity
        
        return {
            "success": False,
            "action_needed": "collect_item",
            "target": step.target,
            "quantity_needed": remaining,
            "location": step.location
        }
    
    def _execute_go_to_location(self, step: QuestStep, game_state: GameState) -> Dict:
        """Exécute une étape de déplacement"""
        if not step.location:
            return {"success": False, "reason": "No target location"}
        
        current_pos = (game_state.character.position.x, game_state.character.position.y)
        distance = self._calculate_distance(current_pos, step.location)
        
        if distance < 2:
            return {"success": True, "reason": "Already at location"}
        
        return {
            "success": False,
            "action_needed": "navigate",
            "target_location": step.location,
            "distance": distance
        }
    
    def _load_execution_strategies(self) -> Dict:
        """Charge les stratégies d'exécution"""
        return {
            QuestStepType.TALK_TO_NPC: self._execute_talk_to_npc,
            QuestStepType.KILL_MONSTER: self._execute_kill_monster,
            QuestStepType.COLLECT_ITEM: self._execute_collect_item,
            QuestStepType.GO_TO_LOCATION: self._execute_go_to_location,
        }
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance entre deux positions"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5
    
    def _find_npc(self, npc_name: Optional[str], game_state: GameState) -> bool:
        """Recherche un NPC"""
        # À implémenter avec vision
        return False
    
    def _find_monster(self, monster_name: Optional[str], game_state: GameState) -> bool:
        """Recherche un monstre"""
        # À implémenter avec vision
        return False
    
    def _check_inventory(self, item_name: Optional[str], game_state: GameState) -> int:
        """Vérifie la quantité d'un item dans l'inventaire"""
        # À implémenter
        return 0


class GanymedeIntegration(IModule):
    """
    Intégration Ganymede pour suivi de quêtes optimisé
    """
    
    def __init__(self, name: str = "ganymede_integration"):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Composants
        self.parser = GanymedeParser()
        self.executor = QuestExecutor()
        
        # Quêtes
        self.active_quests: Dict[str, Quest] = {}
        self.completed_quests: List[Quest] = []
        self.available_quests: List[str] = []
        
        # Configuration
        self.auto_accept_quests = True
        self.abandon_on_difficulty = True
        self.max_attempts_per_step = 3
        
        # Métriques
        self.metrics = {
            "quests_completed": 0,
            "quests_abandoned": 0,
            "total_steps_executed": 0,
            "success_rate": 0.0
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module"""
        try:
            self.status = ModuleStatus.INITIALIZING
            
            # Configuration
            self.auto_accept_quests = config.get("auto_accept_quests", True)
            self.abandon_on_difficulty = config.get("abandon_on_difficulty", True)
            self.max_attempts_per_step = config.get("max_attempts_per_step", 3)
            
            # Chargement des quêtes disponibles
            self._load_available_quests()
            
            self.status = ModuleStatus.ACTIVE
            self.logger.info("Intégration Ganymede initialisée")
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur initialisation: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour l'exécution des quêtes"""
        if not self.is_active():
            return None
        
        try:
            # Exécution des quêtes actives
            for quest_id, quest in list(self.active_quests.items()):
                if quest.is_completed():
                    self._complete_quest(quest)
                    continue
                
                # Exécution de l'étape courante
                current_step = quest.get_current_step()
                if current_step and not current_step.completed:
                    result = self.executor.execute_step(current_step, game_state)
                    
                    # Gestion du résultat
                    if result.get("success"):
                        quest.current_step += 1
                        self.metrics["total_steps_executed"] += 1
                    
                    elif current_step.attempts >= self.max_attempts_per_step:
                        # Abandon si trop d'échecs
                        if self.abandon_on_difficulty:
                            self._abandon_quest(quest, "Too many failed attempts")
            
            return {
                "ganymede": {
                    "active_quests": len(self.active_quests),
                    "completed_quests": len(self.completed_quests),
                    "current_quest": list(self.active_quests.keys())[0] if self.active_quests else None
                }
            }
        
        except Exception as e:
            self.logger.error(f"Erreur mise à jour: {e}")
            return None
    
    def start_quest(self, quest_name: str) -> bool:
        """Démarre une quête depuis Ganymede"""
        try:
            # Récupération du guide
            guide_data = self.parser.fetch_quest_guide(quest_name)
            
            if not guide_data:
                self.logger.error(f"Guide non trouvé pour {quest_name}")
                return False
            
            # Parsing en objet Quest
            quest = self.parser.parse_guide_to_quest(guide_data)
            quest.started_at = datetime.now()
            
            # Ajout aux quêtes actives
            self.active_quests[quest.quest_id] = quest
            
            self.logger.info(f"✅ Quête démarrée: {quest.name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur démarrage quête: {e}")
            return False
    
    def _complete_quest(self, quest: Quest):
        """Marque une quête comme terminée"""
        quest.completed_at = datetime.now()
        self.completed_quests.append(quest)
        del self.active_quests[quest.quest_id]
        
        self.metrics["quests_completed"] += 1
        self.logger.info(f"🎉 Quête terminée: {quest.name}")
    
    def _abandon_quest(self, quest: Quest, reason: str):
        """Abandonne une quête"""
        quest.abandoned = True
        quest.abandon_reason = reason
        
        del self.active_quests[quest.quest_id]
        self.metrics["quests_abandoned"] += 1
        
        self.logger.warning(f"⚠️ Quête abandonnée: {quest.name} - {reason}")
    
    def _load_available_quests(self):
        """Charge la liste des quêtes disponibles"""
        # À implémenter: récupérer depuis Ganymede ou fichier local
        self.available_quests = [
            "Astrub Tutorial",
            "The Gobball Hunt",
            "Scaraleaf Dungeon Quest"
        ]
    
    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du module"""
        return {
            "status": self.status.value,
            "active_quests": len(self.active_quests),
            "completed_quests": len(self.completed_quests),
            "metrics": self.metrics
        }
    
    def cleanup(self) -> None:
        """Nettoie le module"""
        try:
            self.logger.info("Intégration Ganymede nettoyée")
        except Exception as e:
            self.logger.error(f"Erreur nettoyage: {e}")
    
    def get_quest_report(self) -> Dict[str, Any]:
        """Génère un rapport sur les quêtes"""
        return {
            "active_quests": [
                {
                    "name": quest.name,
                    "progress": quest.progress_percentage(),
                    "current_step": quest.current_step,
                    "total_steps": len(quest.steps)
                }
                for quest in self.active_quests.values()
            ],
            "completed_today": len([
                q for q in self.completed_quests
                if q.completed_at and (datetime.now() - q.completed_at) < timedelta(days=1)
            ]),
            "metrics": self.metrics
        }
