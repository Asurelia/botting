"""
Module d'automatisation complète des quêtes.

Système intelligent de parsing et d'exécution automatisée des quêtes :
- Parsing des objectifs de quête avec NLP
- Exécution automatisée multi-étapes
- Gestion intelligente des prérequis
- Optimisation des chemins et actions
"""

import logging
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class QuestType(Enum):
    """Types de quêtes."""
    MAIN_STORY = "main_story"           # Quête principale
    SIDE_QUEST = "side_quest"           # Quête secondaire
    DUNGEON_QUEST = "dungeon_quest"     # Quête de donjon
    PROFESSION_QUEST = "profession_quest" # Quête de métier
    DAILY_QUEST = "daily_quest"         # Quête quotidienne
    ACHIEVEMENT = "achievement"         # Succès/Achievement
    COLLECTION = "collection"           # Quête de collection


class QuestStatus(Enum):
    """États des quêtes."""
    NOT_STARTED = "not_started"
    AVAILABLE = "available"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    BLOCKED = "blocked"


class ObjectiveType(Enum):
    """Types d'objectifs de quête."""
    KILL_MONSTERS = "kill_monsters"     # Tuer des monstres
    COLLECT_ITEMS = "collect_items"     # Collecter des objets
    TALK_TO_NPC = "talk_to_npc"        # Parler à un PNJ
    GO_TO_LOCATION = "go_to_location"  # Aller à un endroit
    USE_ITEM = "use_item"              # Utiliser un objet
    CRAFT_ITEM = "craft_item"          # Crafter un objet
    GATHER_RESOURCE = "gather_resource" # Récolter une ressource
    WIN_FIGHT = "win_fight"            # Gagner un combat
    REACH_LEVEL = "reach_level"        # Atteindre un niveau
    SPEND_KAMAS = "spend_kamas"        # Dépenser des kamas


@dataclass
class QuestObjective:
    """Objectif d'une quête."""
    id: str
    description: str
    objective_type: ObjectiveType
    target: str                        # Cible (monstre, objet, PNJ, etc.)
    quantity_required: int = 1
    quantity_current: int = 0
    location: Optional[str] = None     # Lieu où réaliser l'objectif
    prerequisites: List[str] = field(default_factory=list)
    completion_conditions: Dict[str, Any] = field(default_factory=dict)
    is_completed: bool = False
    auto_executable: bool = True       # Peut être exécuté automatiquement


@dataclass
class Quest:
    """Représente une quête complète."""
    id: str
    name: str
    description: str
    quest_type: QuestType
    level_requirement: int = 1
    prerequisites: List[str] = field(default_factory=list)
    objectives: List[QuestObjective] = field(default_factory=list)
    rewards: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: int = 300      # Durée estimée en secondes
    difficulty: int = 1                # Difficulté de 1 à 5
    
    # État d'exécution
    status: QuestStatus = QuestStatus.NOT_STARTED
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    current_objective: int = 0
    attempts: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None


class QuestParser:
    """Parser intelligent des descriptions de quêtes."""
    
    def __init__(self):
        # Patterns de reconnaissance des objectifs
        self.objective_patterns = {
            ObjectiveType.KILL_MONSTERS: [
                r"tuer?\s+(\d+)?\s*(.+?)(?:\s+dans?\s+(.+?))?",
                r"éliminer?\s+(\d+)?\s*(.+?)(?:\s+à\s+(.+?))?",
                r"vaincre?\s+(\d+)?\s*(.+?)(?:\s+en\s+(.+?))?"
            ],
            ObjectiveType.COLLECT_ITEMS: [
                r"collecter?\s+(\d+)?\s*(.+?)(?:\s+dans?\s+(.+?))?",
                r"récupérer?\s+(\d+)?\s*(.+?)(?:\s+à\s+(.+?))?",
                r"ramasser?\s+(\d+)?\s*(.+?)(?:\s+en\s+(.+?))?"
            ],
            ObjectiveType.TALK_TO_NPC: [
                r"parler?\s+(?:à|avec)\s+(.+?)(?:\s+(?:à|dans?)\s+(.+?))?",
                r"voir\s+(.+?)(?:\s+(?:à|dans?)\s+(.+?))?",
                r"rencontrer?\s+(.+?)(?:\s+(?:à|dans?)\s+(.+?))?"
            ],
            ObjectiveType.GO_TO_LOCATION: [
                r"aller?\s+(?:à|dans?|vers)\s+(.+)",
                r"se rendre\s+(?:à|dans?|vers)\s+(.+)",
                r"rejoindre\s+(.+)"
            ],
            ObjectiveType.USE_ITEM: [
                r"utiliser?\s+(.+?)(?:\s+sur\s+(.+?))?",
                r"employer?\s+(.+?)(?:\s+avec\s+(.+?))?"
            ],
            ObjectiveType.CRAFT_ITEM: [
                r"fabriquer?\s+(\d+)?\s*(.+)",
                r"crafter?\s+(\d+)?\s*(.+)",
                r"confectionner?\s+(\d+)?\s*(.+)"
            ],
            ObjectiveType.GATHER_RESOURCE: [
                r"récolter?\s+(\d+)?\s*(.+?)(?:\s+dans?\s+(.+?))?",
                r"miner?\s+(\d+)?\s*(.+?)(?:\s+à\s+(.+?))?",
                r"couper?\s+(\d+)?\s*(.+?)(?:\s+en\s+(.+?))?"
            ]
        }
        
        # Mots-clés de localisation
        self.location_keywords = [
            "dans", "à", "en", "vers", "sur", "chez", "près de", "autour de"
        ]
        
        # Nombres écrits en français
        self.number_words = {
            "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
            "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
            "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50
        }
    
    def parse_quest_text(self, quest_text: str) -> List[QuestObjective]:
        """Parse le texte d'une quête et extrait les objectifs."""
        objectives = []
        
        # Nettoie et divise le texte en phrases
        sentences = self._clean_and_split_text(quest_text)
        
        for i, sentence in enumerate(sentences):
            objective = self._parse_sentence(sentence, f"obj_{i}")
            if objective:
                objectives.append(objective)
        
        return objectives
    
    def _clean_and_split_text(self, text: str) -> List[str]:
        """Nettoie le texte et le divise en phrases."""
        # Supprime les caractères spéciaux et normalise
        text = re.sub(r'[^\w\s\-àâäéèêëïîôùûüÿç.,;:!?()]', ' ', text.lower())
        text = ' '.join(text.split())
        
        # Divise en phrases
        sentences = re.split(r'[.!?;]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _parse_sentence(self, sentence: str, obj_id: str) -> Optional[QuestObjective]:
        """Parse une phrase et extrait un objectif."""
        for obj_type, patterns in self.objective_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, sentence)
                if match:
                    return self._create_objective_from_match(
                        obj_id, sentence, obj_type, match
                    )
        return None
    
    def _create_objective_from_match(self, obj_id: str, sentence: str, 
                                   obj_type: ObjectiveType, match) -> QuestObjective:
        """Crée un objectif à partir d'un match regex."""
        groups = match.groups()
        
        # Extraction de la quantité
        quantity = 1
        if groups[0] and groups[0].isdigit():
            quantity = int(groups[0])
        elif groups[0] and groups[0] in self.number_words:
            quantity = self.number_words[groups[0]]
        
        # Extraction de la cible
        target = groups[1] if len(groups) > 1 and groups[1] else "unknown"
        target = target.strip()
        
        # Extraction de la localisation
        location = None
        if len(groups) > 2 and groups[2]:
            location = groups[2].strip()
        
        return QuestObjective(
            id=obj_id,
            description=sentence,
            objective_type=obj_type,
            target=target,
            quantity_required=quantity,
            location=location
        )
    
    def parse_quest_log_entry(self, quest_data: Dict[str, Any]) -> Quest:
        """Parse une entrée du journal de quêtes."""
        quest = Quest(
            id=quest_data.get('id', 'unknown'),
            name=quest_data.get('name', 'Unknown Quest'),
            description=quest_data.get('description', ''),
            quest_type=self._determine_quest_type(quest_data),
            level_requirement=quest_data.get('level_req', 1)
        )
        
        # Parse les objectifs
        if 'objectives_text' in quest_data:
            quest.objectives = self.parse_quest_text(quest_data['objectives_text'])
        
        # Parse les récompenses
        if 'rewards' in quest_data:
            quest.rewards = quest_data['rewards']
        
        return quest
    
    def _determine_quest_type(self, quest_data: Dict[str, Any]) -> QuestType:
        """Détermine le type de quête à partir des données."""
        name = quest_data.get('name', '').lower()
        description = quest_data.get('description', '').lower()
        
        if 'quotidien' in name or 'daily' in name:
            return QuestType.DAILY_QUEST
        elif 'donjon' in description or 'dungeon' in description:
            return QuestType.DUNGEON_QUEST
        elif 'métier' in description or 'profession' in description:
            return QuestType.PROFESSION_QUEST
        elif 'succès' in name or 'achievement' in name:
            return QuestType.ACHIEVEMENT
        elif quest_data.get('is_main_story', False):
            return QuestType.MAIN_STORY
        else:
            return QuestType.SIDE_QUEST


class QuestExecutor:
    """Exécuteur automatique de quêtes."""
    
    def __init__(self, game_interface=None, navigation_system=None):
        self.game_interface = game_interface
        self.navigation_system = navigation_system
        
        # Stratégies d'exécution par type d'objectif
        self.execution_strategies = {
            ObjectiveType.KILL_MONSTERS: self._execute_kill_monsters,
            ObjectiveType.COLLECT_ITEMS: self._execute_collect_items,
            ObjectiveType.TALK_TO_NPC: self._execute_talk_to_npc,
            ObjectiveType.GO_TO_LOCATION: self._execute_go_to_location,
            ObjectiveType.USE_ITEM: self._execute_use_item,
            ObjectiveType.CRAFT_ITEM: self._execute_craft_item,
            ObjectiveType.GATHER_RESOURCE: self._execute_gather_resource
        }
    
    def execute_objective(self, objective: QuestObjective) -> bool:
        """Exécute un objectif de quête."""
        if not objective.auto_executable:
            logger.warning(f"Objectif non auto-exécutable : {objective.description}")
            return False
        
        if objective.is_completed:
            return True
        
        logger.info(f"Exécution de l'objectif : {objective.description}")
        
        try:
            strategy = self.execution_strategies.get(objective.objective_type)
            if strategy:
                success = strategy(objective)
                if success:
                    objective.is_completed = True
                return success
            else:
                logger.warning(f"Stratégie non implémentée : {objective.objective_type}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'objectif : {e}")
            return False
    
    def _execute_kill_monsters(self, objective: QuestObjective) -> bool:
        """Exécute un objectif de tuer des monstres."""
        logger.info(f"Tuer {objective.quantity_required} {objective.target}")
        
        # Logique pour tuer des monstres
        # 1. Aller à la zone appropriée
        if objective.location and self.navigation_system:
            self.navigation_system.navigate_to(objective.location)
        
        # 2. Chercher et engager les monstres
        monsters_killed = 0
        while monsters_killed < objective.quantity_required:
            # Simuler la recherche et le combat
            time.sleep(5)  # Temps de recherche
            
            if self._find_and_fight_monster(objective.target):
                monsters_killed += 1
                objective.quantity_current = monsters_killed
                logger.info(f"Progrès : {monsters_killed}/{objective.quantity_required}")
            else:
                logger.warning("Impossible de trouver/combattre le monstre")
                break
        
        return monsters_killed >= objective.quantity_required
    
    def _execute_collect_items(self, objective: QuestObjective) -> bool:
        """Exécute un objectif de collecte d'objets."""
        logger.info(f"Collecter {objective.quantity_required} {objective.target}")
        
        # Logique pour collecter des objets
        # 1. Vérifier l'inventaire actuel
        current_count = self._count_items_in_inventory(objective.target)
        
        if current_count >= objective.quantity_required:
            objective.quantity_current = objective.quantity_required
            return True
        
        # 2. Aller à la zone de collecte
        if objective.location and self.navigation_system:
            self.navigation_system.navigate_to(objective.location)
        
        # 3. Collecter les objets manquants
        needed = objective.quantity_required - current_count
        collected = 0
        
        while collected < needed:
            if self._collect_item(objective.target):
                collected += 1
                objective.quantity_current = current_count + collected
            else:
                time.sleep(2)  # Attendre la régénération
        
        return collected >= needed
    
    def _execute_talk_to_npc(self, objective: QuestObjective) -> bool:
        """Exécute un objectif de parler à un PNJ."""
        logger.info(f"Parler à {objective.target}")
        
        # Logique pour parler à un PNJ
        # 1. Aller à la localisation du PNJ
        if objective.location and self.navigation_system:
            self.navigation_system.navigate_to(objective.location)
        
        # 2. Trouver et cliquer sur le PNJ
        if self._find_and_talk_to_npc(objective.target):
            time.sleep(2)  # Temps de dialogue
            return True
        
        return False
    
    def _execute_go_to_location(self, objective: QuestObjective) -> bool:
        """Exécute un objectif d'aller à un endroit."""
        logger.info(f"Aller à {objective.target}")
        
        if self.navigation_system:
            return self.navigation_system.navigate_to(objective.target)
        
        return False
    
    def _execute_use_item(self, objective: QuestObjective) -> bool:
        """Exécute un objectif d'utiliser un objet."""
        logger.info(f"Utiliser {objective.target}")
        
        # Logique pour utiliser un objet
        # 1. Vérifier la présence de l'objet
        if not self._has_item_in_inventory(objective.target):
            logger.error(f"Objet non trouvé : {objective.target}")
            return False
        
        # 2. Utiliser l'objet
        return self._use_item_from_inventory(objective.target)
    
    def _execute_craft_item(self, objective: QuestObjective) -> bool:
        """Exécute un objectif de craft."""
        logger.info(f"Crafter {objective.quantity_required} {objective.target}")
        
        # Logique pour crafter
        crafted = 0
        while crafted < objective.quantity_required:
            if self._craft_single_item(objective.target):
                crafted += 1
                objective.quantity_current = crafted
            else:
                break
        
        return crafted >= objective.quantity_required
    
    def _execute_gather_resource(self, objective: QuestObjective) -> bool:
        """Exécute un objectif de récolte de ressources."""
        logger.info(f"Récolter {objective.quantity_required} {objective.target}")
        
        # Similar à collect_items mais pour les ressources naturelles
        return self._execute_collect_items(objective)
    
    # Méthodes utilitaires (à implémenter selon l'interface du jeu)
    
    def _find_and_fight_monster(self, monster_name: str) -> bool:
        """Trouve et combat un monstre spécifique."""
        # Implémentation de la logique de combat
        time.sleep(10)  # Simulation du combat
        return True
    
    def _count_items_in_inventory(self, item_name: str) -> int:
        """Compte les objets dans l'inventaire."""
        # Implémentation de la vérification d'inventaire
        return 0
    
    def _collect_item(self, item_name: str) -> bool:
        """Collecte un objet spécifique."""
        # Implémentation de la collecte
        time.sleep(3)
        return True
    
    def _find_and_talk_to_npc(self, npc_name: str) -> bool:
        """Trouve et parle à un PNJ."""
        # Implémentation de l'interaction PNJ
        time.sleep(2)
        return True
    
    def _has_item_in_inventory(self, item_name: str) -> bool:
        """Vérifie la présence d'un objet."""
        return True
    
    def _use_item_from_inventory(self, item_name: str) -> bool:
        """Utilise un objet de l'inventaire."""
        time.sleep(2)
        return True
    
    def _craft_single_item(self, item_name: str) -> bool:
        """Craft un seul objet."""
        time.sleep(5)
        return True


class QuestAutomation:
    """
    Système principal d'automatisation des quêtes.
    
    Gère l'ensemble du processus d'automatisation :
    - Parsing des quêtes
    - Planification d'exécution
    - Exécution automatisée
    - Gestion des erreurs et reprises
    """
    
    def __init__(self, game_interface=None, navigation_system=None):
        self.game_interface = game_interface
        self.parser = QuestParser()
        self.executor = QuestExecutor(game_interface, navigation_system)
        
        self.active_quests: Dict[str, Quest] = {}
        self.quest_queue: List[str] = []
        self.completed_quests: List[str] = []
        
        # Configuration
        self.auto_accept_quests = True
        self.max_concurrent_quests = 10
        self.retry_failed_objectives = True
        self.max_quest_duration = 3600  # 1 heure max par quête
        
        # Statistiques
        self.stats = {
            'quests_completed': 0,
            'objectives_completed': 0,
            'total_execution_time': 0,
            'success_rate': 0.0,
            'errors_encountered': 0
        }
        
        logger.info("Système d'automatisation de quêtes initialisé")
    
    def add_quest(self, quest_data: Dict[str, Any]) -> bool:
        """Ajoute une quête au système."""
        try:
            quest = self.parser.parse_quest_log_entry(quest_data)
            
            if quest.id in self.active_quests:
                logger.warning(f"Quête déjà active : {quest.name}")
                return False
            
            self.active_quests[quest.id] = quest
            self._add_to_execution_queue(quest.id)
            
            logger.info(f"Quête ajoutée : {quest.name}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de quête : {e}")
            return False
    
    def add_quest_from_text(self, quest_name: str, quest_text: str, 
                          quest_type: QuestType = QuestType.SIDE_QUEST) -> bool:
        """Ajoute une quête à partir de texte brut."""
        quest_id = f"text_quest_{int(time.time())}"
        objectives = self.parser.parse_quest_text(quest_text)
        
        quest = Quest(
            id=quest_id,
            name=quest_name,
            description=quest_text,
            quest_type=quest_type,
            objectives=objectives
        )
        
        self.active_quests[quest_id] = quest
        self._add_to_execution_queue(quest_id)
        
        logger.info(f"Quête créée depuis texte : {quest_name}")
        return True
    
    def _add_to_execution_queue(self, quest_id: str):
        """Ajoute une quête à la file d'exécution selon sa priorité."""
        quest = self.active_quests[quest_id]
        
        # Calcul de la priorité
        priority_score = self._calculate_quest_priority(quest)
        
        # Insertion triée dans la queue
        inserted = False
        for i, existing_quest_id in enumerate(self.quest_queue):
            existing_quest = self.active_quests[existing_quest_id]
            existing_priority = self._calculate_quest_priority(existing_quest)
            
            if priority_score > existing_priority:
                self.quest_queue.insert(i, quest_id)
                inserted = True
                break
        
        if not inserted:
            self.quest_queue.append(quest_id)
    
    def _calculate_quest_priority(self, quest: Quest) -> float:
        """Calcule la priorité d'une quête."""
        base_priority = {
            QuestType.DAILY_QUEST: 100,
            QuestType.MAIN_STORY: 80,
            QuestType.DUNGEON_QUEST: 60,
            QuestType.PROFESSION_QUEST: 50,
            QuestType.SIDE_QUEST: 30,
            QuestType.ACHIEVEMENT: 20,
            QuestType.COLLECTION: 10
        }.get(quest.quest_type, 30)
        
        # Modificateurs
        if 'xp' in quest.rewards:
            base_priority += quest.rewards['xp'] / 1000
        
        if 'kamas' in quest.rewards:
            base_priority += quest.rewards['kamas'] / 10000
        
        # Pénalité pour les quêtes longues
        if quest.estimated_duration > 1800:  # Plus de 30 minutes
            base_priority *= 0.8
        
        return base_priority
    
    def start_quest_automation(self, quest_id: Optional[str] = None):
        """Lance l'automatisation des quêtes."""
        if quest_id:
            # Exécution d'une quête spécifique
            if quest_id in self.active_quests:
                self._execute_single_quest(quest_id)
        else:
            # Exécution de toute la file
            self._execute_quest_queue()
    
    def _execute_quest_queue(self):
        """Exécute toutes les quêtes en file."""
        logger.info(f"Début de l'exécution de {len(self.quest_queue)} quêtes")
        
        while self.quest_queue:
            quest_id = self.quest_queue.pop(0)
            
            if quest_id in self.active_quests:
                success = self._execute_single_quest(quest_id)
                
                if success:
                    self.completed_quests.append(quest_id)
                    del self.active_quests[quest_id]
                    self.stats['quests_completed'] += 1
                else:
                    # Remettre en fin de file si échec
                    if self.retry_failed_objectives:
                        self.quest_queue.append(quest_id)
        
        logger.info("Fin de l'exécution des quêtes")
        self._update_success_rate()
    
    def _execute_single_quest(self, quest_id: str) -> bool:
        """Exécute une seule quête."""
        quest = self.active_quests[quest_id]
        logger.info(f"Début d'exécution : {quest.name}")
        
        quest.status = QuestStatus.IN_PROGRESS
        quest.start_time = datetime.now()
        quest.attempts += 1
        
        try:
            start_time = time.time()
            
            # Exécution séquentielle des objectifs
            for i, objective in enumerate(quest.objectives):
                quest.current_objective = i
                
                if not objective.is_completed:
                    success = self.executor.execute_objective(objective)
                    
                    if success:
                        self.stats['objectives_completed'] += 1
                        logger.info(f"Objectif terminé : {objective.description}")
                    else:
                        logger.error(f"Échec objectif : {objective.description}")
                        quest.status = QuestStatus.FAILED
                        quest.error_message = f"Échec objectif {i}"
                        return False
                
                # Vérification du timeout
                elapsed = time.time() - start_time
                if elapsed > self.max_quest_duration:
                    logger.error(f"Timeout quête : {quest.name}")
                    quest.status = QuestStatus.FAILED
                    quest.error_message = "Timeout"
                    return False
            
            # Tous les objectifs sont terminés
            quest.status = QuestStatus.COMPLETED
            quest.completion_time = datetime.now()
            execution_time = int(time.time() - start_time)
            self.stats['total_execution_time'] += execution_time
            
            logger.info(f"Quête terminée : {quest.name} ({execution_time}s)")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de {quest.name} : {e}")
            quest.status = QuestStatus.FAILED
            quest.error_message = str(e)
            self.stats['errors_encountered'] += 1
            return False
    
    def get_quest_status(self, quest_id: str) -> Optional[Dict[str, Any]]:
        """Retourne le statut détaillé d'une quête."""
        if quest_id not in self.active_quests:
            return None
        
        quest = self.active_quests[quest_id]
        
        return {
            'id': quest.id,
            'name': quest.name,
            'status': quest.status.value,
            'progress': {
                'current_objective': quest.current_objective,
                'total_objectives': len(quest.objectives),
                'completed_objectives': sum(1 for obj in quest.objectives if obj.is_completed)
            },
            'objectives': [
                {
                    'description': obj.description,
                    'type': obj.objective_type.value,
                    'target': obj.target,
                    'progress': f"{obj.quantity_current}/{obj.quantity_required}",
                    'completed': obj.is_completed
                }
                for obj in quest.objectives
            ],
            'execution_info': {
                'start_time': quest.start_time.isoformat() if quest.start_time else None,
                'completion_time': quest.completion_time.isoformat() if quest.completion_time else None,
                'attempts': quest.attempts,
                'error_message': quest.error_message
            }
        }
    
    def get_automation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'automatisation."""
        return {
            'active_quests': len(self.active_quests),
            'queued_quests': len(self.quest_queue),
            'completed_quests': len(self.completed_quests),
            'statistics': self.stats.copy()
        }
    
    def _update_success_rate(self):
        """Met à jour le taux de succès."""
        total_attempted = self.stats['quests_completed'] + self.stats['errors_encountered']
        if total_attempted > 0:
            self.stats['success_rate'] = (self.stats['quests_completed'] / total_attempted) * 100
    
    def pause_quest(self, quest_id: str):
        """Met en pause une quête spécifique."""
        if quest_id in self.active_quests:
            quest = self.active_quests[quest_id]
            if quest.status == QuestStatus.IN_PROGRESS:
                quest.status = QuestStatus.AVAILABLE
                logger.info(f"Quête mise en pause : {quest.name}")
    
    def resume_quest(self, quest_id: str):
        """Reprend une quête mise en pause."""
        if quest_id in self.active_quests:
            quest = self.active_quests[quest_id]
            if quest.status == QuestStatus.AVAILABLE:
                self._add_to_execution_queue(quest_id)
                logger.info(f"Quête reprise : {quest.name}")
    
    def abandon_quest(self, quest_id: str):
        """Abandonne une quête."""
        if quest_id in self.active_quests:
            quest = self.active_quests[quest_id]
            quest.status = QuestStatus.ABANDONED
            
            # Retirer de la queue si présent
            if quest_id in self.quest_queue:
                self.quest_queue.remove(quest_id)
            
            logger.info(f"Quête abandonnée : {quest.name}")
    
    def clear_completed_quests(self):
        """Nettoie les quêtes terminées."""
        self.completed_quests.clear()
        logger.info("Quêtes terminées nettoyées")