#!/usr/bin/env python3
"""
GuideLoader - Chargeur et interpréteur de guides DOFUS
Supporte JSON, XML et formats de guides personnalisés
"""

import json
import xml.etree.ElementTree as ET
import yaml
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from pathlib import Path
import re

from config import config

logger = logging.getLogger(__name__)

class GuideType(Enum):
    """Types de guides"""
    QUEST_GUIDE = "quest_guide"
    LEVELING_GUIDE = "leveling_guide"
    FARMING_GUIDE = "farming_guide"
    DUNGEON_GUIDE = "dungeon_guide"
    PVP_GUIDE = "pvp_guide"
    PROFESSION_GUIDE = "profession_guide"
    ACHIEVEMENT_GUIDE = "achievement_guide"
    EXPLORATION_GUIDE = "exploration_guide"

class GuideStepType(Enum):
    """Types d'étapes de guide"""
    MOVE_TO = "move_to"
    TALK_TO_NPC = "talk_to_npc"
    KILL_MONSTERS = "kill_monsters"
    COLLECT_ITEMS = "collect_items"
    USE_ITEM = "use_item"
    EQUIP_ITEM = "equip_item"
    LEVEL_UP = "level_up"
    LEARN_SPELL = "learn_spell"
    BANK_ITEMS = "bank_items"
    SELL_ITEMS = "sell_items"
    BUY_ITEMS = "buy_items"
    WAIT = "wait"
    CONDITION = "condition"
    REPEAT = "repeat"
    CUSTOM = "custom"

class GuideConditionType(Enum):
    """Types de conditions"""
    LEVEL_REQUIREMENT = "level_requirement"
    ITEM_REQUIREMENT = "item_requirement"
    QUEST_COMPLETION = "quest_completion"
    KAMAS_REQUIREMENT = "kamas_requirement"
    TIME_CONDITION = "time_condition"
    WEATHER_CONDITION = "weather_condition"

@dataclass
class GuideCondition:
    """Condition pour exécution d'étape"""
    condition_type: GuideConditionType
    operator: str  # ==, !=, >=, <=, >, <
    value: Any
    description: str = ""

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Évalue la condition avec le contexte donné"""
        try:
            if self.condition_type == GuideConditionType.LEVEL_REQUIREMENT:
                player_level = context.get("player_level", 1)
                return self._compare_values(player_level, self.operator, self.value)

            elif self.condition_type == GuideConditionType.ITEM_REQUIREMENT:
                inventory = context.get("inventory", [])
                has_item = any(item.get("name") == self.value for item in inventory)
                return has_item if self.operator == "==" else not has_item

            elif self.condition_type == GuideConditionType.QUEST_COMPLETION:
                completed_quests = context.get("completed_quests", [])
                quest_completed = self.value in completed_quests
                return quest_completed if self.operator == "==" else not quest_completed

            elif self.condition_type == GuideConditionType.KAMAS_REQUIREMENT:
                player_kamas = context.get("kamas", 0)
                return self._compare_values(player_kamas, self.operator, self.value)

            elif self.condition_type == GuideConditionType.TIME_CONDITION:
                current_time = context.get("game_time", time.time())
                return self._compare_values(current_time, self.operator, self.value)

            return False

        except Exception as e:
            logger.warning(f"Erreur évaluation condition: {e}")
            return False

    def _compare_values(self, actual: Any, operator: str, expected: Any) -> bool:
        """Compare deux valeurs selon l'opérateur"""
        if operator == "==":
            return actual == expected
        elif operator == "!=":
            return actual != expected
        elif operator == ">=":
            return actual >= expected
        elif operator == "<=":
            return actual <= expected
        elif operator == ">":
            return actual > expected
        elif operator == "<":
            return actual < expected
        else:
            return False

@dataclass
class GuideReward:
    """Récompense d'étape ou de guide"""
    experience: int = 0
    kamas: int = 0
    items: List[str] = field(default_factory=list)
    reputation: Dict[str, int] = field(default_factory=dict)

@dataclass
class GuideStep:
    """Étape d'un guide"""
    step_id: str
    step_type: GuideStepType
    title: str
    description: str

    # Paramètres spécifiques par type
    target: Optional[str] = None
    location: Optional[Tuple[int, int]] = None
    map_id: Optional[str] = None
    quantity: int = 1
    items: List[str] = field(default_factory=list)

    # Conditions et logique
    conditions: List[GuideCondition] = field(default_factory=list)
    optional: bool = False
    skippable: bool = False

    # Méta-informations
    estimated_time: float = 0.0
    difficulty: str = "easy"  # easy, medium, hard
    notes: str = ""
    tips: List[str] = field(default_factory=list)

    # Récompenses
    rewards: Optional[GuideReward] = None

    # Logique de répétition
    repeat_count: int = 1
    repeat_condition: Optional[GuideCondition] = None

    def is_executable(self, context: Dict[str, Any]) -> bool:
        """Vérifie si l'étape peut être exécutée"""
        for condition in self.conditions:
            if not condition.evaluate(context):
                return False
        return True

    def should_repeat(self, context: Dict[str, Any], current_iteration: int) -> bool:
        """Vérifie si l'étape doit être répétée"""
        if current_iteration >= self.repeat_count:
            return False

        if self.repeat_condition:
            return self.repeat_condition.evaluate(context)

        return current_iteration < self.repeat_count

@dataclass
class Guide:
    """Guide complet"""
    guide_id: str
    title: str
    description: str
    guide_type: GuideType
    version: str = "1.0"

    # Métadonnées
    author: str = ""
    created_date: str = ""
    last_updated: str = ""
    language: str = "fr"

    # Prérequis
    level_requirement: int = 1
    required_quests: List[str] = field(default_factory=list)
    required_items: List[str] = field(default_factory=list)
    required_kamas: int = 0

    # Contenu
    steps: List[GuideStep] = field(default_factory=list)

    # Estimations
    estimated_completion_time: float = 0.0
    estimated_experience_gain: int = 0
    estimated_kamas_gain: int = 0

    # Paramètres d'optimisation
    priority_score: float = 1.0
    difficulty_rating: float = 1.0

    # Tags et catégories
    tags: List[str] = field(default_factory=list)
    category: str = ""

    @property
    def total_steps(self) -> int:
        """Nombre total d'étapes"""
        return len(self.steps)

    @property
    def executable_steps(self) -> int:
        """Nombre d'étapes exécutables"""
        return len([step for step in self.steps if not step.optional])

    def get_next_step(self, current_step_index: int, context: Dict[str, Any]) -> Optional[GuideStep]:
        """Récupère la prochaine étape exécutable"""
        for i in range(current_step_index, len(self.steps)):
            step = self.steps[i]
            if step.is_executable(context):
                return step
        return None

    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Vérifie si le guide peut être exécuté"""
        player_level = context.get("player_level", 1)
        if player_level < self.level_requirement:
            return False

        completed_quests = context.get("completed_quests", [])
        for required_quest in self.required_quests:
            if required_quest not in completed_quests:
                return False

        player_kamas = context.get("kamas", 0)
        if player_kamas < self.required_kamas:
            return False

        inventory = context.get("inventory", [])
        inventory_items = [item.get("name", "") for item in inventory]
        for required_item in self.required_items:
            if required_item not in inventory_items:
                return False

        return True

class GuideParser:
    """Parser pour différents formats de guides"""

    def __init__(self):
        # Patterns de parsing
        self.coordinate_pattern = r'\((\d+),\s*(\d+)\)'
        self.quantity_pattern = r'(\d+)x?\s+(.*)'
        self.level_pattern = r'level?\s*(\d+)'

    def parse_json_guide(self, file_path: Path) -> Guide:
        """Parse un guide JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return self._create_guide_from_dict(data)

        except Exception as e:
            logger.error(f"Erreur parsing JSON {file_path}: {e}")
            raise

    def parse_xml_guide(self, file_path: Path) -> Guide:
        """Parse un guide XML"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Convertir XML en dict
            data = self._xml_to_dict(root)
            return self._create_guide_from_dict(data)

        except Exception as e:
            logger.error(f"Erreur parsing XML {file_path}: {e}")
            raise

    def parse_yaml_guide(self, file_path: Path) -> Guide:
        """Parse un guide YAML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            return self._create_guide_from_dict(data)

        except Exception as e:
            logger.error(f"Erreur parsing YAML {file_path}: {e}")
            raise

    def parse_text_guide(self, file_path: Path) -> Guide:
        """Parse un guide texte simple"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return self._parse_text_content(content, file_path.stem)

        except Exception as e:
            logger.error(f"Erreur parsing texte {file_path}: {e}")
            raise

    def _create_guide_from_dict(self, data: Dict[str, Any]) -> Guide:
        """Crée un Guide depuis un dictionnaire"""

        # Métadonnées de base
        guide = Guide(
            guide_id=data.get("guide_id", f"guide_{int(time.time())}"),
            title=data.get("title", "Guide sans titre"),
            description=data.get("description", ""),
            guide_type=GuideType(data.get("type", "quest_guide")),
            version=data.get("version", "1.0"),
            author=data.get("author", ""),
            created_date=data.get("created_date", ""),
            last_updated=data.get("last_updated", ""),
            language=data.get("language", "fr")
        )

        # Prérequis
        requirements = data.get("requirements", {})
        guide.level_requirement = requirements.get("level", 1)
        guide.required_quests = requirements.get("quests", [])
        guide.required_items = requirements.get("items", [])
        guide.required_kamas = requirements.get("kamas", 0)

        # Estimations
        estimates = data.get("estimates", {})
        guide.estimated_completion_time = estimates.get("time", 0.0)
        guide.estimated_experience_gain = estimates.get("experience", 0)
        guide.estimated_kamas_gain = estimates.get("kamas", 0)

        # Métadonnées supplémentaires
        guide.priority_score = data.get("priority", 1.0)
        guide.difficulty_rating = data.get("difficulty_rating", 1.0)
        guide.tags = data.get("tags", [])
        guide.category = data.get("category", "")

        # Étapes
        steps_data = data.get("steps", [])
        for step_data in steps_data:
            step = self._create_step_from_dict(step_data)
            guide.steps.append(step)

        return guide

    def _create_step_from_dict(self, data: Dict[str, Any]) -> GuideStep:
        """Crée un GuideStep depuis un dictionnaire"""

        step = GuideStep(
            step_id=data.get("step_id", f"step_{int(time.time())}"),
            step_type=GuideStepType(data.get("type", "custom")),
            title=data.get("title", "Étape sans titre"),
            description=data.get("description", ""),
            target=data.get("target"),
            quantity=data.get("quantity", 1),
            items=data.get("items", []),
            optional=data.get("optional", False),
            skippable=data.get("skippable", False),
            estimated_time=data.get("estimated_time", 0.0),
            difficulty=data.get("difficulty", "easy"),
            notes=data.get("notes", ""),
            tips=data.get("tips", []),
            repeat_count=data.get("repeat_count", 1)
        )

        # Location parsing
        if "location" in data:
            location_data = data["location"]
            if isinstance(location_data, list) and len(location_data) == 2:
                step.location = tuple(location_data)
            elif isinstance(location_data, str):
                match = re.search(self.coordinate_pattern, location_data)
                if match:
                    step.location = (int(match.group(1)), int(match.group(2)))

        # Map ID
        step.map_id = data.get("map_id") or data.get("map")

        # Conditions
        conditions_data = data.get("conditions", [])
        for condition_data in conditions_data:
            condition = GuideCondition(
                condition_type=GuideConditionType(condition_data["type"]),
                operator=condition_data.get("operator", "=="),
                value=condition_data["value"],
                description=condition_data.get("description", "")
            )
            step.conditions.append(condition)

        # Repeat condition
        if "repeat_condition" in data:
            repeat_data = data["repeat_condition"]
            step.repeat_condition = GuideCondition(
                condition_type=GuideConditionType(repeat_data["type"]),
                operator=repeat_data.get("operator", "=="),
                value=repeat_data["value"],
                description=repeat_data.get("description", "")
            )

        # Récompenses
        if "rewards" in data:
            rewards_data = data["rewards"]
            step.rewards = GuideReward(
                experience=rewards_data.get("experience", 0),
                kamas=rewards_data.get("kamas", 0),
                items=rewards_data.get("items", []),
                reputation=rewards_data.get("reputation", {})
            )

        return step

    def _parse_text_content(self, content: str, guide_name: str) -> Guide:
        """Parse contenu texte simple"""

        lines = [line.strip() for line in content.split('\n') if line.strip()]

        guide = Guide(
            guide_id=f"text_{guide_name}",
            title=guide_name.replace('_', ' ').title(),
            description="Guide importé depuis texte",
            guide_type=GuideType.QUEST_GUIDE
        )

        current_step_num = 1

        for line in lines:
            # Ignorer commentaires
            if line.startswith('#') or line.startswith('//'):
                continue

            # Détecter étapes
            if any(keyword in line.lower() for keyword in ['goto', 'aller', 'move', 'talk', 'parler', 'kill', 'tuer', 'collect', 'collecter']):
                step = self._parse_text_step(line, current_step_num)
                guide.steps.append(step)
                current_step_num += 1

        return guide

    def _parse_text_step(self, line: str, step_num: int) -> GuideStep:
        """Parse une ligne de texte en étape"""

        line_lower = line.lower()

        # Déterminer type d'étape
        if any(keyword in line_lower for keyword in ['goto', 'aller', 'move']):
            step_type = GuideStepType.MOVE_TO
        elif any(keyword in line_lower for keyword in ['talk', 'parler']):
            step_type = GuideStepType.TALK_TO_NPC
        elif any(keyword in line_lower for keyword in ['kill', 'tuer']):
            step_type = GuideStepType.KILL_MONSTERS
        elif any(keyword in line_lower for keyword in ['collect', 'collecter']):
            step_type = GuideStepType.COLLECT_ITEMS
        else:
            step_type = GuideStepType.CUSTOM

        # Extraire quantité
        quantity = 1
        quantity_match = re.search(self.quantity_pattern, line)
        if quantity_match:
            quantity = int(quantity_match.group(1))

        # Extraire coordonnées
        location = None
        coord_match = re.search(self.coordinate_pattern, line)
        if coord_match:
            location = (int(coord_match.group(1)), int(coord_match.group(2)))

        return GuideStep(
            step_id=f"text_step_{step_num}",
            step_type=step_type,
            title=f"Étape {step_num}",
            description=line,
            quantity=quantity,
            location=location
        )

    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convertit élément XML en dictionnaire"""
        result = {}

        # Attributs
        if element.attrib:
            result.update(element.attrib)

        # Texte
        if element.text and element.text.strip():
            if len(element) == 0:
                return element.text.strip()
            result['text'] = element.text.strip()

        # Enfants
        for child in element:
            child_data = self._xml_to_dict(child)

            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        return result

class GuideLoader:
    """Chargeur principal de guides"""

    def __init__(self, guides_directory: str = "data/guides"):
        self.guides_directory = Path(guides_directory)
        self.parser = GuideParser()

        # Cache des guides
        self.loaded_guides: Dict[str, Guide] = {}
        self.guide_index: Dict[str, List[str]] = {}  # type -> guide_ids

        # Statistiques
        self.total_guides_loaded = 0
        self.loading_errors = 0

        # Créer répertoire si nécessaire
        self.guides_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"GuideLoader initialisé (répertoire: {self.guides_directory})")

    def load_all_guides(self) -> Dict[str, Guide]:
        """Charge tous les guides du répertoire"""

        self.loaded_guides.clear()
        self.guide_index.clear()

        supported_extensions = ['.json', '.xml', '.yml', '.yaml', '.txt']

        for guide_file in self.guides_directory.rglob('*'):
            if guide_file.suffix.lower() in supported_extensions:
                try:
                    guide = self._load_guide_file(guide_file)
                    self.loaded_guides[guide.guide_id] = guide

                    # Indexer par type
                    guide_type = guide.guide_type.value
                    if guide_type not in self.guide_index:
                        self.guide_index[guide_type] = []
                    self.guide_index[guide_type].append(guide.guide_id)

                    self.total_guides_loaded += 1

                except Exception as e:
                    logger.error(f"Erreur chargement {guide_file}: {e}")
                    self.loading_errors += 1

        logger.info(f"Guides chargés: {self.total_guides_loaded}, erreurs: {self.loading_errors}")
        return self.loaded_guides

    def _load_guide_file(self, file_path: Path) -> Guide:
        """Charge un fichier de guide spécifique"""

        extension = file_path.suffix.lower()

        if extension == '.json':
            return self.parser.parse_json_guide(file_path)
        elif extension == '.xml':
            return self.parser.parse_xml_guide(file_path)
        elif extension in ['.yml', '.yaml']:
            return self.parser.parse_yaml_guide(file_path)
        elif extension == '.txt':
            return self.parser.parse_text_guide(file_path)
        else:
            raise ValueError(f"Format de fichier non supporté: {extension}")

    def get_guide(self, guide_id: str) -> Optional[Guide]:
        """Récupère un guide par ID"""
        return self.loaded_guides.get(guide_id)

    def get_guides_by_type(self, guide_type: GuideType) -> List[Guide]:
        """Récupère guides par type"""
        guide_ids = self.guide_index.get(guide_type.value, [])
        return [self.loaded_guides[gid] for gid in guide_ids if gid in self.loaded_guides]

    def find_guides_by_level(self, player_level: int, tolerance: int = 5) -> List[Guide]:
        """Trouve guides appropriés pour un niveau"""
        suitable_guides = []

        for guide in self.loaded_guides.values():
            level_diff = abs(guide.level_requirement - player_level)
            if level_diff <= tolerance:
                suitable_guides.append(guide)

        # Trier par proximité de niveau
        suitable_guides.sort(key=lambda g: abs(g.level_requirement - player_level))
        return suitable_guides

    def find_guides_by_tags(self, tags: List[str]) -> List[Guide]:
        """Trouve guides par tags"""
        matching_guides = []

        for guide in self.loaded_guides.values():
            if any(tag in guide.tags for tag in tags):
                matching_guides.append(guide)

        return matching_guides

    def search_guides(self, query: str) -> List[Guide]:
        """Recherche guides par titre/description"""
        query_lower = query.lower()
        matching_guides = []

        for guide in self.loaded_guides.values():
            if (query_lower in guide.title.lower() or
                query_lower in guide.description.lower() or
                any(query_lower in tag.lower() for tag in guide.tags)):
                matching_guides.append(guide)

        return matching_guides

    def get_recommended_guides(self, context: Dict[str, Any]) -> List[Guide]:
        """Recommande guides selon contexte joueur"""

        player_level = context.get("player_level", 1)
        completed_quests = context.get("completed_quests", [])
        preferences = context.get("preferences", {})

        recommended = []

        for guide in self.loaded_guides.values():
            # Vérifier si exécutable
            if not guide.can_execute(context):
                continue

            # Score de recommandation
            score = 0.0

            # Bonus niveau approprié
            level_diff = abs(guide.level_requirement - player_level)
            if level_diff <= 3:
                score += 3.0 - level_diff

            # Bonus type préféré
            preferred_types = preferences.get("guide_types", [])
            if guide.guide_type.value in preferred_types:
                score += 2.0

            # Bonus efficacité
            if guide.estimated_experience_gain > 0:
                exp_per_hour = guide.estimated_experience_gain / max(guide.estimated_completion_time, 1)
                score += min(2.0, exp_per_hour / 1000)

            # Bonus difficulté appropriée
            preferred_difficulty = preferences.get("difficulty", "medium")
            if guide.difficulty_rating <= 2.0 and preferred_difficulty == "easy":
                score += 1.0
            elif 2.0 < guide.difficulty_rating <= 4.0 and preferred_difficulty == "medium":
                score += 1.0
            elif guide.difficulty_rating > 4.0 and preferred_difficulty == "hard":
                score += 1.0

            if score > 0:
                guide.priority_score = score
                recommended.append(guide)

        # Trier par score
        recommended.sort(key=lambda g: g.priority_score, reverse=True)
        return recommended[:10]  # Top 10

    def create_custom_guide(self, guide_data: Dict[str, Any]) -> Guide:
        """Crée guide personnalisé depuis données"""
        guide = self.parser._create_guide_from_dict(guide_data)
        self.loaded_guides[guide.guide_id] = guide

        # Indexer
        guide_type = guide.guide_type.value
        if guide_type not in self.guide_index:
            self.guide_index[guide_type] = []
        self.guide_index[guide_type].append(guide.guide_id)

        return guide

    def save_guide(self, guide: Guide, file_path: Optional[Path] = None) -> bool:
        """Sauvegarde guide en JSON"""
        try:
            if not file_path:
                file_path = self.guides_directory / f"{guide.guide_id}.json"

            guide_dict = self._guide_to_dict(guide)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(guide_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Guide sauvegardé: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur sauvegarde guide: {e}")
            return False

    def _guide_to_dict(self, guide: Guide) -> Dict[str, Any]:
        """Convertit Guide en dictionnaire"""
        return {
            "guide_id": guide.guide_id,
            "title": guide.title,
            "description": guide.description,
            "type": guide.guide_type.value,
            "version": guide.version,
            "author": guide.author,
            "created_date": guide.created_date,
            "last_updated": guide.last_updated,
            "language": guide.language,
            "requirements": {
                "level": guide.level_requirement,
                "quests": guide.required_quests,
                "items": guide.required_items,
                "kamas": guide.required_kamas
            },
            "estimates": {
                "time": guide.estimated_completion_time,
                "experience": guide.estimated_experience_gain,
                "kamas": guide.estimated_kamas_gain
            },
            "priority": guide.priority_score,
            "difficulty_rating": guide.difficulty_rating,
            "tags": guide.tags,
            "category": guide.category,
            "steps": [self._step_to_dict(step) for step in guide.steps]
        }

    def _step_to_dict(self, step: GuideStep) -> Dict[str, Any]:
        """Convertit GuideStep en dictionnaire"""
        step_dict = {
            "step_id": step.step_id,
            "type": step.step_type.value,
            "title": step.title,
            "description": step.description,
            "quantity": step.quantity,
            "optional": step.optional,
            "skippable": step.skippable,
            "estimated_time": step.estimated_time,
            "difficulty": step.difficulty,
            "notes": step.notes,
            "tips": step.tips,
            "repeat_count": step.repeat_count
        }

        if step.target:
            step_dict["target"] = step.target
        if step.location:
            step_dict["location"] = list(step.location)
        if step.map_id:
            step_dict["map_id"] = step.map_id
        if step.items:
            step_dict["items"] = step.items

        return step_dict

    def get_loading_stats(self) -> Dict[str, Any]:
        """Statistiques de chargement"""
        return {
            "total_guides": len(self.loaded_guides),
            "guides_by_type": {gtype: len(guides) for gtype, guides in self.guide_index.items()},
            "total_loaded": self.total_guides_loaded,
            "loading_errors": self.loading_errors,
            "guides_directory": str(self.guides_directory)
        }

def create_guide_loader(guides_directory: str = "data/guides") -> GuideLoader:
    """Factory function pour créer un GuideLoader"""
    return GuideLoader(guides_directory)