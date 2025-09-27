"""
Quest Tracker - Système de suivi automatique des quêtes
Intègre OCR, IA et planification pour un suivi intelligent

Auteur: Claude Code
Intégration: TacticalBot + HRM + Quest Management
"""

import cv2
import numpy as np
import re
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class QuestStatus(Enum):
    """États des quêtes"""
    AVAILABLE = "available"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    READY_TO_TURN_IN = "ready_to_turn_in"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

class QuestType(Enum):
    """Types de quêtes"""
    KILL = "kill"
    COLLECT = "collect"
    DELIVER = "deliver"
    ESCORT = "escort"
    EXPLORE = "explore"
    INTERACT = "interact"
    CRAFT = "craft"
    DUNGEON = "dungeon"
    DAILY = "daily"
    WEEKLY = "weekly"
    CHAIN = "chain"

@dataclass
class QuestObjective:
    """Objectif de quête"""
    id: str
    description: str
    type: QuestType
    target: str = ""
    current_progress: int = 0
    required_progress: int = 1
    location: Optional[Tuple[int, int]] = None
    completed: bool = False

    @property
    def progress_percentage(self) -> float:
        """Pourcentage de progression"""
        if self.required_progress == 0:
            return 100.0 if self.completed else 0.0
        return min(100.0, (self.current_progress / self.required_progress) * 100.0)

@dataclass
class Quest:
    """Quête complète"""
    id: str
    name: str
    description: str
    giver_npc: str
    status: QuestStatus
    type: QuestType
    level_requirement: int = 1

    objectives: List[QuestObjective] = field(default_factory=list)
    rewards: Dict[str, Any] = field(default_factory=dict)

    # Métadonnées
    accepted_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    priority: int = 5  # 1-10, 10 = max priorité

    # Navigation
    start_location: Optional[Tuple[int, int]] = None
    turn_in_location: Optional[Tuple[int, int]] = None
    objective_locations: List[Tuple[int, int]] = field(default_factory=list)

    # Tracking
    last_progress_update: datetime = field(default_factory=datetime.now)
    progress_history: List[Dict] = field(default_factory=list)

    @property
    def overall_progress(self) -> float:
        """Progression globale de la quête"""
        if not self.objectives:
            return 100.0 if self.status == QuestStatus.COMPLETED else 0.0

        total_progress = sum(obj.progress_percentage for obj in self.objectives)
        return total_progress / len(self.objectives)

    @property
    def is_ready_to_turn_in(self) -> bool:
        """Prête à être rendue"""
        return all(obj.completed for obj in self.objectives)

class QuestTextExtractor:
    """Extracteur de texte pour les quêtes via OCR"""

    def __init__(self):
        self.quest_patterns = self._load_quest_patterns()
        self.ocr_cache = {}
        self.last_extraction_time = 0

    def _load_quest_patterns(self) -> Dict[str, List[str]]:
        """Charge les patterns de reconnaissance de quêtes"""
        return {
            "quest_names": [
                r"Quest:\s*(.+)",
                r"Mission:\s*(.+)",
                r"Task:\s*(.+)"
            ],
            "objectives": [
                r"Kill\s+(\d+)\s+(.+)",
                r"Collect\s+(\d+)\s+(.+)",
                r"Deliver\s+(.+)\s+to\s+(.+)",
                r"Talk\s+to\s+(.+)",
                r"Go\s+to\s+(.+)",
                r"Find\s+(.+)"
            ],
            "progress": [
                r"(\d+)/(\d+)",
                r"Progress:\s*(\d+)%",
                r"Completed:\s*(\d+)\s*of\s*(\d+)"
            ],
            "rewards": [
                r"Reward:\s*(.+)",
                r"XP:\s*(\d+)",
                r"Gold:\s*(\d+)",
                r"Item:\s*(.+)"
            ]
        }

    def extract_quest_info(self, screenshot: np.ndarray, quest_window_region: Optional[Tuple] = None) -> Dict:
        """Extrait les informations de quête depuis l'écran"""
        current_time = time.time()

        # Cache pour éviter les extractions trop fréquentes
        if current_time - self.last_extraction_time < 1.0:
            return self.ocr_cache

        # Isoler la fenêtre de quêtes si spécifiée
        if quest_window_region:
            x, y, w, h = quest_window_region
            quest_region = screenshot[y:y+h, x:x+w]
        else:
            quest_region = screenshot

        # Préprocessing pour améliorer l'OCR
        processed_image = self._preprocess_for_ocr(quest_region)

        # Extraction de texte
        extracted_text = self._ocr_extract_text(processed_image)

        # Parsing du texte
        quest_info = self._parse_quest_text(extracted_text)

        self.ocr_cache = quest_info
        self.last_extraction_time = current_time
        return quest_info

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Préprocesse l'image pour améliorer l'OCR"""
        # Conversion en grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Débruitage
        denoised = cv2.fastNlMeansDenoising(enhanced)

        # Binarisation adaptative
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        # Morphologie pour nettoyer
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return cleaned

    def _ocr_extract_text(self, image: np.ndarray) -> str:
        """Extrait le texte avec OCR"""
        try:
            import pytesseract

            # Configuration OCR optimisée pour le texte de jeu
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:.,/()- '

            text = pytesseract.image_to_string(image, config=config)
            return text.strip()

        except ImportError:
            logger.warning("pytesseract non disponible - OCR désactivé")
            return ""
        except Exception as e:
            logger.error(f"Erreur OCR: {e}")
            return ""

    def _parse_quest_text(self, text: str) -> Dict:
        """Parse le texte extrait pour identifier les éléments de quête"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        quest_info = {
            "quests": [],
            "objectives": [],
            "progress": [],
            "rewards": []
        }

        for line in lines:
            # Recherche de noms de quêtes
            for pattern in self.quest_patterns["quest_names"]:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    quest_info["quests"].append(match.group(1))

            # Recherche d'objectifs
            for pattern in self.quest_patterns["objectives"]:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    quest_info["objectives"].append({
                        "full_text": line,
                        "groups": match.groups()
                    })

            # Recherche de progression
            for pattern in self.quest_patterns["progress"]:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    quest_info["progress"].append({
                        "full_text": line,
                        "groups": match.groups()
                    })

            # Recherche de récompenses
            for pattern in self.quest_patterns["rewards"]:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    quest_info["rewards"].append({
                        "full_text": line,
                        "groups": match.groups()
                    })

        return quest_info

class QuestDatabase:
    """Base de données des quêtes"""

    def __init__(self, db_path: str = "G:/Botting/data/quests.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialise la base de données"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quests (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    giver_npc TEXT,
                    status TEXT,
                    type TEXT,
                    level_requirement INTEGER DEFAULT 1,
                    priority INTEGER DEFAULT 5,
                    accepted_time TIMESTAMP,
                    deadline TIMESTAMP,
                    start_location TEXT,
                    turn_in_location TEXT,
                    rewards TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS quest_objectives (
                    id TEXT PRIMARY KEY,
                    quest_id TEXT,
                    description TEXT,
                    type TEXT,
                    target TEXT,
                    current_progress INTEGER DEFAULT 0,
                    required_progress INTEGER DEFAULT 1,
                    location TEXT,
                    completed BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (quest_id) REFERENCES quests (id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS quest_progress_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    quest_id TEXT,
                    objective_id TEXT,
                    old_progress INTEGER,
                    new_progress INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (quest_id) REFERENCES quests (id),
                    FOREIGN KEY (objective_id) REFERENCES quest_objectives (id)
                )
            """)

    def save_quest(self, quest: Quest):
        """Sauvegarde une quête"""
        with sqlite3.connect(self.db_path) as conn:
            # Sauvegarder la quête principale
            conn.execute("""
                INSERT OR REPLACE INTO quests
                (id, name, description, giver_npc, status, type, level_requirement,
                 priority, accepted_time, deadline, start_location, turn_in_location,
                 rewards, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                quest.id, quest.name, quest.description, quest.giver_npc,
                quest.status.value, quest.type.value, quest.level_requirement,
                quest.priority, quest.accepted_time, quest.deadline,
                json.dumps(quest.start_location), json.dumps(quest.turn_in_location),
                json.dumps(quest.rewards), datetime.now()
            ))

            # Supprimer les anciens objectifs
            conn.execute("DELETE FROM quest_objectives WHERE quest_id = ?", (quest.id,))

            # Sauvegarder les objectifs
            for obj in quest.objectives:
                conn.execute("""
                    INSERT INTO quest_objectives
                    (id, quest_id, description, type, target, current_progress,
                     required_progress, location, completed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    obj.id, quest.id, obj.description, obj.type.value, obj.target,
                    obj.current_progress, obj.required_progress,
                    json.dumps(obj.location), obj.completed
                ))

    def load_quest(self, quest_id: str) -> Optional[Quest]:
        """Charge une quête depuis la DB"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Charger la quête principale
            quest_row = conn.execute("SELECT * FROM quests WHERE id = ?", (quest_id,)).fetchone()
            if not quest_row:
                return None

            # Charger les objectifs
            objective_rows = conn.execute("SELECT * FROM quest_objectives WHERE quest_id = ?", (quest_id,)).fetchall()

            # Construire les objectifs
            objectives = []
            for obj_row in objective_rows:
                objectives.append(QuestObjective(
                    id=obj_row["id"],
                    description=obj_row["description"],
                    type=QuestType(obj_row["type"]),
                    target=obj_row["target"],
                    current_progress=obj_row["current_progress"],
                    required_progress=obj_row["required_progress"],
                    location=json.loads(obj_row["location"]) if obj_row["location"] else None,
                    completed=bool(obj_row["completed"])
                ))

            # Construire la quête
            quest = Quest(
                id=quest_row["id"],
                name=quest_row["name"],
                description=quest_row["description"],
                giver_npc=quest_row["giver_npc"],
                status=QuestStatus(quest_row["status"]),
                type=QuestType(quest_row["type"]),
                level_requirement=quest_row["level_requirement"],
                priority=quest_row["priority"],
                accepted_time=datetime.fromisoformat(quest_row["accepted_time"]) if quest_row["accepted_time"] else None,
                deadline=datetime.fromisoformat(quest_row["deadline"]) if quest_row["deadline"] else None,
                start_location=json.loads(quest_row["start_location"]) if quest_row["start_location"] else None,
                turn_in_location=json.loads(quest_row["turn_in_location"]) if quest_row["turn_in_location"] else None,
                rewards=json.loads(quest_row["rewards"]) if quest_row["rewards"] else {},
                objectives=objectives
            )

            return quest

    def get_active_quests(self) -> List[Quest]:
        """Récupère toutes les quêtes actives"""
        with sqlite3.connect(self.db_path) as conn:
            active_statuses = [QuestStatus.ACCEPTED.value, QuestStatus.IN_PROGRESS.value, QuestStatus.READY_TO_TURN_IN.value]
            placeholders = ','.join('?' for _ in active_statuses)

            quest_ids = conn.execute(f"SELECT id FROM quests WHERE status IN ({placeholders})", active_statuses).fetchall()

            quests = []
            for (quest_id,) in quest_ids:
                quest = self.load_quest(quest_id)
                if quest:
                    quests.append(quest)

            return quests

class QuestTracker:
    """Tracker principal des quêtes"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()

        # Composants
        self.text_extractor = QuestTextExtractor()
        self.database = QuestDatabase(self.config["database_path"])

        # État du tracking
        self.active_quests: Dict[str, Quest] = {}
        self.quest_history: List[Dict] = []
        self.last_update_time = time.time()

        # Cache pour optimisation
        self.screenshot_cache = None
        self.cache_timestamp = 0

        # Chargement des quêtes actives
        self._load_active_quests()

        logger.info("Quest Tracker initialisé")

    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            "database_path": "G:/Botting/data/quests.db",
            "update_interval": 5.0,  # secondes
            "quest_window_region": None,  # Auto-detect
            "auto_accept_quests": False,
            "auto_turn_in_quests": True,
            "priority_threshold": 7,  # Seuil pour quêtes prioritaires
            "max_concurrent_quests": 10
        }

    def update_from_screenshot(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Met à jour le tracking depuis une capture d'écran"""
        current_time = time.time()

        # Vérifier l'intervalle de mise à jour
        if current_time - self.last_update_time < self.config["update_interval"]:
            return self._get_current_status()

        # Extraire les informations de quête
        quest_info = self.text_extractor.extract_quest_info(
            screenshot,
            self.config["quest_window_region"]
        )

        # Traiter les informations extraites
        self._process_extracted_info(quest_info)

        # Mettre à jour les quêtes existantes
        self._update_quest_progress(quest_info)

        # Détecter de nouvelles quêtes
        self._detect_new_quests(quest_info)

        # Nettoyer les quêtes terminées
        self._cleanup_completed_quests()

        self.last_update_time = current_time

        return self._get_current_status()

    def _load_active_quests(self):
        """Charge les quêtes actives depuis la DB"""
        active_quests = self.database.get_active_quests()
        for quest in active_quests:
            self.active_quests[quest.id] = quest

        logger.info(f"Chargé {len(active_quests)} quêtes actives")

    def _process_extracted_info(self, quest_info: Dict):
        """Traite les informations extraites"""
        # Log des informations détectées
        if quest_info["quests"]:
            logger.debug(f"Quêtes détectées: {quest_info['quests']}")

        if quest_info["progress"]:
            logger.debug(f"Progression détectée: {quest_info['progress']}")

    def _update_quest_progress(self, quest_info: Dict):
        """Met à jour la progression des quêtes"""
        for progress_data in quest_info["progress"]:
            # Tenter de matcher avec les quêtes actives
            for quest in self.active_quests.values():
                updated = self._try_update_quest_progress(quest, progress_data)
                if updated:
                    self.database.save_quest(quest)

    def _try_update_quest_progress(self, quest: Quest, progress_data: Dict) -> bool:
        """Tente de mettre à jour la progression d'une quête"""
        text = progress_data["full_text"].lower()
        groups = progress_data["groups"]

        # Pattern numérique simple (ex: "5/10")
        if len(groups) == 2 and groups[0].isdigit() and groups[1].isdigit():
            current = int(groups[0])
            required = int(groups[1])

            # Chercher l'objectif correspondant
            for obj in quest.objectives:
                if not obj.completed and obj.required_progress == required:
                    old_progress = obj.current_progress
                    obj.current_progress = current
                    obj.completed = (current >= required)

                    if old_progress != current:
                        quest.progress_history.append({
                            "objective_id": obj.id,
                            "old_progress": old_progress,
                            "new_progress": current,
                            "timestamp": datetime.now().isoformat()
                        })
                        return True

        return False

    def _detect_new_quests(self, quest_info: Dict):
        """Détecte de nouvelles quêtes"""
        for quest_name in quest_info["quests"]:
            # Vérifier si cette quête n'existe pas déjà
            quest_id = self._generate_quest_id(quest_name)

            if quest_id not in self.active_quests:
                # Créer une nouvelle quête
                new_quest = self._create_quest_from_name(quest_id, quest_name, quest_info)
                if new_quest:
                    self.active_quests[quest_id] = new_quest
                    self.database.save_quest(new_quest)
                    logger.info(f"Nouvelle quête détectée: {quest_name}")

    def _generate_quest_id(self, quest_name: str) -> str:
        """Génère un ID unique pour une quête"""
        import hashlib
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', quest_name.lower())
        hash_suffix = hashlib.md5(quest_name.encode()).hexdigest()[:8]
        return f"quest_{clean_name}_{hash_suffix}"

    def _create_quest_from_name(self, quest_id: str, quest_name: str, quest_info: Dict) -> Optional[Quest]:
        """Crée une quête depuis les informations détectées"""
        try:
            # Inférer le type de quête depuis le nom/objectifs
            quest_type = self._infer_quest_type(quest_name, quest_info)

            # Créer les objectifs depuis les informations extraites
            objectives = self._create_objectives_from_info(quest_id, quest_info)

            quest = Quest(
                id=quest_id,
                name=quest_name,
                description=f"Quête auto-détectée: {quest_name}",
                giver_npc="Inconnu",
                status=QuestStatus.ACCEPTED,
                type=quest_type,
                objectives=objectives,
                accepted_time=datetime.now(),
                priority=5
            )

            return quest

        except Exception as e:
            logger.error(f"Erreur création quête {quest_name}: {e}")
            return None

    def _infer_quest_type(self, quest_name: str, quest_info: Dict) -> QuestType:
        """Infère le type de quête"""
        name_lower = quest_name.lower()

        # Patterns simples de détection
        if any(word in name_lower for word in ["kill", "slay", "defeat", "eliminate"]):
            return QuestType.KILL
        elif any(word in name_lower for word in ["collect", "gather", "find", "obtain"]):
            return QuestType.COLLECT
        elif any(word in name_lower for word in ["deliver", "bring", "take", "transport"]):
            return QuestType.DELIVER
        elif any(word in name_lower for word in ["talk", "speak", "meet", "visit"]):
            return QuestType.INTERACT
        elif any(word in name_lower for word in ["explore", "discover", "scout"]):
            return QuestType.EXPLORE
        elif "daily" in name_lower:
            return QuestType.DAILY
        elif "weekly" in name_lower:
            return QuestType.WEEKLY
        else:
            return QuestType.KILL  # Type par défaut

    def _create_objectives_from_info(self, quest_id: str, quest_info: Dict) -> List[QuestObjective]:
        """Crée les objectifs depuis les informations extraites"""
        objectives = []

        for i, obj_data in enumerate(quest_info["objectives"]):
            obj_id = f"{quest_id}_obj_{i}"
            description = obj_data["full_text"]

            # Inférer le type d'objectif
            obj_type = self._infer_objective_type(description)

            # Extraire target et progression si possible
            groups = obj_data["groups"]
            if len(groups) >= 2 and groups[0].isdigit():
                required_progress = int(groups[0])
                target = groups[1] if len(groups) > 1 else ""
            else:
                required_progress = 1
                target = groups[0] if groups else ""

            objectives.append(QuestObjective(
                id=obj_id,
                description=description,
                type=obj_type,
                target=target,
                required_progress=required_progress
            ))

        return objectives

    def _infer_objective_type(self, description: str) -> QuestType:
        """Infère le type d'objectif"""
        desc_lower = description.lower()

        if any(word in desc_lower for word in ["kill", "slay", "defeat"]):
            return QuestType.KILL
        elif any(word in desc_lower for word in ["collect", "gather", "find"]):
            return QuestType.COLLECT
        elif any(word in desc_lower for word in ["deliver", "bring", "take"]):
            return QuestType.DELIVER
        elif any(word in desc_lower for word in ["talk", "speak", "meet"]):
            return QuestType.INTERACT
        else:
            return QuestType.INTERACT

    def _cleanup_completed_quests(self):
        """Nettoie les quêtes terminées"""
        completed_quests = []

        for quest_id, quest in self.active_quests.items():
            if quest.is_ready_to_turn_in and quest.status != QuestStatus.READY_TO_TURN_IN:
                quest.status = QuestStatus.READY_TO_TURN_IN
                self.database.save_quest(quest)
                logger.info(f"Quête prête à rendre: {quest.name}")

            elif quest.status == QuestStatus.COMPLETED:
                completed_quests.append(quest_id)

        # Retirer les quêtes terminées de la liste active
        for quest_id in completed_quests:
            del self.active_quests[quest_id]

    def _get_current_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel du tracking"""
        return {
            "active_quests_count": len(self.active_quests),
            "active_quests": [
                {
                    "id": quest.id,
                    "name": quest.name,
                    "status": quest.status.value,
                    "progress": quest.overall_progress,
                    "priority": quest.priority,
                    "ready_to_turn_in": quest.is_ready_to_turn_in
                }
                for quest in self.active_quests.values()
            ],
            "priority_quests": [
                quest for quest in self.active_quests.values()
                if quest.priority >= self.config["priority_threshold"]
            ],
            "last_update": self.last_update_time
        }

    # API publique
    def get_next_quest_action(self) -> Optional[Dict]:
        """Retourne la prochaine action de quête recommandée"""
        if not self.active_quests:
            return None

        # Prioriser les quêtes prêtes à rendre
        ready_quests = [q for q in self.active_quests.values() if q.is_ready_to_turn_in]
        if ready_quests:
            quest = max(ready_quests, key=lambda q: q.priority)
            return {
                "action": "turn_in_quest",
                "quest_id": quest.id,
                "quest_name": quest.name,
                "location": quest.turn_in_location,
                "priority": DecisionPriority.HIGH.value
            }

        # Sinon, continuer la quête avec la plus haute priorité
        active_quests = [q for q in self.active_quests.values() if q.status == QuestStatus.IN_PROGRESS]
        if active_quests:
            quest = max(active_quests, key=lambda q: q.priority)
            next_objective = next((obj for obj in quest.objectives if not obj.completed), None)

            if next_objective:
                return {
                    "action": "continue_quest",
                    "quest_id": quest.id,
                    "quest_name": quest.name,
                    "objective": next_objective.description,
                    "location": next_objective.location,
                    "priority": DecisionPriority.NORMAL.value
                }

        return None

    def force_complete_quest(self, quest_id: str) -> bool:
        """Force la completion d'une quête"""
        if quest_id in self.active_quests:
            quest = self.active_quests[quest_id]
            quest.status = QuestStatus.COMPLETED
            for obj in quest.objectives:
                obj.completed = True
                obj.current_progress = obj.required_progress

            self.database.save_quest(quest)
            del self.active_quests[quest_id]
            logger.info(f"Quête forcée complète: {quest.name}")
            return True

        return False

    def get_quest_statistics(self) -> Dict:
        """Retourne les statistiques des quêtes"""
        with sqlite3.connect(self.database.db_path) as conn:
            stats = {}

            # Quêtes par statut
            for status in QuestStatus:
                count = conn.execute("SELECT COUNT(*) FROM quests WHERE status = ?", (status.value,)).fetchone()[0]
                stats[f"quests_{status.value}"] = count

            # Quêtes par type
            for quest_type in QuestType:
                count = conn.execute("SELECT COUNT(*) FROM quests WHERE type = ?", (quest_type.value,)).fetchone()[0]
                stats[f"quests_{quest_type.value}"] = count

            # Statistiques générales
            stats["total_quests"] = conn.execute("SELECT COUNT(*) FROM quests").fetchone()[0]
            stats["active_quests"] = len(self.active_quests)

            return stats