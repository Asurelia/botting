"""
Learning Engine Personnalisé pour DOFUS Unity World Model
Système d'apprentissage adaptatif basé sur les patterns de gameplay
Intégration ML avec HRM et Knowledge Base pour amélioration continue
"""

import numpy as np
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import pickle

@dataclass
class LearningSession:
    """Session d'apprentissage avec métriques de performance"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    player_class: str
    player_level: int
    server: str
    total_actions: int
    successful_actions: int
    combat_sessions: int
    combat_wins: int
    economic_gains: float
    experience_gained: int
    efficiency_score: float
    learned_patterns: List[str]

@dataclass
class ActionPattern:
    """Pattern d'action appris depuis les sessions"""
    pattern_id: str
    context: Dict[str, Any]  # État du jeu quand le pattern est utilisé
    action_sequence: List[Dict]  # Séquence d'actions
    success_rate: float
    usage_frequency: int
    avg_execution_time: float
    conditions: Dict[str, Any]  # Conditions pour appliquer ce pattern
    confidence_level: float

@dataclass
class LearningMetrics:
    """Métriques de performance de l'apprentissage"""
    total_sessions: int
    avg_session_duration: float
    improvement_rate: float
    pattern_accuracy: float
    adaptation_speed: float
    knowledge_coverage: float

class AdaptiveLearningEngine:
    """Moteur d'apprentissage adaptatif pour gameplay DOFUS"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "learning_engine/learning_database.sqlite"
        self.patterns_cache: Dict[str, ActionPattern] = {}
        self.active_session: Optional[LearningSession] = None

        # Métriques en temps réel
        self.session_metrics = {
            "actions_taken": 0,
            "decisions_quality": deque(maxlen=100),
            "response_times": deque(maxlen=50),
            "success_feedback": deque(maxlen=200)
        }

        # Configuration d'apprentissage
        self.learning_config = {
            "min_pattern_frequency": 3,  # Minimum pour considérer un pattern
            "confidence_threshold": 0.7,  # Seuil de confiance pour utiliser un pattern
            "adaptation_rate": 0.1,  # Vitesse d'adaptation aux nouveaux patterns
            "forgetting_factor": 0.95,  # Facteur d'oubli pour patterns obsolètes
            "exploration_rate": 0.15  # Taux d'exploration vs exploitation
        }

        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        self._load_existing_patterns()

    def _initialize_database(self):
        """Initialise la base de données d'apprentissage"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Table des sessions d'apprentissage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    player_class TEXT,
                    player_level INTEGER,
                    server TEXT,
                    total_actions INTEGER,
                    successful_actions INTEGER,
                    combat_sessions INTEGER,
                    combat_wins INTEGER,
                    economic_gains REAL,
                    experience_gained INTEGER,
                    efficiency_score REAL,
                    learned_patterns TEXT,
                    session_data TEXT
                )
            ''')

            # Table des patterns appris
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS action_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    context TEXT,
                    action_sequence TEXT,
                    success_rate REAL,
                    usage_frequency INTEGER,
                    avg_execution_time REAL,
                    conditions TEXT,
                    confidence_level REAL,
                    created_date TEXT,
                    last_used TEXT,
                    pattern_data TEXT
                )
            ''')

            # Table des métriques de performance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    date TEXT PRIMARY KEY,
                    total_sessions INTEGER,
                    avg_session_duration REAL,
                    improvement_rate REAL,
                    pattern_accuracy REAL,
                    adaptation_speed REAL,
                    knowledge_coverage REAL,
                    metrics_data TEXT
                )
            ''')

            conn.commit()

    def _load_existing_patterns(self):
        """Charge les patterns existants depuis la base de données"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM action_patterns')

                for row in cursor.fetchall():
                    pattern_id = row[0]
                    context = json.loads(row[1])
                    action_sequence = json.loads(row[2])
                    success_rate = row[3]
                    usage_frequency = row[4]
                    avg_execution_time = row[5]
                    conditions = json.loads(row[6])
                    confidence_level = row[7]

                    pattern = ActionPattern(
                        pattern_id=pattern_id,
                        context=context,
                        action_sequence=action_sequence,
                        success_rate=success_rate,
                        usage_frequency=usage_frequency,
                        avg_execution_time=avg_execution_time,
                        conditions=conditions,
                        confidence_level=confidence_level
                    )

                    self.patterns_cache[pattern_id] = pattern

                self.logger.info(f"Chargé {len(self.patterns_cache)} patterns existants")

        except Exception as e:
            self.logger.warning(f"Erreur chargement patterns: {e}")

    def start_learning_session(self, player_class: str, player_level: int, server: str) -> str:
        """Démarre une nouvelle session d'apprentissage"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.active_session = LearningSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            player_class=player_class,
            player_level=player_level,
            server=server,
            total_actions=0,
            successful_actions=0,
            combat_sessions=0,
            combat_wins=0,
            economic_gains=0.0,
            experience_gained=0,
            efficiency_score=0.0,
            learned_patterns=[]
        )

        # Reset des métriques de session
        for key in self.session_metrics:
            if hasattr(self.session_metrics[key], 'clear'):
                self.session_metrics[key].clear()
            else:
                self.session_metrics[key] = 0

        self.logger.info(f"Session d'apprentissage démarrée: {session_id}")
        return session_id

    def record_action_outcome(self, action: Dict, outcome: Dict, context: Dict):
        """Enregistre le résultat d'une action pour apprentissage"""
        if not self.active_session:
            return

        # Mise à jour des métriques de session
        self.active_session.total_actions += 1
        self.session_metrics["actions_taken"] += 1

        # Évaluation de la qualité de la décision
        quality_score = self._evaluate_action_quality(action, outcome, context)
        self.session_metrics["decisions_quality"].append(quality_score)

        # Succès de l'action
        is_successful = outcome.get("success", False)
        if is_successful:
            self.active_session.successful_actions += 1

        self.session_metrics["success_feedback"].append(is_successful)

        # Temps de réponse
        response_time = outcome.get("execution_time", 0)
        if response_time > 0:
            self.session_metrics["response_times"].append(response_time)

        # Apprentissage de patterns
        self._learn_from_action(action, outcome, context, is_successful)

        # Mise à jour gains économiques et expérience
        self.active_session.economic_gains += outcome.get("economic_gain", 0)
        self.active_session.experience_gained += outcome.get("experience_gain", 0)

    def _evaluate_action_quality(self, action: Dict, outcome: Dict, context: Dict) -> float:
        """Évalue la qualité d'une action sur une échelle 0-1"""
        score = 0.5  # Score de base

        # Facteurs positifs
        if outcome.get("success", False):
            score += 0.3

        if outcome.get("optimal", False):
            score += 0.2

        # Efficacité temporelle
        expected_time = action.get("expected_duration", 1.0)
        actual_time = outcome.get("execution_time", expected_time)
        if actual_time <= expected_time:
            score += 0.1

        # Adaptation au contexte
        if self._action_fits_context(action, context):
            score += 0.1

        # Facteurs négatifs
        if outcome.get("error", False):
            score -= 0.3

        if outcome.get("suboptimal", False):
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _action_fits_context(self, action: Dict, context: Dict) -> bool:
        """Vérifie si l'action est appropriée au contexte"""
        action_type = action.get("type", "")

        # Vérifications contextuelles de base
        if context.get("in_combat", False):
            return action_type in ["spell_cast", "movement", "item_use"]
        elif context.get("in_exploration", False):
            return action_type in ["movement", "resource_gather", "interaction"]
        elif context.get("in_market", False):
            return action_type in ["buy", "sell", "market_analysis"]

        return True  # Par défaut, accepte l'action

    def _learn_from_action(self, action: Dict, outcome: Dict, context: Dict, success: bool):
        """Apprend un nouveau pattern ou renforce un pattern existant"""

        # Création d'un identifiant de pattern basé sur le contexte
        pattern_key = self._generate_pattern_key(action, context)

        if pattern_key in self.patterns_cache:
            # Renforcement d'un pattern existant
            pattern = self.patterns_cache[pattern_key]
            self._update_existing_pattern(pattern, action, outcome, success)
        else:
            # Création d'un nouveau pattern
            self._create_new_pattern(pattern_key, action, outcome, context, success)

    def _generate_pattern_key(self, action: Dict, context: Dict) -> str:
        """Génère une clé unique pour identifier un pattern"""
        context_key_parts = []

        # Contexte de combat
        if context.get("in_combat", False):
            context_key_parts.extend([
                f"combat",
                f"hp_{context.get('player_hp', 100)//20*20}",  # Paliers de 20% HP
                f"ap_{context.get('available_ap', 6)}",
                f"enemies_{len(context.get('enemies', []))}"
            ])

        # Contexte d'exploration
        elif context.get("in_exploration", False):
            context_key_parts.extend([
                f"exploration",
                f"map_{context.get('map_id', 'unknown')}",
                f"time_{datetime.now().hour//6}"  # Quart de journée
            ])

        # Action principale
        action_type = action.get("type", "unknown")
        action_target = action.get("target", "none")

        context_key_parts.extend([action_type, str(action_target)])

        return "_".join(context_key_parts)

    def _create_new_pattern(self, pattern_key: str, action: Dict, outcome: Dict,
                          context: Dict, success: bool):
        """Crée un nouveau pattern d'action"""

        pattern = ActionPattern(
            pattern_id=pattern_key,
            context=context.copy(),
            action_sequence=[action],
            success_rate=1.0 if success else 0.0,
            usage_frequency=1,
            avg_execution_time=outcome.get("execution_time", 1.0),
            conditions=self._extract_conditions(context),
            confidence_level=0.5  # Confiance initiale faible
        )

        self.patterns_cache[pattern_key] = pattern
        self._save_pattern_to_db(pattern)

        if self.active_session:
            self.active_session.learned_patterns.append(pattern_key)

    def _update_existing_pattern(self, pattern: ActionPattern, action: Dict,
                               outcome: Dict, success: bool):
        """Met à jour un pattern existant avec une nouvelle expérience"""

        # Mise à jour du taux de succès avec moyenne mobile
        alpha = self.learning_config["adaptation_rate"]
        new_success = 1.0 if success else 0.0
        pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * new_success

        # Mise à jour fréquence d'usage
        pattern.usage_frequency += 1

        # Mise à jour temps d'exécution moyen
        execution_time = outcome.get("execution_time", pattern.avg_execution_time)
        pattern.avg_execution_time = (
            (pattern.avg_execution_time * (pattern.usage_frequency - 1) + execution_time)
            / pattern.usage_frequency
        )

        # Mise à jour confiance
        self._update_pattern_confidence(pattern)

        # Sauvegarde en base
        self._save_pattern_to_db(pattern)

    def _extract_conditions(self, context: Dict) -> Dict[str, Any]:
        """Extrait les conditions d'application d'un pattern"""
        conditions = {}

        # Conditions de combat
        if context.get("in_combat", False):
            conditions.update({
                "min_hp": max(0, context.get("player_hp", 100) - 20),
                "max_hp": min(100, context.get("player_hp", 100) + 20),
                "min_ap": context.get("available_ap", 6),
                "enemy_count_range": [
                    max(1, len(context.get("enemies", [])) - 1),
                    len(context.get("enemies", [])) + 1
                ]
            })

        # Conditions économiques
        if context.get("current_kamas"):
            kamas = context["current_kamas"]
            conditions["min_kamas"] = max(0, kamas - kamas * 0.1)

        # Conditions temporelles
        current_hour = datetime.now().hour
        conditions["time_range"] = [max(0, current_hour - 2), min(23, current_hour + 2)]

        return conditions

    def _update_pattern_confidence(self, pattern: ActionPattern):
        """Met à jour le niveau de confiance d'un pattern"""

        # Facteurs de confiance
        frequency_factor = min(1.0, pattern.usage_frequency / 10.0)  # Max confiance à 10 usages
        success_factor = pattern.success_rate
        recency_factor = 1.0  # Pourrait être basé sur la dernière utilisation

        # Calcul de confiance pondéré
        pattern.confidence_level = (
            0.4 * frequency_factor +
            0.5 * success_factor +
            0.1 * recency_factor
        )

    def _save_pattern_to_db(self, pattern: ActionPattern):
        """Sauvegarde un pattern en base de données"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO action_patterns
                    (pattern_id, context, action_sequence, success_rate, usage_frequency,
                     avg_execution_time, conditions, confidence_level, created_date,
                     last_used, pattern_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern_id,
                    json.dumps(pattern.context),
                    json.dumps(pattern.action_sequence),
                    pattern.success_rate,
                    pattern.usage_frequency,
                    pattern.avg_execution_time,
                    json.dumps(pattern.conditions),
                    pattern.confidence_level,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    json.dumps(asdict(pattern))
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"Erreur sauvegarde pattern {pattern.pattern_id}: {e}")

    def get_recommended_action(self, current_context: Dict) -> Optional[Dict]:
        """Recommande une action basée sur les patterns appris"""

        best_pattern = None
        best_score = 0.0

        for pattern in self.patterns_cache.values():
            if self._pattern_matches_context(pattern, current_context):
                score = self._calculate_pattern_score(pattern, current_context)

                if score > best_score and score > self.learning_config["confidence_threshold"]:
                    best_score = score
                    best_pattern = pattern

        if best_pattern:
            # Exploitation du meilleur pattern
            recommended_action = best_pattern.action_sequence[0].copy()
            recommended_action["confidence"] = best_score
            recommended_action["pattern_source"] = best_pattern.pattern_id
            return recommended_action

        # Exploration si aucun pattern confiant trouvé
        if np.random.random() < self.learning_config["exploration_rate"]:
            return self._generate_exploratory_action(current_context)

        return None

    def _pattern_matches_context(self, pattern: ActionPattern, context: Dict) -> bool:
        """Vérifie si un pattern est applicable au contexte actuel"""

        conditions = pattern.conditions

        # Vérification HP
        if "min_hp" in conditions and "max_hp" in conditions:
            current_hp = context.get("player_hp", 100)
            if not (conditions["min_hp"] <= current_hp <= conditions["max_hp"]):
                return False

        # Vérification AP
        if "min_ap" in conditions:
            current_ap = context.get("available_ap", 6)
            if current_ap < conditions["min_ap"]:
                return False

        # Vérification nombre d'ennemis
        if "enemy_count_range" in conditions:
            enemy_count = len(context.get("enemies", []))
            min_enemies, max_enemies = conditions["enemy_count_range"]
            if not (min_enemies <= enemy_count <= max_enemies):
                return False

        # Vérification kamas
        if "min_kamas" in conditions:
            current_kamas = context.get("current_kamas", 0)
            if current_kamas < conditions["min_kamas"]:
                return False

        # Vérification temps
        if "time_range" in conditions:
            current_hour = datetime.now().hour
            min_hour, max_hour = conditions["time_range"]
            if not (min_hour <= current_hour <= max_hour):
                return False

        return True

    def _calculate_pattern_score(self, pattern: ActionPattern, context: Dict) -> float:
        """Calcule le score d'un pattern pour le contexte actuel"""

        base_score = pattern.confidence_level

        # Bonus pour taux de succès élevé
        success_bonus = pattern.success_rate * 0.2

        # Bonus pour fréquence d'utilisation (patterns éprouvés)
        frequency_bonus = min(0.1, pattern.usage_frequency / 50.0)

        # Malus pour temps d'exécution long
        time_penalty = max(0, (pattern.avg_execution_time - 1.0) * 0.05)

        final_score = base_score + success_bonus + frequency_bonus - time_penalty

        return max(0.0, min(1.0, final_score))

    def _generate_exploratory_action(self, context: Dict) -> Dict:
        """Génère une action exploratoire pour découvrir de nouveaux patterns"""

        if context.get("in_combat", False):
            # Actions de combat exploratoires
            exploratory_actions = [
                {"type": "spell_cast", "target": "nearest_enemy", "spell": "random"},
                {"type": "movement", "direction": "strategic"},
                {"type": "item_use", "item": "health_potion"}
            ]
        elif context.get("in_exploration", False):
            # Actions d'exploration
            exploratory_actions = [
                {"type": "movement", "direction": "unexplored"},
                {"type": "resource_gather", "target": "nearest_resource"},
                {"type": "interaction", "target": "npc"}
            ]
        else:
            # Actions génériques
            exploratory_actions = [
                {"type": "wait", "duration": 1.0},
                {"type": "interface_check", "target": "inventory"}
            ]

        action = np.random.choice(exploratory_actions)
        action["exploration"] = True
        action["confidence"] = 0.3  # Confiance faible pour exploration

        return action

    def end_learning_session(self) -> Optional[LearningSession]:
        """Termine la session d'apprentissage active"""
        if not self.active_session:
            return None

        # Calcul du score d'efficacité final
        if self.active_session.total_actions > 0:
            success_rate = self.active_session.successful_actions / self.active_session.total_actions
            avg_quality = np.mean(self.session_metrics["decisions_quality"]) if self.session_metrics["decisions_quality"] else 0.5
            self.active_session.efficiency_score = (success_rate + avg_quality) / 2.0

        # Finalisation de la session
        self.active_session.end_time = datetime.now()

        # Sauvegarde en base
        self._save_session_to_db(self.active_session)

        completed_session = self.active_session
        self.active_session = None

        self.logger.info(f"Session terminée: {completed_session.session_id}, "
                        f"Score: {completed_session.efficiency_score:.3f}")

        return completed_session

    def _save_session_to_db(self, session: LearningSession):
        """Sauvegarde une session en base de données"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO learning_sessions
                    (session_id, start_time, end_time, player_class, player_level,
                     server, total_actions, successful_actions, combat_sessions,
                     combat_wins, economic_gains, experience_gained, efficiency_score,
                     learned_patterns, session_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id,
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.player_class,
                    session.player_level,
                    session.server,
                    session.total_actions,
                    session.successful_actions,
                    session.combat_sessions,
                    session.combat_wins,
                    session.economic_gains,
                    session.experience_gained,
                    session.efficiency_score,
                    json.dumps(session.learned_patterns),
                    json.dumps(asdict(session), default=str)
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"Erreur sauvegarde session {session.session_id}: {e}")

    def get_learning_metrics(self) -> LearningMetrics:
        """Retourne les métriques d'apprentissage actuelles"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Nombre total de sessions
                cursor.execute('SELECT COUNT(*) FROM learning_sessions')
                total_sessions = cursor.fetchone()[0]

                # Durée moyenne des sessions
                cursor.execute('''
                    SELECT AVG(
                        CASE
                            WHEN end_time IS NOT NULL
                            THEN (julianday(end_time) - julianday(start_time)) * 24 * 60
                            ELSE 0
                        END
                    ) FROM learning_sessions WHERE end_time IS NOT NULL
                ''')
                avg_duration = cursor.fetchone()[0] or 0

                # Taux d'amélioration
                cursor.execute('''
                    SELECT efficiency_score FROM learning_sessions
                    WHERE end_time IS NOT NULL
                    ORDER BY start_time DESC LIMIT 10
                ''')
                recent_scores = [row[0] for row in cursor.fetchall()]

                improvement_rate = 0.0
                if len(recent_scores) >= 2:
                    improvement_rate = (recent_scores[0] - recent_scores[-1]) / len(recent_scores)

                # Précision des patterns
                pattern_accuracy = np.mean([p.success_rate for p in self.patterns_cache.values()]) if self.patterns_cache else 0.0

                # Vitesse d'adaptation (patterns appris par session)
                cursor.execute('SELECT AVG(json_array_length(learned_patterns)) FROM learning_sessions WHERE learned_patterns != "[]"')
                adaptation_speed = cursor.fetchone()[0] or 0

                # Couverture des connaissances (diversité des patterns)
                knowledge_coverage = min(1.0, len(self.patterns_cache) / 100.0)  # Normalisation sur 100 patterns

                return LearningMetrics(
                    total_sessions=total_sessions,
                    avg_session_duration=avg_duration,
                    improvement_rate=improvement_rate,
                    pattern_accuracy=pattern_accuracy,
                    adaptation_speed=adaptation_speed,
                    knowledge_coverage=knowledge_coverage
                )

        except Exception as e:
            self.logger.error(f"Erreur calcul métriques: {e}")
            return LearningMetrics(0, 0, 0, 0, 0, 0)

    def export_learned_knowledge(self, filepath: str):
        """Exporte les connaissances apprises pour sauvegarde/partage"""

        export_data = {
            "export_date": datetime.now().isoformat(),
            "total_patterns": len(self.patterns_cache),
            "learning_config": self.learning_config,
            "patterns": {},
            "metrics": asdict(self.get_learning_metrics())
        }

        # Export des patterns avec filtrage (confiance minimale)
        for pattern_id, pattern in self.patterns_cache.items():
            if pattern.confidence_level >= 0.5:  # Seulement patterns fiables
                export_data["patterns"][pattern_id] = asdict(pattern)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Connaissances exportées vers {filepath}")

# Interface simplifié pour intégration
def get_learning_engine() -> AdaptiveLearningEngine:
    """Retourne l'instance globale du moteur d'apprentissage"""
    if not hasattr(get_learning_engine, '_instance'):
        get_learning_engine._instance = AdaptiveLearningEngine()
    return get_learning_engine._instance

if __name__ == "__main__":
    # Test du Learning Engine
    print("[TEST] Learning Engine Adaptatif")

    engine = AdaptiveLearningEngine()

    # Test session d'apprentissage
    session_id = engine.start_learning_session("IOPS", 150, "Julith")
    print(f"Session démarrée: {session_id}")

    # Simulation d'actions et résultats
    for i in range(10):
        action = {"type": "spell_cast", "target": "enemy", "spell": "Pression"}
        outcome = {"success": True, "execution_time": 0.5, "damage": 100}
        context = {"in_combat": True, "player_hp": 80, "available_ap": 6, "enemies": ["Bouftou"]}

        engine.record_action_outcome(action, outcome, context)

    # Test de recommandation
    recommendation = engine.get_recommended_action(context)
    print(f"Recommandation: {recommendation}")

    # Fin de session
    completed_session = engine.end_learning_session()
    print(f"Session terminée avec score: {completed_session.efficiency_score:.3f}")

    # Métriques
    metrics = engine.get_learning_metrics()
    print(f"Métriques: {metrics}")

    print("[OK] Tests Learning Engine réussis")