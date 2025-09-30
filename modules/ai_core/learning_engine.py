"""
Learning Engine - Système d'apprentissage par observation DOFUS Unity
Intelligence adaptive basée sur l'observation des actions du joueur

Fonctionnalités:
- Observation comportementale en temps réel
- Apprentissage automatique des patterns d'action
- Suggestion intelligente basée sur l'historique
- Détection des situations de jeu (combat, exploration, quête)
- Optimisation des séquences d'actions
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@dataclass
class GameAction:
    """Action de jeu observée"""
    timestamp: float
    action_type: str  # click, spell, movement, menu, etc.
    coordinates: Tuple[int, int]
    context: str  # combat, exploration, quest, crafting, etc.
    success: bool
    target_info: Dict[str, Any]
    screen_hash: str  # Hash de l'état d'écran pour contexte

@dataclass
class GameSituation:
    """Situation de jeu identifiée"""
    situation_id: str
    situation_type: str  # combat, quest, exploration, crafting
    confidence: float
    key_elements: List[str]
    recommended_actions: List[str]
    timestamps: List[float]

@dataclass
class LearningPattern:
    """Pattern d'apprentissage détecté"""
    pattern_id: str
    actions_sequence: List[str]
    success_rate: float
    frequency: int
    contexts: List[str]
    avg_execution_time: float
    confidence_score: float

class BehaviorAnalyzer:
    """Analyseur de comportement du joueur"""

    def __init__(self):
        self.action_history: deque = deque(maxlen=10000)
        self.situation_history: deque = deque(maxlen=5000)
        self.patterns: Dict[str, LearningPattern] = {}

        # ML Models
        self.action_scaler = StandardScaler()
        self.situation_classifier = None
        self.action_predictor = None

        self.logger = logging.getLogger(__name__)

    def analyze_action_sequence(self, actions: List[GameAction]) -> List[LearningPattern]:
        """Analyse une séquence d'actions pour détecter des patterns"""
        if len(actions) < 3:
            return []

        patterns = []

        # Analyser par fenêtres glissantes
        for window_size in [3, 4, 5]:
            for i in range(len(actions) - window_size + 1):
                window = actions[i:i + window_size]

                # Créer signature du pattern
                action_types = [a.action_type for a in window]
                context = window[0].context

                pattern_signature = f"{context}_{'-'.join(action_types)}"

                # Calculer métriques
                execution_time = window[-1].timestamp - window[0].timestamp
                success_rate = sum(1 for a in window if a.success) / len(window)

                # Créer pattern
                pattern = LearningPattern(
                    pattern_id=pattern_signature,
                    actions_sequence=action_types,
                    success_rate=success_rate,
                    frequency=1,
                    contexts=[context],
                    avg_execution_time=execution_time,
                    confidence_score=success_rate * 0.8 + (1.0 / execution_time) * 0.2
                )

                patterns.append(pattern)

        return patterns

    def detect_game_situation(self, screenshot: np.ndarray, ocr_results: List) -> GameSituation:
        """Détecte la situation de jeu actuelle"""
        # Analyser éléments visuels
        situation_features = self._extract_situation_features(screenshot, ocr_results)

        # Classification des situations
        if self._is_combat_situation(situation_features):
            return GameSituation(
                situation_id=f"combat_{int(time.time())}",
                situation_type="combat",
                confidence=0.85,
                key_elements=["hp_bar", "action_points", "enemies"],
                recommended_actions=["spell_cast", "movement", "potion_use"],
                timestamps=[time.time()]
            )
        elif self._is_quest_situation(situation_features):
            return GameSituation(
                situation_id=f"quest_{int(time.time())}",
                situation_type="quest",
                confidence=0.75,
                key_elements=["quest_dialog", "npc", "objectives"],
                recommended_actions=["dialog_advance", "movement", "item_use"],
                timestamps=[time.time()]
            )
        elif self._is_exploration_situation(situation_features):
            return GameSituation(
                situation_id=f"exploration_{int(time.time())}",
                situation_type="exploration",
                confidence=0.70,
                key_elements=["minimap", "resources", "movement_path"],
                recommended_actions=["movement", "resource_gather", "map_open"],
                timestamps=[time.time()]
            )
        else:
            return GameSituation(
                situation_id=f"unknown_{int(time.time())}",
                situation_type="unknown",
                confidence=0.30,
                key_elements=[],
                recommended_actions=[],
                timestamps=[time.time()]
            )

    def _extract_situation_features(self, screenshot: np.ndarray, ocr_results: List) -> Dict[str, Any]:
        """Extrait les caractéristiques de la situation"""
        features = {
            "text_elements": [],
            "ui_elements": [],
            "colors": [],
            "regions": {}
        }

        # Analyser texte OCR
        for ocr_result in ocr_results:
            text = ocr_result.text.lower()
            features["text_elements"].append(text)

        # Analyser couleurs dominantes
        resized = cv2.resize(screenshot, (100, 100))
        colors = resized.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(colors)
        features["colors"] = kmeans.cluster_centers_.tolist()

        # Détection d'éléments UI spécifiques
        features["ui_elements"] = self._detect_ui_elements(screenshot)

        return features

    def _detect_ui_elements(self, screenshot: np.ndarray) -> List[str]:
        """Détecte les éléments d'interface utilisateur"""
        ui_elements = []

        # Convertir en niveaux de gris pour détection
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Détection de barres de vie (rectangles rouges/verts)
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        # Masque rouge pour barres de vie
        red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        if cv2.countNonZero(red_mask) > 1000:
            ui_elements.append("health_bar")

        # Masque vert pour barres de vie/mana
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        if cv2.countNonZero(green_mask) > 1000:
            ui_elements.append("resource_bar")

        # Masque bleu pour barres de mana
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        if cv2.countNonZero(blue_mask) > 1000:
            ui_elements.append("mana_bar")

        # Détection de contours pour boutons/interface
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_rectangles = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Interface elements
                large_rectangles += 1

        if large_rectangles > 10:
            ui_elements.append("complex_interface")
        elif large_rectangles > 5:
            ui_elements.append("simple_interface")

        return ui_elements

    def _is_combat_situation(self, features: Dict[str, Any]) -> bool:
        """Détecte si c'est une situation de combat"""
        combat_indicators = [
            "health_bar" in features["ui_elements"],
            "mana_bar" in features["ui_elements"],
            any("pa" in text or "pm" in text for text in features["text_elements"]),
            any("dommage" in text or "damage" in text for text in features["text_elements"]),
            "complex_interface" in features["ui_elements"]
        ]

        return sum(combat_indicators) >= 2

    def _is_quest_situation(self, features: Dict[str, Any]) -> bool:
        """Détecte si c'est une situation de quête"""
        quest_indicators = [
            any("objectif" in text or "objective" in text for text in features["text_elements"]),
            any("parler" in text or "talk" in text for text in features["text_elements"]),
            any("aller" in text or "go" in text for text in features["text_elements"]),
            any("quete" in text or "quest" in text for text in features["text_elements"])
        ]

        return sum(quest_indicators) >= 1

    def _is_exploration_situation(self, features: Dict[str, Any]) -> bool:
        """Détecte si c'est une situation d'exploration"""
        exploration_indicators = [
            "simple_interface" in features["ui_elements"],
            len(features["text_elements"]) < 5,  # Peu de texte visible
            not self._is_combat_situation(features),
            not self._is_quest_situation(features)
        ]

        return sum(exploration_indicators) >= 2

class LearningEngine:
    """Moteur d'apprentissage principal"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Composants d'analyse
        self.behavior_analyzer = BehaviorAnalyzer()

        # État d'apprentissage
        self.current_situation: Optional[GameSituation] = None
        self.learning_active = False
        self.observation_thread: Optional[threading.Thread] = None

        # Cache et optimisation
        self.action_buffer: deque = deque(maxlen=100)
        self.pattern_cache: Dict[str, Any] = {}
        self.last_analysis_time = 0.0

        # Métriques performance
        self.learning_stats = {
            "actions_observed": 0,
            "patterns_learned": 0,
            "situations_detected": 0,
            "accuracy_score": 0.0,
            "uptime": 0.0
        }

        self.logger.info("LearningEngine initialisé")

    def initialize(self) -> bool:
        """Initialise le système d'apprentissage"""
        try:
            # Charger données existantes
            self._load_learning_data()

            # Initialiser modèles ML
            self._initialize_ml_models()

            self.logger.info("LearningEngine initialisé avec succès")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation LearningEngine: {e}")
            return False

    def start_learning(self) -> bool:
        """Démarre l'apprentissage en continu"""
        if self.learning_active:
            self.logger.warning("Apprentissage déjà actif")
            return False

        self.learning_active = True

        self.observation_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True
        )
        self.observation_thread.start()

        self.logger.info("Apprentissage démarré")
        return True

    def stop_learning(self):
        """Arrête l'apprentissage"""
        self.learning_active = False
        if self.observation_thread and self.observation_thread.is_alive():
            self.observation_thread.join(timeout=5.0)

        # Sauvegarder données d'apprentissage
        self._save_learning_data()

        self.logger.info("Apprentissage arrêté")

    def observe_action(self, action: GameAction):
        """Observe une action du joueur"""
        self.action_buffer.append(action)
        self.learning_stats["actions_observed"] += 1

        # Analyser si buffer suffisant
        if len(self.action_buffer) >= 5:
            self._analyze_recent_actions()

    def observe_situation(self, screenshot: np.ndarray, ocr_results: List) -> GameSituation:
        """Observe et analyse la situation actuelle"""
        situation = self.behavior_analyzer.detect_game_situation(screenshot, ocr_results)

        self.current_situation = situation
        self.learning_stats["situations_detected"] += 1

        return situation

    def get_action_recommendations(self, context: str) -> List[Dict[str, Any]]:
        """Retourne des recommandations d'actions basées sur l'apprentissage"""
        if not self.current_situation:
            return []

        # Rechercher patterns similaires
        relevant_patterns = self._find_relevant_patterns(context)

        recommendations = []
        for pattern in relevant_patterns[:5]:  # Top 5
            recommendation = {
                "action_sequence": pattern.actions_sequence,
                "confidence": pattern.confidence_score,
                "success_rate": pattern.success_rate,
                "execution_time": pattern.avg_execution_time,
                "frequency": pattern.frequency
            }
            recommendations.append(recommendation)

        return sorted(recommendations, key=lambda x: x["confidence"], reverse=True)

    def _analyze_recent_actions(self):
        """Analyse les actions récentes pour détecter des patterns"""
        if len(self.action_buffer) < 3:
            return

        recent_actions = list(self.action_buffer)[-10:]  # 10 dernières actions
        patterns = self.behavior_analyzer.analyze_action_sequence(recent_actions)

        # Mettre à jour base de patterns
        for pattern in patterns:
            if pattern.pattern_id in self.behavior_analyzer.patterns:
                # Mettre à jour pattern existant
                existing = self.behavior_analyzer.patterns[pattern.pattern_id]
                existing.frequency += 1
                existing.success_rate = (existing.success_rate + pattern.success_rate) / 2
                existing.avg_execution_time = (existing.avg_execution_time + pattern.avg_execution_time) / 2
            else:
                # Nouveau pattern
                self.behavior_analyzer.patterns[pattern.pattern_id] = pattern
                self.learning_stats["patterns_learned"] += 1

    def _find_relevant_patterns(self, context: str) -> List[LearningPattern]:
        """Trouve les patterns pertinents pour le contexte donné"""
        relevant = []

        for pattern in self.behavior_analyzer.patterns.values():
            if context in pattern.contexts:
                relevant.append(pattern)

        return sorted(relevant, key=lambda x: x.confidence_score, reverse=True)

    def _learning_loop(self):
        """Boucle principale d'apprentissage"""
        start_time = time.time()

        while self.learning_active:
            current_time = time.time()

            # Analyser périodiquement
            if current_time - self.last_analysis_time > 30.0:  # Toutes les 30 secondes
                self._periodic_analysis()
                self.last_analysis_time = current_time

            # Mise à jour métriques
            self.learning_stats["uptime"] = current_time - start_time

            time.sleep(1.0)

    def _periodic_analysis(self):
        """Analyse périodique pour optimiser l'apprentissage"""
        # Nettoyer patterns peu utilisés
        patterns_to_remove = []
        for pattern_id, pattern in self.behavior_analyzer.patterns.items():
            if pattern.frequency < 2 and pattern.confidence_score < 0.3:
                patterns_to_remove.append(pattern_id)

        for pattern_id in patterns_to_remove:
            del self.behavior_analyzer.patterns[pattern_id]

        # Calculer score de précision
        total_patterns = len(self.behavior_analyzer.patterns)
        high_confidence = sum(1 for p in self.behavior_analyzer.patterns.values()
                             if p.confidence_score > 0.7)

        if total_patterns > 0:
            self.learning_stats["accuracy_score"] = high_confidence / total_patterns

        self.logger.debug(f"Analyse périodique: {total_patterns} patterns, "
                         f"précision {self.learning_stats['accuracy_score']:.2f}")

    def _initialize_ml_models(self):
        """Initialise les modèles d'apprentissage automatique"""
        # TODO: Initialiser sklearn models pour classification et prédiction
        pass

    def _load_learning_data(self):
        """Charge les données d'apprentissage sauvegardées"""
        patterns_file = self.data_dir / "learned_patterns.json"

        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Reconstituer patterns
                for pattern_data in data.get("patterns", []):
                    pattern = LearningPattern(**pattern_data)
                    self.behavior_analyzer.patterns[pattern.pattern_id] = pattern

                self.learning_stats.update(data.get("stats", {}))
                self.logger.info(f"Chargé {len(self.behavior_analyzer.patterns)} patterns")

            except Exception as e:
                self.logger.error(f"Erreur chargement données: {e}")

    def _save_learning_data(self):
        """Sauvegarde les données d'apprentissage"""
        patterns_file = self.data_dir / "learned_patterns.json"

        try:
            data = {
                "patterns": [asdict(pattern) for pattern in self.behavior_analyzer.patterns.values()],
                "stats": self.learning_stats,
                "timestamp": datetime.now().isoformat()
            }

            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Sauvegardé {len(self.behavior_analyzer.patterns)} patterns")

        except Exception as e:
            self.logger.error(f"Erreur sauvegarde données: {e}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'apprentissage"""
        return {
            **self.learning_stats,
            "patterns_count": len(self.behavior_analyzer.patterns),
            "current_situation": self.current_situation.situation_type if self.current_situation else "unknown",
            "buffer_size": len(self.action_buffer)
        }

    def cleanup(self):
        """Nettoyage des ressources"""
        self.stop_learning()
        self.action_buffer.clear()
        self.pattern_cache.clear()
        self.logger.info("LearningEngine nettoyé")

# Factory function
def create_learning_engine(data_dir: Path) -> LearningEngine:
    """Crée une instance LearningEngine configurée"""
    engine = LearningEngine(data_dir)
    if engine.initialize():
        return engine
    else:
        raise RuntimeError("Impossible d'initialiser LearningEngine")

# Test de base
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        from pathlib import Path
        data_dir = Path("data/learning")

        engine = create_learning_engine(data_dir)

        print("Test LearningEngine...")

        # Test observation d'action
        test_action = GameAction(
            timestamp=time.time(),
            action_type="spell_cast",
            coordinates=(500, 300),
            context="combat",
            success=True,
            target_info={"type": "enemy", "hp": 50},
            screen_hash="test_hash"
        )

        engine.observe_action(test_action)

        # Test recommandations
        recommendations = engine.get_action_recommendations("combat")
        print(f"Recommandations: {len(recommendations)}")

        # Stats
        stats = engine.get_learning_statistics()
        print(f"Stats: {stats}")

    except Exception as e:
        print(f"Erreur test: {e}")
    finally:
        if 'engine' in locals():
            engine.cleanup()