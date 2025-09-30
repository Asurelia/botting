"""
Mode Vision Passive - DOFUS Unity World Model AI
Capture et apprentissage passif des patterns de jeu
Connexion sécurisée à DOFUS Unity pour observation uniquement
"""

import time
import json
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import cv2

# Imports vision
from .screenshot_capture import DofusWindowCapture
from .unity_interface_reader import DofusUnityInterfaceReader, GameState
from .combat_grid_analyzer import DofusCombatGridAnalyzer

# Imports apprentissage
import sys
sys.path.append(str(Path(__file__).parent.parent))
from learning_engine.adaptive_learning_engine import get_learning_engine
from world_model.hrm_dofus_integration import DofusGameState, DofusClass

@dataclass
class PassiveLearningSession:
    """Session d'apprentissage passif"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_captures: int = 0
    combat_sequences: int = 0
    exploration_sequences: int = 0
    patterns_learned: int = 0
    errors_encountered: int = 0
    avg_fps: float = 0.0
    data_quality_score: float = 0.0

@dataclass
class PatternObservation:
    """Observation d'un pattern de gameplay"""
    timestamp: float
    pattern_type: str  # "spell_sequence", "movement", "targeting", etc.
    context: Dict[str, Any]
    game_state: DofusGameState
    screenshot_path: Optional[str] = None
    confidence_score: float = 0.0
    learning_value: float = 0.0

class PassiveDofusLearningEngine:
    """Moteur d'apprentissage passif pour DOFUS Unity"""

    def __init__(self, session_dir: str = "passive_learning_sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)

        # Composants vision
        self.window_capture = DofusWindowCapture()
        self.interface_reader = DofusUnityInterfaceReader()
        self.grid_analyzer = DofusCombatGridAnalyzer()

        # Learning engine
        self.learning_engine = get_learning_engine()

        # Session courante
        self.current_session: Optional[PassiveLearningSession] = None
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None

        # Configuration
        self.config = {
            "capture_interval": 0.5,  # Capture toutes les 500ms
            "max_session_duration": 3600,  # 1 heure max
            "min_confidence_threshold": 0.7,
            "save_screenshots": True,
            "max_screenshots_per_session": 1000,
            "pattern_detection_enabled": True,
            "anti_detection_delays": True
        }

        # Patterns observés
        self.observed_patterns: List[PatternObservation] = []
        self.pattern_callbacks: Dict[str, List[Callable]] = {}

        # Métriques temps réel
        self.real_time_metrics = {
            "captures_per_minute": 0,
            "patterns_detected": 0,
            "current_game_state": None,
            "last_activity_time": 0,
            "session_quality": 0.0
        }

    def start_passive_learning_session(self, player_class: str = "IOPS",
                                     player_level: int = 150,
                                     session_name: Optional[str] = None) -> str:
        """Démarre une session d'apprentissage passif"""

        if self.is_running:
            raise RuntimeError("Une session d'apprentissage est déjà en cours")

        # Créer session
        session_id = f"passive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if session_name:
            session_id = f"{session_id}_{session_name}"

        self.current_session = PassiveLearningSession(
            session_id=session_id,
            start_time=time.time()
        )

        # Créer répertoire de session
        self.session_path = self.session_dir / session_id
        self.session_path.mkdir(exist_ok=True)

        if self.config["save_screenshots"]:
            (self.session_path / "screenshots").mkdir(exist_ok=True)

        # Initialiser learning engine
        learning_session_id = self.learning_engine.start_learning_session(
            player_class, player_level, "PassiveLearning"
        )

        print(f"🔍 Session d'apprentissage passif démarrée : {session_id}")
        print(f"📂 Répertoire : {self.session_path}")

        # Démarrer capture
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._passive_capture_loop, daemon=True)
        self.capture_thread.start()

        return session_id

    def _passive_capture_loop(self):
        """Boucle principale de capture passive"""
        print("🎯 Démarrage de la capture passive...")

        capture_count = 0
        last_metrics_update = time.time()

        while self.is_running and self.current_session:
            try:
                start_capture = time.time()

                # Vérifier durée maximale
                if (start_capture - self.current_session.start_time) > self.config["max_session_duration"]:
                    print("⏰ Durée maximale de session atteinte")
                    break

                # Vérifier limite de captures
                if capture_count >= self.config["max_screenshots_per_session"]:
                    print("📸 Limite de captures atteinte")
                    break

                # Capturer écran DOFUS
                screenshot = self.window_capture.capture_dofus_window()
                if screenshot is None:
                    time.sleep(1.0)  # Attendre si pas de fenêtre DOFUS
                    continue

                # Analyser interface
                game_state = self._analyze_game_state(screenshot)
                if game_state is None:
                    time.sleep(self.config["capture_interval"])
                    continue

                # Sauvegarder screenshot si configuré
                screenshot_path = None
                if self.config["save_screenshots"]:
                    screenshot_filename = f"capture_{capture_count:05d}.png"
                    screenshot_path = self.session_path / "screenshots" / screenshot_filename
                    cv2.imwrite(str(screenshot_path), screenshot)

                # Détecter patterns si activé
                if self.config["pattern_detection_enabled"]:
                    self._detect_and_record_patterns(game_state, screenshot, str(screenshot_path))

                # Enregistrer dans learning engine
                self._record_passive_observation(game_state)

                # Mettre à jour métriques
                capture_count += 1
                self.current_session.total_captures = capture_count

                if game_state.in_combat:
                    self.current_session.combat_sequences += 1
                else:
                    self.current_session.exploration_sequences += 1

                # Mettre à jour métriques temps réel
                if time.time() - last_metrics_update >= 60:  # Chaque minute
                    self._update_real_time_metrics()
                    last_metrics_update = time.time()

                # Anti-détection : délais variables
                if self.config["anti_detection_delays"]:
                    delay = self.config["capture_interval"] + np.random.normal(0, 0.1)
                    delay = max(0.1, delay)  # Minimum 100ms
                    time.sleep(delay)
                else:
                    time.sleep(self.config["capture_interval"])

            except Exception as e:
                print(f"❌ Erreur capture passive : {e}")
                self.current_session.errors_encountered += 1
                time.sleep(2.0)  # Pause en cas d'erreur

        print("⏹️ Fin de la boucle de capture passive")

    def _analyze_game_state(self, screenshot: np.ndarray) -> Optional[DofusGameState]:
        """Analyse l'état du jeu à partir d'un screenshot"""
        try:
            # Analyser interface avec OCR
            interface_state = self.interface_reader.analyze_game_interface(screenshot)

            if not interface_state or interface_state.confidence < self.config["min_confidence_threshold"]:
                return None

            # Analyser grille de combat si en combat
            combat_analysis = None
            if interface_state.in_combat:
                combat_analysis = self.grid_analyzer.analyze_combat_situation(screenshot)

            # Construire DofusGameState
            game_state = DofusGameState(
                player_class=DofusClass.IOPS,  # TODO: Détecter depuis interface
                player_level=interface_state.player_level or 150,
                current_server="Auto-detected",
                current_map_id=interface_state.map_id or 0,
                in_combat=interface_state.in_combat,
                available_ap=interface_state.available_ap or 0,
                available_mp=interface_state.available_mp or 0,
                current_health=interface_state.current_hp or 100,
                max_health=interface_state.max_hp or 100,
                player_position=combat_analysis.player_position if combat_analysis else (0, 0),
                enemies_positions=combat_analysis.enemies_positions if combat_analysis else [],
                allies_positions=combat_analysis.allies_positions if combat_analysis else [],
                interface_elements_visible=interface_state.visible_elements or [],
                spell_cooldowns=interface_state.spell_cooldowns or {},
                inventory_items={},
                current_kamas=interface_state.kamas or 0,
                market_opportunities=[],
                timestamp=time.time(),
                screenshot_path=None
            )

            return game_state

        except Exception as e:
            print(f"❌ Erreur analyse état : {e}")
            return None

    def _detect_and_record_patterns(self, game_state: DofusGameState,
                                   screenshot: np.ndarray, screenshot_path: str):
        """Détecte et enregistre les patterns de gameplay"""
        patterns_detected = []

        try:
            # Pattern 1: Séquence de sorts en combat
            if game_state.in_combat and game_state.available_ap > 0:
                spell_pattern = self._detect_spell_sequence_pattern(game_state, screenshot)
                if spell_pattern:
                    patterns_detected.append(spell_pattern)

            # Pattern 2: Mouvement tactique
            if game_state.in_combat:
                movement_pattern = self._detect_movement_pattern(game_state, screenshot)
                if movement_pattern:
                    patterns_detected.append(movement_pattern)

            # Pattern 3: Gestion des ressources
            resource_pattern = self._detect_resource_management_pattern(game_state)
            if resource_pattern:
                patterns_detected.append(resource_pattern)

            # Enregistrer patterns détectés
            for pattern in patterns_detected:
                observation = PatternObservation(
                    timestamp=time.time(),
                    pattern_type=pattern["type"],
                    context=pattern["context"],
                    game_state=game_state,
                    screenshot_path=screenshot_path,
                    confidence_score=pattern["confidence"],
                    learning_value=pattern.get("learning_value", 0.5)
                )

                self.observed_patterns.append(observation)
                self.current_session.patterns_learned += 1

                # Callbacks
                if pattern["type"] in self.pattern_callbacks:
                    for callback in self.pattern_callbacks[pattern["type"]]:
                        callback(observation)

        except Exception as e:
            print(f"❌ Erreur détection patterns : {e}")

    def _detect_spell_sequence_pattern(self, game_state: DofusGameState,
                                     screenshot: np.ndarray) -> Optional[Dict[str, Any]]:
        """Détecte les patterns de séquences de sorts"""
        # Logique de détection des séquences de sorts
        # Basé sur l'analyse des boutons de sorts disponibles et leur utilisation

        if not game_state.spell_cooldowns:
            return None

        available_spells = [spell for spell, cd in game_state.spell_cooldowns.items() if cd == 0]

        if len(available_spells) >= 2:
            return {
                "type": "spell_sequence",
                "context": {
                    "available_spells": available_spells,
                    "current_ap": game_state.available_ap,
                    "enemies_count": len(game_state.enemies_positions),
                    "optimal_sequence": available_spells[:2]  # Simplification
                },
                "confidence": 0.8,
                "learning_value": 0.9
            }

        return None

    def _detect_movement_pattern(self, game_state: DofusGameState,
                               screenshot: np.ndarray) -> Optional[Dict[str, Any]]:
        """Détecte les patterns de mouvement tactique"""
        if not game_state.enemies_positions or not game_state.player_position:
            return None

        # Calculer distance aux ennemis
        player_x, player_y = game_state.player_position
        distances = []

        for enemy_x, enemy_y in game_state.enemies_positions:
            distance = abs(enemy_x - player_x) + abs(enemy_y - player_y)  # Distance Manhattan
            distances.append(distance)

        if distances:
            min_distance = min(distances)
            optimal_distance = 2  # Distance optimale pour Iop

            if min_distance != optimal_distance:
                return {
                    "type": "movement",
                    "context": {
                        "current_distance": min_distance,
                        "optimal_distance": optimal_distance,
                        "available_mp": game_state.available_mp,
                        "movement_needed": optimal_distance - min_distance
                    },
                    "confidence": 0.75,
                    "learning_value": 0.7
                }

        return None

    def _detect_resource_management_pattern(self, game_state: DofusGameState) -> Optional[Dict[str, Any]]:
        """Détecte les patterns de gestion des ressources"""
        hp_percent = game_state.current_health / game_state.max_health

        # Pattern : Santé basse nécessitant soins ou potion
        if hp_percent < 0.3 and game_state.in_combat:
            return {
                "type": "resource_management",
                "context": {
                    "health_percent": hp_percent,
                    "situation": "low_health_combat",
                    "action_needed": "healing_or_potion"
                },
                "confidence": 0.9,
                "learning_value": 0.8
            }

        return None

    def _record_passive_observation(self, game_state: DofusGameState):
        """Enregistre une observation passive dans le learning engine"""
        try:
            # Créer contexte d'observation
            context = {
                "observation_type": "passive",
                "in_combat": game_state.in_combat,
                "player_hp_percent": game_state.current_health / game_state.max_health,
                "available_ap": game_state.available_ap,
                "available_mp": game_state.available_mp,
                "enemies_count": len(game_state.enemies_positions),
                "timestamp": game_state.timestamp
            }

            # Action observée (simulée pour l'apprentissage passif)
            observed_action = {
                "type": "observation",
                "context": "passive_learning",
                "game_state_quality": "high" if game_state.available_ap > 0 else "medium"
            }

            # Outcome basé sur l'analyse de la situation
            outcome = {
                "success": True,  # Observation réussie
                "learning_value": 0.6,  # Valeur modérée pour apprentissage passif
                "observation_quality": "high"
            }

            # Enregistrer dans learning engine
            self.learning_engine.record_action_outcome(observed_action, outcome, context)

        except Exception as e:
            print(f"❌ Erreur enregistrement observation : {e}")

    def _update_real_time_metrics(self):
        """Met à jour les métriques temps réel"""
        if not self.current_session:
            return

        current_time = time.time()
        session_duration = current_time - self.current_session.start_time

        # Captures par minute
        if session_duration > 0:
            self.real_time_metrics["captures_per_minute"] = (
                self.current_session.total_captures / session_duration * 60
            )

        # Patterns détectés
        self.real_time_metrics["patterns_detected"] = len(self.observed_patterns)

        # Qualité de session
        total_sequences = (self.current_session.combat_sequences +
                          self.current_session.exploration_sequences)

        if total_sequences > 0:
            pattern_ratio = self.current_session.patterns_learned / total_sequences
            error_ratio = self.current_session.errors_encountered / self.current_session.total_captures

            self.real_time_metrics["session_quality"] = min(1.0, pattern_ratio - error_ratio)

        print(f"📊 Métriques temps réel:")
        print(f"  - Captures/min: {self.real_time_metrics['captures_per_minute']:.1f}")
        print(f"  - Patterns détectés: {self.real_time_metrics['patterns_detected']}")
        print(f"  - Qualité session: {self.real_time_metrics['session_quality']:.2f}")

    def stop_passive_learning_session(self) -> PassiveLearningSession:
        """Arrête la session d'apprentissage passif"""
        if not self.is_running or not self.current_session:
            raise RuntimeError("Aucune session active à arrêter")

        print("🛑 Arrêt de la session d'apprentissage passif...")

        # Arrêter capture
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)

        # Finaliser session
        self.current_session.end_time = time.time()

        session_duration = self.current_session.end_time - self.current_session.start_time
        if session_duration > 0:
            self.current_session.avg_fps = self.current_session.total_captures / session_duration

        # Calculer score de qualité
        if self.current_session.total_captures > 0:
            pattern_quality = self.current_session.patterns_learned / self.current_session.total_captures
            error_penalty = self.current_session.errors_encountered / self.current_session.total_captures
            self.current_session.data_quality_score = max(0, pattern_quality - error_penalty)

        # Sauvegarder session
        self._save_learning_session()

        # Finaliser learning engine
        try:
            self.learning_engine.end_learning_session()
        except:
            pass  # Learning engine peut ne pas avoir de session active

        completed_session = self.current_session
        self.current_session = None

        print("✅ Session d'apprentissage passif terminée")
        return completed_session

    def _save_learning_session(self):
        """Sauvegarde la session d'apprentissage"""
        if not self.current_session or not self.session_path:
            return

        try:
            # Sauvegarder métadonnées de session
            session_file = self.session_path / "session_metadata.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.current_session), f, indent=2, ensure_ascii=False)

            # Sauvegarder patterns observés
            if self.observed_patterns:
                patterns_file = self.session_path / "observed_patterns.json"
                patterns_data = []

                for pattern in self.observed_patterns:
                    # Sérialiser PatternObservation
                    pattern_dict = asdict(pattern)
                    # Convertir DofusGameState en dict
                    pattern_dict["game_state"] = {
                        "player_class": pattern.game_state.player_class.value,
                        "player_level": pattern.game_state.player_level,
                        "in_combat": pattern.game_state.in_combat,
                        "available_ap": pattern.game_state.available_ap,
                        "available_mp": pattern.game_state.available_mp,
                        "current_health": pattern.game_state.current_health,
                        "timestamp": pattern.game_state.timestamp
                    }
                    patterns_data.append(pattern_dict)

                with open(patterns_file, 'w', encoding='utf-8') as f:
                    json.dump(patterns_data, f, indent=2, ensure_ascii=False)

            # Sauvegarder métriques temps réel
            metrics_file = self.session_path / "real_time_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.real_time_metrics, f, indent=2, ensure_ascii=False)

            print(f"💾 Session sauvegardée : {self.session_path}")

        except Exception as e:
            print(f"❌ Erreur sauvegarde session : {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de la session courante"""
        if not self.current_session:
            return {"error": "Aucune session active"}

        return {
            "session_id": self.current_session.session_id,
            "duration": time.time() - self.current_session.start_time,
            "total_captures": self.current_session.total_captures,
            "patterns_learned": self.current_session.patterns_learned,
            "combat_sequences": self.current_session.combat_sequences,
            "exploration_sequences": self.current_session.exploration_sequences,
            "errors": self.current_session.errors_encountered,
            "real_time_metrics": self.real_time_metrics
        }

    def register_pattern_callback(self, pattern_type: str, callback: Callable):
        """Enregistre un callback pour un type de pattern"""
        if pattern_type not in self.pattern_callbacks:
            self.pattern_callbacks[pattern_type] = []
        self.pattern_callbacks[pattern_type].append(callback)

# Factory function
def get_passive_learning_engine(session_dir: str = "passive_learning_sessions") -> PassiveDofusLearningEngine:
    """Factory pour créer le moteur d'apprentissage passif"""
    return PassiveDofusLearningEngine(session_dir)

if __name__ == "__main__":
    # Test rapide
    engine = get_passive_learning_engine()

    print("🧪 Test Moteur d'Apprentissage Passif")
    print("Démarrage d'une session de test...")

    try:
        session_id = engine.start_passive_learning_session("IOPS", 150, "test")

        print(f"Session démarrée : {session_id}")
        print("Appuyez sur Ctrl+C pour arrêter...")

        # Laisser tourner quelques secondes pour test
        time.sleep(10)

        session = engine.stop_passive_learning_session()
        print(f"Session terminée : {session.session_id}")
        print(f"Captures: {session.total_captures}")
        print(f"Patterns: {session.patterns_learned}")

    except KeyboardInterrupt:
        print("\\nArrêt demandé par l'utilisateur")
        if engine.is_running:
            engine.stop_passive_learning_session()
    except Exception as e:
        print(f"❌ Erreur test : {e}")