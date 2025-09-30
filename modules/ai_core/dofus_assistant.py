"""
DOFUS Assistant - Intelligence Artificielle Ultime 2025
Orchestre tous les composants pour créer l'assistant IA complet

Fonctionnalités:
- Coordination de tous les modules (vision, apprentissage, overlay, outils)
- Interface utilisateur unifiée
- Gestion des états et modes de fonctionnement
- Système de recommandations intelligentes
- Automation adaptative basée sur l'apprentissage
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from enum import Enum
import json

# Imports des modules développés
from ..vision.game_capture import GameCapture
from ..vision.ocr_engine import OCREngine
from .learning_engine import LearningEngine, GameAction, GameSituation
from ..overlay.intelligent_overlay import IntelligentOverlay, OverlayConfig, OverlayType
from ..external.tool_integration import ToolIntegrationManager

class AssistantMode(Enum):
    LEARNING = "learning"           # Mode apprentissage passif
    ADVISORY = "advisory"           # Mode conseil temps réel
    AUTOMATION = "automation"       # Mode automation (avec supervision)
    HYBRID = "hybrid"              # Mode hybride (défaut)

class AssistantState(Enum):
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class AssistantConfig:
    """Configuration de l'assistant"""
    mode: AssistantMode = AssistantMode.HYBRID
    enable_learning: bool = True
    enable_overlay: bool = True
    enable_external_tools: bool = True
    capture_fps: int = 30
    analysis_interval: float = 1.0
    overlay_transparency: float = 0.8
    auto_start_tools: bool = False
    save_sessions: bool = True

class DofusAssistant:
    """Assistant IA DOFUS Ultime - Coordinateur principal"""

    def __init__(self, config: AssistantConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # État assistant
        self.state = AssistantState.INACTIVE
        self.current_mode = config.mode

        # Composants principaux
        self.game_capture: Optional[GameCapture] = None
        self.ocr_engine: Optional[OCREngine] = None
        self.learning_engine: Optional[LearningEngine] = None
        self.overlay: Optional[IntelligentOverlay] = None
        self.tool_manager: Optional[ToolIntegrationManager] = None

        # Threads de traitement
        self.main_thread: Optional[threading.Thread] = None
        self.running = False

        # État de jeu
        self.current_situation: Optional[GameSituation] = None
        self.last_screenshot = None
        self.last_ocr_results = []

        # Métriques et statistiques
        self.session_stats = {
            "start_time": 0.0,
            "actions_observed": 0,
            "recommendations_given": 0,
            "automations_performed": 0,
            "accuracy_score": 0.0
        }

        # Cache pour optimisation
        self.recommendation_cache: Dict[str, Any] = {}
        self.last_analysis_time = 0.0

        self.logger.info("DofusAssistant initialisé")

    def initialize(self) -> bool:
        """Initialise tous les composants de l'assistant"""
        try:
            self.state = AssistantState.INITIALIZING
            self.logger.info("Initialisation assistant IA DOFUS...")

            # 1. Initialiser capture de jeu
            self.logger.info("Initialisation GameCapture...")
            self.game_capture = GameCapture()
            if not self.game_capture.initialize():
                self.logger.error("Échec initialisation GameCapture")
                return False

            # 2. Initialiser OCR
            self.logger.info("Initialisation OCR Engine...")
            self.ocr_engine = OCREngine()
            if not self.ocr_engine.initialize():
                self.logger.error("Échec initialisation OCR Engine")
                return False

            # 3. Initialiser apprentissage
            if self.config.enable_learning:
                self.logger.info("Initialisation Learning Engine...")
                self.learning_engine = LearningEngine(self.data_dir / "learning")
                if not self.learning_engine.initialize():
                    self.logger.error("Échec initialisation Learning Engine")
                    return False

            # 4. Initialiser overlay
            if self.config.enable_overlay:
                self.logger.info("Initialisation Overlay...")
                overlay_config = OverlayConfig(
                    enable_overlay=True,
                    transparency=self.config.overlay_transparency,
                    max_elements=10
                )
                self.overlay = IntelligentOverlay(overlay_config)
                if not self.overlay.initialize():
                    self.logger.error("Échec initialisation Overlay")
                    return False

            # 5. Initialiser intégration outils
            if self.config.enable_external_tools:
                self.logger.info("Initialisation Tool Manager...")
                self.tool_manager = ToolIntegrationManager(self.data_dir / "tools")
                if not self.tool_manager.initialize():
                    self.logger.error("Échec initialisation Tool Manager")
                    return False

            self.state = AssistantState.ACTIVE
            self.logger.info("Assistant IA DOFUS initialisé avec succès!")
            return True

        except Exception as e:
            self.state = AssistantState.ERROR
            self.logger.error(f"Erreur initialisation assistant: {e}")
            return False

    def start(self) -> bool:
        """Démarre l'assistant"""
        if self.state != AssistantState.ACTIVE:
            self.logger.error("Assistant non initialisé")
            return False

        if self.running:
            self.logger.warning("Assistant déjà en cours")
            return False

        try:
            self.running = True
            self.session_stats["start_time"] = time.time()

            # Démarrer composants
            if self.learning_engine:
                self.learning_engine.start_learning()

            if self.overlay:
                self.overlay.start()

            if self.tool_manager:
                self.tool_manager.start_sync()

            # Lancer outils externes si configuré
            if self.config.auto_start_tools and self.tool_manager:
                self._auto_start_external_tools()

            # Démarrer thread principal
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()

            self.logger.info(f"Assistant démarré en mode {self.current_mode.value}")
            return True

        except Exception as e:
            self.logger.error(f"Erreur démarrage assistant: {e}")
            return False

    def stop(self):
        """Arrête l'assistant"""
        self.logger.info("Arrêt assistant...")
        self.running = False

        # Arrêter thread principal
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)

        # Arrêter composants
        if self.learning_engine:
            self.learning_engine.stop_learning()

        if self.overlay:
            self.overlay.stop()

        if self.tool_manager:
            self.tool_manager.stop_sync()

        # Sauvegarder session
        if self.config.save_sessions:
            self._save_session()

        self.state = AssistantState.INACTIVE
        self.logger.info("Assistant arrêté")

    def _main_loop(self):
        """Boucle principale de l'assistant"""
        self.logger.info("Boucle principale démarrée")

        while self.running:
            try:
                start_time = time.time()

                # 1. Capturer écran
                screenshot = self.game_capture.capture_screenshot_fast()
                if screenshot is None:
                    time.sleep(0.1)
                    continue

                self.last_screenshot = screenshot

                # 2. Analyser périodiquement
                if start_time - self.last_analysis_time >= self.config.analysis_interval:
                    self._analyze_game_state(screenshot)
                    self.last_analysis_time = start_time

                # 3. Traiter selon mode
                if self.current_mode == AssistantMode.LEARNING:
                    self._process_learning_mode()
                elif self.current_mode == AssistantMode.ADVISORY:
                    self._process_advisory_mode()
                elif self.current_mode == AssistantMode.AUTOMATION:
                    self._process_automation_mode()
                elif self.current_mode == AssistantMode.HYBRID:
                    self._process_hybrid_mode()

                # 4. Maintenir FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / self.config.capture_fps) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Erreur boucle principale: {e}")
                time.sleep(1.0)

    def _analyze_game_state(self, screenshot):
        """Analyse l'état actuel du jeu"""
        try:
            # 1. OCR pour extraire texte
            self.last_ocr_results = self.ocr_engine.extract_text_multi_engine(screenshot)

            # 2. Détecter situation avec learning engine
            if self.learning_engine:
                self.current_situation = self.learning_engine.observe_situation(
                    screenshot, self.last_ocr_results
                )

            # 3. Mettre à jour overlay avec infos
            if self.overlay:
                self._update_overlay_information()

        except Exception as e:
            self.logger.error(f"Erreur analyse état jeu: {e}")

    def _process_learning_mode(self):
        """Mode apprentissage - observation passive"""
        # Observer les actions sans intervenir
        pass

    def _process_advisory_mode(self):
        """Mode conseil - recommandations visuelles"""
        if not self.current_situation or not self.overlay:
            return

        try:
            # Obtenir recommandations basées sur situation
            recommendations = self._get_context_recommendations()

            # Afficher conseils via overlay
            for i, rec in enumerate(recommendations[:3]):  # Max 3 conseils
                if rec["type"] == "spell":
                    self.overlay.highlight_spell(
                        rec["position"], rec["name"], priority=rec["priority"]
                    )
                elif rec["type"] == "movement":
                    self.overlay.suggest_movement(
                        rec["from"], rec["to"], rec["reason"]
                    )
                elif rec["type"] == "target":
                    self.overlay.mark_target_priority(
                        rec["position"], rec["priority"]
                    )

            self.session_stats["recommendations_given"] += len(recommendations)

        except Exception as e:
            self.logger.error(f"Erreur mode advisory: {e}")

    def _process_automation_mode(self):
        """Mode automation - actions automatiques avec supervision"""
        # TODO: Implémenter automation sécurisée
        pass

    def _process_hybrid_mode(self):
        """Mode hybride - combinaison apprentissage et conseil"""
        self._process_learning_mode()
        self._process_advisory_mode()

    def _get_context_recommendations(self) -> List[Dict[str, Any]]:
        """Obtient recommandations contextuelles"""
        if not self.current_situation:
            return []

        recommendations = []

        try:
            # Recommandations du learning engine
            if self.learning_engine:
                learned_recs = self.learning_engine.get_action_recommendations(
                    self.current_situation.situation_type
                )

                for rec in learned_recs:
                    recommendations.append({
                        "type": "learned_action",
                        "confidence": rec["confidence"],
                        "actions": rec["action_sequence"],
                        "source": "learning"
                    })

            # Recommandations des outils externes
            if self.tool_manager:
                # Quêtes
                if self.current_situation.situation_type == "quest":
                    quest_guidance = self._get_quest_recommendations()
                    recommendations.extend(quest_guidance)

                # Crafting
                if self.current_situation.situation_type == "crafting":
                    craft_guidance = self._get_craft_recommendations()
                    recommendations.extend(craft_guidance)

            # Recommandations basiques selon situation
            basic_recs = self._get_basic_recommendations()
            recommendations.extend(basic_recs)

        except Exception as e:
            self.logger.error(f"Erreur génération recommandations: {e}")

        return sorted(recommendations, key=lambda x: x.get("confidence", 0.5), reverse=True)

    def _get_quest_recommendations(self) -> List[Dict[str, Any]]:
        """Recommandations spécifiques aux quêtes"""
        recommendations = []

        # Rechercher quête active via OCR
        quest_names = []
        for ocr_result in self.last_ocr_results:
            text = ocr_result.text.lower()
            if any(word in text for word in ["objectif", "quete", "mission"]):
                quest_names.append(text)

        # Obtenir guidage depuis outils externes
        for quest_name in quest_names[:1]:  # Première quête trouvée
            quest_data = self.tool_manager.get_quest_guidance(quest_name)
            if quest_data and quest_data.completion_steps:
                recommendations.append({
                    "type": "quest_step",
                    "quest_name": quest_data.name,
                    "next_step": quest_data.completion_steps[0],
                    "confidence": 0.8,
                    "source": "external_tools"
                })

        return recommendations

    def _get_craft_recommendations(self) -> List[Dict[str, Any]]:
        """Recommandations de crafting"""
        recommendations = []

        # Détecter items craftables via OCR
        for ocr_result in self.last_ocr_results:
            text = ocr_result.text
            recipe = self.tool_manager.get_crafting_optimization(text)
            if recipe:
                recommendations.append({
                    "type": "craft_recipe",
                    "item": recipe.item_name,
                    "ingredients": recipe.ingredients,
                    "success_rate": recipe.success_rate,
                    "confidence": 0.7,
                    "source": "external_tools"
                })

        return recommendations

    def _get_basic_recommendations(self) -> List[Dict[str, Any]]:
        """Recommandations basiques selon contexte"""
        recommendations = []

        if not self.current_situation:
            return recommendations

        situation_type = self.current_situation.situation_type

        # Recommandations par type de situation
        if situation_type == "combat":
            recommendations.extend([
                {
                    "type": "spell",
                    "name": "Sort optimal",
                    "position": (400, 300),
                    "priority": 7,
                    "confidence": 0.6,
                    "source": "basic"
                }
            ])
        elif situation_type == "exploration":
            recommendations.extend([
                {
                    "type": "movement",
                    "from": (300, 300),
                    "to": (400, 250),
                    "reason": "Zone intéressante",
                    "confidence": 0.5,
                    "source": "basic"
                }
            ])

        return recommendations

    def _update_overlay_information(self):
        """Met à jour les informations d'overlay"""
        if not self.overlay:
            return

        try:
            # Afficher situation actuelle
            if self.current_situation:
                situation_text = f"Situation: {self.current_situation.situation_type}\n"
                situation_text += f"Confiance: {self.current_situation.confidence:.2f}"

                # Statistiques de session
                stats = self.get_session_statistics()
                stats_text = "\n".join(f"{k}: {v}" for k, v in stats.items())

                self.overlay.show_performance_stats(
                    {"situation": situation_text, **stats},
                    position=(10, 10)
                )

        except Exception as e:
            self.logger.error(f"Erreur mise à jour overlay: {e}")

    def _auto_start_external_tools(self):
        """Lance automatiquement les outils externes"""
        if not self.tool_manager:
            return

        tools_to_start = ["Dofus Guide", "Ganymede"]

        for tool_name in tools_to_start:
            try:
                if self.tool_manager.launch_tool(tool_name):
                    self.logger.info(f"Outil {tool_name} lancé automatiquement")
                    time.sleep(2.0)  # Attendre entre lancements
            except Exception as e:
                self.logger.error(f"Erreur lancement auto {tool_name}: {e}")

    def observe_user_action(self, action_type: str, coordinates: Tuple[int, int],
                          success: bool = True, target_info: Dict[str, Any] = None):
        """Observe une action utilisateur pour apprentissage"""
        if not self.learning_engine:
            return

        action = GameAction(
            timestamp=time.time(),
            action_type=action_type,
            coordinates=coordinates,
            context=self.current_situation.situation_type if self.current_situation else "unknown",
            success=success,
            target_info=target_info or {},
            screen_hash=str(hash(str(self.last_screenshot.tobytes()))) if self.last_screenshot is not None else ""
        )

        self.learning_engine.observe_action(action)
        self.session_stats["actions_observed"] += 1

    def change_mode(self, new_mode: AssistantMode):
        """Change le mode de fonctionnement"""
        self.current_mode = new_mode
        self.logger.info(f"Mode changé vers: {new_mode.value}")

        # Effacer overlay si changement vers mode learning
        if new_mode == AssistantMode.LEARNING and self.overlay:
            self.overlay.clear_overlay()

    def pause(self):
        """Met en pause l'assistant"""
        self.state = AssistantState.PAUSED
        self.logger.info("Assistant mis en pause")

    def resume(self):
        """Reprend l'assistant"""
        if self.state == AssistantState.PAUSED:
            self.state = AssistantState.ACTIVE
            self.logger.info("Assistant repris")

    def get_session_statistics(self) -> Dict[str, Any]:
        """Retourne statistiques de session"""
        current_time = time.time()
        uptime = current_time - self.session_stats["start_time"] if self.session_stats["start_time"] > 0 else 0

        stats = {
            "uptime_minutes": round(uptime / 60, 1),
            "state": self.state.value,
            "mode": self.current_mode.value,
            "actions_observed": self.session_stats["actions_observed"],
            "recommendations_given": self.session_stats["recommendations_given"]
        }

        # Ajouter stats des composants
        if self.learning_engine:
            stats.update(self.learning_engine.get_learning_statistics())

        if self.game_capture:
            perf_stats = self.game_capture.get_performance_stats()
            stats["capture_fps"] = perf_stats["actual_fps"]

        return stats

    def _save_session(self):
        """Sauvegarde session actuelle"""
        try:
            session_file = self.data_dir / f"session_{int(time.time())}.json"
            session_data = {
                "config": {
                    "mode": self.config.mode.value,
                    "capture_fps": self.config.capture_fps
                },
                "statistics": self.get_session_statistics(),
                "timestamp": time.time()
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Session sauvegardée: {session_file}")

        except Exception as e:
            self.logger.error(f"Erreur sauvegarde session: {e}")

    def cleanup(self):
        """Nettoyage complet des ressources"""
        self.stop()

        if self.game_capture:
            self.game_capture.cleanup()

        if self.ocr_engine:
            self.ocr_engine.cleanup()

        if self.learning_engine:
            self.learning_engine.cleanup()

        if self.overlay:
            self.overlay.cleanup()

        if self.tool_manager:
            self.tool_manager.cleanup()

        self.logger.info("DofusAssistant nettoyé")

# Factory function
def create_dofus_assistant(config: Optional[AssistantConfig] = None,
                          data_dir: Optional[Path] = None) -> DofusAssistant:
    """Crée instance DofusAssistant configurée"""
    if config is None:
        config = AssistantConfig()

    if data_dir is None:
        data_dir = Path("data")

    assistant = DofusAssistant(config, data_dir)
    if assistant.initialize():
        return assistant
    else:
        raise RuntimeError("Impossible d'initialiser DofusAssistant")

# Test de base
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        config = AssistantConfig(
            mode=AssistantMode.HYBRID,
            enable_learning=True,
            enable_overlay=True,
            capture_fps=30
        )

        assistant = create_dofus_assistant(config, Path("data"))

        print("Test Assistant IA DOFUS...")

        if assistant.start():
            print("Assistant démarré - Test 10 secondes")

            # Simuler action utilisateur
            assistant.observe_user_action("spell_cast", (400, 300), True)

            time.sleep(10)

            # Afficher stats
            stats = assistant.get_session_statistics()
            print(f"Statistiques: {stats}")

            assistant.stop()
        else:
            print("Impossible de démarrer assistant")

    except Exception as e:
        print(f"Erreur test: {e}")
    finally:
        if 'assistant' in locals():
            assistant.cleanup()