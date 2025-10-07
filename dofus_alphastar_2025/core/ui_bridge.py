#!/usr/bin/env python3
"""
UI Bridge - Pont entre l'interface utilisateur et les systèmes core
Connecte le dashboard moderne avec calibration, map system, observation mode, etc.
"""

import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

# Ajouter le root au path pour importer bot_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from bot_config import DOFUS_WINDOW_TITLE

# Imports des systèmes core
from core.calibration import create_calibrator
from core.map_system import create_map_graph, create_map_discovery
from core.external_data import create_dofusdb_client
from core.safety import create_observation_mode
from core.game_loop import create_game_engine
from core.vision_engine_v2.realtime_vision import create_realtime_vision
from core.actions import create_action_system
from core.decision.autonomous_brain import create_autonomous_brain


@dataclass
class UIState:
    """État global de l'UI"""
    bot_running: bool = False
    observation_mode: bool = True
    calibrated: bool = False
    current_map: Optional[str] = None
    character_hp: int = 100
    character_max_hp: int = 100
    character_pa: int = 6
    character_max_pa: int = 6
    actions_per_minute: float = 0.0
    safety_score: float = 100.0
    total_actions: int = 0
    blocked_actions: int = 0
    session_duration: float = 0.0


class UIBridge:
    """
    Pont entre l'UI et les systèmes core

    Gère la communication bidirectionnelle:
    - UI -> Core: Commandes utilisateur (start/stop, calibrate, etc.)
    - Core -> UI: Mises à jour d'état (stats, logs, progression)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # État de l'UI
        self.state = UIState()

        # Systèmes core
        self.calibrator = None
        self.map_graph = None
        self.map_discovery = None
        self.dofusdb_client = None
        self.observation_mode = None
        
        # Game Engine
        self.game_engine = None
        self.vision = None
        self.actions = None
        self.brain = None

        # Callbacks UI
        self.ui_update_callback: Optional[Callable] = None
        self.log_callback: Optional[Callable] = None

        # Thread de monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

        # Historique
        self.action_history: List[Dict[str, Any]] = []
        self.log_history: List[Dict[str, Any]] = []

        self.logger.info("UIBridge initialisé")

    # ========================================================================
    # INITIALISATION DES SYSTÈMES
    # ========================================================================

    def initialize_systems(self):
        """Initialise tous les systèmes core"""
        self.logger.info("Initialisation des systèmes core...")

        try:
            # Calibration
            self.calibrator = create_calibrator()
            self._log("Calibrator initialisé", "INFO")

            # Map System
            self.map_graph = create_map_graph()
            self.map_discovery = create_map_discovery(self.map_graph)  # Passer map_graph
            self._log("Map System initialisé", "INFO")

            # DofusDB Client
            self.dofusdb_client = create_dofusdb_client()
            self._log("DofusDB Client initialisé", "INFO")

            # Safety: Observation Mode (CRITIQUE!)
            self.observation_mode = create_observation_mode()
            self.state.observation_mode = self.observation_mode.is_enabled()
            self._log("Observation Mode initialisé (SÉCURITÉ ACTIVE)", "WARNING")

            self.logger.info("[OK] Tous les systèmes initialisés")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation systèmes: {e}")
            self._log(f"ERREUR initialisation: {e}", "ERROR")
            return False
    # COMMANDES UI -> CORE
    # ========================================================================

    def start_bot(self, observation_only: bool = True):
        """Démarre le bot"""
        if self.state.bot_running:
            self._log("Bot déjà en cours d'exécution", "WARNING")
            return False

        # Calibration optionnelle maintenant
        if not self.state.calibrated:
            self._log("ATTENTION: Bot démarre sans calibration (utilise valeurs par défaut)", "WARNING")

        self._log(f"Démarrage du bot (observation: {observation_only})...", "INFO")

        # Créer Game Engine si nécessaire
        if not self.game_engine:
            self._log("Initialisation Game Engine...", "INFO")
            try:
                self.game_engine = create_game_engine(target_fps=5, observation_mode=observation_only)
                self.vision = create_realtime_vision(window_title=DOFUS_WINDOW_TITLE)
                self.actions = create_action_system(window_title=DOFUS_WINDOW_TITLE)
                self.brain = create_autonomous_brain()
                
                # Connecter
                self.game_engine.vision_system = self.vision
                self.game_engine.action_system = self.actions
                self.game_engine.decision_engine = self.brain
                
                self._log("Game Engine initialisé", "INFO")
            except Exception as e:
                self._log(f"ERREUR Game Engine: {e}", "ERROR")
                return False
        
        # Initialiser actions
        if not self.actions.initialize():
            self._log("ERREUR: Impossible d'initialiser actions", "ERROR")
            return False
        
        # Démarrer Game Engine
        if not self.game_engine.start():
            self._log("ERREUR: Impossible de démarrer Game Engine", "ERROR")
            return False

        # Active/désactive observation mode
        if observation_only and self.observation_mode:
            self.observation_mode.enable()
            self._log("MODE OBSERVATION ACTIVÉ - Aucune action ne sera exécutée", "WARNING")

        self.state.bot_running = True
        self.state.observation_mode = observation_only

        # Démarre monitoring
        self._start_monitoring()

        self._log("✅ Bot démarré avec succès", "INFO")
        self._update_ui()
        return True

    def stop_bot(self):
        """Arrête le bot"""
        if not self.state.bot_running:
            self._log("Bot n'est pas en cours d'exécution", "WARNING")
            return False

        self._log("Arrêt du bot...", "INFO")

        # Arrêter Game Engine
        if self.game_engine:
            self.game_engine.stop()
            self._log("Game Engine arrêté", "INFO")

        self.state.bot_running = False
        self._stop_monitoring()

        # Sauvegarde observations si en mode observation
        if self.observation_mode and self.observation_mode.is_enabled():
            self.observation_mode.save_observations()
            self._log("Observations sauvegardées", "INFO")

        self._log("✅ Bot arrêté", "INFO")
        self._update_ui()
        return True

    def run_calibration(self):
        """Lance la calibration automatique"""
        self._log("Lancement calibration automatique...", "INFO")

        if not self.calibrator:
            self._log("Calibrator non initialisé!", "ERROR")
            return False

        try:
            # Calibration en thread séparé pour ne pas bloquer l'UI
            def calibrate_thread():
                result = self.calibrator.run_full_calibration()

                if result.success:
                    self.state.calibrated = True
                    self._log(f"[OK] Calibration réussie ({result.duration_seconds:.1f}s)", "INFO")
                    self._log(f"  Éléments UI: {len(result.ui_elements)}", "INFO")
                    self._log(f"  Raccourcis: {len(result.shortcuts)}", "INFO")
                else:
                    self._log("Calibration échouée", "ERROR")

                self._update_ui()

            thread = threading.Thread(target=calibrate_thread, daemon=True)
            thread.start()

            return True

        except Exception as e:
            self.logger.error(f"Erreur calibration: {e}")
            self._log(f"ERREUR calibration: {e}", "ERROR")
            return False

    def toggle_observation_mode(self):
        """Active/désactive le mode observation"""
        if not self.observation_mode:
            self._log("Observation Mode non initialisé", "ERROR")
            return False

        if self.observation_mode.is_enabled():
            # DANGER: Désactiver observation
            self._log("[WARNING] DÉSACTIVATION MODE OBSERVATION [WARNING]", "CRITICAL")
            self._log("[WARNING] LE BOT VA AGIR RÉELLEMENT [WARNING]", "CRITICAL")
            self._log("[WARNING] COMPTE JETABLE REQUIS [WARNING]", "CRITICAL")
            self.observation_mode.disable()
            self.state.observation_mode = False
        else:
            # Safe: Activer observation
            self.observation_mode.enable()
            self.state.observation_mode = True
            self._log("MODE OBSERVATION ACTIVÉ (sécurité)", "INFO")

        self._update_ui()
        return True

    def test_dofusdb_connection(self):
        """Teste la connexion à DofusDB"""
        self._log("Test connexion DofusDB...", "INFO")

        if not self.dofusdb_client:
            self._log("DofusDB Client non initialisé", "ERROR")
            return False

        try:
            # Recherche test
            items = self.dofusdb_client.search_items("Dofus", limit=5)

            if items:
                self._log(f"[OK] DofusDB connecté ({len(items)} résultats)", "INFO")
                for item in items[:3]:
                    self._log(f"  - {item.name} (lvl {item.level})", "INFO")
                return True
            else:
                self._log("DofusDB: Aucun résultat (API offline?)", "WARNING")
                return False

        except Exception as e:
            self.logger.error(f"Erreur test DofusDB: {e}")
            self._log(f"ERREUR DofusDB: {e}", "ERROR")
            return False

    # ========================================================================
    # MONITORING TEMPS RÉEL
    # ========================================================================

    def _start_monitoring(self):
        """Démarre le thread de monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Monitoring démarré")

    def _stop_monitoring(self):
        """Arrête le thread de monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        self.logger.info("Monitoring arrêté")

    def _monitoring_loop(self):
        """Boucle de monitoring (mise à jour stats, etc.)"""
        start_time = time.time()

        while self.monitoring_active:
            try:
                # Mise à jour durée session
                self.state.session_duration = time.time() - start_time

                # Mise à jour stats observation mode
                if self.observation_mode:
                    stats = self.observation_mode.get_stats()
                    self.state.total_actions = stats['total_decisions']
                    self.state.blocked_actions = stats['actions_blocked']
                    self.state.actions_per_minute = stats.get('actions_per_minute', 0)

                    # Safety score
                    if stats['total_decisions'] > 10:
                        analysis = self.observation_mode.analyze_observations()
                        self.state.safety_score = analysis.get('safety_score', 100.0)

                # Mise à jour UI
                self._update_ui()

                # Pause
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Erreur monitoring: {e}")
                time.sleep(5.0)

    # ========================================================================
    # COMMUNICATION AVEC UI
    # ========================================================================

    def set_ui_update_callback(self, callback: Callable):
        """Définit le callback pour mettre à jour l'UI"""
        self.ui_update_callback = callback
        self.logger.info("UI update callback configuré")

    def set_log_callback(self, callback: Callable):
        """Définit le callback pour les logs"""
        self.log_callback = callback
        self.logger.info("Log callback configuré")

    def _update_ui(self):
        """Envoie une mise à jour à l'UI"""
        if self.ui_update_callback:
            try:
                self.ui_update_callback(asdict(self.state))
            except Exception as e:
                self.logger.error(f"Erreur callback UI: {e}")

    def _log(self, message: str, level: str = "INFO"):
        """Envoie un log à l'UI"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }

        self.log_history.append(log_entry)

        # Limite historique
        if len(self.log_history) > 1000:
            self.log_history = self.log_history[-1000:]

        if self.log_callback:
            try:
                self.log_callback(log_entry)
            except Exception as e:
                self.logger.error(f"Erreur callback log: {e}")

    # ========================================================================
    # ACCESSEURS
    # ========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état actuel"""
        return asdict(self.state)

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retourne les logs récents"""
        return self.log_history[-limit:]

    def get_observation_stats(self) -> Optional[Dict[str, Any]]:
        """Retourne les statistiques d'observation"""
        if self.observation_mode:
            return self.observation_mode.get_stats()
        return None

    def get_dofusdb_stats(self) -> Optional[Dict[str, Any]]:
        """Retourne les statistiques DofusDB"""
        if self.dofusdb_client:
            return self.dofusdb_client.get_stats()
        return None


def create_ui_bridge() -> UIBridge:
    """Factory function pour créer UIBridge"""
    bridge = UIBridge()
    bridge.initialize_systems()
    return bridge