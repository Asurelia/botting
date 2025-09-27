"""
Système HRM Intelligence Principal - Point d'entrée unifiée
Orchestrateur central pour tous les composants d'intelligence HRM

Auteur: Claude Code
Version: 1.0.0
"""

import sys
import os
import time
import json
import logging
import threading
import signal
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

# Ajouter le chemin du module HRM
sys.path.append(str(Path(__file__).parent))

# Imports du système HRM
try:
    import hrm_core
    import adaptive_learner
    import intelligent_decision_maker
    import quest_tracker

    from hrm_core import HRMBot, GameState, HRMDecision
    from adaptive_learner import AdaptiveLearner, LearningExperience
    from intelligent_decision_maker import IntelligentDecisionMaker
    from quest_tracker import QuestTracker, Quest
except ImportError as e:
    print(f"Erreur d'import HRM: {e}")
    print("Assurez-vous que tous les modules HRM sont installés")
    sys.exit(1)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('G:/Botting/logs/hrm_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HRMSystemConfig:
    """Configuration du système HRM"""
    # Chemins
    models_path: str = "G:/Botting/models"
    data_path: str = "G:/Botting/data/hrm"
    logs_path: str = "G:/Botting/logs"

    # Paramètres d'apprentissage
    learning_enabled: bool = True
    auto_save_interval: int = 300  # 5 minutes
    max_experiences: int = 10000

    # Paramètres de performance
    decision_timeout: float = 1.0  # 1 seconde max par décision
    screenshot_interval: float = 0.5  # Capture d'écran toutes les 0.5s
    quest_check_interval: float = 5.0  # Vérification quêtes toutes les 5s

    # Paramètres de comportement
    human_like_delays: bool = True
    random_actions_probability: float = 0.02  # 2% d'actions aléatoires
    adaptive_learning_rate: float = 0.001

    # Paramètres de debug
    debug_mode: bool = False
    save_screenshots: bool = False
    verbose_logging: bool = False

class HRMIntelligenceSystem:
    """Système d'intelligence HRM unifié"""

    def __init__(self, config: Optional[HRMSystemConfig] = None, player_id: str = "tactical_bot"):
        self.config = config or HRMSystemConfig()
        self.player_id = player_id
        self.running = False
        self.paused = False

        # Statistiques du système
        self.stats = {
            "start_time": time.time(),
            "decisions_made": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "quests_completed": 0,
            "learning_sessions": 0,
            "total_reward": 0.0
        }

        # Créer les répertoires nécessaires
        self._setup_directories()

        # Initialiser les composants
        self._initialize_components()

        # Handlers pour signaux système
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"🤖 Système HRM Intelligence initialisé pour {player_id}")

    def _setup_directories(self):
        """Crée les répertoires nécessaires"""
        directories = [
            self.config.models_path,
            self.config.data_path,
            self.config.logs_path,
            f"{self.config.data_path}/screenshots",
            f"{self.config.data_path}/learning",
            f"{self.config.data_path}/quests"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _initialize_components(self):
        """Initialise tous les composants HRM"""
        try:
            # Modèle HRM principal
            model_path = f"{self.config.models_path}/hrm_model_{self.player_id}.pth"
            self.hrm_bot = HRMBot(model_path if Path(model_path).exists() else None)

            # Apprentissage adaptatif
            self.adaptive_learner = AdaptiveLearner(
                player_id=self.player_id,
                max_experiences=self.config.max_experiences
            )

            # Décideur intelligent
            self.decision_maker = IntelligentDecisionMaker()

            # Tracker de quêtes
            quest_db_path = f"{self.config.data_path}/quests/quests_{self.player_id}.db"
            self.quest_tracker = QuestTracker(db_path=quest_db_path)

            # Timers pour actions périodiques
            self.last_save_time = time.time()
            self.last_screenshot_time = time.time()
            self.last_quest_check_time = time.time()

            logger.info("✅ Tous les composants HRM initialisés avec succès")

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {e}")
            raise

    def _signal_handler(self, signum, frame):
        """Gestionnaire pour les signaux système"""
        logger.info(f"Signal {signum} reçu, arrêt en cours...")
        self.stop()

    def get_current_game_state(self) -> Optional[GameState]:
        """
        Récupère l'état actuel du jeu
        À adapter selon l'interface de votre jeu spécifique
        """
        try:
            # TODO: Remplacer par la vraie logique de capture d'état
            # Ceci est un exemple de base

            current_time = time.time()

            # État simulé pour les tests
            mock_state = GameState(
                player_position=(100, 200),  # À remplacer par vraie position
                player_health=85.0,  # À remplacer par vraie santé
                player_mana=60.0,    # À remplacer par vrai mana
                player_level=15,     # À remplacer par vrai niveau
                nearby_entities=[],  # À remplacer par vraies entités
                available_actions=["move_up", "move_down", "attack", "use_potion"],
                current_quest=None,  # Sera mis à jour par quest_tracker
                inventory_state={},  # À remplacer par vrai inventaire
                timestamp=current_time,
                game_time=datetime.now().strftime("%H:%M"),
                fps=60.0,           # À remplacer par vrai FPS
                latency=25.0        # À remplacer par vraie latence
            )

            # Mettre à jour avec les quêtes actives
            active_quests = self.quest_tracker.get_active_quests()
            if active_quests:
                mock_state.current_quest = active_quests[0].title

            return mock_state

        except Exception as e:
            logger.error(f"Erreur lors de la capture d'état: {e}")
            return None

    def make_intelligent_decision(self, game_state: GameState) -> Optional[Dict[str, Any]]:
        """Prend une décision intelligente basée sur l'état du jeu"""
        try:
            start_time = time.time()

            # 1. Décision de base via HRM
            base_decision = self.hrm_bot.decide_action(game_state)

            # 2. Enrichissement par le décideur intelligent
            enhanced_decision = self.decision_maker.make_enhanced_decision(
                base_action=base_decision.action,
                confidence=base_decision.confidence,
                game_context={
                    "current_quest": game_state.current_quest,
                    "player_health": game_state.player_health,
                    "nearby_entities": len(game_state.nearby_entities),
                    "inventory_items": len(game_state.inventory_state)
                }
            )

            # 3. Adaptation comportementale humaine
            if self.config.human_like_delays:
                behavior_config = self.adaptive_learner.get_human_like_behavior()
                enhanced_decision["human_delay"] = behavior_config.get("reaction_delay", 0.1)

            # 4. Mise à jour des statistiques
            self.stats["decisions_made"] += 1
            decision_time = time.time() - start_time

            decision_data = {
                "base_decision": asdict(base_decision),
                "enhanced_decision": enhanced_decision,
                "decision_time": decision_time,
                "game_state_snapshot": {
                    "timestamp": game_state.timestamp,
                    "player_health": game_state.player_health,
                    "current_quest": game_state.current_quest
                }
            }

            if self.config.debug_mode:
                logger.debug(f"Décision prise en {decision_time:.3f}s: {enhanced_decision['final_action']}")

            return decision_data

        except Exception as e:
            logger.error(f"Erreur lors de la prise de décision: {e}")
            return None

    def learn_from_action_result(self, decision_data: Dict[str, Any], success: bool, reward: float = 0.0):
        """Apprend du résultat d'une action"""
        try:
            # Mise à jour des statistiques
            if success:
                self.stats["successful_actions"] += 1
            else:
                self.stats["failed_actions"] += 1

            self.stats["total_reward"] += reward

            # Apprentissage HRM
            base_decision = HRMDecision(**decision_data["base_decision"])
            self.hrm_bot.learn_from_outcome(base_decision, success, reward)

            # Apprentissage adaptatif
            if self.config.learning_enabled:
                experience = LearningExperience(
                    state_data=decision_data["game_state_snapshot"],
                    action_taken=decision_data["enhanced_decision"]["final_action"],
                    reward_received=reward,
                    outcome_success=success,
                    context_info=decision_data["enhanced_decision"].get("context", {})
                )

                self.adaptive_learner.add_experience(experience)
                self.stats["learning_sessions"] += 1

            if self.config.debug_mode:
                logger.debug(f"Apprentissage: Action={'SUCCÈS' if success else 'ÉCHEC'}, Reward={reward}")

        except Exception as e:
            logger.error(f"Erreur lors de l'apprentissage: {e}")

    def update_quest_progress(self):
        """Met à jour la progression des quêtes"""
        try:
            current_time = time.time()

            if current_time - self.last_quest_check_time >= self.config.quest_check_interval:
                # Détecter les quêtes via OCR
                detected_quests = self.quest_tracker.detect_quests_from_screen()

                # Obtenir des recommandations
                game_state = self.get_current_game_state()
                if game_state:
                    recommendations = self.quest_tracker.get_quest_recommendations({
                        "player_level": game_state.player_level,
                        "current_area": "auto_detected"  # À améliorer
                    })

                    if self.config.verbose_logging and recommendations:
                        logger.info(f"Recommandations de quêtes: {len(recommendations)} trouvées")

                self.last_quest_check_time = current_time

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des quêtes: {e}")

    def periodic_save(self):
        """Sauvegarde périodique des données"""
        try:
            current_time = time.time()

            if current_time - self.last_save_time >= self.config.auto_save_interval:
                # Sauvegarder le modèle HRM
                model_path = f"{self.config.models_path}/hrm_model_{self.player_id}.pth"
                self.hrm_bot.save_model(model_path)

                # Sauvegarder les données d'apprentissage
                learning_path = f"{self.config.data_path}/learning/adaptive_learning_{self.player_id}.json"
                self.adaptive_learner.save_learning_data(learning_path)

                # Sauvegarder les statistiques
                stats_path = f"{self.config.data_path}/system_stats_{self.player_id}.json"
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(self.stats, f, indent=2)

                self.last_save_time = current_time
                logger.info("💾 Sauvegarde automatique effectuée")

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")

    def run_main_loop(self):
        """Boucle principale du système"""
        logger.info("🚀 Démarrage de la boucle principale HRM Intelligence")
        self.running = True

        try:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue

                # Obtenir l'état du jeu
                game_state = self.get_current_game_state()
                if not game_state:
                    time.sleep(0.1)
                    continue

                # Prendre une décision intelligente
                decision_data = self.make_intelligent_decision(game_state)
                if not decision_data:
                    time.sleep(0.1)
                    continue

                # TODO: Exécuter l'action dans le jeu
                # action_result = execute_game_action(decision_data["enhanced_decision"]["final_action"])

                # Pour les tests, simuler un résultat
                import random
                simulated_success = random.random() > 0.3  # 70% de succès
                simulated_reward = random.uniform(1.0, 10.0) if simulated_success else 0.0

                # Apprendre du résultat
                self.learn_from_action_result(decision_data, simulated_success, simulated_reward)

                # Tâches périodiques
                self.update_quest_progress()
                self.periodic_save()

                # Délai basé sur la configuration
                if self.config.human_like_delays:
                    delay = decision_data.get("enhanced_decision", {}).get("human_delay", 0.1)
                else:
                    delay = 0.05

                time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("Interruption clavier détectée")
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale: {e}")
        finally:
            self.stop()

    def start(self):
        """Démarre le système en arrière-plan"""
        if self.running:
            logger.warning("Le système est déjà en cours d'exécution")
            return

        self.main_thread = threading.Thread(target=self.run_main_loop, daemon=True)
        self.main_thread.start()
        logger.info("✅ Système HRM Intelligence démarré en arrière-plan")

    def stop(self):
        """Arrête le système proprement"""
        logger.info("🛑 Arrêt du système HRM Intelligence...")
        self.running = False

        # Sauvegarde finale
        try:
            self.periodic_save()
            logger.info("💾 Sauvegarde finale effectuée")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde finale: {e}")

        logger.info("✅ Système HRM Intelligence arrêté")

    def pause(self):
        """Met en pause le système"""
        self.paused = True
        logger.info("⏸️  Système mis en pause")

    def resume(self):
        """Reprend l'exécution du système"""
        self.paused = False
        logger.info("▶️  Système repris")

    def get_system_status(self) -> Dict[str, Any]:
        """Retourne le statut du système"""
        uptime = time.time() - self.stats["start_time"]

        return {
            "running": self.running,
            "paused": self.paused,
            "uptime_seconds": uptime,
            "uptime_formatted": f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m {uptime%60:.0f}s",
            "statistics": self.stats.copy(),
            "performance": self.hrm_bot.get_performance_report(),
            "config": asdict(self.config)
        }

def main():
    """Point d'entrée principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Système HRM Intelligence pour TacticalBot")
    parser.add_argument("--player-id", default="tactical_bot", help="ID du joueur")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    parser.add_argument("--test", action="store_true", help="Mode test avec simulation")
    parser.add_argument("--config", help="Fichier de configuration JSON")

    args = parser.parse_args()

    # Configuration
    config = HRMSystemConfig()
    if args.config and Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    if args.debug:
        config.debug_mode = True
        config.verbose_logging = True

    # Initialisation du système
    try:
        system = HRMIntelligenceSystem(config=config, player_id=args.player_id)

        if args.test:
            logger.info("🧪 Mode test activé - exécution pour 30 secondes")
            system.start()
            time.sleep(30)
            system.stop()
        else:
            logger.info("🎮 Mode production - appuyez sur Ctrl+C pour arrêter")
            system.run_main_loop()

    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()