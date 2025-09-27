"""
Test d'intégration complet du système HRM Intelligence
Valide tous les composants ensemble et teste la fonctionnalité end-to-end

Auteur: Claude Code
Version: 1.0.0
"""

import sys
import os
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

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
    print(f"Erreur d'import: {e}")
    print("Assurez-vous que tous les modules HRM sont présents")
    sys.exit(1)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('G:/Botting/logs/hrm_integration_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HRMIntegrationTester:
    """Testeur d'intégration pour le système HRM complet"""

    def __init__(self):
        self.test_results = {
            "hrm_core": False,
            "adaptive_learner": False,
            "decision_maker": False,
            "quest_tracker": False,
            "integration": False
        }

        # Créer les répertoires nécessaires
        os.makedirs("G:/Botting/logs", exist_ok=True)
        os.makedirs("G:/Botting/data/hrm", exist_ok=True)
        os.makedirs("G:/Botting/models", exist_ok=True)

    def create_mock_game_state(self) -> GameState:
        """Crée un état de jeu simulé pour les tests"""
        return GameState(
            player_position=(100, 200),
            player_health=85.5,
            player_mana=60.0,
            player_level=15,
            nearby_entities=[
                {"type": "enemy", "distance": 50, "health": 100},
                {"type": "npc", "distance": 30, "health": 100}
            ],
            available_actions=["move_up", "move_down", "attack", "use_potion"],
            current_quest="Defeat 5 Goblins",
            inventory_state={"health_potion": 3, "mana_potion": 2, "sword": 1},
            timestamp=time.time(),
            game_time="14:30",
            fps=60.0,
            latency=25.0
        )

    def test_hrm_core(self) -> bool:
        """Test du module HRM Core"""
        logger.info("Test du module HRM Core...")

        try:
            # Initialisation du bot HRM
            hrm_bot = HRMBot()

            # Test de décision basique
            game_state = self.create_mock_game_state()
            decision = hrm_bot.decide_action(game_state)

            # Vérifications
            assert isinstance(decision, HRMDecision), "La décision doit être de type HRMDecision"
            assert decision.action in hrm_bot.action_mapping.values(), "Action invalide"
            assert 0.0 <= decision.confidence <= 1.0, "Confiance doit être entre 0 et 1"
            assert decision.execution_time > 0, "Temps d'exécution doit être positif"

            # Test d'apprentissage
            hrm_bot.learn_from_outcome(decision, outcome_success=True, reward=10.0)

            # Test de sauvegarde/chargement
            model_path = "G:/Botting/models/test_hrm_model.pth"
            hrm_bot.save_model(model_path)

            # Nouveau bot pour test de chargement
            hrm_bot2 = HRMBot(model_path=model_path)

            # Test de rapport de performance
            report = hrm_bot.get_performance_report()
            assert "total_decisions" in report, "Rapport manquant"

            logger.info("✅ HRM Core test RÉUSSI")
            return True

        except Exception as e:
            logger.error(f"❌ HRM Core test ÉCHOUÉ: {e}")
            logger.error(traceback.format_exc())
            return False

    def test_adaptive_learner(self) -> bool:
        """Test du module Adaptive Learner"""
        logger.info("Test du module Adaptive Learner...")

        try:
            # Initialisation de l'apprenant adaptatif
            learner = AdaptiveLearner(player_id="test_player")

            # Création d'expériences d'apprentissage
            game_state = self.create_mock_game_state()

            experience = LearningExperience(
                state_data=game_state.__dict__,
                action_taken="attack",
                reward_received=15.0,
                outcome_success=True,
                context_info={"enemy_type": "goblin", "weapon_used": "sword"}
            )

            # Test d'ajout d'expérience
            learner.add_experience(experience)

            # Test d'adaptation de stratégie
            new_strategy = learner.adapt_strategy(game_state)
            assert isinstance(new_strategy, dict), "Stratégie doit être un dictionnaire"

            # Test de comportement humain
            behavior_config = learner.get_human_like_behavior()
            assert "reaction_delay" in behavior_config, "Configuration comportement manquante"

            # Test de mise à jour des métriques
            learner.update_performance_metrics(success=True, execution_time=0.1)

            # Test de sauvegarde
            learner.save_learning_data("G:/Botting/data/hrm/test_learning.json")

            logger.info("✅ Adaptive Learner test RÉUSSI")
            return True

        except Exception as e:
            logger.error(f"❌ Adaptive Learner test ÉCHOUÉ: {e}")
            logger.error(traceback.format_exc())
            return False

    def test_decision_maker(self) -> bool:
        """Test du module Decision Maker"""
        logger.info("Test du module Decision Maker...")

        try:
            # Initialisation du décideur intelligent
            decision_maker = IntelligentDecisionMaker()

            # Mock d'une capture d'écran
            mock_screenshot = Mock()
            mock_screenshot.size = (1920, 1080)

            # Test d'analyse avec screenshot simulé
            with patch('PIL.ImageGrab.grab', return_value=mock_screenshot):
                enhanced_decision = decision_maker.make_enhanced_decision(
                    base_decision="attack",
                    confidence=0.8,
                    game_context={"area": "dungeon", "time": "night"}
                )

                assert isinstance(enhanced_decision, dict), "Décision enrichie doit être un dict"
                assert "final_action" in enhanced_decision, "Action finale manquante"
                assert "risk_assessment" in enhanced_decision, "Évaluation risque manquante"

            # Test de planification stratégique
            strategy_plan = decision_maker.strategic_planner.create_long_term_plan({
                "current_level": 15,
                "current_quest": "Defeat 5 Goblins",
                "inventory": {"health_potion": 3}
            })

            assert isinstance(strategy_plan, dict), "Plan stratégique doit être un dict"

            logger.info("✅ Decision Maker test RÉUSSI")
            return True

        except Exception as e:
            logger.error(f"❌ Decision Maker test ÉCHOUÉ: {e}")
            logger.error(traceback.format_exc())
            return False

    def test_quest_tracker(self) -> bool:
        """Test du module Quest Tracker"""
        logger.info("Test du module Quest Tracker...")

        try:
            # Initialisation du tracker de quêtes
            quest_tracker = QuestTracker(db_path="G:/Botting/data/hrm/test_quests.db")

            # Test de création de quête
            quest = Quest(
                id="quest_001",
                title="Defeat Goblins",
                description="Defeat 5 goblins in the forest",
                category="combat",
                estimated_duration=300,
                objectives=[
                    {"id": "obj_001", "description": "Kill goblin warrior", "required_count": 3, "current_count": 1},
                    {"id": "obj_002", "description": "Kill goblin archer", "required_count": 2, "current_count": 0}
                ],
                rewards={"experience": 500, "gold": 100}
            )

            # Test d'ajout de quête
            quest_tracker.add_quest(quest)

            # Test de récupération de quête
            retrieved_quest = quest_tracker.get_quest("quest_001")
            assert retrieved_quest is not None, "Quête non récupérée"
            assert retrieved_quest.title == "Defeat Goblins", "Titre incorrect"

            # Test de mise à jour de progression
            quest_tracker.update_quest_progress("quest_001", "obj_001", 2)

            # Test de détection automatique via OCR (simulé)
            mock_text = "Quest: Defeat Goblins - Progress: 2/5 Goblins defeated"
            with patch('pytesseract.image_to_string', return_value=mock_text):
                detected_quests = quest_tracker.detect_quests_from_screen()
                # Note: Ce test peut échouer selon la configuration OCR

            # Test de recommandations
            recommendations = quest_tracker.get_quest_recommendations({
                "player_level": 15,
                "current_area": "forest"
            })

            assert isinstance(recommendations, list), "Recommandations doivent être une liste"

            logger.info("✅ Quest Tracker test RÉUSSI")
            return True

        except Exception as e:
            logger.error(f"❌ Quest Tracker test ÉCHOUÉ: {e}")
            logger.error(traceback.format_exc())
            return False

    def test_integration(self) -> bool:
        """Test d'intégration end-to-end"""
        logger.info("Test d'intégration complète...")

        try:
            # Initialisation de tous les composants
            hrm_bot = HRMBot()
            adaptive_learner = AdaptiveLearner(player_id="integration_test")
            decision_maker = IntelligentDecisionMaker()
            quest_tracker = QuestTracker(db_path="G:/Botting/data/hrm/integration_test.db")

            # Simulation d'un cycle de jeu complet
            game_state = self.create_mock_game_state()

            # 1. Décision HRM de base
            base_decision = hrm_bot.decide_action(game_state)

            # 2. Enrichissement par le décideur intelligent
            with patch('PIL.ImageGrab.grab', return_value=Mock(size=(1920, 1080))):
                enhanced_decision = decision_maker.make_enhanced_decision(
                    base_decision.action,
                    base_decision.confidence,
                    {"current_quest": game_state.current_quest}
                )

            # 3. Ajout d'expérience à l'apprenant adaptatif
            experience = LearningExperience(
                state_data=game_state.__dict__,
                action_taken=enhanced_decision["final_action"],
                reward_received=10.0,
                outcome_success=True,
                context_info=enhanced_decision.get("context", {})
            )
            adaptive_learner.add_experience(experience)

            # 4. Mise à jour des quêtes
            if game_state.current_quest:
                # Simuler la progression de quête
                quest = Quest(
                    id="integration_quest",
                    title=game_state.current_quest,
                    description="Quest test d'intégration",
                    category="test"
                )
                quest_tracker.add_quest(quest)

            # 5. Apprentissage du résultat
            hrm_bot.learn_from_outcome(base_decision, outcome_success=True, reward=10.0)

            # Vérification que tous les composants fonctionnent ensemble
            assert base_decision.action is not None, "Décision de base manquante"
            assert enhanced_decision["final_action"] is not None, "Décision enrichie manquante"
            assert len(adaptive_learner.replay_buffer.experiences) > 0, "Expérience non ajoutée"

            logger.info("✅ Test d'intégration RÉUSSI")
            return True

        except Exception as e:
            logger.error(f"❌ Test d'intégration ÉCHOUÉ: {e}")
            logger.error(traceback.format_exc())
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Exécute tous les tests et retourne les résultats"""
        logger.info("🚀 Début des tests d'intégration HRM Intelligence")

        # Exécution des tests
        self.test_results["hrm_core"] = self.test_hrm_core()
        self.test_results["adaptive_learner"] = self.test_adaptive_learner()
        self.test_results["decision_maker"] = self.test_decision_maker()
        self.test_results["quest_tracker"] = self.test_quest_tracker()
        self.test_results["integration"] = self.test_integration()

        # Rapport final
        successful_tests = sum(self.test_results.values())
        total_tests = len(self.test_results)

        logger.info(f"\n{'='*50}")
        logger.info(f"RAPPORT FINAL DES TESTS")
        logger.info(f"{'='*50}")

        for test_name, result in self.test_results.items():
            status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
            logger.info(f"{test_name.upper()}: {status}")

        logger.info(f"\nRÉSULTAT GLOBAL: {successful_tests}/{total_tests} tests réussis")

        if successful_tests == total_tests:
            logger.info("🎉 TOUS LES TESTS ONT RÉUSSI ! Le système HRM Intelligence est opérationnel.")
        else:
            logger.warning(f"⚠️  {total_tests - successful_tests} test(s) ont échoué. Vérifiez les logs.")

        return self.test_results

    def generate_test_report(self) -> str:
        """Génère un rapport de test détaillé"""
        report = {
            "timestamp": time.time(),
            "test_results": self.test_results,
            "system_info": {
                "python_version": sys.version,
                "platform": os.name,
                "working_directory": os.getcwd()
            }
        }

        report_path = "G:/Botting/data/hrm/integration_test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"📄 Rapport de test sauvegardé: {report_path}")
        return report_path

def main():
    """Point d'entrée principal pour les tests"""
    try:
        tester = HRMIntegrationTester()
        results = tester.run_all_tests()
        report_path = tester.generate_test_report()

        # Code de sortie basé sur les résultats
        if all(results.values()):
            sys.exit(0)  # Succès
        else:
            sys.exit(1)  # Échec

    except Exception as e:
        logger.error(f"Erreur fatale lors des tests: {e}")
        logger.error(traceback.format_exc())
        sys.exit(2)

if __name__ == "__main__":
    main()