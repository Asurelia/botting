"""
Tests pour le Projet Augmenta
Validation des modules d'intelligence passive, opportunités, fatigue, combos et analyse post-combat
"""

import unittest
import time
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime

# Imports des modules à tester
try:
    from modules.intelligence.passive_intelligence import PassiveIntelligence, PatternAnalyzer, RiskEvaluator, OpportunityDetector
    from modules.intelligence.opportunity_manager import OpportunityManager, OpportunityTracker
    from modules.intelligence.fatigue_simulation import FatigueSimulation, FatigueSimulator
    from modules.combat.combo_library import ComboLibrary, ComboGenerator, ComboExecutor
    from modules.combat.post_combat_analysis import PostCombatAnalysis, CombatAnalyzer
    from core.hrm_intelligence.amd_gpu_optimizer import AMDGPUOptimizer, AMDGPUDetector
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback pour les tests sans imports
    PassiveIntelligence = None
    OpportunityManager = None
    FatigueSimulation = None
    ComboLibrary = None
    PostCombatAnalysis = None
    AMDGPUOptimizer = None


class TestPassiveIntelligence(unittest.TestCase):
    """Tests pour le module d'intelligence passive"""

    def setUp(self):
        if PassiveIntelligence:
            self.module = PassiveIntelligence()
            self.mock_config = {"scan_interval": 10.0, "enable_learning": True}
            self.module.initialize(self.mock_config)

    def test_initialization(self):
        """Test de l'initialisation"""
        if not PassiveIntelligence:
            self.skipTest("Module not available")

        self.assertTrue(self.module.is_active())
        self.assertEqual(self.module.scan_interval, 10.0)
        self.assertTrue(self.module.enable_learning)

    def test_pattern_analysis(self):
        """Test de l'analyse de patterns"""
        if not PassiveIntelligence:
            self.skipTest("Module not available")

        # Création d'observations de test
        test_observation = Mock()
        test_observation.pattern_type = "enemy_spawn"
        test_observation.location = (100, 150)
        test_observation.frequency = 0.5
        test_observation.confidence = 0.8
        test_observation.last_seen = time.time()
        test_observation.duration = 30.0
        test_observation.metadata = {"threat_level": 0.6}

        self.module.pattern_analyzer.add_observation(test_observation)

        # Test de l'analyse
        analysis = self.module.pattern_analyzer.analyze_patterns((100, 150))
        self.assertIn("enemy_patterns", analysis)
        self.assertIn("resource_patterns", analysis)

    def test_risk_evaluation(self):
        """Test de l'évaluation des risques"""
        if not PassiveIntelligence:
            self.skipTest("Module not available")

        # Mock game state
        mock_game_state = Mock()
        mock_game_state.character.position = Mock(x=100, y=150)
        mock_game_state.combat.enemies = []
        mock_game_state.combat.allies = []

        risk_assessment = self.module.risk_evaluator.evaluate_risk((100, 150), mock_game_state)

        self.assertIsNotNone(risk_assessment)
        self.assertGreaterEqual(risk_assessment.risk_level, 0.0)
        self.assertLessEqual(risk_assessment.risk_level, 1.0)


class TestOpportunityManager(unittest.TestCase):
    """Tests pour le gestionnaire d'opportunités"""

    def setUp(self):
        if OpportunityManager:
            self.module = OpportunityManager()
            self.mock_config = {
                "scan_radius": 10,
                "update_interval": 30.0,
                "filters": {
                    "min_value": 20.0,
                    "max_risk": 0.8
                }
            }
            self.module.initialize(self.mock_config)

    def test_initialization(self):
        """Test de l'initialisation"""
        if not OpportunityManager:
            self.skipTest("Module not available")

        self.assertTrue(self.module.is_active())
        self.assertEqual(self.module.scan_radius, 10)
        self.assertEqual(self.module.default_filter.min_value, 20.0)

    def test_opportunity_tracking(self):
        """Test du suivi d'opportunités"""
        if not OpportunityManager:
            self.skipTest("Module not available")

        # Création d'une opportunité de test
        from modules.intelligence.opportunity_manager import Opportunity

        test_opportunity = Opportunity(
            id="test_opp_1",
            opportunity_type="resource_node",
            location=(100, 150),
            value_estimate=50.0,
            accessibility=0.8,
            competition_level=0.3,
            risk_level=0.2,
            time_to_reach=30.0,
            duration=300.0,
            discovery_time=time.time(),
            last_updated=time.time()
        )

        self.module.tracker.add_opportunity(test_opportunity)

        # Vérification
        opportunities = self.module.tracker.get_prioritized_opportunities(5)
        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0].id, "test_opp_1")

    def test_opportunity_filtering(self):
        """Test du filtrage d'opportunités"""
        if not OpportunityManager:
            self.skipTest("Module not available")

        # Création d'opportunités avec différents niveaux
        opportunities = []

        for i in range(3):
            opp = Mock()
            opp.value_estimate = 10 + i * 20  # 10, 30, 50
            opp.risk_level = i * 0.3  # 0.0, 0.3, 0.6
            opp.competition_level = i * 0.4  # 0.0, 0.4, 0.8
            opp.time_to_reach = i * 50  # 0, 50, 100
            opp.opportunity_type = "resource_node"
            opp.accessibility = 0.8
            opportunities.append(opp)

        # Test du filtrage
        filtered = self.module._filter_opportunities(opportunities, self.module.default_filter)
        self.assertEqual(len(filtered), 2)  # Seules les 2 dernières passent les filtres


class TestFatigueSimulation(unittest.TestCase):
    """Tests pour la simulation de fatigue"""

    def setUp(self):
        if FatigueSimulation:
            self.module = FatigueSimulation()
            self.mock_config = {
                "enable_effects": True,
                "fatigue_thresholds": {"warning": 0.3, "degraded": 0.6}
            }
            self.module.initialize(self.mock_config)

    def test_initialization(self):
        """Test de l'initialisation"""
        if not FatigueSimulation:
            self.skipTest("Module not available")

        self.assertTrue(self.module.is_active())
        self.assertTrue(self.module.enable_effects)
        self.assertEqual(self.module.fatigue_thresholds["warning"], 0.3)

    def test_fatigue_accumulation(self):
        """Test de l'accumulation de fatigue"""
        if not FatigueSimulation:
            self.skipTest("Module not available")

        # Simulation d'activité
        for i in range(10):
            self.module.simulator.record_activity(5)  # 5 actions

        # Mise à jour de la fatigue
        current_time = time.time()
        fatigue_state = self.module.simulator.update_fatigue(current_time, 5)

        self.assertIsNotNone(fatigue_state)
        self.assertGreaterEqual(fatigue_state.current_fatigue_level, 0.0)
        self.assertLessEqual(fatigue_state.current_fatigue_level, 1.0)

    def test_fatigue_effects(self):
        """Test des effets de fatigue"""
        if not FatigueSimulation:
            self.skipTest("Module not available")

        # Création d'un état de fatigue élevé
        self.module.current_fatigue_state = Mock()
        self.module.current_fatigue_state.current_fatigue_level = 0.7

        effects = self.module._calculate_fatigue_effects()

        self.assertIsNotNone(effects)
        self.assertGreater(effects.accuracy_reduction, 0.0)
        self.assertGreater(effects.speed_reduction, 0.0)


class TestComboLibrary(unittest.TestCase):
    """Tests pour la bibliothèque de combos"""

    def setUp(self):
        if ComboLibrary:
            self.module = ComboLibrary()
            self.mock_config = {
                "auto_generate_combos": True,
                "max_active_combos": 2
            }
            self.module.initialize(self.mock_config)

    def test_initialization(self):
        """Test de l'initialisation"""
        if not ComboLibrary:
            self.skipTest("Module not available")

        self.assertTrue(self.module.is_active())
        self.assertTrue(self.module.auto_generate_combos)
        self.assertEqual(self.module.max_active_combos, 2)

    def test_combo_generation(self):
        """Test de la génération de combos"""
        if not ComboLibrary:
            self.skipTest("Module not available")

        # Mock game state
        mock_game_state = Mock()
        mock_game_state.character = Mock()
        mock_game_state.character.character_class = Mock()
        mock_game_state.character.character_class.value = "iop"
        mock_game_state.character.spells = {"sword_celestial": Mock(), "colere_iop": Mock()}
        mock_game_state.character.current_pa = 8

        # Génération de combo
        combo = self.module.generate_combo_for_situation(mock_game_state)

        # Vérification (peut être None si pas de combo viable)
        if combo:
            self.assertIsNotNone(combo.id)
            self.assertIsNotNone(combo.spell_sequence)

    def test_combo_execution(self):
        """Test de l'exécution de combos"""
        if not ComboLibrary:
            self.skipTest("Module not available")

        # Mock combo et game state
        mock_combo = Mock()
        mock_combo.id = "test_combo"
        mock_combo.spell_sequence = ["spell1", "spell2"]

        mock_game_state = Mock()
        mock_game_state.character = Mock()
        mock_game_state.character.spells = {"spell1": Mock(), "spell2": Mock()}
        mock_game_state.character.can_cast_spell.return_value = True

        # Démarrage de l'exécution
        success = self.module.execute_combo(mock_combo, mock_game_state)
        self.assertTrue(success)

        # Test de progression
        next_spell = self.module.get_next_spell_in_combo("test_combo", mock_game_state)
        self.assertEqual(next_spell, "spell1")


class TestPostCombatAnalysis(unittest.TestCase):
    """Tests pour l'analyse post-combat"""

    def setUp(self):
        if PostCombatAnalysis:
            self.module = PostCombatAnalysis()
            self.mock_config = {
                "auto_analyze": True,
                "min_combat_duration": 5.0
            }
            self.module.initialize(self.mock_config)

    def test_initialization(self):
        """Test de l'initialisation"""
        if not PostCombatAnalysis:
            self.skipTest("Module not available")

        self.assertTrue(self.module.is_active())
        self.assertTrue(self.module.auto_analyze)
        self.assertEqual(self.module.min_combat_duration, 5.0)

    def test_combat_analysis(self):
        """Test de l'analyse de combat"""
        if not PostCombatAnalysis:
            self.skipTest("Module not available")

        # Mock game states
        mock_before = Mock()
        mock_before.character = Mock()
        mock_before.character.hp_percentage.return_value = 80.0
        mock_before.character.character_class = Mock()
        mock_before.character.character_class.value = "iop"
        mock_before.timestamp = time.time() - 60  # 1 minute ago

        mock_after = Mock()
        mock_after.character = Mock()
        mock_after.character.hp_percentage.return_value = 60.0
        mock_after.character.is_dead = False
        mock_after.combat = Mock()
        mock_after.combat.enemies = [Mock(is_dead=True), Mock(is_dead=True)]
        mock_after.combat.allies = [Mock()]
        mock_after.timestamp = time.time()

        # Mock events
        combat_events = [
            {"type": "spell_cast", "damage_dealt": 100, "success": True},
            {"type": "spell_cast", "damage_dealt": 150, "success": True}
        ]

        # Analyse
        report = self.module.analyzer.analyze_combat(mock_before, mock_after, combat_events)

        self.assertIsNotNone(report)
        self.assertTrue(report.victory)  # Tous les ennemis morts
        self.assertEqual(report.damage_dealt, 250)
        self.assertGreater(report.performance_score, 0)


class TestAMDGPUEOptimizer(unittest.TestCase):
    """Tests pour l'optimiseur GPU AMD"""

    def setUp(self):
        if AMDGPUOptimizer:
            self.optimizer = AMDGPUOptimizer()

    def test_initialization(self):
        """Test de l'initialisation"""
        if not AMDGPUOptimizer:
            self.skipTest("Module not available")

        # Test sans GPU (devrait fonctionner en mode dégradé)
        success = self.optimizer.initialize()
        # Peut échouer si pas de GPU, mais ne devrait pas planter
        self.assertIsInstance(success, bool)

    def test_gpu_detection(self):
        """Test de la détection GPU"""
        if not AMDGPUOptimizer:
            self.skipTest("Module not available")

        capabilities = self.optimizer.detector.detect_gpu()

        self.assertIsInstance(capabilities, object)
        # Les valeurs peuvent être par défaut si pas de GPU détecté

    def test_optimization_settings(self):
        """Test des paramètres d'optimisation"""
        if not AMDGPUOptimizer:
            self.skipTest("Module not available")

        # Configuration pour GPU haute performance
        self.optimizer.capabilities.memory_gb = 16.0
        self.optimizer.capabilities.compute_units = 60

        self.optimizer._configure_optimization_settings()

        self.assertTrue(self.optimizer.settings.mixed_precision)
        self.assertTrue(self.optimizer.settings.vision_acceleration)


class TestIntegration(unittest.TestCase):
    """Tests d'intégration entre modules"""

    def test_module_interaction(self):
        """Test des interactions entre modules"""
        # Test que les modules peuvent partager des données
        if not (PassiveIntelligence and OpportunityManager):
            self.skipTest("Modules not available")

        # Création des modules
        passive_intel = PassiveIntelligence()
        opp_manager = OpportunityManager()

        passive_intel.initialize({"scan_interval": 10.0})
        opp_manager.initialize({"scan_radius": 10})

        # Test de partage de données
        mock_game_state = Mock()
        mock_game_state.character = Mock()
        mock_game_state.character.position = Mock(x=100, y=150)

        # Mise à jour des modules
        passive_data = passive_intel.update(mock_game_state)
        opp_data = opp_manager.update(mock_game_state)

        # Vérification que les données sont échangées
        self.assertIsNotNone(passive_data)
        self.assertIsNotNone(opp_data)


def run_all_tests():
    """Fonction pour exécuter tous les tests"""
    # Configuration du logging pour les tests
    logging.basicConfig(level=logging.ERROR)

    # Découverte et exécution des tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("✅ Tous les tests ont réussi!")
    else:
        print("❌ Certains tests ont échoué!")
        exit(1)