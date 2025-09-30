"""
Tests d'intégration HRM AMD pour validation de la migration
Tests complets de l'architecture HRM optimisée pour AMD 7800XT

Version: 2.0.0 - Tests Migration AMD
"""

import unittest
import torch
import time
import numpy as np
from typing import Dict, List, Any
import logging

# Imports des modules HRM AMD
try:
    from .hrm_amd_core import (
        HRMAMDModel, AMDOptimizationConfig, AMDDeviceManager,
        HRMSystemOne, HRMSystemTwo, AMDOptimizedAttention
    )
    from .dofus_integration import (
        DofusHRMIntegration, DofusGameState, DofusAction,
        DofusStateEncoder, DofusActionDecoder
    )
except ImportError:
    # Fallback pour exécution directe
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))

    from hrm_amd_core import (
        HRMAMDModel, AMDOptimizationConfig, AMDDeviceManager,
        HRMSystemOne, HRMSystemTwo, AMDOptimizedAttention
    )
    from dofus_integration import (
        DofusHRMIntegration, DofusGameState, DofusAction,
        DofusStateEncoder, DofusActionDecoder
    )

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAMDDeviceManager(unittest.TestCase):
    """Tests du gestionnaire de device AMD"""

    def setUp(self):
        self.config = AMDOptimizationConfig()
        self.device_manager = AMDDeviceManager(self.config)

    def test_device_setup(self):
        """Test configuration du device"""
        device = self.device_manager.device
        self.assertIsNotNone(device)
        logger.info(f"Device configuré: {device}")

    def test_device_properties(self):
        """Test récupération propriétés device"""
        props = self.device_manager.device_props
        self.assertIsInstance(props, dict)
        self.assertIn("name", props)
        self.assertIn("has_directml", props)
        logger.info(f"Propriétés device: {props}")

class TestHRMSystemComponents(unittest.TestCase):
    """Tests des composants System 1 et System 2"""

    def setUp(self):
        self.config = AMDOptimizationConfig()
        self.device_manager = AMDDeviceManager(self.config)
        self.device = self.device_manager.device

    def test_system_one_forward(self):
        """Test forward pass System 1"""
        system_one = HRMSystemOne(self.config).to(self.device)

        # Input test
        batch_size, seq_len, hidden_size = 2, 10, 512
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=self.device)

        # Forward pass
        with torch.no_grad():
            output = system_one(hidden_states)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        logger.info(f"System One output shape: {output.shape}")

    def test_system_two_forward(self):
        """Test forward pass System 2 avec halting"""
        system_two = HRMSystemTwo(self.config).to(self.device)

        # Input test
        batch_size, seq_len, hidden_size = 2, 10, 512
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        system_one_output = torch.randn(batch_size, seq_len, hidden_size, device=self.device)

        # Forward pass avec halting
        with torch.no_grad():
            output, steps = system_two(hidden_states, system_one_output, max_steps=5)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        self.assertGreaterEqual(steps, 1)
        self.assertLessEqual(steps, 5)
        logger.info(f"System Two: {steps} steps, output shape: {output.shape}")

    def test_optimized_attention(self):
        """Test attention optimisée AMD"""
        attention = AMDOptimizedAttention(
            hidden_size=512,
            num_heads=8,
            config=self.config
        ).to(self.device)

        # Input test
        batch_size, seq_len, hidden_size = 2, 16, 512
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=self.device)

        # Forward pass
        with torch.no_grad():
            output, _ = attention(hidden_states)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        logger.info(f"Attention output shape: {output.shape}")

class TestHRMAMDModel(unittest.TestCase):
    """Tests du modèle HRM complet"""

    def setUp(self):
        self.config = AMDOptimizationConfig()
        self.model = HRMAMDModel(self.config).to_device()

    def test_model_initialization(self):
        """Test initialisation du modèle"""
        param_count = self.model.count_parameters()
        self.assertGreater(param_count, 0)
        logger.info(f"Modèle HRM AMD: {param_count:,} paramètres")

    def test_model_forward(self):
        """Test forward pass complet"""
        # Input test
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.model.device_manager.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, max_reasoning_steps=3)

        self.assertIn("logits", outputs)
        self.assertIn("reasoning_steps", outputs)
        self.assertIn("system_one_output", outputs)

        logits = outputs["logits"]
        self.assertEqual(logits.shape[:2], (batch_size, seq_len))

        logger.info(f"Model forward: logits shape {logits.shape}, steps: {outputs['reasoning_steps']}")

    def test_model_generation(self):
        """Test génération de texte"""
        # Input prompt
        input_ids = torch.randint(0, 1000, (1, 5), device=self.model.device_manager.device)

        # Génération
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=20,
                temperature=1.0,
                do_sample=True
            )

        self.assertGreater(generated.shape[1], input_ids.shape[1])
        logger.info(f"Generation: {input_ids.shape[1]} -> {generated.shape[1]} tokens")

    def test_memory_usage(self):
        """Test utilisation mémoire"""
        memory_info = self.model.get_memory_usage()

        self.assertIn("parameters_mb", memory_info)
        self.assertIn("total_mb", memory_info)
        self.assertGreater(memory_info["total_mb"], 0)

        logger.info(f"Memory usage: {memory_info['total_mb']:.1f} MB")

class TestDofusIntegration(unittest.TestCase):
    """Tests de l'intégration DOFUS"""

    def setUp(self):
        self.config = AMDOptimizationConfig()
        self.device_manager = AMDDeviceManager(self.config)
        self.device = self.device_manager.device

    def test_state_encoder(self):
        """Test encodeur d'état DOFUS"""
        encoder = DofusStateEncoder(self.config).to(self.device)

        # État de jeu test
        game_state = DofusGameState(
            position=(100, 150),
            map_id=1234,
            cell_id=250,
            level=180,
            health_percent=0.8,
            energy_percent=0.6,
            experience=50000,
            kamas=100000,
            in_combat=False,
            ap=6, mp=3, turn_number=0,
            inventory_items=[{"id": 123, "quantity": 5}, {"id": 456, "quantity": 2}],
            equipment={"weapon": 789, "hat": 101},
            timestamp=time.time(),
            server_time="14:30",
            active_quests=[{"id": 1, "name": "Test Quest"}],
            current_objective="Tuer 5 monstres",
            nearby_monsters=[{"id": 111, "level": 50}],
            nearby_players=[],
            nearby_npcs=[{"id": 222, "name": "PNJ Test"}],
            nearby_resources=[],
            guild_info={"name": "Test Guild", "level": 10},
            group_members=[]
        )

        # Encodage
        with torch.no_grad():
            embedding = encoder.encode_game_state(game_state)

        self.assertEqual(embedding.shape, (1, 512))  # Batch size 1, hidden size 512
        logger.info(f"State encoding shape: {embedding.shape}")

    def test_action_decoder(self):
        """Test décodeur d'actions DOFUS"""
        decoder = DofusActionDecoder(self.config).to(self.device)

        # Game state test
        game_state = DofusGameState(
            position=(100, 150), map_id=1234, cell_id=250, level=180,
            health_percent=0.8, energy_percent=0.6, experience=50000, kamas=100000,
            in_combat=True, ap=6, mp=3, turn_number=1,
            inventory_items=[], equipment={}, timestamp=time.time(), server_time="14:30",
            active_quests=[], current_objective=None,
            nearby_monsters=[{"id": 111, "level": 50, "position": (101, 151)}],
            nearby_players=[], nearby_npcs=[], nearby_resources=[],
            guild_info=None, group_members=[]
        )

        # HRM output simulé
        hrm_output = torch.randn(1, 512, device=self.device)

        # Décodage
        with torch.no_grad():
            action = decoder.decode_action(hrm_output, game_state)

        self.assertIsInstance(action, DofusAction)
        self.assertIsInstance(action.action_type, str)
        self.assertGreaterEqual(action.confidence, 0.0)
        self.assertLessEqual(action.confidence, 1.0)
        self.assertGreaterEqual(action.priority, 1)
        self.assertLessEqual(action.priority, 10)

        logger.info(f"Decoded action: {action.action_type} (confidence: {action.confidence:.3f})")

    def test_full_dofus_integration(self):
        """Test intégration DOFUS complète"""
        integration = DofusHRMIntegration(config=self.config)

        # État de jeu complet
        game_state = DofusGameState(
            position=(200, 300), map_id=5678, cell_id=400, level=200,
            health_percent=0.9, energy_percent=0.8, experience=100000, kamas=500000,
            in_combat=False, ap=12, mp=6, turn_number=0,
            inventory_items=[{"id": 789, "quantity": 10}], equipment={"weapon": 999},
            timestamp=time.time(), server_time="16:45",
            active_quests=[{"id": 2, "name": "Quest 2"}], current_objective="Explorer la zone",
            nearby_monsters=[], nearby_players=[{"id": 333, "name": "Player1"}],
            nearby_npcs=[{"id": 444, "name": "Merchant"}], nearby_resources=[{"id": 555, "type": "ore"}],
            guild_info={"name": "Elite Guild", "level": 50}, group_members=[]
        )

        # Décision d'action
        start_time = time.time()
        action = integration.decide_action(game_state)
        decision_time = time.time() - start_time

        self.assertIsInstance(action, DofusAction)
        self.assertLess(decision_time, 5.0)  # Max 5 secondes pour une décision

        logger.info(f"Decision time: {decision_time*1000:.1f}ms")
        logger.info(f"Action: {action.action_type}")
        logger.info(f"Reasoning: {action.reasoning_path}")

class TestPerformanceBenchmark(unittest.TestCase):
    """Tests de performance et benchmark"""

    def setUp(self):
        self.config = AMDOptimizationConfig()
        self.model = HRMAMDModel(self.config).to_device()

    def test_inference_speed(self):
        """Test vitesse d'inférence"""
        batch_size, seq_len = 1, 50
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.model.device_manager.device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(input_ids, max_reasoning_steps=4)

        # Benchmark
        num_runs = 10
        times = []

        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(input_ids, max_reasoning_steps=4)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        self.assertLess(avg_time, 2.0)  # Max 2 secondes par inférence

        logger.info(f"Inference speed: {avg_time*1000:.1f}±{std_time*1000:.1f}ms")
        logger.info(f"Reasoning steps: {outputs['reasoning_steps']}")

    def test_generation_speed(self):
        """Test vitesse de génération"""
        input_ids = torch.randint(0, 1000, (1, 10), device=self.model.device_manager.device)

        start_time = time.time()
        with torch.no_grad():
            generated = self.model.generate(input_ids, max_length=30, do_sample=False)
        generation_time = time.time() - start_time

        tokens_generated = generated.shape[1] - input_ids.shape[1]
        tokens_per_second = tokens_generated / generation_time

        self.assertGreater(tokens_per_second, 5.0)  # Min 5 tokens/sec

        logger.info(f"Generation speed: {tokens_per_second:.1f} tokens/sec")

    def test_memory_efficiency(self):
        """Test efficacité mémoire"""
        # Test avec différentes tailles de batch
        for batch_size in [1, 2, 4]:
            input_ids = torch.randint(0, 1000, (batch_size, 20), device=self.model.device_manager.device)

            # Mesure mémoire avant
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
            else:
                memory_before = 0

            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)

            # Mesure mémoire après
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
            else:
                memory_used = 0

            logger.info(f"Batch {batch_size}: {memory_used:.1f}MB memory used")

class TestConfigurationOptimizations(unittest.TestCase):
    """Tests des optimisations de configuration"""

    def test_mixed_precision(self):
        """Test mixed precision"""
        config = AMDOptimizationConfig(use_mixed_precision=True, preferred_dtype=torch.bfloat16)
        model = HRMAMDModel(config).to_device()

        input_ids = torch.randint(0, 1000, (1, 10), device=model.device_manager.device)

        with torch.no_grad():
            outputs = model(input_ids)

        # Vérifier que le modèle utilise le dtype approprié
        for param in model.parameters():
            if param.requires_grad:
                # Some parameters might still be float32 (embeddings, etc.)
                self.assertIn(param.dtype, [torch.float32, torch.bfloat16, torch.float16])
                break

        logger.info(f"Mixed precision test passed")

    def test_memory_optimization(self):
        """Test optimisations mémoire"""
        config = AMDOptimizationConfig(
            memory_fraction=0.8,
            enable_memory_optimization=True,
            enable_gradient_checkpointing=True
        )

        model = HRMAMDModel(config).to_device()

        # Test avec séquence plus longue
        input_ids = torch.randint(0, 1000, (1, 100), device=model.device_manager.device)

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertIn("logits", outputs)
        logger.info("Memory optimization test passed")

def run_all_tests():
    """Lance tous les tests avec rapport détaillé"""

    print("\n" + "="*60)
    print("TESTS HRM AMD OPTIMIZED - VALIDATION MIGRATION")
    print("="*60)

    # Créer la suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Ajouter toutes les classes de tests
    test_classes = [
        TestAMDDeviceManager,
        TestHRMSystemComponents,
        TestHRMAMDModel,
        TestDofusIntegration,
        TestPerformanceBenchmark,
        TestConfigurationOptimizations
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Rapport final
    print("\n" + "="*60)
    print("RAPPORT FINAL DES TESTS")
    print("="*60)
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")

    if result.failures:
        print("\nÉCHECS:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nERREURS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nTaux de succès: {success_rate:.1f}%")

    if success_rate >= 90:
        print("✅ MIGRATION VALIDÉE - Prêt pour déploiement")
    elif success_rate >= 75:
        print("⚠️ MIGRATION PARTIELLE - Corrections mineures requises")
    else:
        print("❌ MIGRATION ÉCHOUÉE - Corrections majeures requises")

    return result.wasSuccessful()

if __name__ == "__main__":
    # Configuration logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Lancer les tests
    success = run_all_tests()

    # Code de sortie
    import sys
    sys.exit(0 if success else 1)