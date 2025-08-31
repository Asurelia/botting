"""
Exemple d'Utilisation Complète du Système RL DOFUS
=================================================

Ce fichier démontre l'utilisation complète du système d'apprentissage
par renforcement avancé pour DOFUS, incluant tous les composants principaux.

Auteur: Système RL DOFUS
Version: 1.0.0
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from pathlib import Path

# Import du module RL
from . import (
    AdvancedRLAgent, RLConfig,
    DofusEnvironment, EnvironmentFactory, 
    RewardSystemFactory,
    BehaviorCloningTrainer, BehaviorCloningConfig,
    MetaLearningFactory, AdaptationStrategy,
    ExpertiseLevel, TaskType
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_rl_training():
    """Exemple d'entraînement RL basique avec PPO"""
    logger.info("=== Exemple: Entraînement RL Basique ===")
    
    # Configuration
    rl_config = RLConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_episodes=1000,
        ppo_lr=3e-4,
        save_frequency=100
    )
    
    # Création de l'environnement
    env = EnvironmentFactory.create_training_env(
        window_title="Dofus",
        target_resolution=(800, 600),
        fps_limit=10
    )
    
    # Création de l'agent RL
    agent = AdvancedRLAgent(rl_config, env.action_space, env.observation_space)
    agent.add_agent("ppo")
    
    # Entraînement
    logger.info("Début de l'entraînement PPO...")
    training_stats = agent.train_ppo(env, num_episodes=100)
    
    logger.info(f"Entraînement terminé: {training_stats}")
    
    # Évaluation
    eval_stats = agent.evaluate(env, num_episodes=10)
    logger.info(f"Évaluation: {eval_stats}")
    
    # Nettoyage
    env.close()
    agent.close()

def example_behavior_cloning():
    """Exemple d'apprentissage par imitation"""
    logger.info("=== Exemple: Behavior Cloning ===")
    
    # Configuration
    bc_config = BehaviorCloningConfig(
        data_dir="G:/Botting/data/demonstrations",
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=50
    )
    
    # Collecteur de démonstrations
    from .behavior_cloning import DemonstrationCollector
    collector = DemonstrationCollector(bc_config)
    
    # Simulation de collecte (en réalité, l'utilisateur jouerait)
    logger.info("Collecte de démonstrations simulée...")
    
    # Dans un vrai scénario:
    # collector.start_collection(ExpertiseLevel.INTERMEDIATE)
    # input("Jouez normalement et appuyez sur Entrée pour arrêter...")
    # collector.stop_collection()
    
    # Pour la démo, on génère des données fictives
    demonstrations = []
    for i in range(100):
        from .behavior_cloning import ActionDemonstration
        demo = ActionDemonstration()
        demo.screenshot = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        demo.action = np.random.randint(0, 16)
        demo.expertise_level = ExpertiseLevel.INTERMEDIATE
        demonstrations.append(demo)
    
    if demonstrations:
        # Entraînement
        trainer = BehaviorCloningTrainer(bc_config)
        train_loader, val_loader, test_loader = trainer.prepare_data(demonstrations)
        
        # Construction du modèle
        num_actions = 16
        from .behavior_cloning import BehaviorCloningDataset
        dataset = BehaviorCloningDataset(demonstrations, bc_config)
        class_weights = dataset.get_class_weights()
        trainer.build_model(num_actions, class_weights)
        
        # Entraînement
        logger.info("Entraînement behavior cloning...")
        history = trainer.train(train_loader, val_loader)
        
        # Test
        test_results = trainer.test(test_loader)
        logger.info(f"Résultats test: {test_results['overall_accuracy']:.2f}% précision")

def example_meta_learning():
    """Exemple de meta-learning pour adaptation rapide"""
    logger.info("=== Exemple: Meta-Learning ===")
    
    # Système MAML
    logger.info("Test du système MAML...")
    maml_system = MetaLearningFactory.create_maml_system(
        num_epochs=50,
        meta_batch_size=4,  # Réduit pour la démo
        maml_inner_steps=3
    )
    
    # Entraînement méta (version courte pour démo)
    logger.info("Entraînement méta MAML...")
    history = maml_system.train_meta(20)  # Version courte
    
    # Test d'adaptation à une nouvelle tâche
    logger.info("Test d'adaptation à une nouvelle tâche...")
    test_task = maml_system.task_generator.generate_task(TaskType.COMBAT_MONSTER)
    logger.info(f"Tâche générée: {test_task['task_type'].name} (difficulté: {test_task['difficulty_estimate']:.2f})")
    
    # Génération de données de support
    test_support = maml_system._generate_test_data(test_task, 5)
    
    # Adaptation du modèle
    adapted_model = maml_system.adapt_to_task(test_task, test_support, adaptation_steps=5)
    
    # Statistiques
    stats = maml_system.get_adaptation_statistics()
    logger.info(f"Statistiques d'adaptation: {stats}")
    
    # Test d'autres systèmes
    logger.info("Test du système prototypique...")
    proto_system = MetaLearningFactory.create_prototypical_system()
    
    logger.info("Test du système avec mémoire...")
    memory_system = MetaLearningFactory.create_memory_system()

def example_advanced_reward_system():
    """Exemple de système de récompenses avancé"""
    logger.info("=== Exemple: Système de Récompenses Avancé ===")
    
    # Création du système de récompenses
    reward_system = RewardSystemFactory.create_training_system(
        enable_adaptation=True,
        enable_curiosity=True,
        enable_shaping=True
    )
    
    # Simulation d'épisode avec récompenses
    logger.info("Simulation d'un épisode avec calcul de récompenses...")
    
    total_reward = 0.0
    for step in range(50):
        # Simulation d'une capture d'écran et action
        dummy_screenshot = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        dummy_action = np.random.randint(0, 16)
        dummy_game_info = {
            'health': max(0.1, np.random.uniform(0.3, 1.0)),
            'mana': np.random.uniform(0.2, 1.0),
            'entities': [{'type': 'monster', 'distance': np.random.uniform(10, 100)}] if np.random.random() > 0.7 else []
        }
        
        # Calcul de la récompense
        reward = reward_system.calculate_reward(
            action=dummy_action,
            current_state=None,
            screenshot=dummy_screenshot,
            game_info=dummy_game_info
        )
        
        total_reward += reward
        
        if step % 10 == 0:
            logger.info(f"Step {step}: Reward={reward:.3f}, Total={total_reward:.3f}")
    
    # Décomposition des récompenses
    breakdown = reward_system.get_reward_breakdown()
    logger.info(f"Décomposition finale des récompenses: {breakdown}")
    
    # Test avec curriculum
    logger.info("Test avec curriculum learning...")
    from .reward_system import CurriculumRewardManager
    curriculum_system = CurriculumRewardManager(reward_system)
    
    # Simulation de progression du curriculum
    for stage in range(3):
        performance = np.random.uniform(0.1, 0.9)
        curriculum_system.update_curriculum(performance)
        stage_info = curriculum_system.get_current_stage_info()
        logger.info(f"Curriculum stage {stage}: {stage_info}")

def example_complete_pipeline():
    """Exemple de pipeline complet: BC -> RL -> Meta-learning"""
    logger.info("=== Exemple: Pipeline Complet ===")
    
    # Phase 1: Behavior Cloning pour initialisation
    logger.info("Phase 1: Apprentissage par imitation...")
    
    # Configuration BC
    bc_config = BehaviorCloningConfig(
        data_dir="G:/Botting/data/demonstrations",
        batch_size=8,
        num_epochs=20
    )
    
    # Génération de données factices pour la démo
    demonstrations = []
    for i in range(50):
        from .behavior_cloning import ActionDemonstration
        demo = ActionDemonstration()
        demo.screenshot = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        demo.action = np.random.randint(0, 16)
        demonstrations.append(demo)
    
    # Entraînement BC
    if demonstrations:
        trainer = BehaviorCloningTrainer(bc_config)
        train_loader, val_loader, test_loader = trainer.prepare_data(demonstrations)
        
        from .behavior_cloning import BehaviorCloningDataset
        dataset = BehaviorCloningDataset(demonstrations, bc_config)
        trainer.build_model(16, dataset.get_class_weights())
        
        logger.info("Entraînement BC rapide...")
        trainer.train(train_loader, val_loader)
    
    # Phase 2: Fine-tuning avec RL
    logger.info("Phase 2: Fine-tuning avec apprentissage par renforcement...")
    
    rl_config = RLConfig(
        device="cpu",  # CPU pour la démo
        max_episodes=50,
        ppo_lr=1e-4
    )
    
    # Environnement simulé pour la démo
    class MockEnvironment:
        def __init__(self):
            self.action_space = type('ActionSpace', (), {'n': 16})()
            self.observation_space = type('ObservationSpace', (), {
                'shape': (3, 224, 224)
            })()
        
        def reset(self):
            return {'image': np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)}
        
        def step(self, action):
            obs = {'image': np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)}
            reward = np.random.uniform(-1, 1)
            done = np.random.random() < 0.05
            info = {}
            return obs, reward, done, info
        
        def close(self):
            pass
    
    mock_env = MockEnvironment()
    
    # Agent RL avec initialisation BC
    agent = AdvancedRLAgent(rl_config, mock_env.action_space, mock_env.observation_space)
    agent.add_agent("ppo")
    
    # Chargement des poids BC (simulation)
    logger.info("Chargement des poids BC...")
    
    # Entraînement RL court
    logger.info("Fine-tuning RL...")
    # agent.train_ppo(mock_env, num_episodes=20)
    
    # Phase 3: Meta-learning pour adaptation
    logger.info("Phase 3: Meta-learning pour adaptation rapide...")
    
    meta_system = MetaLearningFactory.create_maml_system(
        num_epochs=10,
        meta_batch_size=2
    )
    
    # Entraînement méta court
    logger.info("Entraînement méta rapide...")
    meta_system.train_meta(5)
    
    # Test d'adaptation
    test_task = meta_system.task_generator.generate_task(TaskType.RESOURCE_GATHERING)
    test_support = meta_system._generate_test_data(test_task, 3)
    adapted_model = meta_system.adapt_to_task(test_task, test_support)
    
    logger.info("Pipeline complet terminé avec succès!")
    
    mock_env.close()

def example_gpu_optimization():
    """Exemple d'optimisations GPU avancées"""
    logger.info("=== Exemple: Optimisations GPU ===")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA non disponible, exemple en mode CPU")
        return
    
    # Configuration GPU optimisée
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    
    # Configuration avec optimisations GPU
    rl_config = RLConfig(
        device="cuda",
        ppo_batch_size=128,  # Batch plus grand pour GPU
        ppo_lr=3e-4,
        hidden_dim=1024,    # Réseau plus grand
        use_transformer=True
    )
    
    logger.info(f"GPU sélectionné: {torch.cuda.get_device_name()}")
    logger.info(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test de performance GPU
    logger.info("Test de performance GPU...")
    
    # Simulation de données
    batch_size = 64
    dummy_images = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_actions = torch.randint(0, 16, (batch_size,), device=device)
    
    # Test de forward pass
    from .rl_agent import VisionEncoder
    vision_encoder = VisionEncoder(rl_config).to(device)
    
    # Mesure du temps
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            features = vision_encoder(dummy_images)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    logger.info(f"Temps moyen forward pass: {avg_time*1000:.2f}ms")
    logger.info(f"Throughput: {batch_size / avg_time:.0f} images/sec")
    
    # Mémoire GPU utilisée
    memory_used = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Mémoire GPU utilisée: {memory_used:.2f} GB")

def main():
    """Fonction principale démontrant tous les exemples"""
    logger.info("Début des exemples d'utilisation du système RL DOFUS")
    
    try:
        # Exemples basiques
        # example_basic_rl_training()
        
        # Behavior cloning
        example_behavior_cloning()
        
        # Meta-learning  
        example_meta_learning()
        
        # Système de récompenses
        example_advanced_reward_system()
        
        # Pipeline complet
        example_complete_pipeline()
        
        # Optimisations GPU
        example_gpu_optimization()
        
        logger.info("Tous les exemples terminés avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des exemples: {e}")
        raise
    
    logger.info("Exemples d'utilisation terminés")

if __name__ == "__main__":
    main()