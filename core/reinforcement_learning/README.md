# Système d'Apprentissage par Renforcement Avancé - DOFUS

## Vue d'ensemble

Ce module implémente un système d'apprentissage par renforcement state-of-the-art spécialement conçu pour l'automatisation intelligente du jeu DOFUS. Il combine plusieurs techniques avancées d'IA pour créer un agent adaptatif capable d'apprendre et de s'optimiser en temps réel.

## Fonctionnalités Principales

### 🧠 Algorithmes RL State-of-the-Art
- **PPO (Proximal Policy Optimization)** avec optimisations avancées
- **DQN (Deep Q-Network)** avec améliorations Rainbow
- **A3C (Asynchronous Actor-Critic)** pour entraînement distribué
- Vision Transformer intégrée pour reconnaissance d'écran
- Support multi-GPU et distributed training

### 🎯 Système de Récompenses Intelligent
- Récompenses multi-objectifs (XP, Kamas, Survie, Efficacité)
- Adaptation dynamique des pondérations
- Reward shaping pour guide l'apprentissage
- Détection automatique des objectifs du joueur
- Système de curriculum learning progressif

### 👨‍🏫 Apprentissage par Imitation (Behavior Cloning)
- Collecte automatique des démonstrations utilisateur
- Preprocessing avancé des données d'entrée
- Architecture de réseau optimisée pour l'imitation
- Techniques d'augmentation des données spécialisées
- Interface pour supervision humaine

### 🚀 Meta-Learning pour Adaptation Rapide
- Model-Agnostic Meta-Learning (MAML)
- Prototypical Networks pour few-shot learning
- Memory-Augmented Neural Networks
- Task-Conditional Networks adaptatifs
- Adaptation rapide à de nouveaux contextes DOFUS

### 🖼️ Vision Avancée
- CNN + Vision Transformer hybride
- Détection d'état du jeu automatique
- Reconnaissance d'entités et d'objets
- Support multi-résolution et multi-fenêtre
- Optimisations GPU pour traitement en temps réel

## Architecture du Système

```
core/reinforcement_learning/
├── __init__.py                 # Module principal
├── core_components.py          # Composants principaux
├── rl_agent.py                # Agent RL avec PPO/DQN/A3C
├── environment_wrapper.py     # Environnement DOFUS compatible Gym
├── reward_system.py           # Système de récompenses adaptatif
├── behavior_cloning.py        # Apprentissage par imitation
├── meta_learning.py           # Meta-learning pour adaptation
├── example_usage.py           # Exemples d'utilisation
└── README.md                  # Cette documentation
```

## Installation et Dépendances

### Prérequis
- Python 3.8+
- CUDA 11.0+ (optionnel, pour GPU)
- DOFUS installé et accessible

### Dépendances Python
```bash
pip install torch torchvision torchaudio
pip install gymnasium
pip install numpy opencv-python pillow
pip install matplotlib seaborn
pip install h5py albumentations
pip install ultralytics  # Pour YOLO
pip install pytesseract   # Pour OCR
pip install keyboard mouse  # Pour capture d'entrées
pip install psutil win32gui win32ui  # Windows uniquement
pip install mss  # Pour capture d'écran rapide
```

## Utilisation Rapide

### 1. Entraînement RL Basique

```python
from core.reinforcement_learning import RLSystem, RLSystemConfig

# Configuration
config = RLSystemConfig(
    device="cuda",
    max_episodes=1000,
    learning_rate=3e-4
)

# Système RL
rl_system = RLSystem(config)

# Entraînement
history = rl_system.train(500)

# Évaluation
results = rl_system.evaluate(20)
print(f"Performance: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
```

### 2. Apprentissage par Imitation

```python
from core.reinforcement_learning.behavior_cloning import (
    DemonstrationCollector, BehaviorCloningTrainer, BehaviorCloningConfig
)

# Configuration
bc_config = BehaviorCloningConfig(batch_size=32, num_epochs=100)

# Collecte de démonstrations
collector = DemonstrationCollector(bc_config)
collector.start_collection()
# ... jouer normalement ...
collector.stop_collection()

# Entraînement
trainer = BehaviorCloningTrainer(bc_config)
# ... (voir example_usage.py pour détails complets) ...
```

### 3. Meta-Learning

```python
from core.reinforcement_learning.meta_learning import MetaLearningFactory

# Création d'un système MAML
meta_system = MetaLearningFactory.create_maml_system(
    num_epochs=200,
    meta_batch_size=16
)

# Entraînement méta
history = meta_system.train_meta(100)

# Adaptation à une nouvelle tâche
task = meta_system.task_generator.generate_task(TaskType.COMBAT_MONSTER)
adapted_model = meta_system.adapt_to_task(task, support_data)
```

## Configuration Avancée

### Optimisations GPU

```python
config = RLSystemConfig(
    device="cuda",
    batch_size=128,        # Batch plus grand pour GPU
    hidden_dim=1024,       # Réseau plus complexe
    use_vision_transformer=True,
    enable_meta_learning=True
)
```

### Réglage du Système de Récompenses

```python
from core.reinforcement_learning.reward_system import RewardConfig

reward_config = RewardConfig(
    enable_adaptation=True,
    enable_curiosity=True,
    objective_weights={
        'survival': 1.0,
        'xp_gain': 0.8,
        'efficiency': 0.6
    }
)
```

## Métriques et Monitoring

Le système fournit des métriques détaillées :
- Récompenses par épisode et moyennes mobiles
- Précision du behavior cloning
- Scores d'adaptation du meta-learning
- Utilisation GPU et performance
- Analyse des patterns d'action
- Décomposition des récompenses par objectif

## Exemples d'Utilisation

Voir `example_usage.py` pour des exemples complets incluant :
- Pipeline complet BC → RL → Meta-learning
- Optimisations GPU avancées
- Système de récompenses adaptatif
- Intégration avec l'environnement DOFUS

## Optimisations Performance

### GPU
- Utilisation de mixed precision training
- Batch processing optimisé
- Memory pooling pour éviter les allocations
- Distributed training multi-GPU

### CPU
- Preprocessing asynchrone des images
- Cache intelligent des captures d'écran
- Threading pour I/O non-bloquant

### Mémoire
- Replay buffers avec compression
- Garbage collection optimisé
- Memory mapping pour gros datasets

## Architecture Technique

### Agent RL
- Encoder vision CNN + Vision Transformer
- Policy et value heads séparés
- Techniques de régularisation avancées
- Curriculum learning intégré

### Environnement
- Interface OpenAI Gym compatible
- Capture d'écran multi-méthodes (MSS, Win32, PIL)
- Détection d'état par vision + OCR + templates
- Actions discrètes et continues supportées

### Système de Récompenses
- Multi-objectifs avec pondération adaptative
- Reward shaping basé sur les heuristiques DOFUS
- Curiosité intrinsèque pour exploration
- Métriques de performance en temps réel

## Troubleshooting

### Problèmes Courants

1. **Fenêtre DOFUS non détectée**
   - Vérifier le titre exact de la fenêtre
   - S'assurer que DOFUS est en mode fenêtré
   - Ajuster les paramètres de capture dans EnvironmentConfig

2. **Performance GPU faible**
   - Vérifier les drivers CUDA
   - Augmenter la taille de batch
   - Activer cudnn.benchmark = True

3. **Récompenses instables**
   - Ajuster les poids du système de récompenses
   - Activer la normalisation des récompenses
   - Utiliser un learning rate plus faible

### Débogage

```python
# Activer le logging détaillé
import logging
logging.basicConfig(level=logging.DEBUG)

# Sauvegarder les screenshots pour analyse
env_config.save_screenshots = True
env_config.screenshot_dir = "debug/screenshots"

# Activer les métriques detaillées
config.log_frequency = 1
```

## Contribution

Pour contribuer au système :
1. Créer une branche feature
2. Implémenter les améliorations
3. Ajouter des tests
4. Mettre à jour la documentation
5. Créer une pull request

## Licence

Ce système est développé pour l'usage éducatif et de recherche. 
Respecter les conditions d'utilisation de DOFUS.

## Contact

Pour questions et support :
- Issues GitHub pour bugs et feature requests
- Documentation technique dans le code
- Exemples d'usage dans example_usage.py