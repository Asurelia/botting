"""
Composants Principaux du Système RL DOFUS
=========================================

Classes et fonctions principales regroupées pour un système RL complet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
from dataclasses import dataclass
from collections import deque
from enum import IntEnum
import cv2
import logging

logger = logging.getLogger(__name__)

# Configuration principale
@dataclass
class RLSystemConfig:
    """Configuration globale du système RL"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "G:/Botting/models/rl_system"
    hidden_dim: int = 512
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_episodes: int = 1000
    
    # Vision
    use_vision_transformer: bool = True
    vision_backbone: str = "efficientnet_v2_s"
    
    # Reward System
    enable_adaptive_rewards: bool = True
    enable_curiosity: bool = True
    
    # Meta-learning
    enable_meta_learning: bool = True
    adaptation_strategy: str = "maml"

class ActionType(IntEnum):
    """Actions possibles dans DOFUS"""
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    ATTACK = 4
    USE_SPELL_1 = 5
    USE_SPELL_2 = 6
    USE_SPELL_3 = 7
    USE_ITEM = 8
    OPEN_INVENTORY = 9
    CLOSE_INTERFACE = 10
    WAIT = 11

class GameState(IntEnum):
    """États du jeu DOFUS"""
    UNKNOWN = 0
    IN_GAME = 1
    IN_COMBAT = 2
    DEAD = 3
    LOADING = 4

class VisionEncoder(nn.Module):
    """Encodeur vision optimisé pour DOFUS"""
    
    def __init__(self, config: RLSystemConfig):
        super().__init__()
        self.config = config
        
        # CNN backbone simplifié pour la démo
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, config.hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class PPOAgent(nn.Module):
    """Agent PPO simplifié"""
    
    def __init__(self, config: RLSystemConfig, num_actions: int):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(config)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.vision_encoder(observations)
        policy_logits = self.policy_head(features)
        values = self.value_head(features)
        return policy_logits, values
    
    def get_action(self, observation: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sélectionne une action"""
        policy_logits, value = self.forward(observation.unsqueeze(0))
        
        # Échantillonnage de l'action
        action_probs = F.softmax(policy_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()

class RewardCalculator:
    """Calculateur de récompenses adaptatif"""
    
    def __init__(self, config: RLSystemConfig):
        self.config = config
        self.episode_rewards = deque(maxlen=100)
        
        # Poids adaptatifs
        self.reward_weights = {
            'survival': 1.0,
            'progress': 0.8,
            'efficiency': 0.5,
            'exploration': 0.3
        }
    
    def calculate_reward(self, action: int, game_state: Dict, screenshot: np.ndarray) -> float:
        """Calcule la récompense multi-objectif"""
        reward = 0.0
        
        # Récompense de survie
        health = game_state.get('health', 0.5)
        if health > 0:
            reward += 0.1 * self.reward_weights['survival']
        else:
            reward -= 5.0  # Pénalité mort
        
        # Récompense de progression (XP, niveau, etc.)
        if game_state.get('xp_gained', 0) > 0:
            reward += game_state['xp_gained'] * 0.001 * self.reward_weights['progress']
        
        # Bonus efficacité (éviter actions répétitives)
        if hasattr(self, 'last_action') and self.last_action == action:
            reward -= 0.05 * self.reward_weights['efficiency']
        
        self.last_action = action
        self.episode_rewards.append(reward)
        
        return reward

class DofusEnvironment:
    """Environnement DOFUS simplifié"""
    
    def __init__(self, config: RLSystemConfig):
        self.config = config
        self.reward_calculator = RewardCalculator(config)
        
        # Espaces d'action et observation
        self.action_space = gym.spaces.Discrete(len(ActionType))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(3, 224, 224), 
            dtype=np.uint8
        )
        
        # État interne
        self.current_state = GameState.UNKNOWN
        self.step_count = 0
        self.episode_reward = 0.0
        
    def reset(self) -> Dict[str, Any]:
        """Reset l'environnement"""
        self.step_count = 0
        self.episode_reward = 0.0
        self.current_state = GameState.IN_GAME
        
        # Observation factice pour démo
        observation = {
            'image': np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            'game_state': int(self.current_state),
            'health': 1.0,
            'mana': 1.0
        }
        
        return observation
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Exécute une action"""
        self.step_count += 1
        
        # Simulation d'état de jeu
        game_state = {
            'health': max(0.0, 1.0 - self.step_count * 0.01),  # Santé diminue
            'mana': 1.0,
            'xp_gained': 1 if action == ActionType.ATTACK else 0
        }
        
        # Nouvelle observation
        observation = {
            'image': np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            'game_state': int(self.current_state),
            'health': game_state['health'],
            'mana': game_state['mana']
        }
        
        # Calcul de la récompense
        reward = self.reward_calculator.calculate_reward(
            action, game_state, observation['image']
        )
        self.episode_reward += reward
        
        # Conditions de fin
        done = (self.step_count >= 100 or game_state['health'] <= 0)
        
        # Informations supplémentaires
        info = {
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'game_state': self.current_state.name
        }
        
        return observation, reward, done, info

class RLTrainer:
    """Entraîneur RL principal"""
    
    def __init__(self, config: RLSystemConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Agent
        num_actions = len(ActionType)
        self.agent = PPOAgent(config, num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate)
        
        # Environnement
        self.env = DofusEnvironment(config)
        
        # Historiques
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': []
        }
    
    def train(self, num_episodes: int = None) -> Dict[str, List]:
        """Entraîne l'agent"""
        num_episodes = num_episodes or self.config.max_episodes
        
        logger.info(f"Début entraînement PPO ({num_episodes} épisodes)")
        
        for episode in range(num_episodes):
            episode_reward = 0.0
            episode_length = 0
            
            # Reset environnement
            obs = self.env.reset()
            done = False
            
            # Collecte d'un épisode
            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': []
            }
            
            while not done:
                # Conversion observation pour agent
                obs_tensor = torch.FloatTensor(obs['image']).to(self.device)
                
                # Action de l'agent
                action, log_prob, value = self.agent.get_action(obs_tensor)
                
                # Step environnement
                next_obs, reward, done, info = self.env.step(action)
                
                # Stockage
                episode_data['observations'].append(obs_tensor)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['values'].append(value)
                episode_data['log_probs'].append(log_prob)
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
            
            # Mise à jour agent (simplifiée)
            loss = self._update_agent(episode_data)
            
            # Historique
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['losses'].append(loss)
            
            # Log périodique
            if episode % 10 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                           f"Avg={avg_reward:.2f}, Length={episode_length}")
        
        logger.info("Entraînement terminé")
        return self.training_history
    
    def _update_agent(self, episode_data: Dict) -> float:
        """Met à jour l'agent (version simplifiée)"""
        # Conversion en tenseurs
        rewards = torch.FloatTensor(episode_data['rewards']).to(self.device)
        values = torch.stack(episode_data['values'])
        log_probs = torch.stack(episode_data['log_probs'])
        
        # Calcul des returns (simple)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Calcul des avantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Perte simplifiée
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        total_loss = policy_loss + 0.5 * value_loss
        
        # Mise à jour
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Évalue l'agent"""
        self.agent.eval()
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                obs_tensor = torch.FloatTensor(obs['image']).to(self.device)
                
                with torch.no_grad():
                    action, _, _ = self.agent.get_action(obs_tensor)
                
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        self.agent.train()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths)
        }

class RLSystem:
    """Système RL principal intégré"""
    
    def __init__(self, config: RLSystemConfig = None):
        self.config = config or RLSystemConfig()
        self.trainer = RLTrainer(self.config)
        
        logger.info(f"Système RL DOFUS initialisé sur {self.config.device}")
    
    def train(self, num_episodes: int = None) -> Dict[str, List]:
        """Interface d'entraînement simplifiée"""
        return self.trainer.train(num_episodes)
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Interface d'évaluation simplifiée"""
        return self.trainer.evaluate(num_episodes)
    
    def save_model(self, filepath: str):
        """Sauvegarde le modèle"""
        torch.save({
            'model_state_dict': self.trainer.agent.state_dict(),
            'config': self.config,
            'training_history': self.trainer.training_history
        }, filepath)
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath: str):
        """Charge un modèle"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        self.trainer.agent.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Modèle chargé: {filepath}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    config = RLSystemConfig(
        device="cpu",  # CPU pour démo
        max_episodes=50,
        learning_rate=1e-4
    )
    
    # Système RL
    rl_system = RLSystem(config)
    
    # Entraînement
    history = rl_system.train(20)
    print(f"Entraînement terminé: {len(history['episode_rewards'])} épisodes")
    
    # Évaluation
    eval_results = rl_system.evaluate(5)
    print(f"Évaluation: {eval_results}")
    
    # Sauvegarde
    rl_system.save_model("G:/Botting/models/rl_demo_model.pt")
    
    logger.info("Démonstration RL DOFUS terminée!")