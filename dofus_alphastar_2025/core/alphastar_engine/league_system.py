"""
League System - Système de league multi-agent pour AlphaStar DOFUS
Inspired by DeepMind AlphaStar League Training
"""

import torch
import numpy as np
import random
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import threading

from ..networks import AlphaStarModel, AlphaStarHRMModel
from ..hrm_reasoning import DofusHRMAgent
from config import config

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types d'agents dans la league"""
    MAIN_AGENT = "main_agent"
    EXPLOITER = "exploiter"
    LEAGUE_EXPLOITER = "league_exploiter"
    MAIN_EXPLOITER = "main_exploiter"

@dataclass
class AgentStats:
    """Statistiques d'un agent"""
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo_rating: float = 1200.0
    win_rate: float = 0.0
    avg_game_length: float = 0.0
    total_reward: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def update_from_game(self, won: bool, game_length: float, reward: float):
        """Met à jour les stats après un match"""
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1

        self.total_reward += reward
        self.win_rate = self.wins / max(self.games_played, 1)

        # Moyenne mobile de la longueur des parties
        if self.games_played == 1:
            self.avg_game_length = game_length
        else:
            self.avg_game_length = 0.9 * self.avg_game_length + 0.1 * game_length

        self.last_updated = time.time()

    def update_elo(self, opponent_elo: float, won: bool, k_factor: float = 32):
        """Met à jour l'ELO après un match"""
        expected_score = 1 / (1 + 10 ** ((opponent_elo - self.elo_rating) / 400))
        actual_score = 1.0 if won else 0.0

        self.elo_rating += k_factor * (actual_score - expected_score)

@dataclass
class LeagueAgent:
    """Agent dans la league avec métadonnées"""
    agent_id: str
    agent_type: AgentType
    model: torch.nn.Module
    stats: AgentStats = field(default_factory=AgentStats)
    creation_time: float = field(default_factory=time.time)
    last_checkpoint_time: float = field(default_factory=time.time)
    is_training: bool = True
    payoff: float = 0.0  # Payoff against main agents

    # Spécialisation
    specialization_target: Optional[str] = None  # Pour exploiters
    diversity_objective: float = 0.0

class AgentPool:
    """Pool d'agents pour la league"""

    def __init__(self, max_pool_size: int = 50):
        self.agents: Dict[str, LeagueAgent] = {}
        self.max_pool_size = max_pool_size

        # Indices par type
        self.agents_by_type: Dict[AgentType, List[str]] = defaultdict(list)

        # Historique des performances
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        logger.info(f"Agent Pool initialisé (taille max: {max_pool_size})")

    def add_agent(self, agent: LeagueAgent) -> bool:
        """Ajoute un agent au pool"""
        if len(self.agents) >= self.max_pool_size:
            # Retirer l'agent le moins performant du même type
            self._remove_weakest_agent(agent.agent_type)

        self.agents[agent.agent_id] = agent
        self.agents_by_type[agent.agent_type].append(agent.agent_id)

        logger.info(f"Agent ajouté: {agent.agent_id} ({agent.agent_type.value})")
        return True

    def get_agent(self, agent_id: str) -> Optional[LeagueAgent]:
        """Récupère un agent par ID"""
        return self.agents.get(agent_id)

    def get_agents_by_type(self, agent_type: AgentType) -> List[LeagueAgent]:
        """Récupère tous les agents d'un type donné"""
        agent_ids = self.agents_by_type[agent_type]
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]

    def get_top_agents(self, n: int = 10, sort_by: str = "elo") -> List[LeagueAgent]:
        """Récupère les meilleurs agents"""
        agents = list(self.agents.values())

        if sort_by == "elo":
            agents.sort(key=lambda a: a.stats.elo_rating, reverse=True)
        elif sort_by == "win_rate":
            agents.sort(key=lambda a: a.stats.win_rate, reverse=True)
        elif sort_by == "games":
            agents.sort(key=lambda a: a.stats.games_played, reverse=True)

        return agents[:n]

    def _remove_weakest_agent(self, agent_type: AgentType):
        """Supprime l'agent le plus faible d'un type donné"""
        agents_of_type = self.get_agents_by_type(agent_type)

        if not agents_of_type:
            return

        # Trouver l'agent avec le plus bas ELO
        weakest = min(agents_of_type, key=lambda a: a.stats.elo_rating)

        # Supprimer
        del self.agents[weakest.agent_id]
        self.agents_by_type[agent_type].remove(weakest.agent_id)

        logger.info(f"Agent supprimé (faible performance): {weakest.agent_id}")

    def get_diversity_score(self, agent1: LeagueAgent, agent2: LeagueAgent) -> float:
        """Calcule un score de diversité entre deux agents"""
        # Diversité basée sur l'historique de performance
        hist1 = list(self.performance_history[agent1.agent_id])
        hist2 = list(self.performance_history[agent2.agent_id])

        if not hist1 or not hist2:
            return 1.0  # Max diversité si pas d'historique

        # Corrélation des performances (inverse = diversité)
        if len(hist1) < 3 or len(hist2) < 3:
            return 1.0

        correlation = np.corrcoef(hist1[-10:], hist2[-10:])[0, 1]
        return max(0.0, 1.0 - abs(correlation))

class MatchmakingSystem:
    """Système de matchmaking pour la league"""

    def __init__(self, agent_pool: AgentPool):
        self.agent_pool = agent_pool

        # Configuration matchmaking
        self.elo_range_limit = 200  # Différence ELO max
        self.diversity_weight = 0.3
        self.recent_opponent_penalty = 0.5

        # Historique des matchs
        self.recent_matches: deque = deque(maxlen=1000)
        self.opponent_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

        logger.info("Matchmaking System initialisé")

    def find_opponent(self,
                     agent_id: str,
                     preferred_types: Optional[List[AgentType]] = None,
                     exclude_agents: Optional[List[str]] = None) -> Optional[str]:
        """Trouve un adversaire optimal pour un agent"""

        agent = self.agent_pool.get_agent(agent_id)
        if not agent:
            return None

        candidates = []
        exclude_agents = exclude_agents or []

        # Collecter candidats
        if preferred_types:
            for agent_type in preferred_types:
                candidates.extend(self.agent_pool.get_agents_by_type(agent_type))
        else:
            candidates = list(self.agent_pool.agents.values())

        # Filtrer
        candidates = [c for c in candidates if c.agent_id != agent_id and c.agent_id not in exclude_agents]

        if not candidates:
            return None

        # Scorer les candidats
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_matchmaking_score(agent, candidate)
            scored_candidates.append((candidate, score))

        # Sélectionner le meilleur
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            best_opponent = scored_candidates[0][0]

            # Enregistrer le match
            self._record_match_intent(agent_id, best_opponent.agent_id)

            return best_opponent.agent_id

        return None

    def _calculate_matchmaking_score(self, agent: LeagueAgent, candidate: LeagueAgent) -> float:
        """Calcule un score de qualité de match"""
        score = 0.0

        # 1. Proximité ELO (plus proche = meilleur)
        elo_diff = abs(agent.stats.elo_rating - candidate.stats.elo_rating)
        elo_score = max(0.0, 1.0 - (elo_diff / self.elo_range_limit))
        score += elo_score * 0.4

        # 2. Diversité comportementale
        diversity_score = self.agent_pool.get_diversity_score(agent, candidate)
        score += diversity_score * self.diversity_weight

        # 3. Éviter adversaires récents
        recent_games = self.opponent_history[agent.agent_id]
        if candidate.agent_id in recent_games:
            recency_penalty = (len(recent_games) - list(recent_games).index(candidate.agent_id)) / len(recent_games)
            score -= recency_penalty * self.recent_opponent_penalty

        # 4. Équilibrage des types d'agents
        type_bonus = self._get_type_matching_bonus(agent.agent_type, candidate.agent_type)
        score += type_bonus * 0.2

        return max(0.0, score)

    def _get_type_matching_bonus(self, type1: AgentType, type2: AgentType) -> float:
        """Bonus pour certaines combinaisons de types"""
        # Main agents vs tous types
        if type1 == AgentType.MAIN_AGENT:
            return 1.0

        # Exploiters vs main agents (leur objectif)
        if type1 in [AgentType.EXPLOITER, AgentType.MAIN_EXPLOITER] and type2 == AgentType.MAIN_AGENT:
            return 1.2

        # League exploiters vs autres league exploiters
        if type1 == AgentType.LEAGUE_EXPLOITER and type2 == AgentType.LEAGUE_EXPLOITER:
            return 0.8

        return 0.5

    def _record_match_intent(self, agent1_id: str, agent2_id: str):
        """Enregistre l'intention de match"""
        self.opponent_history[agent1_id].append(agent2_id)
        self.opponent_history[agent2_id].append(agent1_id)

class LeagueManager:
    """Gestionnaire principal de la league"""

    def __init__(self,
                 league_size: int = 32,
                 main_agents_ratio: float = 0.35,
                 exploiters_ratio: float = 0.25,
                 league_exploiters_ratio: float = 0.40):

        self.league_size = league_size
        self.main_agents_ratio = main_agents_ratio
        self.exploiters_ratio = exploiters_ratio
        self.league_exploiters_ratio = league_exploiters_ratio

        # Composants
        self.agent_pool = AgentPool(max_pool_size=league_size)
        self.matchmaking = MatchmakingSystem(self.agent_pool)

        # État de la league
        self.generation = 0
        self.total_games_played = 0
        self.league_stats = {
            "games_per_generation": 0,
            "avg_game_length": 0.0,
            "diversity_score": 0.0
        }

        # Threading pour matchmaking
        self.match_queue = deque()
        self.active_matches: Dict[str, Dict] = {}
        self.match_lock = threading.Lock()

        # Initialiser agents de base
        self._initialize_base_agents()

        logger.info(f"League Manager initialisé (taille: {league_size})")

    def _initialize_base_agents(self):
        """Initialise les agents de base de la league"""

        # Calculer nombres d'agents par type
        num_main = max(1, int(self.league_size * self.main_agents_ratio))
        num_exploiters = max(1, int(self.league_size * self.exploiters_ratio))
        num_league_exploiters = self.league_size - num_main - num_exploiters

        agent_counts = {
            AgentType.MAIN_AGENT: num_main,
            AgentType.EXPLOITER: num_exploiters,
            AgentType.LEAGUE_EXPLOITER: num_league_exploiters
        }

        for agent_type, count in agent_counts.items():
            for i in range(count):
                agent = self._create_agent(agent_type, i)
                self.agent_pool.add_agent(agent)

        logger.info(f"Agents initialisés: {dict(agent_counts)}")

    def _create_agent(self, agent_type: AgentType, index: int) -> LeagueAgent:
        """Crée un nouvel agent"""

        agent_id = f"{agent_type.value}_{index}_{self.generation}"

        # Modèle selon le type
        if agent_type == AgentType.MAIN_AGENT:
            model = AlphaStarHRMModel(use_hrm=True)  # HRM pour agents principaux
        else:
            model = AlphaStarModel(use_hrm=False)  # Plus léger pour exploiters

        # Optimisations AMD
        model = model.to(config.amd.device if hasattr(config.amd, 'device') else 'cpu')

        agent = LeagueAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            model=model,
            stats=AgentStats()
        )

        return agent

    def step_league(self) -> Dict[str, Any]:
        """Avance la league d'une étape"""

        step_start = time.time()

        # 1. Générer des matchs
        matches_created = self._generate_matches()

        # 2. Traiter résultats des matchs terminés
        results_processed = self._process_match_results()

        # 3. Évolution des agents
        agents_evolved = self._evolve_agents()

        # 4. Mise à jour statistiques
        self._update_league_stats()

        step_time = time.time() - step_start

        step_info = {
            "generation": self.generation,
            "matches_created": matches_created,
            "results_processed": results_processed,
            "agents_evolved": agents_evolved,
            "total_games": self.total_games_played,
            "step_time": step_time,
            "active_matches": len(self.active_matches),
            "league_stats": self.league_stats.copy()
        }

        return step_info

    def _generate_matches(self) -> int:
        """Génère de nouveaux matchs"""
        matches_created = 0

        # Sélectionner agents disponibles
        available_agents = [
            a for a in self.agent_pool.agents.values()
            if a.agent_id not in self.active_matches and a.is_training
        ]

        random.shuffle(available_agents)

        # Créer matchs par paires
        i = 0
        while i + 1 < len(available_agents):
            agent1 = available_agents[i]

            # Trouver adversaire optimal
            opponent_id = self.matchmaking.find_opponent(
                agent1.agent_id,
                exclude_agents=[a.agent_id for a in available_agents[:i+1]]
            )

            if opponent_id and opponent_id in [a.agent_id for a in available_agents[i+1:]]:
                # Créer le match
                match_info = {
                    "match_id": f"match_{self.total_games_played}",
                    "agent1_id": agent1.agent_id,
                    "agent2_id": opponent_id,
                    "start_time": time.time(),
                    "status": "pending"
                }

                with self.match_lock:
                    self.active_matches[match_info["match_id"]] = match_info
                    self.match_queue.append(match_info["match_id"])

                matches_created += 1
                self.total_games_played += 1

                # Retirer l'adversaire de la liste
                available_agents = [a for a in available_agents if a.agent_id != opponent_id]

            i += 1

        return matches_created

    def _process_match_results(self) -> int:
        """Traite les résultats des matchs terminés"""
        # Cette méthode serait appelée par le système d'entraînement
        # pour signaler les résultats des matchs
        return 0

    def record_match_result(self,
                           match_id: str,
                           winner_id: Optional[str],
                           loser_id: Optional[str],
                           game_length: float,
                           rewards: Dict[str, float]):
        """Enregistre le résultat d'un match"""

        with self.match_lock:
            if match_id not in self.active_matches:
                logger.warning(f"Match inconnu: {match_id}")
                return

            match_info = self.active_matches[match_id]
            agent1_id = match_info["agent1_id"]
            agent2_id = match_info["agent2_id"]

            # Récupérer agents
            agent1 = self.agent_pool.get_agent(agent1_id)
            agent2 = self.agent_pool.get_agent(agent2_id)

            if not agent1 or not agent2:
                logger.error(f"Agents introuvables pour match {match_id}")
                return

            # Mettre à jour statistiques
            reward1 = rewards.get(agent1_id, 0.0)
            reward2 = rewards.get(agent2_id, 0.0)

            if winner_id == agent1_id:
                agent1.stats.update_from_game(True, game_length, reward1)
                agent2.stats.update_from_game(False, game_length, reward2)

                # Mise à jour ELO
                agent1.stats.update_elo(agent2.stats.elo_rating, True)
                agent2.stats.update_elo(agent1.stats.elo_rating, False)

            elif winner_id == agent2_id:
                agent1.stats.update_from_game(False, game_length, reward1)
                agent2.stats.update_from_game(True, game_length, reward2)

                agent1.stats.update_elo(agent2.stats.elo_rating, False)
                agent2.stats.update_elo(agent1.stats.elo_rating, True)

            else:
                # Match nul
                agent1.stats.draws += 1
                agent2.stats.draws += 1

            # Mettre à jour historiques de performance
            self.agent_pool.performance_history[agent1_id].append(reward1)
            self.agent_pool.performance_history[agent2_id].append(reward2)

            # Retirer match des actifs
            del self.active_matches[match_id]

        logger.info(f"Match {match_id} terminé: {winner_id} vs {loser_id}")

    def _evolve_agents(self) -> int:
        """Fait évoluer la league (agents, génération)"""
        agents_evolved = 0

        # Logique d'évolution selon les performances
        # - Promouvoir de bons exploiters en main agents
        # - Créer de nouveaux exploiters ciblant les faiblesses
        # - Retirer agents sous-performants

        top_exploiters = [
            a for a in self.agent_pool.get_agents_by_type(AgentType.EXPLOITER)
            if a.stats.win_rate > 0.7 and a.stats.games_played > 10
        ]

        for exploiter in top_exploiters[:2]:  # Promouvoir max 2 par génération
            # Créer nouveau main agent basé sur l'exploiter
            new_main = self._create_agent(AgentType.MAIN_AGENT, len(self.agent_pool.agents))

            # Copier poids (simplification)
            try:
                new_main.model.load_state_dict(exploiter.model.state_dict())
                self.agent_pool.add_agent(new_main)
                agents_evolved += 1
                logger.info(f"Exploiter {exploiter.agent_id} promu en main agent")
            except Exception as e:
                logger.error(f"Erreur promotion agent: {e}")

        return agents_evolved

    def _update_league_stats(self):
        """Met à jour les statistiques de la league"""
        all_agents = list(self.agent_pool.agents.values())

        if all_agents:
            total_games = sum(a.stats.games_played for a in all_agents)
            avg_elo = np.mean([a.stats.elo_rating for a in all_agents])
            avg_win_rate = np.mean([a.stats.win_rate for a in all_agents])

            # Diversité basée sur spread ELO
            elo_ratings = [a.stats.elo_rating for a in all_agents]
            elo_std = np.std(elo_ratings)
            diversity_score = min(1.0, elo_std / 200.0)  # Normaliser

            self.league_stats.update({
                "total_games": total_games,
                "avg_elo": avg_elo,
                "avg_win_rate": avg_win_rate,
                "diversity_score": diversity_score,
                "agent_count": len(all_agents),
                "generation": self.generation
            })

    def get_league_ranking(self) -> List[Dict[str, Any]]:
        """Retourne le classement de la league"""
        agents = self.agent_pool.get_top_agents(n=20, sort_by="elo")

        ranking = []
        for i, agent in enumerate(agents):
            ranking.append({
                "rank": i + 1,
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type.value,
                "elo": round(agent.stats.elo_rating, 1),
                "win_rate": round(agent.stats.win_rate, 3),
                "games_played": agent.stats.games_played,
                "wins": agent.stats.wins,
                "losses": agent.stats.losses
            })

        return ranking

    def get_league_summary(self) -> Dict[str, Any]:
        """Retourne un résumé complet de la league"""
        return {
            "generation": self.generation,
            "league_size": len(self.agent_pool.agents),
            "total_games_played": self.total_games_played,
            "active_matches": len(self.active_matches),
            "league_stats": self.league_stats,
            "top_agents": self.get_league_ranking()[:5],
            "agents_by_type": {
                agent_type.value: len(self.agent_pool.get_agents_by_type(agent_type))
                for agent_type in AgentType
            }
        }

    def save_league_state(self, save_path: str):
        """Sauvegarde l'état de la league"""
        save_data = {
            "generation": self.generation,
            "total_games_played": self.total_games_played,
            "league_stats": self.league_stats,
            "agents": {}
        }

        # Sauvegarder agents (métadonnées seulement, pas les modèles)
        for agent_id, agent in self.agent_pool.agents.items():
            save_data["agents"][agent_id] = {
                "agent_type": agent.agent_type.value,
                "stats": {
                    "games_played": agent.stats.games_played,
                    "wins": agent.stats.wins,
                    "losses": agent.stats.losses,
                    "elo_rating": agent.stats.elo_rating,
                    "win_rate": agent.stats.win_rate
                },
                "creation_time": agent.creation_time
            }

        # Sauvegarder
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"État league sauvegardé: {save_path}")

# Factory function
def create_league_system(league_size: int = 32) -> LeagueManager:
    """Crée un système de league configuré"""
    return LeagueManager(league_size=league_size)