"""
Module d'Apprentissage Génétique pour l'IA DOFUS Évolutive
Phase 4 : Méta-Évolution & Auto-Amélioration

Ce module implémente des algorithmes génétiques pour optimiser
les stratégies, les paramètres et l'architecture de l'IA.
"""

import asyncio
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import json
import time
import copy
from datetime import datetime, timedelta
import logging
import hashlib

logger = logging.getLogger(__name__)

class GeneType(Enum):
    PARAMETER = "parameter"
    STRATEGY = "strategy"
    ARCHITECTURE = "architecture"
    BEHAVIOR = "behavior"
    THRESHOLD = "threshold"

class MutationType(Enum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    DISCRETE = "discrete"
    SWAP = "swap"
    INVERSION = "inversion"

class SelectionMethod(Enum):
    ROULETTE = "roulette"
    TOURNAMENT = "tournament"
    ELITISM = "elitism"
    RANK_BASED = "rank_based"

@dataclass
class Gene:
    """Représente un gène dans le génome de l'IA"""
    gene_id: str
    gene_type: GeneType
    value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    dependencies: List[str] = field(default_factory=list)

@dataclass
class Individual:
    """Représente un individu dans la population génétique"""
    individual_id: str
    genome: Dict[str, Gene]
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    performance_history: List[float] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)

@dataclass
class EvolutionParameters:
    """Paramètres de l'évolution génétique"""
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 5
    max_generations: int = 100
    convergence_threshold: float = 0.001
    diversity_threshold: float = 0.1

class GeneticFitnessEvaluator:
    """Évaluateur de fitness pour les individus"""

    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []
        self.fitness_weights = {
            "performance": 0.4,
            "efficiency": 0.3,
            "stability": 0.2,
            "adaptability": 0.1
        }

    async def evaluate_individual(self, individual: Individual,
                                environment_state: Dict[str, Any]) -> float:
        """Évalue la fitness d'un individu"""
        try:
            # Simulation de l'évaluation basée sur le génome
            performance_score = await self._evaluate_performance(individual, environment_state)
            efficiency_score = await self._evaluate_efficiency(individual, environment_state)
            stability_score = await self._evaluate_stability(individual, environment_state)
            adaptability_score = await self._evaluate_adaptability(individual, environment_state)

            # Calcul de la fitness pondérée
            fitness = (
                performance_score * self.fitness_weights["performance"] +
                efficiency_score * self.fitness_weights["efficiency"] +
                stability_score * self.fitness_weights["stability"] +
                adaptability_score * self.fitness_weights["adaptability"]
            )

            # Bonus pour la diversité
            diversity_bonus = await self._calculate_diversity_bonus(individual)
            fitness += diversity_bonus

            # Pénalité pour l'âge (éviter la stagnation)
            age_penalty = min(individual.age * 0.001, 0.05)
            fitness -= age_penalty

            individual.fitness = max(0.0, min(1.0, fitness))
            individual.performance_history.append(individual.fitness)

            # Limitation de l'historique
            if len(individual.performance_history) > 20:
                individual.performance_history = individual.performance_history[-15:]

            logger.debug(f"Individu {individual.individual_id}: fitness = {individual.fitness:.3f}")

            return individual.fitness

        except Exception as e:
            logger.error(f"Erreur évaluation fitness: {e}")
            return 0.0

    async def _evaluate_performance(self, individual: Individual,
                                  environment_state: Dict[str, Any]) -> float:
        """Évalue la performance de l'individu"""
        try:
            # Simulation basée sur les gènes de stratégie
            performance = 0.5  # Base

            # Analyse des gènes de stratégie
            for gene_id, gene in individual.genome.items():
                if gene.gene_type == GeneType.STRATEGY:
                    if isinstance(gene.value, str):
                        strategy_bonus = {
                            "aggressive": 0.8,
                            "balanced": 0.6,
                            "conservative": 0.4,
                            "adaptive": 0.9
                        }.get(gene.value, 0.5)
                        performance += strategy_bonus * 0.2

                elif gene.gene_type == GeneType.PARAMETER:
                    # Optimisation des paramètres
                    if "efficiency" in gene_id.lower():
                        if isinstance(gene.value, (int, float)):
                            normalized_value = min(max(gene.value, 0), 1)
                            performance += normalized_value * 0.1

            # Facteur environnemental
            env_difficulty = environment_state.get("difficulty", 0.5)
            performance *= (1.0 - env_difficulty * 0.3)

            return min(1.0, max(0.0, performance))

        except Exception as e:
            logger.error(f"Erreur évaluation performance: {e}")
            return 0.5

    async def _evaluate_efficiency(self, individual: Individual,
                                 environment_state: Dict[str, Any]) -> float:
        """Évalue l'efficacité de l'individu"""
        try:
            efficiency = 0.5

            # Analyse des gènes de seuils
            threshold_genes = [g for g in individual.genome.values()
                             if g.gene_type == GeneType.THRESHOLD]

            if threshold_genes:
                # Efficacité basée sur l'optimisation des seuils
                threshold_scores = []
                for gene in threshold_genes:
                    if isinstance(gene.value, (int, float)):
                        # Optimal autour de 0.7 pour la plupart des seuils
                        optimal_distance = abs(gene.value - 0.7)
                        threshold_score = 1.0 - optimal_distance
                        threshold_scores.append(max(0.0, threshold_score))

                if threshold_scores:
                    efficiency = np.mean(threshold_scores)

            # Bonus pour la complexité modérée
            genome_complexity = len(individual.genome) / 20.0  # Normalisation
            if 0.3 <= genome_complexity <= 0.7:
                efficiency += 0.1

            return min(1.0, max(0.0, efficiency))

        except Exception as e:
            logger.error(f"Erreur évaluation efficacité: {e}")
            return 0.5

    async def _evaluate_stability(self, individual: Individual,
                                environment_state: Dict[str, Any]) -> float:
        """Évalue la stabilité de l'individu"""
        try:
            # Stabilité basée sur l'historique de performance
            if len(individual.performance_history) < 3:
                return 0.5

            # Calcul de la variance de performance
            performance_variance = np.var(individual.performance_history)
            stability = 1.0 - min(performance_variance * 2, 1.0)

            # Bonus pour performance constante
            recent_performances = individual.performance_history[-5:]
            if len(recent_performances) >= 3:
                performance_trend = np.polyfit(range(len(recent_performances)),
                                             recent_performances, 1)[0]

                # Tendance positive légère = bon
                if 0 <= performance_trend <= 0.05:
                    stability += 0.1

            return min(1.0, max(0.0, stability))

        except Exception as e:
            logger.error(f"Erreur évaluation stabilité: {e}")
            return 0.5

    async def _evaluate_adaptability(self, individual: Individual,
                                   environment_state: Dict[str, Any]) -> float:
        """Évalue l'adaptabilité de l'individu"""
        try:
            adaptability = 0.5

            # Adaptabilité basée sur les gènes de comportement
            behavior_genes = [g for g in individual.genome.values()
                            if g.gene_type == GeneType.BEHAVIOR]

            if behavior_genes:
                adaptability_scores = []
                for gene in behavior_genes:
                    if isinstance(gene.value, dict):
                        # Vérifier la diversité des comportements
                        behavior_diversity = len(gene.value) / 10.0  # Normalisation
                        adaptability_scores.append(min(behavior_diversity, 1.0))

                if adaptability_scores:
                    adaptability = np.mean(adaptability_scores)

            # Bonus pour l'âge modéré (expérience vs flexibilité)
            if 5 <= individual.age <= 15:
                adaptability += 0.1

            return min(1.0, max(0.0, adaptability))

        except Exception as e:
            logger.error(f"Erreur évaluation adaptabilité: {e}")
            return 0.5

    async def _calculate_diversity_bonus(self, individual: Individual) -> float:
        """Calcule le bonus de diversité"""
        try:
            # Bonus basé sur l'unicité du génome
            genome_hash = self._hash_genome(individual.genome)

            # Simulation de vérification de diversité
            # En pratique, on comparerait avec les autres individus
            diversity_score = random.uniform(0.0, 0.05)  # Bonus maximum 5%

            return diversity_score

        except Exception as e:
            logger.error(f"Erreur calcul diversité: {e}")
            return 0.0

    def _hash_genome(self, genome: Dict[str, Gene]) -> str:
        """Calcule un hash unique du génome"""
        try:
            genome_str = ""
            for gene_id in sorted(genome.keys()):
                gene = genome[gene_id]
                genome_str += f"{gene_id}:{gene.value}:"

            return hashlib.md5(genome_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Erreur hash génome: {e}")
            return "error_hash"

class GeneticOperators:
    """Opérateurs génétiques (sélection, croisement, mutation)"""

    def __init__(self):
        self.crossover_methods = {
            "single_point": self._single_point_crossover,
            "uniform": self._uniform_crossover,
            "arithmetic": self._arithmetic_crossover
        }

    async def selection(self, population: List[Individual],
                       method: SelectionMethod = SelectionMethod.TOURNAMENT,
                       **kwargs) -> List[Individual]:
        """Sélection des parents pour la reproduction"""
        try:
            if method == SelectionMethod.TOURNAMENT:
                return await self._tournament_selection(population, **kwargs)
            elif method == SelectionMethod.ROULETTE:
                return await self._roulette_selection(population, **kwargs)
            elif method == SelectionMethod.ELITISM:
                return await self._elitism_selection(population, **kwargs)
            elif method == SelectionMethod.RANK_BASED:
                return await self._rank_based_selection(population, **kwargs)
            else:
                return population[:len(population)//2]

        except Exception as e:
            logger.error(f"Erreur sélection: {e}")
            return population[:len(population)//2]

    async def _tournament_selection(self, population: List[Individual],
                                  tournament_size: int = 5,
                                  num_parents: int = None) -> List[Individual]:
        """Sélection par tournoi"""
        try:
            if num_parents is None:
                num_parents = len(population) // 2

            selected = []
            for _ in range(num_parents):
                tournament = random.sample(population, min(tournament_size, len(population)))
                winner = max(tournament, key=lambda ind: ind.fitness)
                selected.append(winner)

            return selected

        except Exception as e:
            logger.error(f"Erreur sélection tournoi: {e}")
            return population[:num_parents] if num_parents else population[:len(population)//2]

    async def _roulette_selection(self, population: List[Individual],
                                num_parents: int = None) -> List[Individual]:
        """Sélection par roulette"""
        try:
            if num_parents is None:
                num_parents = len(population) // 2

            # Calcul des probabilités
            total_fitness = sum(ind.fitness for ind in population)
            if total_fitness == 0:
                return random.sample(population, num_parents)

            probabilities = [ind.fitness / total_fitness for ind in population]

            selected = []
            for _ in range(num_parents):
                spin = random.random()
                cumulative = 0
                for i, prob in enumerate(probabilities):
                    cumulative += prob
                    if spin <= cumulative:
                        selected.append(population[i])
                        break

            return selected

        except Exception as e:
            logger.error(f"Erreur sélection roulette: {e}")
            return population[:num_parents] if num_parents else population[:len(population)//2]

    async def _elitism_selection(self, population: List[Individual],
                               elite_ratio: float = 0.2) -> List[Individual]:
        """Sélection élitiste"""
        try:
            num_elite = max(1, int(len(population) * elite_ratio))
            sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
            return sorted_pop[:num_elite]

        except Exception as e:
            logger.error(f"Erreur sélection élitiste: {e}")
            return population[:max(1, len(population)//5)]

    async def _rank_based_selection(self, population: List[Individual],
                                  num_parents: int = None) -> List[Individual]:
        """Sélection basée sur le rang"""
        try:
            if num_parents is None:
                num_parents = len(population) // 2

            # Tri par fitness
            sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)

            # Attribution des probabilités basées sur le rang
            ranks = list(range(len(sorted_pop), 0, -1))
            total_rank = sum(ranks)

            selected = []
            for _ in range(num_parents):
                spin = random.uniform(0, total_rank)
                cumulative = 0
                for i, rank in enumerate(ranks):
                    cumulative += rank
                    if spin <= cumulative:
                        selected.append(sorted_pop[i])
                        break

            return selected

        except Exception as e:
            logger.error(f"Erreur sélection par rang: {e}")
            return population[:num_parents] if num_parents else population[:len(population)//2]

    async def crossover(self, parent1: Individual, parent2: Individual,
                       method: str = "single_point") -> Tuple[Individual, Individual]:
        """Croisement entre deux parents"""
        try:
            crossover_func = self.crossover_methods.get(method, self._single_point_crossover)
            return await crossover_func(parent1, parent2)

        except Exception as e:
            logger.error(f"Erreur croisement: {e}")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    async def _single_point_crossover(self, parent1: Individual,
                                    parent2: Individual) -> Tuple[Individual, Individual]:
        """Croisement à un point"""
        try:
            # Création des enfants
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

            # Nouvel ID pour les enfants
            child1.individual_id = f"child_{int(time.time() * 1000000) % 1000000}"
            child2.individual_id = f"child_{int(time.time() * 1000000 + 1) % 1000000}"

            # Reset des métriques
            child1.fitness = 0.0
            child2.fitness = 0.0
            child1.performance_history = []
            child2.performance_history = []
            child1.age = 0
            child2.age = 0
            child1.parent_ids = [parent1.individual_id, parent2.individual_id]
            child2.parent_ids = [parent1.individual_id, parent2.individual_id]

            # Croisement des gènes
            gene_ids = list(parent1.genome.keys())
            if len(gene_ids) > 1:
                crossover_point = random.randint(1, len(gene_ids) - 1)

                for i, gene_id in enumerate(gene_ids):
                    if i >= crossover_point:
                        # Échange des gènes
                        if gene_id in parent2.genome:
                            child1.genome[gene_id] = copy.deepcopy(parent2.genome[gene_id])
                        if gene_id in parent1.genome:
                            child2.genome[gene_id] = copy.deepcopy(parent1.genome[gene_id])

            return child1, child2

        except Exception as e:
            logger.error(f"Erreur croisement un point: {e}")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    async def _uniform_crossover(self, parent1: Individual,
                               parent2: Individual) -> Tuple[Individual, Individual]:
        """Croisement uniforme"""
        try:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

            # Nouvel ID pour les enfants
            child1.individual_id = f"child_{int(time.time() * 1000000) % 1000000}"
            child2.individual_id = f"child_{int(time.time() * 1000000 + 1) % 1000000}"

            # Reset des métriques
            child1.fitness = 0.0
            child2.fitness = 0.0
            child1.performance_history = []
            child2.performance_history = []
            child1.age = 0
            child2.age = 0
            child1.parent_ids = [parent1.individual_id, parent2.individual_id]
            child2.parent_ids = [parent1.individual_id, parent2.individual_id]

            # Croisement uniforme des gènes
            for gene_id in parent1.genome.keys():
                if gene_id in parent2.genome and random.random() < 0.5:
                    child1.genome[gene_id] = copy.deepcopy(parent2.genome[gene_id])
                    child2.genome[gene_id] = copy.deepcopy(parent1.genome[gene_id])

            return child1, child2

        except Exception as e:
            logger.error(f"Erreur croisement uniforme: {e}")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    async def _arithmetic_crossover(self, parent1: Individual,
                                  parent2: Individual) -> Tuple[Individual, Individual]:
        """Croisement arithmétique pour les gènes numériques"""
        try:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

            # Nouvel ID et reset
            child1.individual_id = f"child_{int(time.time() * 1000000) % 1000000}"
            child2.individual_id = f"child_{int(time.time() * 1000000 + 1) % 1000000}"
            child1.fitness = 0.0
            child2.fitness = 0.0
            child1.performance_history = []
            child2.performance_history = []
            child1.age = 0
            child2.age = 0
            child1.parent_ids = [parent1.individual_id, parent2.individual_id]
            child2.parent_ids = [parent1.individual_id, parent2.individual_id]

            alpha = random.uniform(0.3, 0.7)

            # Croisement arithmétique pour les gènes numériques
            for gene_id in parent1.genome.keys():
                if gene_id in parent2.genome:
                    gene1 = parent1.genome[gene_id]
                    gene2 = parent2.genome[gene_id]

                    if isinstance(gene1.value, (int, float)) and isinstance(gene2.value, (int, float)):
                        # Croisement arithmétique
                        value1 = alpha * gene1.value + (1 - alpha) * gene2.value
                        value2 = (1 - alpha) * gene1.value + alpha * gene2.value

                        child1.genome[gene_id].value = value1
                        child2.genome[gene_id].value = value2

            return child1, child2

        except Exception as e:
            logger.error(f"Erreur croisement arithmétique: {e}")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    async def mutation(self, individual: Individual,
                      mutation_rate: float = 0.1) -> Individual:
        """Mutation d'un individu"""
        try:
            mutated = copy.deepcopy(individual)

            for gene_id, gene in mutated.genome.items():
                if random.random() < mutation_rate:
                    await self._mutate_gene(gene)

            return mutated

        except Exception as e:
            logger.error(f"Erreur mutation: {e}")
            return individual

    async def _mutate_gene(self, gene: Gene):
        """Mutation d'un gène spécifique"""
        try:
            if gene.gene_type == GeneType.PARAMETER and isinstance(gene.value, (int, float)):
                # Mutation gaussienne pour les paramètres numériques
                mutation_value = random.gauss(0, gene.mutation_strength)
                new_value = gene.value + mutation_value

                # Contraintes min/max
                if gene.min_value is not None:
                    new_value = max(new_value, gene.min_value)
                if gene.max_value is not None:
                    new_value = min(new_value, gene.max_value)

                gene.value = new_value

            elif gene.gene_type == GeneType.STRATEGY and isinstance(gene.value, str):
                # Mutation discrète pour les stratégies
                strategies = ["aggressive", "balanced", "conservative", "adaptive"]
                if random.random() < gene.mutation_rate:
                    gene.value = random.choice(strategies)

            elif gene.gene_type == GeneType.THRESHOLD and isinstance(gene.value, (int, float)):
                # Mutation pour les seuils
                mutation_value = random.uniform(-gene.mutation_strength, gene.mutation_strength)
                new_value = gene.value + mutation_value
                gene.value = max(0.0, min(1.0, new_value))

            elif gene.gene_type == GeneType.BEHAVIOR and isinstance(gene.value, dict):
                # Mutation pour les comportements
                if gene.value and random.random() < gene.mutation_rate:
                    key = random.choice(list(gene.value.keys()))
                    if isinstance(gene.value[key], (int, float)):
                        mutation = random.uniform(-0.1, 0.1)
                        gene.value[key] = max(0.0, min(1.0, gene.value[key] + mutation))

        except Exception as e:
            logger.error(f"Erreur mutation gène: {e}")

class GeneticLearningEngine:
    """Moteur principal d'apprentissage génétique"""

    def __init__(self):
        self.population: List[Individual] = []
        self.fitness_evaluator = GeneticFitnessEvaluator()
        self.genetic_operators = GeneticOperators()
        self.generation = 0
        self.evolution_history: List[Dict[str, Any]] = []

        # Paramètres par défaut
        self.parameters = EvolutionParameters()

        # Statistiques
        self.best_individual: Optional[Individual] = None
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    async def initialize_population(self, population_size: int = 50):
        """Initialise la population de base"""
        try:
            self.population = []
            self.generation = 0

            for i in range(population_size):
                individual = await self._create_random_individual(f"init_{i}")
                self.population.append(individual)

            logger.info(f"Population initialisée: {len(self.population)} individus")

        except Exception as e:
            logger.error(f"Erreur initialisation population: {e}")

    async def _create_random_individual(self, base_id: str) -> Individual:
        """Crée un individu aléatoire"""
        try:
            genome = {}

            # Gènes de stratégie
            genome["combat_strategy"] = Gene(
                gene_id="combat_strategy",
                gene_type=GeneType.STRATEGY,
                value=random.choice(["aggressive", "balanced", "conservative", "adaptive"]),
                mutation_rate=0.1
            )

            genome["resource_strategy"] = Gene(
                gene_id="resource_strategy",
                gene_type=GeneType.STRATEGY,
                value=random.choice(["efficiency", "speed", "safety", "balanced"]),
                mutation_rate=0.1
            )

            # Gènes de paramètres
            genome["learning_rate"] = Gene(
                gene_id="learning_rate",
                gene_type=GeneType.PARAMETER,
                value=random.uniform(0.01, 0.3),
                min_value=0.001,
                max_value=0.5,
                mutation_strength=0.02
            )

            genome["exploration_factor"] = Gene(
                gene_id="exploration_factor",
                gene_type=GeneType.PARAMETER,
                value=random.uniform(0.1, 0.9),
                min_value=0.0,
                max_value=1.0,
                mutation_strength=0.05
            )

            # Gènes de seuils
            genome["confidence_threshold"] = Gene(
                gene_id="confidence_threshold",
                gene_type=GeneType.THRESHOLD,
                value=random.uniform(0.3, 0.9),
                min_value=0.0,
                max_value=1.0,
                mutation_strength=0.05
            )

            genome["risk_threshold"] = Gene(
                gene_id="risk_threshold",
                gene_type=GeneType.THRESHOLD,
                value=random.uniform(0.2, 0.8),
                min_value=0.0,
                max_value=1.0,
                mutation_strength=0.05
            )

            # Gènes de comportement
            genome["social_behavior"] = Gene(
                gene_id="social_behavior",
                gene_type=GeneType.BEHAVIOR,
                value={
                    "cooperation": random.uniform(0.3, 0.9),
                    "competition": random.uniform(0.1, 0.7),
                    "communication": random.uniform(0.2, 0.8),
                    "leadership": random.uniform(0.1, 0.6)
                },
                mutation_rate=0.15
            )

            # Gènes d'architecture
            genome["decision_complexity"] = Gene(
                gene_id="decision_complexity",
                gene_type=GeneType.ARCHITECTURE,
                value=random.randint(3, 10),
                min_value=1,
                max_value=15,
                mutation_strength=1
            )

            individual = Individual(
                individual_id=f"{base_id}_{int(time.time() * 1000) % 10000}",
                genome=genome,
                generation=self.generation
            )

            return individual

        except Exception as e:
            logger.error(f"Erreur création individu: {e}")
            # Individu minimal en cas d'erreur
            return Individual(
                individual_id=f"error_{base_id}",
                genome={"default": Gene("default", GeneType.PARAMETER, 0.5)},
                generation=self.generation
            )

    async def evolve_generation(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Évolution d'une génération"""
        try:
            generation_start = datetime.now()

            # 1. Évaluation de la fitness
            await self._evaluate_population(environment_state)

            # 2. Sélection des parents
            parents = await self.genetic_operators.selection(
                self.population,
                SelectionMethod.TOURNAMENT,
                tournament_size=self.parameters.tournament_size,
                num_parents=self.parameters.population_size // 2
            )

            # 3. Génération des enfants par croisement
            children = []
            for i in range(0, len(parents) - 1, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

                if random.random() < self.parameters.crossover_rate:
                    child1, child2 = await self.genetic_operators.crossover(parent1, parent2)
                    children.extend([child1, child2])

            # 4. Mutation des enfants
            for child in children:
                if random.random() < self.parameters.mutation_rate:
                    child = await self.genetic_operators.mutation(child, self.parameters.mutation_rate)

            # 5. Sélection élitiste
            elite = await self.genetic_operators.selection(
                self.population,
                SelectionMethod.ELITISM,
                elite_ratio=self.parameters.elite_size / self.parameters.population_size
            )

            # 6. Nouvelle population
            new_population = elite + children

            # Ajustement de la taille de population
            if len(new_population) > self.parameters.population_size:
                new_population = sorted(new_population, key=lambda x: x.fitness, reverse=True)
                new_population = new_population[:self.parameters.population_size]
            elif len(new_population) < self.parameters.population_size:
                # Compléter avec des individus aléatoires
                needed = self.parameters.population_size - len(new_population)
                for i in range(needed):
                    random_individual = await self._create_random_individual(f"fill_{self.generation}_{i}")
                    new_population.append(random_individual)

            # 7. Mise à jour de la population
            self.population = new_population
            self.generation += 1

            # 8. Mise à jour de l'âge
            for individual in self.population:
                individual.age += 1

            # 9. Statistiques
            generation_stats = await self._calculate_generation_stats()

            # 10. Mise à jour du meilleur individu
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(current_best)

            self.best_fitness_history.append(current_best.fitness)

            generation_duration = (datetime.now() - generation_start).total_seconds()

            result = {
                "generation": self.generation,
                "best_fitness": current_best.fitness,
                "average_fitness": generation_stats["average_fitness"],
                "diversity": generation_stats["diversity"],
                "elite_preserved": len(elite),
                "children_created": len(children),
                "duration_seconds": generation_duration,
                "convergence_rate": self._calculate_convergence_rate()
            }

            self.evolution_history.append(result)

            logger.info(f"Génération {self.generation}: "
                       f"Meilleure fitness = {current_best.fitness:.3f}, "
                       f"Moyenne = {generation_stats['average_fitness']:.3f}")

            return result

        except Exception as e:
            logger.error(f"Erreur évolution génération: {e}")
            return {"generation": self.generation, "error": str(e)}

    async def _evaluate_population(self, environment_state: Dict[str, Any]):
        """Évalue la fitness de toute la population"""
        try:
            evaluation_tasks = []
            for individual in self.population:
                task = self.fitness_evaluator.evaluate_individual(individual, environment_state)
                evaluation_tasks.append(task)

            # Évaluation en parallèle
            await asyncio.gather(*evaluation_tasks)

        except Exception as e:
            logger.error(f"Erreur évaluation population: {e}")

    async def _calculate_generation_stats(self) -> Dict[str, Any]:
        """Calcule les statistiques de la génération"""
        try:
            fitnesses = [ind.fitness for ind in self.population]

            stats = {
                "average_fitness": np.mean(fitnesses),
                "std_fitness": np.std(fitnesses),
                "min_fitness": min(fitnesses),
                "max_fitness": max(fitnesses),
                "diversity": self._calculate_population_diversity()
            }

            self.diversity_history.append(stats["diversity"])

            return stats

        except Exception as e:
            logger.error(f"Erreur calcul statistiques: {e}")
            return {
                "average_fitness": 0.0,
                "std_fitness": 0.0,
                "min_fitness": 0.0,
                "max_fitness": 0.0,
                "diversity": 0.0
            }

    def _calculate_population_diversity(self) -> float:
        """Calcule la diversité de la population"""
        try:
            if len(self.population) < 2:
                return 0.0

            # Diversité basée sur les différences de génomes
            diversity_scores = []

            for i in range(len(self.population)):
                for j in range(i + 1, len(self.population)):
                    similarity = self._calculate_genome_similarity(
                        self.population[i].genome,
                        self.population[j].genome
                    )
                    diversity_scores.append(1.0 - similarity)

            return np.mean(diversity_scores) if diversity_scores else 0.0

        except Exception as e:
            logger.error(f"Erreur calcul diversité: {e}")
            return 0.0

    def _calculate_genome_similarity(self, genome1: Dict[str, Gene],
                                   genome2: Dict[str, Gene]) -> float:
        """Calcule la similarité entre deux génomes"""
        try:
            common_genes = set(genome1.keys()) & set(genome2.keys())
            if not common_genes:
                return 0.0

            similarities = []
            for gene_id in common_genes:
                gene1 = genome1[gene_id]
                gene2 = genome2[gene_id]

                if gene1.gene_type == gene2.gene_type:
                    if isinstance(gene1.value, (int, float)) and isinstance(gene2.value, (int, float)):
                        # Similarité numérique
                        max_diff = abs(gene1.max_value - gene1.min_value) if gene1.max_value and gene1.min_value else 1.0
                        diff = abs(gene1.value - gene2.value)
                        similarity = 1.0 - min(diff / max_diff, 1.0)
                        similarities.append(similarity)

                    elif gene1.value == gene2.value:
                        similarities.append(1.0)
                    else:
                        similarities.append(0.0)

            return np.mean(similarities) if similarities else 0.0

        except Exception as e:
            logger.error(f"Erreur calcul similarité génome: {e}")
            return 0.0

    def _calculate_convergence_rate(self) -> float:
        """Calcule le taux de convergence"""
        try:
            if len(self.best_fitness_history) < 5:
                return 0.0

            recent_improvements = []
            for i in range(1, min(6, len(self.best_fitness_history))):
                improvement = self.best_fitness_history[-i] - self.best_fitness_history[-i-1]
                recent_improvements.append(max(0, improvement))

            return np.mean(recent_improvements)

        except Exception as e:
            logger.error(f"Erreur calcul convergence: {e}")
            return 0.0

    async def get_best_configuration(self) -> Dict[str, Any]:
        """Retourne la meilleure configuration trouvée"""
        try:
            if not self.best_individual:
                return {"error": "No best individual found"}

            config = {
                "individual_id": self.best_individual.individual_id,
                "fitness": self.best_individual.fitness,
                "generation": self.best_individual.generation,
                "age": self.best_individual.age,
                "genome": {}
            }

            for gene_id, gene in self.best_individual.genome.items():
                config["genome"][gene_id] = {
                    "type": gene.gene_type.value,
                    "value": gene.value
                }

            return config

        except Exception as e:
            logger.error(f"Erreur récupération meilleure config: {e}")
            return {"error": str(e)}

# Intégration principale
class GeneticLearningModule:
    """Module principal d'apprentissage génétique"""

    def __init__(self):
        self.genetic_engine = GeneticLearningEngine()
        self.is_active = False
        self.learning_cycles = 0

    async def initialize(self, config: Dict[str, Any]):
        """Initialise le module d'apprentissage génétique"""
        try:
            # Configuration des paramètres génétiques
            genetic_config = config.get("genetic_learning", {})

            if "population_size" in genetic_config:
                self.genetic_engine.parameters.population_size = genetic_config["population_size"]

            if "mutation_rate" in genetic_config:
                self.genetic_engine.parameters.mutation_rate = genetic_config["mutation_rate"]

            if "crossover_rate" in genetic_config:
                self.genetic_engine.parameters.crossover_rate = genetic_config["crossover_rate"]

            # Initialisation de la population
            await self.genetic_engine.initialize_population(
                self.genetic_engine.parameters.population_size
            )

            self.is_active = True
            logger.info("Module d'apprentissage génétique initialisé")

        except Exception as e:
            logger.error(f"Erreur initialisation apprentissage génétique: {e}")

    async def run_learning_cycle(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute un cycle d'apprentissage génétique"""
        try:
            if not self.is_active:
                return {"learning_active": False}

            self.learning_cycles += 1

            # Évolution d'une génération
            evolution_result = await self.genetic_engine.evolve_generation(environment_state)

            return {
                "learning_active": True,
                "cycle": self.learning_cycles,
                "evolution_result": evolution_result,
                "best_individual": await self.genetic_engine.get_best_configuration(),
                "population_size": len(self.genetic_engine.population),
                "generation": self.genetic_engine.generation
            }

        except Exception as e:
            logger.error(f"Erreur cycle apprentissage: {e}")
            return {"learning_active": False, "error": str(e)}

    async def get_learning_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques d'apprentissage"""
        try:
            return {
                "total_cycles": self.learning_cycles,
                "current_generation": self.genetic_engine.generation,
                "population_size": len(self.genetic_engine.population),
                "best_fitness_history": self.genetic_engine.best_fitness_history[-20:],  # 20 dernières
                "diversity_history": self.genetic_engine.diversity_history[-20:],
                "best_configuration": await self.genetic_engine.get_best_configuration(),
                "convergence_rate": self.genetic_engine._calculate_convergence_rate()
            }

        except Exception as e:
            logger.error(f"Erreur récupération métriques apprentissage: {e}")
            return {"error": str(e)}