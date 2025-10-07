"""
Autonomous Brain INTEGRATED - Cerveau décisionnel autonome avec tous les systèmes avancés
Intègre: HRM, Decision Engine, Strategy Selector, Passive Intelligence, Opportunity Manager,
Fatigue Simulation, Combat System, Navigation, Vision V2
"""

import time
import logging
import random
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

# Imports des systèmes avancés
from core.decision.decision_engine import DecisionEngine, DecisionContext, Decision, Priority, ActionType
from core.decision.strategy_selector import StrategySelector, StrategyType, NavigationPriority
from core.decision.config import DecisionConfigManager

from core.intelligence.passive_intelligence import PassiveIntelligence
from core.intelligence.opportunity_manager import OpportunityManager
from core.intelligence.fatigue_simulation import FatigueSimulator

from core.combat.combo_library import ComboLibrary, CharacterClass
from core.combat.after_action_report import AARManager, CombatOutcome
from core.combat.post_combat_analysis import CombatAnalyzer

from core.navigation_system.ganymede_navigator import GanymedeNavigator, NavigationPriority as NavPriority
from core.navigation_system.pathfinding_engine import create_pathfinding_engine

from core.quest_system import QuestManager
from core.professions import ProfessionManager

logger = logging.getLogger(__name__)


@dataclass
class Objective:
    """Objectif du bot"""
    name: str
    priority: int  # 1 = max priorité, 10 = min
    condition: str  # Condition pour activer l'objectif
    tasks: List[str]  # Tâches à accomplir


class IntegratedAutonomousBrain:
    """
    Cerveau autonome intégré avec tous les systèmes avancés

    Architecture hiérarchique:
    1. Survie (priorité max)
    2. Combat (avec ComboLibrary)
    3. Objectifs (farming, questing, etc.)
    4. Idle/Exploration

    Systèmes intégrés:
    - DecisionEngine: Évaluation multi-critères
    - StrategySelector: Adaptation stratégique
    - PassiveIntelligence: Apprentissage continu
    - OpportunityManager: Détection opportunités
    - FatigueSimulator: Simulation fatigue humaine
    - ComboLibrary: Combos de sorts par classe
    - GanymedeNavigator: Navigation intelligente
    - AARManager: Analyse post-combat
    """

    def __init__(self, character_class: CharacterClass = CharacterClass.IOP):
        # Configuration
        self.config_manager = DecisionConfigManager()

        # Systèmes de décision
        self.decision_engine = DecisionEngine()
        self.strategy_selector = StrategySelector()
        self.config_manager.configure_decision_engine(self.decision_engine)
        self.config_manager.configure_strategy_selector(self.strategy_selector)

        # Systèmes d'intelligence
        self.passive_intelligence = PassiveIntelligence()
        self.opportunity_manager = OpportunityManager()
        self.fatigue_simulator = FatigueSimulator()

        # Systèmes de combat
        self.combo_library = ComboLibrary()
        self.character_class = character_class
        self.aar_manager = AARManager()
        self.combat_analyzer = CombatAnalyzer()

        # Systèmes de navigation
        self.ganymede_navigator = GanymedeNavigator()
        self.pathfinder = create_pathfinding_engine()

        # Systèmes de jeu
        self.quest_manager = QuestManager()
        self.profession_manager = ProfessionManager()

        # État actuel
        self.current_objective = "idle"
        self.current_strategy = StrategyType.BALANCED
        self.objectives = self._init_objectives()

        # Historique et stats
        self.decision_history = []
        self.last_action_time = 0
        self.action_cooldown = 0.5

        self.stats = {
            'decisions_made': 0,
            'combat_decisions': 0,
            'navigation_decisions': 0,
            'idle_decisions': 0,
            'opportunities_detected': 0,
            'combos_executed': 0
        }

        # Charger templates monstres
        self.monster_templates = self._load_monster_templates()

        logger.info(f"IntegratedAutonomousBrain initialisé (classe: {character_class.value})")

    def _init_objectives(self) -> List[Objective]:
        """Initialise les objectifs disponibles"""
        return [
            Objective(
                name="survival",
                priority=1,
                condition="hp_low",
                tasks=["heal", "flee", "use_potion"]
            ),
            Objective(
                name="combat",
                priority=2,
                condition="in_combat",
                tasks=["attack", "use_combo", "move_tactical", "defend"]
            ),
            Objective(
                name="farming",
                priority=5,
                condition="farming_mode",
                tasks=["detect_monsters", "engage_combat", "harvest_loot"]
            ),
            Objective(
                name="questing",
                priority=4,
                condition="has_active_quest",
                tasks=["follow_quest", "complete_objectives"]
            ),
            Objective(
                name="idle",
                priority=10,
                condition="no_objective",
                tasks=["detect_opportunities", "explore"]
            )
        ]

    def _load_monster_templates(self) -> Dict[str, Any]:
        """Charge les templates de monstres"""
        templates_file = Path("assets/templates/monster/monster_templates.json")

        if templates_file.exists():
            try:
                with open(templates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Templates de monstres chargés: {len(data.get('templates', {}))} types")
                    return data.get('templates', {})
            except Exception as e:
                logger.error(f"Erreur chargement templates: {e}")

        return {}

    def decide(self, game_state: Any, vision_data: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Prend une décision basée sur l'état du jeu avec tous les systèmes

        Args:
            game_state: GameState object
            vision_data: Données de détection (monstres, ressources, etc.)

        Returns:
            Decision dict ou None si pas d'action
        """
        # Cooldown entre actions
        if time.time() - self.last_action_time < self.action_cooldown:
            return None

        self.stats['decisions_made'] += 1

        # Appliquer fatigue (dégrade les performances au fil du temps)
        self.fatigue_simulator.update()
        fatigue_level = self.fatigue_simulator.get_fatigue_level()

        # Construire contexte de décision
        context = self._build_decision_context(game_state, vision_data)

        # Sélectionner stratégie adaptée
        strategy_type, strategy_config = self.strategy_selector.select_strategy(context)
        self.current_strategy = strategy_type

        # Détecter opportunités
        opportunities = self.opportunity_manager.detect_opportunities(
            game_state,
            vision_data or {}
        )

        if opportunities:
            self.stats['opportunities_detected'] += len(opportunities)
            logger.info(f"{len(opportunities)} opportunités détectées")

        # Analyse passive de la situation
        passive_analysis = self.passive_intelligence.analyze_situation(game_state)

        # === HIÉRARCHIE DE DÉCISION ===

        decision = None

        # PRIORITÉ 1: SURVIE
        if game_state.character.hp_percent < 30:
            decision = self._decide_survival(game_state, context)
            if decision:
                self._record_decision(decision, context)
                return decision

        # PRIORITÉ 2: COMBAT
        if game_state.combat.in_combat:
            decision = self._decide_combat_advanced(game_state, context)
            self.stats['combat_decisions'] += 1

        # PRIORITÉ 3: OPPORTUNITÉS DÉTECTÉES
        elif opportunities:
            decision = self._decide_opportunity(opportunities[0], game_state, context)

        # PRIORITÉ 4: OBJECTIF ACTUEL
        elif game_state.current_objective == "farming":
            decision = self._decide_farming_advanced(game_state, vision_data, context)

        elif game_state.current_objective == "questing":
            decision = self._decide_questing_advanced(game_state, context)

        # PRIORITÉ 5: IDLE/EXPLORATION
        else:
            decision = self._decide_idle_advanced(game_state, context)
            self.stats['idle_decisions'] += 1

        # Enregistrer et retourner
        if decision:
            self._record_decision(decision, context)
            self.last_action_time = time.time()

        return decision

    def _build_decision_context(self, game_state: Any, vision_data: Optional[Dict]) -> DecisionContext:
        """Construit le contexte de décision"""
        return DecisionContext(
            health_percent=game_state.character.hp_percent,
            mana_percent=game_state.character.mana_percent if hasattr(game_state.character, 'mana_percent') else 100.0,
            pod_percent=game_state.character.pod_percent if hasattr(game_state.character, 'pod_percent') else 0.0,
            in_combat=game_state.combat.in_combat,
            enemies_count=len(game_state.combat.get_alive_enemies()) if game_state.combat.in_combat else 0,
            allies_count=0,  # TODO
            combat_difficulty=game_state.combat.difficulty if hasattr(game_state.combat, 'difficulty') else 0.0,
            current_map=getattr(game_state.environment, 'current_map', ''),
            safe_zone=getattr(game_state.environment, 'safe_zone', True),
            resources_available=vision_data.get('resources', []) if vision_data else [],
            current_objective=game_state.current_objective,
            objective_progress=0.0,  # TODO
            session_time=game_state.session_time,
            risk_tolerance=0.5,  # TODO: configurable
            efficiency_focus=0.5
        )

    def _decide_survival(self, game_state: Any, context: DecisionContext) -> Optional[Dict[str, Any]]:
        """Décisions de survie avec système avancé"""
        logger.warning(f"MODE SURVIE: HP {game_state.character.hp_percent}%")

        possible_decisions = [
            Decision(
                action_id="flee_combat",
                action_type=ActionType.SURVIVAL,
                priority=Priority.CRITICAL,
                confidence=0.9,
                estimated_duration=2.0,
                success_probability=0.8,
                risk_level=0.2,
                reward_estimate=100.0,  # Sauver sa vie
                module_source="survival"
            ),
            Decision(
                action_id="use_heal_potion",
                action_type=ActionType.SURVIVAL,
                priority=Priority.CRITICAL,
                confidence=0.85,
                estimated_duration=1.0,
                success_probability=0.95,
                risk_level=0.1,
                reward_estimate=80.0,
                module_source="survival"
            )
        ]

        best_decision = self.decision_engine.make_decision(possible_decisions, context)

        if best_decision:
            return {
                'action_type': 'survival',
                'action_id': best_decision.action_id,
                'details': {},
                'reason': 'hp_critical',
                'confidence': best_decision.confidence
            }

        return None

    def _decide_combat_advanced(self, game_state: Any, context: DecisionContext) -> Optional[Dict[str, Any]]:
        """Décisions de combat avec ComboLibrary et système avancé"""
        if not game_state.combat.my_turn:
            return None

        enemies = game_state.combat.get_alive_enemies()
        if not enemies:
            return None

        # Récupérer meilleur combo pour la situation
        best_combo = self.combo_library.get_best_combo(
            character_class=self.character_class,
            available_pa=game_state.character.pa,
            available_pm=getattr(game_state.character, 'pm', 3),
            enemy_count=len(enemies),
            hp_percent=game_state.character.hp_percent
        )

        if best_combo:
            target = game_state.combat.get_weakest_enemy()

            if target:
                self.stats['combos_executed'] += 1

                return {
                    'action_type': 'combat_combo',
                    'combo_id': best_combo.id,
                    'combo_name': best_combo.name,
                    'spell_sequence': best_combo.spell_sequence,
                    'details': {
                        'target_x': target.position[0],
                        'target_y': target.position[1]
                    },
                    'reason': f'execute_combo_{best_combo.category.value}',
                    'expected_damage': best_combo.effects.get('damage_bonus', 0)
                }

        # Fallback: attaque simple
        target = game_state.combat.get_weakest_enemy()
        if target and game_state.character.pa >= 3:
            return {
                'action_type': 'spell',
                'details': {
                    'spell_key': '1',
                    'target_x': target.position[0],
                    'target_y': target.position[1]
                },
                'reason': 'attack_weakest_enemy'
            }

        # Passer tour
        return {
            'action_type': 'shortcut',
            'details': {'keys': 'space'},
            'reason': 'no_action_available'
        }

    def _decide_farming_advanced(
        self,
        game_state: Any,
        vision_data: Optional[Dict],
        context: DecisionContext
    ) -> Optional[Dict[str, Any]]:
        """Décisions de farming avec détection monsters et navigation intelligente"""

        if not vision_data:
            return self._decide_exploration_advanced(game_state, context)

        # Détecter monstres
        monsters_detected = vision_data.get('monsters', [])

        if monsters_detected:
            # Filtrer monstres appropriés pour le niveau
            player_level = getattr(game_state.character, 'level', 1)
            suitable_monsters = [
                m for m in monsters_detected
                if self._is_suitable_target(m, player_level)
            ]

            if suitable_monsters:
                # Choisir la cible la plus proche
                nearest_monster = min(
                    suitable_monsters,
                    key=lambda m: self._distance(
                        game_state.character.position,
                        m.get('center', (0, 0))
                    )
                )

                logger.info(f"Monstre détecté: {nearest_monster.get('name', 'Unknown')} (confiance: {nearest_monster.get('confidence', 0):.2f})")

                return {
                    'action_type': 'engage_monster',
                    'monster_type': nearest_monster.get('type'),
                    'monster_name': nearest_monster.get('name'),
                    'details': {
                        'click_x': nearest_monster.get('center', (0, 0))[0],
                        'click_y': nearest_monster.get('center', (0, 0))[1]
                    },
                    'reason': 'farming_detected_monster',
                    'confidence': nearest_monster.get('confidence', 0)
                }

        # Pas de monstres: chercher ressources
        resources = vision_data.get('resources', [])
        if resources:
            nearest_resource = resources[0]
            return {
                'action_type': 'harvest',
                'details': {
                    'x': nearest_resource.get('position', (0, 0))[0],
                    'y': nearest_resource.get('position', (0, 0))[1]
                },
                'reason': 'harvest_resource'
            }

        # Rien trouvé: navigation intelligente vers zone de farm
        return self._decide_navigation_to_farming_spot(game_state, context)

    def _decide_navigation_to_farming_spot(
        self,
        game_state: Any,
        context: DecisionContext
    ) -> Optional[Dict[str, Any]]:
        """Navigation intelligente vers une zone de farm"""
        player_level = getattr(game_state.character, 'level', 1)

        # Demander au navigator de trouver un spot de farm approprié
        route = self.ganymede_navigator.navigate_to_location(
            target_location="ganymede_east",  # Forêt = bon spot
            player_level=player_level,
            priority=NavPriority.EFFICIENCY
        )

        if route and route.steps:
            first_step = route.steps[0]

            return {
                'action_type': 'navigate',
                'details': {
                    'x': first_step.to_position[0],
                    'y': first_step.to_position[1],
                    'direction': first_step.description
                },
                'reason': 'navigate_to_farming_spot',
                'route_id': route.route_id
            }

        # Fallback: exploration aléatoire
        return self._decide_exploration_advanced(game_state, context)

    def _decide_questing_advanced(
        self,
        game_state: Any,
        context: DecisionContext
    ) -> Optional[Dict[str, Any]]:
        """Décisions de quête avec QuestManager"""
        active_quest = self.quest_manager.get_active_quest()

        if active_quest:
            # Obtenir objectif actuel
            current_objective = active_quest.get_current_objective()

            if current_objective:
                objective_type = current_objective.get('type')

                if objective_type == 'dialogue':
                    npc_position = current_objective.get('location', {}).get('position', [0, 0])
                    return {
                        'action_type': 'talk_to_npc',
                        'details': {'x': npc_position[0], 'y': npc_position[1]},
                        'reason': 'quest_objective_dialogue'
                    }

                elif objective_type == 'combat':
                    # Se comporte comme farming mais pour un monstre spécifique
                    return self._decide_farming_advanced(game_state, None, context)

        return None

    def _decide_opportunity(
        self,
        opportunity: Dict[str, Any],
        game_state: Any,
        context: DecisionContext
    ) -> Optional[Dict[str, Any]]:
        """Exploite une opportunité détectée"""
        opp_type = opportunity.get('type')

        if opp_type == 'combat':
            return {
                'action_type': 'engage_opportunity',
                'details': opportunity.get('details', {}),
                'reason': 'opportunity_combat',
                'value': opportunity.get('value', 0)
            }

        elif opp_type == 'resource':
            return {
                'action_type': 'harvest_opportunity',
                'details': opportunity.get('details', {}),
                'reason': 'opportunity_resource',
                'value': opportunity.get('value', 0)
            }

        return None

    def _decide_idle_advanced(
        self,
        game_state: Any,
        context: DecisionContext
    ) -> Optional[Dict[str, Any]]:
        """Décisions idle avec intelligence passive"""
        # Analyser environnement
        analysis = self.passive_intelligence.analyze_situation(game_state)

        # Si détecte des patterns intéressants, agir
        if analysis.get('suggested_action'):
            return analysis['suggested_action']

        # Sinon, explorer intelligemment
        return self._decide_exploration_advanced(game_state, context)

    def _decide_exploration_advanced(
        self,
        game_state: Any,
        context: DecisionContext
    ) -> Optional[Dict[str, Any]]:
        """Exploration avec pathfinding intelligent"""
        # Navigation aléatoire mais intelligente
        rand_x = random.randint(200, 800)
        rand_y = random.randint(200, 500)

        return {
            'action_type': 'move',
            'details': {'x': rand_x, 'y': rand_y},
            'reason': 'intelligent_exploration',
            'strategy': self.current_strategy.value
        }

    def _is_suitable_target(self, monster: Dict, player_level: int) -> bool:
        """Vérifie si un monstre est une cible appropriée"""
        monster_template = self.monster_templates.get(monster.get('type', ''))

        if not monster_template:
            return True  # Par défaut, accepter

        level_range = monster_template.get('level_range', [1, 100])
        min_level, max_level = level_range

        # Accepter si niveau du monstre est dans +/-5 niveaux du joueur
        return (player_level - 5) <= max_level <= (player_level + 10)

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule distance euclidienne"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def _record_decision(self, decision: Dict[str, Any], context: DecisionContext):
        """Enregistre une décision pour analyse"""
        self.decision_history.append({
            'time': time.time(),
            'decision': decision,
            'context': {
                'hp': context.health_percent,
                'in_combat': context.in_combat,
                'strategy': self.current_strategy.value
            }
        })

        # Limiter historique
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

    def record_combat_outcome(
        self,
        combat_id: str,
        outcome: CombatOutcome,
        duration: float,
        damage_dealt: int,
        damage_received: int
    ):
        """Enregistre le résultat d'un combat pour apprentissage"""
        self.aar_manager.record_combat(
            combat_id=combat_id,
            outcome=outcome,
            duration=duration,
            damage_dealt=damage_dealt,
            damage_received=damage_received
        )

    def set_objective(self, objective: str):
        """Change l'objectif actuel"""
        self.current_objective = objective
        logger.info(f"Objectif changé: {objective}")

    def get_stats(self) -> Dict[str, Any]:
        """Retourne statistiques complètes"""
        return {
            **self.stats,
            'fatigue_level': self.fatigue_simulator.get_fatigue_level(),
            'current_strategy': self.current_strategy.value,
            'decision_engine_stats': self.decision_engine.get_decision_stats(),
            'strategy_analytics': self.strategy_selector.get_strategy_analytics(),
            'passive_intelligence_insights': self.passive_intelligence.get_insights(),
            'opportunities_total': self.opportunity_manager.get_stats()
        }

    def save_state(self, save_path: str = "config/brain_state"):
        """Sauvegarde l'état complet du brain"""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        self.config_manager.save_engine_state(self.decision_engine)
        self.config_manager.save_strategy_state(self.strategy_selector)

        logger.info(f"État du brain sauvegardé dans {save_path}")


def create_integrated_brain(character_class: CharacterClass = CharacterClass.IOP) -> IntegratedAutonomousBrain:
    """Factory function pour créer un cerveau intégré"""
    return IntegratedAutonomousBrain(character_class)
