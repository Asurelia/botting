"""
Intelligent Decision Maker - Système de prise de décision multi-couches
Combine HRM, logique de jeu, et stratégies avancées

Auteur: Claude Code
Intégration: TacticalBot + HRM + Serpent.AI
"""

import numpy as np
import torch
import cv2
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

try:
    from hrm_core import GameState, HRMDecision, HRMBot
    from adaptive_learner import AdaptiveLearner, PerformanceMetrics
except ImportError:
    # Fallback pour imports relatifs
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from hrm_core import GameState, HRMDecision, HRMBot
    from adaptive_learner import AdaptiveLearner, PerformanceMetrics

logger = logging.getLogger(__name__)

class DecisionPriority(Enum):
    """Priorités de décision"""
    EMERGENCY = 10      # Danger immédiat
    URGENT = 8          # Action critique
    HIGH = 6            # Important
    NORMAL = 4          # Routine
    LOW = 2             # Optionnel
    IDLE = 1            # Inactivité

class ActionCategory(Enum):
    """Catégories d'actions"""
    COMBAT = "combat"
    MOVEMENT = "movement"
    INTERACTION = "interaction"
    INVENTORY = "inventory"
    QUEST = "quest"
    SOCIAL = "social"
    RESOURCE = "resource"
    UTILITY = "utility"

@dataclass
class GameContext:
    """Contexte étendu du jeu"""
    # État de base
    base_state: GameState

    # Analyse visuelle
    screen_analysis: Dict = field(default_factory=dict)
    minimap_data: Dict = field(default_factory=dict)
    ui_elements: List[Dict] = field(default_factory=list)

    # Contexte de jeu
    current_objective: Optional[str] = None
    active_quests: List[Dict] = field(default_factory=list)
    party_members: List[Dict] = field(default_factory=list)
    enemies_nearby: List[Dict] = field(default_factory=list)
    npcs_nearby: List[Dict] = field(default_factory=list)

    # Ressources et économie
    currency: Dict = field(default_factory=dict)
    resources: Dict = field(default_factory=dict)
    market_prices: Dict = field(default_factory=dict)

    # Progression
    experience_gained: float = 0.0
    skills_progress: Dict = field(default_factory=dict)
    achievements_unlocked: List[str] = field(default_factory=list)

@dataclass
class AdvancedDecision:
    """Décision avancée avec contexte enrichi"""
    # Décision de base HRM
    base_decision: HRMDecision

    # Enrichissement contextuel
    category: ActionCategory
    priority: DecisionPriority
    risk_assessment: float  # 0.0 = sûr, 1.0 = très risqué
    time_to_execute: float
    success_probability: float
    resource_cost: Dict = field(default_factory=dict)

    # Prédictions
    expected_rewards: Dict = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)

    # Contexte stratégique
    fits_strategy: bool = True
    long_term_benefit: float = 0.0
    opportunity_cost: float = 0.0

class VisionAnalyzer:
    """Analyseur de vision pour extraire l'information de l'écran"""

    def __init__(self):
        self.template_cache = {}
        self.last_analysis_time = 0
        self.analysis_cache = {}

    def analyze_screen(self, screenshot: np.ndarray) -> Dict:
        """Analyse complète de l'écran"""
        current_time = time.time()

        # Cache pour éviter les analyses trop fréquentes
        if current_time - self.last_analysis_time < 0.1:  # 100ms cache
            return self.analysis_cache

        analysis = {
            "ui_elements": self._detect_ui_elements(screenshot),
            "minimap": self._analyze_minimap(screenshot),
            "health_mana": self._detect_health_mana(screenshot),
            "enemies": self._detect_enemies(screenshot),
            "npcs": self._detect_npcs(screenshot),
            "items": self._detect_items(screenshot),
            "quest_indicators": self._detect_quest_indicators(screenshot),
            "text_info": self._extract_text(screenshot)
        }

        self.analysis_cache = analysis
        self.last_analysis_time = current_time
        return analysis

    def _detect_ui_elements(self, screenshot: np.ndarray) -> List[Dict]:
        """Détecte les éléments d'interface"""
        # Exemple basique - à adapter selon votre jeu
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Détection de contours pour les boutons
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ui_elements = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:  # Filtrer par taille
                x, y, w, h = cv2.boundingRect(contour)
                ui_elements.append({
                    "type": "button",
                    "position": (x, y),
                    "size": (w, h),
                    "area": area
                })

        return ui_elements

    def _analyze_minimap(self, screenshot: np.ndarray) -> Dict:
        """Analyse la minimap pour extraire la position et les POI"""
        # Zone typique de minimap (coin supérieur droit)
        h, w = screenshot.shape[:2]
        minimap_region = screenshot[0:h//4, 3*w//4:w]

        return {
            "player_position": self._find_player_on_minimap(minimap_region),
            "points_of_interest": self._find_poi_on_minimap(minimap_region),
            "path_to_objective": self._calculate_minimap_path(minimap_region)
        }

    def _detect_health_mana(self, screenshot: np.ndarray) -> Dict:
        """Détecte les barres de vie et mana"""
        # Recherche de barres colorées (rouge/bleu typiquement)
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        # Masque pour le rouge (vie)
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)

        # Masque pour le bleu (mana)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        return {
            "health_percentage": self._calculate_bar_percentage(red_mask),
            "mana_percentage": self._calculate_bar_percentage(blue_mask),
            "status_effects": self._detect_status_effects(screenshot)
        }

    def _detect_enemies(self, screenshot: np.ndarray) -> List[Dict]:
        """Détecte les ennemis à l'écran"""
        # Utiliser la détection de couleur, contours, ou templates
        enemies = []

        # Exemple: recherche de noms d'ennemis en rouge
        red_text_regions = self._find_colored_text(screenshot, color_range="red")

        for region in red_text_regions:
            enemies.append({
                "position": region["position"],
                "estimated_level": self._estimate_enemy_level(region),
                "threat_level": self._assess_threat_level(region),
                "type": "unknown"
            })

        return enemies

    def _detect_npcs(self, screenshot: np.ndarray) -> List[Dict]:
        """Détecte les NPCs"""
        # Recherche de noms en vert/jaune (NPCs typiques)
        npcs = []
        green_text_regions = self._find_colored_text(screenshot, color_range="green")

        for region in green_text_regions:
            npcs.append({
                "position": region["position"],
                "type": self._classify_npc_type(region),
                "has_quest": self._has_quest_indicator(region),
                "interactable": True
            })

        return npcs

    def _detect_items(self, screenshot: np.ndarray) -> List[Dict]:
        """Détecte les objets au sol"""
        return []  # Implémentation dépendante du jeu

    def _detect_quest_indicators(self, screenshot: np.ndarray) -> List[Dict]:
        """Détecte les indicateurs de quête"""
        return []  # Points d'exclamation, flèches, etc.

    def _extract_text(self, screenshot: np.ndarray) -> Dict:
        """Extrait le texte visible (OCR)"""
        try:
            import pytesseract
            text = pytesseract.image_to_string(screenshot)
            return {"extracted_text": text, "confidence": 0.8}
        except ImportError:
            logger.warning("pytesseract non disponible pour l'OCR")
            return {"extracted_text": "", "confidence": 0.0}

    # Méthodes utilitaires
    def _find_player_on_minimap(self, minimap: np.ndarray) -> Tuple[int, int]:
        """Trouve la position du joueur sur la minimap"""
        return (0, 0)  # Placeholder

    def _find_poi_on_minimap(self, minimap: np.ndarray) -> List[Dict]:
        """Trouve les points d'intérêt sur la minimap"""
        return []

    def _calculate_minimap_path(self, minimap: np.ndarray) -> List[Tuple[int, int]]:
        """Calcule un chemin sur la minimap"""
        return []

    def _calculate_bar_percentage(self, mask: np.ndarray) -> float:
        """Calcule le pourcentage d'une barre"""
        if mask.size == 0:
            return 0.0
        return np.sum(mask > 0) / mask.size

    def _detect_status_effects(self, screenshot: np.ndarray) -> List[str]:
        """Détecte les effets de statut"""
        return []

    def _find_colored_text(self, screenshot: np.ndarray, color_range: str) -> List[Dict]:
        """Trouve du texte coloré"""
        return []

    def _estimate_enemy_level(self, region: Dict) -> int:
        """Estime le niveau d'un ennemi"""
        return 1

    def _assess_threat_level(self, region: Dict) -> float:
        """Évalue le niveau de menace"""
        return 0.5

    def _classify_npc_type(self, region: Dict) -> str:
        """Classifie le type de NPC"""
        return "vendor"

    def _has_quest_indicator(self, region: Dict) -> bool:
        """Vérifie si un NPC a une quête"""
        return False

class StrategicPlanner:
    """Planificateur stratégique long-terme"""

    def __init__(self):
        self.current_objectives = []
        self.long_term_goals = []
        self.resource_targets = {}
        self.skill_priorities = {}

    def plan_session(self, game_context: GameContext, session_duration: float) -> List[str]:
        """Planifie une session de jeu"""
        plans = []

        # Objectifs par priorité
        if game_context.base_state.player_health < 0.3:
            plans.append("heal_priority")

        if game_context.active_quests:
            plans.extend([f"complete_quest_{q['id']}" for q in game_context.active_quests])

        if session_duration > 3600:  # Sessions longues
            plans.append("resource_gathering")
            plans.append("skill_training")

        return plans

    def evaluate_opportunity_cost(self, action: str, alternatives: List[str]) -> float:
        """Évalue le coût d'opportunité d'une action"""
        # Comparer la valeur attendue de l'action vs alternatives
        return 0.1  # Placeholder

    def should_continue_current_activity(self, current_activity: str, new_opportunity: str) -> bool:
        """Détermine s'il faut continuer l'activité actuelle"""
        # Logique de switching cost
        return True

class IntelligentDecisionMaker:
    """Système principal de prise de décision intelligente"""

    def __init__(self, hrm_bot: HRMBot, adaptive_learner: AdaptiveLearner):
        self.hrm_bot = hrm_bot
        self.adaptive_learner = adaptive_learner
        self.vision_analyzer = VisionAnalyzer()
        self.strategic_planner = StrategicPlanner()

        # Cache de décisions
        self.decision_cache = {}
        self.last_decisions = []

        # Règles de sécurité
        self.safety_rules = self._load_safety_rules()

        # Filtres d'actions
        self.action_filters = self._setup_action_filters()

    def make_intelligent_decision(self, screenshot: np.ndarray,
                                base_game_state: GameState) -> AdvancedDecision:
        """Prend une décision intelligente multi-couches"""

        # 1. Analyse visuelle de l'écran
        screen_analysis = self.vision_analyzer.analyze_screen(screenshot)

        # 2. Construction du contexte enrichi
        game_context = self._build_game_context(base_game_state, screen_analysis)

        # 3. Décision de base avec HRM
        base_decision = self.hrm_bot.decide_action(base_game_state)

        # 4. Enrichissement contextuel
        enriched_decision = self._enrich_decision(base_decision, game_context)

        # 5. Validation de sécurité
        safe_decision = self._apply_safety_filters(enriched_decision, game_context)

        # 6. Optimisation stratégique
        final_decision = self._apply_strategic_optimization(safe_decision, game_context)

        # 7. Apprentissage de l'expérience
        self._record_decision_for_learning(final_decision, game_context)

        return final_decision

    def _build_game_context(self, base_state: GameState, screen_analysis: Dict) -> GameContext:
        """Construit le contexte de jeu enrichi"""
        return GameContext(
            base_state=base_state,
            screen_analysis=screen_analysis,
            minimap_data=screen_analysis.get("minimap", {}),
            ui_elements=screen_analysis.get("ui_elements", []),
            enemies_nearby=screen_analysis.get("enemies", []),
            npcs_nearby=screen_analysis.get("npcs", []),
            current_objective=self._determine_current_objective(screen_analysis),
            active_quests=self._extract_active_quests(screen_analysis)
        )

    def _enrich_decision(self, base_decision: HRMDecision, context: GameContext) -> AdvancedDecision:
        """Enrichit la décision HRM avec le contexte"""

        # Catégorisation de l'action
        category = self._categorize_action(base_decision.action)

        # Évaluation des risques
        risk_assessment = self._assess_risk(base_decision, context)

        # Prédiction de succès
        success_probability = self._predict_success(base_decision, context)

        # Calcul des récompenses attendues
        expected_rewards = self._calculate_expected_rewards(base_decision, context)

        return AdvancedDecision(
            base_decision=base_decision,
            category=category,
            priority=DecisionPriority(base_decision.priority),
            risk_assessment=risk_assessment,
            time_to_execute=self._estimate_execution_time(base_decision),
            success_probability=success_probability,
            expected_rewards=expected_rewards,
            fits_strategy=self._fits_current_strategy(base_decision),
            long_term_benefit=self._calculate_long_term_benefit(base_decision, context)
        )

    def _apply_safety_filters(self, decision: AdvancedDecision, context: GameContext) -> AdvancedDecision:
        """Applique les filtres de sécurité"""

        # Règle 1: Éviter les actions dangereuses avec peu de vie
        if context.base_state.player_health < 0.2 and decision.risk_assessment > 0.7:
            # Forcer une action de guérison ou fuite
            decision.base_decision.action = "use_potion"
            decision.priority = DecisionPriority.EMERGENCY
            decision.risk_assessment = 0.1

        # Règle 2: Éviter les combats inégaux
        if decision.category == ActionCategory.COMBAT and len(context.enemies_nearby) > 3:
            decision.base_decision.action = "retreat"
            decision.priority = DecisionPriority.URGENT

        # Règle 3: Respecter les cooldowns
        if not self._action_available(decision.base_decision.action):
            decision.base_decision.action = "wait"
            decision.priority = DecisionPriority.LOW

        return decision

    def _apply_strategic_optimization(self, decision: AdvancedDecision, context: GameContext) -> AdvancedDecision:
        """Applique l'optimisation stratégique"""

        # Vérifier si l'action s'aligne avec les objectifs long-terme
        if not decision.fits_strategy:
            alternative = self._find_strategic_alternative(decision, context)
            if alternative:
                decision = alternative

        # Optimiser selon la stratégie actuelle
        current_strategy = self.adaptive_learner.strategy_evolution.current_strategy
        decision = self._optimize_for_strategy(decision, current_strategy, context)

        return decision

    def _record_decision_for_learning(self, decision: AdvancedDecision, context: GameContext):
        """Enregistre la décision pour l'apprentissage futur"""
        self.last_decisions.append({
            "decision": decision,
            "context": context,
            "timestamp": time.time()
        })

        # Garder seulement les 100 dernières décisions
        if len(self.last_decisions) > 100:
            self.last_decisions = self.last_decisions[-100:]

    # Méthodes utilitaires
    def _load_safety_rules(self) -> Dict:
        """Charge les règles de sécurité"""
        return {
            "min_health_for_combat": 0.3,
            "max_enemies_to_engage": 2,
            "forbidden_actions": ["delete_character", "trade_all_items"],
            "required_items": ["healing_potion"]
        }

    def _setup_action_filters(self) -> List[Callable]:
        """Configure les filtres d'actions"""
        return [
            self._filter_dangerous_actions,
            self._filter_resource_costly_actions,
            self._filter_time_consuming_actions
        ]

    def _categorize_action(self, action: str) -> ActionCategory:
        """Catégorise une action"""
        action_categories = {
            "attack": ActionCategory.COMBAT,
            "defend": ActionCategory.COMBAT,
            "move": ActionCategory.MOVEMENT,
            "interact": ActionCategory.INTERACTION,
            "use_potion": ActionCategory.INVENTORY,
            "accept_quest": ActionCategory.QUEST,
            "gather": ActionCategory.RESOURCE
        }

        for key, category in action_categories.items():
            if key in action.lower():
                return category

        return ActionCategory.UTILITY

    def _assess_risk(self, decision: HRMDecision, context: GameContext) -> float:
        """Évalue le risque d'une décision"""
        risk = 0.0

        # Risque basé sur la santé
        if context.base_state.player_health < 0.5:
            risk += 0.3

        # Risque basé sur les ennemis
        risk += len(context.enemies_nearby) * 0.1

        # Risque basé sur l'action
        risky_actions = ["attack", "explore", "trade"]
        if any(risky in decision.action for risky in risky_actions):
            risk += 0.2

        return min(1.0, risk)

    def _predict_success(self, decision: HRMDecision, context: GameContext) -> float:
        """Prédit la probabilité de succès"""
        # Base sur la confiance HRM
        base_probability = decision.confidence

        # Ajustements contextuels
        if context.base_state.player_level > 10:
            base_probability += 0.1

        if len(context.enemies_nearby) > 1:
            base_probability -= 0.2

        return max(0.1, min(1.0, base_probability))

    def _calculate_expected_rewards(self, decision: HRMDecision, context: GameContext) -> Dict:
        """Calcule les récompenses attendues"""
        return {
            "experience": 10.0,
            "gold": 5.0,
            "items": 0.1,
            "quest_progress": 0.2
        }

    def _estimate_execution_time(self, decision: HRMDecision) -> float:
        """Estime le temps d'exécution"""
        time_estimates = {
            "move": 2.0,
            "attack": 1.5,
            "interact": 3.0,
            "use_item": 1.0,
            "wait": 0.5
        }

        for action_type, time_est in time_estimates.items():
            if action_type in decision.action:
                return time_est

        return 2.0  # Défaut

    def _fits_current_strategy(self, decision: HRMDecision) -> bool:
        """Vérifie si la décision s'aligne avec la stratégie"""
        current_strategy = self.adaptive_learner.strategy_evolution.current_strategy

        strategy_preferences = {
            "aggressive": ["attack", "engage", "fight"],
            "defensive": ["defend", "heal", "retreat"],
            "explorer": ["move", "explore", "discover"],
            "questfocused": ["quest", "objective", "complete"]
        }

        preferred_actions = strategy_preferences.get(current_strategy, [])
        return any(pref in decision.action.lower() for pref in preferred_actions)

    def _calculate_long_term_benefit(self, decision: HRMDecision, context: GameContext) -> float:
        """Calcule le bénéfice long-terme"""
        return 0.5  # Placeholder

    def _action_available(self, action: str) -> bool:
        """Vérifie si une action est disponible (cooldowns, etc.)"""
        return True  # Placeholder

    def _find_strategic_alternative(self, decision: AdvancedDecision, context: GameContext) -> Optional[AdvancedDecision]:
        """Trouve une alternative stratégique"""
        return None  # Placeholder

    def _optimize_for_strategy(self, decision: AdvancedDecision, strategy: str, context: GameContext) -> AdvancedDecision:
        """Optimise la décision pour la stratégie"""
        return decision  # Placeholder

    def _determine_current_objective(self, screen_analysis: Dict) -> Optional[str]:
        """Détermine l'objectif actuel"""
        return None

    def _extract_active_quests(self, screen_analysis: Dict) -> List[Dict]:
        """Extrait les quêtes actives"""
        return []

    # Filtres d'actions
    def _filter_dangerous_actions(self, decision: AdvancedDecision, context: GameContext) -> bool:
        """Filtre les actions dangereuses"""
        return decision.risk_assessment < 0.8

    def _filter_resource_costly_actions(self, decision: AdvancedDecision, context: GameContext) -> bool:
        """Filtre les actions coûteuses en ressources"""
        return True

    def _filter_time_consuming_actions(self, decision: AdvancedDecision, context: GameContext) -> bool:
        """Filtre les actions qui prennent trop de temps"""
        return decision.time_to_execute < 10.0