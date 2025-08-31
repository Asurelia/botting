"""
Système anti-détection avec patterns comportementaux
Évite la détection par des comportements naturels et variables
"""

import time
import random
import logging
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class DetectionRisk(Enum):
    """Niveaux de risque de détection"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BehaviorCategory(Enum):
    """Catégories de comportements surveillés"""
    TIMING = "timing"
    MOVEMENT = "movement"
    INTERACTION = "interaction"
    PROGRESSION = "progression"
    SOCIAL = "social"
    ECONOMIC = "economic"


@dataclass
class BehaviorPattern:
    """Modèle de pattern comportemental"""
    name: str
    category: BehaviorCategory
    weight: float = 1.0
    min_variance: float = 0.1
    max_variance: float = 0.3
    cooldown_seconds: float = 0.0
    risk_multiplier: float = 1.0


@dataclass
class DetectionMetrics:
    """Métriques de détection et surveillance"""
    actions_per_minute: float = 0.0
    click_precision_variance: float = 0.0
    timing_regularity: float = 0.0
    path_efficiency: float = 0.0
    error_rate: float = 0.0
    pause_pattern_score: float = 0.0
    interaction_diversity: float = 0.0
    progression_consistency: float = 0.0


class AntiDetectionSystem:
    """
    Système anti-détection complet avec analyse comportementale
    Maintient des patterns humains naturels et évite les signaux suspects
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.detection_history = deque(maxlen=10000)  # Historique limité
        self.behavior_patterns = {}
        self.current_metrics = DetectionMetrics()
        self.risk_factors = defaultdict(float)
        self.last_analysis_time = time.time()
        
        # Patterns comportementaux par défaut
        self._initialize_default_patterns()
        
        # Surveillance en temps réel
        self.action_timestamps = deque(maxlen=1000)
        self.click_positions = deque(maxlen=500)
        self.path_segments = deque(maxlen=100)
        self.interaction_types = deque(maxlen=200)
        
        # État de camouflage
        self.camouflage_active = False
        self.stealth_level = 0.5
        self.adaptive_factor = 1.0
        
        # Chargement de la configuration
        self._load_configuration()
    
    def _initialize_default_patterns(self):
        """Initialise les patterns comportementaux par défaut"""
        
        # Patterns de timing
        self.behavior_patterns.update({
            "reaction_time": BehaviorPattern(
                name="reaction_time",
                category=BehaviorCategory.TIMING,
                weight=2.0,
                min_variance=0.15,
                max_variance=0.4,
                risk_multiplier=1.5
            ),
            "action_intervals": BehaviorPattern(
                name="action_intervals", 
                category=BehaviorCategory.TIMING,
                weight=1.8,
                min_variance=0.2,
                max_variance=0.5,
                risk_multiplier=1.3
            ),
            "pause_frequency": BehaviorPattern(
                name="pause_frequency",
                category=BehaviorCategory.TIMING,
                weight=1.5,
                min_variance=0.3,
                max_variance=0.7,
                risk_multiplier=1.2
            )
        })
        
        # Patterns de mouvement
        self.behavior_patterns.update({
            "mouse_trajectory": BehaviorPattern(
                name="mouse_trajectory",
                category=BehaviorCategory.MOVEMENT,
                weight=1.7,
                min_variance=0.1,
                max_variance=0.3,
                risk_multiplier=1.4
            ),
            "click_accuracy": BehaviorPattern(
                name="click_accuracy",
                category=BehaviorCategory.MOVEMENT,
                weight=1.6,
                min_variance=0.05,
                max_variance=0.2,
                risk_multiplier=1.3
            ),
            "movement_speed": BehaviorPattern(
                name="movement_speed",
                category=BehaviorCategory.MOVEMENT,
                weight=1.4,
                min_variance=0.2,
                max_variance=0.4,
                risk_multiplier=1.1
            )
        })
        
        # Patterns d'interaction
        self.behavior_patterns.update({
            "menu_usage": BehaviorPattern(
                name="menu_usage",
                category=BehaviorCategory.INTERACTION,
                weight=1.3,
                min_variance=0.2,
                max_variance=0.6,
                risk_multiplier=1.0
            ),
            "error_recovery": BehaviorPattern(
                name="error_recovery",
                category=BehaviorCategory.INTERACTION,
                weight=1.5,
                min_variance=0.3,
                max_variance=0.8,
                risk_multiplier=1.2
            )
        })
    
    def analyze_current_behavior(self) -> DetectionRisk:
        """
        Analyse le comportement actuel et évalue le risque de détection
        """
        current_time = time.time()
        
        # Mise à jour des métriques
        self._update_metrics()
        
        # Calcul des scores de risque par catégorie
        risk_scores = {}
        for category in BehaviorCategory:
            risk_scores[category] = self._calculate_category_risk(category)
        
        # Score de risque global
        total_risk = sum(score * self._get_category_weight(cat) 
                        for cat, score in risk_scores.items())
        
        # Normalisation du score (0-100)
        normalized_risk = min(100, max(0, total_risk * 100))
        
        # Détermination du niveau de risque
        risk_level = self._get_risk_level(normalized_risk)
        
        # Sauvegarde de l'analyse
        analysis_result = {
            'timestamp': current_time,
            'risk_level': risk_level.value,
            'risk_score': normalized_risk,
            'category_risks': {cat.value: score for cat, score in risk_scores.items()},
            'metrics': asdict(self.current_metrics)
        }
        
        self.detection_history.append(analysis_result)
        self.last_analysis_time = current_time
        
        # Adaptation automatique si risque élevé
        if risk_level in [DetectionRisk.HIGH, DetectionRisk.CRITICAL]:
            self._activate_defensive_measures(risk_level)
        
        logger.debug(f"Analyse comportementale: {risk_level.value} ({normalized_risk:.1f}%)")
        
        return risk_level
    
    def get_human_timing_variation(self, base_time: float, action_type: str = "generic") -> float:
        """
        Retourne une variation de timing humaine pour éviter la régularité
        """
        # Pattern spécifique selon le type d'action
        pattern_key = f"{action_type}_timing"
        if pattern_key not in self.behavior_patterns:
            pattern = self.behavior_patterns.get("action_intervals")
        else:
            pattern = self.behavior_patterns[pattern_key]
        
        # Facteur de variance basé sur le risque actuel
        current_risk = self.risk_factors.get("timing", 0.3)
        variance_factor = pattern.min_variance + (current_risk * (pattern.max_variance - pattern.min_variance))
        
        # Application de la variance avec distribution normale
        variation = random.normalvariate(1.0, variance_factor * self.adaptive_factor)
        variation = max(0.1, variation)  # Minimum pour éviter les temps négatifs
        
        # Ajustement anti-pattern
        adjusted_time = base_time * variation
        
        # Enregistrement pour analyse
        self.action_timestamps.append(time.time())
        
        return adjusted_time
    
    def get_mouse_position_variation(self, target_pos: Tuple[int, int], 
                                   accuracy_level: float = 0.8) -> Tuple[int, int]:
        """
        Ajoute une variation naturelle à la position de la souris
        """
        x, y = target_pos
        
        # Pattern de précision des clics
        pattern = self.behavior_patterns.get("click_accuracy")
        base_variance = pattern.min_variance if pattern else 0.05
        max_variance = pattern.max_variance if pattern else 0.2
        
        # Calcul de la variance selon le niveau de précision
        position_variance = base_variance + ((1.0 - accuracy_level) * (max_variance - base_variance))
        
        # Zone de variance (pixels)
        variance_pixels = int(position_variance * 50)  # Max 10 pixels de base
        
        # Application de la variation avec distribution normale tronquée
        dx = int(np.clip(random.normalvariate(0, variance_pixels/2), -variance_pixels, variance_pixels))
        dy = int(np.clip(random.normalvariate(0, variance_pixels/2), -variance_pixels, variance_pixels))
        
        final_pos = (x + dx, y + dy)
        
        # Enregistrement pour analyse
        self.click_positions.append({
            'target': target_pos,
            'actual': final_pos,
            'variance': (dx, dy),
            'timestamp': time.time()
        })
        
        return final_pos
    
    def should_introduce_error(self, base_error_rate: float = 0.02) -> bool:
        """
        Détermine si une erreur intentionnelle doit être introduite
        """
        # Ajustement selon le pattern d'erreur
        pattern = self.behavior_patterns.get("error_recovery")
        if pattern:
            error_multiplier = 0.5 + (pattern.weight * 0.5)
            adjusted_error_rate = base_error_rate * error_multiplier
        else:
            adjusted_error_rate = base_error_rate
        
        # Facteur adaptatif selon le risque
        risk_factor = self.risk_factors.get("interaction", 0.5)
        final_error_rate = adjusted_error_rate * (1.0 + risk_factor)
        
        should_error = random.random() < final_error_rate
        
        if should_error:
            logger.debug(f"Erreur intentionnelle introduite (taux: {final_error_rate:.3f})")
        
        return should_error
    
    def get_natural_pause_suggestion(self) -> Optional[Tuple[float, str]]:
        """
        Suggère une pause naturelle si les patterns l'indiquent
        """
        current_time = time.time()
        
        # Analyse de la fréquence d'action récente
        recent_actions = [ts for ts in self.action_timestamps 
                         if current_time - ts < 300]  # 5 dernières minutes
        
        if len(recent_actions) < 10:
            return None
        
        # Calcul de la densité d'actions
        action_density = len(recent_actions) / 300  # Actions par seconde
        
        # Seuils de pause selon la densité
        if action_density > 0.3:  # Plus de 90 actions/5min
            return (random.uniform(10, 30), "high_activity")
        elif action_density > 0.2 and random.random() < 0.1:  # 10% chance
            return (random.uniform(3, 10), "regular_break")
        elif random.random() < 0.02:  # 2% chance pause aléatoire
            return (random.uniform(1, 5), "random_pause")
        
        return None
    
    def record_game_event(self, event_type: str, data: Dict[str, Any]):
        """
        Enregistre un événement de jeu pour l'analyse comportementale
        """
        timestamp = time.time()
        
        event_record = {
            'timestamp': timestamp,
            'type': event_type,
            'data': data
        }
        
        # Traitement spécialisé selon le type d'événement
        if event_type == "movement":
            self._process_movement_event(data)
        elif event_type == "combat":
            self._process_combat_event(data)
        elif event_type == "interaction":
            self._process_interaction_event(data)
        elif event_type == "progression":
            self._process_progression_event(data)
        
        # Stockage générique
        self.interaction_types.append(event_record)
        
        # Analyse périodique
        if timestamp - self.last_analysis_time > 60:  # Analyse chaque minute
            self.analyze_current_behavior()
    
    def get_stealth_recommendations(self) -> List[str]:
        """
        Retourne des recommandations pour améliorer la furtivité
        """
        recommendations = []
        
        # Analyse des métriques actuelles
        if self.current_metrics.actions_per_minute > 60:
            recommendations.append("Réduire la fréquence d'actions (>60/min détecté)")
        
        if self.current_metrics.timing_regularity > 0.8:
            recommendations.append("Augmenter la variance des timings (trop régulier)")
        
        if self.current_metrics.click_precision_variance < 0.1:
            recommendations.append("Augmenter la variance de précision (trop précis)")
        
        if self.current_metrics.error_rate < 0.01:
            recommendations.append("Introduire plus d'erreurs occasionnelles")
        
        if self.current_metrics.pause_pattern_score < 0.3:
            recommendations.append("Ajouter plus de pauses naturelles")
        
        if not recommendations:
            recommendations.append("Comportement dans les normes humaines")
        
        return recommendations
    
    def activate_stealth_mode(self, duration_minutes: int = 30):
        """
        Active un mode furtif renforcé temporairement
        """
        self.camouflage_active = True
        self.stealth_level = min(1.0, self.stealth_level + 0.3)
        self.adaptive_factor = 1.5
        
        logger.info(f"Mode furtif activé pour {duration_minutes}min (niveau: {self.stealth_level:.2f})")
        
        # Désactivation automatique après la durée spécifiée
        def deactivate_stealth():
            time.sleep(duration_minutes * 60)
            self.camouflage_active = False
            self.stealth_level = max(0.3, self.stealth_level - 0.2)
            self.adaptive_factor = 1.0
            logger.info("Mode furtif désactivé")
        
        import threading
        threading.Thread(target=deactivate_stealth, daemon=True).start()
    
    def _update_metrics(self):
        """Met à jour les métriques comportementales"""
        current_time = time.time()
        
        # Actions par minute (dernières 5 minutes)
        recent_actions = [ts for ts in self.action_timestamps 
                         if current_time - ts < 300]
        self.current_metrics.actions_per_minute = len(recent_actions) / 5.0
        
        # Variance de précision des clics
        if len(self.click_positions) > 10:
            variances = [abs(pos['variance'][0]) + abs(pos['variance'][1]) 
                        for pos in list(self.click_positions)[-50:]]
            self.current_metrics.click_precision_variance = np.std(variances) if variances else 0.0
        
        # Régularité des timings
        if len(self.action_timestamps) > 5:
            intervals = []
            timestamps = list(self.action_timestamps)[-20:]
            for i in range(1, len(timestamps)):
                intervals.append(timestamps[i] - timestamps[i-1])
            
            if intervals:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                # Plus le coefficient de variation est faible, plus c'est régulier (suspect)
                cv = std_interval / mean_interval if mean_interval > 0 else 1
                self.current_metrics.timing_regularity = 1.0 - min(1.0, cv)
        
        # Score de pattern de pause
        pause_score = self._calculate_pause_naturalness()
        self.current_metrics.pause_pattern_score = pause_score
        
        # Taux d'erreur récent
        self.current_metrics.error_rate = self._calculate_recent_error_rate()
    
    def _calculate_category_risk(self, category: BehaviorCategory) -> float:
        """Calcule le risque pour une catégorie de comportement"""
        risk_score = 0.0
        pattern_count = 0
        
        for pattern_name, pattern in self.behavior_patterns.items():
            if pattern.category == category:
                pattern_risk = self._calculate_pattern_risk(pattern)
                risk_score += pattern_risk * pattern.weight
                pattern_count += 1
        
        return risk_score / pattern_count if pattern_count > 0 else 0.0
    
    def _calculate_pattern_risk(self, pattern: BehaviorPattern) -> float:
        """Calcule le risque pour un pattern spécifique"""
        # Implémentation simplifiée - à développer selon les besoins
        base_risk = 0.3
        
        # Ajustements selon les métriques actuelles
        if pattern.name == "timing_regularity":
            base_risk = self.current_metrics.timing_regularity
        elif pattern.name == "click_accuracy":
            base_risk = 1.0 - self.current_metrics.click_precision_variance
        elif pattern.name == "action_intervals":
            base_risk = min(1.0, self.current_metrics.actions_per_minute / 100)
        
        return base_risk * pattern.risk_multiplier
    
    def _get_category_weight(self, category: BehaviorCategory) -> float:
        """Retourne le poids d'une catégorie pour le calcul global"""
        weights = {
            BehaviorCategory.TIMING: 2.0,
            BehaviorCategory.MOVEMENT: 1.8,
            BehaviorCategory.INTERACTION: 1.5,
            BehaviorCategory.PROGRESSION: 1.3,
            BehaviorCategory.SOCIAL: 1.0,
            BehaviorCategory.ECONOMIC: 1.2
        }
        return weights.get(category, 1.0)
    
    def _get_risk_level(self, risk_score: float) -> DetectionRisk:
        """Convertit un score de risque en niveau"""
        if risk_score >= 80:
            return DetectionRisk.CRITICAL
        elif risk_score >= 60:
            return DetectionRisk.HIGH
        elif risk_score >= 40:
            return DetectionRisk.MEDIUM
        else:
            return DetectionRisk.LOW
    
    def _activate_defensive_measures(self, risk_level: DetectionRisk):
        """Active des mesures défensives selon le niveau de risque"""
        if risk_level == DetectionRisk.HIGH:
            self.adaptive_factor = 1.3
            logger.warning("Risque élevé détecté - Augmentation de la variance comportementale")
            
        elif risk_level == DetectionRisk.CRITICAL:
            self.adaptive_factor = 1.6
            self.activate_stealth_mode(60)  # 1h de mode furtif
            logger.error("Risque critique détecté - Activation du mode furtif renforcé")
    
    def _calculate_pause_naturalness(self) -> float:
        """Calcule un score de naturalité des pauses"""
        if len(self.action_timestamps) < 10:
            return 0.5
        
        # Analyse des intervalles d'actions
        timestamps = list(self.action_timestamps)[-50:]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        # Détection des pauses (intervalles > 2 secondes)
        pauses = [interval for interval in intervals if interval > 2.0]
        
        if not intervals:
            return 0.0
        
        # Score basé sur la fréquence et la variabilité des pauses
        pause_ratio = len(pauses) / len(intervals)
        
        if pauses:
            pause_variance = np.std(pauses)
            # Plus il y a de variance dans les pauses, plus c'est naturel
            naturalness = min(1.0, pause_ratio * 2 + pause_variance / 10)
        else:
            # Aucune pause = peu naturel
            naturalness = 0.1
        
        return naturalness
    
    def _calculate_recent_error_rate(self) -> float:
        """Calcule le taux d'erreur récent"""
        # Simplification - à implémenter selon le tracking d'erreurs spécifique
        return 0.02  # 2% par défaut
    
    def _process_movement_event(self, data: Dict[str, Any]):
        """Traite un événement de mouvement"""
        if 'path' in data:
            self.path_segments.append({
                'timestamp': time.time(),
                'path': data['path'],
                'efficiency': data.get('efficiency', 0.8)
            })
    
    def _process_combat_event(self, data: Dict[str, Any]):
        """Traite un événement de combat"""
        # Analyse des patterns de combat pour la détection
        pass
    
    def _process_interaction_event(self, data: Dict[str, Any]):
        """Traite un événement d'interaction"""
        # Suivi de la diversité des interactions
        pass
    
    def _process_progression_event(self, data: Dict[str, Any]):
        """Traite un événement de progression"""
        # Analyse de la cohérence de progression
        pass
    
    def _load_configuration(self):
        """Charge la configuration depuis un fichier"""
        if not self.config_file:
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Chargement des patterns personnalisés
            if 'behavior_patterns' in config:
                for name, pattern_data in config['behavior_patterns'].items():
                    pattern = BehaviorPattern(
                        name=name,
                        category=BehaviorCategory(pattern_data['category']),
                        weight=pattern_data.get('weight', 1.0),
                        min_variance=pattern_data.get('min_variance', 0.1),
                        max_variance=pattern_data.get('max_variance', 0.3)
                    )
                    self.behavior_patterns[name] = pattern
            
        except Exception as e:
            logger.error(f"Erreur chargement configuration anti-détection: {e}")


# Configuration par défaut pour différents niveaux de prudence
STEALTH_CONFIGS = {
    "paranoid": {
        "base_error_rate": 0.04,
        "timing_variance": 0.6,
        "position_variance": 0.3,
        "pause_frequency": 0.25
    },
    "careful": {
        "base_error_rate": 0.02,
        "timing_variance": 0.4,
        "position_variance": 0.2,
        "pause_frequency": 0.15
    },
    "normal": {
        "base_error_rate": 0.015,
        "timing_variance": 0.3,
        "position_variance": 0.15,
        "pause_frequency": 0.1
    },
    "bold": {
        "base_error_rate": 0.01,
        "timing_variance": 0.2,
        "position_variance": 0.1,
        "pause_frequency": 0.05
    }
}


if __name__ == "__main__":
    # Test du système anti-détection
    logging.basicConfig(level=logging.DEBUG)
    
    system = AntiDetectionSystem()
    
    print("=== Test du système anti-détection ===")
    
    # Simulation d'activité
    for i in range(20):
        # Enregistrement d'actions
        system.record_game_event("movement", {"path": [(i*10, i*10), ((i+1)*10, (i+1)*10)]})
        
        # Test des variations
        timing_var = system.get_human_timing_variation(1.0, "click")
        pos_var = system.get_mouse_position_variation((100, 100))
        
        print(f"Action {i+1}: timing={timing_var:.3f}s, pos_variation={pos_var}")
        
        time.sleep(0.1)  # Simulation
    
    # Analyse finale
    risk = system.analyze_current_behavior()
    recommendations = system.get_stealth_recommendations()
    
    print(f"\nRisque de détection: {risk.value}")
    print("Recommandations:")
    for rec in recommendations:
        print(f"  - {rec}")