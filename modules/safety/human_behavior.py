"""
Module de simulation du comportement humain
Simule des actions réalistes avec délais variables, erreurs et pauses naturelles
"""

import random
import time
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types d'actions pour personnaliser les comportements"""
    MOVEMENT = "movement"
    CLICK = "click"
    TYPING = "typing"
    COMBAT = "combat"
    PROFESSION = "profession"
    MENU = "menu"


@dataclass
class HumanProfile:
    """Profil comportemental humain personnalisable"""
    reaction_time_base: float = 0.3  # Temps de réaction de base (secondes)
    reaction_variance: float = 0.2   # Variance du temps de réaction
    error_rate: float = 0.02         # Taux d'erreur (2%)
    fatigue_factor: float = 0.001    # Facteur de fatigue par action
    pause_frequency: float = 0.1     # Fréquence des pauses (10%)
    typing_wpm: int = 45             # Mots par minute de frappe
    double_click_chance: float = 0.05 # Chance de double-clic accidentel
    misclick_chance: float = 0.03     # Chance de clic raté
    hesitation_chance: float = 0.08   # Chance d'hésitation


class HumanBehaviorSimulator:
    """
    Simulateur de comportement humain réaliste
    Ajoute des délais naturels, des erreurs occasionnelles et des patterns humains
    """
    
    def __init__(self, profile: Optional[HumanProfile] = None):
        self.profile = profile or HumanProfile()
        self.session_start_time = time.time()
        self.actions_performed = 0
        self.current_fatigue = 0.0
        self.last_action_time = time.time()
        self.error_history = []
        self.pause_history = []
        
        # Patterns comportementaux
        self.action_patterns = {
            ActionType.MOVEMENT: {"min_delay": 0.1, "max_delay": 0.4, "variance": 0.3},
            ActionType.CLICK: {"min_delay": 0.05, "max_delay": 0.2, "variance": 0.15},
            ActionType.TYPING: {"min_delay": 0.08, "max_delay": 0.3, "variance": 0.2},
            ActionType.COMBAT: {"min_delay": 0.2, "max_delay": 0.8, "variance": 0.4},
            ActionType.PROFESSION: {"min_delay": 0.3, "max_delay": 1.2, "variance": 0.5},
            ActionType.MENU: {"min_delay": 0.15, "max_delay": 0.6, "variance": 0.25}
        }
    
    def get_human_delay(self, action_type: ActionType) -> float:
        """
        Calcule un délai humain réaliste basé sur le type d'action et l'état actuel
        """
        pattern = self.action_patterns[action_type]
        
        # Délai de base avec variance
        base_delay = random.normalvariate(
            self.profile.reaction_time_base,
            self.profile.reaction_variance
        )
        
        # Ajustement selon le type d'action
        action_delay = random.uniform(pattern["min_delay"], pattern["max_delay"])
        
        # Facteur de fatigue progressif
        fatigue_multiplier = 1.0 + self.current_fatigue
        
        # Variation naturelle
        variance = random.normalvariate(1.0, pattern["variance"])
        variance = max(0.1, variance)  # Éviter les valeurs négatives
        
        # Calcul final avec tous les facteurs
        total_delay = (base_delay + action_delay) * fatigue_multiplier * variance
        
        # Assurer un délai minimum réaliste
        return max(0.05, total_delay)
    
    def should_make_error(self, action_type: ActionType) -> bool:
        """
        Détermine si une erreur humaine doit être simulée
        """
        # Augmentation du taux d'erreur avec la fatigue
        adjusted_error_rate = self.profile.error_rate * (1.0 + self.current_fatigue)
        
        # Facteur spécifique au type d'action
        type_multiplier = {
            ActionType.TYPING: 2.0,      # Plus d'erreurs en tapant
            ActionType.CLICK: 1.5,       # Erreurs de clic modérées
            ActionType.MOVEMENT: 1.2,    # Mouvements parfois imprécis
            ActionType.COMBAT: 0.8,      # Plus concentré en combat
            ActionType.PROFESSION: 1.0,  # Erreurs normales
            ActionType.MENU: 0.9         # Menus familiers
        }.get(action_type, 1.0)
        
        final_error_rate = adjusted_error_rate * type_multiplier
        return random.random() < final_error_rate
    
    def should_pause(self) -> bool:
        """
        Détermine si une pause naturelle doit être prise
        """
        # Facteur de temps écoulé depuis la dernière action
        time_factor = min(time.time() - self.last_action_time, 300) / 300  # Max 5min
        
        # Augmentation de la probabilité de pause avec la fatigue
        pause_chance = self.profile.pause_frequency * (1.0 + self.current_fatigue + time_factor)
        
        return random.random() < pause_chance
    
    def get_pause_duration(self) -> float:
        """
        Calcule une durée de pause naturelle
        """
        # Pauses courtes fréquentes, pauses longues rares
        if random.random() < 0.7:  # 70% pauses courtes
            return random.uniform(0.5, 3.0)
        elif random.random() < 0.9:  # 20% pauses moyennes
            return random.uniform(3.0, 8.0)
        else:  # 10% pauses longues
            return random.uniform(8.0, 20.0)
    
    def simulate_typing_delay(self, text: str) -> List[float]:
        """
        Simule des délais de frappe réalistes pour un texte
        """
        delays = []
        wpm = self.profile.typing_wpm
        base_delay = 60.0 / (wpm * 5)  # 5 caractères par mot en moyenne
        
        for i, char in enumerate(text):
            # Délai de base
            delay = base_delay
            
            # Caractères spéciaux plus lents
            if char.isupper():
                delay *= 1.3  # Majuscules (Shift)
            elif char in "!@#$%^&*()_+-=[]{}|;':\",./<>?":
                delay *= 1.5  # Caractères spéciaux
            elif char == ' ':
                delay *= 0.8  # Espaces plus rapides
            
            # Variation naturelle
            delay *= random.normalvariate(1.0, 0.3)
            delay = max(0.05, delay)  # Minimum 50ms
            
            # Erreurs de frappe occasionnelles
            if self.should_make_error(ActionType.TYPING):
                delay *= 2.0  # Temps pour corriger
            
            delays.append(delay)
        
        return delays
    
    def simulate_mouse_movement_delay(self, distance: float) -> float:
        """
        Simule le délai de mouvement de souris basé sur la distance
        """
        # Loi de Fitts simplifiée : T = a + b * log2(D/W + 1)
        # où D est la distance et W la largeur de la cible
        a, b = 0.1, 0.05  # Constantes empiriques
        target_width = 20  # Largeur de cible moyenne en pixels
        
        fitts_delay = a + b * np.log2(max(distance, 1) / target_width + 1)
        
        # Ajout de variance naturelle
        variance_factor = random.normalvariate(1.0, 0.2)
        total_delay = fitts_delay * variance_factor * (1.0 + self.current_fatigue)
        
        return max(0.05, total_delay)
    
    def simulate_hesitation(self) -> bool:
        """
        Simule une hésitation humaine (arrêt momentané)
        """
        if random.random() < self.profile.hesitation_chance:
            hesitation_time = random.uniform(0.3, 1.5)
            logger.debug(f"Hésitation simulée: {hesitation_time:.2f}s")
            time.sleep(hesitation_time)
            return True
        return False
    
    def perform_action(self, action_type: ActionType, action_func: Callable, *args, **kwargs):
        """
        Exécute une action avec simulation comportementale humaine complète
        """
        logger.debug(f"Début action {action_type.value}")
        
        # Hésitation occasionnelle avant l'action
        hesitated = self.simulate_hesitation()
        
        # Pause naturelle si nécessaire
        if self.should_pause():
            pause_duration = self.get_pause_duration()
            logger.info(f"Pause naturelle: {pause_duration:.1f}s")
            time.sleep(pause_duration)
            self.pause_history.append({
                'time': time.time(),
                'duration': pause_duration,
                'reason': 'natural'
            })
        
        # Délai pré-action
        delay = self.get_human_delay(action_type)
        logger.debug(f"Délai pré-action: {delay:.3f}s")
        time.sleep(delay)
        
        # Simulation d'erreur si nécessaire
        error_occurred = False
        if self.should_make_error(action_type):
            error_occurred = True
            logger.debug(f"Erreur simulée pour {action_type.value}")
            self.error_history.append({
                'time': time.time(),
                'action_type': action_type.value,
                'fatigue_level': self.current_fatigue
            })
            
            # Délai de correction d'erreur
            correction_delay = random.uniform(0.5, 2.0)
            time.sleep(correction_delay)
        
        # Exécution de l'action réelle
        try:
            result = action_func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Erreur lors de l'action {action_type.value}: {e}")
            result = None
            success = False
        
        # Mise à jour de l'état
        self.actions_performed += 1
        self.current_fatigue += self.profile.fatigue_factor
        self.last_action_time = time.time()
        
        # Délai post-action
        post_delay = random.uniform(0.05, 0.2)
        time.sleep(post_delay)
        
        logger.debug(f"Action {action_type.value} terminée (succès: {success}, erreur: {error_occurred})")
        
        return {
            'result': result,
            'success': success,
            'error_simulated': error_occurred,
            'hesitated': hesitated,
            'delay_used': delay,
            'fatigue_level': self.current_fatigue
        }
    
    def get_session_stats(self) -> Dict:
        """
        Retourne les statistiques de la session actuelle
        """
        session_duration = time.time() - self.session_start_time
        
        return {
            'session_duration': session_duration,
            'actions_performed': self.actions_performed,
            'current_fatigue': self.current_fatigue,
            'errors_made': len(self.error_history),
            'pauses_taken': len(self.pause_history),
            'average_error_rate': len(self.error_history) / max(self.actions_performed, 1),
            'actions_per_minute': self.actions_performed / (session_duration / 60) if session_duration > 0 else 0
        }
    
    def reset_fatigue(self):
        """
        Remet à zéro la fatigue (après une pause longue)
        """
        logger.info("Fatigue remise à zéro")
        self.current_fatigue = 0.0
        self.error_history.clear()
        self.pause_history.clear()
    
    def adjust_profile(self, **kwargs):
        """
        Ajuste dynamiquement le profil comportemental
        """
        for key, value in kwargs.items():
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)
                logger.info(f"Profil ajusté: {key} = {value}")
            else:
                logger.warning(f"Attribut de profil inconnu: {key}")


# Profils prédéfinis pour différents types d'utilisateurs
BEGINNER_PROFILE = HumanProfile(
    reaction_time_base=0.5,
    reaction_variance=0.3,
    error_rate=0.05,
    pause_frequency=0.15,
    typing_wpm=25,
    hesitation_chance=0.12
)

EXPERIENCED_PROFILE = HumanProfile(
    reaction_time_base=0.25,
    reaction_variance=0.15,
    error_rate=0.015,
    pause_frequency=0.08,
    typing_wpm=60,
    hesitation_chance=0.05
)

CASUAL_PROFILE = HumanProfile(
    reaction_time_base=0.4,
    reaction_variance=0.25,
    error_rate=0.03,
    pause_frequency=0.2,
    typing_wpm=35,
    hesitation_chance=0.15
)


if __name__ == "__main__":
    # Test du simulateur
    simulator = HumanBehaviorSimulator(CASUAL_PROFILE)
    
    # Fonction de test
    def test_action():
        print("Action exécutée!")
        return "success"
    
    # Test de plusieurs actions
    for i in range(5):
        result = simulator.perform_action(ActionType.CLICK, test_action)
        print(f"Résultat {i+1}: {result}")
    
    # Affichage des statistiques
    stats = simulator.get_session_stats()
    print("\nStatistiques de session:")
    for key, value in stats.items():
        print(f"  {key}: {value}")