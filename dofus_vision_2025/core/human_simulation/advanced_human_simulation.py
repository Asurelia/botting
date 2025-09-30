"""
Simulation Humaine Perfectionnée pour DOFUS Unity
Système avancé de simulation des interactions humaines avec patterns naturels
Anti-détection optimisé pour gaming compétitif
"""

import random
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import math

@dataclass
class HumanProfile:
    """Profil de joueur humain avec caractéristiques uniques"""
    reaction_time_base: float  # Temps de réaction de base (ms)
    reaction_time_variance: float  # Variance du temps de réaction
    click_precision_level: float  # Précision des clics (0.0-1.0)
    movement_style: str  # Style de mouvement ("aggressive", "cautious", "balanced")
    fatigue_resistance: float  # Résistance à la fatigue (0.0-1.0)
    micro_pause_frequency: float  # Fréquence des micro-pauses
    preferred_spell_timing: Dict[str, float]  # Timing préféré par sort
    keyboard_rhythm: Dict[str, float]  # Rythme de frappe par touche

class AdvancedHumanSimulator:
    """Simulateur humain avancé avec patterns comportementaux réalistes"""

    def __init__(self):
        self.current_profile = self._generate_random_profile()
        self.session_start_time = time.time()
        self.total_actions = 0
        self.fatigue_level = 0.0

        # Historique des actions pour patterns
        self.action_history: List[Dict] = []
        self.timing_patterns: Dict[str, List[float]] = {}

        # Coordonnées de la dernière action
        self.last_mouse_pos = (0, 0)
        self.last_action_time = time.time()

        # Patterns de sessions humaines
        self.session_patterns = self._load_session_patterns()

    def _generate_random_profile(self) -> HumanProfile:
        """Génère un profil de joueur humain réaliste"""

        # Styles de jeu différents
        styles = ["aggressive", "cautious", "balanced", "tryhard", "casual"]
        style = random.choice(styles)

        if style == "aggressive":
            return HumanProfile(
                reaction_time_base=random.uniform(180, 250),
                reaction_time_variance=random.uniform(30, 60),
                click_precision_level=random.uniform(0.85, 0.95),
                movement_style="aggressive",
                fatigue_resistance=random.uniform(0.7, 0.9),
                micro_pause_frequency=random.uniform(0.02, 0.05),
                preferred_spell_timing={
                    "instant": random.uniform(0.1, 0.3),
                    "aimed": random.uniform(0.3, 0.6),
                    "strategic": random.uniform(0.5, 1.0)
                },
                keyboard_rhythm={
                    "spell_keys": random.uniform(0.08, 0.15),
                    "movement": random.uniform(0.05, 0.12),
                    "interface": random.uniform(0.12, 0.25)
                }
            )
        elif style == "cautious":
            return HumanProfile(
                reaction_time_base=random.uniform(300, 450),
                reaction_time_variance=random.uniform(40, 80),
                click_precision_level=random.uniform(0.92, 0.98),
                movement_style="cautious",
                fatigue_resistance=random.uniform(0.8, 0.95),
                micro_pause_frequency=random.uniform(0.08, 0.15),
                preferred_spell_timing={
                    "instant": random.uniform(0.4, 0.8),
                    "aimed": random.uniform(0.8, 1.5),
                    "strategic": random.uniform(1.2, 2.5)
                },
                keyboard_rhythm={
                    "spell_keys": random.uniform(0.15, 0.25),
                    "movement": random.uniform(0.12, 0.20),
                    "interface": random.uniform(0.20, 0.35)
                }
            )
        else:  # balanced, tryhard, casual
            return HumanProfile(
                reaction_time_base=random.uniform(220, 320),
                reaction_time_variance=random.uniform(35, 70),
                click_precision_level=random.uniform(0.88, 0.94),
                movement_style="balanced",
                fatigue_resistance=random.uniform(0.75, 0.88),
                micro_pause_frequency=random.uniform(0.05, 0.10),
                preferred_spell_timing={
                    "instant": random.uniform(0.2, 0.5),
                    "aimed": random.uniform(0.5, 1.0),
                    "strategic": random.uniform(0.8, 1.8)
                },
                keyboard_rhythm={
                    "spell_keys": random.uniform(0.10, 0.18),
                    "movement": random.uniform(0.08, 0.15),
                    "interface": random.uniform(0.15, 0.28)
                }
            )

    def _load_session_patterns(self) -> Dict:
        """Charge les patterns de sessions humaines réalistes"""
        return {
            "peak_performance_periods": [
                (10*60, 45*60),    # 10-45 min après début
                (90*60, 120*60),   # Après pause
                (180*60, 200*60)   # Regain après fatigue
            ],
            "fatigue_curves": {
                "light_session": lambda t: min(0.3, t / (2 * 3600)),  # 2h max
                "normal_session": lambda t: min(0.7, t / (4 * 3600)),  # 4h max
                "intense_session": lambda t: min(1.0, t / (6 * 3600))  # 6h max
            },
            "break_patterns": {
                "micro_break": (3, 8),      # 3-8 secondes
                "small_break": (30, 90),    # 30s-1.5min
                "medium_break": (300, 900), # 5-15 min
                "long_break": (1800, 3600)  # 30min-1h
            }
        }

    def calculate_human_delay(self, action_type: str, complexity: float = 1.0) -> float:
        """Calcule le délai humain réaliste pour une action"""

        # Temps de base selon le profil
        base_time = self.current_profile.reaction_time_base / 1000.0

        # Variance naturelle
        variance = np.random.normal(0, self.current_profile.reaction_time_variance / 1000.0)

        # Facteur de complexité
        complexity_factor = 1.0 + (complexity - 1.0) * 0.3

        # Fatigue progressive
        fatigue_factor = 1.0 + self.fatigue_level * 0.4

        # Timing spécialisé par action
        action_modifiers = {
            "spell_cast": self.current_profile.preferred_spell_timing.get("instant", 0.2),
            "targeted_spell": self.current_profile.preferred_spell_timing.get("aimed", 0.5),
            "movement": 0.1,
            "item_use": 0.15,
            "interface_click": 0.25,
            "strategic_pause": self.current_profile.preferred_spell_timing.get("strategic", 1.0)
        }

        action_modifier = action_modifiers.get(action_type, 0.2)

        # Calcul final
        total_delay = (base_time + variance) * complexity_factor * fatigue_factor + action_modifier

        # Assurer un minimum réaliste
        return max(0.05, total_delay)

    def generate_mouse_movement(self, start_pos: Tuple[int, int],
                              target_pos: Tuple[int, int],
                              precision_required: float = 0.9) -> List[Tuple[int, int, float]]:
        """Génère un mouvement de souris humain naturel"""

        start_x, start_y = start_pos
        target_x, target_y = target_pos

        distance = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)

        # Nombre de points intermédiaires selon la distance
        num_points = max(3, int(distance / 50) + random.randint(1, 3))

        # Génération de la courbe de Bézier humaine
        points = []

        # Point de départ
        points.append((start_x, start_y, 0.0))

        # Points intermédiaires avec tremblements naturels
        for i in range(1, num_points):
            progress = i / num_points

            # Interpolation de base
            base_x = start_x + (target_x - start_x) * progress
            base_y = start_y + (target_y - start_y) * progress

            # Ajout de variations humaines
            jitter_strength = (1.0 - self.current_profile.click_precision_level) * 3.0
            jitter_x = random.uniform(-jitter_strength, jitter_strength)
            jitter_y = random.uniform(-jitter_strength, jitter_strength)

            # Courbe naturelle (légère overshoot au milieu)
            if 0.3 < progress < 0.7:
                curve_factor = math.sin(progress * math.pi) * 0.1
                perpendicular_x = -(target_y - start_y) / distance * curve_factor * distance * 0.1
                perpendicular_y = (target_x - start_x) / distance * curve_factor * distance * 0.1
            else:
                perpendicular_x = perpendicular_y = 0

            final_x = int(base_x + jitter_x + perpendicular_x)
            final_y = int(base_y + jitter_y + perpendicular_y)

            # Timing entre les points
            point_delay = random.uniform(0.008, 0.020)  # 8-20ms entre points

            points.append((final_x, final_y, point_delay))

        # Point final avec petite correction de précision
        precision_error = (1.0 - precision_required) * 5.0
        final_error_x = random.uniform(-precision_error, precision_error)
        final_error_y = random.uniform(-precision_error, precision_error)

        final_x = int(target_x + final_error_x)
        final_y = int(target_y + final_error_y)

        points.append((final_x, final_y, random.uniform(0.01, 0.03)))

        return points

    def should_take_break(self) -> Optional[Tuple[str, float]]:
        """Détermine si le joueur devrait prendre une pause"""

        current_time = time.time()
        session_duration = current_time - self.session_start_time

        # Vérification des patterns de pause naturels
        if self.total_actions > 0:
            # Micro-pauses fréquentes
            if random.random() < self.current_profile.micro_pause_frequency / 100:
                duration = random.uniform(*self.session_patterns["break_patterns"]["micro_break"])
                return ("micro", duration)

            # Pauses selon la fatigue
            fatigue_break_probability = self.fatigue_level * 0.002  # 0.2% par point de fatigue
            if random.random() < fatigue_break_probability:
                if self.fatigue_level < 0.3:
                    duration = random.uniform(*self.session_patterns["break_patterns"]["small_break"])
                    return ("small", duration)
                elif self.fatigue_level < 0.7:
                    duration = random.uniform(*self.session_patterns["break_patterns"]["medium_break"])
                    return ("medium", duration)
                else:
                    duration = random.uniform(*self.session_patterns["break_patterns"]["long_break"])
                    return ("long", duration)

        # Pauses naturelles selon temps de session
        if session_duration > 3600:  # Après 1h
            if random.random() < 0.001:  # 0.1% chance par action
                duration = random.uniform(*self.session_patterns["break_patterns"]["medium_break"])
                return ("scheduled", duration)

        return None

    def update_fatigue(self, action_intensity: float = 1.0):
        """Met à jour le niveau de fatigue du joueur"""

        session_duration = time.time() - self.session_start_time

        # Augmentation de fatigue basée sur durée et intensité
        base_fatigue_increase = action_intensity * 0.0001  # Base très faible
        time_fatigue = session_duration * 0.00002  # Fatigue temporelle

        # Facteur de résistance personnel
        resistance_factor = self.current_profile.fatigue_resistance

        fatigue_increase = (base_fatigue_increase + time_fatigue) / resistance_factor

        self.fatigue_level = min(1.0, self.fatigue_level + fatigue_increase)

        # Récupération durant les pics de performance
        for start, end in self.session_patterns["peak_performance_periods"]:
            if start <= session_duration <= end:
                self.fatigue_level *= 0.998  # Légère récupération
                break

    def generate_keyboard_rhythm(self, key_sequence: List[str]) -> List[Tuple[str, float]]:
        """Génère un rythme de frappe humain pour une séquence de touches"""

        timed_sequence = []

        for i, key in enumerate(key_sequence):
            # Catégorisation de la touche
            if key in "123456789":
                category = "spell_keys"
            elif key in "wasd":
                category = "movement"
            else:
                category = "interface"

            # Temps de base pour cette catégorie
            base_time = self.current_profile.keyboard_rhythm[category]

            # Variation humaine
            variation = random.uniform(-0.03, 0.03)

            # Effet de fatigue
            fatigue_factor = 1.0 + self.fatigue_level * 0.2

            # Pattern de frappe (plus rapide pour les combinaisons habituelles)
            if i > 0 and key == key_sequence[i-1]:
                base_time *= 0.8  # Répétition plus rapide

            final_time = (base_time + variation) * fatigue_factor
            final_time = max(0.05, final_time)  # Minimum humain

            timed_sequence.append((key, final_time))

        return timed_sequence

    def simulate_spell_casting_sequence(self, spell_name: str, target_pos: Optional[Tuple[int, int]] = None) -> Dict:
        """Simule la séquence complète de lancement d'un sort"""

        sequence = {
            "preparation_delay": self.calculate_human_delay("strategic_pause", 0.5),
            "key_press_timing": None,
            "mouse_movement": None,
            "cast_delay": self.calculate_human_delay("spell_cast"),
            "confirmation_delay": self.calculate_human_delay("interface_click", 0.3)
        }

        # Simulation de la pression de touche du sort
        spell_key = spell_name[0].lower()  # Première lettre comme raccourci
        sequence["key_press_timing"] = self.generate_keyboard_rhythm([spell_key])[0]

        # Si le sort nécessite un ciblage
        if target_pos:
            mouse_movement = self.generate_mouse_movement(
                self.last_mouse_pos,
                target_pos,
                precision_required=0.95
            )
            sequence["mouse_movement"] = mouse_movement
            sequence["targeting_delay"] = self.calculate_human_delay("targeted_spell", 1.2)

            # Mise à jour de la position de souris
            self.last_mouse_pos = target_pos

        # Mise à jour des statistiques
        self.total_actions += 1
        self.update_fatigue(1.0)
        self.last_action_time = time.time()

        # Enregistrement dans l'historique
        self.action_history.append({
            "type": "spell_cast",
            "spell": spell_name,
            "timestamp": time.time(),
            "sequence": sequence
        })

        return sequence

    def export_session_analysis(self, filepath: str):
        """Exporte une analyse de la session pour amélioration"""

        session_data = {
            "profile": {
                "movement_style": self.current_profile.movement_style,
                "reaction_time_base": self.current_profile.reaction_time_base,
                "click_precision": self.current_profile.click_precision_level,
                "fatigue_resistance": self.current_profile.fatigue_resistance
            },
            "session_stats": {
                "total_actions": self.total_actions,
                "session_duration": time.time() - self.session_start_time,
                "final_fatigue_level": self.fatigue_level,
                "average_action_interval": len(self.action_history) / max(1, (time.time() - self.session_start_time))
            },
            "patterns_detected": {
                "most_used_spells": self._analyze_spell_usage(),
                "timing_consistency": self._analyze_timing_consistency(),
                "break_patterns": self._analyze_break_patterns()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

    def _analyze_spell_usage(self) -> Dict:
        """Analyse l'usage des sorts durant la session"""
        spell_counts = {}
        for action in self.action_history:
            if action["type"] == "spell_cast":
                spell = action["spell"]
                spell_counts[spell] = spell_counts.get(spell, 0) + 1

        return dict(sorted(spell_counts.items(), key=lambda x: x[1], reverse=True))

    def _analyze_timing_consistency(self) -> float:
        """Analyse la cohérence du timing (plus cohérent = plus humain)"""
        if len(self.action_history) < 3:
            return 0.5

        intervals = []
        for i in range(1, len(self.action_history)):
            interval = self.action_history[i]["timestamp"] - self.action_history[i-1]["timestamp"]
            intervals.append(interval)

        if not intervals:
            return 0.5

        # Coefficient de variation (plus bas = plus cohérent)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv = std_interval / mean_interval if mean_interval > 0 else 1.0

        # Normalisation (0.2-0.8 est un range humain typique)
        consistency = max(0.0, min(1.0, 1.0 - cv))
        return consistency

    def _analyze_break_patterns(self) -> Dict:
        """Analyse les patterns de pauses"""
        # Détection des gaps importants dans l'historique
        breaks = []
        for i in range(1, len(self.action_history)):
            gap = self.action_history[i]["timestamp"] - self.action_history[i-1]["timestamp"]
            if gap > 5.0:  # Gap de plus de 5 secondes
                breaks.append(gap)

        return {
            "total_breaks": len(breaks),
            "average_break_duration": np.mean(breaks) if breaks else 0,
            "longest_break": max(breaks) if breaks else 0
        }

# Fonctions utilitaires pour intégration
def get_human_simulator() -> AdvancedHumanSimulator:
    """Retourne une instance globale du simulateur humain"""
    if not hasattr(get_human_simulator, '_instance'):
        get_human_simulator._instance = AdvancedHumanSimulator()
    return get_human_simulator._instance

def simulate_human_action(action_type: str, **kwargs) -> Dict:
    """Interface simplifiée pour simuler une action humaine"""
    simulator = get_human_simulator()

    if action_type == "spell_cast":
        return simulator.simulate_spell_casting_sequence(
            kwargs.get("spell_name", "Sort"),
            kwargs.get("target_pos")
        )
    elif action_type == "movement":
        start_pos = kwargs.get("start_pos", simulator.last_mouse_pos)
        target_pos = kwargs.get("target_pos", (100, 100))
        return {
            "mouse_movement": simulator.generate_mouse_movement(start_pos, target_pos),
            "movement_delay": simulator.calculate_human_delay("movement")
        }
    elif action_type == "interface_click":
        return {
            "click_delay": simulator.calculate_human_delay("interface_click"),
            "precision": simulator.current_profile.click_precision_level
        }
    else:
        return {
            "generic_delay": simulator.calculate_human_delay(action_type),
            "fatigue_level": simulator.fatigue_level
        }

if __name__ == "__main__":
    # Test du système
    print("[TEST] Simulation Humaine Perfectionnée")

    simulator = AdvancedHumanSimulator()
    print(f"Profil généré: {simulator.current_profile.movement_style}")
    print(f"Temps de réaction base: {simulator.current_profile.reaction_time_base:.0f}ms")

    # Test de lancement de sort
    spell_sequence = simulator.simulate_spell_casting_sequence("Pression", (150, 200))
    print(f"Séquence sort: {spell_sequence}")

    # Test de mouvement de souris
    movement = simulator.generate_mouse_movement((0, 0), (200, 150))
    print(f"Mouvement souris: {len(movement)} points")

    # Simulation de fatigue
    for i in range(100):
        simulator.update_fatigue(1.0)
    print(f"Fatigue après 100 actions: {simulator.fatigue_level:.3f}")

    print("[OK] Tests simulation humaine réussis")