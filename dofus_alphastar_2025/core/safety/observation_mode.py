#!/usr/bin/env python3
"""
ObservationMode - Mode observation SÉCURISÉ
Le bot OBSERVE sans JAMAIS agir - CRUCIAL pour tests initiaux

[WARNING] IMPORTANT: Mode OBLIGATOIRE pour premier test sur compte jetable
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

@dataclass
class ObservationLog:
    """Log d'une décision observée"""
    timestamp: float
    action_type: str
    action_details: Dict[str, Any]
    game_state: Dict[str, Any]
    decision_reason: str
    would_execute: bool = False

class ObservationMode:
    """
    Mode Observation - Le bot observe mais N'AGIT JAMAIS

    **RÈGLE D'OR**: Toutes les actions sont interceptées et bloquées
    **UTILISATION**: OBLIGATOIRE pour premiers tests

    Features:
    - Intercepte 100% des actions
    - Log toutes les décisions
    - Analyse l'état du jeu
    - Aucune interaction avec Dofus
    - Export des logs pour analyse
    """

    def __init__(self, log_file: str = "logs/observation.json", auto_enabled: bool = True):
        self.logger = logging.getLogger(__name__)

        # Mode OBSERVATION ACTIVÉ PAR DÉFAUT (sécurité!)
        self.enabled = auto_enabled

        # Configuration
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Logs des décisions
        self.observations: List[ObservationLog] = []

        # Statistiques
        self.stats = {
            'total_decisions': 0,
            'actions_blocked': 0,
            'keyboard_inputs': 0,
            'mouse_clicks': 0,
            'navigation_attempts': 0,
            'combat_actions': 0,
            'start_time': time.time()
        }

        if self.enabled:
            self.logger.warning("=" * 70)
            self.logger.warning("[WARNING]  MODE OBSERVATION ACTIVÉ")
            self.logger.warning("[WARNING]  AUCUNE ACTION NE SERA EXÉCUTÉE")
            self.logger.warning("[WARNING]  Le bot observe seulement")
            self.logger.warning("=" * 70)

    def intercept_action(self, action_type: str, action_details: Dict[str, Any],
                        game_state: Dict[str, Any], reason: str = "") -> Optional[Any]:
        """
        Intercepte TOUTES les actions

        Args:
            action_type: Type d'action (move, click, spell, etc.)
            action_details: Détails de l'action
            game_state: État actuel du jeu
            reason: Raison de la décision

        Returns:
            None si mode observation (bloque l'action)
            action_details sinon (laisse passer)
        """
        self.stats['total_decisions'] += 1

        # Log l'observation
        observation = ObservationLog(
            timestamp=time.time(),
            action_type=action_type,
            action_details=action_details,
            game_state=game_state,
            decision_reason=reason,
            would_execute=not self.enabled
        )

        self.observations.append(observation)

        # Catégorise pour stats
        self._categorize_action(action_type)

        if self.enabled:
            # MODE OBSERVATION: Bloque l'action
            self.stats['actions_blocked'] += 1

            self.logger.debug(f"[OBSERVATION] {action_type}: {action_details}")
            self.logger.debug(f"  Raison: {reason}")
            self.logger.debug(f"  [ERROR] ACTION BLOQUÉE")

            return None  # Aucune action n'est exécutée

        else:
            # Mode normal: Laisse passer
            self.logger.debug(f"[EXECUTE] {action_type}")
            return action_details

    def _categorize_action(self, action_type: str):
        """Catégorise l'action pour statistiques"""
        if action_type in ['key_press', 'key_release', 'type_text']:
            self.stats['keyboard_inputs'] += 1

        elif action_type in ['mouse_click', 'mouse_move']:
            self.stats['mouse_clicks'] += 1

        elif action_type in ['navigate', 'navigation', 'pathfind', 'move_to']:
            self.stats['navigation_attempts'] += 1

        elif action_type in ['cast_spell', 'spell_cast', 'attack', 'use_item']:
            self.stats['combat_actions'] += 1

    def log_decision(self, action_type: str, details: Dict, game_state: Dict, reason: str):
        """
        Log une décision (alias de intercept_action pour compatibilité)
        """
        return self.intercept_action(action_type, details, game_state, reason)

    def get_observations(self, limit: Optional[int] = None) -> List[ObservationLog]:
        """
        Récupère les observations

        Args:
            limit: Nombre max d'observations (None = toutes)

        Returns:
            Liste d'observations
        """
        if limit:
            return self.observations[-limit:]
        return self.observations

    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques"""
        duration = time.time() - self.stats['start_time']

        return {
            **self.stats,
            'duration_seconds': duration,
            'observations_count': len(self.observations),
            'actions_per_minute': (self.stats['total_decisions'] / duration * 60) if duration > 0 else 0
        }

    def save_observations(self, output_file: Optional[str] = None):
        """
        Sauvegarde toutes les observations en JSON

        Args:
            output_file: Fichier de sortie (None = fichier par défaut)
        """
        output_path = Path(output_file) if output_file else self.log_file

        data = {
            'mode': 'observation',
            'enabled': self.enabled,
            'start_time': self.stats['start_time'],
            'end_time': time.time(),
            'statistics': self.get_stats(),
            'observations': [asdict(obs) for obs in self.observations]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"[OK] Observations sauvegardées: {output_path}")
        self.logger.info(f"  Total: {len(self.observations)} décisions loggées")

    def analyze_observations(self) -> Dict[str, Any]:
        """
        Analyse les observations pour insights

        Returns:
            Rapport d'analyse
        """
        if not self.observations:
            return {
                'error': 'Aucune observation à analyser'
            }

        # Compte par type d'action
        action_types = {}
        for obs in self.observations:
            action_types[obs.action_type] = action_types.get(obs.action_type, 0) + 1

        # Top 5 actions
        top_actions = sorted(action_types.items(), key=lambda x: x[1], reverse=True)[:5]

        # Analyse temporelle
        first_obs = self.observations[0].timestamp
        last_obs = self.observations[-1].timestamp
        duration = last_obs - first_obs

        # Patterns
        patterns = self._detect_patterns()

        report = {
            'total_observations': len(self.observations),
            'duration_seconds': duration,
            'actions_per_second': len(self.observations) / duration if duration > 0 else 0,
            'action_types': action_types,
            'top_actions': dict(top_actions),
            'patterns_detected': patterns,
            'safety_score': self._compute_safety_score(),
            'recommendations': self._generate_recommendations()
        }

        return report

    def _detect_patterns(self) -> Dict[str, Any]:
        """Détecte des patterns dans les observations"""
        patterns = {
            'repetitive_actions': False,
            'suspicious_timing': False,
            'bot_like_behavior': False
        }

        if len(self.observations) < 10:
            return patterns

        # Détection d'actions répétitives
        recent_actions = [obs.action_type for obs in self.observations[-20:]]
        if len(set(recent_actions)) < 3:
            patterns['repetitive_actions'] = True

        # TODO: Analyser timing, séquences suspectes, etc.

        return patterns

    def _compute_safety_score(self) -> float:
        """
        Calcule un score de sécurité (0-100)

        Plus le score est bas, plus le comportement est suspect
        """
        score = 100.0

        patterns = self._detect_patterns()

        if patterns['repetitive_actions']:
            score -= 20

        if patterns['suspicious_timing']:
            score -= 30

        if patterns['bot_like_behavior']:
            score -= 40

        # Actions trop rapides
        stats = self.get_stats()
        if stats['actions_per_minute'] > 60:
            score -= 10

        return max(0, score)

    def _generate_recommendations(self) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []

        patterns = self._detect_patterns()
        safety_score = self._compute_safety_score()

        if safety_score < 50:
            recommendations.append("[WARNING] Score de sécurité faible - Comportement potentiellement détectable")

        if patterns['repetitive_actions']:
            recommendations.append("Ajouter plus de variabilité dans les actions")

        if patterns['suspicious_timing']:
            recommendations.append("Randomiser les délais entre actions")

        stats = self.get_stats()
        if stats['actions_per_minute'] > 60:
            recommendations.append("Réduire la vitesse d'exécution (trop d'actions/minute)")

        if not recommendations:
            recommendations.append("[OK] Comportement semble naturel")

        return recommendations

    def print_report(self):
        """Affiche un rapport d'analyse dans la console"""
        print("\n" + "=" * 70)
        print("RAPPORT D'OBSERVATION")
        print("=" * 70)

        stats = self.get_stats()
        print(f"\nStatistiques:")
        print(f"  Duree: {stats['duration_seconds']:.1f}s")
        print(f"  Decisions: {stats['total_decisions']}")
        print(f"  Actions bloquees: {stats['actions_blocked']}")
        print(f"  Actions/min: {stats['actions_per_minute']:.1f}")

        analysis = self.analyze_observations()
        print(f"\nScore de securite: {analysis['safety_score']:.1f}/100")

        print(f"\nTop 5 actions:")
        for action, count in analysis['top_actions'].items():
            print(f"  {action}: {count}")

        print(f"\nRecommandations:")
        for rec in analysis['recommendations']:
            print(f"  {rec}")

        print("\n" + "=" * 70)

    def enable(self):
        """Active le mode observation"""
        self.enabled = True
        self.logger.warning("[WARNING] MODE OBSERVATION ACTIVÉ - Aucune action ne sera exécutée")

    def disable(self):
        """Désactive le mode observation (DANGER!)"""
        if self.enabled:
            self.logger.critical("[WARNING][WARNING][WARNING] MODE OBSERVATION DÉSACTIVÉ [WARNING][WARNING][WARNING]")
            self.logger.critical("[WARNING] LE BOT VA MAINTENANT AGIR RÉELLEMENT [WARNING]")
            self.logger.critical("[WARNING] UTILISER UNIQUEMENT SUR COMPTE JETABLE [WARNING]")

        self.enabled = False

    def is_enabled(self) -> bool:
        """Vérifie si le mode observation est activé"""
        return self.enabled

def create_observation_mode(log_file: str = "logs/observation.json",
                           auto_enabled: bool = True) -> ObservationMode:
    """
    Factory function pour créer ObservationMode

    Args:
        log_file: Fichier de log
        auto_enabled: Active automatiquement (True par défaut pour sécurité)

    Returns:
        Instance d'ObservationMode
    """
    return ObservationMode(log_file, auto_enabled)