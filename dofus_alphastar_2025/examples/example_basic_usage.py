#!/usr/bin/env python3
"""
Exemple 1: Utilisation Basique du Bot
Démontre l'utilisation minimale en mode observation
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.safety import create_observation_mode
from core.calibration import create_calibrator

def main():
    print("=" * 70)
    print("EXEMPLE 1: Utilisation Basique")
    print("=" * 70)

    # 1. Créer observation mode (SÉCURITÉ!)
    print("\n[1] Création Observation Mode...")
    obs = create_observation_mode(auto_enabled=True)
    print(f"[OK] Observation Mode créé (enabled: {obs.is_enabled()})")

    # 2. Simuler quelques décisions
    print("\n[2] Simulation de décisions...")

    game_state = {'hp': 100, 'pa': 6}

    # Décision 1: Navigation
    action1 = obs.intercept_action(
        action_type='navigation',
        action_details={'target': (200, 250)},
        game_state=game_state,
        reason='Explorer la map'
    )
    print(f"Action 1 (navigation): {action1}")  # None = bloquée

    # Décision 2: Mouse click
    action2 = obs.intercept_action(
        action_type='mouse_click',
        action_details={'position': (150, 180)},
        game_state=game_state,
        reason='Collecter ressource'
    )
    print(f"Action 2 (mouse_click): {action2}")  # None = bloquée

    # Décision 3: Spell cast
    action3 = obs.intercept_action(
        action_type='cast_spell',
        action_details={'spell_id': 1, 'target': 'enemy'},
        game_state=game_state,
        reason='Attaquer monstre'
    )
    print(f"Action 3 (cast_spell): {action3}")  # None = bloquée

    # 3. Analyser les observations
    print("\n[3] Analyse des observations...")
    stats = obs.get_stats()
    print(f"Total décisions: {stats['total_decisions']}")
    print(f"Actions bloquées: {stats['actions_blocked']}")

    analysis = obs.analyze_observations()
    print(f"Safety score: {analysis['safety_score']:.1f}/100")
    print(f"Recommandations: {analysis['recommendations']}")

    # 4. Sauvegarder
    print("\n[4] Sauvegarde observations...")
    obs.save_observations("logs/example1_observations.json")
    print("[OK] Observations sauvegardées")

    print("\n" + "=" * 70)
    print("[OK] Exemple terminé avec succès!")
    print("=" * 70)


if __name__ == "__main__":
    main()