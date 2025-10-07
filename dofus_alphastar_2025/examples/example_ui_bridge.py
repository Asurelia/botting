#!/usr/bin/env python3
"""
Exemple 4: Utilisation du UI Bridge
Démontre l'intégration complète UI <-> Core
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ui_bridge import create_ui_bridge

def on_state_update(state):
    """Callback pour mises à jour d'état"""
    print(f"\n[STATE UPDATE]")
    print(f"  Bot running: {state['bot_running']}")
    print(f"  Observation: {state['observation_mode']}")
    print(f"  Safety score: {state['safety_score']:.1f}/100")
    print(f"  Total actions: {state['total_actions']}")

def on_log(log_entry):
    """Callback pour logs"""
    timestamp = log_entry['timestamp'].split('T')[1].split('.')[0]
    level = log_entry['level']
    message = log_entry['message']
    print(f"[{timestamp}] [{level}] {message}")

def main():
    print("=" * 70)
    print("EXEMPLE 4: UI Bridge - Intégration Complète")
    print("=" * 70)

    # 1. Créer le bridge
    print("\n[1] Création UIBridge...")
    bridge = create_ui_bridge()
    print("[OK] Bridge créé et systèmes initialisés")

    # 2. Configurer callbacks
    print("\n[2] Configuration callbacks...")
    bridge.set_ui_update_callback(on_state_update)
    bridge.set_log_callback(on_log)
    print("[OK] Callbacks configurés")

    # 3. État initial
    print("\n[3] État initial...")
    state = bridge.get_state()
    print(f"Calibrated: {state['calibrated']}")
    print(f"Observation mode: {state['observation_mode']}")

    # 4. Test DofusDB
    print("\n[4] Test DofusDB...")
    bridge.test_dofusdb_connection()
    time.sleep(2)  # Laisser temps pour callback

    # 5. Démarrer bot (observation mode)
    print("\n[5] Démarrage bot en mode observation...")
    success = bridge.start_bot(observation_only=True)

    if success:
        print("[OK] Bot démarré")

        # Simuler une session courte
        print("\n[6] Session observation (5 secondes)...")
        for i in range(5):
            time.sleep(1)
            print(f"  Session: {i+1}s...")

            # Le monitoring thread met à jour automatiquement l'état
            state = bridge.get_state()

        # Arrêter
        print("\n[7] Arrêt du bot...")
        bridge.stop_bot()
        print("[OK] Bot arrêté")

    else:
        print("[FAIL] Impossible de démarrer (calibration requise?)")

    # 8. Stats finales
    print("\n[8] Statistiques finales...")
    obs_stats = bridge.get_observation_stats()
    if obs_stats:
        print(f"Total décisions: {obs_stats['total_decisions']}")
        print(f"Actions bloquées: {obs_stats['actions_blocked']}")
        print(f"Durée: {obs_stats['duration_seconds']:.1f}s")

    dofusdb_stats = bridge.get_dofusdb_stats()
    if dofusdb_stats:
        print(f"\nDofusDB requêtes: {dofusdb_stats['requests']}")
        print(f"Cache ratio: {dofusdb_stats['cache_ratio']}")

    # 9. Logs récents
    print("\n[9] Logs récents...")
    recent_logs = bridge.get_logs(limit=5)
    for log in recent_logs:
        timestamp = log['timestamp'].split('T')[1].split('.')[0]
        print(f"  [{timestamp}] {log['level']}: {log['message']}")

    print("\n" + "=" * 70)
    print("[OK] Exemple terminé avec succès!")
    print("=" * 70)


if __name__ == "__main__":
    main()