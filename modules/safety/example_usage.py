"""
Exemple d'utilisation du module Safety
Démontre comment utiliser les systèmes de sécurité pour un bot réaliste
"""

import time
import logging
from typing import Callable

from . import (
    SafetyManager,
    HumanBehaviorSimulator, 
    SessionManager,
    AntiDetectionSystem,
    ActionType,
    SAFETY_PROFILES
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def simulate_game_action(action_name: str) -> str:
    """
    Simule une action de jeu (fonction exemple)
    """
    print(f"  -> Exécution de l'action: {action_name}")
    time.sleep(0.1)  # Simulation du temps d'exécution
    
    # Simulation d'échec occasionnel
    import random
    if random.random() < 0.05:  # 5% d'échec
        raise Exception(f"Échec simulé de l'action: {action_name}")
    
    return f"Succès: {action_name}"


def demonstrate_human_behavior():
    """
    Démontre l'utilisation du simulateur de comportement humain
    """
    print("\n=== DÉMONSTRATION: Simulateur de Comportement Humain ===")
    
    # Création d'un simulateur avec profil casual
    from . import CASUAL_PROFILE
    simulator = HumanBehaviorSimulator(CASUAL_PROFILE)
    
    print(f"Profil utilisé: Casual (erreurs: {CASUAL_PROFILE.error_rate}, WPM: {CASUAL_PROFILE.typing_wpm})")
    
    # Test de différents types d'actions
    action_types = [ActionType.CLICK, ActionType.MOVEMENT, ActionType.COMBAT, ActionType.TYPING]
    
    for action_type in action_types:
        print(f"\nTest {action_type.value}:")
        
        # Exécution de plusieurs actions du même type
        for i in range(3):
            result = simulator.perform_action(
                action_type, 
                simulate_game_action, 
                f"{action_type.value}_{i+1}"
            )
            
            print(f"  Action {i+1}: délai={result['delay_used']:.3f}s, "
                  f"erreur={result['error_simulated']}, "
                  f"fatigue={result['fatigue_level']:.3f}")
    
    # Affichage des statistiques finales
    stats = simulator.get_session_stats()
    print(f"\nStatistiques de session:")
    print(f"  Actions effectuées: {stats['actions_performed']}")
    print(f"  Erreurs simulées: {stats['errors_made']}")
    print(f"  Pauses prises: {stats['pauses_taken']}")
    print(f"  Taux d'erreur moyen: {stats['average_error_rate']:.3f}")


def demonstrate_session_management():
    """
    Démontre l'utilisation du gestionnaire de sessions
    """
    print("\n=== DÉMONSTRATION: Gestionnaire de Sessions ===")
    
    # Configuration avec limites courtes pour la démo
    from . import SessionLimits
    test_limits = SessionLimits(
        max_continuous_minutes=2,  # 2 minutes max pour la démo
        short_break_frequency=1,   # Pause courte chaque minute
        short_break_duration=0.5   # 30 secondes de pause
    )
    
    manager = SessionManager(test_limits)
    
    if not manager.start_session():
        print("❌ Impossible de démarrer la session")
        return
    
    print("✅ Session démarrée avec succès")
    
    # Simulation d'activité
    for i in range(10):
        manager.record_action(f"action_{i+1}")
        
        # Affichage de l'état de session
        info = manager.get_session_info()
        print(f"Action {i+1} - État: {info['status']}, "
              f"Durée active: {info['active_duration']:.1f}s, "
              f"Actions: {info['actions_performed']}")
        
        # Vérification des pauses obligatoires
        if info['status'] == 'mandatory_break':
            print("⏸️  Pause obligatoire en cours...")
            time.sleep(2)  # Simulation de pause courte
            manager.resume_session()
            print("▶️  Session reprise")
        
        time.sleep(0.5)  # Délai entre actions
    
    # Fin de session
    manager.end_session()
    print("🏁 Session terminée")


def demonstrate_anti_detection():
    """
    Démontre l'utilisation du système anti-détection
    """
    print("\n=== DÉMONSTRATION: Système Anti-Détection ===")
    
    system = AntiDetectionSystem()
    
    print("🕵️ Système anti-détection initialisé")
    
    # Simulation de différents patterns d'activité
    scenarios = [
        ("Activité normale", 20, 0.5),
        ("Activité intensive", 50, 0.1), 
        ("Activité très régulière", 30, 0.0),
        ("Activité avec pauses", 25, 1.0)
    ]
    
    for scenario_name, action_count, pause_factor in scenarios:
        print(f"\n--- Scénario: {scenario_name} ---")
        
        # Réinitialisation pour chaque scénario
        system = AntiDetectionSystem()
        
        for i in range(action_count):
            # Enregistrement d'événements variés
            system.record_game_event("movement", {
                "path": [(i*10, i*10), ((i+1)*10, (i+1)*10)],
                "efficiency": 0.8 + (i % 3) * 0.1
            })
            
            # Variations de timing et position
            timing_var = system.get_human_timing_variation(1.0, "click")
            pos_var = system.get_mouse_position_variation((100, 100))
            
            # Introduction d'erreurs occasionnelles
            if system.should_introduce_error():
                print(f"  ❌ Erreur introduite à l'action {i+1}")
            
            # Pauses selon le scénario
            if pause_factor > 0 and i % 10 == 0:
                pause_suggestion = system.get_natural_pause_suggestion()
                if pause_suggestion:
                    duration, reason = pause_suggestion
                    print(f"  ⏸️  Pause suggérée: {duration:.1f}s ({reason})")
                    time.sleep(min(duration, 2))  # Pause limitée pour la démo
            
            time.sleep(0.1 * (1 + pause_factor))
        
        # Analyse du risque pour ce scénario
        risk_level = system.analyze_current_behavior()
        recommendations = system.get_stealth_recommendations()
        
        print(f"  📊 Risque de détection: {risk_level.value}")
        print(f"  💡 Recommandations:")
        for rec in recommendations[:3]:  # Limiter à 3 recommandations
            print(f"     - {rec}")


def demonstrate_integrated_safety():
    """
    Démontre l'utilisation intégrée de tous les systèmes de sécurité
    """
    print("\n=== DÉMONSTRATION: Gestionnaire de Sécurité Intégré ===")
    
    # Test avec différents profils
    profiles = ["conservative", "balanced", "aggressive"]
    
    for profile_name in profiles:
        print(f"\n--- Test du profil: {profile_name.upper()} ---")
        
        # Initialisation du gestionnaire de sécurité
        safety_manager = SafetyManager(profile_name)
        
        if not safety_manager.start_safe_session():
            print(f"❌ Impossible de démarrer la session {profile_name}")
            continue
        
        print(f"✅ Session {profile_name} démarrée")
        
        # Simulation d'activités variées
        activities = [
            ("click", "Clic sur interface"),
            ("movement", "Déplacement de personnage"), 
            ("combat", "Action de combat"),
            ("typing", "Saisie de texte"),
            ("menu", "Navigation dans menu")
        ]
        
        for activity_type, activity_description in activities:
            try:
                result = safety_manager.perform_safe_action(
                    activity_type,
                    simulate_game_action,
                    activity_description
                )
                
                print(f"  ✅ {activity_description}: "
                      f"délai={result['delay_used']:.3f}s, "
                      f"erreur={result['error_simulated']}")
                
            except Exception as e:
                print(f"  ❌ Erreur lors de {activity_description}: {e}")
            
            time.sleep(0.2)
        
        # État de sécurité final
        safety_status = safety_manager.check_safety_status()
        print(f"  📊 État final:")
        print(f"     Risque détection: {safety_status['detection_risk']}")
        print(f"     Actions effectuées: {safety_status['behavior_stats']['actions_performed']}")
        print(f"     Fatigue: {safety_status['behavior_stats']['current_fatigue']:.3f}")
        
        # Fin de session
        safety_manager.end_safe_session()
        print(f"🏁 Session {profile_name} terminée")


def run_comprehensive_demo():
    """
    Lance une démonstration complète de tous les modules
    """
    print("🚀 DÉMONSTRATION COMPLÈTE DU MODULE SAFETY")
    print("=" * 60)
    
    try:
        # 1. Démonstration du simulateur de comportement
        demonstrate_human_behavior()
        
        # 2. Démonstration de la gestion de session
        demonstrate_session_management() 
        
        # 3. Démonstration de l'anti-détection
        demonstrate_anti_detection()
        
        # 4. Démonstration intégrée
        demonstrate_integrated_safety()
        
        print("\n" + "=" * 60)
        print("✅ Démonstration terminée avec succès!")
        print("\nLe module Safety fournit:")
        print("  - Simulation réaliste du comportement humain")
        print("  - Gestion intelligente des sessions avec pauses")
        print("  - Système anti-détection avancé")
        print("  - Intégration complète pour un usage sécurisé")
        
    except KeyboardInterrupt:
        print("\n⏹️  Démonstration interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur pendant la démonstration: {e}")
        logger.exception("Erreur détaillée:")


if __name__ == "__main__":
    # Lancement de la démonstration complète
    run_comprehensive_demo()
    
    # Exemple d'usage simple
    print("\n" + "=" * 40)
    print("EXEMPLE D'USAGE SIMPLE:")
    print("=" * 40)
    
    # Usage minimal recommandé
    safety = SafetyManager("balanced")
    
    if safety.start_safe_session():
        # Vos actions de bot ici, par exemple:
        for i in range(5):
            safety.perform_safe_action("click", lambda: print(f"Action {i+1} exécutée"))
            time.sleep(1)
        
        print(f"État final: {safety.check_safety_status()['detection_risk']}")
        safety.end_safe_session()
    
    print("\n💡 Pour utiliser dans votre bot:")
    print("   from modules.safety import SafetyManager")
    print("   safety = SafetyManager('balanced')")
    print("   safety.start_safe_session()")
    print("   safety.perform_safe_action('click', your_function)")
    print("   safety.end_safe_session()")