#!/usr/bin/env python3
"""
Exemple d'utilisation du système d'alertes archimonstres.
Démonstration des différentes configurations et fonctionnalités.
"""

import sys
import time
import json
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.archmonster import (
    ArchmonsterSystem,
    create_archmonster_system,
    quick_setup_discord,
    quick_setup_basic,
    NotificationMode,
    AlertPriority
)


def example_basic_setup():
    """Exemple de configuration basique."""
    print("=== Configuration Basique ===")
    
    # Configuration simple avec alertes sonores et système
    system = quick_setup_basic(sound_enabled=True)
    
    # Configurer archimonstres à surveiller
    watched_archs = [
        'Chafer Rōnin',
        'Batofu Royal',
        'Gelée Royale Bleue'
    ]
    system.set_watched_archmonsters(watched_archs)
    
    # Configurer zones
    watched_zones = [
        'Cimetière des Torturés',
        'Plaine des Porkass',
        'Territoire des Bandits'
    ]
    system.set_watched_zones(watched_zones)
    
    print(f"Surveillance configurée pour {len(watched_archs)} archimonstres")
    print(f"et {len(watched_zones)} zones")
    
    return system


def example_discord_setup():
    """Exemple avec Discord."""
    print("=== Configuration Discord ===")
    
    # Remplacer par votre URL de webhook Discord
    webhook_url = "https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN"
    
    # Configuration avec Discord
    system = quick_setup_discord(
        webhook_url=webhook_url,
        watched_archmonsters=['Chafer Rōnin', 'Wa Wabbit'],
        watched_zones=['Cimetière des Torturés', 'Île de la Tortue']
    )
    
    print("Système configuré avec Discord")
    return system


def example_advanced_setup():
    """Exemple de configuration avancée."""
    print("=== Configuration Avancée ===")
    
    # Configuration personnalisée complète
    config = {
        'detector': {
            'scan_interval': 1.0,  # Scan plus fréquent
            'enable_visual_detection': True,
            'enable_chat_detection': True,
            'watched_archmonsters': [
                'Chafer Rōnin', 'Batofu Royal', 'Gelée Royale Bleue',
                'Tofu Royal', 'Wa Wabbit', 'Moon'
            ],
            'watched_zones': [
                'Cimetière des Torturés', 'Plaine des Porkass',
                'Territoire des Bandits', 'Champs de Cania'
            ]
        },
        'alerts': {
            'alert_channels': [
                {
                    'channel': 'discord',
                    'enabled': True,
                    'priority_threshold': 'LOW',
                    'rate_limit': 30,
                    'config': {
                        'webhook_url': 'YOUR_DISCORD_WEBHOOK_HERE',
                        'username': 'DOFUS Archimonstres Bot',
                        'mention_roles': ['ROLE_ID']
                    }
                },
                {
                    'channel': 'sound',
                    'enabled': True,
                    'priority_threshold': 'MEDIUM',
                    'config': {
                        'sounds_path': 'data/sounds',
                        'default_sound': 'alert.wav',
                        'archmonster_sounds': {
                            'Chafer Rōnin': 'chafer_special.wav',
                            'Wa Wabbit': 'wabbit_rare.wav'
                        }
                    }
                },
                {
                    'channel': 'system',
                    'enabled': True,
                    'priority_threshold': 'HIGH',
                    'config': {'timeout': 5000}
                }
            ]
        },
        'mode': 'aggressive',  # Mode agressif
        'enabled': True
    }
    
    system = create_archmonster_system(config)
    print("Configuration avancée chargée")
    
    return system


def example_custom_rules():
    """Exemple d'ajout de règles personnalisées."""
    print("=== Règles Personnalisées ===")
    
    system = quick_setup_basic()
    
    # Règle pour archimonstres rares
    from modules.archmonster import (
        NotificationRule, NotificationFilter, FilterType, AlertPriority
    )
    
    rare_filter = NotificationFilter(
        filter_type=FilterType.ARCHMONSTER,
        config={'whitelist': ['Chafer Rōnin', 'Wa Wabbit', 'Moon']},
        description="Archimonstres légendaires"
    )
    
    rare_rule = NotificationRule(
        name="Archimonstres Légendaires",
        priority=AlertPriority.CRITICAL,
        filters=[rare_filter],
        channels=['discord', 'sound', 'system'],
        custom_message="🌟 ARCHIMONSTRE LÉGENDAIRE DÉTECTÉ ! 🌟"
    )
    
    system.notification_manager.add_notification_rule(rare_rule)
    
    # Règle pour heures actives
    time_filter = NotificationFilter(
        filter_type=FilterType.TIME_WINDOW,
        config={'allowed_hours': [18, 19, 20, 21, 22]},
        description="Heures de prime time"
    )
    
    primetime_rule = NotificationRule(
        name="Prime Time",
        priority=AlertPriority.MEDIUM,
        filters=[time_filter],
        custom_message="🕐 Spawn pendant prime time !"
    )
    
    system.notification_manager.add_notification_rule(primetime_rule)
    
    print("Règles personnalisées ajoutées")
    return system


def example_predictions():
    """Exemple d'utilisation des prédictions."""
    print("=== Prédictions de Spawns ===")
    
    system = quick_setup_basic()
    
    # Simuler démarrage pour initialiser tracker
    print("Initialisation du tracker...")
    
    # Récupérer prédictions pour les prochaines 24h
    predictions = system.get_predictions(hours_ahead=24)
    
    if predictions:
        print(f"Prédictions trouvées: {len(predictions)}")
        for pred in predictions[:5]:  # Afficher les 5 premiers
            print(f"- {pred.archmonster} en {pred.zone}")
            print(f"  Prévu: {pred.predicted_time.strftime('%d/%m %H:%M')}")
            print(f"  Confiance: {pred.confidence:.2%}")
            print(f"  Probabilité: {pred.probability:.2%}")
            print()
    else:
        print("Aucune prédiction disponible (pas assez de données)")
    
    return system


def example_statistics():
    """Exemple d'affichage des statistiques."""
    print("=== Statistiques du Système ===")
    
    system = quick_setup_basic()
    
    # Récupérer toutes les statistiques
    stats = system.get_statistics()
    
    print("Statistiques générales:")
    print(f"- Système actif: {stats['notification_manager']['is_running']}")
    print(f"- Mode: {stats['notification_manager']['current_mode']}")
    print(f"- Règles actives: {stats['notification_manager']['active_rules']}")
    
    print("\nBase de données:")
    db_stats = stats['database']
    print(f"- Détections totales: {db_stats.get('detections_count', 0)}")
    print(f"- Taille DB: {db_stats.get('db_size_mb', 0):.1f} MB")
    
    print("\nTracker:")
    tracker_stats = stats['tracker']
    print(f"- Patterns détectés: {tracker_stats.get('total_patterns', 0)}")
    
    print("\nDétecteur:")
    detector_stats = stats['detector']
    print(f"- En cours: {detector_stats.get('is_running', False)}")
    
    return system


def example_monitoring_loop():
    """Exemple de boucle de monitoring."""
    print("=== Boucle de Monitoring ===")
    
    system = quick_setup_basic()
    
    # Configurer surveillance
    system.set_watched_archmonsters(['Chafer Rōnin', 'Batofu Royal'])
    system.set_watched_zones(['Cimetière des Torturés', 'Plaine des Porkass'])
    
    try:
        print("Démarrage du système...")
        system.start()
        
        print("Système actif ! Surveillance en cours...")
        print("Appuyez sur Ctrl+C pour arrêter\n")
        
        iteration = 0
        while True:
            time.sleep(30)  # Attendre 30 secondes
            iteration += 1
            
            print(f"--- Itération {iteration} ---")
            
            # Afficher prédictions courtes
            predictions = system.get_predictions(hours_ahead=2)
            if predictions:
                print(f"Prochains spawns (2h): {len(predictions)}")
                for pred in predictions[:2]:
                    time_left = (pred.predicted_time - pred.predicted_time.now()).total_seconds() / 60
                    print(f"- {pred.archmonster}: dans {time_left:.0f}min")
            
            # Test notification occasionnel
            if iteration % 10 == 0:  # Toutes les 5 minutes
                print("Test notification...")
                system.test_alerts()
            
            print()
    
    except KeyboardInterrupt:
        print("\nArrêt demandé...")
        
    finally:
        system.stop()
        print("Système arrêté.")


def main():
    """Fonction principale - menu d'exemples."""
    examples = {
        '1': ('Configuration Basique', example_basic_setup),
        '2': ('Configuration Discord', example_discord_setup), 
        '3': ('Configuration Avancée', example_advanced_setup),
        '4': ('Règles Personnalisées', example_custom_rules),
        '5': ('Prédictions', example_predictions),
        '6': ('Statistiques', example_statistics),
        '7': ('Monitoring (Ctrl+C pour arrêter)', example_monitoring_loop)
    }
    
    print("=== Exemples Système Archimonstres DOFUS ===\n")
    
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    print("\nChoisissez un exemple (1-7) ou 'q' pour quitter: ", end='')
    
    try:
        choice = input().strip()
        
        if choice.lower() == 'q':
            print("Au revoir !")
            return
        
        if choice in examples:
            name, example_func = examples[choice]
            print(f"\n--- {name} ---")
            
            try:
                system = example_func()
                
                # Pour les exemples non-monitoring, proposer test
                if choice != '7':
                    print("\nVoulez-vous tester les alertes ? (y/n): ", end='')
                    if input().strip().lower() == 'y':
                        print("Test des alertes...")
                        if hasattr(system, 'test_alerts'):
                            system.test_alerts()
                        time.sleep(2)
                    
                    # Nettoyer
                    if hasattr(system, 'stop'):
                        system.stop()
            
            except Exception as e:
                print(f"Erreur lors de l'exemple: {e}")
        
        else:
            print("Choix invalide !")
    
    except KeyboardInterrupt:
        print("\nAu revoir !")
    
    except Exception as e:
        print(f"Erreur: {e}")


if __name__ == "__main__":
    main()