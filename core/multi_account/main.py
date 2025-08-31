#!/usr/bin/env python3
"""
Point d'entrée principal pour le système de gestion multi-comptes DOFUS.

Ce script permet de lancer le système multi-comptes avec différentes options :
- Interface graphique complète
- Mode console pour tests
- Démo avec comptes factices
- Outils de maintenance

Usage:
    python main.py [options]
    
Options:
    --gui           Lance l'interface graphique (défaut)
    --console       Mode console interactif
    --demo          Démo avec comptes factices
    --test          Tests automatiques
    --maintenance   Outils de maintenance
    --help          Affiche cette aide
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Ajout du répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('G:/Botting/logs/multi_account.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def ensure_directories():
    """Assure que tous les répertoires nécessaires existent."""
    directories = [
        "G:/Botting/data",
        "G:/Botting/logs",
        "G:/Botting/config",
        "G:/Botting/temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Répertoire créé/vérifié: {directory}")

def launch_gui():
    """Lance l'interface graphique."""
    try:
        from . import MultiAccountSystem
        
        logger.info("Démarrage de l'interface graphique multi-comptes")
        
        # Créer et démarrer le système
        system = MultiAccountSystem()
        if system.start():
            logger.info("Système multi-comptes démarré avec succès")
            return system.launch_gui()
        else:
            logger.error("Échec du démarrage du système")
            return 1
            
    except Exception as e:
        logger.error(f"Erreur lors du lancement de l'interface: {e}")
        return 1

def launch_console():
    """Lance le mode console interactif."""
    try:
        from . import MultiAccountSystem
        
        logger.info("Démarrage du mode console multi-comptes")
        
        # Créer et démarrer le système
        system = MultiAccountSystem()
        if not system.start():
            logger.error("Échec du démarrage du système")
            return 1
        
        print("\n" + "="*60)
        print("       DOFUS Multi-Account Manager - Mode Console")
        print("="*60)
        print()
        
        try:
            console_loop(system)
        finally:
            system.stop()
            
        return 0
        
    except Exception as e:
        logger.error(f"Erreur dans le mode console: {e}")
        return 1

def console_loop(system):
    """Boucle principale du mode console."""
    commands = {
        'help': show_help,
        'list': lambda: list_accounts(system),
        'add': lambda: add_account_interactive(system),
        'remove': lambda: remove_account_interactive(system),
        'start': lambda: start_account_interactive(system),
        'stop': lambda: stop_account_interactive(system),
        'stats': lambda: show_statistics(system),
        'schedule': lambda: schedule_interactive(system),
        'group': lambda: create_group_interactive(system),
        'quit': lambda: None,
        'exit': lambda: None
    }
    
    show_help()
    
    while True:
        try:
            command = input("\nMultiAccount> ").strip().lower()
            
            if command in ['quit', 'exit']:
                print("Au revoir!")
                break
            
            if command in commands:
                result = commands[command]()
                if result is False:
                    break
            else:
                print(f"Commande inconnue: {command}. Tapez 'help' pour l'aide.")
                
        except KeyboardInterrupt:
            print("\nInterrompu par l'utilisateur. Au revoir!")
            break
        except EOFError:
            print("\nAu revoir!")
            break
        except Exception as e:
            print(f"Erreur: {e}")
            logger.error(f"Erreur dans la console: {e}")

def show_help():
    """Affiche l'aide des commandes console."""
    help_text = """
Commandes disponibles:
  help         - Affiche cette aide
  list         - Liste tous les comptes
  add          - Ajoute un nouveau compte
  remove       - Supprime un compte
  start        - Démarre un compte
  stop         - Arrête un compte
  stats        - Affiche les statistiques
  schedule     - Planifie une session
  group        - Crée un groupe de comptes
  quit/exit    - Quitte le programme
"""
    print(help_text)

def list_accounts(system):
    """Liste tous les comptes."""
    accounts = system.account_manager.get_all_accounts()
    
    if not accounts:
        print("Aucun compte configuré.")
        return
    
    print("\nComptes configurés:")
    print("-" * 80)
    print(f"{'ID':<8} {'Nom':<15} {'Personnage':<15} {'Serveur':<12} {'Statut':<12}")
    print("-" * 80)
    
    for account in accounts:
        print(f"{account.id[:8]:<8} {account.credentials.username:<15} "
              f"{account.credentials.character_name:<15} {account.credentials.server:<12} "
              f"{account.status.value:<12}")

def add_account_interactive(system):
    """Ajoute un compte de manière interactive."""
    try:
        print("\nAjout d'un nouveau compte:")
        username = input("Nom d'utilisateur: ").strip()
        password = input("Mot de passe: ").strip()
        character = input("Nom du personnage: ").strip()
        
        print("\nServeurs disponibles:")
        servers = ["Ily", "Meriana", "Julith", "Dodge", "Terra Cogita", "Echo", "Eratz"]
        for i, server in enumerate(servers, 1):
            print(f"{i}. {server}")
        
        server_choice = input("Choisissez un serveur (1-7): ").strip()
        if server_choice.isdigit() and 1 <= int(server_choice) <= 7:
            server = servers[int(server_choice) - 1]
        else:
            server = "Ily"  # Défaut
        
        if not all([username, password, character]):
            print("Tous les champs sont obligatoires.")
            return
        
        account_id = system.add_account(username, password, character, server)
        print(f"Compte ajouté avec succès. ID: {account_id}")
        
    except Exception as e:
        print(f"Erreur lors de l'ajout du compte: {e}")

def remove_account_interactive(system):
    """Supprime un compte de manière interactive."""
    list_accounts(system)
    
    account_id = input("\nID du compte à supprimer (8 premiers caractères): ").strip()
    
    if len(account_id) < 8:
        print("ID invalide.")
        return
    
    # Trouver le compte complet
    accounts = system.account_manager.get_all_accounts()
    target_account = None
    
    for account in accounts:
        if account.id.startswith(account_id):
            target_account = account
            break
    
    if not target_account:
        print("Compte non trouvé.")
        return
    
    confirm = input(f"Supprimer le compte {target_account.credentials.username}? (y/N): ").strip().lower()
    if confirm == 'y':
        if system.account_manager.remove_account(target_account.id):
            print("Compte supprimé avec succès.")
        else:
            print("Erreur lors de la suppression.")

def start_account_interactive(system):
    """Démarre un compte de manière interactive."""
    list_accounts(system)
    
    account_id = input("\nID du compte à démarrer (8 premiers caractères): ").strip()
    
    # Trouver et démarrer le compte
    accounts = system.account_manager.get_all_accounts()
    for account in accounts:
        if account.id.startswith(account_id):
            if system.launch_account(account.id):
                print(f"Compte {account.credentials.username} en cours de démarrage...")
            else:
                print("Erreur lors du démarrage.")
            return
    
    print("Compte non trouvé.")

def stop_account_interactive(system):
    """Arrête un compte de manière interactive."""
    active_accounts = system.account_manager.get_active_accounts()
    
    if not active_accounts:
        print("Aucun compte actif.")
        return
    
    print("\nComptes actifs:")
    for i, account in enumerate(active_accounts, 1):
        print(f"{i}. {account.credentials.username} ({account.status.value})")
    
    choice = input("Choisissez un compte à arrêter (numéro): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(active_accounts):
        account = active_accounts[int(choice) - 1]
        system.account_manager.disconnect_account(account.id)
        if system.window_manager.close_account_window(account.id):
            print(f"Compte {account.credentials.username} arrêté.")
        else:
            print("Erreur lors de l'arrêt.")

def show_statistics(system):
    """Affiche les statistiques du système."""
    stats = system.get_statistics()
    
    print("\n" + "="*50)
    print("           STATISTIQUES SYSTÈME")
    print("="*50)
    
    # Statistiques des comptes
    account_stats = stats.get('accounts', {})
    print(f"\nComptes:")
    print(f"  Total: {account_stats.get('total_accounts', 0)}")
    print(f"  Actifs: {account_stats.get('active_accounts', 0)}")
    print(f"  Inactifs: {account_stats.get('inactive_accounts', 0)}")
    print(f"  Groupes: {account_stats.get('groups_count', 0)}")
    
    # Statistiques des fenêtres
    window_stats = stats.get('windows', {})
    print(f"\nFenêtres:")
    print(f"  Ouvertes: {window_stats.get('window_count', 0)}")
    print(f"  Mémoire totale: {window_stats.get('total_memory_mb', 0)} MB")
    print(f"  Mémoire moyenne: {window_stats.get('average_memory_mb', 0)} MB")
    
    # Statistiques de synchronisation
    sync_stats = stats.get('synchronization', {})
    print(f"\nSynchronisation:")
    print(f"  Actions traitées: {sync_stats.get('actions_processed', 0)}")
    print(f"  Actions réussies: {sync_stats.get('actions_successful', 0)}")
    print(f"  Actions échouées: {sync_stats.get('actions_failed', 0)}")
    
    # Statistiques de planification
    sched_stats = stats.get('scheduling', {})
    print(f"\nPlanification:")
    print(f"  Sessions planifiées: {sched_stats.get('sessions_scheduled', 0)}")
    print(f"  Sessions terminées: {sched_stats.get('sessions_completed', 0)}")

def schedule_interactive(system):
    """Planifie une session de manière interactive."""
    accounts = system.account_manager.get_all_accounts()
    
    if not accounts:
        print("Aucun compte disponible.")
        return
    
    print("\nComptes disponibles:")
    for i, account in enumerate(accounts, 1):
        print(f"{i}. {account.credentials.username}")
    
    choice = input("Choisissez un compte (numéro): ").strip()
    
    if not choice.isdigit() or not (1 <= int(choice) <= len(accounts)):
        print("Choix invalide.")
        return
    
    account = accounts[int(choice) - 1]
    
    print("\nTypes de session disponibles:")
    session_types = ["farming", "questing", "dungeon", "pvp", "trading"]
    for i, stype in enumerate(session_types, 1):
        print(f"{i}. {stype}")
    
    type_choice = input("Choisissez un type (numéro): ").strip()
    
    if not type_choice.isdigit() or not (1 <= int(type_choice) <= len(session_types)):
        session_type = "farming"
    else:
        session_type = session_types[int(type_choice) - 1]
    
    # Heure de début (dans 5 minutes par défaut)
    start_time = datetime.now() + timedelta(minutes=5)
    
    # Durée (2 heures par défaut)
    duration_input = input("Durée en minutes (défaut: 120): ").strip()
    if duration_input.isdigit():
        duration = timedelta(minutes=int(duration_input))
    else:
        duration = timedelta(hours=2)
    
    try:
        schedule_id = system.schedule_session(
            account.id, session_type, start_time, duration
        )
        print(f"Session planifiée avec succès. ID: {schedule_id}")
        print(f"Début: {start_time.strftime('%d/%m/%Y %H:%M')}")
        print(f"Durée: {int(duration.total_seconds() / 60)} minutes")
        
    except Exception as e:
        print(f"Erreur lors de la planification: {e}")

def create_group_interactive(system):
    """Crée un groupe de manière interactive."""
    accounts = system.account_manager.get_all_accounts()
    
    if len(accounts) < 2:
        print("Au moins 2 comptes sont nécessaires pour créer un groupe.")
        return
    
    print("\nComptes disponibles:")
    for i, account in enumerate(accounts, 1):
        print(f"{i}. {account.credentials.username}")
    
    leader_choice = input("Choisissez le leader (numéro): ").strip()
    
    if not leader_choice.isdigit() or not (1 <= int(leader_choice) <= len(accounts)):
        print("Choix invalide.")
        return
    
    leader = accounts[int(leader_choice) - 1]
    
    print(f"\nLeader sélectionné: {leader.credentials.username}")
    print("Choisissez les membres (numéros séparés par des virgules):")
    
    members_input = input("Membres: ").strip()
    member_indices = []
    
    try:
        for num in members_input.split(','):
            idx = int(num.strip())
            if 1 <= idx <= len(accounts) and idx != int(leader_choice):
                member_indices.append(idx - 1)
    except ValueError:
        print("Format invalide.")
        return
    
    if not member_indices:
        print("Aucun membre sélectionné.")
        return
    
    member_ids = [accounts[i].id for i in member_indices]
    group_name = input("Nom du groupe: ").strip() or "Groupe Auto"
    
    try:
        group_id = system.create_group(group_name, leader.id, member_ids)
        print(f"Groupe '{group_name}' créé avec succès. ID: {group_id}")
        print(f"Leader: {leader.credentials.username}")
        print(f"Membres: {', '.join([accounts[i].credentials.username for i in member_indices])}")
        
    except Exception as e:
        print(f"Erreur lors de la création du groupe: {e}")

def launch_demo():
    """Lance une démo avec des comptes factices."""
    try:
        from . import MultiAccountSystem
        
        logger.info("Démarrage de la démo multi-comptes")
        
        # Créer le système
        system = MultiAccountSystem()
        if not system.start():
            logger.error("Échec du démarrage du système")
            return 1
        
        print("\n" + "="*60)
        print("        DOFUS Multi-Account Manager - DEMO")
        print("="*60)
        print()
        print("Création de comptes de démonstration...")
        
        # Créer des comptes fictifs
        demo_accounts = [
            ("demo_user1", "password123", "Iop-Fighter", "Ily"),
            ("demo_user2", "password456", "Eni-Healer", "Ily"),
            ("demo_user3", "password789", "Sadi-Support", "Meriana"),
            ("demo_user4", "passwordABC", "Cra-Archer", "Meriana")
        ]
        
        account_ids = []
        for username, password, character, server in demo_accounts:
            try:
                account_id = system.add_account(username, password, character, server)
                account_ids.append(account_id)
                print(f"✓ Compte créé: {username} ({character}) sur {server}")
            except Exception as e:
                print(f"✗ Erreur pour {username}: {e}")
        
        if len(account_ids) >= 2:
            # Créer un groupe de démo
            try:
                group_id = system.create_group(
                    "Groupe Démo", 
                    account_ids[0], 
                    account_ids[1:3]
                )
                print(f"✓ Groupe de démo créé: {group_id}")
            except Exception as e:
                print(f"✗ Erreur création groupe: {e}")
            
            # Planifier quelques sessions
            try:
                for i, account_id in enumerate(account_ids[:2]):
                    start_time = datetime.now() + timedelta(minutes=5 + i*10)
                    schedule_id = system.schedule_session(
                        account_id,
                        "farming",
                        start_time,
                        timedelta(hours=1)
                    )
                    print(f"✓ Session planifiée: {schedule_id}")
            except Exception as e:
                print(f"✗ Erreur planification: {e}")
        
        print("\n" + "="*60)
        print("Démo configurée! Statistiques:")
        show_statistics(system)
        
        print("\nLancement de l'interface graphique...")
        return system.launch_gui()
        
    except Exception as e:
        logger.error(f"Erreur dans la démo: {e}")
        return 1

def run_tests():
    """Exécute les tests automatiques du système."""
    print("\n" + "="*60)
    print("         TESTS AUTOMATIQUES MULTI-COMPTES")
    print("="*60)
    
    try:
        from . import MultiAccountSystem
        
        # Test 1: Initialisation du système
        print("\n[1/5] Test d'initialisation...")
        system = MultiAccountSystem()
        assert system.start(), "Échec du démarrage du système"
        print("✓ Système initialisé")
        
        # Test 2: Ajout de comptes
        print("\n[2/5] Test d'ajout de comptes...")
        account_id1 = system.add_account("test1", "pass1", "char1", "Ily")
        account_id2 = system.add_account("test2", "pass2", "char2", "Ily")
        assert account_id1 and account_id2, "Échec de l'ajout de comptes"
        print("✓ Comptes ajoutés")
        
        # Test 3: Création de groupe
        print("\n[3/5] Test de création de groupe...")
        group_id = system.create_group("Test Group", account_id1, [account_id2])
        assert group_id, "Échec de la création de groupe"
        print("✓ Groupe créé")
        
        # Test 4: Planification
        print("\n[4/5] Test de planification...")
        start_time = datetime.now() + timedelta(minutes=1)
        schedule_id = system.schedule_session(
            account_id1, "farming", start_time, timedelta(minutes=30)
        )
        assert schedule_id, "Échec de la planification"
        print("✓ Session planifiée")
        
        # Test 5: Statistiques
        print("\n[5/5] Test des statistiques...")
        stats = system.get_statistics()
        assert isinstance(stats, dict), "Échec des statistiques"
        assert stats.get('accounts', {}).get('total_accounts', 0) >= 2, "Nombre de comptes incorrect"
        print("✓ Statistiques OK")
        
        print("\n" + "="*60)
        print("         TOUS LES TESTS RÉUSSIS!")
        print("="*60)
        
        # Nettoyage
        system.stop()
        return 0
        
    except Exception as e:
        print(f"\n✗ ÉCHEC DU TEST: {e}")
        logger.error(f"Erreur dans les tests: {e}")
        return 1

def maintenance_mode():
    """Mode maintenance pour les outils de gestion."""
    print("\n" + "="*60)
    print("           MODE MAINTENANCE")
    print("="*60)
    
    maintenance_commands = {
        '1': ("Nettoyer les logs", clean_logs),
        '2': ("Réinitialiser la base de données", reset_database),
        '3': ("Vérifier l'intégrité du système", check_system_integrity),
        '4': ("Exporter la configuration", export_config),
        '5': ("Importer la configuration", import_config),
        '0': ("Retour", None)
    }
    
    while True:
        print("\nOptions de maintenance:")
        for key, (desc, _) in maintenance_commands.items():
            print(f"  {key}. {desc}")
        
        choice = input("\nChoix: ").strip()
        
        if choice == '0':
            break
        
        if choice in maintenance_commands:
            desc, func = maintenance_commands[choice]
            if func:
                print(f"\n{desc}...")
                try:
                    func()
                    print("✓ Opération terminée")
                except Exception as e:
                    print(f"✗ Erreur: {e}")
        else:
            print("Choix invalide.")

def clean_logs():
    """Nettoie les anciens logs."""
    import glob
    log_files = glob.glob("G:/Botting/logs/*.log")
    
    for log_file in log_files:
        try:
            # Garder seulement les 100 dernières lignes
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) > 100:
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.writelines(lines[-100:])
                print(f"  Nettoyé: {log_file}")
        except Exception as e:
            print(f"  Erreur avec {log_file}: {e}")

def reset_database():
    """Réinitialise la base de données."""
    import glob
    db_files = glob.glob("G:/Botting/data/*.db")
    
    confirm = input("⚠️  Supprimer toutes les données? (y/N): ").strip().lower()
    if confirm == 'y':
        for db_file in db_files:
            try:
                os.remove(db_file)
                print(f"  Supprimé: {db_file}")
            except Exception as e:
                print(f"  Erreur avec {db_file}: {e}")

def check_system_integrity():
    """Vérifie l'intégrité du système."""
    # Vérifier les répertoires
    directories = ["G:/Botting/data", "G:/Botting/logs", "G:/Botting/config"]
    for directory in directories:
        if os.path.exists(directory):
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} (manquant)")
    
    # Vérifier les dépendances
    required_modules = ['PySide6', 'cryptography', 'psutil', 'schedule']
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (manquant)")

def export_config():
    """Exporte la configuration."""
    print("  Exportation non encore implémentée")

def import_config():
    """Importe la configuration."""
    print("  Importation non encore implémentée")

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="DOFUS Multi-Account Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--gui', action='store_true', default=True,
        help='Lance l\'interface graphique (défaut)'
    )
    parser.add_argument(
        '--console', action='store_true',
        help='Mode console interactif'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Démo avec comptes factices'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Tests automatiques'
    )
    parser.add_argument(
        '--maintenance', action='store_true',
        help='Outils de maintenance'
    )
    
    args = parser.parse_args()
    
    # Assurer que les répertoires existent
    ensure_directories()
    
    try:
        if args.console:
            return launch_console()
        elif args.demo:
            return launch_demo()
        elif args.test:
            return run_tests()
        elif args.maintenance:
            maintenance_mode()
            return 0
        else:  # GUI par défaut
            return launch_gui()
            
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.")
        return 1
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        print(f"Erreur fatale: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())