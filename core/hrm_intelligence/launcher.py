"""
Lanceur HRM Intelligence - Point d'entrée principal
Lance l'interface graphique ou le système en mode console

Auteur: Claude Code
Version: 1.0.0
"""

import sys
import os
import argparse
from pathlib import Path

# Ajouter le chemin du module HRM
sys.path.append(str(Path(__file__).parent))

def launch_gui():
    """Lance l'interface graphique"""
    try:
        from hrm_gui import HRMControlInterface
        print("Lancement de l'interface graphique HRM Intelligence...")

        app = HRMControlInterface()
        app.root.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.run()

    except ImportError as e:
        print(f"❌ Erreur d'import GUI: {e}")
        print("Assurez-vous que tkinter et matplotlib sont installés")
        print("pip install matplotlib")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lancement GUI: {e}")
        sys.exit(1)

def launch_console(args):
    """Lance le système en mode console"""
    try:
        from main_hrm_system import HRMIntelligenceSystem, HRMSystemConfig

        print("Lancement du système HRM Intelligence en mode console...")

        # Configuration
        config = HRMSystemConfig()
        if args.debug:
            config.debug_mode = True
            config.verbose_logging = True

        config.human_like_delays = not args.no_delays
        config.learning_enabled = not args.no_learning

        # Initialisation
        system = HRMIntelligenceSystem(
            config=config,
            player_id=args.player_id
        )

        if args.test:
            print("Mode test - execution pour 30 secondes...")
            system.start()
            import time
            time.sleep(30)
            system.stop()
        else:
            print("Mode production - Ctrl+C pour arreter")
            system.run_main_loop()

    except KeyboardInterrupt:
        print("\nArret demande par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur système: {e}")
        sys.exit(1)

def run_tests():
    """Exécute les tests d'intégration"""
    try:
        from integration_test import HRMIntegrationTester

        print("Lancement des tests d'integration HRM...")

        tester = HRMIntegrationTester()
        results = tester.run_all_tests()
        report_path = tester.generate_test_report()

        successful_tests = sum(results.values())
        total_tests = len(results)

        print(f"\nRESULTATS: {successful_tests}/{total_tests} tests reussis")
        print(f"Rapport detaille: {report_path}")

        if successful_tests == total_tests:
            print("TOUS LES TESTS ONT REUSSI!")
            sys.exit(0)
        else:
            print("Certains tests ont echoue")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Erreur tests: {e}")
        sys.exit(1)

def show_status():
    """Affiche le statut du système"""
    print("Statut HRM Intelligence")
    print("=" * 50)

    # Vérifier les fichiers
    required_files = [
        "hrm_core.py",
        "adaptive_learner.py",
        "intelligent_decision_maker.py",
        "quest_tracker.py",
        "main_hrm_system.py",
        "hrm_gui.py"
    ]

    current_dir = Path(__file__).parent

    for file in required_files:
        file_path = current_dir / file
        status = "OK" if file_path.exists() else "MANQUANT"
        print(f"{status} {file}")

    # Vérifier les dépendances
    dependencies = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("tkinter", "Tkinter"),
        ("matplotlib", "Matplotlib")
    ]

    print("\nDependances:")
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"OK {name}")
        except ImportError:
            print(f"MANQUANT {name}")

    # Vérifier les répertoires
    directories = [
        "G:/Botting/logs",
        "G:/Botting/data/hrm",
        "G:/Botting/models"
    ]

    print("\nRepertoires:")
    for directory in directories:
        dir_path = Path(directory)
        status = "OK" if dir_path.exists() else "MANQUANT"
        print(f"{status} {directory}")

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="🤖 HRM Intelligence - Bot intelligent pour TacticalBot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python launcher.py                    # Lance l'interface graphique
  python launcher.py --console          # Mode console
  python launcher.py --test             # Tests d'intégration
  python launcher.py --status           # Vérifier le statut
  python launcher.py --console --debug  # Console avec debug
        """
    )

    # Modes de lancement
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--console", action="store_true",
                           help="Lance en mode console (sans GUI)")
    mode_group.add_argument("--test", action="store_true",
                           help="Exécute les tests d'intégration")
    mode_group.add_argument("--status", action="store_true",
                           help="Affiche le statut du système")

    # Options console
    parser.add_argument("--player-id", default="tactical_bot",
                       help="ID du joueur (défaut: tactical_bot)")
    parser.add_argument("--debug", action="store_true",
                       help="Active le mode debug")
    parser.add_argument("--no-delays", action="store_true",
                       help="Désactive les délais humains")
    parser.add_argument("--no-learning", action="store_true",
                       help="Désactive l'apprentissage")

    args = parser.parse_args()

    # Header
    print("HRM Intelligence v1.0.0")
    print("Bot intelligent pour TacticalBot")
    print("=" * 50)

    try:
        if args.status:
            show_status()
        elif args.test:
            run_tests()
        elif args.console:
            launch_console(args)
        else:
            # Mode GUI par défaut
            launch_gui()

    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
    except Exception as e:
        print(f"\nErreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()