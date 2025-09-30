"""
Lanceur Unifié - DOFUS Unity World Model AI
Script principal pour démarrer tous les modes disponibles
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent))

def print_banner():
    """Affiche la bannière du système"""
    banner = """
=======================================================
DOFUS UNITY WORLD MODEL AI - VERSION 2025.1.0
Systeme d'Intelligence Artificielle Avancee
=======================================================
"""
    print(banner)

def check_dependencies():
    """Vérifie les dépendances système"""
    print("[INIT] Vérification des dépendances...")

    required_packages = [
        'opencv-python', 'numpy', 'pillow', 'requests',
        'psutil', 'sqlite3'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            elif package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                from PIL import Image
            else:
                __import__(package.replace('-', '_'))
            print(f"  [OK] {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  [MISSING] {package}")

    if missing_packages:
        print(f"\n[WARNING] Packages manquants: {', '.join(missing_packages)}")
        print("Installez-les avec: pip install " + " ".join(missing_packages))
        return False

    print("[SUCCESS] Toutes les dépendances sont installées")
    return True

def run_system_tests():
    """Lance les tests système complets"""
    print("\n[TESTS] Lancement des tests système...")

    try:
        result = subprocess.run([
            sys.executable, "tests/test_complete_system.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode == 0:
            print("[SUCCESS] Tests système réussis")
            return True
        else:
            print(f"[ERROR] Tests échoués - Code: {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Dernières 500 chars
            if result.stderr:
                print("STDERR:", result.stderr[-500:])
            return False

    except Exception as e:
        print(f"[ERROR] Erreur tests: {e}")
        return False

def run_synthetic_environment():
    """Lance l'environnement de test synthétique"""
    print("\n[SYNTHETIC] Génération environnement synthétique...")

    try:
        result = subprocess.run([
            sys.executable, "tests/test_synthetic_environment.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode == 0:
            print("[SUCCESS] Environnement synthétique créé")
            return True
        else:
            print(f"[ERROR] Erreur environnement synthétique")
            return False

    except Exception as e:
        print(f"[ERROR] Erreur synthétique: {e}")
        return False

def run_passive_learning(duration: int = 300):
    """Lance le mode d'apprentissage passif"""
    print(f"\n[PASSIVE] Démarrage apprentissage passif ({duration}s)...")

    try:
        # Import du moteur d'apprentissage passif
        from core.vision_engine.passive_learning_mode import get_passive_learning_engine

        engine = get_passive_learning_engine()

        print("[INFO] Recherche de fenêtre DOFUS...")
        session_id = engine.start_passive_learning_session("IOPS", 150, "unified_launcher")

        print(f"[RUNNING] Session active: {session_id}")
        print(f"[INFO] Apprentissage en cours... ({duration}s)")
        print("[INFO] Appuyez sur Ctrl+C pour arrêter")

        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\n[USER] Arrêt demandé par l'utilisateur")

        session = engine.stop_passive_learning_session()

        print(f"[COMPLETE] Session terminée:")
        print(f"  - Captures: {session.total_captures}")
        print(f"  - Patterns appris: {session.patterns_learned}")
        print(f"  - Qualité: {session.data_quality_score:.2f}")

        return True

    except Exception as e:
        print(f"[ERROR] Erreur apprentissage passif: {e}")
        return False

def run_assistant_interface():
    """Lance l'interface assistante GUI"""
    print("\n[GUI] Lancement interface assistante...")

    try:
        subprocess.run([
            sys.executable, "assistant_interface/intelligent_assistant.py"
        ], cwd=Path(__file__).parent)
        return True

    except Exception as e:
        print(f"[ERROR] Erreur interface: {e}")
        return False

def run_api_tests():
    """Lance les tests des APIs externes"""
    print("\n[API] Test des connecteurs API externes...")

    try:
        from core.external_integration.dofus_api_connector import get_unified_api_manager

        manager = get_unified_api_manager()

        # Test Dofapi
        print("  Testing Dofapi...")
        response = manager.get_class_spells_comprehensive("iop")
        if response.success:
            print(f"    [OK] Sorts Iop: {len(response.data)} sorts")
        else:
            print(f"    [ERROR] {response.error_message}")

        # Test recherche item
        print("  Testing Item Search...")
        response = manager.get_comprehensive_item_data("Dofus")
        if response.success:
            print(f"    [OK] Items trouvés: {len(response.data)}")
        else:
            print(f"    [ERROR] {response.error_message}")

        stats = manager.get_statistics()
        print(f"  [STATS] Requêtes: {stats['current_session']}")

        return True

    except Exception as e:
        print(f"[ERROR] Erreur tests API: {e}")
        return False

def run_ganymede_tests():
    """Lance les tests du connecteur Ganymede"""
    print("\n[GANYMEDE] Test du connecteur Ganymede...")

    try:
        from core.external_integration.ganymede_guides_connector import get_ganymede_connector

        connector = get_ganymede_connector()

        # Test récupération guides
        print("  Testing Guides List...")
        guides_list = connector.scraper.get_guides_list(category="quetes")
        print(f"    [OK] Guides trouvés: {len(guides_list)}")

        if guides_list:
            # Test contenu guide
            print("  Testing Guide Content...")
            first_guide = guides_list[0]
            guide_content = connector.scraper.get_guide_content(first_guide['url'])
            if guide_content:
                print(f"    [OK] Guide chargé: {guide_content.title}")
                print(f"    [INFO] Étapes: {len(guide_content.steps)}")
            else:
                print("    [WARNING] Contenu guide non chargé")

        # Test recherche
        print("  Testing Guide Search...")
        search_results = connector.search_guides("temple")
        print(f"    [OK] Résultats recherche 'temple': {len(search_results)}")

        # Test recommandations contextuelles
        print("  Testing Contextual Recommendations...")
        from core.knowledge_base.knowledge_integration import get_knowledge_base
        kb = get_knowledge_base()
        recommendations = kb.get_contextual_guide_recommendations()

        if recommendations.success:
            rec_count = len(recommendations.data.get('recommendations', []))
            print(f"    [OK] Recommandations contextuelles: {rec_count}")
        else:
            print("    [WARNING] Recommandations non disponibles")

        return True

    except Exception as e:
        print(f"[ERROR] Erreur tests Ganymede: {e}")
        return False

def run_reverse_engineering():
    """Lance le framework de reverse engineering"""
    print("\n[REVERSE] Analyse applications externes...")

    try:
        from core.external_integration.reverse_engineering_framework import get_reverse_engineering_framework

        orchestrator = get_reverse_engineering_framework()

        # Découvrir processus
        processes = orchestrator.discover_target_applications()

        if processes:
            print(f"[FOUND] {len(processes)} processus cibles trouvés")

            # Analyser le premier processus
            print("[ANALYZE] Analyse du premier processus...")
            analysis = orchestrator.analyze_application(processes[0], capture_duration=60)

            print(f"[COMPLETE] Analyse terminée - APIs: {len(analysis.get('discovered_apis', {}))}")

            # Générer guide
            guide = orchestrator.generate_integration_guide(analysis['session_id'])
            print("[GUIDE] Guide d'intégration généré")

        else:
            print("[INFO] Aucun processus cible trouvé")
            print("[INFO] Démarrez Ganymede, DOFUS Guide ou DOFUS pour l'analyse")

        return True

    except Exception as e:
        print(f"[ERROR] Erreur reverse engineering: {e}")
        return False

def show_menu():
    """Affiche le menu principal"""
    menu = """
[MENU] Modes disponibles:

1. [TESTS]     Tests système complets
2. [SYNTHETIC] Environnement de test synthétique
3. [PASSIVE]   Apprentissage passif DOFUS Unity
4. [GUI]       Interface assistante complète
5. [API]       Tests connecteurs API externes
6. [GANYMEDE]  Tests connecteur guides Ganymede
7. [REVERSE]   Reverse engineering applications
8. [ALL]       Exécuter tous les modules
9. [EXIT]      Quitter

Choisissez une option (1-9):"""

    print(menu)

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="DOFUS Unity World Model AI - Lanceur Unifié")
    parser.add_argument("--mode", choices=[
        "tests", "synthetic", "passive", "gui", "api", "ganymede", "reverse", "all"
    ], help="Mode à lancer directement")
    parser.add_argument("--duration", type=int, default=300,
                       help="Durée apprentissage passif (secondes)")
    parser.add_argument("--skip-deps", action="store_true",
                       help="Ignorer vérification dépendances")

    args = parser.parse_args()

    print_banner()

    # Vérifier dépendances
    if not args.skip_deps:
        if not check_dependencies():
            print("\n[ERROR] Dépendances manquantes - arrêt")
            return 1

    # Mode direct si spécifié
    if args.mode:
        success = execute_mode(args.mode, args.duration)
        return 0 if success else 1

    # Menu interactif
    while True:
        show_menu()

        try:
            choice = input().strip()

            if choice == "9" or choice.lower() == "exit":
                print("\n[EXIT] Au revoir !")
                break

            mode_map = {
                "1": "tests",
                "2": "synthetic",
                "3": "passive",
                "4": "gui",
                "5": "api",
                "6": "ganymede",
                "7": "reverse",
                "8": "all"
            }

            if choice in mode_map:
                mode = mode_map[choice]
                print(f"\n[EXEC] Exécution mode: {mode}")
                success = execute_mode(mode, args.duration)

                if success:
                    print(f"[SUCCESS] Mode {mode} terminé avec succès")
                else:
                    print(f"[ERROR] Mode {mode} a échoué")

                input("\nAppuyez sur Entrée pour continuer...")
            else:
                print("\n[ERROR] Option invalide")

        except KeyboardInterrupt:
            print("\n\n[EXIT] Interruption utilisateur - Au revoir !")
            break
        except Exception as e:
            print(f"\n[ERROR] Erreur: {e}")

    return 0

def execute_mode(mode: str, duration: int = 300) -> bool:
    """Exécute un mode spécifique"""

    if mode == "tests":
        return run_system_tests()

    elif mode == "synthetic":
        return run_synthetic_environment()

    elif mode == "passive":
        return run_passive_learning(duration)

    elif mode == "gui":
        return run_assistant_interface()

    elif mode == "api":
        return run_api_tests()

    elif mode == "ganymede":
        return run_ganymede_tests()

    elif mode == "reverse":
        return run_reverse_engineering()

    elif mode == "all":
        print("[ALL] Exécution de tous les modules...")

        results = {}
        results["tests"] = run_system_tests()
        results["synthetic"] = run_synthetic_environment()
        results["api"] = run_api_tests()
        results["ganymede"] = run_ganymede_tests()

        # Afficher résumé
        print("\n[SUMMARY] Résultats:")
        for mode_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  - {mode_name}: {status}")

        return all(results.values())

    else:
        print(f"[ERROR] Mode inconnu: {mode}")
        return False

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n[FATAL] Erreur fatale: {e}")
        sys.exit(1)