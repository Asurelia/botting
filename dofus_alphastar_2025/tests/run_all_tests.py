#!/usr/bin/env python3
"""
Master Test Runner - Execute ALL bot tests
Tests everything WITHOUT requiring Dofus to be running
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """Print formatted header"""
    print("\n")
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_test_module(module_name: str, description: str):
    """Run a test module and return success status"""
    print_header(f"[TEST] {description}")

    try:
        if module_name == "test_imports":
            from tests.test_imports import run_import_tests
            return run_import_tests()

        elif module_name == "test_calibration":
            from tests.test_calibration import run_calibration_tests
            return run_calibration_tests()

        elif module_name == "test_map_system":
            from tests.test_map_system import run_map_tests
            return run_map_tests()

        elif module_name == "test_dofusdb":
            from tests.test_dofusdb import run_dofusdb_tests
            return run_dofusdb_tests()

        elif module_name == "test_safety":
            from tests.test_safety import run_safety_tests
            return run_safety_tests()

        else:
            print(f"[ERROR] Module de test inconnu: {module_name}")
            return False

    except Exception as e:
        print(f"\n[ERROR] ERREUR lors de l'exécution de {module_name}:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner"""
    print("\n")
    print("=" * 80)
    print("  DOFUS ALPHASTAR 2025 - SUITE DE TESTS COMPLETE")
    print("  Tests SANS Dofus (aucun risque de ban)")
    print("=" * 80)

    start_time = time.time()

    # Test modules to run
    test_modules = [
        ("test_imports", "Vérification des imports et dépendances"),
        ("test_safety", "Tests des systèmes de sécurité"),
        ("test_calibration", "Tests du système de calibration"),
        ("test_map_system", "Tests du système de cartes"),
        ("test_dofusdb", "Tests du client DofusDB"),
    ]

    results = {}

    # Run all tests
    for module_name, description in test_modules:
        success = run_test_module(module_name, description)
        results[module_name] = success
        time.sleep(0.5)  # Small pause between test suites

    # Summary
    duration = time.time() - start_time

    print("\n")
    print("=" * 80)
    print("  RESUME GLOBAL DES TESTS")
    print("=" * 80)

    print(f"\nDuree totale: {duration:.2f}s\n")

    all_passed = True
    for module_name, description in test_modules:
        success = results.get(module_name, False)
        status = "[OK] SUCCES" if success else "[!!] ECHEC"
        print(f"  {status}  {description}")

        if not success:
            all_passed = False

    print("\n" + "=" * 80)

    if all_passed:
        print("TOUS LES TESTS SONT PASSES")
        print("\nProchaines etapes:")
        print("   1. Verifier que toutes les dependances sont installees")
        print("   2. Lancer calibration: python launch_safe.py --calibrate")
        print("   3. Tester en mode observation: python launch_safe.py --observe 10")
        print("   4. Analyser les logs avant d'activer le mode reel")
    else:
        print("CERTAINS TESTS ONT ECHOUE")
        print("\nActions recommandees:")
        print("   1. Installer les dependances manquantes: pip install -r requirements.txt")
        print("   2. Verifier les erreurs ci-dessus")
        print("   3. Relancer les tests apres corrections")

    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())