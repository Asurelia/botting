#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Test Runner - NO EMOJIS for Windows compatibility
Execute ALL bot tests WITHOUT requiring Dofus
"""

import sys
import os
import time
import unittest
from pathlib import Path

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Run all test suites"""

    print("\n" + "=" * 80)
    print(" DOFUS ALPHASTAR 2025 - SUITE DE TESTS COMPLETE")
    print(" Tests SANS Dofus (aucun risque de ban)")
    print("=" * 80 + "\n")

    start_time = time.time()

    # Discover all tests
    loader = unittest.TestLoader()
    tests_dir = Path(__file__).parent
    suite = loader.discover(str(tests_dir), pattern='test_*.py')

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    duration = time.time() - start_time

    # Summary
    print("\n" + "=" * 80)
    print(" RESUME GLOBAL")
    print("=" * 80)
    print(f"\nDuree totale: {duration:.2f}s")
    print(f"Tests executes: {result.testsRun}")
    print(f"Succes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Echecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n[OK] TOUS LES TESTS SONT PASSES")
        print("\nProchaines etapes:")
        print("  1. Verifier que toutes les dependances sont installees")
        print("  2. Lancer calibration: python launch_safe.py --calibrate")
        print("  3. Tester en mode observation: python launch_safe.py --observe 10")
    else:
        print("\n[!!] CERTAINS TESTS ONT ECHOUE")
        print("\nActions recommandees:")
        print("  1. Installer les dependances: pip install -r requirements.txt")
        print("  2. Verifier les erreurs ci-dessus")
        print("  3. Relancer les tests")

    print("=" * 80 + "\n")

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_all_tests())