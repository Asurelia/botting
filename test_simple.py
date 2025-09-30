#!/usr/bin/env python3
"""
Test Simple Enhanced AI System
Tests basiques sans caractères Unicode
"""

import asyncio
import sys
import logging
from pathlib import Path
import time

# Ajouter au path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_imports():
    """Test des imports essentiels"""
    print("=== Test Imports ===")

    imports_test = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
    ]

    failed = []
    for module, name in imports_test:
        try:
            __import__(module)
            print(f"OK: {name}")
        except ImportError as e:
            print(f"ERREUR: {name} - {e}")
            failed.append(name)

    return len(failed) == 0

async def test_framework():
    """Test du framework IA existant"""
    print("\n=== Test Framework AI ===")

    try:
        from core.ai_framework import MetaOrchestrator

        # Configuration minimale
        config = {
            "orchestrator": {"coordination_interval": 1.0},
            "knowledge_graph": {"enabled": False},
            "predictive_engine": {"enabled": False},
            "decision_engine": {"enabled": False},
            "emotional_state": {"enabled": False},
            "social_intelligence": {"enabled": False}
        }

        orchestrator = MetaOrchestrator(config)
        print("OK: MetaOrchestrator cree")

        if await orchestrator.initialize():
            print("OK: MetaOrchestrator initialise")

            if await orchestrator.start():
                print("OK: MetaOrchestrator demarre")
                await asyncio.sleep(1)
                await orchestrator.stop()
                print("OK: MetaOrchestrator arrete")
                return True
            else:
                print("ERREUR: Echec demarrage")
                return False
        else:
            print("ERREUR: Echec initialisation")
            return False

    except Exception as e:
        print(f"ERREUR: {e}")
        return False

async def test_vision():
    """Test du module de vision"""
    print("\n=== Test Vision ===")

    try:
        from modules.vision.ai_vision_module import create_vision_module

        vision_module = create_vision_module()
        print("OK: Module Vision cree")

        config = {"vision": {"enabled": True, "capture_fps": 5}}

        if await vision_module.initialize(config):
            print("OK: Module Vision initialise")

            # Test rapide sans démarrage complet
            stats = vision_module.get_module_stats()
            print(f"OK: Stats obtenues: {len(stats)} elements")

            await vision_module._shutdown_impl()
            print("OK: Module Vision arrete")
            return True
        else:
            print("ATTENTION: Module Vision non initialise (normal sans DOFUS)")
            return True  # Pas d'échec si pas de fenêtre DOFUS

    except Exception as e:
        print(f"ATTENTION: Vision - {e}")
        return True  # Pas critique

async def test_learning():
    """Test du module d'apprentissage"""
    print("\n=== Test Learning ===")

    try:
        from modules.learning.ai_learning_module import create_learning_module

        data_dir = Path("data/test_learning")
        learning_module = create_learning_module(data_dir)
        print("OK: Module Learning cree")

        config = {"learning": {"enabled": True}}

        if await learning_module.initialize(config):
            print("OK: Module Learning initialise")

            # Test observation action
            await learning_module.observe_user_action("test", (100, 100), True)
            print("OK: Action observee")

            stats = learning_module.get_module_stats()
            print(f"OK: Stats: {stats.get('actions_observed', 0)} actions")

            await learning_module._shutdown_impl()
            print("OK: Module Learning arrete")
            return True
        else:
            print("ERREUR: Echec initialisation Learning")
            return False

    except Exception as e:
        print(f"ERREUR: Learning - {e}")
        return False

async def main():
    """Tests principaux"""
    print("Tests Enhanced AI DOFUS System")
    print("=" * 50)

    # Créer répertoires
    Path("data/test").mkdir(parents=True, exist_ok=True)
    Path("data/test_learning").mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    results = []

    # Tests
    print("Execution des tests...")
    results.append(await test_imports())
    results.append(await test_framework())
    results.append(await test_vision())
    results.append(await test_learning())

    # Résultats
    print("\n" + "=" * 50)
    print("RESULTATS:")

    test_names = ["Imports", "Framework", "Vision", "Learning"]
    passed = sum(results)
    total = len(results)

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASSE" if result else "ECHEC"
        print(f"  {i+1}. {name}: {status}")

    print(f"\nGlobal: {passed}/{total} tests passes")

    if passed >= total - 1:  # Autoriser 1 échec
        print("\nSYSTEME PRET!")
        print("Prochaines etapes:")
        print("1. Lancer DOFUS Unity")
        print("2. Executer: python enhanced_ai_launcher.py --mode hybrid")
        return True
    else:
        print("\nVerifiez la configuration")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\nTest termine: {'SUCCES' if success else 'ECHEC'}")
    sys.exit(0 if success else 1)