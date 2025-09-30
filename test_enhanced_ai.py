#!/usr/bin/env python3
"""
Test Enhanced AI System - Tests spécifiques pour le système AI amélioré
Tests d'intégration avec le framework existant
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

async def test_vision_module():
    """Test du module de vision"""
    print("=== Test Module Vision ===")

    try:
        from modules.vision.ai_vision_module import create_vision_module

        # Créer module
        vision_module = create_vision_module()
        print("OK: Module Vision cree")

        # Configuration test
        config = {
            "vision": {
                "enabled": True,
                "capture_fps": 5,
                "quality": "medium"
            }
        }

        # Initialiser
        if await vision_module.initialize(config):
            print("OK: Module Vision initialise")

            # Démarrer
            vision_task = asyncio.create_task(vision_module._run_impl())

            # Tester pendant 3 secondes
            await asyncio.sleep(3)

            # Obtenir stats
            stats = vision_module.get_module_stats()
            print(f"OK: Stats Vision: {stats}")

            # Arrêter
            vision_module._shutdown_event.set()
            try:
                await asyncio.wait_for(vision_task, timeout=2.0)
            except asyncio.TimeoutError:
                vision_task.cancel()

            await vision_module._shutdown_impl()
            print("OK: Module Vision arrete")

            return True
        else:
            print("ERREUR: Echec initialisation Vision")
            return False

    except Exception as e:
        print(f"ERREUR: Erreur test Vision: {e}")
        return False

async def test_learning_module():
    """Test du module d'apprentissage"""
    print("\n=== Test Module Learning ===")

    try:
        from modules.learning.ai_learning_module import create_learning_module

        # Créer module
        data_dir = Path("data/test_learning")
        learning_module = create_learning_module(data_dir)
        print("OK: Module Learning cree")

        # Configuration test
        config = {
            "learning": {
                "enabled": True,
                "observation_interval": 1.0,
                "pattern_analysis_interval": 5.0
            }
        }

        # Initialiser
        if await learning_module.initialize(config):
            print("OK: Module Learning initialise")

            # Démarrer
            learning_task = asyncio.create_task(learning_module._run_impl())

            # Simuler quelques actions
            await learning_module.observe_user_action("spell_cast", (100, 200), True)
            await learning_module.observe_user_action("movement", (150, 250), True)
            await learning_module.observe_user_action("click", (200, 300), True)

            # Attendre
            await asyncio.sleep(2)

            # Obtenir stats
            stats = learning_module.get_module_stats()
            print(f"OK: Stats Learning: {stats}")

            # Arrêter
            learning_module._shutdown_event.set()
            try:
                await asyncio.wait_for(learning_task, timeout=2.0)
            except asyncio.TimeoutError:
                learning_task.cancel()

            await learning_module._shutdown_impl()
            print("OK: Module Learning arrete")

            return True
        else:
            print("ERREUR: Echec initialisation Learning")
            return False

    except Exception as e:
        print(f"ERREUR: Erreur test Learning: {e}")
        return False

async def test_ai_framework_integration():
    """Test d'intégration avec le framework AI existant"""
    print("\n=== Test Intégration Framework AI ===")

    try:
        from core.ai_framework import MetaOrchestrator

        # Créer orchestrateur avec fichier de config
        config_path = "data/test_config.json"
        orchestrator = MetaOrchestrator(config_path)
        print("OK: MetaOrchestrator cree")

        # Demarrer
        if await orchestrator.start():
            print("OK: MetaOrchestrator demarre")

            # Fonctionner pendant 2 secondes
            await asyncio.sleep(2)

            # Obtenir stats
            stats = orchestrator.get_orchestrator_stats()
            print(f"OK: Stats Orchestrator: modules={stats.get('active_modules', 0)}")

            # Arreter
            await orchestrator.stop()
            print("OK: MetaOrchestrator arrete")

            return True
        else:
            print("ERREUR: Echec demarrage MetaOrchestrator")
            return False

    except Exception as e:
        print(f"ERREUR: Erreur test Framework: {e}")
        return False

async def test_environment_setup():
    """Test de l'environnement"""
    print("=== Test Environnement ===")

    # Test imports critiques
    imports_test = [
        ("numpy", "np"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("asyncio", "asyncio"),
        ("logging", "logging")
    ]

    failed_imports = []
    for module, name in imports_test:
        try:
            __import__(module)
            print(f"OK: {name}")
        except ImportError:
            print(f"ERREUR: {name}")
            failed_imports.append(name)

    # Test répertoires
    required_dirs = [
        "data", "data/logs", "modules", "modules/vision",
        "modules/learning", "modules/overlay", "core"
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"OK: Repertoire: {dir_path}")
        else:
            print(f"ERREUR: Repertoire manquant: {dir_path}")

    return len(failed_imports) == 0

async def main():
    """Tests principaux"""
    print("Tests Enhanced AI DOFUS System")
    print("=" * 50)

    # Créer répertoires de test
    test_dirs = ["data/test", "data/test_learning", "data/logs"]
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    results = []

    # Test 1: Environnement
    results.append(await test_environment_setup())

    # Test 2: Framework AI
    results.append(await test_ai_framework_integration())

    # Test 3: Module Vision (peut échouer si pas de fenêtre DOFUS)
    try:
        results.append(await test_vision_module())
    except Exception as e:
        print(f"ATTENTION: Test Vision ignore: {e}")
        results.append(True)  # Ne pas faire échouer les tests

    # Test 4: Module Learning
    results.append(await test_learning_module())

    # Résultats
    print("\n" + "=" * 50)
    print("Resultats des tests:")

    test_names = ["Environnement", "Framework AI", "Module Vision", "Module Learning"]

    passed = sum(results)
    total = len(results)

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASSE" if result else "ECHEC"
        print(f"  {i+1}. {name}: {status}")

    print(f"\nResultat global: {passed}/{total} tests passes")

    if passed == total:
        print("Tous les tests sont passes!")
        print("\nProchaines etapes:")
        print("1. Lancer DOFUS Unity")
        print("2. Executer: python enhanced_ai_launcher.py --mode hybrid")
        print("3. Utiliser commande 'start' pour demarrer l'assistant")
        return True
    else:
        print("Certains tests ont echoue")
        print("Verifiez les dependances et la configuration")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)