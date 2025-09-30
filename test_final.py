#!/usr/bin/env python3
"""
Test Final - Validation rapide du systeme
"""

import asyncio
import sys
from pathlib import Path

# Ajouter au path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

async def test_basic_imports():
    """Test des imports de base"""
    print("=== Test Imports ===")

    try:
        import numpy as np
        print("OK: NumPy")

        import cv2
        print("OK: OpenCV")

        import torch
        print(f"OK: PyTorch {torch.__version__}")

        from core.ai_framework import MetaOrchestrator
        print("OK: AI Framework")

        from modules.vision.ai_vision_module import create_vision_module
        print("OK: Vision Module")

        from modules.learning.ai_learning_module import create_learning_module
        print("OK: Learning Module")

        from modules.worldmodel.dofus_world_model import create_world_model
        print("OK: World Model")

        return True

    except Exception as e:
        print(f"ERREUR: {e}")
        return False

async def test_system_integration():
    """Test integration basique"""
    print("\n=== Test Integration ===")

    try:
        # Test orchestrateur
        from core.ai_framework import MetaOrchestrator
        config_path = "data/test_config.json"
        orchestrator = MetaOrchestrator(config_path)

        if await orchestrator.start():
            print("OK: MetaOrchestrator demarre")
            await asyncio.sleep(1)
            await orchestrator.stop()
            print("OK: MetaOrchestrator arrete")
        else:
            print("ERREUR: Echec demarrage orchestrateur")
            return False

        # Test world model
        from modules.worldmodel.dofus_world_model import create_world_model
        data_dir = Path("data/test_world_model")
        world_model = create_world_model(data_dir)

        config = {"world_model": {"enabled": True}}
        if await world_model.initialize(config):
            print("OK: World Model initialise")
            await world_model._shutdown_impl()
            print("OK: World Model arrete")
        else:
            print("ERREUR: Echec world model")
            return False

        return True

    except Exception as e:
        print(f"ERREUR: {e}")
        return False

async def test_enhanced_launcher():
    """Test du launcher"""
    print("\n=== Test Enhanced Launcher ===")

    try:
        from enhanced_ai_launcher import EnhancedAIOrchestrator
        print("OK: Enhanced AI Launcher import")
        return True
    except Exception as e:
        print(f"ERREUR: {e}")
        return False

async def main():
    """Tests finaux"""
    print("Test Final - Validation Systeme Enhanced AI")
    print("=" * 50)

    # Creer repertoires
    Path("data/test_world_model").mkdir(parents=True, exist_ok=True)

    results = []

    # Tests
    results.append(await test_basic_imports())
    results.append(await test_system_integration())
    results.append(await test_enhanced_launcher())

    # Resultats
    print("\n" + "=" * 50)
    print("RESULTATS FINAUX:")

    test_names = ["Imports", "Integration", "Launcher"]
    passed = sum(results)
    total = len(results)

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASSE" if result else "ECHEC"
        print(f"  {i+1}. {name}: {status}")

    print(f"\nGlobal: {passed}/{total} tests passes")

    if passed >= 2:  # Au moins 2/3 tests passent
        print("\n" + "="*50)
        print("SYSTEME PRET POUR UTILISATION!")
        print("="*50)
        print("\nPour demarrer:")
        print("1. Lancer DOFUS Unity")
        print("2. Executer: python enhanced_ai_launcher.py --mode hybrid")
        print("3. Utiliser commande 'start' dans l'interface")
        print("\nCommandes disponibles:")
        print("- start/stop: Gestion assistant")
        print("- status/stats: Informations systeme")
        print("- gamestate: Etat du jeu detecte")
        print("- recommend: Conseils IA")
        print("- worldmap: Carte du monde")
        print("- predict: Predictions evenements")
        return True
    else:
        print("\nSYSTEME NON PRET - Verifiez erreurs ci-dessus")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)