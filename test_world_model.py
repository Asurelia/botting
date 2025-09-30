#!/usr/bin/env python3
"""
Test World Model System - Tests specifiques pour le world model DOFUS
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

async def test_world_model_creation():
    """Test de creation du world model"""
    print("=== Test Creation World Model ===")

    try:
        from modules.worldmodel.dofus_world_model import create_world_model

        # Creer module
        data_dir = Path("data/test_world_model")
        world_model = create_world_model(data_dir)
        print("OK: World Model cree")

        # Configuration test
        config = {
            "world_model": {
                "enabled": True,
                "spatial_memory": True,
                "temporal_predictions": True,
                "danger_assessment": True
            }
        }

        # Initialiser
        if await world_model.initialize(config):
            print("OK: World Model initialise")

            # Test ajout map cell
            from modules.worldmodel.dofus_world_model import MapCellType
            world_model.update_map_cell("Test_Zone", 100, 200, MapCellType.WALKABLE, True)
            print("OK: Map cell ajoutee")

            # Test ajout entite
            from modules.worldmodel.dofus_world_model import WorldEntity, WorldPosition
            entity = WorldEntity(
                entity_id="test_tofu_1",
                entity_type="monster",
                name="Tofu",
                position=WorldPosition("Test_Zone", 150, 250)
            )
            world_model.add_entity(entity)
            print("OK: Entite ajoutee")

            # Test pathfinding
            path = await world_model.get_optimal_path(200, 200)
            print(f"OK: Pathfinding: {len(path) if path else 0} points")

            # Test opportunities
            opportunities = await world_model.get_nearby_opportunities()
            print(f"OK: Opportunities: {len(opportunities)} trouvees")

            # Stats
            stats = world_model.get_module_stats()
            print(f"OK: Stats: {stats}")

            # Arreter
            await world_model._shutdown_impl()
            print("OK: World Model arrete")

            return True
        else:
            print("ERREUR: Echec initialisation World Model")
            return False

    except Exception as e:
        print(f"ERREUR: Test World Model: {e}")
        return False

async def test_world_model_integration():
    """Test integration avec le framework AI"""
    print("\n=== Test Integration World Model ===")

    try:
        from core.ai_framework import MetaOrchestrator
        from modules.worldmodel.dofus_world_model import create_world_model

        # Creer orchestrateur
        config_path = "data/test_config.json"
        orchestrator = MetaOrchestrator(config_path)
        print("OK: MetaOrchestrator cree")

        # Creer world model
        data_dir = Path("data/test_world_model")
        world_model = create_world_model(data_dir)
        print("OK: World Model cree")

        # Demarrer orchestrateur
        if await orchestrator.start():
            print("OK: MetaOrchestrator initialise")

            # Ajouter world model comme module
            await orchestrator.register_module(world_model)
            print("OK: World Model ajoute a l'orchestrateur")

            print("OK: Orchestrateur avec World Model demarre")

            # Fonctionner 2 secondes
            await asyncio.sleep(2)

            # Stats
            stats = orchestrator.get_orchestrator_stats()
            print(f"OK: Stats integration: {stats.get('active_modules', 0)} modules")

            # Arreter
            await orchestrator.stop()
            print("OK: Integration stoppee")

            return True
        else:
            print("ERREUR: Echec demarrage orchestrateur")
            return False

    except Exception as e:
        print(f"ERREUR: Test integration: {e}")
        return False

async def test_world_model_persistence():
    """Test de persistance des donnees"""
    print("\n=== Test Persistance World Model ===")

    try:
        from modules.worldmodel.dofus_world_model import create_world_model

        data_dir = Path("data/test_world_model")

        # Premier world model - ajouter donnees
        world_model1 = create_world_model(data_dir)
        config = {"world_model": {"enabled": True}}

        await world_model1.initialize(config)
        from modules.worldmodel.dofus_world_model import MapCellType, WorldEntity, WorldPosition
        world_model1.update_map_cell("Persistent_Zone", 300, 400, MapCellType.WALKABLE, True)
        entity = WorldEntity(
            entity_id="test_tree_99",
            entity_type="resource",
            name="Tree",
            position=WorldPosition("Persistent_Zone", 350, 450)
        )
        world_model1.add_entity(entity)
        await world_model1._shutdown_impl()
        print("OK: Donnees sauvegardees")

        # Deuxieme world model - charger donnees
        world_model2 = create_world_model(data_dir)
        await world_model2.initialize(config)

        # Verifier que les donnees sont conservees
        stats = world_model2.get_module_stats()
        if stats.get("maps_known", 0) > 0 and stats.get("entities_tracked", 0) > 0:
            print("OK: Donnees rechargees avec succes")
            result = True
        else:
            print("ERREUR: Donnees non conservees")
            result = False

        await world_model2._shutdown_impl()
        return result

    except Exception as e:
        print(f"ERREUR: Test persistance: {e}")
        return False

async def main():
    """Tests principaux world model"""
    print("Tests World Model DOFUS System")
    print("=" * 50)

    # Creer repertoires de test
    test_dirs = ["data/test_world_model"]
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    results = []

    # Tests
    results.append(await test_world_model_creation())
    results.append(await test_world_model_integration())
    results.append(await test_world_model_persistence())

    # Resultats
    print("\n" + "=" * 50)
    print("Resultats des tests World Model:")

    test_names = ["Creation", "Integration", "Persistance"]

    passed = sum(results)
    total = len(results)

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASSE" if result else "ECHEC"
        print(f"  {i+1}. {name}: {status}")

    print(f"\nResultat global: {passed}/{total} tests passes")

    if passed == total:
        print("World Model pret a l'utilisation!")
        return True
    else:
        print("Certains tests World Model ont echoue")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)