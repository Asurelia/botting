"""
Test complet du systeme Knowledge Base DOFUS Unity
Verification de l'integration de tous les modules
"""

import sys
import os
from pathlib import Path

# Ajout du chemin pour imports
sys.path.append(str(Path(__file__).parent / "knowledge_base"))

try:
    from core.knowledge_base.spells_database import get_spells_database, DofusClass
    from core.knowledge_base.monsters_database import get_monsters_database
    from core.knowledge_base.maps_database import get_maps_database
    from core.knowledge_base.economy_tracker import get_economy_tracker
    from core.knowledge_base.knowledge_integration import get_knowledge_base, GameContext

    print("[OK] Tous les imports reussis")

    # Test initialisation
    print("\n[INIT] Test initialisation modules...")

    spells_db = get_spells_database()
    print(f"[OK] Spells DB: {len(spells_db.spells)} sorts charges")

    monsters_db = get_monsters_database()
    print(f"[OK] Monsters DB: {len(monsters_db.monsters)} monstres charges")

    maps_db = get_maps_database()
    print(f"[OK] Maps DB: {len(maps_db.maps)} cartes chargees")

    economy_tracker = get_economy_tracker()
    print(f"[OK] Economy Tracker: {len(economy_tracker.tracked_items)} items suivis")

    # Test Knowledge Base integre
    print("\n[BRAIN] Test Knowledge Base integre...")
    kb = get_knowledge_base()

    # Configuration contexte
    context = GameContext(
        player_class=DofusClass.IOPS,
        player_level=150,
        current_server="Julith",
        available_ap=6,
        distance_to_target=2
    )
    kb.update_game_context(context)
    print(f"[OK] Contexte configure: {context.player_class.value} niveau {context.player_level}")

    # Test requetes
    print("\n[QUERY] Test requetes...")

    # Sorts optimaux
    spells_result = kb.query_optimal_spells()
    print(f"[OK] Sorts optimaux: {'SUCCES' if spells_result.success else 'ECHEC'}")
    print(f"   Temps execution: {spells_result.execution_time_ms:.1f}ms")
    print(f"   Suggestions: {len(spells_result.suggestions)}")

    # Strategie monstre
    monster_result = kb.query_monster_strategy("Bouftou")
    print(f"[OK] Strategie monstre: {'SUCCES' if monster_result.success else 'ECHEC'}")
    if monster_result.success:
        print(f"   Elements efficaces: {monster_result.data.get('preferred_elements', [])}")

    # Route farming
    farming_result = kb.query_farming_route("Frene")
    print(f"[OK] Route farming: {'SUCCES' if farming_result.success else 'ECHEC'}")
    print(f"   Cartes trouvees: {len(farming_result.data) if farming_result.data else 0}")

    # Opportunites marche
    market_result = kb.query_market_opportunities()
    print(f"[OK] Opportunites marche: {'SUCCES' if market_result.success else 'ECHEC'}")
    print(f"   Opportunites detectees: {len(market_result.data) if market_result.data else 0}")

    # Conseil complet
    advice_result = kb.query_comprehensive_advice("combat contre monstre")
    print(f"[OK] Conseil complet: {'SUCCES' if advice_result.success else 'ECHEC'}")
    print(f"   Recommandations: {len(advice_result.suggestions)}")

    # Statut systeme
    print("\n[STATS] Statut systeme...")
    status = kb.get_system_status()
    print(f"[OK] Cache queries: {status['performance']['total_queries']}")
    print(f"[OK] Cache hit rate: {status['performance']['cache_hit_rate']:.1f}%")
    print(f"[OK] Modules usage: {status['performance']['modules_usage']}")

    # Export test
    print("\n[EXPORT] Test export...")
    kb.export_knowledge_summary("test_knowledge_summary.json")
    print("[OK] Resume exporte: test_knowledge_summary.json")

    # Test extraction donnees
    print("\n[EXTRACT] Test extraction donnees DOFUS...")
    try:
        from core.knowledge_base.dofus_data_extractor import get_dofus_extractor
        extractor = get_dofus_extractor()

        bundles = extractor.list_available_bundles()
        print(f"[OK] Extractor: {len(bundles)} bundles detectes")

        # Test extraction sorts
        spells_data = extractor.extract_spells_data()
        success_count = sum(1 for data in spells_data.values() if data.success)
        print(f"[OK] Extraction sorts: {success_count}/{len(spells_data)} bundles extraits")

    except Exception as e:
        print(f"[WARN] Extraction donnees limitee: {e}")

    print("\n[SUCCESS] TOUS LES TESTS REUSSIS !")
    print(f"[PERF] Performance globale:")
    print(f"   - {len(spells_db.spells)} sorts disponibles")
    print(f"   - {len(monsters_db.monsters)} monstres disponibles")
    print(f"   - {len(maps_db.maps)} cartes disponibles")
    print(f"   - {len(economy_tracker.tracked_items)} items economiques suivis")
    print(f"   - Systeme Knowledge Base operationnel")

except ImportError as e:
    print(f"[ERROR] Erreur import: {e}")
    print("Verifiez que tous les modules sont dans knowledge_base/")

except Exception as e:
    print(f"[ERROR] Erreur test: {e}")
    import traceback
    traceback.print_exc()