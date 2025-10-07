#!/usr/bin/env python3
"""
DOFUS AlphaStar - Lancement SÉCURISÉ
Mode observation OBLIGATOIRE pour premiers tests

[WARNING] AVERTISSEMENT:
- Utiliser UNIQUEMENT sur compte jetable
- Mode observation activé par défaut
- Aucune action exécutée (observation seulement)
"""

import sys
import argparse
import logging
from pathlib import Path

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.safety import create_observation_mode, create_safety_manager
from core.calibration import create_calibrator
from core.map_system import create_map_graph
from core.external_data import create_dofusdb_client

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def print_banner():
    """Affiche le banner de sécurité"""
    banner = """
===============================================================================

           DOFUS ALPHASTAR 2025 - MODE SECURISE

  AVERTISSEMENTS CRITIQUES:

  1. Utiliser UNIQUEMENT sur compte jetable
  2. Mode observation active par defaut (AUCUNE action)
  3. Premiers tests: 5-10 minutes MAX
  4. Analyser les logs AVANT d'activer le mode reel

  Features disponibles:
   - Calibration automatique (--calibrate)
   - Mode observation (defaut)
   - Map discovery (--explore)
   - DofusDB integration

===============================================================================
    """
    print(banner)

def run_calibration():
    """Lance la calibration automatique"""
    logger.info("=" * 70)
    logger.info("CALIBRATION AUTOMATIQUE")
    logger.info("=" * 70)

    logger.info("\n[SEARCH] Pré-requis:")
    logger.info("  1. Dofus doit être lancé et connecté")
    logger.info("  2. Personnage sur une map vide (village de départ)")
    logger.info("  3. Ne pas toucher pendant 5-10 minutes\n")

    input("Appuyez sur ENTRÉE quand prêt...")

    calibrator = create_calibrator()
    result = calibrator.run_full_calibration()

    if result.success:
        logger.info("\n[OK] Calibration terminée avec succès!")
        logger.info(f"  Éléments UI: {len(result.ui_elements)}")
        logger.info(f"  Raccourcis: {len(result.shortcuts)}")
        logger.info(f"  Durée: {result.duration_seconds:.1f}s")
        logger.info(f"\n  Fichier: config/dofus_knowledge.json")
    else:
        logger.error("\n[FAIL] Calibration échouée")
        sys.exit(1)

def run_observation_mode(duration_minutes: int = 10):
    """Lance le mode observation"""
    logger.info("=" * 70)
    logger.info("MODE OBSERVATION")
    logger.info("=" * 70)

    logger.info("\n[WARNING]  MODE SÉCURISÉ ACTIF:")
    logger.info("  - Aucune action ne sera exécutée")
    logger.info("  - Le bot observe seulement")
    logger.info("  - Logs sauvegardés pour analyse")
    logger.info(f"  - Durée: {duration_minutes} minutes\n")

    observation = create_observation_mode()

    logger.info("[OK] Mode observation initialisé")
    logger.info("\n[NOTE] Le bot va maintenant observer...")
    logger.info("  Le bot va logger toutes les décisions qu'il AURAIT prises\n")

    # TODO: Intégrer avec la boucle principale du bot
    # Pour l'instant, juste un exemple de monitoring

    import time
    start_time = time.time()
    duration_seconds = duration_minutes * 60
    
    logger.info(f"Début observation: {duration_seconds}s")

    try:
        iteration = 0
        while time.time() - start_time < duration_seconds:
            iteration += 1
            elapsed = time.time() - start_time
            remaining = duration_seconds - elapsed
            
            logger.info(f"[{iteration}] Temps écoulé: {elapsed:.0f}s / {duration_seconds}s (reste: {remaining:.0f}s)")
            
            # Exemple: Simule des décisions
            game_state = {
                'hp': 100,
                'position': (150, 200),
                'map': '(5,-18)'
            }

            # Le bot prendrait normalement des décisions ici
            # Mais observation_mode.intercept_action() les bloque

            action = observation.intercept_action(
                action_type='navigation',
                action_details={'target': (200, 250)},
                game_state=game_state,
                reason='Exploration de la map'
            )

            # action sera None car observation est activé

            time.sleep(5)  # Check toutes les 5 secondes
        
        logger.info(f"Boucle terminée après {iteration} iterations")

    except KeyboardInterrupt:
        logger.info("\n[WARNING] Arrêt demandé par l'utilisateur")

    # Sauvegarde et analyse
    observation.save_observations()

    logger.info("\n[STATS] Analyse des observations...")
    observation.print_report()

    logger.info("\n[OK] Session d'observation terminée")
    logger.info(f"  Logs: {observation.log_file}")

def run_map_exploration():
    """Lance l'exploration automatique des maps"""
    logger.info("=" * 70)
    logger.info("EXPLORATION DES MAPS")
    logger.info("=" * 70)

    logger.warning("\n[WARNING] Fonctionnalité en développement")
    logger.warning("  Mode observation recommandé\n")

    map_graph = create_map_graph()

    logger.info(f"[OK] Map Graph chargé")
    logger.info(f"  Maps connues: {len(map_graph.maps)}")
    logger.info(f"  Découvertes: {len(map_graph.discovered_maps)}")

    # TODO: Implémenter exploration réelle

def test_dofusdb():
    """Teste l'intégration DofusDB"""
    logger.info("=" * 70)
    logger.info("TEST DOFUSDB API")
    logger.info("=" * 70)

    dofusdb = create_dofusdb_client()

    logger.info("\n[API] Test de connexion DofusDB...")

    # Test: Récupère un item célèbre
    logger.info("  Recherche: 'Dofus'")
    items = dofusdb.search_items("Dofus", limit=5)

    if items:
        logger.info(f"\n[OK] {len(items)} résultats trouvés:")
        for item in items:
            logger.info(f"    - {item.name} (lvl {item.level})")

        # Stats du client
        stats = dofusdb.get_stats()
        logger.info(f"\n[STATS] Statistiques:")
        logger.info(f"    - Requêtes: {stats['requests']}")
        logger.info(f"    - Cache hits: {stats['cache_hits']}")
        logger.info(f"    - Cache ratio: {stats['cache_ratio']}")

    else:
        logger.warning("[FAIL] Aucun résultat (API indisponible?)")

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description='DOFUS AlphaStar - Bot AlphaStar-like pour Dofus Unity',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Lance la calibration automatique (première fois)'
    )

    parser.add_argument(
        '--observe',
        type=int,
        metavar='MINUTES',
        default=0,
        help='Mode observation (N minutes, défaut: désactivé)'
    )

    parser.add_argument(
        '--explore',
        action='store_true',
        help='Mode exploration des maps'
    )

    parser.add_argument(
        '--test-dofusdb',
        action='store_true',
        help='Test de l\'API DofusDB'
    )

    parser.add_argument(
        '--unsafe',
        action='store_true',
        help='Désactive le mode observation (DANGER!)'
    )

    args = parser.parse_args()

    # Banner
    print_banner()

    # Avertissement si mode unsafe
    if args.unsafe:
        logger.critical("\n" + "=" * 70)
        logger.critical("[WARNING][WARNING][WARNING]  MODE UNSAFE DEMANDÉ  [WARNING][WARNING][WARNING]")
        logger.critical("[WARNING]  LE BOT VA AGIR RÉELLEMENT  [WARNING]")
        logger.critical("[WARNING]  COMPTE JETABLE REQUIS  [WARNING]")
        logger.critical("=" * 70)

        confirmation = input("\nTaper 'JE COMPRENDS LES RISQUES' pour continuer: ")
        if confirmation != "JE COMPRENDS LES RISQUES":
            logger.info("Annulation")
            sys.exit(0)

    # Exécution
    try:
        if args.calibrate:
            run_calibration()

        elif args.observe > 0:
            run_observation_mode(args.observe)

        elif args.explore:
            run_map_exploration()

        elif args.test_dofusdb:
            test_dofusdb()

        else:
            # Par défaut: Mode observation 10 minutes
            logger.info("\n[INFO] Aucun mode spécifié, lancement observation 10min")
            logger.info("   Utiliser --help pour voir toutes les options\n")

            run_observation_mode(10)

    except KeyboardInterrupt:
        logger.info("\n\n[WARNING] Arrêt demandé par l'utilisateur")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\n[ERROR] Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()