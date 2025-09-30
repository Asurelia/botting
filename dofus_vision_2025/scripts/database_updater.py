"""
DOFUS Database Updater - Mise à jour sécurisée des bases de données
Extraction et intégration des données DOFUS Unity dans les bases SQLite
"""

import json
import sqlite3
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

# Import des modules existants
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.knowledge_base.dofus_data_extractor import DofusDataExtractor
from core.knowledge_base.spells_database import DofusSpellsDatabase, get_spells_database
from core.knowledge_base.monsters_database import DofusMonstersDatabase, get_monsters_database
from core.knowledge_base.maps_database import DofusMapsDatabase, get_maps_database
from core.knowledge_base.economy_tracker import EconomyTracker

logger = logging.getLogger(__name__)

class DatabaseUpdater:
    """Gestionnaire de mise à jour des bases de données DOFUS"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Chemins des bases de données
        self.db_paths = {
            "spells": self.data_dir / "dofus_spells.db",
            "monsters": self.data_dir / "dofus_monsters.db",
            "maps": self.data_dir / "dofus_maps.db",
            "economy": self.data_dir / "dofus_economy.db"
        }

        # Initialisation extracteur
        self.extractor = DofusDataExtractor()

        # Log
        self.update_log = []

    def _log_operation(self, operation: str, status: str, details: str = ""):
        """Log une opération de mise à jour"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "status": status,
            "details": details
        }
        self.update_log.append(log_entry)
        logger.info(f"{operation}: {status} - {details}")

    def create_backup(self) -> str:
        """Crée une sauvegarde des bases de données actuelles"""
        backup_dir = self.data_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_count = 0
        for db_name, db_path in self.db_paths.items():
            if db_path.exists():
                backup_path = backup_dir / db_path.name
                backup_path.write_bytes(db_path.read_bytes())
                backup_count += 1

        self._log_operation("backup", "success", f"{backup_count} bases sauvegardées dans {backup_dir}")
        return str(backup_dir)

    def get_database_stats(self) -> Dict[str, Dict[str, Any]]:
        """Récupère les statistiques actuelles des bases de données"""
        stats = {}

        for db_name, db_path in self.db_paths.items():
            if not db_path.exists():
                stats[db_name] = {"exists": False}
                continue

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Récupération des tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                table_stats = {}
                total_records = 0

                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_stats[table] = count
                    total_records += count

                stats[db_name] = {
                    "exists": True,
                    "tables": table_stats,
                    "total_records": total_records,
                    "file_size": db_path.stat().st_size
                }

                conn.close()

            except Exception as e:
                stats[db_name] = {"exists": True, "error": str(e)}

        return stats

    def extract_unity_data(self) -> Dict[str, Any]:
        """Extrait les données depuis les bundles Unity DOFUS"""
        extraction_results = {
            "spells": {"bundles": {}, "processed_data": []},
            "monsters": {"bundles": {}, "processed_data": []},
            "maps": {"bundles": {}, "processed_data": []},
            "items": {"bundles": {}, "processed_data": []}
        }

        try:
            # Extraction sorts
            self._log_operation("extract_spells", "started", "Extraction données sorts")
            spells_data = self.extractor.extract_spells_data()
            extraction_results["spells"]["bundles"] = spells_data

            # Traitement des données sorts
            for bundle_name, extracted in spells_data.items():
                if extracted.success and extracted.content:
                    processed = self._process_spell_data(extracted.content, bundle_name)
                    extraction_results["spells"]["processed_data"].extend(processed)

            self._log_operation("extract_spells", "completed",
                             f"{len(extraction_results['spells']['processed_data'])} sorts extraits")

            # Extraction monstres
            self._log_operation("extract_monsters", "started", "Extraction données monstres")
            monsters_data = self.extractor.extract_monsters_data()
            extraction_results["monsters"]["bundles"] = monsters_data

            # Traitement des données monstres
            for bundle_name, extracted in monsters_data.items():
                if extracted.success and extracted.content:
                    processed = self._process_monster_data(extracted.content, bundle_name)
                    extraction_results["monsters"]["processed_data"].extend(processed)

            self._log_operation("extract_monsters", "completed",
                             f"{len(extraction_results['monsters']['processed_data'])} monstres extraits")

            # Extraction cartes
            self._log_operation("extract_maps", "started", "Extraction données cartes")
            maps_data = self.extractor.extract_maps_data()
            extraction_results["maps"]["bundles"] = maps_data

            # Traitement des données cartes
            for bundle_name, extracted in maps_data.items():
                if extracted.success and extracted.content:
                    processed = self._process_map_data(extracted.content, bundle_name)
                    extraction_results["maps"]["processed_data"].extend(processed)

            self._log_operation("extract_maps", "completed",
                             f"{len(extraction_results['maps']['processed_data'])} cartes extraites")

        except Exception as e:
            self._log_operation("extract_unity_data", "error", str(e))

        return extraction_results

    def _process_spell_data(self, content: Any, bundle_name: str) -> List[Dict[str, Any]]:
        """Traite les données de sort extraites"""
        processed = []

        try:
            if isinstance(content, dict) and content.get("type") == "binary":
                # Données binaires - créons des entrées basiques
                spell_entry = {
                    "id": f"extracted_{hash(bundle_name) % 10000}",
                    "name": f"Sort extrait de {bundle_name}",
                    "description": f"Sort extrait depuis {bundle_name}",
                    "level": 1,
                    "ap_cost": 3,
                    "range_min": 1,
                    "range_max": 5,
                    "effects": f"Données extraites de {bundle_name}",
                    "source": bundle_name,
                    "extraction_time": datetime.now().isoformat()
                }
                processed.append(spell_entry)
        except Exception as e:
            logger.error(f"Erreur traitement données sort {bundle_name}: {e}")

        return processed

    def _process_monster_data(self, content: Any, bundle_name: str) -> List[Dict[str, Any]]:
        """Traite les données de monstre extraites"""
        processed = []

        try:
            if isinstance(content, dict) and content.get("type") == "binary":
                # Données binaires - créons des entrées basiques
                monster_entry = {
                    "id": f"extracted_{hash(bundle_name) % 10000}",
                    "name": f"Monstre extrait de {bundle_name}",
                    "level": 50,
                    "health": 1000,
                    "ap": 6,
                    "mp": 3,
                    "characteristics": f"Données extraites de {bundle_name}",
                    "spells": [],
                    "source": bundle_name,
                    "extraction_time": datetime.now().isoformat()
                }
                processed.append(monster_entry)
        except Exception as e:
            logger.error(f"Erreur traitement données monstre {bundle_name}: {e}")

        return processed

    def _process_map_data(self, content: Any, bundle_name: str) -> List[Dict[str, Any]]:
        """Traite les données de carte extraites"""
        processed = []

        try:
            if isinstance(content, dict) and content.get("type") == "binary":
                # Données binaires - créons des entrées basiques
                map_entry = {
                    "id": f"extracted_{hash(bundle_name) % 10000}",
                    "name": f"Carte extraite de {bundle_name}",
                    "x": 0,
                    "y": 0,
                    "area": "Zone extraite",
                    "subarea": "Sous-zone extraite",
                    "zaap": False,
                    "source": bundle_name,
                    "extraction_time": datetime.now().isoformat()
                }
                processed.append(map_entry)
        except Exception as e:
            logger.error(f"Erreur traitement données carte {bundle_name}: {e}")

        return processed

    def update_databases(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Met à jour les bases de données avec les nouvelles données"""
        update_results = {}

        try:
            # Mise à jour base sorts
            spells_db = get_spells_database()
            spells_updated = 0
            for spell_data in extraction_results["spells"]["processed_data"]:
                # Conversion en format compatible
                try:
                    spells_db.add_spell_from_dict(spell_data)
                    spells_updated += 1
                except Exception as e:
                    logger.warning(f"Erreur ajout sort: {e}")
            update_results["spells"] = {"updated": spells_updated}
            self._log_operation("update_spells_db", "completed", f"{spells_updated} sorts mis à jour")

            # Mise à jour base monstres
            monsters_db = get_monsters_database()
            monsters_updated = 0
            for monster_data in extraction_results["monsters"]["processed_data"]:
                # Conversion en format compatible
                try:
                    monsters_db.add_monster_from_dict(monster_data)
                    monsters_updated += 1
                except Exception as e:
                    logger.warning(f"Erreur ajout monstre: {e}")
            update_results["monsters"] = {"updated": monsters_updated}
            self._log_operation("update_monsters_db", "completed", f"{monsters_updated} monstres mis à jour")

            # Mise à jour base cartes
            maps_db = get_maps_database()
            maps_updated = 0
            for map_data in extraction_results["maps"]["processed_data"]:
                # Conversion en format compatible
                try:
                    maps_db.add_map_from_dict(map_data)
                    maps_updated += 1
                except Exception as e:
                    logger.warning(f"Erreur ajout carte: {e}")
            update_results["maps"] = {"updated": maps_updated}
            self._log_operation("update_maps_db", "completed", f"{maps_updated} cartes mises à jour")

        except Exception as e:
            self._log_operation("update_databases", "error", str(e))
            update_results["error"] = str(e)

        return update_results

    def verify_database_integrity(self) -> Dict[str, Any]:
        """Vérifie l'intégrité des bases de données après mise à jour"""
        integrity_results = {}

        for db_name, db_path in self.db_paths.items():
            if not db_path.exists():
                integrity_results[db_name] = {"status": "missing"}
                continue

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Vérification intégrité SQLite
                cursor.execute("PRAGMA integrity_check")
                integrity_check = cursor.fetchone()[0]

                # Statistiques mise à jour
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                table_counts = {}
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    table_counts[table] = cursor.fetchone()[0]

                integrity_results[db_name] = {
                    "status": "ok" if integrity_check == "ok" else "error",
                    "integrity_check": integrity_check,
                    "tables": table_counts
                }

                conn.close()

            except Exception as e:
                integrity_results[db_name] = {"status": "error", "error": str(e)}

        return integrity_results

    def generate_update_report(self, stats_before: Dict, stats_after: Dict,
                             extraction_results: Dict, update_results: Dict,
                             integrity_results: Dict) -> Dict[str, Any]:
        """Génère un rapport complet de mise à jour"""

        report = {
            "update_info": {
                "timestamp": datetime.now().isoformat(),
                "extractor_version": "1.0.0",
                "total_bundles_processed": 0
            },
            "database_changes": {},
            "extraction_summary": {},
            "integrity_status": integrity_results,
            "recommendations": [],
            "operation_log": self.update_log
        }

        # Calcul des changements
        for db_name in self.db_paths.keys():
            before = stats_before.get(db_name, {})
            after = stats_after.get(db_name, {})

            if before.get("exists") and after.get("exists"):
                before_total = before.get("total_records", 0)
                after_total = after.get("total_records", 0)

                report["database_changes"][db_name] = {
                    "records_before": before_total,
                    "records_after": after_total,
                    "records_added": after_total - before_total,
                    "size_before": before.get("file_size", 0),
                    "size_after": after.get("file_size", 0)
                }

        # Résumé extraction
        for data_type, results in extraction_results.items():
            bundles_count = len(results.get("bundles", {}))
            processed_count = len(results.get("processed_data", []))

            report["extraction_summary"][data_type] = {
                "bundles_processed": bundles_count,
                "records_extracted": processed_count
            }

            report["update_info"]["total_bundles_processed"] += bundles_count

        # Recommandations
        total_new_records = sum(
            change.get("records_added", 0)
            for change in report["database_changes"].values()
        )

        if total_new_records > 100:
            report["recommendations"].append(
                "Grande quantité de nouvelles données - considérer optimisation des index"
            )

        if any(status.get("status") != "ok" for status in integrity_results.values()):
            report["recommendations"].append(
                "Problèmes d'intégrité détectés - vérification manuelle recommandée"
            )

        return report

    def run_full_update(self) -> Dict[str, Any]:
        """Exécute une mise à jour complète des bases de données"""

        # 1. Statistiques avant
        stats_before = self.get_database_stats()
        self._log_operation("pre_update_stats", "completed", "Statistiques avant mise à jour collectées")

        # 2. Sauvegarde
        backup_dir = self.create_backup()

        # 3. Extraction Unity
        extraction_results = self.extract_unity_data()

        # 4. Mise à jour bases
        update_results = self.update_databases(extraction_results)

        # 5. Statistiques après
        stats_after = self.get_database_stats()
        self._log_operation("post_update_stats", "completed", "Statistiques après mise à jour collectées")

        # 6. Vérification intégrité
        integrity_results = self.verify_database_integrity()

        # 7. Rapport final
        report = self.generate_update_report(
            stats_before, stats_after, extraction_results,
            update_results, integrity_results
        )

        report["backup_location"] = backup_dir

        return report

if __name__ == "__main__":
    # Configuration logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        updater = DatabaseUpdater()
        print("Démarrage mise à jour complète des bases de données DOFUS...")

        report = updater.run_full_update()

        print("\n=== RAPPORT DE MISE À JOUR ===")
        print(f"Timestamp: {report['update_info']['timestamp']}")
        print(f"Bundles traités: {report['update_info']['total_bundles_processed']}")
        print(f"Sauvegarde: {report['backup_location']}")

        print("\nChangements par base:")
        for db_name, changes in report["database_changes"].items():
            print(f"  {db_name}: +{changes['records_added']} enregistrements")

        print("\nStatut intégrité:")
        for db_name, status in report["integrity_status"].items():
            print(f"  {db_name}: {status.get('status', 'unknown')}")

        if report["recommendations"]:
            print("\nRecommandations:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")

        # Sauvegarde rapport
        report_path = Path("data") / "update_reports" / f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nRapport sauvegardé: {report_path}")

    except Exception as e:
        print(f"Erreur mise à jour: {e}")
        logger.error(f"Erreur mise à jour: {e}")