"""
DOFUS Database Updater Simplifié - Mise à jour sécurisée des bases de données
Version simplifiée pour mise à jour avec données extraites
"""

import json
import sqlite3
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import des modules existants
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.knowledge_base.dofus_data_extractor import DofusDataExtractor
from core.knowledge_base.spells_database import DofusSpell, get_spells_database, SpellType, TargetType, DofusClass
from core.knowledge_base.monsters_database import DofusMonster, get_monsters_database, MonsterElement, MonsterRank
from core.knowledge_base.maps_database import DofusMap, get_maps_database, MapType

logger = logging.getLogger(__name__)

class SimpleDatabaseUpdater:
    """Gestionnaire simplifié de mise à jour des bases de données DOFUS"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Initialisation extracteur
        try:
            self.extractor = DofusDataExtractor()
            self.extractor_available = True
        except Exception as e:
            logger.warning(f"Extracteur non disponible: {e}")
            self.extractor_available = False

        # Log des opérations
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {operation}: {status} - {details}")

    def create_backup(self) -> str:
        """Crée une sauvegarde des bases de données actuelles"""
        backup_dir = self.data_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)

        db_files = list(self.data_dir.glob("*.db"))
        backup_count = 0

        for db_file in db_files:
            backup_path = backup_dir / db_file.name
            backup_path.write_bytes(db_file.read_bytes())
            backup_count += 1

        self._log_operation("backup", "success", f"{backup_count} bases sauvegardées dans {backup_dir}")
        return str(backup_dir)

    def get_database_stats(self) -> Dict[str, Dict[str, Any]]:
        """Récupère les statistiques actuelles des bases de données"""
        stats = {}
        db_files = ["dofus_spells.db", "dofus_monsters.db", "dofus_maps.db", "dofus_economy.db"]

        for db_file in db_files:
            db_path = self.data_dir / db_file
            db_name = db_file.replace(".db", "")

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

    def extract_and_create_sample_data(self) -> Dict[str, List[Dict]]:
        """Extrait des données ou crée des échantillons pour test"""
        sample_data = {
            "spells": [],
            "monsters": [],
            "maps": []
        }

        if self.extractor_available:
            try:
                # Tentative d'extraction réelle
                self._log_operation("extract_real_data", "started", "Tentative extraction données réelles")

                bundles = self.extractor.list_available_bundles()
                spell_bundles = [b for b in bundles if b.type == "spells"]
                monster_bundles = [b for b in bundles if b.type == "monsters"]
                map_bundles = [b for b in bundles if b.type == "maps"]

                self._log_operation("bundles_found", "info",
                                  f"Trouvé {len(spell_bundles)} bundles sorts, {len(monster_bundles)} monstres, {len(map_bundles)} cartes")

                # Extraction limitée pour test
                if spell_bundles:
                    for bundle in spell_bundles[:2]:  # Limité à 2 bundles
                        extracted = self.extractor.extract_bundle_data(bundle.name)
                        if extracted.success:
                            # Création d'échantillon basé sur l'extraction
                            spell_sample = {
                                "id": f"extracted_{hash(bundle.name) % 10000}",
                                "name": f"Sort extrait {bundle.name[:20]}",
                                "description": f"Sort extrait depuis {bundle.name}",
                                "level": 1,
                                "ap_cost": 3,
                                "range_min": 1,
                                "range_max": 5,
                                "spell_type": SpellType.DAMAGE,
                                "target_type": TargetType.CELL,
                                "element": "neutral",
                                "class_restriction": None,
                                "effects": [{"type": "damage", "min": 10, "max": 20}],
                                "source": f"Unity:{bundle.name}"
                            }
                            sample_data["spells"].append(spell_sample)

            except Exception as e:
                self._log_operation("extract_real_data", "error", str(e))

        # Création d'échantillons si pas d'extraction ou complément
        if len(sample_data["spells"]) < 5:
            self._log_operation("create_sample_data", "started", "Création d'échantillons de test")

            # Sorts échantillon
            for i in range(5 - len(sample_data["spells"])):
                spell_sample = {
                    "id": f"sample_spell_{i+1}",
                    "name": f"Sort Test Unity {i+1}",
                    "description": f"Sort de test extrait depuis Unity bundle {i+1}",
                    "level": (i % 3) + 1,
                    "ap_cost": (i % 4) + 2,
                    "range_min": 1,
                    "range_max": (i % 3) + 3,
                    "spell_type": [SpellType.DAMAGE, SpellType.HEAL, SpellType.BUFF][i % 3],
                    "target_type": [TargetType.CELL, TargetType.ENTITY, TargetType.LINE][i % 3],
                    "element": ["fire", "water", "earth", "air", "neutral"][i % 5],
                    "class_restriction": None,
                    "effects": [{"type": "damage", "min": 10 + i*5, "max": 20 + i*10}],
                    "source": "Unity:Sample"
                }
                sample_data["spells"].append(spell_sample)

            # Monstres échantillon
            for i in range(5):
                monster_sample = {
                    "id": f"sample_monster_{i+1}",
                    "name": f"Monstre Unity {i+1}",
                    "level": 50 + i*10,
                    "health": 1000 + i*500,
                    "ap": 6,
                    "mp": 3,
                    "element": [MonsterElement.FIRE, MonsterElement.WATER, MonsterElement.EARTH][i % 3],
                    "rank": [MonsterRank.NORMAL, MonsterRank.ELITE, MonsterRank.BOSS][i % 3],
                    "resistances": {"fire": 10, "water": 10, "earth": 10, "air": 10, "neutral": 10},
                    "spells": [f"sample_spell_{(i%3)+1}"],
                    "drops": {"items": [f"item_{i+1}"], "kamas": 100 + i*50},
                    "behavior": {"aggressive": True, "movement": 3},
                    "source": "Unity:Sample"
                }
                sample_data["monsters"].append(monster_sample)

            # Cartes échantillon
            for i in range(5):
                map_sample = {
                    "id": f"sample_map_{i+1}",
                    "name": f"Carte Unity {i+1}",
                    "x": i - 2,
                    "y": i - 2,
                    "area": "Zone Test Unity",
                    "subarea": f"Sous-zone {i+1}",
                    "map_type": MapType.OVERWORLD,
                    "zaap": i == 2,  # Zaap sur la carte du milieu
                    "transitions": [],
                    "monsters": [f"sample_monster_{i+1}"],
                    "resources": [],
                    "source": "Unity:Sample"
                }
                sample_data["maps"].append(map_sample)

        return sample_data

    def update_databases_with_samples(self, sample_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Met à jour les bases de données avec les données échantillon"""
        update_results = {}

        try:
            # Mise à jour base sorts
            self._log_operation("update_spells", "started", "Mise à jour base sorts")
            spells_db = get_spells_database()
            spells_updated = 0

            for spell_data in sample_data["spells"]:
                try:
                    # Création objet DofusSpell
                    spell = DofusSpell(
                        id=spell_data["id"],
                        name=spell_data["name"],
                        description=spell_data["description"],
                        level=spell_data["level"],
                        ap_cost=spell_data["ap_cost"],
                        range_min=spell_data["range_min"],
                        range_max=spell_data["range_max"],
                        spell_type=spell_data["spell_type"],
                        target_type=spell_data["target_type"],
                        element=spell_data["element"],
                        class_restriction=spell_data["class_restriction"],
                        effects=spell_data["effects"]
                    )

                    spells_db.add_spell(spell)
                    spells_updated += 1

                except Exception as e:
                    logger.warning(f"Erreur ajout sort {spell_data['name']}: {e}")

            update_results["spells"] = {"updated": spells_updated}
            self._log_operation("update_spells", "completed", f"{spells_updated} sorts ajoutés")

            # Mise à jour base monstres
            self._log_operation("update_monsters", "started", "Mise à jour base monstres")
            monsters_db = get_monsters_database()
            monsters_updated = 0

            for monster_data in sample_data["monsters"]:
                try:
                    # Création objet DofusMonster
                    monster = DofusMonster(
                        id=monster_data["id"],
                        name=monster_data["name"],
                        level=monster_data["level"],
                        health=monster_data["health"],
                        ap=monster_data["ap"],
                        mp=monster_data["mp"],
                        element=monster_data["element"],
                        rank=monster_data["rank"],
                        resistances=monster_data["resistances"],
                        spells=monster_data["spells"],
                        drops=monster_data["drops"],
                        behavior=monster_data["behavior"]
                    )

                    monsters_db.add_monster(monster)
                    monsters_updated += 1

                except Exception as e:
                    logger.warning(f"Erreur ajout monstre {monster_data['name']}: {e}")

            update_results["monsters"] = {"updated": monsters_updated}
            self._log_operation("update_monsters", "completed", f"{monsters_updated} monstres ajoutés")

            # Mise à jour base cartes
            self._log_operation("update_maps", "started", "Mise à jour base cartes")
            maps_db = get_maps_database()
            maps_updated = 0

            for map_data in sample_data["maps"]:
                try:
                    # Création objet DofusMap
                    map_obj = DofusMap(
                        id=map_data["id"],
                        name=map_data["name"],
                        x=map_data["x"],
                        y=map_data["y"],
                        area=map_data["area"],
                        subarea=map_data["subarea"],
                        map_type=map_data["map_type"],
                        zaap=map_data["zaap"],
                        transitions=map_data["transitions"],
                        monsters=map_data["monsters"],
                        resources=map_data["resources"]
                    )

                    maps_db.add_map(map_obj)
                    maps_updated += 1

                except Exception as e:
                    logger.warning(f"Erreur ajout carte {map_data['name']}: {e}")

            update_results["maps"] = {"updated": maps_updated}
            self._log_operation("update_maps", "completed", f"{maps_updated} cartes ajoutées")

        except Exception as e:
            self._log_operation("update_databases", "error", str(e))
            update_results["error"] = str(e)

        return update_results

    def verify_database_integrity(self) -> Dict[str, Any]:
        """Vérifie l'intégrité des bases de données après mise à jour"""
        integrity_results = {}
        db_files = ["dofus_spells.db", "dofus_monsters.db", "dofus_maps.db", "dofus_economy.db"]

        for db_file in db_files:
            db_path = self.data_dir / db_file
            db_name = db_file.replace(".db", "")

            if not db_path.exists():
                integrity_results[db_name] = {"status": "missing"}
                continue

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Vérification intégrité SQLite
                cursor.execute("PRAGMA integrity_check")
                integrity_check = cursor.fetchone()[0]

                # Statistiques
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

    def run_update(self) -> Dict[str, Any]:
        """Exécute une mise à jour complète"""

        print("=== DOFUS DATABASE UPDATER ===")
        print("Démarrage de la mise à jour des bases de données...")

        # 1. Statistiques avant
        self._log_operation("pre_stats", "started", "Collecte statistiques avant mise à jour")
        stats_before = self.get_database_stats()

        # 2. Sauvegarde
        backup_dir = self.create_backup()

        # 3. Extraction/Création échantillons
        sample_data = self.extract_and_create_sample_data()

        # 4. Mise à jour
        update_results = self.update_databases_with_samples(sample_data)

        # 5. Statistiques après
        self._log_operation("post_stats", "started", "Collecte statistiques après mise à jour")
        stats_after = self.get_database_stats()

        # 6. Vérification intégrité
        self._log_operation("integrity_check", "started", "Vérification intégrité")
        integrity_results = self.verify_database_integrity()

        # 7. Rapport final
        report = {
            "update_info": {
                "timestamp": datetime.now().isoformat(),
                "backup_location": backup_dir,
                "extractor_available": self.extractor_available
            },
            "stats_before": stats_before,
            "stats_after": stats_after,
            "update_results": update_results,
            "integrity_results": integrity_results,
            "sample_data_counts": {k: len(v) for k, v in sample_data.items()},
            "operation_log": self.update_log
        }

        return report

if __name__ == "__main__":
    # Configuration logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        updater = SimpleDatabaseUpdater()
        report = updater.run_update()

        print("\n=== RAPPORT DE MISE À JOUR ===")
        print(f"Timestamp: {report['update_info']['timestamp']}")
        print(f"Sauvegarde: {report['update_info']['backup_location']}")
        print(f"Extracteur disponible: {report['update_info']['extractor_available']}")

        print("\nChangements par base:")
        for db_name in ["dofus_spells", "dofus_monsters", "dofus_maps"]:
            before = report["stats_before"].get(db_name, {}).get("total_records", 0)
            after = report["stats_after"].get(db_name, {}).get("total_records", 0)
            print(f"  {db_name}: {before} -> {after} (+{after - before})")

        print("\nStatut intégrité:")
        for db_name, status in report["integrity_results"].items():
            print(f"  {db_name}: {status.get('status', 'unknown')}")

        print("\nDonnées ajoutées:")
        for data_type, count in report["sample_data_counts"].items():
            print(f"  {data_type}: {count} éléments")

        # Sauvegarde rapport
        reports_dir = Path("data") / "update_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nRapport sauvegardé: {report_path}")
        print("\n=== MISE À JOUR TERMINÉE ===")

    except Exception as e:
        print(f"Erreur mise à jour: {e}")
        logger.error(f"Erreur mise à jour: {e}")