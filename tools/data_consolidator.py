"""
Data Consolidator - Consolidation et enrichissement des données
Combine les données locales avec les fansites pour une base complète

Fonctionnalités:
- Charge toutes les données locales existantes
- Complète avec données des fansites (DofusDB, etc.)
- Valide et normalise les données
- Génère une base de données unifiée
"""

import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from bs4 import BeautifulSoup
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FansiteAPI:
    """API pour récupérer données depuis fansites"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cache = {}
    
    def get_monster_data(self, monster_name: str) -> Optional[Dict]:
        """Récupère données d'un monstre depuis DofusDB"""
        if monster_name in self.cache:
            return self.cache[monster_name]
        
        try:
            # DofusDB API (exemple - à adapter selon API réelle)
            url = f"https://api.dofusdb.fr/monsters"
            params = {"name": monster_name, "lang": "fr"}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    monster_data = self._normalize_monster_data(data[0] if isinstance(data, list) else data)
                    self.cache[monster_name] = monster_data
                    return monster_data
        
        except Exception as e:
            logger.debug(f"Erreur récupération {monster_name}: {e}")
        
        return None
    
    def get_spell_data(self, spell_name: str) -> Optional[Dict]:
        """Récupère données d'un sort"""
        try:
            url = f"https://api.dofusdb.fr/spells"
            params = {"name": spell_name, "lang": "fr"}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    return self._normalize_spell_data(data[0] if isinstance(data, list) else data)
        
        except Exception as e:
            logger.debug(f"Erreur récupération sort {spell_name}: {e}")
        
        return None
    
    def get_item_data(self, item_name: str) -> Optional[Dict]:
        """Récupère données d'un item"""
        try:
            url = f"https://api.dofusdb.fr/items"
            params = {"name": item_name, "lang": "fr"}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    return self._normalize_item_data(data[0] if isinstance(data, list) else data)
        
        except Exception as e:
            logger.debug(f"Erreur récupération item {item_name}: {e}")
        
        return None
    
    def _normalize_monster_data(self, raw_data: Dict) -> Dict:
        """Normalise les données de monstre au format du bot"""
        return {
            "id": raw_data.get("id", raw_data.get("name", "").lower().replace(" ", "_")),
            "name": raw_data.get("name", "Unknown"),
            "level": raw_data.get("level", 1),
            "health": raw_data.get("lifePoints", raw_data.get("hp", 100)),
            "ap": raw_data.get("actionPoints", 6),
            "mp": raw_data.get("movementPoints", 3),
            "resistances": raw_data.get("resistances", {
                "neutral": 0, "earth": 0, "fire": 0, "water": 0, "air": 0
            }),
            "damages": {
                "min": raw_data.get("minDamage", 10),
                "max": raw_data.get("maxDamage", 20),
                "element": raw_data.get("element", "neutral")
            },
            "locations": raw_data.get("locations", []),
            "drops": raw_data.get("drops", []),
            "xp_reward": raw_data.get("xp", 50),
            "kamas_reward": raw_data.get("kamas", 10),
            "difficulty": raw_data.get("difficulty", "medium"),
            "source": "fansite",
            "fetched_at": datetime.now().isoformat()
        }
    
    def _normalize_spell_data(self, raw_data: Dict) -> Dict:
        """Normalise les données de sort"""
        return {
            "id": raw_data.get("id", raw_data.get("name", "").lower().replace(" ", "_")),
            "name": raw_data.get("name", "Unknown"),
            "level": raw_data.get("level", 1),
            "ap_cost": raw_data.get("apCost", 3),
            "range": raw_data.get("range", 5),
            "damage": raw_data.get("damage", {}),
            "effects": raw_data.get("effects", []),
            "element": raw_data.get("element", "neutral"),
            "source": "fansite",
            "fetched_at": datetime.now().isoformat()
        }
    
    def _normalize_item_data(self, raw_data: Dict) -> Dict:
        """Normalise les données d'item"""
        return {
            "id": raw_data.get("id", raw_data.get("name", "").lower().replace(" ", "_")),
            "name": raw_data.get("name", "Unknown"),
            "level": raw_data.get("level", 1),
            "type": raw_data.get("type", "misc"),
            "stats": raw_data.get("stats", {}),
            "value": raw_data.get("averagePrice", 0),
            "source": "fansite",
            "fetched_at": datetime.now().isoformat()
        }


class DataConsolidator:
    """Consolidateur de données"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.fansite_api = FansiteAPI()
        
        self.consolidated_data = {
            "monsters": {},
            "spells": {},
            "items": {},
            "maps": {},
            "resources": {},
            "npcs": {},
            "quests": {}
        }
    
    def consolidate_all(self):
        """Consolide toutes les données"""
        logger.info("🔄 Consolidation des données...")
        
        # 1. Chargement données locales
        self._load_local_data()
        
        # 2. Enrichissement avec fansites
        self._enrich_from_fansites()
        
        # 3. Validation
        self._validate_data()
        
        # 4. Sauvegarde
        self._save_consolidated_data()
        
        logger.info("✅ Consolidation terminée !")
        return self.consolidated_data
    
    def _load_local_data(self):
        """Charge toutes les données locales"""
        logger.info("📂 Chargement données locales...")
        
        # Monstres
        monsters_file = self.data_dir / "monsters" / "monster_database.json"
        if monsters_file.exists():
            with open(monsters_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._extract_monsters_from_db(data)
                logger.info(f"   ✅ Monstres: {len(self.consolidated_data['monsters'])} entrées")
        
        # Maps
        maps_file = self.data_dir / "maps" / "map_coordinates.json"
        if maps_file.exists():
            with open(maps_file, 'r', encoding='utf-8') as f:
                self.consolidated_data['maps'] = json.load(f)
                logger.info(f"   ✅ Maps: {len(self.consolidated_data['maps'])} entrées")
        
        # Ressources
        resources_file = self.data_dir / "resources" / "agricultural_resources.json"
        if resources_file.exists():
            with open(resources_file, 'r', encoding='utf-8') as f:
                self.consolidated_data['resources'] = json.load(f)
                logger.info(f"   ✅ Ressources: {len(self.consolidated_data['resources'])} entrées")
    
    def _extract_monsters_from_db(self, data: Dict):
        """Extrait les monstres de la base de données"""
        if "categories" in data:
            for category, category_data in data["categories"].items():
                if "monsters" in category_data:
                    for monster_id, monster_data in category_data["monsters"].items():
                        monster_data["source"] = "local"
                        self.consolidated_data["monsters"][monster_id] = monster_data
    
    def _enrich_from_fansites(self):
        """Enrichit avec données des fansites"""
        logger.info("🌐 Enrichissement depuis fansites...")
        
        # Liste de monstres communs à compléter
        common_monsters = [
            "Tofu Maléfique", "Craqueleur Légendaire", "Blop Royal",
            "Wa Wabbit", "Chafer Rōnin", "Abraknyde Ancestral"
        ]
        
        enriched_count = 0
        for monster_name in common_monsters:
            monster_id = monster_name.lower().replace(" ", "_")
            
            # Si pas déjà dans les données locales
            if monster_id not in self.consolidated_data["monsters"]:
                logger.info(f"   🔍 Recherche {monster_name}...")
                monster_data = self.fansite_api.get_monster_data(monster_name)
                
                if monster_data:
                    self.consolidated_data["monsters"][monster_id] = monster_data
                    enriched_count += 1
                    logger.info(f"      ✅ Ajouté")
                    time.sleep(0.5)  # Rate limiting
                else:
                    logger.debug(f"      ⚠️ Non trouvé")
        
        logger.info(f"   ✅ {enriched_count} monstres ajoutés depuis fansites")
    
    def _validate_data(self):
        """Valide la cohérence des données"""
        logger.info("🔍 Validation des données...")
        
        # Validation monstres
        valid_monsters = 0
        for monster_id, monster in self.consolidated_data["monsters"].items():
            if self._is_valid_monster(monster):
                valid_monsters += 1
        
        logger.info(f"   ✅ Monstres valides: {valid_monsters}/{len(self.consolidated_data['monsters'])}")
    
    def _is_valid_monster(self, monster: Dict) -> bool:
        """Vérifie qu'un monstre a les champs requis"""
        required_fields = ["name", "level", "health"]
        return all(field in monster for field in required_fields)
    
    def _save_consolidated_data(self):
        """Sauvegarde les données consolidées"""
        logger.info("💾 Sauvegarde données consolidées...")
        
        output_dir = self.data_dir / "consolidated"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for category, data in self.consolidated_data.items():
            if data:
                output_file = output_dir / f"{category}_consolidated.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"   ✅ {category}: {len(data)} entrées → {output_file.name}")
        
        # Métadonnées
        metadata = {
            "generated_at": timestamp,
            "total_entries": sum(len(data) for data in self.consolidated_data.values()),
            "sources": ["local", "fansite"],
            "categories": {
                category: len(data) 
                for category, data in self.consolidated_data.items()
            }
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def get_monster(self, monster_name: str) -> Optional[Dict]:
        """Récupère un monstre (local ou fansite)"""
        monster_id = monster_name.lower().replace(" ", "_")
        
        # 1. Recherche locale
        if monster_id in self.consolidated_data["monsters"]:
            return self.consolidated_data["monsters"][monster_id]
        
        # 2. Recherche fansite
        logger.info(f"🌐 Récupération {monster_name} depuis fansite...")
        monster_data = self.fansite_api.get_monster_data(monster_name)
        
        if monster_data:
            # Ajout au cache
            self.consolidated_data["monsters"][monster_id] = monster_data
            return monster_data
        
        return None


def main():
    """Point d'entrée principal"""
    print("=" * 70)
    print("🔄 CONSOLIDATION DES DONNÉES DOFUS")
    print("=" * 70)
    print()
    
    consolidator = DataConsolidator()
    
    # Consolidation
    data = consolidator.consolidate_all()
    
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    
    for category, entries in data.items():
        if entries:
            print(f"  • {category.capitalize():15} : {len(entries):4} entrées")
    
    total = sum(len(entries) for entries in data.values())
    print("-" * 70)
    print(f"  • {'TOTAL':15} : {total:4} entrées")
    
    print()
    print("=" * 70)
    print("✅ CONSOLIDATION TERMINÉE")
    print("=" * 70)
    print()
    print("📁 Fichiers générés dans: data/consolidated/")
    print()
    print("💡 Utilisation dans le bot:")
    print("   from tools.data_consolidator import DataConsolidator")
    print("   consolidator = DataConsolidator()")
    print("   monster = consolidator.get_monster('Bouftou')")
    print()


if __name__ == "__main__":
    main()
