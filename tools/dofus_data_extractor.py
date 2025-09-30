"""
Dofus Unity Data Extractor
Analyse les fichiers du jeu Dofus Unity pour extraire les données
SANS MODIFIER les fichiers originaux

Fonctionnalités:
- Recherche automatique installation Dofus Unity
- Extraction données JSON/XML/Assets
- Parsing bases de données internes
- Génération fichiers pour le bot
- Support multi-sources (local + fansites)
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DofusInstallation:
    """Installation Dofus détectée"""
    path: Path
    version: str
    install_type: str  # "Steam", "Ankama", "Standalone"
    data_folders: List[Path] = field(default_factory=list)


class DofusPathFinder:
    """Recherche l'installation Dofus Unity"""
    
    COMMON_PATHS = [
        # Steam
        r"C:\Program Files (x86)\Steam\steamapps\common\Dofus Unity",
        r"D:\Steam\steamapps\common\Dofus Unity",
        r"E:\Steam\steamapps\common\Dofus Unity",
        
        # Ankama Launcher
        r"C:\Users\{username}\AppData\Local\Ankama\Dofus",
        r"C:\Program Files\Ankama\Dofus",
        r"C:\Program Files (x86)\Ankama\Dofus",
        
        # Standalone
        r"C:\Dofus",
        r"D:\Dofus",
        r"C:\Games\Dofus",
    ]
    
    def find_installations(self) -> List[DofusInstallation]:
        """Recherche toutes les installations Dofus"""
        installations = []
        
        # Expansion des chemins avec username
        username = os.getenv("USERNAME", "")
        expanded_paths = [
            path.replace("{username}", username) 
            for path in self.COMMON_PATHS
        ]
        
        for path_str in expanded_paths:
            path = Path(path_str)
            if path.exists():
                installation = self._analyze_installation(path)
                if installation:
                    installations.append(installation)
                    logger.info(f"✅ Installation trouvée: {path}")
        
        # Recherche dans tous les disques
        for drive in "CDEFGH":
            drive_path = Path(f"{drive}:\\")
            if drive_path.exists():
                installations.extend(self._search_drive(drive_path))
        
        return installations
    
    def _analyze_installation(self, path: Path) -> Optional[DofusInstallation]:
        """Analyse une installation potentielle"""
        # Vérification fichiers clés
        key_files = ["Dofus.exe", "DofusInvoker.exe", "Dofus Unity.exe"]
        
        for key_file in key_files:
            if (path / key_file).exists():
                # Détection type d'installation
                install_type = self._detect_install_type(path)
                
                # Recherche dossiers de données
                data_folders = self._find_data_folders(path)
                
                # Lecture version
                version = self._read_version(path)
                
                return DofusInstallation(
                    path=path,
                    version=version,
                    install_type=install_type,
                    data_folders=data_folders
                )
        
        return None
    
    def _detect_install_type(self, path: Path) -> str:
        """Détecte le type d'installation"""
        if "Steam" in str(path):
            return "Steam"
        elif "Ankama" in str(path):
            return "Ankama"
        else:
            return "Standalone"
    
    def _find_data_folders(self, path: Path) -> List[Path]:
        """Trouve les dossiers de données"""
        data_folders = []
        
        # Dossiers courants
        common_data_dirs = [
            "Data", "StreamingAssets", "Resources", 
            "Dofus_Data", "Assets", "Content"
        ]
        
        for dir_name in common_data_dirs:
            data_path = path / dir_name
            if data_path.exists():
                data_folders.append(data_path)
        
        return data_folders
    
    def _read_version(self, path: Path) -> str:
        """Lit la version du jeu"""
        # Recherche fichier version
        version_files = ["version.txt", "buildinfo.txt", "app.info"]
        
        for version_file in version_files:
            version_path = path / version_file
            if version_path.exists():
                try:
                    with open(version_path, 'r') as f:
                        return f.read().strip()
                except:
                    pass
        
        return "Unknown"
    
    def _search_drive(self, drive_path: Path, max_depth: int = 3) -> List[DofusInstallation]:
        """Recherche récursive dans un disque"""
        installations = []
        
        try:
            for item in drive_path.iterdir():
                if item.is_dir() and "dofus" in item.name.lower():
                    installation = self._analyze_installation(item)
                    if installation:
                        installations.append(installation)
        except (PermissionError, OSError):
            pass
        
        return installations


class DofusDataExtractor:
    """Extracteur de données Dofus"""
    
    def __init__(self, installation: DofusInstallation, output_dir: str = "data/extracted"):
        self.installation = installation
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extracted_data = {
            "monsters": {},
            "spells": {},
            "items": {},
            "maps": {},
            "npcs": {},
            "quests": {},
            "resources": {}
        }
    
    def extract_all(self):
        """Extrait toutes les données"""
        logger.info("🔍 Début extraction données Dofus Unity...")
        
        # 1. Recherche fichiers JSON
        self._extract_json_files()
        
        # 2. Recherche fichiers XML
        self._extract_xml_files()
        
        # 3. Analyse Assets Unity
        self._extract_unity_assets()
        
        # 4. Extraction depuis bases de données
        self._extract_databases()
        
        # 5. Sauvegarde résultats
        self._save_extracted_data()
        
        logger.info("✅ Extraction terminée !")
        return self.extracted_data
    
    def _extract_json_files(self):
        """Extrait les fichiers JSON"""
        logger.info("📄 Extraction fichiers JSON...")
        
        for data_folder in self.installation.data_folders:
            json_files = list(data_folder.rglob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Classification par type
                    self._classify_json_data(json_file.stem, data)
                    
                except Exception as e:
                    logger.debug(f"Erreur lecture {json_file.name}: {e}")
    
    def _extract_xml_files(self):
        """Extrait les fichiers XML"""
        logger.info("📄 Extraction fichiers XML...")
        
        try:
            from xml.etree import ElementTree as ET
            
            for data_folder in self.installation.data_folders:
                xml_files = list(data_folder.rglob("*.xml"))
                
                for xml_file in xml_files:
                    try:
                        tree = ET.parse(xml_file)
                        root = tree.getroot()
                        
                        # Conversion XML → Dict
                        data = self._xml_to_dict(root)
                        self._classify_json_data(xml_file.stem, data)
                        
                    except Exception as e:
                        logger.debug(f"Erreur lecture {xml_file.name}: {e}")
        
        except ImportError:
            logger.warning("Module XML non disponible")
    
    def _extract_unity_assets(self):
        """Extrait les Unity Assets"""
        logger.info("🎮 Analyse Unity Assets...")
        
        # Recherche fichiers .assets
        for data_folder in self.installation.data_folders:
            asset_files = list(data_folder.rglob("*.assets"))
            
            for asset_file in asset_files:
                # Note: Nécessite UnityPy pour extraction complète
                logger.debug(f"Asset trouvé: {asset_file.name}")
    
    def _extract_databases(self):
        """Extrait depuis bases de données internes"""
        logger.info("💾 Extraction bases de données...")
        
        # Recherche fichiers .db, .sqlite, .dat
        db_extensions = [".db", ".sqlite", ".sqlite3", ".dat"]
        
        for data_folder in self.installation.data_folders:
            for ext in db_extensions:
                db_files = list(data_folder.rglob(f"*{ext}"))
                
                for db_file in db_files:
                    logger.debug(f"Base de données trouvée: {db_file.name}")
                    # Note: Nécessite sqlite3 pour extraction
    
    def _classify_json_data(self, filename: str, data: Any):
        """Classifie les données JSON par type"""
        filename_lower = filename.lower()
        
        # Détection par nom de fichier
        if any(keyword in filename_lower for keyword in ["monster", "mob", "creature"]):
            self._process_monster_data(data)
        
        elif any(keyword in filename_lower for keyword in ["spell", "sort", "skill"]):
            self._process_spell_data(data)
        
        elif any(keyword in filename_lower for keyword in ["item", "equipment", "objet"]):
            self._process_item_data(data)
        
        elif any(keyword in filename_lower for keyword in ["map", "carte", "world"]):
            self._process_map_data(data)
        
        elif any(keyword in filename_lower for keyword in ["npc", "pnj"]):
            self._process_npc_data(data)
        
        elif any(keyword in filename_lower for keyword in ["quest", "quete", "mission"]):
            self._process_quest_data(data)
        
        elif any(keyword in filename_lower for keyword in ["resource", "ressource", "harvest"]):
            self._process_resource_data(data)
    
    def _process_monster_data(self, data: Any):
        """Traite les données de monstres"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and "level" in value:
                    monster_id = value.get("id", key)
                    self.extracted_data["monsters"][monster_id] = value
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "level" in item:
                    monster_id = item.get("id", item.get("name", "unknown"))
                    self.extracted_data["monsters"][monster_id] = item
    
    def _process_spell_data(self, data: Any):
        """Traite les données de sorts"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and any(k in value for k in ["damage", "ap_cost", "range"]):
                    spell_id = value.get("id", key)
                    self.extracted_data["spells"][spell_id] = value
    
    def _process_item_data(self, data: Any):
        """Traite les données d'items"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and any(k in value for k in ["type", "level", "stats"]):
                    item_id = value.get("id", key)
                    self.extracted_data["items"][item_id] = value
    
    def _process_map_data(self, data: Any):
        """Traite les données de cartes"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and any(k in value for k in ["coordinates", "area", "subarea"]):
                    map_id = value.get("id", key)
                    self.extracted_data["maps"][map_id] = value
    
    def _process_npc_data(self, data: Any):
        """Traite les données de NPCs"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and any(k in value for k in ["dialogue", "shop", "quest"]):
                    npc_id = value.get("id", key)
                    self.extracted_data["npcs"][npc_id] = value
    
    def _process_quest_data(self, data: Any):
        """Traite les données de quêtes"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and any(k in value for k in ["objectives", "rewards", "steps"]):
                    quest_id = value.get("id", key)
                    self.extracted_data["quests"][quest_id] = value
    
    def _process_resource_data(self, data: Any):
        """Traite les données de ressources"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and any(k in value for k in ["harvest_level", "location"]):
                    resource_id = value.get("id", key)
                    self.extracted_data["resources"][resource_id] = value
    
    def _xml_to_dict(self, element) -> Dict:
        """Convertit XML en dictionnaire"""
        result = {}
        
        # Attributs
        if element.attrib:
            result.update(element.attrib)
        
        # Texte
        if element.text and element.text.strip():
            result["_text"] = element.text.strip()
        
        # Enfants
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    def _save_extracted_data(self):
        """Sauvegarde les données extraites"""
        logger.info("💾 Sauvegarde données extraites...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for data_type, data in self.extracted_data.items():
            if data:
                output_file = self.output_dir / f"{data_type}_{timestamp}.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"  ✅ {data_type}: {len(data)} entrées → {output_file.name}")


class FansiteDataFetcher:
    """Récupère données depuis fansites"""
    
    FANSITES = {
        "dofus_db": "https://dofusdb.fr",
        "dofus_pour_les_noobs": "https://www.dofuspourlesnoobs.com",
        "dofus_book": "https://dofusbook.net",
        "krosmoz": "https://www.krosmoz.com/fr/dofus"
    }
    
    def __init__(self, output_dir: str = "data/fansite"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_monster_data(self, monster_name: str) -> Optional[Dict]:
        """Récupère données d'un monstre depuis fansites"""
        logger.info(f"🌐 Recherche {monster_name} sur fansites...")
        
        # Tentative DofusDB
        data = self._fetch_from_dofusdb(monster_name, "monsters")
        if data:
            return data
        
        # Tentative Dofus Pour Les Noobs
        data = self._fetch_from_dpln(monster_name)
        if data:
            return data
        
        return None
    
    def _fetch_from_dofusdb(self, name: str, category: str) -> Optional[Dict]:
        """Récupère depuis DofusDB"""
        try:
            # Recherche
            search_url = f"{self.FANSITES['dofus_db']}/fr/search"
            params = {"q": name, "type": category}
            
            response = self.session.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Parsing HTML (à adapter selon structure réelle)
                data = self._parse_dofusdb_page(soup)
                return data
        
        except Exception as e:
            logger.debug(f"Erreur DofusDB: {e}")
        
        return None
    
    def _fetch_from_dpln(self, name: str) -> Optional[Dict]:
        """Récupère depuis Dofus Pour Les Noobs"""
        try:
            # URL de recherche
            search_url = f"{self.FANSITES['dofus_pour_les_noobs']}/search"
            params = {"q": name}
            
            response = self.session.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                data = self._parse_dpln_page(soup)
                return data
        
        except Exception as e:
            logger.debug(f"Erreur DPLN: {e}")
        
        return None
    
    def _parse_dofusdb_page(self, soup: BeautifulSoup) -> Dict:
        """Parse page DofusDB"""
        # À implémenter selon structure HTML réelle
        return {}
    
    def _parse_dpln_page(self, soup: BeautifulSoup) -> Dict:
        """Parse page DPLN"""
        # À implémenter selon structure HTML réelle
        return {}


class DofusDataManager:
    """Gestionnaire principal des données Dofus"""
    
    def __init__(self):
        self.path_finder = DofusPathFinder()
        self.fansite_fetcher = FansiteDataFetcher()
        self.installations = []
        self.extractors = []
    
    def setup(self):
        """Configuration initiale"""
        logger.info("🚀 Configuration Dofus Data Manager...")
        
        # 1. Recherche installations
        self.installations = self.path_finder.find_installations()
        
        if not self.installations:
            logger.warning("⚠️ Aucune installation Dofus trouvée localement")
            logger.info("💡 Utilisation des fansites uniquement")
            return False
        
        # 2. Création extracteurs
        for installation in self.installations:
            extractor = DofusDataExtractor(installation)
            self.extractors.append(extractor)
        
        logger.info(f"✅ {len(self.installations)} installation(s) trouvée(s)")
        return True
    
    def extract_all_data(self):
        """Extrait toutes les données disponibles"""
        all_data = {
            "monsters": {},
            "spells": {},
            "items": {},
            "maps": {},
            "npcs": {},
            "quests": {},
            "resources": {}
        }
        
        # Extraction locale
        for extractor in self.extractors:
            extracted = extractor.extract_all()
            
            # Fusion données
            for category, data in extracted.items():
                all_data[category].update(data)
        
        # Complément avec fansites si nécessaire
        if not all_data["monsters"]:
            logger.info("📡 Récupération données depuis fansites...")
            # À implémenter: fetch depuis fansites
        
        return all_data
    
    def get_monster_info(self, monster_name: str) -> Optional[Dict]:
        """Récupère infos d'un monstre (local ou fansite)"""
        # 1. Recherche locale
        for extractor in self.extractors:
            if monster_name in extractor.extracted_data["monsters"]:
                return extractor.extracted_data["monsters"][monster_name]
        
        # 2. Recherche fansite
        return self.fansite_fetcher.fetch_monster_data(monster_name)


def main():
    """Point d'entrée principal"""
    print("=" * 60)
    print("🎮 DOFUS UNITY DATA EXTRACTOR")
    print("=" * 60)
    print()
    
    manager = DofusDataManager()
    
    # Configuration
    has_local = manager.setup()
    print()
    
    if has_local:
        # Extraction données locales
        print("📊 Extraction des données locales...")
        all_data = manager.extract_all_data()
        
        print()
        print("✅ RÉSULTATS:")
        for category, data in all_data.items():
            if data:
                print(f"  • {category.capitalize()}: {len(data)} entrées")
        
    else:
        print("💡 Mode fansite uniquement")
        print("   Les données seront récupérées en ligne à la demande")
    
    print()
    print("=" * 60)
    print("✅ Extraction terminée !")
    print("=" * 60)


if __name__ == "__main__":
    main()
