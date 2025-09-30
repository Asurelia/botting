"""
DOFUS Data Extractor - Extraction Securisee depuis Installation Unity
Extraction des vraies donnees DOFUS depuis les bundles Unity
Approche 100% lecture seule - Aucune modification des fichiers originaux
"""

import json
import struct
import os
import zipfile
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import hashlib
from dataclasses import dataclass
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class BundleInfo:
    """Informations sur un bundle Unity"""
    name: str
    path: str
    size: int
    hash: str
    type: str  # "spells", "monsters", "items", etc.

@dataclass
class ExtractedData:
    """Donnees extraites d'un bundle"""
    bundle_name: str
    data_type: str
    content: Any
    extraction_time: str
    success: bool
    error_message: Optional[str] = None

class DofusDataExtractor:
    """
    Extracteur securise pour donnees DOFUS Unity
    Lecture seule - Preservation integrite fichiers originaux
    """

    def __init__(self, dofus_path: str = r"C:\Users\rafai\AppData\Local\Ankama\Dofus-dofus3"):
        self.dofus_path = Path(dofus_path)
        self.data_path = self.dofus_path / "Dofus_Data" / "StreamingAssets" / "Content" / "Data"

        # Cache et securite
        self.extracted_cache: Dict[str, ExtractedData] = {}
        self.bundle_info_cache: Dict[str, BundleInfo] = {}

        # Verification acces
        self._verify_access()

        # Base de donnees cache
        self.cache_db_path = Path("data/extraction_cache.db")
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache_db()

        logger.info(f"DofusDataExtractor initialise: {self.data_path}")

    def _verify_access(self):
        """Verifie l'acces aux fichiers DOFUS"""
        if not self.dofus_path.exists():
            raise FileNotFoundError(f"Installation DOFUS non trouvee: {self.dofus_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dossier donnees non trouve: {self.data_path}")

        # Test lecture sur un fichier
        test_files = list(self.data_path.glob("*.bundle"))
        if not test_files:
            raise FileNotFoundError("Aucun bundle trouve dans le dossier donnees")

        test_file = test_files[0]
        try:
            with open(test_file, 'rb') as f:
                f.read(10)  # Test lecture
            logger.info("Acces verification reussie")
        except PermissionError:
            raise PermissionError(f"Permissions insuffisantes pour lire: {test_file}")

    def _init_cache_db(self):
        """Initialise la base de donnees cache"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extraction_cache (
                bundle_name TEXT PRIMARY KEY,
                data_type TEXT,
                content_hash TEXT,
                extraction_time TEXT,
                success BOOLEAN,
                data_json TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def get_bundle_info(self, bundle_name: str) -> Optional[BundleInfo]:
        """Recupere les informations d'un bundle"""
        if bundle_name in self.bundle_info_cache:
            return self.bundle_info_cache[bundle_name]

        bundle_path = self.data_path / bundle_name
        if not bundle_path.exists():
            return None

        # Calcul hash pour verification integrite
        with open(bundle_path, 'rb') as f:
            content = f.read()
            file_hash = hashlib.sha256(content).hexdigest()

        # Determination du type depuis le nom
        data_type = self._determine_data_type(bundle_name)

        bundle_info = BundleInfo(
            name=bundle_name,
            path=str(bundle_path),
            size=len(content),
            hash=file_hash,
            type=data_type
        )

        self.bundle_info_cache[bundle_name] = bundle_info
        return bundle_info

    def _determine_data_type(self, bundle_name: str) -> str:
        """Determine le type de donnees depuis le nom du bundle"""
        name_lower = bundle_name.lower()

        if "spells" in name_lower:
            return "spells"
        elif "monsters" in name_lower:
            return "monsters"
        elif "items" in name_lower:
            return "items"
        elif "maps" in name_lower:
            return "maps"
        elif "recipes" in name_lower:
            return "recipes"
        elif "interactives" in name_lower:
            return "resources"
        elif "areas" in name_lower or "subareas" in name_lower:
            return "areas"
        else:
            return "other"

    def list_available_bundles(self) -> List[BundleInfo]:
        """Liste tous les bundles disponibles"""
        bundles = []

        for bundle_file in self.data_path.glob("*.bundle"):
            bundle_info = self.get_bundle_info(bundle_file.name)
            if bundle_info:
                bundles.append(bundle_info)

        # Tri par type puis taille
        bundles.sort(key=lambda x: (x.type, -x.size))
        return bundles

    def _try_extract_as_json(self, data: bytes) -> Optional[Any]:
        """Tente d'extraire comme JSON"""
        try:
            # Test direct JSON
            text = data.decode('utf-8')
            return json.loads(text)
        except:
            pass

        try:
            # Test UTF-8 avec BOM
            if data.startswith(b'\xef\xbb\xbf'):
                text = data[3:].decode('utf-8')
                return json.loads(text)
        except:
            pass

        return None

    def _try_extract_as_unity_text(self, data: bytes) -> Optional[Any]:
        """Tente d'extraire comme Unity TextAsset"""
        try:
            # Unity TextAsset peut avoir un header
            # Recherche patterns JSON dans les donnees
            text = data.decode('utf-8', errors='ignore')

            # Recherche debut JSON
            start_pos = text.find('{')
            end_pos = text.rfind('}')

            if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
                json_text = text[start_pos:end_pos+1]
                return json.loads(json_text)

            # Recherche array JSON
            start_pos = text.find('[')
            end_pos = text.rfind(']')

            if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
                json_text = text[start_pos:end_pos+1]
                return json.loads(json_text)

        except Exception as e:
            logger.debug(f"Erreur extraction Unity text: {e}")

        return None

    def _try_extract_as_compressed(self, data: bytes) -> Optional[Any]:
        """Tente d'extraire comme archive compressee"""
        try:
            # Test ZIP
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(data)
                temp_file.flush()

                with zipfile.ZipFile(temp_file.name, 'r') as zip_file:
                    for file_name in zip_file.namelist():
                        if file_name.endswith('.json'):
                            with zip_file.open(file_name) as json_file:
                                return json.load(json_file)
        except:
            pass

        return None

    def _extract_bundle_content(self, bundle_path: Path) -> Optional[Any]:
        """Extrait le contenu d'un bundle Unity"""
        try:
            with open(bundle_path, 'rb') as f:
                data = f.read()

            # Tentatives d'extraction multiples
            extraction_methods = [
                self._try_extract_as_json,
                self._try_extract_as_unity_text,
                self._try_extract_as_compressed
            ]

            for method in extraction_methods:
                try:
                    result = method(data)
                    if result is not None:
                        logger.info(f"Extraction reussie: {bundle_path.name} via {method.__name__}")
                        return result
                except Exception as e:
                    logger.debug(f"Methode {method.__name__} echouee pour {bundle_path.name}: {e}")

            # Si aucune methode ne fonctionne, analyse binaire basique
            logger.warning(f"Extraction JSON echouee pour {bundle_path.name}, analyse binaire...")
            return self._analyze_binary_content(data)

        except Exception as e:
            logger.error(f"Erreur lecture bundle {bundle_path}: {e}")
            return None

    def _analyze_binary_content(self, data: bytes) -> Dict[str, Any]:
        """Analyse basique du contenu binaire"""
        analysis = {
            "type": "binary",
            "size": len(data),
            "header": data[:32].hex() if len(data) >= 32 else data.hex(),
            "contains_text": self._detect_text_content(data),
            "possible_json_blocks": self._find_json_blocks(data)
        }

        return analysis

    def _detect_text_content(self, data: bytes) -> bool:
        """Detecte si les donnees contiennent du texte"""
        try:
            text = data.decode('utf-8', errors='ignore')
            # Ratio de caracteres printables
            printable_ratio = sum(1 for c in text if c.isprintable()) / len(text)
            return printable_ratio > 0.7
        except:
            return False

    def _find_json_blocks(self, data: bytes) -> List[str]:
        """Trouve les blocs JSON potentiels dans les donnees binaires"""
        json_blocks = []

        try:
            text = data.decode('utf-8', errors='ignore')

            # Recherche tous les blocs { ... }
            brace_level = 0
            start_pos = -1

            for i, char in enumerate(text):
                if char == '{':
                    if brace_level == 0:
                        start_pos = i
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == 0 and start_pos != -1:
                        json_candidate = text[start_pos:i+1]
                        if len(json_candidate) > 50:  # Filtre les petits blocs
                            try:
                                json.loads(json_candidate)
                                json_blocks.append(json_candidate[:200] + "..." if len(json_candidate) > 200 else json_candidate)
                            except:
                                pass

        except Exception as e:
            logger.debug(f"Erreur recherche JSON blocks: {e}")

        return json_blocks[:5]  # Max 5 blocs pour limiter la taille

    def extract_bundle_data(self, bundle_name: str, force_refresh: bool = False) -> ExtractedData:
        """Extrait les donnees d'un bundle specifique"""

        # Verification cache
        if not force_refresh and bundle_name in self.extracted_cache:
            return self.extracted_cache[bundle_name]

        # Verification cache DB
        if not force_refresh:
            cached_data = self._load_from_cache_db(bundle_name)
            if cached_data:
                self.extracted_cache[bundle_name] = cached_data
                return cached_data

        bundle_info = self.get_bundle_info(bundle_name)
        if not bundle_info:
            error_data = ExtractedData(
                bundle_name=bundle_name,
                data_type="unknown",
                content=None,
                extraction_time="",
                success=False,
                error_message=f"Bundle non trouve: {bundle_name}"
            )
            return error_data

        # Extraction
        logger.info(f"Extraction bundle: {bundle_name} ({bundle_info.size} bytes)")

        try:
            content = self._extract_bundle_content(Path(bundle_info.path))

            extracted_data = ExtractedData(
                bundle_name=bundle_name,
                data_type=bundle_info.type,
                content=content,
                extraction_time=str(Path(bundle_info.path).stat().st_mtime),
                success=content is not None
            )

            if not extracted_data.success:
                extracted_data.error_message = "Echec extraction contenu"

            # Sauvegarde cache
            self.extracted_cache[bundle_name] = extracted_data
            self._save_to_cache_db(extracted_data)

            return extracted_data

        except Exception as e:
            error_data = ExtractedData(
                bundle_name=bundle_name,
                data_type=bundle_info.type,
                content=None,
                extraction_time="",
                success=False,
                error_message=str(e)
            )
            return error_data

    def _load_from_cache_db(self, bundle_name: str) -> Optional[ExtractedData]:
        """Charge depuis le cache DB"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT data_type, extraction_time, success, data_json FROM extraction_cache WHERE bundle_name = ?",
                (bundle_name,)
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                data_type, extraction_time, success, data_json = row
                content = json.loads(data_json) if data_json else None

                return ExtractedData(
                    bundle_name=bundle_name,
                    data_type=data_type,
                    content=content,
                    extraction_time=extraction_time,
                    success=bool(success)
                )
        except Exception as e:
            logger.debug(f"Erreur chargement cache: {e}")

        return None

    def _save_to_cache_db(self, extracted_data: ExtractedData):
        """Sauvegarde dans le cache DB"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            data_json = json.dumps(extracted_data.content) if extracted_data.content else None
            content_hash = hashlib.md5(data_json.encode() if data_json else b"").hexdigest()

            cursor.execute('''
                INSERT OR REPLACE INTO extraction_cache
                (bundle_name, data_type, content_hash, extraction_time, success, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                extracted_data.bundle_name,
                extracted_data.data_type,
                content_hash,
                extracted_data.extraction_time,
                extracted_data.success,
                data_json
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Erreur sauvegarde cache: {e}")

    def extract_spells_data(self) -> Dict[str, ExtractedData]:
        """Extrait toutes les donnees de sorts"""
        spell_bundles = [
            "data_assets_spellsdataroot.asset.bundle",
            "data_assets_spelllevelsdataroot.asset.bundle",
            "data_assets_spellstatesdataroot.asset.bundle",
            "data_assets_spelltypesdataroot.asset.bundle"
        ]

        results = {}
        for bundle_name in spell_bundles:
            results[bundle_name] = self.extract_bundle_data(bundle_name)

        return results

    def extract_monsters_data(self) -> Dict[str, ExtractedData]:
        """Extrait toutes les donnees de monstres"""
        monster_bundles = [
            "data_assets_monstersdataroot.asset.bundle",
            "data_assets_monsterracesdataroot.asset.bundle",
            "data_assets_monstersuperracesdataroot.asset.bundle"
        ]

        results = {}
        for bundle_name in monster_bundles:
            results[bundle_name] = self.extract_bundle_data(bundle_name)

        return results

    def extract_maps_data(self) -> Dict[str, ExtractedData]:
        """Extrait toutes les donnees de cartes"""
        map_bundles = [
            "data_assets_mapsinformationdataroot.asset.bundle",
            "data_assets_mapscoordinatesdataroot.asset.bundle",
            "data_assets_areasdataroot.asset.bundle",
            "data_assets_subareasdataroot.asset.bundle"
        ]

        results = {}
        for bundle_name in map_bundles:
            results[bundle_name] = self.extract_bundle_data(bundle_name)

        return results

    def extract_items_data(self) -> Dict[str, ExtractedData]:
        """Extrait toutes les donnees d'objets"""
        item_bundles = [
            "data_assets_itemsdataroot.asset.bundle",
            "data_assets_itemsetsdataroot.asset.bundle",
            "data_assets_itemtypesdataroot.asset.bundle"
        ]

        results = {}
        for bundle_name in item_bundles:
            results[bundle_name] = self.extract_bundle_data(bundle_name)

        return results

    def generate_extraction_report(self) -> Dict[str, Any]:
        """Genere un rapport d'extraction complet"""
        bundles = self.list_available_bundles()

        report = {
            "total_bundles": len(bundles),
            "bundles_by_type": {},
            "extraction_summary": {},
            "recommendations": []
        }

        # Groupement par type
        for bundle in bundles:
            if bundle.type not in report["bundles_by_type"]:
                report["bundles_by_type"][bundle.type] = []
            report["bundles_by_type"][bundle.type].append({
                "name": bundle.name,
                "size_kb": bundle.size // 1024
            })

        # Recommandations d'extraction prioritaire
        priority_bundles = [
            ("spells", "data_assets_spellsdataroot.asset.bundle"),
            ("monsters", "data_assets_monstersdataroot.asset.bundle"),
            ("items", "data_assets_itemsdataroot.asset.bundle"),
            ("maps", "data_assets_mapsinformationdataroot.asset.bundle")
        ]

        for data_type, bundle_name in priority_bundles:
            bundle_info = self.get_bundle_info(bundle_name)
            if bundle_info:
                report["recommendations"].append({
                    "type": data_type,
                    "bundle": bundle_name,
                    "size_mb": bundle_info.size / (1024 * 1024),
                    "priority": "high"
                })

        return report

# Instance globale
_extractor_instance = None

def get_dofus_extractor() -> DofusDataExtractor:
    """Retourne l'instance singleton de l'extracteur"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = DofusDataExtractor()
    return _extractor_instance

# Test du module
if __name__ == "__main__":
    try:
        extractor = DofusDataExtractor()

        # Rapport general
        report = extractor.generate_extraction_report()
        print(f"Bundles disponibles: {report['total_bundles']}")
        print("Types de donnees:")
        for data_type, bundles in report["bundles_by_type"].items():
            print(f"  {data_type}: {len(bundles)} bundles")

        # Test extraction sorts
        print("\nTest extraction sorts...")
        spells_data = extractor.extract_spells_data()
        for bundle_name, extracted in spells_data.items():
            status = "SUCCES" if extracted.success else "ECHEC"
            print(f"  {bundle_name}: {status}")
            if extracted.error_message:
                print(f"    Erreur: {extracted.error_message}")

    except Exception as e:
        print(f"Erreur test extracteur: {e}")