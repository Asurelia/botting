"""
Fetch All Monsters from DofusDB
Récupère automatiquement TOUS les monstres depuis DofusDB
"""

import json
import requests
import time
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DofusDBFetcher:
    """Récupère tous les monstres depuis DofusDB"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.base_url = "https://dofusdb.fr"
        self.monsters = {}
    
    def fetch_all_monsters(self, max_monsters: int = 100):
        """
        Récupère tous les monstres depuis DofusDB
        
        Args:
            max_monsters: Nombre maximum de monstres à récupérer (pour éviter surcharge)
        """
        logger.info(f"🌐 Récupération de {max_monsters} monstres depuis DofusDB...")
        
        # Liste des monstres populaires par niveau
        monster_list = self._get_monster_list()
        
        fetched = 0
        for i, monster_name in enumerate(monster_list[:max_monsters], 1):
            logger.info(f"   [{i}/{min(max_monsters, len(monster_list))}] {monster_name}...")
            
            monster_data = self._fetch_monster(monster_name)
            
            if monster_data:
                monster_id = monster_name.lower().replace(" ", "_").replace("'", "")
                self.monsters[monster_id] = monster_data
                fetched += 1
                logger.info(f"      ✅ Récupéré")
            else:
                logger.debug(f"      ⚠️ Non trouvé")
            
            # Rate limiting (1 requête par seconde)
            time.sleep(1)
        
        logger.info(f"✅ {fetched} monstres récupérés avec succès !")
        return self.monsters
    
    def _get_monster_list(self) -> list:
        """
        Liste complète des monstres Dofus par niveau
        Source: https://dofusdb.fr/fr/database/monsters/
        """
        return [
            # Niveau 1-20
            "Piou", "Larve Bleue", "Bouftou", "Tofu", "Rose Démoniaque",
            "Arakne", "Moskito", "Chafer", "Craqueleur", "Larve Orange",
            "Pissenlit Diabolique", "Tournesol Sauvage", "Corbac",
            "Kwoan", "Bambouto", "Boo", "Wabbit",
            
            # Niveau 20-50
            "Tofu Maléfique", "Bouftou Royal", "Bwork", "Bwork Mage",
            "Koalak", "Kanigrou", "Prespic", "Abra Kadabra",
            "Sanglier", "Crocodaille", "Dragodinde Sauvage",
            "Mulou", "Firefoux", "Biblop", "Bworkette",
            
            # Niveau 50-100
            "Craqueleur Légendaire", "Blop Royal", "Wa Wabbit",
            "Chafer Rōnin", "Abraknyde Ancestral", "Gelée Royale",
            "Scarafeuille", "Scarabosse", "Gourlo le Terrible",
            "Kanniboul", "Meulou", "Pandikaze", "Bulbiflore",
            "Chiendent", "Craqueboule", "Maître Corbac",
            
            # Niveau 100-150
            "Dragon Cochon", "Rat Blanc", "Rat Noir", "Rasboul",
            "Shin Larve", "Tanukoui", "Tanukouisan", "Kitsou Nae",
            "Kitsou Nah", "Tengu Nae", "Tengu Nah", "Oni Nae",
            "Oni Nah", "Kaniglou", "Kanigloo", "Kanigroula",
            
            # Niveau 150-200
            "Minotot", "Minotoror", "Megamimog", "Grozilla",
            "Grozepin", "Tynril", "Tynril Ahuri", "Obsidiantre",
            "Kolosso", "Missiz Frizz", "Mufafah", "Sphincter Cell",
            "Gourlo le Terrible", "Moon", "Anerice", "Klime",
            
            # Boss et Archimonstres
            "Wa Wabbit", "Shin Larve", "Rat Blanc", "Rat Noir",
            "Tofu Royal", "Bouftou Royal", "Craqueleur Légendaire",
            "Blop Royal Multicolore", "Gelée Royale Bleue",
            "Gelée Royale Menthe", "Gelée Royale Fraise",
            "Dragon Cochon", "Kimbo", "Tanukoui San", "Tengu Nae",
            
            # Donjons populaires
            "Maître Corbac", "Skeunk", "Chafer Draugr", "Chafer Rōnin",
            "Grozilla", "Grozepin", "Tynril", "Obsidiantre",
            "Kolosso", "Missiz Frizz", "Mufafah", "Sphincter Cell",
            
            # Frigost
            "Gourlo", "Kaniglou", "Kanigloo", "Kanigroula",
            "Frozard", "Glaçon", "Bwork Givrefoux", "Floristile",
            "Fungus", "Champodonte", "Champbis", "Champaknyde",
            
            # Zones diverses
            "Pichon", "Pichon Blanc", "Pichon Noir", "Pichon Kloune",
            "Boo", "Fantôme Boo", "Fantôme Ardent", "Fantôme Apero",
            "Abraknyde", "Abraknyde Vénérable", "Abraknyde Ancestral",
            "Bulbe", "Bulbiflore", "Bulbuisson", "Bulbambou"
        ]
    
    def _fetch_monster(self, monster_name: str) -> dict:
        """Récupère les données d'un monstre spécifique"""
        try:
            # Note: L'API réelle de DofusDB peut être différente
            # Ceci est un exemple - à adapter selon l'API disponible
            
            # Tentative 1: API directe (si disponible)
            api_url = f"{self.base_url}/api/monsters"
            params = {"name": monster_name, "lang": "fr"}
            
            response = self.session.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    return self._normalize_monster(data[0] if isinstance(data, list) else data)
            
            # Tentative 2: Scraping page (fallback)
            return self._scrape_monster_page(monster_name)
        
        except Exception as e:
            logger.debug(f"Erreur fetch {monster_name}: {e}")
            return None
    
    def _scrape_monster_page(self, monster_name: str) -> dict:
        """Scrape la page du monstre (fallback)"""
        try:
            from bs4 import BeautifulSoup
            
            # Recherche du monstre
            search_url = f"{self.base_url}/fr/recherche"
            params = {"q": monster_name}
            
            response = self.session.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Parsing HTML (à adapter selon structure réelle)
                # ...
                pass
        
        except Exception as e:
            logger.debug(f"Erreur scraping {monster_name}: {e}")
        
        return None
    
    def _normalize_monster(self, raw_data: dict) -> dict:
        """Normalise les données au format du bot"""
        return {
            "id": raw_data.get("id", ""),
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
            "source": "dofusdb",
            "fetched_at": datetime.now().isoformat()
        }
    
    def save_to_file(self, output_file: str = "data/dofusdb_monsters.json"):
        """Sauvegarde les monstres dans un fichier"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.monsters, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Sauvegardé dans {output_path}")


def main():
    print("=" * 70)
    print("🌐 RÉCUPÉRATION MONSTRES DEPUIS DOFUSDB")
    print("=" * 70)
    print()
    
    fetcher = DofusDBFetcher()
    
    # Demander combien de monstres
    print("Combien de monstres voulez-vous récupérer ?")
    print("  - 10  : Test rapide")
    print("  - 50  : Collection moyenne")
    print("  - 100 : Collection complète (recommandé)")
    print("  - 200 : Tous les monstres principaux")
    print()
    
    try:
        max_monsters = int(input("Nombre (défaut 100): ") or "100")
    except:
        max_monsters = 100
    
    print()
    
    # Récupération
    monsters = fetcher.fetch_all_monsters(max_monsters)
    
    # Sauvegarde
    fetcher.save_to_file()
    
    print()
    print("=" * 70)
    print(f"✅ {len(monsters)} monstres récupérés et sauvegardés !")
    print("=" * 70)
    print()
    print("📁 Fichier: data/dofusdb_monsters.json")
    print()
    print("💡 Pour intégrer au bot:")
    print("   python tools/data_consolidator.py")
    print()


if __name__ == "__main__":
    main()
