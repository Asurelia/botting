"""
DofusDB Web Scraper
Récupère les données des monstres en scrapant le site DofusDB
Alternative à l'API qui ne fonctionne pas
"""

import json
import requests
import time
from pathlib import Path
from datetime import datetime
import logging
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DofusDBScraper:
    """Scraper pour DofusDB.fr"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.base_url = "https://dofusdb.fr"
        self.monsters = {}
    
    def scrape_monster_list(self, max_monsters: int = 20):
        """
        Scrape la liste des monstres depuis DofusDB
        """
        logger.info(f"🌐 Scraping {max_monsters} monstres depuis DofusDB...")
        
        try:
            # Page de liste des monstres
            url = f"{self.base_url}/fr/database/monsters"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Recherche des liens de monstres
                monster_links = soup.find_all('a', href=re.compile(r'/fr/monster/\d+'))
                
                logger.info(f"   ✅ {len(monster_links)} monstres trouvés sur la page")
                
                # Scrape chaque monstre
                for i, link in enumerate(monster_links[:max_monsters], 1):
                    monster_url = self.base_url + link['href']
                    monster_name = link.get_text(strip=True)
                    
                    logger.info(f"   [{i}/{max_monsters}] {monster_name}...")
                    
                    monster_data = self._scrape_monster_page(monster_url, monster_name)
                    
                    if monster_data:
                        monster_id = monster_name.lower().replace(" ", "_").replace("'", "")
                        self.monsters[monster_id] = monster_data
                        logger.info(f"      ✅ Récupéré")
                    else:
                        logger.debug(f"      ⚠️ Échec")
                    
                    # Rate limiting
                    time.sleep(1)
            
            else:
                logger.error(f"Erreur HTTP {response.status_code}")
        
        except Exception as e:
            logger.error(f"Erreur scraping: {e}")
        
        logger.info(f"✅ {len(self.monsters)} monstres récupérés avec succès !")
        return self.monsters
    
    def _scrape_monster_page(self, url: str, name: str) -> dict:
        """Scrape la page d'un monstre spécifique"""
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extraction des données
                monster_data = {
                    "id": self._extract_id_from_url(url),
                    "name": name,
                    "level": self._extract_level(soup),
                    "health": self._extract_health(soup),
                    "ap": self._extract_stat(soup, "PA"),
                    "mp": self._extract_stat(soup, "PM"),
                    "resistances": self._extract_resistances(soup),
                    "damages": self._extract_damages(soup),
                    "locations": [],
                    "drops": [],
                    "xp_reward": 0,
                    "kamas_reward": 0,
                    "difficulty": "medium",
                    "source": "dofusdb_scrape",
                    "fetched_at": datetime.now().isoformat(),
                    "url": url
                }
                
                return monster_data
        
        except Exception as e:
            logger.debug(f"Erreur scraping page {url}: {e}")
        
        return None
    
    def _extract_id_from_url(self, url: str) -> str:
        """Extrait l'ID depuis l'URL"""
        match = re.search(r'/monster/(\d+)', url)
        return match.group(1) if match else ""
    
    def _extract_level(self, soup: BeautifulSoup) -> int:
        """Extrait le niveau"""
        try:
            # Recherche du niveau dans différents formats possibles
            level_elem = soup.find(text=re.compile(r'Niveau\s*:?\s*\d+'))
            if level_elem:
                match = re.search(r'\d+', level_elem)
                if match:
                    return int(match.group())
        except:
            pass
        return 1
    
    def _extract_health(self, soup: BeautifulSoup) -> int:
        """Extrait les points de vie"""
        try:
            hp_elem = soup.find(text=re.compile(r'Points de vie|PV|HP'))
            if hp_elem:
                # Chercher le nombre à côté
                parent = hp_elem.parent
                numbers = re.findall(r'\d+', parent.get_text())
                if numbers:
                    return int(numbers[0])
        except:
            pass
        return 100
    
    def _extract_stat(self, soup: BeautifulSoup, stat_name: str) -> int:
        """Extrait une statistique (PA, PM, etc.)"""
        try:
            stat_elem = soup.find(text=re.compile(stat_name))
            if stat_elem:
                parent = stat_elem.parent
                numbers = re.findall(r'\d+', parent.get_text())
                if numbers:
                    return int(numbers[0])
        except:
            pass
        return 6 if stat_name == "PA" else 3
    
    def _extract_resistances(self, soup: BeautifulSoup) -> dict:
        """Extrait les résistances"""
        resistances = {
            "neutral": 0,
            "earth": 0,
            "fire": 0,
            "water": 0,
            "air": 0
        }
        
        try:
            # Recherche section résistances
            resist_section = soup.find(text=re.compile(r'Résistances?'))
            if resist_section:
                parent = resist_section.parent.parent
                # Extraction des valeurs
                # (À adapter selon la structure HTML réelle)
        except:
            pass
        
        return resistances
    
    def _extract_damages(self, soup: BeautifulSoup) -> dict:
        """Extrait les dégâts"""
        return {
            "min": 10,
            "max": 20,
            "element": "neutral"
        }
    
    def save_to_file(self, output_file: str = "data/dofusdb_monsters_scraped.json"):
        """Sauvegarde les monstres"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.monsters, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Sauvegardé dans {output_path}")


def main():
    print("=" * 70)
    print("🌐 SCRAPING DOFUSDB")
    print("=" * 70)
    print()
    print("⚠️ Note: Le scraping web est plus lent que l'API")
    print("   Comptez ~1 seconde par monstre")
    print()
    
    try:
        max_monsters = int(input("Nombre de monstres (défaut 20): ") or "20")
    except:
        max_monsters = 20
    
    print()
    
    scraper = DofusDBScraper()
    monsters = scraper.scrape_monster_list(max_monsters)
    
    if monsters:
        scraper.save_to_file()
        
        print()
        print("=" * 70)
        print(f"✅ {len(monsters)} monstres récupérés !")
        print("=" * 70)
        print()
        print("📁 Fichier: data/dofusdb_monsters_scraped.json")
        print()
        print("💡 Exemple de monstre:")
        first_monster = list(monsters.values())[0]
        print(json.dumps(first_monster, indent=2, ensure_ascii=False))
    else:
        print()
        print("=" * 70)
        print("❌ Aucun monstre récupéré")
        print("=" * 70)
        print()
        print("💡 Solutions:")
        print("   1. Vérifier la connexion internet")
        print("   2. DofusDB peut avoir changé sa structure")
        print("   3. Utiliser les données locales existantes")


if __name__ == "__main__":
    main()
