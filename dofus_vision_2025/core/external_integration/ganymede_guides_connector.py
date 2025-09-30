"""
Connecteur Guides Ganymede - DOFUS Unity World Model AI
Intégration des guides communautaires Ganymede dans le système IA
"""

import json
import time
import requests
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import sqlite3
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

@dataclass
class GanymedeGuide:
    """Guide Ganymede structuré"""
    id: str
    title: str
    category: str
    language: str
    description: str
    content: str
    steps: List[Dict[str, Any]]
    author: str
    creation_date: str
    last_updated: str
    votes: int
    difficulty_level: str
    tags: List[str]
    url: str

@dataclass
class GuideStep:
    """Étape d'un guide"""
    step_number: int
    title: str
    content: str
    images: List[str]
    coordinates: Optional[Dict[str, int]] = None
    items_needed: Optional[List[str]] = None
    monsters_involved: Optional[List[str]] = None

class GanymedeGuidesScraper:
    """Scraper pour les guides Ganymede"""

    def __init__(self):
        self.base_url = "https://ganymede-app.com"
        self.guides_url = f"{self.base_url}/guides"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DOFUS-Unity-World-Model-AI/1.0 (+https://github.com/your-repo)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        # Cache local
        self.cache_file = "ganymede_guides_cache.json"
        self.cache_duration = 3600  # 1 heure
        self._load_cache()

    def _load_cache(self):
        """Charge le cache depuis le fichier"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except Exception as e:
            print(f"[ERROR] Erreur chargement cache: {e}")
            self.cache = {}

    def _save_cache(self):
        """Sauvegarde le cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Erreur sauvegarde cache: {e}")

    def _is_cache_valid(self, key: str) -> bool:
        """Vérifie si l'entrée cache est valide"""
        if key not in self.cache:
            return False

        cache_time = self.cache[key].get('timestamp', 0)
        return (time.time() - cache_time) < self.cache_duration

    def get_guides_list(self, category: str = None, language: str = "fr") -> List[Dict[str, Any]]:
        """Récupère la liste des guides disponibles"""
        cache_key = f"guides_list_{category}_{language}"

        if self._is_cache_valid(cache_key):
            print("[CACHE] Guides charges depuis le cache")
            return self.cache[cache_key]['data']

        print("[FETCH] Récupération guides Ganymede...")

        try:
            # Construire URL
            url = self.guides_url
            params = {}
            if category:
                params['category'] = category
            if language:
                params['lang'] = language

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Parser HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            guides = self._parse_guides_list_page(soup)

            # Mettre en cache
            self.cache[cache_key] = {
                'data': guides,
                'timestamp': time.time()
            }
            self._save_cache()

            print(f"[SUCCESS] {len(guides)} guides récupérés")
            return guides

        except Exception as e:
            print(f"[ERROR] Erreur récupération guides: {e}")
            return []

    def _parse_guides_list_page(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse la page de liste des guides"""
        guides = []

        try:
            # Rechercher les éléments guides (à adapter selon la structure HTML réelle)
            guide_elements = soup.find_all(['div', 'article'], class_=re.compile(r'guide|card'))

            for element in guide_elements:
                guide_data = self._extract_guide_info_from_element(element)
                if guide_data:
                    guides.append(guide_data)

            # Si pas de structure spécifique trouvée, essayer approche générique
            if not guides:
                guides = self._parse_guides_generic(soup)

        except Exception as e:
            print(f"[ERROR] Erreur parsing guides: {e}")

        return guides

    def _extract_guide_info_from_element(self, element) -> Optional[Dict[str, Any]]:
        """Extrait les infos d'un guide depuis un élément HTML"""
        try:
            guide_info = {}

            # Titre
            title_elem = element.find(['h1', 'h2', 'h3', 'h4'], string=re.compile(r'.+'))
            if title_elem:
                guide_info['title'] = title_elem.get_text().strip()

            # Lien
            link_elem = element.find('a', href=True)
            if link_elem:
                guide_info['url'] = urljoin(self.base_url, link_elem['href'])
                guide_info['id'] = self._extract_guide_id_from_url(guide_info['url'])

            # Description
            desc_elem = element.find(['p', 'div'], class_=re.compile(r'desc|summary'))
            if desc_elem:
                guide_info['description'] = desc_elem.get_text().strip()

            # Catégorie
            cat_elem = element.find(['span', 'div'], class_=re.compile(r'cat|tag'))
            if cat_elem:
                guide_info['category'] = cat_elem.get_text().strip()

            # Retourner seulement si on a au moins titre et URL
            if 'title' in guide_info and 'url' in guide_info:
                return guide_info

        except Exception as e:
            print(f"[ERROR] Erreur extraction guide: {e}")

        return None

    def _parse_guides_generic(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse générique si structure spécifique non trouvée"""
        guides = []

        try:
            # Rechercher tous les liens contenant "guide"
            guide_links = soup.find_all('a', href=re.compile(r'/guide'))

            for link in guide_links:
                guide_info = {
                    'title': link.get_text().strip() or 'Guide sans titre',
                    'url': urljoin(self.base_url, link['href']),
                    'id': self._extract_guide_id_from_url(link['href']),
                    'description': '',
                    'category': 'general'
                }

                # Éviter doublons
                if not any(g['url'] == guide_info['url'] for g in guides):
                    guides.append(guide_info)

        except Exception as e:
            print(f"[ERROR] Erreur parsing générique: {e}")

        return guides

    def _extract_guide_id_from_url(self, url: str) -> str:
        """Extrait l'ID du guide depuis l'URL"""
        try:
            # Patterns courants : /guides/123, /guide/123-nom-guide, etc.
            patterns = [
                r'/guides?/(\d+)',
                r'/guides?/([^/]+)',
                r'id=(\d+)'
            ]

            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)

            # Fallback : utiliser la dernière partie de l'URL
            return url.split('/')[-1] or str(hash(url))

        except:
            return str(hash(url))

    def get_guide_content(self, guide_url: str) -> Optional[GanymedeGuide]:
        """Récupère le contenu détaillé d'un guide"""
        cache_key = f"guide_content_{hash(guide_url)}"

        if self._is_cache_valid(cache_key):
            return GanymedeGuide(**self.cache[cache_key]['data'])

        print(f"[GUIDE] Récupération guide: {guide_url}")

        try:
            response = self.session.get(guide_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            guide_data = self._parse_guide_content(soup, guide_url)

            if guide_data:
                # Mettre en cache
                self.cache[cache_key] = {
                    'data': asdict(guide_data),
                    'timestamp': time.time()
                }
                self._save_cache()

                return guide_data

        except Exception as e:
            print(f"[ERROR] Erreur récupération contenu guide: {e}")

        return None

    def _parse_guide_content(self, soup: BeautifulSoup, url: str) -> Optional[GanymedeGuide]:
        """Parse le contenu détaillé d'un guide"""
        try:
            # Extraire titre
            title = "Guide sans titre"
            title_elem = soup.find(['h1', 'h2'])
            if title_elem:
                title = title_elem.get_text().strip()

            # Extraire contenu principal
            content_elem = soup.find(['main', 'article', 'div'], class_=re.compile(r'content|guide|main'))
            content = ""
            if content_elem:
                content = content_elem.get_text().strip()

            # Extraire étapes
            steps = self._extract_guide_steps(soup)

            # Métadonnées
            guide = GanymedeGuide(
                id=self._extract_guide_id_from_url(url),
                title=title,
                category=self._extract_category(soup),
                language="fr",  # Par défaut
                description=self._extract_description(soup),
                content=content,
                steps=steps,
                author=self._extract_author(soup),
                creation_date=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                votes=self._extract_votes(soup),
                difficulty_level=self._extract_difficulty(soup),
                tags=self._extract_tags(soup),
                url=url
            )

            return guide

        except Exception as e:
            print(f"[ERROR] Erreur parsing contenu guide: {e}")
            return None

    def _extract_guide_steps(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extrait les étapes du guide"""
        steps = []

        try:
            # Rechercher éléments d'étapes
            step_elements = soup.find_all(['div', 'section'], class_=re.compile(r'step|etape'))

            if not step_elements:
                # Fallback : rechercher par numérotation
                step_elements = soup.find_all(text=re.compile(r'^(\d+\.|\d+\)|\d+ -)', re.MULTILINE))

            step_num = 1
            for element in step_elements[:20]:  # Limiter à 20 étapes max
                step_data = {
                    'step_number': step_num,
                    'title': f"Étape {step_num}",
                    'content': '',
                    'images': [],
                    'coordinates': None,
                    'items_needed': [],
                    'monsters_involved': []
                }

                if hasattr(element, 'get_text'):
                    step_data['content'] = element.get_text().strip()
                else:
                    step_data['content'] = str(element).strip()

                # Extraire images de l'étape
                if hasattr(element, 'find_all'):
                    img_elements = element.find_all('img')
                    step_data['images'] = [
                        urljoin(self.base_url, img.get('src', ''))
                        for img in img_elements if img.get('src')
                    ]

                steps.append(step_data)
                step_num += 1

        except Exception as e:
            print(f"[ERROR] Erreur extraction étapes: {e}")

        return steps

    def _extract_category(self, soup: BeautifulSoup) -> str:
        """Extrait la catégorie du guide"""
        try:
            cat_elem = soup.find(['span', 'div'], class_=re.compile(r'cat|tag'))
            if cat_elem:
                return cat_elem.get_text().strip()
        except:
            pass
        return "general"

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extrait la description du guide"""
        try:
            desc_elem = soup.find(['meta'], attrs={'name': 'description'})
            if desc_elem:
                return desc_elem.get('content', '')

            desc_elem = soup.find(['p'], class_=re.compile(r'desc|summary'))
            if desc_elem:
                return desc_elem.get_text().strip()
        except:
            pass
        return ""

    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extrait l'auteur du guide"""
        try:
            author_elem = soup.find(['span', 'div'], class_=re.compile(r'author|creator'))
            if author_elem:
                return author_elem.get_text().strip()
        except:
            pass
        return "Auteur inconnu"

    def _extract_votes(self, soup: BeautifulSoup) -> int:
        """Extrait le nombre de votes"""
        try:
            votes_elem = soup.find(text=re.compile(r'\d+\s*(votes?|likes?)'))
            if votes_elem:
                numbers = re.findall(r'\d+', str(votes_elem))
                if numbers:
                    return int(numbers[0])
        except:
            pass
        return 0

    def _extract_difficulty(self, soup: BeautifulSoup) -> str:
        """Extrait le niveau de difficulté"""
        try:
            diff_elem = soup.find(['span', 'div'], class_=re.compile(r'diff|level'))
            if diff_elem:
                text = diff_elem.get_text().lower()
                if any(word in text for word in ['facile', 'easy', 'débutant']):
                    return "facile"
                elif any(word in text for word in ['difficile', 'hard', 'expert']):
                    return "difficile"
                elif any(word in text for word in ['moyen', 'medium', 'intermédiaire']):
                    return "moyen"
        except:
            pass
        return "moyen"

    def _extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """Extrait les tags du guide"""
        tags = []
        try:
            tag_elements = soup.find_all(['span', 'div'], class_=re.compile(r'tag'))
            for elem in tag_elements:
                tag_text = elem.get_text().strip()
                if tag_text and len(tag_text) < 50:  # Éviter contenu trop long
                    tags.append(tag_text)
        except:
            pass
        return tags

class GanymedeKnowledgeIntegration:
    """Intégration des guides Ganymede dans le système de connaissances"""

    def __init__(self, database_path: str = "ganymede_knowledge.db"):
        self.database_path = database_path
        self.scraper = GanymedeGuidesScraper()
        self._init_database()

    def _init_database(self):
        """Initialise la base de données des guides"""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ganymede_guides (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    category TEXT,
                    language TEXT,
                    description TEXT,
                    content TEXT,
                    author TEXT,
                    difficulty_level TEXT,
                    votes INTEGER,
                    url TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS guide_steps (
                    guide_id TEXT,
                    step_number INTEGER,
                    title TEXT,
                    content TEXT,
                    coordinates TEXT,
                    items_needed TEXT,
                    monsters_involved TEXT,
                    FOREIGN KEY (guide_id) REFERENCES ganymede_guides (id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS guide_tags (
                    guide_id TEXT,
                    tag TEXT,
                    FOREIGN KEY (guide_id) REFERENCES ganymede_guides (id)
                )
            """)

    def sync_guides_database(self, categories: List[str] = None) -> Dict[str, Any]:
        """Synchronise la base de données avec les guides Ganymede"""
        print("[SYNC] Synchronisation guides Ganymede...")

        stats = {
            "guides_found": 0,
            "guides_updated": 0,
            "guides_new": 0,
            "errors": 0
        }

        try:
            # Catégories par défaut
            if not categories:
                categories = ["quetes", "donjons", "metiers", "pvp", "general"]

            for category in categories:
                print(f"[CATEGORY] Catégorie: {category}")

                # Récupérer liste guides
                guides_list = self.scraper.get_guides_list(category=category)
                stats["guides_found"] += len(guides_list)

                for guide_info in guides_list:
                    try:
                        # Récupérer contenu détaillé
                        guide = self.scraper.get_guide_content(guide_info['url'])

                        if guide:
                            # Sauvegarder en base
                            self._save_guide_to_database(guide)
                            stats["guides_updated"] += 1

                        # Délai entre requêtes
                        time.sleep(1)

                    except Exception as e:
                        print(f"[ERROR] Erreur guide {guide_info.get('title', 'inconnu')}: {e}")
                        stats["errors"] += 1

                # Délai entre catégories
                time.sleep(2)

            print(f"[SUCCESS] Synchronisation terminée: {stats}")
            return stats

        except Exception as e:
            print(f"[ERROR] Erreur synchronisation: {e}")
            stats["errors"] += 1
            return stats

    def _save_guide_to_database(self, guide: GanymedeGuide):
        """Sauvegarde un guide en base de données"""
        with sqlite3.connect(self.database_path) as conn:
            # Sauvegarder guide principal
            conn.execute("""
                INSERT OR REPLACE INTO ganymede_guides
                (id, title, category, language, description, content, author,
                 difficulty_level, votes, url, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                guide.id, guide.title, guide.category, guide.language,
                guide.description, guide.content, guide.author,
                guide.difficulty_level, guide.votes, guide.url,
                guide.creation_date, guide.last_updated
            ))

            # Supprimer anciennes étapes
            conn.execute("DELETE FROM guide_steps WHERE guide_id = ?", (guide.id,))

            # Sauvegarder étapes
            for step in guide.steps:
                conn.execute("""
                    INSERT INTO guide_steps
                    (guide_id, step_number, title, content, coordinates, items_needed, monsters_involved)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    guide.id, step['step_number'], step['title'], step['content'],
                    json.dumps(step.get('coordinates')),
                    json.dumps(step.get('items_needed', [])),
                    json.dumps(step.get('monsters_involved', []))
                ))

            # Supprimer anciens tags
            conn.execute("DELETE FROM guide_tags WHERE guide_id = ?", (guide.id,))

            # Sauvegarder tags
            for tag in guide.tags:
                conn.execute("INSERT INTO guide_tags (guide_id, tag) VALUES (?, ?)", (guide.id, tag))

    def search_guides(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """Recherche guides par mots-clés"""
        with sqlite3.connect(self.database_path) as conn:
            sql = """
                SELECT * FROM ganymede_guides
                WHERE (title LIKE ? OR description LIKE ? OR content LIKE ?)
            """
            params = [f"%{query}%", f"%{query}%", f"%{query}%"]

            if category:
                sql += " AND category = ?"
                params.append(category)

            sql += " ORDER BY votes DESC, title"

            cursor = conn.execute(sql, params)
            guides = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]

            return guides

    def get_guide_by_topic(self, topic: str) -> Optional[Dict[str, Any]]:
        """Récupère le meilleur guide pour un sujet donné"""
        guides = self.search_guides(topic)

        if guides:
            # Retourner le guide avec le plus de votes
            best_guide = max(guides, key=lambda g: g.get('votes', 0))

            # Récupérer étapes
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM guide_steps WHERE guide_id = ? ORDER BY step_number",
                    (best_guide['id'],)
                )
                steps = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
                best_guide['steps'] = steps

            return best_guide

        return None

# Factory function
def get_ganymede_connector(database_path: str = "ganymede_knowledge.db") -> GanymedeKnowledgeIntegration:
    """Factory pour créer le connecteur Ganymede"""
    return GanymedeKnowledgeIntegration(database_path)

if __name__ == "__main__":
    # Test du connecteur
    print("[TEST] Test Connecteur Guides Ganymede")
    print("=" * 50)

    connector = get_ganymede_connector()

    # Test récupération guides
    print("\n[LIST] Test récupération guides...")
    try:
        stats = connector.sync_guides_database(categories=["quetes"])
        print(f"Résultats: {stats}")

        # Test recherche
        print("\n[SEARCH] Test recherche...")
        guides = connector.search_guides("temple")
        print(f"Guides trouvés pour 'temple': {len(guides)}")

        for guide in guides[:3]:
            print(f"  - {guide['title']} ({guide['votes']} votes)")

    except Exception as e:
        print(f"[ERROR] Erreur test: {e}")

    print("\n[SUCCESS] Test terminé")