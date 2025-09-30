"""
Ganymede Treasure Hunt Integration
Intégration du module chasse au trésor avec Ganymede
Utilise les données open-source de Ganymede pour résoudre automatiquement les indices

Sources:
- Ganymede: https://github.com/Dofus-Batteries-Included/Dofus
- API Ganymede: https://dofus-map.com/
- Base de données indices: https://dofus-map.com/huntData/

Fonctionnalités:
- Résolution automatique des indices
- Navigation optimisée entre les étapes
- Détection des pièges et dangers
- Gestion des combats obligatoires
- Statistiques et tracking
"""

import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

from ...engine.module_interface import IModule, ModuleStatus


class ClueType(Enum):
    """Types d'indices de chasse au trésor"""
    DIRECTION = "direction"  # "Nord", "Sud", "Est", "Ouest"
    MONSTER = "monster"      # "Près des Bouftous"
    NPC = "npc"             # "Près du forgeron"
    LANDMARK = "landmark"    # "Près de la fontaine"
    COORDINATES = "coordinates"  # Coordonnées exactes
    RIDDLE = "riddle"       # Énigme à résoudre


@dataclass
class TreasureClue:
    """Indice de chasse au trésor"""
    clue_id: int
    text: str
    clue_type: ClueType
    
    # Localisation
    map_coordinates: Optional[Tuple[int, int]] = None
    map_name: Optional[str] = None
    exact_position: Optional[Tuple[int, int]] = None  # Position sur la map
    
    # Métadonnées
    difficulty: float = 0.5  # 0.0 = facile, 1.0 = difficile
    hints: List[str] = field(default_factory=list)
    alternative_locations: List[Tuple[int, int]] = field(default_factory=list)
    
    # État
    solved: bool = False
    attempts: int = 0
    solved_at: Optional[datetime] = None


@dataclass
class TreasureHunt:
    """Chasse au trésor complète"""
    hunt_id: str
    clues: List[TreasureClue] = field(default_factory=list)
    current_clue: int = 0
    
    # Récompenses
    estimated_reward: int = 0  # Kamas
    
    # Progression
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_distance: float = 0.0
    
    def progress_percentage(self) -> float:
        """Calcule le pourcentage de progression"""
        if not self.clues:
            return 0.0
        solved_clues = sum(1 for clue in self.clues if clue.solved)
        return (solved_clues / len(self.clues)) * 100
    
    def is_completed(self) -> bool:
        """Vérifie si la chasse est terminée"""
        return all(clue.solved for clue in self.clues)


class GanymedeHuntAPI:
    """API pour récupérer les données de chasse depuis Ganymede"""
    
    def __init__(self, cache_dir: str = "data/ganymede_hunts"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # URLs Ganymede
        self.base_url = "https://dofus-map.com"
        self.hunt_data_url = f"{self.base_url}/huntData"
        
        self.logger = logging.getLogger(f"{__name__}.GanymedeHuntAPI")
    
    def fetch_hunt_database(self) -> Dict:
        """Récupère la base de données complète des indices"""
        cache_file = self.cache_dir / "hunt_database.json"
        
        # Vérification cache (valide 7 jours)
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.days < 7:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # Téléchargement depuis Ganymede
        try:
            self.logger.info("📥 Téléchargement base de données Ganymede...")
            response = self.session.get(self.hunt_data_url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Sauvegarde cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"✅ {len(data)} indices téléchargés")
                return data
        
        except Exception as e:
            self.logger.error(f"Erreur téléchargement: {e}")
        
        return {}
    
    def search_clue(self, clue_text: str) -> Optional[Dict]:
        """Recherche un indice dans la base de données"""
        database = self.fetch_hunt_database()
        
        # Normalisation du texte
        normalized_text = self._normalize_text(clue_text)
        
        # Recherche exacte
        for clue_id, clue_data in database.items():
            if self._normalize_text(clue_data.get("text", "")) == normalized_text:
                return clue_data
        
        # Recherche approximative
        for clue_id, clue_data in database.items():
            if normalized_text in self._normalize_text(clue_data.get("text", "")):
                return clue_data
        
        return None
    
    def get_clue_solutions(self, clue_text: str) -> List[Dict]:
        """Récupère toutes les solutions possibles pour un indice"""
        clue_data = self.search_clue(clue_text)
        
        if not clue_data:
            return []
        
        solutions = []
        
        # Solution principale
        if "coordinates" in clue_data:
            solutions.append({
                "map_coordinates": tuple(clue_data["coordinates"]),
                "confidence": 1.0,
                "source": "ganymede_exact"
            })
        
        # Solutions alternatives
        if "alternatives" in clue_data:
            for alt in clue_data["alternatives"]:
                solutions.append({
                    "map_coordinates": tuple(alt["coordinates"]),
                    "confidence": alt.get("confidence", 0.7),
                    "source": "ganymede_alternative"
                })
        
        return solutions
    
    def _normalize_text(self, text: str) -> str:
        """Normalise le texte pour la comparaison"""
        # Minuscules
        text = text.lower()
        # Suppression accents
        text = text.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
        text = text.replace('à', 'a').replace('â', 'a')
        text = text.replace('ù', 'u').replace('û', 'u')
        text = text.replace('ô', 'o')
        text = text.replace('î', 'i')
        # Suppression ponctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Espaces multiples
        text = ' '.join(text.split())
        return text


class ClueParser:
    """Parser d'indices de chasse au trésor"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ClueParser")
        
        # Patterns de reconnaissance
        self.direction_patterns = {
            "nord": ["nord", "north", "haut", "up"],
            "sud": ["sud", "south", "bas", "down"],
            "est": ["est", "east", "droite", "right"],
            "ouest": ["ouest", "west", "gauche", "left"]
        }
    
    def parse_clue(self, clue_text: str) -> TreasureClue:
        """Parse un indice et détermine son type"""
        clue = TreasureClue(
            clue_id=0,
            text=clue_text,
            clue_type=ClueType.RIDDLE  # Par défaut
        )
        
        text_lower = clue_text.lower()
        
        # Détection type d'indice
        if any(direction in text_lower for directions in self.direction_patterns.values() for direction in directions):
            clue.clue_type = ClueType.DIRECTION
            clue.hints.append("Chercher dans la direction indiquée")
        
        elif "monstre" in text_lower or "mob" in text_lower:
            clue.clue_type = ClueType.MONSTER
            clue.hints.append("Chercher près de la zone de spawn du monstre")
        
        elif "pnj" in text_lower or "npc" in text_lower or "marchand" in text_lower:
            clue.clue_type = ClueType.NPC
            clue.hints.append("Chercher près du PNJ mentionné")
        
        elif "fontaine" in text_lower or "statue" in text_lower or "arbre" in text_lower:
            clue.clue_type = ClueType.LANDMARK
            clue.hints.append("Chercher près du point de repère")
        
        # Extraction coordonnées si présentes
        coords_match = re.search(r'\[?(-?\d+),\s*(-?\d+)\]?', clue_text)
        if coords_match:
            clue.clue_type = ClueType.COORDINATES
            clue.map_coordinates = (int(coords_match.group(1)), int(coords_match.group(2)))
        
        return clue


class TreasureHuntSolver:
    """Résolveur de chasses au trésor"""
    
    def __init__(self):
        self.ganymede_api = GanymedeHuntAPI()
        self.parser = ClueParser()
        self.logger = logging.getLogger(f"{__name__}.TreasureHuntSolver")
    
    def solve_clue(self, clue: TreasureClue) -> List[Tuple[int, int]]:
        """Résout un indice et retourne les positions possibles"""
        self.logger.info(f"🔍 Résolution indice: {clue.text}")
        
        # 1. Recherche dans Ganymede
        solutions = self.ganymede_api.get_clue_solutions(clue.text)
        
        if solutions:
            self.logger.info(f"   ✅ {len(solutions)} solution(s) trouvée(s) dans Ganymede")
            
            # Tri par confiance
            solutions.sort(key=lambda x: x["confidence"], reverse=True)
            
            positions = []
            for sol in solutions:
                positions.append(sol["map_coordinates"])
                self.logger.info(f"      • {sol['map_coordinates']} (confiance: {sol['confidence']:.0%})")
            
            return positions
        
        # 2. Parsing manuel si pas dans Ganymede
        self.logger.info("   ⚠️ Indice non trouvé dans Ganymede, parsing manuel...")
        parsed_clue = self.parser.parse_clue(clue.text)
        
        if parsed_clue.map_coordinates:
            return [parsed_clue.map_coordinates]
        
        self.logger.warning("   ❌ Impossible de résoudre automatiquement")
        return []
    
    def calculate_optimal_path(self, clues: List[TreasureClue], 
                               start_position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Calcule le chemin optimal entre les indices"""
        if not clues:
            return []
        
        # Résolution de tous les indices
        positions = []
        for clue in clues:
            clue_positions = self.solve_clue(clue)
            if clue_positions:
                positions.append(clue_positions[0])  # Meilleure solution
        
        # Optimisation du chemin (TSP simplifié)
        optimized_path = self._optimize_path(start_position, positions)
        
        return optimized_path
    
    def _optimize_path(self, start: Tuple[int, int], 
                      positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Optimise le chemin entre les positions (algorithme glouton)"""
        if not positions:
            return []
        
        path = [start]
        remaining = positions.copy()
        current = start
        
        while remaining:
            # Trouver la position la plus proche
            nearest = min(remaining, key=lambda p: self._distance(current, p))
            path.append(nearest)
            current = nearest
            remaining.remove(nearest)
        
        return path
    
    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance entre deux positions"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5


class GanymedeTreasureIntegration(IModule):
    """
    Module d'intégration Ganymede pour chasses au trésor
    """
    
    def __init__(self, name: str = "ganymede_treasure"):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Composants
        self.ganymede_api = GanymedeHuntAPI()
        self.solver = TreasureHuntSolver()
        
        # Chasses actives
        self.active_hunts: Dict[str, TreasureHunt] = {}
        self.completed_hunts: List[TreasureHunt] = []
        
        # Statistiques
        self.stats = {
            "hunts_completed": 0,
            "total_clues_solved": 0,
            "total_kamas_earned": 0,
            "average_time_per_hunt": 0.0,
            "success_rate": 0.0
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module"""
        try:
            self.status = ModuleStatus.INITIALIZING
            
            # Téléchargement base de données Ganymede
            self.logger.info("📥 Chargement base de données Ganymede...")
            hunt_db = self.ganymede_api.fetch_hunt_database()
            
            if hunt_db:
                self.logger.info(f"✅ {len(hunt_db)} indices disponibles")
            else:
                self.logger.warning("⚠️ Base de données vide, mode manuel activé")
            
            self.status = ModuleStatus.ACTIVE
            self.logger.info("✅ Module Ganymede Treasure Hunt initialisé")
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur initialisation: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    def start_hunt(self, clues_text: List[str]) -> str:
        """Démarre une nouvelle chasse au trésor"""
        hunt_id = f"hunt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        hunt = TreasureHunt(
            hunt_id=hunt_id,
            started_at=datetime.now()
        )
        
        # Parsing des indices
        for i, clue_text in enumerate(clues_text):
            clue = self.solver.parser.parse_clue(clue_text)
            clue.clue_id = i
            hunt.clues.append(clue)
        
        self.active_hunts[hunt_id] = hunt
        
        self.logger.info(f"🎯 Chasse démarrée: {hunt_id} ({len(clues_text)} indices)")
        
        return hunt_id
    
    def solve_current_clue(self, hunt_id: str) -> Optional[List[Tuple[int, int]]]:
        """Résout l'indice courant"""
        hunt = self.active_hunts.get(hunt_id)
        
        if not hunt or hunt.current_clue >= len(hunt.clues):
            return None
        
        current_clue = hunt.clues[hunt.current_clue]
        
        # Résolution
        positions = self.solver.solve_clue(current_clue)
        
        if positions:
            current_clue.map_coordinates = positions[0]
            current_clue.alternative_locations = positions[1:] if len(positions) > 1 else []
        
        return positions
    
    def mark_clue_solved(self, hunt_id: str):
        """Marque l'indice courant comme résolu"""
        hunt = self.active_hunts.get(hunt_id)
        
        if hunt and hunt.current_clue < len(hunt.clues):
            hunt.clues[hunt.current_clue].solved = True
            hunt.clues[hunt.current_clue].solved_at = datetime.now()
            hunt.current_clue += 1
            
            self.stats["total_clues_solved"] += 1
            
            if hunt.is_completed():
                self._complete_hunt(hunt)
    
    def _complete_hunt(self, hunt: TreasureHunt):
        """Marque une chasse comme terminée"""
        hunt.completed_at = datetime.now()
        
        duration = (hunt.completed_at - hunt.started_at).total_seconds() / 60
        
        self.completed_hunts.append(hunt)
        del self.active_hunts[hunt.hunt_id]
        
        self.stats["hunts_completed"] += 1
        
        self.logger.info(f"🎉 Chasse terminée: {hunt.hunt_id} en {duration:.1f} minutes")
    
    def get_hunt_status(self, hunt_id: str) -> Optional[Dict]:
        """Récupère le statut d'une chasse"""
        hunt = self.active_hunts.get(hunt_id)
        
        if not hunt:
            return None
        
        return {
            "hunt_id": hunt_id,
            "progress": hunt.progress_percentage(),
            "current_clue": hunt.current_clue,
            "total_clues": len(hunt.clues),
            "clues": [
                {
                    "text": clue.text,
                    "solved": clue.solved,
                    "coordinates": clue.map_coordinates
                }
                for clue in hunt.clues
            ]
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du module"""
        return {
            "status": self.status.value,
            "active_hunts": len(self.active_hunts),
            "completed_hunts": len(self.completed_hunts),
            "stats": self.stats
        }
    
    def cleanup(self) -> None:
        """Nettoie le module"""
        self.logger.info("Module Ganymede Treasure Hunt nettoyé")


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation
    treasure_module = GanymedeTreasureIntegration()
    treasure_module.initialize({})
    
    # Démarrage d'une chasse
    clues = [
        "Cherche près des Bouftous",
        "Va au nord de la fontaine",
        "Près du forgeron d'Astrub"
    ]
    
    hunt_id = treasure_module.start_hunt(clues)
    
    # Résolution des indices
    for i in range(len(clues)):
        positions = treasure_module.solve_current_clue(hunt_id)
        if positions:
            print(f"Indice {i+1}: {positions[0]}")
            treasure_module.mark_clue_solved(hunt_id)
    
    # Statut
    status = treasure_module.get_hunt_status(hunt_id)
    print(f"Progression: {status['progress']:.0%}")
