"""
DOFUS Monsters Database - Knowledge Base Integration
Bestiaire complet avec resistances, comportements et strategies
Approche 100% vision - Reconnaissance automatique monstres via templates
"""

import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MonsterRank(Enum):
    """Rangs des monstres"""
    NORMAL = "normal"
    ELITE = "elite"
    BOSS = "boss"
    ARCHMONSTER = "archmonster"
    DUNGEON_BOSS = "dungeon_boss"

class MonsterElement(Enum):
    """Elements des monstres"""
    NEUTRAL = "neutre"
    FIRE = "feu"
    WATER = "eau"
    EARTH = "terre"
    AIR = "air"

class AIPattern(Enum):
    """Patterns d'IA des monstres"""
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    SUPPORT = "support"
    TACTICAL = "tactical"
    BERSERKER = "berserker"
    COWARD = "coward"

@dataclass
class MonsterResistances:
    """Resistances elementaires d'un monstre"""
    fire_resist: int = 0
    water_resist: int = 0
    earth_resist: int = 0
    air_resist: int = 0
    neutral_resist: int = 0

    # Resistances speciales
    critical_resist: int = 0
    pushback_resist: int = 0

@dataclass
class MonsterStats:
    """Statistiques de base d'un monstre"""
    health: int
    action_points: int
    movement_points: int

    # Stats offensives
    strength: int = 0
    intelligence: int = 0
    chance: int = 0
    agility: int = 0

    # Stats defensives
    vitality: int = 0
    wisdom: int = 0

    # Niveau et grade
    level: int = 1
    grade: int = 1

@dataclass
class MonsterSpell:
    """Sort utilise par un monstre"""
    name: str
    min_damage: int
    max_damage: int
    element: MonsterElement
    range_min: int
    range_max: int
    area_size: int
    ap_cost: int
    cooldown: int = 0
    special_effects: List[str] = None
    probability: float = 1.0  # Probabilite d'utilisation

@dataclass
class MonsterBehavior:
    """Comportement IA d'un monstre"""
    ai_pattern: AIPattern
    aggro_range: int
    preferred_distance: int  # Distance preferee pour combat
    spell_priorities: List[str]  # Noms des sorts par priorite

    # Behaviors specifiques
    uses_summons: bool = False
    heals_allies: bool = False
    teleports: bool = False
    changes_form: bool = False

    # Conditions speciales
    enrage_health_threshold: float = 0.25  # S'enrage sous 25% PV
    retreat_health_threshold: float = 0.1   # Fuit sous 10% PV

@dataclass
class MonsterDrops:
    """Loots d'un monstre"""
    kamas_min: int = 0
    kamas_max: int = 0

    # Items principaux (nom -> probabilite)
    common_drops: Dict[str, float] = None
    rare_drops: Dict[str, float] = None
    epic_drops: Dict[str, float] = None

    # Experience
    base_experience: int = 0

    def __post_init__(self):
        if self.common_drops is None:
            self.common_drops = {}
        if self.rare_drops is None:
            self.rare_drops = {}
        if self.epic_drops is None:
            self.epic_drops = {}

@dataclass
class MonsterLocation:
    """Localisation d'un monstre"""
    map_name: str
    zone_name: str
    coordinates: Tuple[int, int]  # Position sur la carte monde
    spawn_conditions: Optional[str] = None  # Conditions speciales
    spawn_rate: float = 1.0  # Taux d'apparition

@dataclass
class DofusMonster:
    """Representation complete d'un monstre DOFUS"""
    id: int
    name: str
    family: str  # Ex: "Bouftou", "Bwork", etc.
    rank: MonsterRank

    # Caracteristiques
    stats: MonsterStats
    resistances: MonsterResistances
    spells: List[MonsterSpell]
    behavior: MonsterBehavior
    drops: MonsterDrops

    # Localisation
    locations: List[MonsterLocation]

    # Reconnaissance visuelle
    icon_template: Optional[str] = None
    sprite_keywords: List[str] = None

    # Metadata
    description: str = ""
    weaknesses: List[str] = None  # Elements faibles
    immunities: List[str] = None  # Immunites

    def __post_init__(self):
        if self.weaknesses is None:
            self.weaknesses = []
        if self.immunities is None:
            self.immunities = []
        if self.sprite_keywords is None:
            self.sprite_keywords = []

class DofusMonstersDatabase:
    """
    Base de donnees complete des monstres DOFUS Unity
    Reconnaissance automatique + strategies optimales
    """

    def __init__(self, db_path: str = "data/dofus_monsters.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.monsters: Dict[int, DofusMonster] = {}
        self.family_monsters: Dict[str, List[DofusMonster]] = {}
        self.zone_monsters: Dict[str, List[DofusMonster]] = {}

        # Templates pour reconnaissance
        self.monster_templates = {}

        self._init_database()
        self._load_monster_data()

        logger.info(f"MonstersDatabase initialise: {len(self.monsters)} monstres")

    def _init_database(self):
        """Initialise la base de donnees SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monsters (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                family TEXT,
                rank TEXT,
                data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monster_locations (
                monster_id INTEGER,
                map_name TEXT,
                zone_name TEXT,
                coordinates TEXT,
                FOREIGN KEY (monster_id) REFERENCES monsters (id)
            )
        ''')

        conn.commit()
        conn.close()

    def _load_monster_data(self):
        """Charge les donnees depuis DB et JSON"""
        json_path = self.db_path.parent / "monsters_data.json"
        if json_path.exists():
            self._load_from_json(json_path)
        else:
            self._create_base_monsters()

    def _load_from_json(self, json_path: Path):
        """Charge depuis fichier JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for monster_data in data.get('monsters', []):
                monster = self._dict_to_monster(monster_data)
                self.add_monster(monster)

        except Exception as e:
            logger.error(f"Erreur chargement monsters JSON: {e}")

    def _dict_to_monster(self, data: Dict) -> DofusMonster:
        """Convertit dictionnaire en DofusMonster"""
        # Reconstruction des objets complexes
        stats = MonsterStats(**data['stats'])
        resistances = MonsterResistances(**data['resistances'])

        spells = []
        for spell_data in data['spells']:
            spell = MonsterSpell(**spell_data)
            if spell.special_effects is None:
                spell.special_effects = []
            spells.append(spell)

        behavior = MonsterBehavior(**data['behavior'])
        drops = MonsterDrops(**data['drops'])

        locations = []
        for loc_data in data['locations']:
            locations.append(MonsterLocation(**loc_data))

        return DofusMonster(
            id=data['id'],
            name=data['name'],
            family=data['family'],
            rank=MonsterRank(data['rank']),
            stats=stats,
            resistances=resistances,
            spells=spells,
            behavior=behavior,
            drops=drops,
            locations=locations,
            icon_template=data.get('icon_template'),
            sprite_keywords=data.get('sprite_keywords', []),
            description=data.get('description', ''),
            weaknesses=data.get('weaknesses', []),
            immunities=data.get('immunities', [])
        )

    def _create_base_monsters(self):
        """Cree les monstres de base pour test"""
        base_monsters = [
            # Bouftou Basique
            DofusMonster(
                id=1, name="Bouftou", family="Bouftou", rank=MonsterRank.NORMAL,
                stats=MonsterStats(health=45, action_points=3, movement_points=3,
                                 strength=25, level=5, grade=1),
                resistances=MonsterResistances(fire_resist=-10, water_resist=5),
                spells=[
                    MonsterSpell("Charge", 8, 12, MonsterElement.NEUTRAL, 1, 1, 1, 3,
                               special_effects=["pushback"], probability=0.8),
                    MonsterSpell("Coup de Corne", 6, 10, MonsterElement.EARTH, 1, 2, 1, 2,
                               probability=0.6)
                ],
                behavior=MonsterBehavior(
                    ai_pattern=AIPattern.AGGRESSIVE, aggro_range=6, preferred_distance=1,
                    spell_priorities=["Charge", "Coup de Corne"]
                ),
                drops=MonsterDrops(
                    kamas_min=5, kamas_max=15, base_experience=25,
                    common_drops={"Poil de Bouftou": 0.8, "Langue de Bouftou": 0.3}
                ),
                locations=[
                    MonsterLocation("Plaine de Cania", "Zone Debutant", (-25, -36))
                ],
                description="Monstre de base, tres agressif",
                weaknesses=["feu"], sprite_keywords=["bouftou", "bleu", "cornes"]
            ),

            # Crabe Basique
            DofusMonster(
                id=2, name="Crabe", family="Crabe", rank=MonsterRank.NORMAL,
                stats=MonsterStats(health=38, action_points=3, movement_points=2,
                                 agility=20, level=3, grade=1),
                resistances=MonsterResistances(water_resist=15, fire_resist=-15),
                spells=[
                    MonsterSpell("Pince", 5, 9, MonsterElement.WATER, 1, 1, 1, 2,
                               probability=0.9),
                    MonsterSpell("Bulle", 3, 6, MonsterElement.WATER, 2, 4, 2, 3,
                               special_effects=["heal"], probability=0.4)
                ],
                behavior=MonsterBehavior(
                    ai_pattern=AIPattern.DEFENSIVE, aggro_range=4, preferred_distance=2,
                    spell_priorities=["Pince", "Bulle"], heals_allies=True
                ),
                drops=MonsterDrops(
                    kamas_min=3, kamas_max=10, base_experience=18,
                    common_drops={"Carapace de Crabe": 0.7, "Chair de Crabe": 0.5}
                ),
                locations=[
                    MonsterLocation("Plage de Corail", "Zone Cote", (-27, -38))
                ],
                description="Crabe defensif avec capacites de soin",
                weaknesses=["feu"], immunities=["noyade"],
                sprite_keywords=["crabe", "rouge", "pinces", "carapace"]
            )
        ]

        for monster in base_monsters:
            self.add_monster(monster)

    def add_monster(self, monster: DofusMonster):
        """Ajoute un monstre a la base"""
        self.monsters[monster.id] = monster

        # Index par famille
        if monster.family not in self.family_monsters:
            self.family_monsters[monster.family] = []
        self.family_monsters[monster.family].append(monster)

        # Index par zone
        for location in monster.locations:
            if location.zone_name not in self.zone_monsters:
                self.zone_monsters[location.zone_name] = []
            self.zone_monsters[location.zone_name].append(monster)

        self._save_monster_to_db(monster)

    def _save_monster_to_db(self, monster: DofusMonster):
        """Sauvegarde en SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        monster_dict = asdict(monster)
        # Conversion enums
        monster_dict['rank'] = monster.rank.value
        for spell in monster_dict['spells']:
            spell['element'] = spell['element'].value
        monster_dict['behavior']['ai_pattern'] = monster.behavior.ai_pattern.value

        cursor.execute('''
            INSERT OR REPLACE INTO monsters
            (id, name, family, rank, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (monster.id, monster.name, monster.family,
              monster.rank.value, json.dumps(monster_dict)))

        conn.commit()
        conn.close()

    def get_monster(self, monster_id: int) -> Optional[DofusMonster]:
        """Recupere un monstre par ID"""
        return self.monsters.get(monster_id)

    def get_monster_by_name(self, name: str) -> Optional[DofusMonster]:
        """Recupere un monstre par nom"""
        for monster in self.monsters.values():
            if monster.name.lower() == name.lower():
                return monster
        return None

    def get_monsters_by_family(self, family: str) -> List[DofusMonster]:
        """Recupere tous les monstres d'une famille"""
        return self.family_monsters.get(family, [])

    def get_monsters_by_zone(self, zone: str) -> List[DofusMonster]:
        """Recupere tous les monstres d'une zone"""
        return self.zone_monsters.get(zone, [])

    def get_monsters_by_level_range(self, min_level: int, max_level: int) -> List[DofusMonster]:
        """Recupere monstres par niveau"""
        return [monster for monster in self.monsters.values()
                if min_level <= monster.stats.level <= max_level]

    def calculate_weakness_damage(self, monster: DofusMonster, element: str, base_damage: int) -> int:
        """Calcule degats avec resistances"""
        resistance = 0

        if element == "feu":
            resistance = monster.resistances.fire_resist
        elif element == "eau":
            resistance = monster.resistances.water_resist
        elif element == "terre":
            resistance = monster.resistances.earth_resist
        elif element == "air":
            resistance = monster.resistances.air_resist
        else:
            resistance = monster.resistances.neutral_resist

        # Formule simplifiee DOFUS
        if resistance > 0:
            final_damage = base_damage * (100 - resistance) // 100
        else:
            final_damage = base_damage * (100 + abs(resistance)) // 100

        return max(1, final_damage)  # Minimum 1 degat

    def find_optimal_targets(self, player_level: int, zone: Optional[str] = None) -> List[DofusMonster]:
        """Trouve les monstres optimaux pour un joueur"""
        optimal_monsters = []

        # Niveau recommande: +/- 10 niveaux
        level_min = max(1, player_level - 10)
        level_max = player_level + 15

        candidates = self.get_monsters_by_level_range(level_min, level_max)

        if zone:
            candidates = [m for m in candidates if any(loc.zone_name == zone for loc in m.locations)]

        # Tri par rapport XP/difficulte
        def monster_value(monster):
            xp_ratio = monster.drops.base_experience / monster.stats.level
            hp_difficulty = monster.stats.health / (player_level * 10)
            return xp_ratio / max(0.1, hp_difficulty)

        candidates.sort(key=monster_value, reverse=True)
        return candidates[:10]

    def get_counter_strategy(self, monster: DofusMonster) -> Dict[str, any]:
        """Retourne une strategie optimale contre un monstre"""
        strategy = {
            "preferred_elements": [],
            "avoid_elements": [],
            "optimal_distance": monster.behavior.preferred_distance + 1,
            "priority_target": False,
            "special_tactics": []
        }

        # Elements efficaces (resistances negatives)
        resistances = {
            "feu": monster.resistances.fire_resist,
            "eau": monster.resistances.water_resist,
            "terre": monster.resistances.earth_resist,
            "air": monster.resistances.air_resist
        }

        for element, resist in resistances.items():
            if resist < 0:
                strategy["preferred_elements"].append(element)
            elif resist > 50:
                strategy["avoid_elements"].append(element)

        # Tactiques speciales
        if monster.behavior.heals_allies:
            strategy["priority_target"] = True
            strategy["special_tactics"].append("Eliminer en priorite (soigneur)")

        if monster.behavior.uses_summons:
            strategy["special_tactics"].append("Attention aux invocations")

        if monster.behavior.teleports:
            strategy["special_tactics"].append("Garde distance - peut teleporter")

        return strategy

    def export_to_json(self, output_path: str):
        """Exporte la base vers JSON"""
        export_data = {
            'version': '1.0',
            'total_monsters': len(self.monsters),
            'monsters': []
        }

        for monster in self.monsters.values():
            monster_dict = asdict(monster)
            # Conversion enums
            monster_dict['rank'] = monster.rank.value
            for spell in monster_dict['spells']:
                spell['element'] = spell['element'].value
            monster_dict['behavior']['ai_pattern'] = monster.behavior.ai_pattern.value
            export_data['monsters'].append(monster_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Base monstres exportee vers {output_path}")

# Instance globale
_monsters_db_instance = None

def get_monsters_database() -> DofusMonstersDatabase:
    """Retourne l'instance singleton"""
    global _monsters_db_instance
    if _monsters_db_instance is None:
        _monsters_db_instance = DofusMonstersDatabase()
    return _monsters_db_instance

# Test du module
if __name__ == "__main__":
    db = DofusMonstersDatabase()

    # Test recherche
    bouftou = db.get_monster_by_name("Bouftou")
    if bouftou:
        print(f"Bouftou trouve: Niveau {bouftou.stats.level}, PV {bouftou.stats.health}")

        # Test strategie
        strategy = db.get_counter_strategy(bouftou)
        print(f"Strategie: Elements efficaces {strategy['preferred_elements']}")

        # Test degats
        fire_damage = db.calculate_weakness_damage(bouftou, "feu", 100)
        print(f"Degats feu (100 base): {fire_damage}")

    # Test monstres optimaux
    optimal = db.find_optimal_targets(player_level=10)
    print(f"Monstres optimaux niveau 10: {[m.name for m in optimal]}")

    # Export
    db.export_to_json("test_monsters_export.json")