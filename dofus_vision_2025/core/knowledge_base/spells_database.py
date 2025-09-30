"""
DOFUS Spells Database - Knowledge Base Integration
Base de donnees complete des sorts pour toutes les classes DOFUS Unity
Approche 100% vision - Reconnaissance automatique sorts via templates
"""

import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DofusClass(Enum):
    """Classes jouables DOFUS"""
    IOPS = "iop"
    CRA = "cra"
    SRAM = "sram"
    XELOR = "xelor"
    ECAFLIP = "ecaflip"
    ENUTROF = "enutrof"
    SADIDA = "sadida"
    OSAMODAS = "osamodas"
    FECA = "feca"
    PANDAWA = "pandawa"
    ROUBLARD = "roublard"
    ZOBAL = "zobal"
    STEAMER = "steamer"
    ELIOTROPE = "eliotrope"
    HUPPERMAGE = "huppermage"
    OUGINAK = "ouginak"
    FORGELANCE = "forgelance"

class SpellType(Enum):
    """Types d'effets des sorts"""
    DAMAGE = "damage"
    HEAL = "heal"
    BUFF = "buff"
    DEBUFF = "debuff"
    INVOCATION = "invocation"
    TELEPORT = "teleport"
    UTILITY = "utility"

class TargetType(Enum):
    """Types de ciblage"""
    CELL = "cell"
    ENTITY = "entity"
    LINE = "line"
    CIRCLE = "circle"
    CROSS = "cross"
    SELF = "self"
    ZONE = "zone"

@dataclass
class SpellEffect:
    """Effet d'un sort"""
    type: SpellType
    min_value: int
    max_value: int
    element: str  # feu, eau, terre, air, neutre
    description: str
    conditions: Optional[str] = None

@dataclass
class SpellRange:
    """Portee et zone d'effet d'un sort"""
    min_range: int
    max_range: int
    area_size: int
    area_shape: str  # circle, cross, line, etc.
    line_of_sight: bool
    diagonal: bool

@dataclass
class SpellCost:
    """Cout d'un sort"""
    ap_cost: int
    mp_cost: int = 0
    special_cost: Optional[str] = None  # ex: "1 invocation"

@dataclass
class DofusSpell:
    """Representation complete d'un sort DOFUS"""
    id: int
    name: str
    class_type: DofusClass
    level: int
    description: str

    # Caracteristiques techniques
    cost: SpellCost
    range_info: SpellRange
    effects: List[SpellEffect]

    # Mecaniques
    cooldown: int
    casts_per_turn: int
    casts_per_target: int
    target_type: TargetType

    # Visuels pour reconnaissance
    icon_template: Optional[str] = None
    animation_keywords: List[str] = None

    # Metadata
    element_primary: str = "neutre"
    is_weapon_spell: bool = False
    unlock_level: int = 1

class DofusSpellsDatabase:
    """
    Base de donnees complete des sorts DOFUS Unity
    Reconnaissance automatique + donnees strategiques
    """

    def __init__(self, db_path: str = "data/dofus_spells.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.spells: Dict[int, DofusSpell] = {}
        self.class_spells: Dict[DofusClass, List[DofusSpell]] = {}

        # Templates pour reconnaissance visuelle
        self.spell_templates = {}

        self._init_database()
        self._load_spell_data()

        logger.info(f"SpellsDatabase initialise: {len(self.spells)} sorts")

    def _init_database(self):
        """Initialise la base de donnees SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spells (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                class_type TEXT NOT NULL,
                level INTEGER,
                description TEXT,
                data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spell_templates (
                spell_id INTEGER,
                template_path TEXT,
                template_hash TEXT,
                FOREIGN KEY (spell_id) REFERENCES spells (id)
            )
        ''')

        conn.commit()
        conn.close()

    def _load_spell_data(self):
        """Charge les donnees de sorts depuis la DB et JSON"""
        # Chargement depuis SQLite
        self._load_from_sqlite()

        # Chargement depuis fichiers JSON (backup/import)
        json_path = self.db_path.parent / "spells_data.json"
        if json_path.exists():
            self._load_from_json(json_path)
        else:
            # Creation des sorts de base si pas de donnees
            self._create_base_spells()

    def _load_from_sqlite(self):
        """Charge les sorts depuis SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM spells")
        rows = cursor.fetchall()

        for row in rows:
            spell_data = json.loads(row[5])  # data JSON column
            spell = self._dict_to_spell(spell_data)
            self.add_spell(spell)

        conn.close()

    def _load_from_json(self, json_path: Path):
        """Charge les sorts depuis fichier JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for spell_data in data.get('spells', []):
                spell = self._dict_to_spell(spell_data)
                self.add_spell(spell)

        except Exception as e:
            logger.error(f"Erreur chargement JSON: {e}")

    def _dict_to_spell(self, data: Dict) -> DofusSpell:
        """Convertit un dictionnaire en objet DofusSpell"""
        # Reconstruction des objets complexes
        cost = SpellCost(**data['cost'])
        range_info = SpellRange(**data['range_info'])
        effects = [SpellEffect(**effect) for effect in data['effects']]

        return DofusSpell(
            id=data['id'],
            name=data['name'],
            class_type=DofusClass(data['class_type']),
            level=data['level'],
            description=data['description'],
            cost=cost,
            range_info=range_info,
            effects=effects,
            cooldown=data['cooldown'],
            casts_per_turn=data['casts_per_turn'],
            casts_per_target=data['casts_per_target'],
            target_type=TargetType(data['target_type']),
            icon_template=data.get('icon_template'),
            animation_keywords=data.get('animation_keywords', []),
            element_primary=data.get('element_primary', 'neutre'),
            is_weapon_spell=data.get('is_weapon_spell', False),
            unlock_level=data.get('unlock_level', 1)
        )

    def _create_base_spells(self):
        """Cree les sorts de base pour test (exemples Iop)"""
        base_spells = [
            # Sorts Iop niveau 1-20
            DofusSpell(
                id=1, name="Compulsion", class_type=DofusClass.IOPS, level=1,
                description="Sort de base Iop - Dommages Feu",
                cost=SpellCost(ap_cost=3),
                range_info=SpellRange(min_range=1, max_range=3, area_size=1,
                                    area_shape="circle", line_of_sight=True, diagonal=True),
                effects=[SpellEffect(SpellType.DAMAGE, 8, 12, "feu", "Dommages Feu")],
                cooldown=0, casts_per_turn=3, casts_per_target=1,
                target_type=TargetType.ENTITY, element_primary="feu", unlock_level=1
            ),
            DofusSpell(
                id=2, name="Epee du Jugement", class_type=DofusClass.IOPS, level=3,
                description="Invoque une epee - Dommages zone",
                cost=SpellCost(ap_cost=4),
                range_info=SpellRange(min_range=2, max_range=4, area_size=3,
                                    area_shape="cross", line_of_sight=True, diagonal=False),
                effects=[SpellEffect(SpellType.DAMAGE, 12, 18, "neutre", "Dommages Neutre zone")],
                cooldown=3, casts_per_turn=1, casts_per_target=1,
                target_type=TargetType.CELL, element_primary="neutre", unlock_level=3
            ),
            # Ajouter plus de sorts...
        ]

        for spell in base_spells:
            self.add_spell(spell)

    def add_spell(self, spell: DofusSpell):
        """Ajoute un sort a la base de donnees"""
        self.spells[spell.id] = spell

        # Index par classe
        if spell.class_type not in self.class_spells:
            self.class_spells[spell.class_type] = []
        self.class_spells[spell.class_type].append(spell)

        # Sauvegarde en SQLite
        self._save_spell_to_db(spell)

    def _save_spell_to_db(self, spell: DofusSpell):
        """Sauvegarde un sort en SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        spell_dict = asdict(spell)
        # Conversion des enums en strings pour JSON
        spell_dict['class_type'] = spell.class_type.value
        spell_dict['target_type'] = spell.target_type.value
        for effect in spell_dict['effects']:
            if hasattr(effect['type'], 'value'):
                effect['type'] = effect['type'].value

        cursor.execute('''
            INSERT OR REPLACE INTO spells
            (id, name, class_type, level, description, data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (spell.id, spell.name, spell.class_type.value,
              spell.level, spell.description, json.dumps(spell_dict)))

        conn.commit()
        conn.close()

    def get_spell(self, spell_id: int) -> Optional[DofusSpell]:
        """Recupere un sort par ID"""
        return self.spells.get(spell_id)

    def get_spell_by_name(self, name: str) -> Optional[DofusSpell]:
        """Recupere un sort par nom"""
        for spell in self.spells.values():
            if spell.name.lower() == name.lower():
                return spell
        return None

    def get_class_spells(self, class_type: DofusClass, max_level: int = 200) -> List[DofusSpell]:
        """Recupere tous les sorts d'une classe jusqu'a un niveau"""
        class_spells = self.class_spells.get(class_type, [])
        return [spell for spell in class_spells if spell.unlock_level <= max_level]

    def get_spells_by_element(self, element: str) -> List[DofusSpell]:
        """Recupere tous les sorts d'un element"""
        return [spell for spell in self.spells.values()
                if spell.element_primary == element]

    def get_damage_spells(self, class_type: Optional[DofusClass] = None) -> List[DofusSpell]:
        """Recupere tous les sorts de degats"""
        damage_spells = []
        for spell in self.spells.values():
            if class_type and spell.class_type != class_type:
                continue
            for effect in spell.effects:
                if effect.type == SpellType.DAMAGE:
                    damage_spells.append(spell)
                    break
        return damage_spells

    def calculate_spell_damage(self, spell: DofusSpell, character_level: int = 200) -> Tuple[int, int]:
        """Calcule les degats min/max d'un sort selon le niveau"""
        total_min = 0
        total_max = 0

        for effect in spell.effects:
            if effect.type == SpellType.DAMAGE:
                # Formule simplifiee (ajuster selon DOFUS reel)
                level_bonus = character_level // 10
                total_min += effect.min_value + level_bonus
                total_max += effect.max_value + level_bonus

        return total_min, total_max

    def find_optimal_spells(self, ap_available: int, distance: int,
                          class_type: DofusClass) -> List[DofusSpell]:
        """Trouve les sorts optimaux selon PA disponibles et distance"""
        available_spells = self.get_class_spells(class_type)
        optimal_spells = []

        for spell in available_spells:
            # Verifications de base
            if (spell.cost.ap_cost <= ap_available and
                spell.range_info.min_range <= distance <= spell.range_info.max_range):
                optimal_spells.append(spell)

        # Tri par efficacite (degats / PA)
        def spell_efficiency(spell):
            min_dmg, max_dmg = self.calculate_spell_damage(spell)
            avg_damage = (min_dmg + max_dmg) / 2
            return avg_damage / spell.cost.ap_cost if spell.cost.ap_cost > 0 else 0

        optimal_spells.sort(key=spell_efficiency, reverse=True)
        return optimal_spells[:5]  # Top 5

    def export_to_json(self, output_path: str):
        """Exporte toute la base vers JSON"""
        export_data = {
            'version': '1.0',
            'total_spells': len(self.spells),
            'spells': []
        }

        for spell in self.spells.values():
            spell_dict = asdict(spell)
            # Conversion enums pour JSON
            spell_dict['class_type'] = spell.class_type.value
            spell_dict['target_type'] = spell.target_type.value
            for effect in spell_dict['effects']:
                effect['type'] = effect['type'].value
            export_data['spells'].append(spell_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Base exportee vers {output_path}")

# Instance globale
_spells_db_instance = None

def get_spells_database() -> DofusSpellsDatabase:
    """Retourne l'instance singleton de la base de sorts"""
    global _spells_db_instance
    if _spells_db_instance is None:
        _spells_db_instance = DofusSpellsDatabase()
    return _spells_db_instance

# Test du module
if __name__ == "__main__":
    db = DofusSpellsDatabase()

    # Test recherche sorts Iop
    iop_spells = db.get_class_spells(DofusClass.IOPS)
    print(f"Sorts Iop disponibles: {len(iop_spells)}")

    # Test sorts optimaux
    optimal = db.find_optimal_spells(ap_available=6, distance=2, class_type=DofusClass.IOPS)
    print(f"Sorts optimaux (6 PA, distance 2): {[s.name for s in optimal]}")

    # Export test
    db.export_to_json("test_spells_export.json")