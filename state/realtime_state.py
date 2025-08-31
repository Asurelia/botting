"""
Système d'état temps réel complet du jeu DOFUS
Maintient une représentation fidèle et à jour de tous les éléments du jeu
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import copy


class MapType(Enum):
    """Types de cartes dans DOFUS"""
    OUTDOOR = "outdoor"
    INDOOR = "indoor"
    DUNGEON = "dungeon"
    PVP = "pvp"
    GUILD_AREA = "guild_area"


class CombatState(Enum):
    """États du combat"""
    NO_COMBAT = "no_combat"
    STARTING = "starting"
    IN_COMBAT = "in_combat"
    PLACEMENT_PHASE = "placement_phase"
    MY_TURN = "my_turn"
    ENEMY_TURN = "enemy_turn"
    ENDING = "ending"


class CharacterClass(Enum):
    """Classes de personnages DOFUS"""
    IOP = "iop"
    CRA = "cra"
    ENIRIPSA = "eniripsa"
    ENUTROF = "enutrof"
    SRAM = "sram"
    XELOR = "xelor"
    ECAFLIP = "ecaflip"
    SACRIEUR = "sacrieur"
    SADIDA = "sadida"
    OSMODAS = "osmodas"
    PANDAWA = "pandawa"
    FECA = "feca"


@dataclass
class GridPosition:
    """Position sur la grille isométrique DOFUS"""
    x: int
    y: int
    cell_id: int = -1  # ID de cellule DOFUS
    
    def distance_to(self, other: 'GridPosition') -> int:
        """Calcule la distance Manhattan entre deux positions"""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def is_adjacent(self, other: 'GridPosition') -> bool:
        """Vérifie si deux positions sont adjacentes"""
        return self.distance_to(other) == 1


@dataclass
class Spell:
    """Informations sur un sort"""
    spell_id: int
    name: str
    level: int
    pa_cost: int
    range_min: int
    range_max: int
    is_modifiable_range: bool = False
    line_of_sight: bool = True
    cooldown_turns: int = 0
    current_cooldown: int = 0
    effects: List[str] = field(default_factory=list)
    damage_range: Tuple[int, int] = (0, 0)
    ap_cost: int = 0  # Pour compatibilité AP/PA
    
    def __post_init__(self):
        if self.ap_cost == 0:
            self.ap_cost = self.pa_cost


@dataclass
class Equipment:
    """Équipement du personnage"""
    helmet: Optional[str] = None
    amulet: Optional[str] = None
    cape: Optional[str] = None
    weapon: Optional[str] = None
    shield: Optional[str] = None
    ring1: Optional[str] = None
    ring2: Optional[str] = None
    belt: Optional[str] = None
    boots: Optional[str] = None
    pet: Optional[str] = None
    
    def get_all_items(self) -> List[str]:
        """Retourne tous les items équipés"""
        items = []
        for slot, item in self.__dict__.items():
            if item is not None:
                items.append(item)
        return items


@dataclass
class Character:
    """État complet du personnage"""
    name: str = ""
    level: int = 1
    class_name: CharacterClass = CharacterClass.IOP
    
    # Points de vie
    current_hp: int = 100
    max_hp: int = 100
    
    # Points d'action/mouvement
    current_pa: int = 6
    max_pa: int = 6
    current_pm: int = 3
    max_pm: int = 3
    
    # Position
    position: Optional[GridPosition] = None
    map_coordinates: Tuple[int, int] = (0, 0)
    
    # Ressources
    kamas: int = 0
    energy: int = 10000
    energy_max: int = 10000
    
    # Inventaire
    pods_used: int = 0
    pods_max: int = 1000
    
    # Sorts disponibles
    spells: Dict[int, Spell] = field(default_factory=dict)
    
    # Équipement
    equipment: Equipment = field(default_factory=Equipment)
    
    # Statistiques
    vitality: int = 0
    wisdom: int = 0
    strength: int = 0
    intelligence: int = 0
    chance: int = 0
    agility: int = 0
    
    # États temporaires
    is_dead: bool = False
    is_sitting: bool = False
    is_moving: bool = False
    
    def hp_percentage(self) -> float:
        """Pourcentage de HP actuel"""
        if self.max_hp == 0:
            return 0.0
        return (self.current_hp / self.max_hp) * 100
    
    def energy_percentage(self) -> float:
        """Pourcentage d'énergie actuel"""
        if self.energy_max == 0:
            return 0.0
        return (self.energy / self.energy_max) * 100
    
    def pods_percentage(self) -> float:
        """Pourcentage de pods utilisé"""
        if self.pods_max == 0:
            return 0.0
        return (self.pods_used / self.pods_max) * 100
    
    def can_cast_spell(self, spell_id: int) -> bool:
        """Vérifie si un sort peut être lancé"""
        if spell_id not in self.spells:
            return False
        
        spell = self.spells[spell_id]
        return (self.current_pa >= spell.pa_cost and 
                spell.current_cooldown == 0)


@dataclass
class CombatEntity:
    """Entité en combat (joueur, monstre, invocation)"""
    entity_id: int
    name: str
    level: int
    current_hp: int
    max_hp: int
    position: Optional[GridPosition] = None
    is_ally: bool = False
    is_player: bool = False
    is_monster: bool = False
    is_summon: bool = False
    
    # États temporaires
    states: Set[str] = field(default_factory=set)
    
    def hp_percentage(self) -> float:
        """Pourcentage de HP"""
        if self.max_hp == 0:
            return 0.0
        return (self.current_hp / self.max_hp) * 100
    
    def is_dead(self) -> bool:
        """Vérifie si l'entité est morte"""
        return self.current_hp <= 0
    
    def has_state(self, state: str) -> bool:
        """Vérifie si l'entité a un état spécifique"""
        return state in self.states


@dataclass
class Resource:
    """Ressource sur la carte"""
    resource_id: int
    name: str
    position: GridPosition
    profession_required: str  # "farmer", "lumberjack", "miner", etc.
    level_required: int
    estimated_respawn_time: Optional[datetime] = None
    
    def is_harvestable_by_level(self, player_level: int) -> bool:
        """Vérifie si la ressource peut être récoltée"""
        return player_level >= self.level_required


@dataclass
class NPC:
    """PNJ sur la carte"""
    npc_id: int
    name: str
    position: GridPosition
    npc_type: str  # "merchant", "quest", "bank", etc.
    is_interactable: bool = True


@dataclass
class Player:
    """Autre joueur visible"""
    player_id: int
    name: str
    level: int
    position: GridPosition
    guild: Optional[str] = None
    is_in_combat: bool = False


@dataclass
class Monster:
    """Monstre sur la carte"""
    monster_id: int
    name: str
    level: int
    position: GridPosition
    is_aggressive: bool = False
    group_size: int = 1


@dataclass
class MapState:
    """État de la carte actuelle"""
    coordinates: Tuple[int, int] = (0, 0)
    map_type: MapType = MapType.OUTDOOR
    area_name: str = ""
    subarea_name: str = ""
    
    # Entités visibles
    resources: List[Resource] = field(default_factory=list)
    npcs: List[NPC] = field(default_factory=list)
    players: List[Player] = field(default_factory=list)
    monsters: List[Monster] = field(default_factory=list)
    
    # Éléments interactifs
    zaap_present: bool = False
    bank_present: bool = False
    hdv_present: bool = False
    
    def get_resources_by_profession(self, profession: str) -> List[Resource]:
        """Retourne les ressources d'une profession donnée"""
        return [r for r in self.resources if r.profession_required == profession]
    
    def get_harvestable_resources(self, player_level_dict: Dict[str, int]) -> List[Resource]:
        """Retourne les ressources récoltables selon les niveaux du joueur"""
        harvestable = []
        for resource in self.resources:
            player_level = player_level_dict.get(resource.profession_required, 0)
            if resource.is_harvestable_by_level(player_level):
                harvestable.append(resource)
        return harvestable


@dataclass
class CombatInfo:
    """Informations sur le combat en cours"""
    state: CombatState = CombatState.NO_COMBAT
    turn_number: int = 0
    current_fighter_id: Optional[int] = None
    my_turn: bool = False
    time_remaining: float = 30.0
    max_turn_time: float = 30.0
    
    # Participants au combat
    allies: List[CombatEntity] = field(default_factory=list)
    enemies: List[CombatEntity] = field(default_factory=list)
    
    # Ordre des tours
    turn_order: List[int] = field(default_factory=list)  # IDs des entités
    
    # Grille de combat
    blocked_cells: Set[int] = field(default_factory=set)
    
    def get_all_entities(self) -> List[CombatEntity]:
        """Retourne toutes les entités du combat"""
        return self.allies + self.enemies
    
    def get_entity_by_id(self, entity_id: int) -> Optional[CombatEntity]:
        """Trouve une entité par son ID"""
        for entity in self.get_all_entities():
            if entity.entity_id == entity_id:
                return entity
        return None
    
    def is_my_turn(self) -> bool:
        """Vérifie si c'est le tour du joueur"""
        return self.my_turn and self.state == CombatState.MY_TURN


@dataclass
class InventoryItem:
    """Item dans l'inventaire"""
    item_id: int
    name: str
    quantity: int
    category: str
    estimated_value: int = 0
    is_consumable: bool = False
    is_equipment: bool = False


class GameState:
    """
    État temps réel complet du jeu DOFUS
    Thread-safe et optimisé pour les mises à jour fréquentes
    """
    
    def __init__(self):
        # Synchronisation thread-safe
        self._lock = threading.RLock()
        self._last_update = datetime.now()
        
        # Configuration du logging
        self.logger = logging.getLogger(f"{__name__}.GameState")
        
        # État du personnage
        self.character = Character()
        
        # État de la carte
        self.map_state = MapState()
        
        # État du combat
        self.combat = CombatInfo()
        
        # Inventaire
        self.inventory: Dict[int, InventoryItem] = {}
        
        # Statistiques de session
        self.session_stats = {
            "session_start": datetime.now(),
            "total_updates": 0,
            "combat_count": 0,
            "maps_visited": set(),
            "resources_harvested": 0,
            "xp_gained": 0,
            "kamas_gained": 0
        }
        
        # Cache pour optimisation
        self._state_cache = {}
        self._cache_ttl = {}
        self._cache_duration = 0.1  # 100ms de cache
        
        # Historique des changements pour debug
        self._change_history = []
        self._max_history = 1000
        
    def update_from_screen(self, screen_data: Dict[str, Any]) -> bool:
        """
        Met à jour l'état complet depuis l'analyse d'écran
        
        Args:
            screen_data: Données extraites de l'écran par le module vision
            
        Returns:
            bool: True si la mise à jour a réussi
        """
        try:
            with self._lock:
                old_state = self._create_state_snapshot()
                
                # Mise à jour des différents composants
                self._update_character(screen_data.get("character", {}))
                self._update_map_state(screen_data.get("map", {}))
                self._update_combat_state(screen_data.get("combat", {}))
                self._update_inventory(screen_data.get("inventory", {}))
                
                # Mise à jour des métadonnées
                self._last_update = datetime.now()
                self.session_stats["total_updates"] += 1
                
                # Enregistrement des changements
                new_state = self._create_state_snapshot()
                changes = self._calculate_changes(old_state, new_state)
                if changes:
                    self._record_changes(changes)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour de l'état: {e}")
            return False
    
    def _update_character(self, char_data: Dict[str, Any]) -> None:
        """Met à jour les informations du personnage"""
        if not char_data:
            return
        
        # HP/PA/PM
        self.character.current_hp = char_data.get("current_hp", self.character.current_hp)
        self.character.max_hp = char_data.get("max_hp", self.character.max_hp)
        self.character.current_pa = char_data.get("current_pa", self.character.current_pa)
        self.character.max_pa = char_data.get("max_pa", self.character.max_pa)
        self.character.current_pm = char_data.get("current_pm", self.character.current_pm)
        self.character.max_pm = char_data.get("max_pm", self.character.max_pm)
        
        # Position
        if "position" in char_data:
            pos_data = char_data["position"]
            self.character.position = GridPosition(
                x=pos_data.get("x", 0),
                y=pos_data.get("y", 0),
                cell_id=pos_data.get("cell_id", -1)
            )
        
        # Ressources
        self.character.kamas = char_data.get("kamas", self.character.kamas)
        self.character.energy = char_data.get("energy", self.character.energy)
        
        # États
        self.character.is_dead = char_data.get("is_dead", False)
        self.character.is_sitting = char_data.get("is_sitting", False)
        self.character.is_moving = char_data.get("is_moving", False)
        
        # Mise à jour des cooldowns des sorts
        if "spell_cooldowns" in char_data:
            for spell_id, cooldown in char_data["spell_cooldowns"].items():
                if spell_id in self.character.spells:
                    self.character.spells[spell_id].current_cooldown = cooldown
    
    def _update_map_state(self, map_data: Dict[str, Any]) -> None:
        """Met à jour l'état de la carte"""
        if not map_data:
            return
        
        # Coordonnées
        if "coordinates" in map_data:
            new_coords = tuple(map_data["coordinates"])
            if new_coords != self.map_state.coordinates:
                self.session_stats["maps_visited"].add(new_coords)
                self.map_state.coordinates = new_coords
        
        # Type de carte et zone
        self.map_state.map_type = MapType(map_data.get("type", "outdoor"))
        self.map_state.area_name = map_data.get("area_name", "")
        self.map_state.subarea_name = map_data.get("subarea_name", "")
        
        # Ressources
        if "resources" in map_data:
            self.map_state.resources = []
            for res_data in map_data["resources"]:
                resource = Resource(
                    resource_id=res_data["id"],
                    name=res_data["name"],
                    position=GridPosition(res_data["x"], res_data["y"]),
                    profession_required=res_data["profession"],
                    level_required=res_data["level"]
                )
                self.map_state.resources.append(resource)
        
        # NPCs
        if "npcs" in map_data:
            self.map_state.npcs = []
            for npc_data in map_data["npcs"]:
                npc = NPC(
                    npc_id=npc_data["id"],
                    name=npc_data["name"],
                    position=GridPosition(npc_data["x"], npc_data["y"]),
                    npc_type=npc_data.get("type", "unknown")
                )
                self.map_state.npcs.append(npc)
        
        # Joueurs
        if "players" in map_data:
            self.map_state.players = []
            for player_data in map_data["players"]:
                player = Player(
                    player_id=player_data["id"],
                    name=player_data["name"],
                    level=player_data["level"],
                    position=GridPosition(player_data["x"], player_data["y"]),
                    guild=player_data.get("guild")
                )
                self.map_state.players.append(player)
        
        # Monstres
        if "monsters" in map_data:
            self.map_state.monsters = []
            for monster_data in map_data["monsters"]:
                monster = Monster(
                    monster_id=monster_data["id"],
                    name=monster_data["name"],
                    level=monster_data["level"],
                    position=GridPosition(monster_data["x"], monster_data["y"]),
                    is_aggressive=monster_data.get("aggressive", False),
                    group_size=monster_data.get("group_size", 1)
                )
                self.map_state.monsters.append(monster)
    
    def _update_combat_state(self, combat_data: Dict[str, Any]) -> None:
        """Met à jour l'état du combat"""
        if not combat_data:
            # Vérifier si on est sorti du combat
            if self.combat.state != CombatState.NO_COMBAT:
                self.combat.state = CombatState.NO_COMBAT
                self.combat.allies.clear()
                self.combat.enemies.clear()
            return
        
        # État du combat
        old_state = self.combat.state
        new_state = CombatState(combat_data.get("state", "no_combat"))
        
        if old_state == CombatState.NO_COMBAT and new_state != CombatState.NO_COMBAT:
            self.session_stats["combat_count"] += 1
        
        self.combat.state = new_state
        self.combat.turn_number = combat_data.get("turn_number", 0)
        self.combat.my_turn = combat_data.get("my_turn", False)
        self.combat.time_remaining = combat_data.get("time_remaining", 30.0)
        
        # Entités alliées
        if "allies" in combat_data:
            self.combat.allies = []
            for ally_data in combat_data["allies"]:
                entity = CombatEntity(
                    entity_id=ally_data["id"],
                    name=ally_data["name"],
                    level=ally_data["level"],
                    current_hp=ally_data["current_hp"],
                    max_hp=ally_data["max_hp"],
                    position=GridPosition(ally_data["x"], ally_data["y"]) if "x" in ally_data else None,
                    is_ally=True,
                    is_player=ally_data.get("is_player", False)
                )
                self.combat.allies.append(entity)
        
        # Entités ennemies
        if "enemies" in combat_data:
            self.combat.enemies = []
            for enemy_data in combat_data["enemies"]:
                entity = CombatEntity(
                    entity_id=enemy_data["id"],
                    name=enemy_data["name"],
                    level=enemy_data["level"],
                    current_hp=enemy_data["current_hp"],
                    max_hp=enemy_data["max_hp"],
                    position=GridPosition(enemy_data["x"], enemy_data["y"]) if "x" in enemy_data else None,
                    is_ally=False,
                    is_monster=enemy_data.get("is_monster", True)
                )
                self.combat.enemies.append(entity)
    
    def _update_inventory(self, inventory_data: Dict[str, Any]) -> None:
        """Met à jour l'inventaire"""
        if not inventory_data:
            return
        
        # Pods
        self.character.pods_used = inventory_data.get("pods_used", self.character.pods_used)
        self.character.pods_max = inventory_data.get("pods_max", self.character.pods_max)
        
        # Items
        if "items" in inventory_data:
            self.inventory.clear()
            for item_data in inventory_data["items"]:
                item = InventoryItem(
                    item_id=item_data["id"],
                    name=item_data["name"],
                    quantity=item_data["quantity"],
                    category=item_data.get("category", "misc"),
                    estimated_value=item_data.get("value", 0),
                    is_consumable=item_data.get("consumable", False),
                    is_equipment=item_data.get("equipment", False)
                )
                self.inventory[item.item_id] = item
    
    def get_context_for_decision(self) -> Dict[str, Any]:
        """
        Retourne le contexte nécessaire pour la prise de décision IA
        Données optimisées et structurées pour les algorithmes de décision
        
        Returns:
            Dict contenant toutes les informations contextuelles
        """
        with self._lock:
            # Utilisation du cache si disponible
            cache_key = "decision_context"
            if self._is_cache_valid(cache_key):
                return self._state_cache[cache_key]
            
            context = {
                # Informations personnage critiques
                "character": {
                    "hp_percentage": self.character.hp_percentage(),
                    "current_pa": self.character.current_pa,
                    "current_pm": self.character.current_pm,
                    "position": {
                        "x": self.character.position.x if self.character.position else -1,
                        "y": self.character.position.y if self.character.position else -1
                    },
                    "is_dead": self.character.is_dead,
                    "energy_percentage": self.character.energy_percentage(),
                    "pods_percentage": self.character.pods_percentage()
                },
                
                # État du combat
                "combat": {
                    "in_combat": self.combat.state != CombatState.NO_COMBAT,
                    "is_my_turn": self.combat.is_my_turn(),
                    "time_remaining": self.combat.time_remaining,
                    "allies_count": len(self.combat.allies),
                    "enemies_count": len(self.combat.enemies),
                    "turn_number": self.combat.turn_number
                },
                
                # Environnement
                "map": {
                    "coordinates": self.map_state.coordinates,
                    "type": self.map_state.map_type.value,
                    "resources_count": len(self.map_state.resources),
                    "monsters_count": len(self.map_state.monsters),
                    "players_count": len(self.map_state.players)
                },
                
                # Objectifs potentiels
                "opportunities": {
                    "harvestable_resources": len(self._get_harvestable_resources()),
                    "can_bank": self.map_state.bank_present,
                    "can_use_zaap": self.map_state.zaap_present,
                    "inventory_full": self.character.pods_percentage() > 90
                },
                
                # Métadonnées temporelles
                "timing": {
                    "last_update": self._last_update.timestamp(),
                    "session_duration": (datetime.now() - self.session_stats["session_start"]).total_seconds(),
                    "updates_count": self.session_stats["total_updates"]
                }
            }
            
            # Mise en cache
            self._cache_result(cache_key, context)
            return context
    
    def validate_state(self) -> List[str]:
        """
        Valide la cohérence de l'état du jeu
        
        Returns:
            List des incohérences détectées
        """
        issues = []
        
        # Validation du personnage
        if self.character.current_hp < 0:
            issues.append("HP négatif détecté")
        if self.character.current_hp > self.character.max_hp:
            issues.append("HP courant supérieur au maximum")
        
        # Validation combat
        if self.combat.state != CombatState.NO_COMBAT:
            if not self.combat.allies and not self.combat.enemies:
                issues.append("Combat actif sans participants")
            
            if self.combat.my_turn and self.character.current_pa == 0:
                issues.append("Tour du joueur sans PA")
        
        # Validation position
        if self.character.position and self.combat.state != CombatState.NO_COMBAT:
            # En combat, la position doit être cohérente avec les cellules de combat
            pass
        
        return issues
    
    def diff_state(self, other_state: 'GameState') -> Dict[str, Any]:
        """
        Compare avec un autre état et retourne les différences
        
        Args:
            other_state: Autre état à comparer
            
        Returns:
            Dict contenant les différences
        """
        differences = {}
        
        # Comparaison HP
        if self.character.current_hp != other_state.character.current_hp:
            differences["hp_change"] = self.character.current_hp - other_state.character.current_hp
        
        # Comparaison position
        if self.character.position != other_state.character.position:
            differences["position_changed"] = True
            differences["new_position"] = self.character.position
        
        # Comparaison carte
        if self.map_state.coordinates != other_state.map_state.coordinates:
            differences["map_changed"] = True
            differences["new_map"] = self.map_state.coordinates
        
        # Comparaison combat
        if self.combat.state != other_state.combat.state:
            differences["combat_state_changed"] = True
            differences["new_combat_state"] = self.combat.state
        
        return differences
    
    def _get_harvestable_resources(self) -> List[Resource]:
        """Retourne les ressources récoltables (méthode interne)"""
        # Cette méthode nécessiterait les niveaux de métiers du personnage
        # Pour l'instant, retourne toutes les ressources
        return self.map_state.resources
    
    def _create_state_snapshot(self) -> Dict[str, Any]:
        """Crée un snapshot de l'état actuel pour comparaison"""
        return {
            "character_hp": self.character.current_hp,
            "character_position": copy.deepcopy(self.character.position),
            "map_coordinates": self.map_state.coordinates,
            "combat_state": self.combat.state,
            "timestamp": datetime.now()
        }
    
    def _calculate_changes(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les changements entre deux états"""
        changes = {}
        
        for key in old_state:
            if key == "timestamp":
                continue
            if old_state[key] != new_state[key]:
                changes[key] = {
                    "old": old_state[key],
                    "new": new_state[key]
                }
        
        return changes
    
    def _record_changes(self, changes: Dict[str, Any]) -> None:
        """Enregistre les changements dans l'historique"""
        change_record = {
            "timestamp": datetime.now(),
            "changes": changes
        }
        
        self._change_history.append(change_record)
        
        # Limitation de l'historique
        if len(self._change_history) > self._max_history:
            self._change_history.pop(0)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Vérifie si une entrée de cache est encore valide"""
        if key not in self._state_cache:
            return False
        
        if key not in self._cache_ttl:
            return False
        
        return datetime.now() < self._cache_ttl[key]
    
    def _cache_result(self, key: str, result: Any) -> None:
        """Met en cache un résultat"""
        self._state_cache[key] = result
        self._cache_ttl[key] = datetime.now() + timedelta(seconds=self._cache_duration)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de la session"""
        with self._lock:
            return {
                "session": dict(self.session_stats),
                "current_state": {
                    "character_level": self.character.level,
                    "hp_percentage": self.character.hp_percentage(),
                    "current_map": self.map_state.coordinates,
                    "combat_active": self.combat.state != CombatState.NO_COMBAT
                },
                "performance": {
                    "total_updates": self.session_stats["total_updates"],
                    "last_update": self._last_update,
                    "cache_hits": len(self._state_cache),
                    "history_size": len(self._change_history)
                }
            }