"""
Game State - État complet du jeu à un instant T
Extrait de la vision et des observations
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class BotState(Enum):
    """État général du bot"""
    IDLE = "idle"                     # Inactif, en attente
    NAVIGATING = "navigating"         # En déplacement
    IN_COMBAT = "in_combat"           # En combat
    FARMING = "farming"               # Récolte de ressources
    QUESTING = "questing"             # Fait une quête
    TRADING = "trading"               # Commerce/HDV
    CRAFTING = "crafting"             # Craft
    INTERACTING = "interacting"       # Interaction NPC/UI
    DEAD = "dead"                     # Mort
    LOADING = "loading"               # Chargement/Transition
    ERROR = "error"                   # Erreur/Bloqué


class CombatState(Enum):
    """État en combat"""
    NOT_IN_COMBAT = "not_in_combat"
    COMBAT_STARTING = "combat_starting"
    MY_TURN = "my_turn"
    ENEMY_TURN = "enemy_turn"
    ALLY_TURN = "ally_turn"
    COMBAT_ENDING = "combat_ending"


@dataclass
class CharacterStats:
    """Statistiques du personnage"""
    # Vie
    hp: int = 100
    max_hp: int = 100
    hp_percent: float = 100.0
    
    # Points d'action/mouvement
    pa: int = 6
    max_pa: int = 6
    pm: int = 3
    max_pm: int = 3
    
    # Level & XP
    level: int = 1
    xp: int = 0
    xp_percent: float = 0.0
    
    # Position
    map_pos: Tuple[int, int] = (0, 0)  # Coordonnées map (x, y)
    cell_pos: int = 0  # Position cellule sur la map
    
    # Kamas & inventaire
    kamas: int = 0
    pods: int = 0
    max_pods: int = 1000
    pods_percent: float = 0.0
    
    # Combat
    in_combat: bool = False
    combat_turn: int = 0
    
    def is_low_hp(self, threshold: float = 30.0) -> bool:
        """HP faibles?"""
        return self.hp_percent < threshold
    
    def is_full_hp(self) -> bool:
        """HP pleins?"""
        return self.hp >= self.max_hp
    
    def can_cast_spell(self, pa_cost: int) -> bool:
        """Peut lancer un sort?"""
        return self.pa >= pa_cost
    
    def can_move(self, pm_cost: int) -> bool:
        """Peut se déplacer?"""
        return self.pm >= pm_cost
    
    def is_overweight(self) -> bool:
        """Surcharge?"""
        return self.pods_percent > 90.0


@dataclass
class EntityInfo:
    """Information sur une entité (mob, joueur, PNJ, ressource)"""
    entity_id: str
    entity_type: str  # "mob", "player", "npc", "resource", "item"
    name: str
    level: int = 0
    position: Tuple[int, int] = (0, 0)
    cell: int = 0
    hp: int = 100
    max_hp: int = 100
    is_ally: bool = False
    is_enemy: bool = False
    distance: float = 0.0
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CombatInfo:
    """Informations de combat"""
    in_combat: bool = False
    state: CombatState = CombatState.NOT_IN_COMBAT
    turn_number: int = 0
    my_turn: bool = False
    time_left: float = 30.0  # Secondes restantes
    
    # Entités en combat
    allies: List[EntityInfo] = field(default_factory=list)
    enemies: List[EntityInfo] = field(default_factory=list)
    
    # Tactique
    current_target: Optional[EntityInfo] = None
    threat_level: float = 0.0  # 0.0 = facile, 1.0 = mortel
    
    def get_alive_enemies(self) -> List[EntityInfo]:
        """Ennemis vivants"""
        return [e for e in self.enemies if e.hp > 0]
    
    def get_weakest_enemy(self) -> Optional[EntityInfo]:
        """Ennemi le plus faible"""
        alive = self.get_alive_enemies()
        if not alive:
            return None
        return min(alive, key=lambda e: e.hp)
    
    def get_closest_enemy(self) -> Optional[EntityInfo]:
        """Ennemi le plus proche"""
        alive = self.get_alive_enemies()
        if not alive:
            return None
        return min(alive, key=lambda e: e.distance)


@dataclass
class UIState:
    """État de l'interface utilisateur"""
    window_active: bool = False
    window_pos: Tuple[int, int] = (0, 0)
    window_size: Tuple[int, int] = (1280, 720)
    
    # UI visible
    inventory_open: bool = False
    spell_book_open: bool = False
    chat_visible: bool = True
    map_visible: bool = True
    
    # Messages
    last_chat_message: str = ""
    last_notification: str = ""
    
    # Curseur
    cursor_pos: Tuple[int, int] = (0, 0)


@dataclass
class EnvironmentState:
    """État de l'environnement"""
    current_map: str = "unknown"  # "(5,-18)" format
    map_name: str = "Inconnue"
    map_type: str = "normal"  # "normal", "dungeon", "house", "pvp"
    
    # Entités visibles
    entities: List[EntityInfo] = field(default_factory=list)
    
    # Ressources disponibles
    resources: List[EntityInfo] = field(default_factory=list)
    
    # Dangers
    danger_level: float = 0.0  # 0.0 = sûr, 1.0 = très dangereux
    aggressive_mobs_nearby: int = 0
    
    # Météo/Heure
    time_of_day: str = "unknown"  # "day", "night"
    weather: str = "clear"
    
    def get_harvestable_resources(self) -> List[EntityInfo]:
        """Ressources récoltables"""
        return [r for r in self.resources if r.visible]
    
    def get_nearest_resource(self) -> Optional[EntityInfo]:
        """Ressource la plus proche"""
        harvestable = self.get_harvestable_resources()
        if not harvestable:
            return None
        return min(harvestable, key=lambda r: r.distance)


@dataclass
class GameState:
    """
    État complet du jeu à un instant T
    Mis à jour par la vision et les observations
    """
    # Métadonnées
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0
    
    # État général
    bot_state: BotState = BotState.IDLE
    previous_state: BotState = BotState.IDLE
    
    # Personnage
    character: CharacterStats = field(default_factory=CharacterStats)
    
    # Combat
    combat: CombatInfo = field(default_factory=CombatInfo)
    
    # UI
    ui: UIState = field(default_factory=UIState)
    
    # Environnement
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    
    # Objectif actuel
    current_objective: str = "idle"
    current_task: str = "waiting"
    
    # Dernière action
    last_action: str = "none"
    last_action_time: float = 0.0
    last_action_success: bool = True
    
    # Flags
    is_loading: bool = False
    is_moving: bool = False
    is_interacting: bool = False
    
    def update_timestamp(self):
        """Met à jour le timestamp"""
        self.timestamp = time.time()
        self.frame_number += 1
    
    def change_state(self, new_state: BotState, reason: str = ""):
        """Change l'état du bot"""
        if self.bot_state != new_state:
            logger.info(f"État changé: {self.bot_state.value} → {new_state.value} ({reason})")
            self.previous_state = self.bot_state
            self.bot_state = new_state
    
    def enter_combat(self):
        """Entre en combat"""
        self.combat.in_combat = True
        self.combat.state = CombatState.COMBAT_STARTING
        self.change_state(BotState.IN_COMBAT, "Combat détecté")
    
    def exit_combat(self):
        """Sort du combat"""
        self.combat.in_combat = False
        self.combat.state = CombatState.NOT_IN_COMBAT
        self.change_state(BotState.IDLE, "Combat terminé")
    
    def is_safe(self) -> bool:
        """Environnement sûr?"""
        return (
            self.environment.danger_level < 0.3 and
            not self.combat.in_combat and
            not self.character.is_low_hp()
        )
    
    def can_act(self) -> bool:
        """Peut agir maintenant?"""
        # Simplification: toujours pouvoir agir sauf si mort/erreur/loading
        return (
            not self.is_loading and
            not self.is_moving and
            self.bot_state not in [BotState.DEAD, BotState.ERROR]
        )
    
    def needs_healing(self) -> bool:
        """Besoin de soin?"""
        return self.character.hp_percent < 50.0
    
    def needs_rest(self) -> bool:
        """Besoin de repos?"""
        # À implémenter avec simulation de fatigue
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'timestamp': self.timestamp,
            'frame': self.frame_number,
            'bot_state': self.bot_state.value,
            'character': {
                'hp': self.character.hp,
                'max_hp': self.character.max_hp,
                'hp_percent': self.character.hp_percent,
                'level': self.character.level,
                'map': self.character.map_pos,
                'kamas': self.character.kamas
            },
            'combat': {
                'in_combat': self.combat.in_combat,
                'my_turn': self.combat.my_turn,
                'enemies': len(self.combat.enemies),
                'allies': len(self.combat.allies)
            },
            'environment': {
                'map': self.environment.current_map,
                'entities': len(self.environment.entities),
                'resources': len(self.environment.resources),
                'danger': self.environment.danger_level
            }
        }
    
    def __str__(self) -> str:
        """Représentation string"""
        return (
            f"GameState(frame={self.frame_number}, "
            f"state={self.bot_state.value}, "
            f"hp={self.character.hp}/{self.character.max_hp}, "
            f"map={self.environment.current_map}, "
            f"combat={self.combat.in_combat})"
        )


def create_game_state() -> GameState:
    """Factory function"""
    return GameState()
