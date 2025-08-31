"""
Module de gestion de groupe avec coordination intelligente
Fonctionnalités:
- Suivi du leader et synchronisation
- Coordination des combats en groupe
- Partage équitable du loot
- Communication intelligente
- Stratégies de groupe adaptatives
- Gestion des rôles et formations
"""

import cv2
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import re
import math

class GroupRole(Enum):
    """Rôles dans le groupe"""
    LEADER = "leader"
    TANK = "tank"
    DPS = "dps"
    HEALER = "healer"
    SUPPORT = "support"
    FOLLOWER = "follower"

class GroupFormation(Enum):
    """Formations de groupe"""
    LINE = "line"  # En ligne
    CIRCLE = "circle"  # En cercle
    TRIANGLE = "triangle"  # En triangle
    CROSS = "cross"  # En croix
    SCATTERED = "scattered"  # Dispersé
    CUSTOM = "custom"  # Formation personnalisée

class CombatPhase(Enum):
    """Phases de combat"""
    PREPARATION = "preparation"
    ENGAGEMENT = "engagement"
    COMBAT = "combat"
    CLEANUP = "cleanup"
    LOOTING = "looting"
    REGROUPING = "regrouping"

class LootPriority(Enum):
    """Priorités de loot"""
    NEED = "need"  # Besoin absolu
    GREED = "greed"  # Envie/peut servir
    PASS = "pass"  # Ne veut pas
    AUTO = "auto"  # Distribution automatique

@dataclass
class GroupMember:
    """Membre du groupe"""
    name: str
    role: GroupRole
    level: int
    classe: str = ""
    position: Tuple[int, int] = (0, 0)
    health_percent: float = 100.0
    mp: int = 6
    ap: int = 6
    alive: bool = True
    online: bool = True
    combat_score: int = 0
    loot_contribution: float = 0.0
    preferred_loot: List[str] = field(default_factory=list)
    last_action: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['role'] = self.role.value
        data['last_action'] = self.last_action.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GroupMember':
        data['role'] = GroupRole(data['role'])
        data['last_action'] = datetime.fromisoformat(data['last_action'])
        return cls(**data)

@dataclass
class LootItem:
    """Item de loot"""
    name: str
    rarity: str
    estimated_value: int
    item_type: str  # arme, armure, ressource, etc.
    suitable_for: List[str] = field(default_factory=list)  # Classes ou rôles
    quantity: int = 1
    pickup_position: Tuple[int, int] = (0, 0)
    priority_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class CombatTarget:
    """Cible de combat"""
    name: str
    position: Tuple[int, int]
    health_percent: float
    threat_level: int  # 1-5
    priority: int  # 1-10
    assigned_to: Optional[str] = None
    damage_taken: int = 0
    effects: List[str] = field(default_factory=list)

class GroupOCR:
    """OCR spécialisé pour l'interface de groupe"""
    
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['fr', 'en'], gpu=True)
        
        # Régions de l'interface
        self.group_regions = {
            'member_list': (10, 10, 200, 400),  # Liste des membres
            'combat_info': (220, 10, 300, 200),  # Infos de combat
            'loot_window': (400, 300, 400, 300),  # Fenêtre de loot
            'chat_tactical': (50, 500, 600, 100),  # Chat tactique
            'minimap': (950, 10, 200, 200)  # Minimap pour positions
        }
    
    def read_group_members(self, screenshot: np.ndarray) -> List[Dict]:
        """Lit les informations des membres du groupe"""
        region = self._extract_region(screenshot, 'member_list')
        
        # Préprocessing pour améliorer la lecture
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.8, beta=20)
        
        results = self.reader.readtext(enhanced)
        
        members = []
        for (bbox, text, confidence) in results:
            if confidence > 0.6:
                # Parser format: "Nom [Niveau] Classe HP:X% MP:X AP:X"
                member_match = re.match(
                    r'^(.+?)\s*\[(\d+)\]\s*(\w+)?\s*(?:HP:(\d+)%?)?\s*(?:MP:(\d+))?\s*(?:AP:(\d+))?',
                    text.strip()
                )
                
                if member_match:
                    name, level, classe, hp, mp, ap = member_match.groups()
                    
                    member_info = {
                        'name': name.strip(),
                        'level': int(level),
                        'classe': classe or "",
                        'health_percent': float(hp) if hp else 100.0,
                        'mp': int(mp) if mp else 6,
                        'ap': int(ap) if ap else 6,
                        'confidence': confidence
                    }
                    members.append(member_info)
        
        return members
    
    def read_combat_targets(self, screenshot: np.ndarray) -> List[Dict]:
        """Lit les informations des cibles en combat"""
        region = self._extract_region(screenshot, 'combat_info')
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        
        targets = []
        for (bbox, text, confidence) in results:
            if confidence > 0.7:
                # Parser format ennemi: "Nom HP:X% Niv:X"
                target_match = re.match(
                    r'^(.+?)\s*(?:HP:(\d+)%?)?\s*(?:Niv:(\d+))?',
                    text.strip()
                )
                
                if target_match:
                    name, hp, level = target_match.groups()
                    
                    target_info = {
                        'name': name.strip(),
                        'health_percent': float(hp) if hp else 100.0,
                        'level': int(level) if level else 0,
                        'position': self._estimate_position_from_bbox(bbox),
                        'confidence': confidence
                    }
                    targets.append(target_info)
        
        return targets
    
    def read_loot_window(self, screenshot: np.ndarray) -> List[Dict]:
        """Lit le contenu de la fenêtre de loot"""
        region = self._extract_region(screenshot, 'loot_window')
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        
        loot_items = []
        for (bbox, text, confidence) in results:
            if confidence > 0.6:
                # Détecter les items par patterns
                item_patterns = [
                    r'^(.+?)\s*x(\d+)$',  # Item x quantité
                    r'^(.+?)\s*\((.+?)\)$',  # Item (rareté)
                    r'^(.+?)$'  # Item simple
                ]
                
                for pattern in item_patterns:
                    match = re.match(pattern, text.strip())
                    if match:
                        if len(match.groups()) == 2 and 'x' in text:
                            name, quantity = match.groups()
                            quantity = int(quantity)
                        elif len(match.groups()) == 2:
                            name, rarity = match.groups()
                            quantity = 1
                        else:
                            name = match.group(1)
                            quantity = 1
                            rarity = "commun"
                        
                        loot_item = {
                            'name': name.strip(),
                            'quantity': quantity,
                            'rarity': locals().get('rarity', 'commun'),
                            'position': self._estimate_position_from_bbox(bbox),
                            'confidence': confidence
                        }
                        loot_items.append(loot_item)
                        break
        
        return loot_items
    
    def _extract_region(self, screenshot: np.ndarray, region_name: str) -> np.ndarray:
        """Extrait une région spécifique"""
        if region_name not in self.group_regions:
            return screenshot
        
        x, y, w, h = self.group_regions[region_name]
        return screenshot[y:y+h, x:x+w]
    
    def _estimate_position_from_bbox(self, bbox) -> Tuple[int, int]:
        """Estime la position à partir d'une bounding box"""
        # Calculer le centre de la bounding box
        points = np.array(bbox)
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        return (center_x, center_y)

class PositionTracker:
    """Système de suivi des positions des membres"""
    
    def __init__(self):
        self.member_positions = {}
        self.leader_position = (0, 0)
        self.formation = GroupFormation.LINE
        self.formation_positions = {}
        self.movement_history = defaultdict(deque)
        
    def update_member_position(self, member_name: str, position: Tuple[int, int]):
        """Met à jour la position d'un membre"""
        self.member_positions[member_name] = position
        
        # Ajouter à l'historique
        history = self.movement_history[member_name]
        history.append((datetime.now(), position))
        
        # Garder seulement les 20 dernières positions
        if len(history) > 20:
            history.popleft()
    
    def set_leader_position(self, position: Tuple[int, int]):
        """Définit la position du leader"""
        self.leader_position = position
    
    def calculate_formation_positions(self, formation: GroupFormation, 
                                    members: List[str]) -> Dict[str, Tuple[int, int]]:
        """Calcule les positions de formation pour les membres"""
        leader_x, leader_y = self.leader_position
        positions = {}
        
        if formation == GroupFormation.LINE:
            # Formation en ligne
            for i, member in enumerate(members):
                offset_x = i * 40  # 40 pixels entre chaque membre
                positions[member] = (leader_x + offset_x, leader_y)
        
        elif formation == GroupFormation.CIRCLE:
            # Formation en cercle
            radius = 60
            angle_step = 2 * math.pi / len(members)
            
            for i, member in enumerate(members):
                angle = i * angle_step
                offset_x = int(radius * math.cos(angle))
                offset_y = int(radius * math.sin(angle))
                positions[member] = (leader_x + offset_x, leader_y + offset_y)
        
        elif formation == GroupFormation.TRIANGLE:
            # Formation en triangle
            triangle_positions = [
                (0, 0),      # Pointe (leader)
                (-40, 40),   # Arrière gauche
                (40, 40),    # Arrière droite
                (-80, 80),   # Très arrière gauche
                (80, 80),    # Très arrière droite
            ]
            
            for i, member in enumerate(members[:len(triangle_positions)]):
                offset_x, offset_y = triangle_positions[i]
                positions[member] = (leader_x + offset_x, leader_y + offset_y)
        
        elif formation == GroupFormation.CROSS:
            # Formation en croix
            cross_positions = [
                (0, 0),      # Centre
                (0, -40),    # Nord
                (40, 0),     # Est
                (0, 40),     # Sud
                (-40, 0),    # Ouest
            ]
            
            for i, member in enumerate(members[:len(cross_positions)]):
                offset_x, offset_y = cross_positions[i]
                positions[member] = (leader_x + offset_x, leader_y + offset_y)
        
        self.formation_positions = positions
        return positions
    
    def get_distance_to_formation(self, member_name: str) -> float:
        """Calcule la distance d'un membre à sa position de formation"""
        if member_name not in self.formation_positions:
            return 0.0
        
        current_pos = self.member_positions.get(member_name, (0, 0))
        target_pos = self.formation_positions[member_name]
        
        dx = current_pos[0] - target_pos[0]
        dy = current_pos[1] - target_pos[1]
        
        return math.sqrt(dx*dx + dy*dy)
    
    def is_formation_maintained(self, threshold: float = 50.0) -> bool:
        """Vérifie si la formation est maintenue"""
        for member_name in self.formation_positions:
            if self.get_distance_to_formation(member_name) > threshold:
                return False
        return True
    
    def get_member_speed(self, member_name: str) -> float:
        """Calcule la vitesse de déplacement d'un membre"""
        history = self.movement_history.get(member_name, deque())
        
        if len(history) < 2:
            return 0.0
        
        # Calculer sur les 5 dernières positions
        recent_history = list(history)[-5:]
        total_distance = 0.0
        total_time = 0.0
        
        for i in range(1, len(recent_history)):
            time1, pos1 = recent_history[i-1]
            time2, pos2 = recent_history[i]
            
            distance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            time_diff = (time2 - time1).total_seconds()
            
            total_distance += distance
            total_time += time_diff
        
        return total_distance / total_time if total_time > 0 else 0.0

class CombatCoordinator:
    """Coordinateur de combat intelligent"""
    
    def __init__(self):
        self.targets: Dict[str, CombatTarget] = {}
        self.combat_phase = CombatPhase.PREPARATION
        self.target_assignments = {}
        self.combat_strategies = self._load_combat_strategies()
        self.turn_order = []
        self.current_turn = 0
        
    def _load_combat_strategies(self) -> Dict[str, Dict]:
        """Charge les stratégies de combat"""
        return {
            'focus_fire': {
                'description': 'Concentrer les attaques sur une cible',
                'priority_func': lambda targets: max(targets, key=lambda t: t.threat_level),
                'suitable_for': ['boss', 'elite']
            },
            'spread_damage': {
                'description': 'Répartir les dégâts sur plusieurs cibles',
                'priority_func': lambda targets: min(targets, key=lambda t: t.damage_taken),
                'suitable_for': ['multiple_weak']
            },
            'protect_healer': {
                'description': 'Protéger le soigneur en priorité',
                'priority_func': lambda targets: max(targets, key=lambda t: self._threat_to_healer(t)),
                'suitable_for': ['healer_present']
            }
        }
    
    def _threat_to_healer(self, target: CombatTarget) -> float:
        """Évalue la menace d'une cible envers le soigneur"""
        # Logique simplifiée
        return target.threat_level * (1.0 if 'healer' in target.effects else 0.5)
    
    def update_targets(self, target_data: List[Dict]):
        """Met à jour la liste des cibles"""
        current_targets = set()
        
        for data in target_data:
            target_name = data['name']
            current_targets.add(target_name)
            
            if target_name not in self.targets:
                # Nouvelle cible
                target = CombatTarget(
                    name=target_name,
                    position=data.get('position', (0, 0)),
                    health_percent=data.get('health_percent', 100.0),
                    threat_level=self._estimate_threat_level(data),
                    priority=self._calculate_priority(data)
                )
                self.targets[target_name] = target
            else:
                # Mise à jour cible existante
                target = self.targets[target_name]
                target.health_percent = data.get('health_percent', target.health_percent)
                target.position = data.get('position', target.position)
        
        # Supprimer les cibles qui ne sont plus présentes
        inactive_targets = set(self.targets.keys()) - current_targets
        for target_name in inactive_targets:
            del self.targets[target_name]
    
    def _estimate_threat_level(self, target_data: Dict) -> int:
        """Estime le niveau de menace d'une cible"""
        level = target_data.get('level', 1)
        name = target_data['name'].lower()
        
        # Mots-clés indiquant un niveau de menace élevé
        high_threat_keywords = ['boss', 'chef', 'elite', 'champion', 'archonte']
        medium_threat_keywords = ['garde', 'guerrier', 'mage', 'archer']
        
        base_threat = min(level // 20 + 1, 5)  # 1-5 basé sur le niveau
        
        for keyword in high_threat_keywords:
            if keyword in name:
                return min(base_threat + 2, 5)
        
        for keyword in medium_threat_keywords:
            if keyword in name:
                return min(base_threat + 1, 5)
        
        return base_threat
    
    def _calculate_priority(self, target_data: Dict) -> int:
        """Calcule la priorité d'attaque d'une cible"""
        health = target_data.get('health_percent', 100.0)
        level = target_data.get('level', 1)
        
        # Priorité plus élevée pour les cibles blessées et dangereuses
        priority = 5
        
        if health < 25:
            priority += 3  # Finir les cibles faibles
        elif health < 50:
            priority += 1
        
        if level > 100:
            priority += 2  # Cibles de haut niveau
        
        return min(priority, 10)
    
    def assign_targets(self, members: Dict[str, GroupMember]) -> Dict[str, str]:
        """Assigne les cibles aux membres du groupe"""
        available_targets = [t for t in self.targets.values() if t.health_percent > 0]
        available_members = [m for m in members.values() if m.alive and m.role != GroupRole.HEALER]
        
        if not available_targets or not available_members:
            return {}
        
        assignments = {}
        
        # Trier les cibles par priorité
        sorted_targets = sorted(available_targets, key=lambda t: t.priority, reverse=True)
        
        # Stratégie simple : assigner la cible la plus prioritaire au meilleur DPS
        for i, member in enumerate(available_members):
            if i < len(sorted_targets):
                target = sorted_targets[i]
                assignments[member.name] = target.name
                target.assigned_to = member.name
        
        self.target_assignments = assignments
        return assignments
    
    def get_recommended_action(self, member: GroupMember, combat_state: Dict) -> Dict:
        """Recommande une action pour un membre"""
        if not member.alive:
            return {'action': 'wait', 'reason': 'member_dead'}
        
        if member.role == GroupRole.HEALER:
            return self._get_healer_action(member, combat_state)
        elif member.role == GroupRole.TANK:
            return self._get_tank_action(member, combat_state)
        else:
            return self._get_dps_action(member, combat_state)
    
    def _get_healer_action(self, member: GroupMember, combat_state: Dict) -> Dict:
        """Action recommandée pour un soigneur"""
        # Chercher le membre le plus blessé
        members = combat_state.get('members', {})
        injured_members = [(name, m) for name, m in members.items() 
                          if m.alive and m.health_percent < 50]
        
        if injured_members:
            most_injured = min(injured_members, key=lambda x: x[1].health_percent)
            return {
                'action': 'heal',
                'target': most_injured[0],
                'priority': 10 - int(most_injured[1].health_percent / 10),
                'reason': f'heal_{most_injured[0]}_at_{most_injured[1].health_percent}%'
            }
        
        return {'action': 'wait', 'reason': 'no_healing_needed'}
    
    def _get_tank_action(self, member: GroupMember, combat_state: Dict) -> Dict:
        """Action recommandée pour un tank"""
        # Le tank doit attirer l'attention des ennemis les plus dangereux
        high_threat_targets = [t for t in self.targets.values() 
                              if t.threat_level >= 4 and t.health_percent > 0]
        
        if high_threat_targets:
            priority_target = max(high_threat_targets, key=lambda t: t.threat_level)
            return {
                'action': 'taunt',
                'target': priority_target.name,
                'priority': priority_target.threat_level,
                'reason': f'tank_high_threat_{priority_target.name}'
            }
        
        # Sinon, attaquer normalement
        return self._get_dps_action(member, combat_state)
    
    def _get_dps_action(self, member: GroupMember, combat_state: Dict) -> Dict:
        """Action recommandée pour un DPS"""
        assigned_target = self.target_assignments.get(member.name)
        
        if assigned_target and assigned_target in self.targets:
            target = self.targets[assigned_target]
            if target.health_percent > 0:
                return {
                    'action': 'attack',
                    'target': assigned_target,
                    'priority': target.priority,
                    'reason': f'assigned_target_{assigned_target}'
                }
        
        # Pas de cible assignée, choisir la plus prioritaire
        available_targets = [t for t in self.targets.values() if t.health_percent > 0]
        if available_targets:
            best_target = max(available_targets, key=lambda t: t.priority)
            return {
                'action': 'attack',
                'target': best_target.name,
                'priority': best_target.priority,
                'reason': f'priority_target_{best_target.name}'
            }
        
        return {'action': 'wait', 'reason': 'no_targets'}

class LootDistributor:
    """Système de distribution équitable du loot"""
    
    def __init__(self):
        self.loot_history = defaultdict(list)
        self.member_contributions = defaultdict(float)
        self.loot_preferences = {}
        self.distribution_rules = self._load_distribution_rules()
        
    def _load_distribution_rules(self) -> Dict[str, Dict]:
        """Charge les règles de distribution"""
        return {
            'equipment': {
                'priority_order': ['need', 'greed', 'pass'],
                'tie_breaker': 'contribution',
                'auto_assign': False
            },
            'resources': {
                'priority_order': ['contribution'],
                'tie_breaker': 'random',
                'auto_assign': True
            },
            'rare_items': {
                'priority_order': ['need', 'contribution', 'greed'],
                'tie_breaker': 'discussion',
                'auto_assign': False
            }
        }
    
    def analyze_loot(self, loot_items: List[Dict], members: Dict[str, GroupMember]) -> List[LootItem]:
        """Analyse le loot et calcule les priorités"""
        analyzed_items = []
        
        for item_data in loot_items:
            loot_item = LootItem(
                name=item_data['name'],
                rarity=item_data.get('rarity', 'commun'),
                estimated_value=self._estimate_value(item_data),
                item_type=self._classify_item_type(item_data['name']),
                quantity=item_data.get('quantity', 1)
            )
            
            # Calculer les scores de priorité pour chaque membre
            for member_name, member in members.items():
                score = self._calculate_priority_score(loot_item, member)
                loot_item.priority_scores[member_name] = score
            
            # Déterminer pour qui l'item est approprié
            loot_item.suitable_for = self._determine_suitability(loot_item, members)
            
            analyzed_items.append(loot_item)
        
        return analyzed_items
    
    def _estimate_value(self, item_data: Dict) -> int:
        """Estime la valeur d'un item"""
        rarity_multipliers = {
            'commun': 1,
            'inhabituel': 5,
            'rare': 25,
            'mythique': 100,
            'legendaire': 500,
            'relique': 1000
        }
        
        base_value = 1000  # Valeur de base
        rarity = item_data.get('rarity', 'commun').lower()
        multiplier = rarity_multipliers.get(rarity, 1)
        
        return base_value * multiplier
    
    def _classify_item_type(self, item_name: str) -> str:
        """Classifie le type d'item"""
        name_lower = item_name.lower()
        
        weapon_keywords = ['épée', 'baguette', 'arc', 'dague', 'hache', 'marteau']
        armor_keywords = ['casque', 'plastron', 'bottes', 'gants', 'ceinture', 'cape']
        resource_keywords = ['minerai', 'bois', 'céréale', 'poisson', 'pierre']
        
        for keyword in weapon_keywords:
            if keyword in name_lower:
                return 'arme'
        
        for keyword in armor_keywords:
            if keyword in name_lower:
                return 'armure'
        
        for keyword in resource_keywords:
            if keyword in name_lower:
                return 'ressource'
        
        return 'divers'
    
    def _calculate_priority_score(self, item: LootItem, member: GroupMember) -> float:
        """Calcule le score de priorité d'un item pour un membre"""
        base_score = 0.0
        
        # Score basé sur la contribution au combat
        contribution_score = self.member_contributions.get(member.name, 0.0) * 0.3
        
        # Score basé sur la classe/rôle
        role_score = self._get_role_compatibility(item, member) * 0.4
        
        # Score basé sur les préférences
        preference_score = self._get_preference_score(item, member) * 0.3
        
        total_score = base_score + contribution_score + role_score + preference_score
        
        return max(0.0, min(10.0, total_score))  # Normaliser entre 0 et 10
    
    def _get_role_compatibility(self, item: LootItem, member: GroupMember) -> float:
        """Évalue la compatibilité item/rôle"""
        compatibility_matrix = {
            GroupRole.TANK: {
                'arme': 0.8,
                'armure': 1.0,
                'ressource': 0.3,
                'divers': 0.5
            },
            GroupRole.DPS: {
                'arme': 1.0,
                'armure': 0.7,
                'ressource': 0.4,
                'divers': 0.5
            },
            GroupRole.HEALER: {
                'arme': 0.6,
                'armure': 0.8,
                'ressource': 0.5,
                'divers': 0.7
            },
            GroupRole.SUPPORT: {
                'arme': 0.7,
                'armure': 0.7,
                'ressource': 0.8,
                'divers': 1.0
            }
        }
        
        role_compat = compatibility_matrix.get(member.role, {})
        return role_compat.get(item.item_type, 0.5)
    
    def _get_preference_score(self, item: LootItem, member: GroupMember) -> float:
        """Score basé sur les préférences du membre"""
        if item.name in member.preferred_loot:
            return 1.0
        
        # Chercher des préférences partielles
        for pref in member.preferred_loot:
            if pref.lower() in item.name.lower() or item.name.lower() in pref.lower():
                return 0.7
        
        return 0.0
    
    def _determine_suitability(self, item: LootItem, members: Dict[str, GroupMember]) -> List[str]:
        """Détermine pour qui l'item est approprié"""
        suitable_members = []
        
        for member_name, member in members.items():
            score = item.priority_scores.get(member_name, 0.0)
            if score >= 6.0:  # Seuil de convenance
                suitable_members.append(member_name)
        
        return suitable_members
    
    def distribute_loot(self, loot_items: List[LootItem], 
                       members: Dict[str, GroupMember]) -> Dict[str, List[str]]:
        """Distribue le loot entre les membres"""
        distribution = defaultdict(list)
        
        for item in loot_items:
            recipient = self._select_recipient(item, members)
            if recipient:
                distribution[recipient].append(item.name)
                
                # Mettre à jour l'historique
                self.loot_history[recipient].append({
                    'item': item.name,
                    'value': item.estimated_value,
                    'timestamp': datetime.now()
                })
        
        return dict(distribution)
    
    def _select_recipient(self, item: LootItem, members: Dict[str, GroupMember]) -> Optional[str]:
        """Sélectionne le destinataire d'un item"""
        if not item.suitable_for:
            # Distribution équitable pour les items non spécialisés
            return self._fair_distribution_selection(item, members)
        
        # Chercher le membre avec le meilleur score parmi ceux pour qui c'est approprié
        suitable_members = {name: score for name, score in item.priority_scores.items() 
                           if name in item.suitable_for}
        
        if suitable_members:
            return max(suitable_members, key=suitable_members.get)
        
        return None
    
    def _fair_distribution_selection(self, item: LootItem, 
                                   members: Dict[str, GroupMember]) -> Optional[str]:
        """Sélection équitable pour les items génériques"""
        # Calculer la valeur totale reçue par chaque membre
        member_values = {}
        for member_name in members:
            total_value = sum(loot_entry['value'] 
                            for loot_entry in self.loot_history[member_name])
            member_values[member_name] = total_value
        
        # Sélectionner le membre qui a reçu le moins de valeur
        if member_values:
            return min(member_values, key=member_values.get)
        
        return list(members.keys())[0] if members else None
    
    def update_contribution(self, member_name: str, contribution: float):
        """Met à jour la contribution d'un membre"""
        self.member_contributions[member_name] = contribution
    
    def get_loot_statistics(self) -> Dict:
        """Récupère les statistiques de loot"""
        stats = {}
        
        for member_name, history in self.loot_history.items():
            total_value = sum(entry['value'] for entry in history)
            item_count = len(history)
            
            stats[member_name] = {
                'total_value': total_value,
                'item_count': item_count,
                'average_value': total_value / max(item_count, 1),
                'recent_items': [entry['item'] for entry in history[-5:]]
            }
        
        return stats

class GroupManager:
    """Gestionnaire principal du groupe"""
    
    def __init__(self):
        self.group_info = {
            'name': '',
            'leader': '',
            'formation': GroupFormation.LINE,
            'auto_follow': True,
            'auto_combat': True
        }
        
        self.members: Dict[str, GroupMember] = {}
        self.ocr = GroupOCR()
        self.position_tracker = PositionTracker()
        self.combat_coordinator = CombatCoordinator()
        self.loot_distributor = LootDistributor()
        
        self.running = False
        self.update_thread = None
        
        # Callbacks
        self.on_member_joined = None
        self.on_member_left = None
        self.on_combat_started = None
        self.on_loot_distributed = None
        self.on_formation_broken = None
        
    def start_monitoring(self):
        """Démarre la surveillance du groupe"""
        self.running = True
        self.update_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.update_thread.start()
    
    def stop_monitoring(self):
        """Arrête la surveillance"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Boucle principale de surveillance"""
        while self.running:
            try:
                screenshot = self._get_screenshot()
                if screenshot is not None:
                    self.update_from_screenshot(screenshot)
                
                # Vérifier la formation
                self._check_formation()
                
                # Coordonner le combat si nécessaire
                if self.combat_coordinator.combat_phase != CombatPhase.PREPARATION:
                    self._coordinate_combat()
                
                time.sleep(2)  # Vérification toutes les 2 secondes
                
            except Exception as e:
                print(f"Erreur dans la surveillance groupe: {e}")
                time.sleep(10)
    
    def _get_screenshot(self) -> Optional[np.ndarray]:
        """Récupère une capture d'écran"""
        # À implémenter selon votre système
        return None
    
    def update_from_screenshot(self, screenshot: np.ndarray):
        """Met à jour depuis une capture d'écran"""
        try:
            # Lire les membres
            member_data = self.ocr.read_group_members(screenshot)
            self._update_members(member_data)
            
            # Lire les cibles de combat
            target_data = self.ocr.read_combat_targets(screenshot)
            if target_data:
                self.combat_coordinator.update_targets(target_data)
            
            # Lire le loot si présent
            loot_data = self.ocr.read_loot_window(screenshot)
            if loot_data:
                self._process_loot(loot_data)
                
        except Exception as e:
            print(f"Erreur mise à jour screenshot: {e}")
    
    def _update_members(self, member_data: List[Dict]):
        """Met à jour les informations des membres"""
        current_members = set()
        
        for data in member_data:
            name = data['name']
            current_members.add(name)
            
            if name not in self.members:
                # Nouveau membre
                role = self._determine_role(data)
                member = GroupMember(
                    name=name,
                    role=role,
                    level=data['level'],
                    classe=data.get('classe', ''),
                    health_percent=data.get('health_percent', 100.0),
                    mp=data.get('mp', 6),
                    ap=data.get('ap', 6)
                )
                self.members[name] = member
                
                if self.on_member_joined:
                    self.on_member_joined(member)
            else:
                # Mise à jour membre existant
                member = self.members[name]
                member.health_percent = data.get('health_percent', member.health_percent)
                member.mp = data.get('mp', member.mp)
                member.ap = data.get('ap', member.ap)
                member.last_action = datetime.now()
        
        # Détecter les membres qui ont quitté
        left_members = set(self.members.keys()) - current_members
        for member_name in left_members:
            member = self.members[member_name]
            member.online = False
            
            if self.on_member_left:
                self.on_member_left(member)
    
    def _determine_role(self, member_data: Dict) -> GroupRole:
        """Détermine le rôle d'un membre basé sur sa classe"""
        classe = member_data.get('classe', '').lower()
        
        role_mapping = {
            'iop': GroupRole.TANK,
            'sacrieur': GroupRole.TANK,
            'eniripsa': GroupRole.HEALER,
            'sadida': GroupRole.SUPPORT,
            'cra': GroupRole.DPS,
            'sram': GroupRole.DPS,
            'enutrof': GroupRole.SUPPORT,
            'feca': GroupRole.TANK,
            'osamodas': GroupRole.SUPPORT,
            'eca': GroupRole.DPS,
            'pandawa': GroupRole.TANK,
            'roublard': GroupRole.DPS
        }
        
        return role_mapping.get(classe, GroupRole.FOLLOWER)
    
    def _check_formation(self):
        """Vérifie si la formation est maintenue"""
        if not self.group_info['auto_follow']:
            return
        
        formation_positions = self.position_tracker.calculate_formation_positions(
            self.group_info['formation'],
            list(self.members.keys())
        )
        
        if not self.position_tracker.is_formation_maintained():
            if self.on_formation_broken:
                self.on_formation_broken(formation_positions)
    
    def _coordinate_combat(self):
        """Coordonne le combat"""
        if not self.group_info['auto_combat']:
            return
        
        # Assigner les cibles
        assignments = self.combat_coordinator.assign_targets(self.members)
        
        # Générer des recommandations d'actions
        for member_name, member in self.members.items():
            if member.alive and member.online:
                combat_state = {
                    'members': self.members,
                    'targets': self.combat_coordinator.targets,
                    'phase': self.combat_coordinator.combat_phase
                }
                
                action = self.combat_coordinator.get_recommended_action(member, combat_state)
                
                # Ici vous implémenteriez l'exécution de l'action recommandée
                self._execute_member_action(member, action)
    
    def _execute_member_action(self, member: GroupMember, action: Dict):
        """Exécute une action recommandée (à implémenter)"""
        # À implémenter selon votre système de contrôle
        print(f"[COMBAT] {member.name} -> {action['action']} sur {action.get('target', 'N/A')}")
    
    def _process_loot(self, loot_data: List[Dict]):
        """Traite le loot détecté"""
        analyzed_loot = self.loot_distributor.analyze_loot(loot_data, self.members)
        distribution = self.loot_distributor.distribute_loot(analyzed_loot, self.members)
        
        if distribution and self.on_loot_distributed:
            self.on_loot_distributed(distribution)
    
    def set_formation(self, formation: GroupFormation):
        """Définit la formation du groupe"""
        self.group_info['formation'] = formation
        self.position_tracker.formation = formation
    
    def set_leader(self, leader_name: str):
        """Définit le leader du groupe"""
        if leader_name in self.members:
            self.group_info['leader'] = leader_name
            self.members[leader_name].role = GroupRole.LEADER
    
    def follow_leader(self, enable: bool = True):
        """Active/désactive le suivi automatique"""
        self.group_info['auto_follow'] = enable
    
    def set_combat_auto(self, enable: bool = True):
        """Active/désactive le combat automatique"""
        self.group_info['auto_combat'] = enable
    
    def get_group_statistics(self) -> Dict:
        """Récupère les statistiques du groupe"""
        online_members = [m for m in self.members.values() if m.online]
        alive_members = [m for m in online_members if m.alive]
        
        # Statistiques de combat
        total_damage = sum(m.combat_score for m in online_members)
        average_health = sum(m.health_percent for m in alive_members) / max(len(alive_members), 1)
        
        # Statistiques de loot
        loot_stats = self.loot_distributor.get_loot_statistics()
        
        return {
            'group_info': self.group_info,
            'members': {
                'total': len(self.members),
                'online': len(online_members),
                'alive': len(alive_members),
                'average_level': sum(m.level for m in online_members) / max(len(online_members), 1),
                'average_health': average_health
            },
            'combat': {
                'phase': self.combat_coordinator.combat_phase.value,
                'active_targets': len(self.combat_coordinator.targets),
                'total_damage': total_damage,
                'assignments': len(self.combat_coordinator.target_assignments)
            },
            'formation': {
                'type': self.group_info['formation'].value,
                'maintained': self.position_tracker.is_formation_maintained()
            },
            'loot': loot_stats
        }
    
    def save_group_data(self, filepath: str):
        """Sauvegarde les données du groupe"""
        data = {
            'group_info': {
                **self.group_info,
                'formation': self.group_info['formation'].value
            },
            'members': {name: member.to_dict() for name, member in self.members.items()},
            'loot_history': dict(self.loot_distributor.loot_history),
            'contributions': dict(self.loot_distributor.member_contributions),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_group_data(self, filepath: str):
        """Charge les données du groupe"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            group_info = data.get('group_info', {})
            group_info['formation'] = GroupFormation(group_info.get('formation', 'line'))
            self.group_info = group_info
            
            # Charger les membres
            for name, member_data in data.get('members', {}).items():
                self.members[name] = GroupMember.from_dict(member_data)
            
            # Charger l'historique de loot
            loot_history = data.get('loot_history', {})
            for member, history in loot_history.items():
                self.loot_distributor.loot_history[member] = history
            
            # Charger les contributions
            contributions = data.get('contributions', {})
            for member, contrib in contributions.items():
                self.loot_distributor.member_contributions[member] = contrib
                
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    group_manager = GroupManager()
    
    # Callbacks
    def on_member_joined(member):
        print(f"Nouveau membre: {member.name} ({member.role.value})")
    
    def on_loot_distributed(distribution):
        print(f"Loot distribué: {distribution}")
    
    def on_formation_broken(positions):
        print("Formation cassée! Repositionnement requis.")
    
    group_manager.on_member_joined = on_member_joined
    group_manager.on_loot_distributed = on_loot_distributed
    group_manager.on_formation_broken = on_formation_broken
    
    # Configuration
    group_manager.set_formation(GroupFormation.TRIANGLE)
    group_manager.follow_leader(True)
    group_manager.set_combat_auto(True)
    
    # Démarrer la surveillance
    group_manager.start_monitoring()
    
    try:
        while True:
            time.sleep(30)
            stats = group_manager.get_group_statistics()
            print(f"Groupe: {stats['members']['online']} membres en ligne")
    except KeyboardInterrupt:
        group_manager.stop_monitoring()
        group_manager.save_group_data("group_data.json")