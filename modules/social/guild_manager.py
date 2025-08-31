"""
Module de gestion de guilde avec XP, percepteurs et événements
Fonctionnalités:
- Gestion des membres et hiérarchie
- Suivi XP et contributions
- Gestion des percepteurs
- Événements de guilde automatisés
- Système de récompenses
- Coordination des activités
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

class GuildRank(Enum):
    """Rangs dans la guilde"""
    MENEUR = "meneur"
    BRAS_DROIT = "bras_droit"
    PROTECTEUR = "protecteur"
    GARDIEN = "gardien"
    TRESORIER = "tresorier"
    ECLAIREUR = "eclaireur"
    RESERVE = "reserve"
    RECRUE = "recrue"

class EventType(Enum):
    """Types d'événements de guilde"""
    ATTAQUE_PERCEPTEUR = "attaque_percepteur"
    DEFENSE_PERCEPTEUR = "defense_percepteur"
    CONQUETE_PRISME = "conquete_prisme"
    DEFENSE_PRISME = "defense_prisme"
    KOLIZEUM_GUILDE = "kolizeum_guilde"
    RECRUTEMENT = "recrutement"
    PROMOTION = "promotion"
    EXCLUSION = "exclusion"
    CONTRIBUTION_XP = "contribution_xp"
    EVENEMENT_SPECIAL = "evenement_special"

class PerceptorStatus(Enum):
    """États des percepteurs"""
    ACTIF = "actif"
    SOUS_ATTAQUE = "sous_attaque"
    DETRUIT = "detruit"
    COLLECTE_REQUISE = "collecte_requise"
    MAINTENANCE = "maintenance"

@dataclass
class GuildMember:
    """Informations d'un membre de guilde"""
    name: str
    rank: GuildRank
    level: int
    xp_contributed: int = 0
    last_connection: datetime = field(default_factory=datetime.now)
    total_contributions: int = 0
    combat_score: int = 0
    profession_levels: Dict[str, int] = field(default_factory=dict)
    active: bool = True
    join_date: datetime = field(default_factory=datetime.now)
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour sérialisation"""
        data = asdict(self)
        data['last_connection'] = self.last_connection.isoformat()
        data['join_date'] = self.join_date.isoformat()
        data['rank'] = self.rank.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GuildMember':
        """Crée depuis un dictionnaire"""
        data['last_connection'] = datetime.fromisoformat(data['last_connection'])
        data['join_date'] = datetime.fromisoformat(data['join_date'])
        data['rank'] = GuildRank(data['rank'])
        return cls(**data)

@dataclass
class Perceptor:
    """Informations d'un percepteur"""
    id: str
    name: str
    map_position: Tuple[int, int]
    status: PerceptorStatus
    kamas_collected: int = 0
    items_collected: Dict[str, int] = field(default_factory=dict)
    last_collection: datetime = field(default_factory=datetime.now)
    attack_time: Optional[datetime] = None
    defenders: List[str] = field(default_factory=list)
    attacker_info: Optional[str] = None
    collection_rate: float = 0.0  # kamas par heure
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data['last_collection'] = self.last_collection.isoformat()
        data['attack_time'] = self.attack_time.isoformat() if self.attack_time else None
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Perceptor':
        """Crée depuis un dictionnaire"""
        data['last_collection'] = datetime.fromisoformat(data['last_collection'])
        if data['attack_time']:
            data['attack_time'] = datetime.fromisoformat(data['attack_time'])
        data['status'] = PerceptorStatus(data['status'])
        return cls(**data)

@dataclass
class GuildEvent:
    """Événement de guilde"""
    id: str
    event_type: EventType
    title: str
    description: str
    timestamp: datetime
    participants: List[str] = field(default_factory=list)
    rewards: Dict[str, int] = field(default_factory=dict)
    completed: bool = False
    priority: int = 1  # 1-5, 5 étant le plus urgent
    location: Optional[str] = None
    organizer: Optional[str] = None

class GuildOCR:
    """OCR spécialisé pour l'interface de guilde"""
    
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['fr', 'en'], gpu=True)
        
        # Régions de l'interface guilde
        self.guild_regions = {
            'member_list': (200, 150, 400, 500),
            'xp_info': (50, 50, 300, 100),
            'perceptor_list': (650, 150, 350, 400),
            'event_log': (100, 600, 800, 150)
        }
    
    def extract_guild_region(self, screenshot: np.ndarray, region_name: str) -> np.ndarray:
        """Extrait une région spécifique de l'interface guilde"""
        if region_name not in self.guild_regions:
            return screenshot
        
        x, y, w, h = self.guild_regions[region_name]
        return screenshot[y:y+h, x:x+w]
    
    def read_member_list(self, screenshot: np.ndarray) -> List[Dict]:
        """Lit la liste des membres depuis l'interface"""
        region = self.extract_guild_region(screenshot, 'member_list')
        
        # Prétraitement pour améliorer la lecture
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
        
        results = self.reader.readtext(enhanced)
        
        members = []
        for (bbox, text, confidence) in results:
            if confidence > 0.7:
                # Parser le format: "Nom [Niveau] Rang"
                match = re.match(r'^(.+?)\s*\[(\d+)\]\s*(.+)$', text.strip())
                if match:
                    name, level, rank = match.groups()
                    members.append({
                        'name': name.strip(),
                        'level': int(level),
                        'rank_text': rank.strip(),
                        'confidence': confidence
                    })
        
        return members
    
    def read_xp_info(self, screenshot: np.ndarray) -> Dict:
        """Lit les informations d'XP de guilde"""
        region = self.extract_guild_region(screenshot, 'xp_info')
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        
        xp_info = {}
        for (bbox, text, confidence) in results:
            if confidence > 0.6:
                # Chercher des patterns d'XP
                xp_match = re.search(r'(\d+(?:\.\d+)?[kmKM]?)\s*[xX][pP]', text)
                if xp_match:
                    xp_info['current_xp'] = xp_match.group(1)
                
                level_match = re.search(r'[nN]iveau\s*(\d+)', text)
                if level_match:
                    xp_info['guild_level'] = int(level_match.group(1))
        
        return xp_info
    
    def read_perceptor_list(self, screenshot: np.ndarray) -> List[Dict]:
        """Lit la liste des percepteurs"""
        region = self.extract_guild_region(screenshot, 'perceptor_list')
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        
        perceptors = []
        for (bbox, text, confidence) in results:
            if confidence > 0.6:
                # Parser les informations de percepteur
                # Format attendu: "Nom - Carte [X,Y] - Status - Kamas"
                parts = text.split('-')
                if len(parts) >= 3:
                    name = parts[0].strip()
                    
                    # Extraire position
                    pos_match = re.search(r'\[(-?\d+),(-?\d+)\]', parts[1])
                    position = None
                    if pos_match:
                        position = (int(pos_match.group(1)), int(pos_match.group(2)))
                    
                    # Extraire kamas
                    kamas_match = re.search(r'(\d+(?:\.\d+)?[kmKM]?)\s*kamas?', text, re.IGNORECASE)
                    kamas = 0
                    if kamas_match:
                        kamas_text = kamas_match.group(1).lower()
                        if 'k' in kamas_text:
                            kamas = int(float(kamas_text.replace('k', '')) * 1000)
                        elif 'm' in kamas_text:
                            kamas = int(float(kamas_text.replace('m', '')) * 1000000)
                        else:
                            kamas = int(kamas_text)
                    
                    perceptors.append({
                        'name': name,
                        'position': position,
                        'kamas': kamas,
                        'raw_text': text,
                        'confidence': confidence
                    })
        
        return perceptors

class XPTracker:
    """Système de suivi des contributions XP"""
    
    def __init__(self):
        self.xp_history = defaultdict(list)
        self.daily_contributions = defaultdict(int)
        self.weekly_contributions = defaultdict(int)
        self.monthly_contributions = defaultdict(int)
        self.contribution_goals = {}
        
    def add_xp_contribution(self, member_name: str, xp_amount: int, source: str = "unknown"):
        """Ajoute une contribution XP"""
        timestamp = datetime.now()
        
        contribution = {
            'timestamp': timestamp,
            'amount': xp_amount,
            'source': source
        }
        
        self.xp_history[member_name].append(contribution)
        
        # Mettre à jour les totaux
        today = timestamp.date()
        week_start = today - timedelta(days=today.weekday())
        month_start = today.replace(day=1)
        
        self.daily_contributions[f"{member_name}_{today}"] += xp_amount
        self.weekly_contributions[f"{member_name}_{week_start}"] += xp_amount
        self.monthly_contributions[f"{member_name}_{month_start}"] += xp_amount
    
    def get_member_stats(self, member_name: str) -> Dict:
        """Récupère les statistiques XP d'un membre"""
        history = self.xp_history.get(member_name, [])
        
        if not history:
            return {'total': 0, 'daily': 0, 'weekly': 0, 'monthly': 0}
        
        total_xp = sum(contrib['amount'] for contrib in history)
        
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())
        month_start = today.replace(day=1)
        
        daily_xp = self.daily_contributions.get(f"{member_name}_{today}", 0)
        weekly_xp = self.weekly_contributions.get(f"{member_name}_{week_start}", 0)
        monthly_xp = self.monthly_contributions.get(f"{member_name}_{month_start}", 0)
        
        return {
            'total': total_xp,
            'daily': daily_xp,
            'weekly': weekly_xp,
            'monthly': monthly_xp,
            'average_daily': total_xp / max(len(set(c['timestamp'].date() for c in history)), 1),
            'contribution_count': len(history)
        }
    
    def get_top_contributors(self, period: str = 'monthly', limit: int = 10) -> List[Tuple[str, int]]:
        """Récupère le top des contributeurs"""
        if period == 'daily':
            contributions = self.daily_contributions
        elif period == 'weekly':
            contributions = self.weekly_contributions
        else:
            contributions = self.monthly_contributions
        
        # Grouper par nom de membre
        member_totals = defaultdict(int)
        for key, amount in contributions.items():
            member_name = '_'.join(key.split('_')[:-1])  # Retirer la date
            member_totals[member_name] += amount
        
        # Trier et retourner le top
        sorted_contributors = sorted(member_totals.items(), key=lambda x: x[1], reverse=True)
        return sorted_contributors[:limit]

class PerceptorManager:
    """Gestionnaire des percepteurs de guilde"""
    
    def __init__(self):
        self.perceptors: Dict[str, Perceptor] = {}
        self.attack_alerts = deque(maxlen=50)
        self.collection_schedule = {}
        self.defense_strategies = {}
        
    def add_perceptor(self, perceptor: Perceptor):
        """Ajoute un percepteur à la gestion"""
        self.perceptors[perceptor.id] = perceptor
    
    def update_perceptor_status(self, perceptor_id: str, status: PerceptorStatus, 
                              attacker_info: Optional[str] = None):
        """Met à jour le statut d'un percepteur"""
        if perceptor_id not in self.perceptors:
            return
        
        perceptor = self.perceptors[perceptor_id]
        old_status = perceptor.status
        perceptor.status = status
        
        if status == PerceptorStatus.SOUS_ATTAQUE:
            perceptor.attack_time = datetime.now()
            perceptor.attacker_info = attacker_info
            
            # Créer une alerte d'attaque
            alert = {
                'timestamp': datetime.now(),
                'perceptor_id': perceptor_id,
                'perceptor_name': perceptor.name,
                'position': perceptor.map_position,
                'attacker': attacker_info,
                'priority': 5
            }
            self.attack_alerts.append(alert)
    
    def update_collection(self, perceptor_id: str, kamas: int, items: Dict[str, int]):
        """Met à jour la collecte d'un percepteur"""
        if perceptor_id not in self.perceptors:
            return
        
        perceptor = self.perceptors[perceptor_id]
        perceptor.kamas_collected = kamas
        perceptor.items_collected.update(items)
        perceptor.last_collection = datetime.now()
        
        # Calculer le taux de collecte
        time_diff = (datetime.now() - perceptor.last_collection).total_seconds() / 3600
        if time_diff > 0:
            perceptor.collection_rate = kamas / time_diff
    
    def get_attack_alerts(self, active_only: bool = True) -> List[Dict]:
        """Récupère les alertes d'attaque"""
        alerts = list(self.attack_alerts)
        
        if active_only:
            # Garder seulement les alertes récentes (dernière heure)
            cutoff_time = datetime.now() - timedelta(hours=1)
            alerts = [alert for alert in alerts if alert['timestamp'] > cutoff_time]
        
        return sorted(alerts, key=lambda x: (x['priority'], x['timestamp']), reverse=True)
    
    def get_collection_summary(self) -> Dict:
        """Récupère un résumé des collectes"""
        total_kamas = sum(p.kamas_collected for p in self.perceptors.values())
        total_items = defaultdict(int)
        
        for perceptor in self.perceptors.values():
            for item, count in perceptor.items_collected.items():
                total_items[item] += count
        
        # Calculer les percepteurs nécessitant une collecte
        collection_needed = []
        for perceptor in self.perceptors.values():
            if perceptor.status == PerceptorStatus.COLLECTE_REQUISE or \
               perceptor.kamas_collected > 100000:  # Plus de 100k kamas
                collection_needed.append(perceptor.id)
        
        return {
            'total_kamas': total_kamas,
            'total_items': dict(total_items),
            'collection_needed': collection_needed,
            'perceptors_count': len(self.perceptors),
            'active_perceptors': len([p for p in self.perceptors.values() 
                                    if p.status == PerceptorStatus.ACTIF])
        }
    
    def plan_collection_route(self) -> List[str]:
        """Planifie un itinéraire de collecte optimisé"""
        perceptors_to_collect = [
            p for p in self.perceptors.values()
            if p.status in [PerceptorStatus.ACTIF, PerceptorStatus.COLLECTE_REQUISE]
            and p.kamas_collected > 10000  # Seuil minimum
        ]
        
        if not perceptors_to_collect:
            return []
        
        # Trier par priorité : kamas collectés et proximité
        # (Algorithme simple, pourrait être amélioré avec du TSP)
        perceptors_to_collect.sort(key=lambda p: p.kamas_collected, reverse=True)
        
        return [p.id for p in perceptors_to_collect]

class EventManager:
    """Gestionnaire des événements de guilde"""
    
    def __init__(self):
        self.events: Dict[str, GuildEvent] = {}
        self.recurring_events = {}
        self.event_templates = self._load_event_templates()
        
    def _load_event_templates(self) -> Dict[EventType, Dict]:
        """Charge les templates d'événements"""
        return {
            EventType.ATTAQUE_PERCEPTEUR: {
                'title_template': 'Attaque sur {perceptor_name}',
                'description_template': 'Le percepteur {perceptor_name} en {location} est attaqué par {attacker}. Défense requise !',
                'priority': 5,
                'auto_participants': True
            },
            EventType.KOLIZEUM_GUILDE: {
                'title_template': 'Session Kolizéum Guilde',
                'description_template': 'Participation au Kolizéum pour gagner de l\'XP de guilde',
                'priority': 3,
                'auto_participants': False
            },
            EventType.CONTRIBUTION_XP: {
                'title_template': 'Objectif XP {period}',
                'description_template': 'Objectif de contribution XP pour la période {period}',
                'priority': 2,
                'auto_participants': False
            }
        }
    
    def create_event(self, event_type: EventType, **kwargs) -> str:
        """Crée un nouvel événement"""
        event_id = f"{event_type.value}_{int(time.time())}"
        
        template = self.event_templates.get(event_type, {})
        
        title = template.get('title_template', 'Événement Guilde').format(**kwargs)
        description = template.get('description_template', '').format(**kwargs)
        priority = template.get('priority', 1)
        
        event = GuildEvent(
            id=event_id,
            event_type=event_type,
            title=title,
            description=description,
            timestamp=datetime.now(),
            priority=priority,
            **{k: v for k, v in kwargs.items() 
               if k in ['location', 'organizer', 'participants', 'rewards']}
        )
        
        self.events[event_id] = event
        return event_id
    
    def complete_event(self, event_id: str, participants: Optional[List[str]] = None):
        """Marque un événement comme terminé"""
        if event_id not in self.events:
            return
        
        event = self.events[event_id]
        event.completed = True
        
        if participants:
            event.participants.extend(participants)
    
    def get_active_events(self) -> List[GuildEvent]:
        """Récupère les événements actifs"""
        return [event for event in self.events.values() if not event.completed]
    
    def get_priority_events(self, min_priority: int = 3) -> List[GuildEvent]:
        """Récupère les événements prioritaires"""
        events = [event for event in self.events.values() 
                 if not event.completed and event.priority >= min_priority]
        return sorted(events, key=lambda x: (x.priority, x.timestamp), reverse=True)
    
    def schedule_recurring_event(self, event_type: EventType, interval_hours: int, **kwargs):
        """Programme un événement récurrent"""
        recurring_id = f"recurring_{event_type.value}"
        self.recurring_events[recurring_id] = {
            'event_type': event_type,
            'interval_hours': interval_hours,
            'last_created': datetime.now(),
            'kwargs': kwargs
        }
    
    def check_recurring_events(self):
        """Vérifie et crée les événements récurrents si nécessaire"""
        current_time = datetime.now()
        
        for recurring_id, config in self.recurring_events.items():
            time_since_last = (current_time - config['last_created']).total_seconds() / 3600
            
            if time_since_last >= config['interval_hours']:
                self.create_event(config['event_type'], **config['kwargs'])
                config['last_created'] = current_time

class GuildManager:
    """Gestionnaire principal de la guilde"""
    
    def __init__(self):
        self.guild_info = {
            'name': '',
            'level': 1,
            'xp': 0,
            'members_count': 0,
            'max_members': 50
        }
        
        self.members: Dict[str, GuildMember] = {}
        self.ocr = GuildOCR()
        self.xp_tracker = XPTracker()
        self.perceptor_manager = PerceptorManager()
        self.event_manager = EventManager()
        
        self.running = False
        self.update_thread = None
        
        # Callbacks
        self.on_member_joined = None
        self.on_member_left = None
        self.on_perceptor_attack = None
        self.on_important_event = None
        
        # Configuration
        self.auto_collect_threshold = 50000  # Kamas
        self.auto_defense = True
        self.recruitment_active = False
        
    def start_monitoring(self):
        """Démarre la surveillance de la guilde"""
        self.running = True
        self.update_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.update_thread.start()
    
    def stop_monitoring(self):
        """Arrête la surveillance"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Boucle de surveillance principale"""
        while self.running:
            try:
                # Simuler une capture d'écran
                screenshot = self._get_screenshot()
                if screenshot is not None:
                    self.update_from_screenshot(screenshot)
                
                # Vérifier les événements récurrents
                self.event_manager.check_recurring_events()
                
                # Vérifier les alertes de percepteur
                self._check_perceptor_alerts()
                
                time.sleep(30)  # Vérifier toutes les 30 secondes
                
            except Exception as e:
                print(f"Erreur dans la surveillance guilde: {e}")
                time.sleep(60)
    
    def _get_screenshot(self) -> Optional[np.ndarray]:
        """Récupère une capture d'écran"""
        # À implémenter selon votre système
        return None
    
    def update_from_screenshot(self, screenshot: np.ndarray):
        """Met à jour les informations depuis une capture d'écran"""
        try:
            # Lire les membres
            member_data = self.ocr.read_member_list(screenshot)
            self._update_members(member_data)
            
            # Lire les infos XP
            xp_info = self.ocr.read_xp_info(screenshot)
            self._update_guild_xp(xp_info)
            
            # Lire les percepteurs
            perceptor_data = self.ocr.read_perceptor_list(screenshot)
            self._update_perceptors(perceptor_data)
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour depuis screenshot: {e}")
    
    def _update_members(self, member_data: List[Dict]):
        """Met à jour la liste des membres"""
        current_members = set()
        
        for data in member_data:
            name = data['name']
            current_members.add(name)
            
            if name not in self.members:
                # Nouveau membre
                rank = self._parse_rank(data['rank_text'])
                member = GuildMember(
                    name=name,
                    rank=rank,
                    level=data['level']
                )
                self.members[name] = member
                
                if self.on_member_joined:
                    self.on_member_joined(member)
            else:
                # Mise à jour membre existant
                member = self.members[name]
                member.level = data['level']
                member.last_connection = datetime.now()
        
        # Détecter les membres qui ont quitté
        all_members = set(self.members.keys())
        left_members = all_members - current_members
        
        for member_name in left_members:
            member = self.members[member_name]
            member.active = False
            
            if self.on_member_left:
                self.on_member_left(member)
    
    def _parse_rank(self, rank_text: str) -> GuildRank:
        """Parse le texte de rang vers l'enum"""
        rank_mapping = {
            'meneur': GuildRank.MENEUR,
            'bras droit': GuildRank.BRAS_DROIT,
            'protecteur': GuildRank.PROTECTEUR,
            'gardien': GuildRank.GARDIEN,
            'trésorier': GuildRank.TRESORIER,
            'éclaireur': GuildRank.ECLAIREUR,
            'réserve': GuildRank.RESERVE,
            'recrue': GuildRank.RECRUE
        }
        
        rank_lower = rank_text.lower()
        return rank_mapping.get(rank_lower, GuildRank.RECRUE)
    
    def _update_guild_xp(self, xp_info: Dict):
        """Met à jour les informations XP de guilde"""
        if 'guild_level' in xp_info:
            old_level = self.guild_info['level']
            new_level = xp_info['guild_level']
            
            if new_level > old_level:
                # Level up de guilde !
                self.event_manager.create_event(
                    EventType.EVENEMENT_SPECIAL,
                    title=f"Niveau {new_level} atteint !",
                    description=f"La guilde vient d'atteindre le niveau {new_level}!"
                )
            
            self.guild_info['level'] = new_level
        
        if 'current_xp' in xp_info:
            # Parser l'XP (format: "123.4k XP" ou "1.2M XP")
            xp_text = xp_info['current_xp'].lower().replace('xp', '').strip()
            try:
                if 'k' in xp_text:
                    xp = int(float(xp_text.replace('k', '')) * 1000)
                elif 'm' in xp_text:
                    xp = int(float(xp_text.replace('m', '')) * 1000000)
                else:
                    xp = int(xp_text)
                
                self.guild_info['xp'] = xp
            except ValueError:
                pass
    
    def _update_perceptors(self, perceptor_data: List[Dict]):
        """Met à jour les informations des percepteurs"""
        for data in perceptor_data:
            perceptor_id = data['name'].lower().replace(' ', '_')
            
            if perceptor_id not in self.perceptor_manager.perceptors:
                # Nouveau percepteur
                perceptor = Perceptor(
                    id=perceptor_id,
                    name=data['name'],
                    map_position=data['position'] or (0, 0),
                    status=PerceptorStatus.ACTIF,
                    kamas_collected=data['kamas']
                )
                self.perceptor_manager.add_perceptor(perceptor)
            else:
                # Mise à jour
                perceptor = self.perceptor_manager.perceptors[perceptor_id]
                old_kamas = perceptor.kamas_collected
                new_kamas = data['kamas']
                
                if new_kamas != old_kamas:
                    perceptor.kamas_collected = new_kamas
                    perceptor.last_collection = datetime.now()
    
    def _check_perceptor_alerts(self):
        """Vérifie les alertes de percepteur"""
        alerts = self.perceptor_manager.get_attack_alerts()
        
        for alert in alerts:
            if self.on_perceptor_attack:
                self.on_perceptor_attack(alert)
            
            # Créer un événement de défense si auto_defense est activé
            if self.auto_defense:
                event_id = self.event_manager.create_event(
                    EventType.DEFENSE_PERCEPTEUR,
                    perceptor_name=alert['perceptor_name'],
                    location=f"[{alert['position'][0]}, {alert['position'][1]}]",
                    attacker=alert['attacker'] or 'Inconnu'
                )
    
    def add_member_xp(self, member_name: str, xp_amount: int, source: str = "unknown"):
        """Ajoute de l'XP à un membre"""
        if member_name in self.members:
            self.members[member_name].xp_contributed += xp_amount
            self.members[member_name].total_contributions += 1
        
        self.xp_tracker.add_xp_contribution(member_name, xp_amount, source)
    
    def promote_member(self, member_name: str, new_rank: GuildRank, reason: str = ""):
        """Promeut un membre"""
        if member_name not in self.members:
            return False
        
        member = self.members[member_name]
        old_rank = member.rank
        member.rank = new_rank
        
        # Créer un événement de promotion
        self.event_manager.create_event(
            EventType.PROMOTION,
            title=f"Promotion de {member_name}",
            description=f"{member_name} a été promu de {old_rank.value} à {new_rank.value}. {reason}",
            participants=[member_name]
        )
        
        return True
    
    def kick_member(self, member_name: str, reason: str = ""):
        """Exclut un membre"""
        if member_name not in self.members:
            return False
        
        member = self.members[member_name]
        member.active = False
        
        # Créer un événement d'exclusion
        self.event_manager.create_event(
            EventType.EXCLUSION,
            title=f"Exclusion de {member_name}",
            description=f"{member_name} a été exclu de la guilde. Raison: {reason}",
            participants=[member_name]
        )
        
        return True
    
    def plan_perceptor_collection(self) -> List[str]:
        """Planifie une collecte des percepteurs"""
        return self.perceptor_manager.plan_collection_route()
    
    def get_guild_statistics(self) -> Dict:
        """Récupère les statistiques de guilde"""
        active_members = [m for m in self.members.values() if m.active]
        
        # Top contributeurs
        top_contributors = self.xp_tracker.get_top_contributors('monthly', 5)
        
        # Résumé percepteurs
        perceptor_summary = self.perceptor_manager.get_collection_summary()
        
        # Événements actifs
        active_events = len(self.event_manager.get_active_events())
        priority_events = len(self.event_manager.get_priority_events())
        
        return {
            'guild_info': self.guild_info,
            'members': {
                'total': len(self.members),
                'active': len(active_members),
                'average_level': sum(m.level for m in active_members) / max(len(active_members), 1),
                'total_xp_contributed': sum(m.xp_contributed for m in active_members)
            },
            'perceptors': perceptor_summary,
            'events': {
                'active': active_events,
                'priority': priority_events
            },
            'top_contributors': top_contributors
        }
    
    def save_guild_data(self, filepath: str):
        """Sauvegarde les données de guilde"""
        data = {
            'guild_info': self.guild_info,
            'members': {name: member.to_dict() for name, member in self.members.items()},
            'perceptors': {pid: perc.to_dict() for pid, perc in self.perceptor_manager.perceptors.items()},
            'xp_history': dict(self.xp_tracker.xp_history),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_guild_data(self, filepath: str):
        """Charge les données de guilde"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.guild_info = data.get('guild_info', {})
            
            # Charger les membres
            for name, member_data in data.get('members', {}).items():
                self.members[name] = GuildMember.from_dict(member_data)
            
            # Charger les percepteurs
            for pid, perc_data in data.get('perceptors', {}).items():
                perceptor = Perceptor.from_dict(perc_data)
                self.perceptor_manager.perceptors[pid] = perceptor
            
            # Charger l'historique XP
            xp_history = data.get('xp_history', {})
            for member, history in xp_history.items():
                self.xp_tracker.xp_history[member] = history
                
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    guild_manager = GuildManager()
    
    # Callbacks
    def on_member_joined(member):
        print(f"Nouveau membre: {member.name} ({member.rank.value})")
    
    def on_perceptor_attack(alert):
        print(f"ALERTE: Percepteur {alert['perceptor_name']} attaqué par {alert['attacker']}!")
    
    guild_manager.on_member_joined = on_member_joined
    guild_manager.on_perceptor_attack = on_perceptor_attack
    
    # Programmer des événements récurrents
    guild_manager.event_manager.schedule_recurring_event(
        EventType.CONTRIBUTION_XP,
        interval_hours=24,
        period="journalier"
    )
    
    # Démarrer la surveillance
    guild_manager.start_monitoring()
    
    try:
        while True:
            time.sleep(60)
            stats = guild_manager.get_guild_statistics()
            print(f"Stats guilde: {stats['members']['active']} membres actifs")
    except KeyboardInterrupt:
        guild_manager.stop_monitoring()
        guild_manager.save_guild_data("guild_data.json")