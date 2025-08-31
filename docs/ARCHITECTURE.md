# 🏗️ Architecture Technique - TacticalBot

## 📋 Vue d'Ensemble

TacticalBot utilise une architecture modulaire événementielle conçue pour la robustesse, la scalabilité et la maintenabilité. Le système fonctionne en temps réel à 30 FPS avec une couche de décision intelligente à 10 FPS pour optimiser les performances.

## 🎯 Principes Architecturaux

### 1. Architecture Modulaire
- **Découplage** : Modules indépendants communicant via événements
- **Extensibilité** : Ajout/suppression de modules à chaud
- **Responsabilité unique** : Chaque module a une fonction spécifique
- **Interface standardisée** : Tous les modules implémentent `IModule`

### 2. Architecture Événementielle
- **Bus d'événements centralisé** : Communication asynchrone inter-modules
- **Priorités** : Gestion des événements par ordre d'importance
- **Filtrage** : Abonnement sélectif aux types d'événements
- **Performance** : Traitement non-bloquant des événements

### 3. Architecture Temps Réel
- **Boucle principale 30 FPS** : Mise à jour fluide des modules
- **Décisions 10 FPS** : Optimisation computationnelle
- **Threading** : Séparation des tâches longues
- **Monitoring** : Suivi temps réel des performances

## 🏢 Structure Générale

```
TacticalBot/
├── 🔧 Core Engine/              # Moteur central
│   ├── BotEngine               # Orchestrateur principal
│   ├── EventBus                # Système d'événements
│   ├── ModuleInterface         # Interface des modules
│   └── PerformanceMonitor      # Monitoring performances
│
├── 🧠 Intelligence Layer/       # Couche intelligence
│   ├── DecisionEngine          # Moteur de décision IA
│   ├── StrategySelector        # Sélection de stratégies
│   ├── CombatAI                # Intelligence de combat
│   └── SafetyManager           # Gestion de la sécurité
│
├── 🎮 Game Interface/           # Interface jeu
│   ├── VisionSystem            # Système de vision
│   ├── StateManager            # Gestion d'état
│   ├── ActionExecutor          # Exécution d'actions
│   └── InputManager            # Gestion des entrées
│
├── 🔌 Modules Spécialisés/      # Modules métier
│   ├── ProfessionManager       # Gestion des métiers
│   ├── NavigationSystem        # Système de navigation
│   ├── EconomyManager          # Gestion économique
│   ├── SocialManager           # Interactions sociales
│   └── AutomationModules       # Automatisation
│
└── 🗄️ Data & Config/           # Données et configuration
    ├── ConfigManager           # Gestionnaire de configuration
    ├── DatabaseManager         # Base de données
    ├── LoggingSystem           # Système de logs
    └── StatisticsManager       # Gestionnaire de statistiques
```

## 🔧 Couche Core Engine

### BotEngine - Orchestrateur Central

```python
class BotEngine:
    """
    Cœur du système - Orchestre tous les composants
    
    Responsabilités:
    - Gestion du cycle de vie des modules
    - Coordination via le bus d'événements
    - Monitoring des performances
    - Gestion des erreurs et récupération
    - Interface de contrôle principale
    """
```

**Cycle Principal (30 FPS)** :
```
┌─────────────────────┐
│ 1. Mise à jour État │ ←─┐
├─────────────────────┤   │
│ 2. Traitement Events│   │
├─────────────────────┤   │ Cycle 33ms
│ 3. Update Modules   │   │ (30 FPS)
├─────────────────────┤   │
│ 4. Monitoring       │   │
├─────────────────────┤   │
│ 5. Safety Checks    │   │
├─────────────────────┤   │
│ 6. Régulation Timing│ ──┘
└─────────────────────┘
```

### EventBus - Système d'Événements

```python
class EventBus:
    """
    Bus d'événements centralisé pour communication inter-modules
    
    Fonctionnalités:
    - Publication/Abonnement asynchrone
    - Gestion des priorités (CRITICAL > HIGH > NORMAL > LOW)
    - Filtrage par type d'événement
    - Gestion de la surcharge (rate limiting)
    - Statistiques et monitoring
    """
```

**Types d'Événements** :
- `MODULE_ERROR` : Erreur dans un module
- `STATE_CHANGED` : Changement d'état du jeu
- `ACTION_COMPLETED` : Action terminée
- `COMBAT_STARTED` : Début de combat
- `RESOURCE_FOUND` : Ressource détectée
- `SAFETY_ALERT` : Alerte de sécurité
- `CONFIG_CHANGED` : Modification de configuration
- `PERFORMANCE_WARNING` : Alerte performance

### Module Interface - Standardisation

```python
class IModule(ABC):
    """Interface abstraite pour tous les modules"""
    
    @abstractmethod
    def initialize(self, config) -> bool:
        """Initialisation du module"""
        
    @abstractmethod  
    def update(self, game_state) -> Optional[Dict]:
        """Mise à jour principale (30 FPS)"""
        
    @abstractmethod
    def handle_event(self, event) -> bool:
        """Traitement des événements"""
        
    @abstractmethod
    def get_state(self) -> Dict:
        """État actuel du module"""
        
    @abstractmethod
    def cleanup(self) -> None:
        """Nettoyage des ressources"""
```

## 🧠 Couche Intelligence

### Decision Engine - Moteur Décisionnel

Architecture multi-critères avec apprentissage :

```
Input: Actions Possibles + Contexte
          │
          ▼
┌─────────────────────┐
│ Évaluation Multi-   │ ← Critères pondérés
│ Critères            │   - Priorité
├─────────────────────┤   - Confiance  
│ Résolution Conflits │   - Durée estimée
├─────────────────────┤   - Probabilité succès
│ Apprentissage       │   - Niveau de risque
├─────────────────────┤   - Récompense
│ Recommandation      │
└─────────────────────┘
          │
          ▼
Output: Meilleure Action + Explication
```

### Strategy Selector - Sélection Adaptative

```python
# Stratégies disponibles
STRATEGIES = {
    AGGRESSIVE: {      # Maximise les gains, accepte les risques
        'risk_tolerance': 0.8,
        'efficiency_weight': 1.5,
        'safety_weight': 0.3
    },
    DEFENSIVE: {       # Priorité à la sécurité
        'risk_tolerance': 0.2, 
        'efficiency_weight': 0.7,
        'safety_weight': 2.0
    },
    BALANCED: {        # Équilibre optimal
        'risk_tolerance': 0.5,
        'efficiency_weight': 1.0,
        'safety_weight': 1.0
    },
    EFFICIENT: {       # Optimise le temps
        'risk_tolerance': 0.6,
        'efficiency_weight': 2.0,
        'safety_weight': 0.8
    },
    STEALTH: {         # Évite la détection
        'risk_tolerance': 0.3,
        'efficiency_weight': 0.6,
        'safety_weight': 1.8
    },
    SOCIAL: {          # Interactions coopératives
        'risk_tolerance': 0.4,
        'efficiency_weight': 0.8,
        'safety_weight': 1.2
    }
}
```

## 🎮 Interface Jeu

### Vision System - Analyse Visuelle

```
┌─────────────────────┐
│ Capture d'Écran     │
├─────────────────────┤
│ Préprocessing       │ ← Filtres, normalisation
├─────────────────────┤
│ Template Matching   │ ← Base de templates
├─────────────────────┤
│ OCR Intelligent     │ ← Reconnaissance texte
├─────────────────────┤
│ Object Detection    │ ← Détection d'objets
├─────────────────────┤
│ State Extraction    │ ← Extraction informations
└─────────────────────┘
```

**Composants** :
- **ScreenAnalyzer** : Capture et analyse d'écran
- **TemplateManager** : Gestion des templates
- **OCREngine** : Reconnaissance de caractères
- **ObjectDetector** : Détection d'éléments UI

### State Manager - Gestion d'État

```python
class GameState:
    """État complet du jeu en temps réel"""
    
    # État du personnage
    character: CharacterState
    health_percent: float
    mana_percent: float
    level: int
    experience: int
    position: Position
    
    # État de l'interface
    windows_open: List[str]
    inventory_slots: List[Item]
    spells_available: List[Spell]
    
    # État du monde
    current_map: str
    nearby_entities: List[Entity] 
    resources_visible: List[Resource]
    threats_detected: List[Threat]
    
    # État métier
    profession_levels: Dict[str, int]
    current_objective: str
    session_statistics: SessionStats
```

## 🔌 Modules Spécialisés

### Architecture des Modules

Chaque module spécialisé suit le pattern suivant :

```python
class SpecializedModule(IModule):
    def __init__(self):
        super().__init__("module_name")
        self.config = ModuleConfig()
        self.state = ModuleState() 
        self.metrics = ModuleMetrics()
        self.ai = ModuleAI()  # IA spécialisée
        
    def update(self, game_state):
        # 1. Analyser l'état du jeu
        situation = self.analyze_situation(game_state)
        
        # 2. Prendre des décisions
        actions = self.ai.get_recommended_actions(situation)
        
        # 3. Exécuter l'action optimale
        best_action = self.select_best_action(actions)
        result = self.execute_action(best_action)
        
        # 4. Mettre à jour les métriques
        self.metrics.update(result)
        
        # 5. Partager les données pertinentes
        return self.get_shared_data()
```

### Profession Manager - Gestion des Métiers

```
┌─────────────────────┐
│ Farmer              │ ← 29 ressources agricoles
├─────────────────────┤
│ Lumberjack          │ ← 25 types d'arbres  
├─────────────────────┤
│ Miner               │ ← 30 minerais + gemmes
├─────────────────────┤
│ Alchemist           │ ← 16 recettes potions
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Optimisation Globale│
├─────────────────────┤
│ - Routes optimales  │
│ - Synergies         │ ← Inter-métier
│ - ROI calculation   │
│ - Sessions planning │
└─────────────────────┘
```

### Navigation System - Pathfinding

```python
class NavigationSystem:
    """
    Système de navigation intelligent
    
    Algorithmes:
    - A* pour pathfinding optimal  
    - Évitement d'obstacles dynamiques
    - Optimisation de trajets multi-points
    - Gestion des zones dangereuses
    - Cache des chemins fréquents
    """
    
    def find_path(self, start, goal, constraints):
        # A* avec heuristique adaptée
        path = self.astar(start, goal, self.heuristic)
        
        # Optimisation et lissage
        optimized_path = self.smooth_path(path)
        
        # Vérifications sécurité
        safe_path = self.apply_safety_constraints(optimized_path)
        
        return safe_path
```

## 🗄️ Gestion des Données

### Architecture de Persistance

```
┌─────────────────────┐
│ Configuration       │ ← JSON files
├─────────────────────┤   - engine.json
│ Statistics          │   - modules.json
├─────────────────────┤   - professions.json
│ Learning Data       │   - safety.json
├─────────────────────┤
│ Cache               │ ← Templates, paths
├─────────────────────┤
│ Logs                │ ← Structured logging
└─────────────────────┘
```

### Config Manager - Gestion Configuration

```python
class ConfigManager:
    """Gestionnaire centralisé de configuration"""
    
    def __init__(self):
        self.config_files = {
            'engine': 'config/engine.json',
            'modules': 'config/modules.json',
            'professions': 'config/professions.json',
            'safety': 'config/safety.json'
        }
        
    def load_all_configs(self):
        """Charge toutes les configurations"""
        
    def apply_profile(self, profile_name):
        """Applique un profil prédéfini"""
        
    def validate_config(self, config_data):
        """Valide une configuration"""
```

## 🔄 Flux de Données

### Cycle Complet d'Exécution

```
┌─────────────────────┐
│ 1. Capture Vision   │ ← Screenshots, OCR
├─────────────────────┤
│ 2. State Update     │ ← Game state extraction  
├─────────────────────┤
│ 3. Module Analysis  │ ← Chaque module analyse
├─────────────────────┤
│ 4. Decision Making  │ ← IA décisionnelle
├─────────────────────┤
│ 5. Action Selection │ ← Sélection meilleure action
├─────────────────────┤
│ 6. Safety Check     │ ← Vérifications sécurité
├─────────────────────┤
│ 7. Action Execution │ ← Exécution contrôlée
├─────────────────────┤
│ 8. Result Analysis  │ ← Analyse des résultats
├─────────────────────┤
│ 9. Learning Update  │ ← Mise à jour apprentissage
├─────────────────────┤
│ 10. Stats & Logging │ ← Statistiques et logs
└─────────────────────┘
```

### Communication Inter-Modules

```python
# Exemple de communication via événements
class ProfessionModule(IModule):
    def update(self, game_state):
        if self.resource_found:
            # Publier découverte de ressource
            self.engine.event_bus.publish_immediate(
                EventType.RESOURCE_FOUND,
                {
                    'resource_type': 'wheat',
                    'position': (x, y),
                    'confidence': 0.95
                },
                sender='profession_farmer',
                priority=EventPriority.NORMAL
            )
        
class NavigationModule(IModule):
    def handle_event(self, event):
        if event.type == EventType.RESOURCE_FOUND:
            # Calculer chemin vers ressource
            path = self.calculate_path_to(event.data['position'])
            # Publier chemin optimal
            self.publish_navigation_update(path)
```

## 📊 Monitoring et Performance

### Métriques Système

```python
class PerformanceMetrics:
    """Métriques de performance temps réel"""
    
    # Métriques moteur
    fps_actual: float           # FPS réel
    loop_time_avg: float        # Temps cycle moyen
    memory_usage_mb: int        # Usage mémoire
    cpu_usage_percent: float    # Usage CPU
    
    # Métriques modules
    active_modules: int         # Modules actifs
    errors_per_minute: int      # Taux d'erreurs
    events_processed: int       # Événements traités
    
    # Métriques gameplay
    actions_per_minute: float   # Actions/minute
    success_rate: float         # Taux de succès
    xp_per_hour: float         # XP gagnée/heure
    kamas_per_hour: float      # Kamas gagnés/heure
```

### Système de Logging

```
logs/
├── tacticalbot.log          # Log principal
├── modules/                 # Logs par module
│   ├── decision.log
│   ├── professions.log
│   ├── navigation.log
│   └── safety.log
├── performance/             # Métriques performance
│   ├── fps_metrics.log
│   ├── memory_usage.log
│   └── error_rates.log
└── events/                  # Logs d'événements
    ├── combat_events.log
    ├── resource_events.log
    └── safety_events.log
```

## 🔒 Sécurité et Robustesse

### Architecture Sécurisée

```python
class SafetyArchitecture:
    """
    Couches de sécurité multiples
    
    1. Randomisation comportementale
    2. Détection d'anomalies  
    3. Respect limites temporelles
    4. Simulation comportement humain
    5. Monitoring pattern detection
    6. Circuit breakers automatiques
    """
```

### Gestion d'Erreurs

```python
# Stratégies de récupération par niveau
ERROR_RECOVERY = {
    'MODULE_ERROR': [
        'retry_operation',
        'reset_module_state', 
        'restart_module',
        'disable_module'
    ],
    'SYSTEM_ERROR': [
        'clear_cache',
        'reload_configuration',
        'restart_engine',
        'safe_shutdown'
    ],
    'GAME_ERROR': [
        'retry_action',
        'change_strategy',
        'pause_and_wait',
        'emergency_logout'
    ]
}
```

## 🚀 Extensibilité et API

### API pour Développeurs

```python
# Interface simplifiée pour nouveaux modules
class CustomModule(IGameModule):
    def __init__(self):
        super().__init__("custom_module")
        
    def initialize(self, config):
        # Configuration personnalisée
        return True
        
    def get_available_actions(self, game_state):
        # Actions possibles dans l'état actuel
        return []
        
    def execute_action(self, action):
        # Exécution d'action spécifique
        return True
        
    # Méthodes automatiques héritées :
    # - update()
    # - handle_event() 
    # - get_state()
    # - cleanup()
```

### Plugin System (Futur)

```python
# Architecture plugin planifiée
class PluginManager:
    def load_plugin(self, plugin_path):
        """Charge un plugin externe"""
        
    def validate_plugin(self, plugin):
        """Valide sécurité et compatibilité"""
        
    def integrate_plugin(self, plugin):
        """Intègre dans l'écosystème"""
```

## 📈 Scalabilité

### Performance Design

- **Threading** : Tâches longues sur threads séparés
- **Caching** : Cache intelligent pour templates et paths
- **Lazy Loading** : Chargement à la demande des ressources
- **Memory Management** : Nettoyage automatique mémoire
- **CPU Optimization** : Optimisation cycles critiques

### Évolutivité Architecture

- **Modularité** : Ajout facile nouveaux modules
- **Configuration** : Paramétrage sans code
- **API Standardisée** : Interface cohérente
- **Event-Driven** : Communication découplée
- **Monitoring** : Observabilité complète

---

Cette architecture permet à TacticalBot d'être à la fois puissant, flexible et maintenable, tout en offrant les performances nécessaires pour un botting efficace et sécurisé.