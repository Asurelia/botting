# 🧩 Guide Complet des Modules - TacticalBot

## 📋 Vue d'Ensemble

TacticalBot est composé de modules spécialisés qui fonctionnent ensemble pour créer un système de botting intelligent et robuste. Chaque module a une responsabilité spécifique et communique avec les autres via le système d'événements.

## 🏗️ Types de Modules

### 🔧 Modules Core (Système)
Modules fondamentaux nécessaires au fonctionnement du bot.

### 🎯 Modules Intelligence 
Modules d'IA et de prise de décision.

### 🎮 Modules Gameplay
Modules d'interaction avec le jeu.

### 👁️ Modules Vision
Modules d'analyse et reconnaissance visuelle.

### 🤖 Modules Automation
Modules d'automatisation des tâches.

---

## 🔧 Modules Core

### Engine Core
**Fichier** : `engine/core.py`  
**Type** : Module système critique

Le cœur du système qui orchestre tous les autres modules.

#### Fonctionnalités
- **Gestion du cycle de vie** des modules
- **Boucle principale** à 30 FPS
- **Monitoring** des performances
- **Gestion d'erreurs** et récupération automatique
- **Coordination** via le bus d'événements

#### Configuration
```json
{
  "engine": {
    "target_fps": 30,
    "decision_fps": 10,
    "max_modules": 50,
    "auto_recovery": true,
    "safety_checks": true,
    "performance_monitoring": true
  }
}
```

#### Exemple d'utilisation
```python
from engine.core import BotEngine, EngineConfig

# Configuration personnalisée
config = EngineConfig(
    target_fps=30,
    decision_fps=10,
    enable_logging=True,
    log_level="INFO"
)

# Initialisation
bot = BotEngine(config)

# Ajout de modules
bot.register_module(profession_manager, dependencies=["state_manager"])
bot.register_module(decision_engine, dependencies=[])

# Démarrage
if bot.initialize():
    bot.start()
```

### Event Bus
**Fichier** : `engine/event_bus.py`  
**Type** : Module système critique

Système de communication inter-modules via événements.

#### Types d'événements supportés
- `MODULE_ERROR` : Erreur dans un module
- `STATE_CHANGED` : Changement d'état du jeu
- `ACTION_COMPLETED` : Action terminée
- `COMBAT_STARTED` : Début de combat
- `RESOURCE_FOUND` : Ressource détectée
- `SAFETY_ALERT` : Alerte de sécurité
- `CONFIG_CHANGED` : Configuration modifiée
- `PERFORMANCE_WARNING` : Alerte performance

#### Exemple d'utilisation
```python
from engine.event_bus import EventBus, Event, EventType, EventPriority

# S'abonner à des événements
event_bus.subscribe(
    subscriber_id="mon_module",
    event_types={EventType.RESOURCE_FOUND, EventType.COMBAT_STARTED},
    handler=self.handle_event
)

# Publier un événement
event_bus.publish_immediate(
    EventType.RESOURCE_FOUND,
    {"resource_type": "wheat", "position": (100, 200)},
    sender="profession_farmer",
    priority=EventPriority.HIGH
)
```

### State Manager
**Fichier** : `state/state_tracker.py`  
**Type** : Module système important

Gestion de l'état complet du jeu en temps réel.

#### État suivi
```python
class GameState:
    # État personnage
    character: CharacterState
    health_percent: float
    mana_percent: float
    level: int
    position: Position
    
    # Interface
    windows_open: List[str]
    inventory_slots: List[Item]
    spells_available: List[Spell]
    
    # Monde
    current_map: str
    nearby_entities: List[Entity]
    resources_visible: List[Resource]
    threats_detected: List[Threat]
```

---

## 🎯 Modules Intelligence

### Decision Engine
**Fichier** : `modules/decision/decision_engine.py`  
**Type** : Module intelligence critique

Moteur de décision intelligent multi-critères avec apprentissage.

#### Fonctionnalités
- **Évaluation multi-critères** avec pondération
- **Gestion des priorités** (Critique > Élevée > Moyenne > Faible)
- **Apprentissage** des performances des décisions
- **Résolution de conflits** entre modules
- **Explications** des recommandations

#### Critères d'évaluation
- **Priorité** : Importance de l'action
- **Confiance** : Probabilité de succès
- **Durée estimée** : Temps d'exécution
- **Niveau de risque** : Risques associés
- **Récompense** : Gains attendus
- **Contexte** : Situation actuelle

#### Exemple d'utilisation
```python
from modules.decision import DecisionEngine, DecisionContext, Decision, Priority

# Initialisation
engine = DecisionEngine()

# Contexte actuel
context = DecisionContext(
    health_percent=75.0,
    mana_percent=80.0,
    in_combat=False,
    current_objective="farm_cereals"
)

# Actions possibles
decisions = [
    Decision(
        action_id="harvest_wheat",
        action_type=ActionType.PROFESSION,
        priority=Priority.MEDIUM,
        confidence=0.9,
        estimated_duration=15.0,
        success_probability=0.95,
        risk_level=0.1,
        reward_estimate=0.7
    )
]

# Meilleure décision
best_decision = engine.make_decision(decisions, context)
print(f"Action choisie: {best_decision.action_id}")

# Explication
explanation = engine.explain_decision(best_decision, context)
print(f"Raison: {explanation}")
```

### Strategy Selector
**Fichier** : `modules/decision/strategy_selector.py`  
**Type** : Module intelligence important

Sélection adaptative de stratégies selon la situation.

#### Stratégies disponibles
1. **AGGRESSIVE** : Maximise gains, accepte risques
2. **DEFENSIVE** : Priorité sécurité, minimise risques
3. **BALANCED** : Équilibre optimal gain/sécurité
4. **EFFICIENT** : Optimise le temps et l'efficacité
5. **STEALTH** : Évite la détection, discrétion
6. **SOCIAL** : Privilégie interactions coopératives

#### Situations détectées
- `PEACEFUL_FARMING` : Farm tranquille
- `DANGEROUS_AREA` : Zone à risques
- `CROWDED_AREA` : Zone bondée
- `DUNGEON_EXPLORATION` : Exploration donjon
- `PVP_ZONE` : Zone de combat PvP
- `RESOURCE_COMPETITION` : Compétition ressources
- `BOSS_FIGHT` : Combat de boss
- `LOW_RESOURCES` : Ressources faibles
- `INVENTORY_FULL` : Inventaire plein
- `MISSION_CRITICAL` : Mission critique

#### Exemple d'utilisation
```python
from modules.decision.strategy_selector import StrategySelector, StrategyType

selector = StrategySelector()

# Sélection automatique
strategy, config = selector.select_strategy(context)
print(f"Stratégie: {strategy.value}")

# Recommandations détaillées
recommendations = selector.get_strategy_recommendations(context, top_n=3)
for strategy, score, explanation in recommendations:
    print(f"{strategy.value}: {score:.2f} - {explanation}")

# Mise à jour des performances
selector.update_strategy_outcome(
    strategy_type=StrategyType.EFFICIENT,
    success=True,
    reward=0.8,
    duration=120.0
)
```

### Combat AI
**Fichier** : `modules/combat/ai/combat_ai.py`  
**Type** : Module intelligence spécialisé

Intelligence de combat avancée avec support de classes.

#### Classes supportées
- **Cra** : Archer à distance, focus dégâts
- **Iop** : Guerrier mêlée, forte attaque
- **Eniripsa** : Soigneur, support équipe

#### Fonctionnalités
- **Analyse tactique** des situations
- **Priorisation des cibles** intelligente
- **Gestion des sorts** et cooldowns
- **Adaptation** selon la classe
- **Coordination équipe** (si groupe)

#### Exemple d'utilisation
```python
from modules.combat.ai.combat_ai import CombatAI
from modules.combat.classes.cra import CraClass

# Initialisation pour un Cra
combat_ai = CombatAI()
combat_ai.set_character_class(CraClass())

# Analyse de combat
def update(self, game_state):
    if game_state.in_combat:
        # Analyser la situation
        situation = combat_ai.analyze_combat_situation(game_state)
        
        # Obtenir la meilleure action
        best_action = combat_ai.get_best_combat_action(situation)
        
        # Exécuter
        if best_action:
            return {"suggested_action": best_action}
```

---

## 🎮 Modules Gameplay

### Profession Manager
**Fichier** : `modules/professions/profession_manager.py`  
**Type** : Module gameplay principal

Gestion complète des 4 métiers avec optimisation automatique.

#### Métiers supportés
1. **Farmer** (29 ressources) : Céréales, légumes, fruits, plantes
2. **Lumberjack** (25 arbres) : Bois communs, nobles, exotiques
3. **Miner** (30 minerais) : Surface, mines profondes, gemmes
4. **Alchemist** (16 recettes) : Potions de soin, amélioration, rares

#### Stratégies d'optimisation
- **BALANCED** : Équilibre entre tous les métiers
- **XP_FOCUSED** : Maximise l'expérience globale
- **PROFIT_FOCUSED** : Maximise les profits
- **LEVELING** : Rattrapage du métier le plus faible
- **SYNERGY** : Exploite les synergies inter-métiers

#### Exemple d'utilisation
```python
from modules.professions import ProfessionManager, OptimizationStrategy

# Initialisation
manager = ProfessionManager()

# Optimisation session 4h pour profit
session = manager.optimize_global_session(
    duration_hours=4.0,
    strategy=OptimizationStrategy.PROFIT_FOCUSED
)

print(f"Session optimisée:")
for profession_id, time_allocation in session.profession_allocation.items():
    print(f"- {profession_id}: {time_allocation:.1f}h")

# Exécution
results = manager.execute_session(session)
print(f"Résultats: {results['totals']['total_kamas_gained']:,} kamas")

# Métier spécifique
farmer = manager.get_profession("farmer")
optimal_route = farmer.get_optimal_route((1, 30))  # Niveaux 1-30
profitability = farmer.calculate_profitability('ble')
print(f"Blé: {profitability['kamas_per_hour']:.0f} kamas/h")
```

### Navigation System
**Fichier** : `modules/navigation/pathfinding.py`  
**Type** : Module gameplay important

Système de navigation intelligent avec pathfinding A*.

#### Fonctionnalités
- **Pathfinding A*** optimisé
- **Évitement d'obstacles** dynamiques
- **Gestion zones dangereuses**
- **Cache chemins** fréquents
- **Optimisation trajets** multi-points

#### Exemple d'utilisation
```python
from modules.navigation import NavigationSystem, Position

nav = NavigationSystem()

# Navigation simple
start = Position(100, 100)
goal = Position(500, 300)
path = nav.find_path(start, goal)

# Navigation avec contraintes
constraints = {
    'avoid_areas': [(200, 200, 50)],  # Zone à éviter (x, y, rayon)
    'safe_only': True,
    'max_distance': 1000
}
safe_path = nav.find_path(start, goal, constraints)

# Navigation multi-points (tournée)
waypoints = [Position(100, 100), Position(200, 200), Position(300, 150)]
tour = nav.optimize_tour(waypoints)
```

### Economy Manager
**Fichier** : `modules/economy/market_analyzer.py`  
**Type** : Module gameplay spécialisé

Gestion économique avancée avec analyse de marché.

#### Fonctionnalités
- **Analyse des prix** marché
- **Optimisation crafting** avec ROI
- **Gestion automatique** inventaire
- **Prédiction tendances** économiques
- **Stratégies d'investissement**

#### Exemple d'utilisation
```python
from modules.economy import EconomyManager, MarketAnalyzer

economy = EconomyManager()
market = MarketAnalyzer()

# Analyse du marché
wheat_analysis = market.analyze_item("ble")
print(f"Prix moyen blé: {wheat_analysis['average_price']} kamas")
print(f"Tendance: {wheat_analysis['trend']}")

# Recommandations d'investissement
investments = economy.get_investment_recommendations()
for item, recommendation in investments.items():
    print(f"{item}: {recommendation['action']} - ROI: {recommendation['roi']:.1%}")

# Optimisation inventaire
inventory_actions = economy.optimize_inventory(current_inventory)
```

---

## 👁️ Modules Vision

### Screen Analyzer
**Fichier** : `modules/vision/screen_analyzer.py`  
**Type** : Module vision critique

Analyse et reconnaissance d'écran en temps réel.

#### Fonctionnalités
- **Capture d'écran** optimisée
- **Préprocessing** intelligent
- **OCR** reconnaissance texte
- **Détection d'éléments** UI
- **Extraction d'informations** de jeu

#### Exemple d'utilisation
```python
from modules.vision.screen_analyzer import ScreenAnalyzer

analyzer = ScreenAnalyzer()

# Analyse complète de l'écran
screen_data = analyzer.analyze_full_screen()
print(f"Santé: {screen_data['health_percent']}%")
print(f"Position: {screen_data['position']}")

# Recherche d'élément spécifique
health_bar = analyzer.find_element("health_bar")
if health_bar:
    health_value = analyzer.extract_health_value(health_bar)

# OCR sur zone spécifique
text_region = (100, 50, 200, 30)  # x, y, width, height
extracted_text = analyzer.ocr_region(text_region)
```

### Template Matcher
**Fichier** : `modules/vision/template_matcher.py`  
**Type** : Module vision spécialisé

Reconnaissance de patterns et templates visuels.

#### Templates supportés
- **Ressources** : Tous types de ressources des métiers
- **UI Elements** : Boutons, fenêtres, barres
- **Entités** : Monstres, PNJ, joueurs
- **Items** : Équipements, consommables
- **Spells** : Sorts et compétences

#### Exemple d'utilisation
```python
from modules.vision.template_matcher import TemplateMatcher

matcher = TemplateMatcher()

# Recherche de ressource
wheat_locations = matcher.find_template("wheat", confidence=0.8)
for location in wheat_locations:
    print(f"Blé trouvé en {location.position} (confiance: {location.confidence})")

# Template adaptatif
matcher.enable_adaptive_templates(True)
matcher.train_template("new_resource", sample_image)

# Recherche multiple
resources = ["wheat", "barley", "corn"]
found_resources = matcher.find_multiple_templates(resources)
```

---

## 🤖 Modules Automation

### Daily Routine
**Fichier** : `modules/automation/daily_routine.py`  
**Type** : Module automation utile

Automatisation des tâches quotidiennes répétitives.

#### Tâches supportées
- **Connexion quotidienne** : Login automatique
- **Récompenses quotidiennes** : Collecte récompenses
- **Quêtes quotidiennes** : Achèvement automatique
- **Maintenance inventaire** : Tri et nettoyage
- **Vérifications sécurité** : État du compte

#### Exemple d'utilisation
```python
from modules.automation.daily_routine import DailyRoutineManager

routine = DailyRoutineManager()

# Configuration des tâches quotidiennes
routine.configure_daily_tasks({
    "collect_daily_rewards": {"enabled": True, "priority": 1},
    "daily_quests": {"enabled": True, "max_quests": 5},
    "inventory_maintenance": {"enabled": True, "frequency": "weekly"},
    "security_checks": {"enabled": True, "frequency": "daily"}
})

# Exécution automatique
routine.execute_daily_routine()

# Planification
routine.schedule_routine(time="08:00", days=["monday", "wednesday", "friday"])
```

### Leveling Automation
**Fichier** : `modules/automation/leveling_automation.py`  
**Type** : Module automation spécialisé

Automatisation du level avec optimisation XP.

#### Stratégies de level
- **FAST_XP** : Maximum XP/heure
- **SAFE_LEVELING** : Level sécurisé
- **BALANCED_GROWTH** : Croissance équilibrée
- **SKILL_FOCUSED** : Focus sur compétences spécifiques

#### Exemple d'utilisation
```python
from modules.automation.leveling_automation import LevelingAutomation

leveling = LevelingAutomation()

# Configuration
leveling.set_target_level(50)
leveling.set_strategy("FAST_XP")

# Optimisation automatique
optimal_activities = leveling.get_optimal_leveling_activities(
    current_level=25,
    target_level=35,
    time_available=3.0  # 3 heures
)

# Exécution
for activity in optimal_activities:
    result = leveling.execute_activity(activity)
    print(f"Activité {activity.name}: {result['xp_gained']} XP")
```

### Quest Automation
**Fichier** : `modules/automation/quest_automation.py`  
**Type** : Module automation avancé

Gestion automatique des quêtes avec plus de 50 types supportés.

#### Types de quêtes
- **Collecte** : Ramassage d'items/ressources
- **Élimination** : Tuer des monstres
- **Livraison** : Transport d'objets
- **Exploration** : Découverte de zones
- **Craft** : Création d'objets
- **Social** : Interactions avec PNJ/joueurs

#### Exemple d'utilisation
```python
from modules.automation.quest_automation import QuestAutomation

quest_system = QuestAutomation()

# Analyse des quêtes disponibles
available_quests = quest_system.scan_available_quests()
print(f"{len(available_quests)} quêtes disponibles")

# Optimisation automatique
optimal_quests = quest_system.optimize_quest_sequence(
    available_quests,
    max_duration=2.0,  # 2 heures max
    strategy="XP_FOCUSED"
)

# Exécution automatique
for quest in optimal_quests:
    result = quest_system.execute_quest(quest)
    if result['completed']:
        print(f"Quête '{quest.name}' terminée: {result['rewards']}")
```

---

## 🛡️ Modules Sécurité

### Safety Manager
**Fichier** : `modules/safety/detection_avoidance.py`  
**Type** : Module sécurité critique

Évitement de détection avec comportement humain-like.

#### Fonctionnalités sécurité
- **Randomisation** des timings et patterns
- **Comportement humain** avec pauses réalistes
- **Détection d'anomalies** et arrêt automatique
- **Monitoring** des patterns suspects
- **Circuit breakers** de sécurité

#### Exemple d'utilisation
```python
from modules.safety import SafetyManager, HumanBehavior

safety = SafetyManager()

# Configuration du comportement humain
safety.configure_human_behavior({
    "randomization_level": 0.7,  # 70% de randomisation
    "pause_frequency": 0.1,      # Pause 10% du temps
    "break_duration": (30, 300), # Pauses 30s à 5min
    "typing_speed": (80, 120),   # 80-120 CPM
    "reaction_time": (150, 400)  # 150-400ms
})

# Vérifications automatiques
safety.enable_auto_checks({
    "pattern_detection": True,
    "timing_analysis": True,
    "behavior_deviation": True,
    "session_limits": True
})

# Dans la boucle principale
if safety.should_take_break():
    break_duration = safety.calculate_break_duration()
    safety.take_human_break(break_duration)
```

### Session Manager  
**Fichier** : `modules/safety/session_manager.py`  
**Type** : Module sécurité important

Gestion des sessions avec limites de sécurité.

#### Limites configurables
- **Durée maximale** : 4-6h par session
- **Pauses obligatoires** : Toutes les 1-2h
- **Sessions quotidiennes** : Maximum 8-12h/jour
- **Détection fatigue** : Réduction performances

#### Exemple d'utilisation
```python
from modules.safety.session_manager import SessionManager

session = SessionManager()

# Configuration des limites
session.configure_limits({
    "max_session_duration": 4.0,    # 4h max
    "mandatory_break_interval": 1.5, # Pause toutes les 1.5h
    "min_break_duration": 0.25,     # Pause min 15min
    "daily_limit": 8.0              # 8h max/jour
})

# Monitoring automatique
if session.should_end_session():
    reason = session.get_end_reason()
    print(f"Fin de session: {reason}")
    session.end_session()

# Statistiques
stats = session.get_session_stats()
print(f"Temps joué: {stats['active_time']:.1f}h")
print(f"Efficacité: {stats['efficiency']:.1%}")
```

---

## 🔧 API Module Developer

### Créer un Nouveau Module

#### 1. Interface de Base
```python
from engine.module_interface import IModule, ModuleStatus

class MonNouveauModule(IModule):
    def __init__(self):
        super().__init__("mon_nouveau_module")
        self.config = {}
        self.state = {}
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialisation du module"""
        self.config = config
        self.status = ModuleStatus.ACTIVE
        return True
        
    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Mise à jour principale (appelée à 30 FPS)"""
        # Logique principale du module
        return self.get_shared_data()
        
    def handle_event(self, event: Any) -> bool:
        """Gestion des événements du système"""
        if event.type == EventType.RESOURCE_FOUND:
            # Traiter l'événement
            return True
        return False
        
    def get_state(self) -> Dict[str, Any]:
        """État actuel du module"""
        return {
            "status": self.status.value,
            "config": self.config,
            "internal_state": self.state
        }
        
    def cleanup(self) -> None:
        """Nettoyage avant arrêt"""
        # Libérer les ressources
        pass
```

#### 2. Module Gameplay (Actions de jeu)
```python
from engine.module_interface import IGameModule

class MonModuleGameplay(IGameModule):
    def execute_action(self, action: Any) -> bool:
        """Exécute une action dans le jeu"""
        try:
            # Exécuter l'action
            success = self.perform_game_action(action)
            return success
        except Exception as e:
            self.set_error(str(e))
            return False
            
    def get_available_actions(self, game_state: Any) -> List[Any]:
        """Retourne les actions possibles"""
        actions = []
        
        # Analyser l'état et générer les actions
        if self.can_perform_action("harvest", game_state):
            actions.append({"type": "harvest", "target": "wheat"})
            
        return actions
```

#### 3. Module Analysis (Sans actions)
```python
from engine.module_interface import IAnalysisModule

class MonModuleAnalyse(IAnalysisModule):
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyse des données sans action directe"""
        analysis_result = {
            "threats_detected": self.detect_threats(data),
            "opportunities": self.find_opportunities(data),
            "recommendations": self.generate_recommendations(data)
        }
        return analysis_result
```

#### 4. Enregistrement du Module
```python
# Dans le script principal
from engine.core import BotEngine

# Créer et enregistrer le module
mon_module = MonNouveauModule()
bot = BotEngine()

# Enregistrer avec dépendances optionnelles
bot.register_module(
    mon_module, 
    dependencies=["state_manager", "vision_system"]
)

# Initialiser et démarrer
if bot.initialize():
    bot.start()
```

### Bonnes Pratiques Modules

#### Configuration
```python
# Fichier config/mon_module.json
{
  "enabled": true,
  "priority": 5,
  "update_frequency": 30,
  "settings": {
    "param1": "value1",
    "param2": 42
  },
  "safety": {
    "max_errors": 10,
    "timeout_seconds": 30
  }
}
```

#### Logging
```python
import logging

class MonModule(IModule):
    def __init__(self):
        super().__init__("mon_module")
        self.logger = logging.getLogger(f"modules.{self.name}")
        
    def update(self, game_state):
        self.logger.debug("Mise à jour du module")
        try:
            # Logique du module
            pass
        except Exception as e:
            self.logger.error(f"Erreur dans update: {e}")
            self.set_error(str(e))
```

#### Métriques
```python
def get_metrics(self) -> Dict[str, Any]:
    """Métriques personnalisées du module"""
    return {
        "actions_executed": self.action_count,
        "success_rate": self.calculate_success_rate(),
        "average_duration": self.get_average_duration(),
        "last_error": self.get_last_error()
    }
```

---

Cette documentation couvre tous les modules principaux de TacticalBot avec des exemples pratiques d'utilisation. Chaque module peut être utilisé indépendamment ou en coordination avec les autres pour créer un système de botting complet et intelligent.