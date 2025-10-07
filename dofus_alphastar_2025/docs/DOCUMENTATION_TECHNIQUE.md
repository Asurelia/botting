# 📚 DOCUMENTATION TECHNIQUE - DOFUS AlphaStar 2025

**Version:** 1.0.0
**Date:** Janvier 2025
**Auteur:** Système AlphaStar

---

## 📋 TABLE DES MATIÈRES

1. [Architecture Générale](#architecture-générale)
2. [Modules Principaux](#modules-principaux)
3. [Flux de Données](#flux-de-données)
4. [APIs et Interfaces](#apis-et-interfaces)
5. [Base de Données](#base-de-données)
6. [Système de Logging](#système-de-logging)
7. [Tests et Qualité](#tests-et-qualité)
8. [Déploiement](#déploiement)

---

## 🏗️ ARCHITECTURE GÉNÉRALE

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERFACE UTILISATEUR                     │
│          (ui/modern_app/ + ui/alphastar_dashboard.py)        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                     GAME ENGINE                              │
│              (core/game_loop/game_engine.py)                 │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   VISION     │  │    BRAIN     │  │   ACTIONS    │     │
│  │   ENGINE V2  │──┤  INTEGRATED  │──┤   SYSTEM     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    │ Vision  │          │ Decision│          │ Execution│
    │ Analysis│          │ Making  │          │ & Safety │
    └─────────┘          └─────────┘          └──────────┘
```

### Composants Principaux

1. **GameEngine** (`core/game_loop/game_engine.py:45`)
   - Orchestration centrale
   - Boucle de jeu à 5-30 FPS
   - Threading non-bloquant
   - Gestion états et statistiques

2. **Vision Engine V2** (`core/vision_engine_v2/`)
   - SAM 2 (Meta AI) pour segmentation
   - TrOCR (Microsoft) pour OCR
   - Détection multi-méthodes (YOLO + Template Matching)

3. **Integrated Brain** (`core/decision/autonomous_brain_integrated.py:27`)
   - 17 systèmes intégrés
   - Décision hiérarchique
   - HRM Reasoning (108M paramètres)

4. **Action System** (`core/actions/`)
   - Exécution actions avec validation
   - Safety checks intégrés
   - Queue de commandes

---

## 🧩 MODULES PRINCIPAUX

### 1. Vision System

#### Vision V1 (Stable)
**Localisation:** `core/vision/`

**Fonctionnalités:**
- Capture d'écran temps réel (win32 API)
- OCR Tesseract pour texte
- Template matching (OpenCV)
- Détection HP/PA/PM basique

**Exemple d'utilisation:**
```python
from core.vision import VisionOrchestrator

vision = VisionOrchestrator()
game_state = vision.extract_game_state()
print(f"HP: {game_state['character']['hp']}")
```

#### Vision V2 (Avancé)
**Localisation:** `core/vision_engine_v2/`

**Composants:**
- `vision_adapter.py` - Adaptateur principal
- `ocr_detector.py` - TrOCR intégration
- `realtime_vision.py` - Processing temps réel
- `vision_complete_adapter.py` - Adaptateur complet

**Configuration:**
```python
from core.vision_engine_v2 import create_vision_engine

vision = create_vision_engine()
frame = vision.capture_screen()
analysis = vision.analyze_frame(frame)

# Résultats:
# {
#   'character': {'hp': 450, 'max_hp': 500, 'pa': 6, 'pm': 3},
#   'combat': {'in_combat': True, 'my_turn': True},
#   'monsters': [{'type': 'Tofu', 'position': (350, 250)}]
# }
```

---

### 2. Combat Engine

**Localisation:** `core/combat/combat_engine.py`

**Classes Principales:**

#### CombatEngine (`combat_engine.py:102`)
Moteur de combat principal avec IA tactique.

**Méthodes clés:**
```python
def decide_action(self, combat_state: CombatState) -> Dict[str, Any]
```
Décide de l'action optimale selon:
- État HP du joueur (survie si < 30%)
- Positionnement tactique
- Sélection de cibles (4 stratégies)
- Combos optimaux vs sorts simples

**Exemple:**
```python
from core.combat import create_combat_engine, CharacterClass

combat = create_combat_engine(CharacterClass.IOP)

# Créer état de combat
player = create_player_entity("Player", hp=350, max_hp=500, pa=6, pm=3, position=(5, 5))
enemies = [create_enemy_entity("enemy_1", "Tofu", hp=100, max_hp=100, position=(8, 6))]
state = create_combat_state(player, enemies)

# Démarrer combat
combat.start_combat(state)

# Obtenir décision
decision = combat.decide_action(state)
# {'action_type': 'combo', 'combo_name': 'IOP_BURST', 'target_id': 'enemy_1'}

# Logger l'action
combat.log_action(decision, success=True, result={'damage_dealt': 150})
```

**Phases de combat:**
1. `PREPARATION` - Début de tour
2. `POSITIONING` - Déplacements tactiques
3. `BUFFING` - Application de buffs
4. `ATTACKING` - Attaques
5. `FINISHING` - Fin de tour

---

### 3. Economic System

#### Market Analyzer (`core/economy/market_analyzer.py`)

**Fonctionnalités:**
- Scan HDV automatique
- Prédictions ML des prix (LinearRegression + RandomForest)
- Détection arbitrage inter-serveurs
- Historique en SQLite

**Classes:**

##### MLPricePredictor (`market_analyzer.py:168`)
```python
predictor = MLPricePredictor()
predictor.train_model(item_id=1234, market_db=db)
prediction = predictor.predict_price(item_id=1234, market_db=db, horizon_days=7)

# Résultat:
# PricePrediction(
#   predicted_price=1250.0,
#   confidence_interval=(1100.0, 1400.0),
#   trend_direction='up',
#   accuracy_score=0.87
# )
```

##### ArbitrageDetector (`market_analyzer.py:320`)
Détecte opportunités de profit entre serveurs.

**Algorithme:**
1. Compare prix même item sur différents serveurs
2. Calcule frais transaction (2% par défaut)
3. Évalue risque (volatilité, quantité, âge données)
4. Score de confiance multi-facteurs

**Exemple:**
```python
detector = ArbitrageDetector(min_profit_margin=0.1, min_roi=0.05)
opportunities = detector.detect_opportunities(market_data)

for opp in opportunities:
    print(f"{opp.item_id}: acheter à {opp.buy_price} sur {opp.buy_server}")
    print(f"  → vendre à {opp.sell_price} sur {opp.sell_server}")
    print(f"  Profit: {opp.profit_margin} ({opp.roi_percentage}%)")
```

#### Crafting Optimizer (`core/economy/crafting_optimizer.py`)

**Composants:**

##### ProfitabilityCalculator (`crafting_optimizer.py:258`)
Calcule rentabilité des recettes.

**Métriques:**
- Profit par craft
- ROI %
- XP/heure
- Profit/heure
- Point mort (break-even)
- Score de risque
- Saturation marché

**Exemple:**
```python
calc = ProfitabilityCalculator()
calc.update_market_data({
    101: 15.0,  # Blé
    102: 5.0,   # Eau
    201: 80.0   # Pain
})

analysis = calc.calculate_recipe_profitability(pain_recipe)
# CraftAnalysis(
#   profit_per_craft=45.0,
#   roi_percentage=75.0,
#   profit_per_hour=3600.0,
#   xp_per_hour=10000,
#   confidence_score=0.85
# )
```

##### CraftQueue (`crafting_optimizer.py:428`)
Gestionnaire de queue avec priorités.

**Optimisation selon objectif:**
- `PROFIT` - Maximise profit/heure
- `XP` - Maximise XP/heure
- `TIME` - Minimise temps
- `BALANCED` - Équilibre
- `RESOURCES` - Optimise ressources

---

### 4. Navigation System

#### Map System (`core/map_system/`)

##### MapGraph (`map_graph.py`)
Graphe NetworkX pour navigation.

**Fonctionnalités:**
- Pathfinding A* (`pathfinding.py:18`)
- Découverte progressive (`map_discovery.py`)
- Graphe persistant

**Exemple:**
```python
from core.map_system import MapGraph

graph = MapGraph()
graph.add_map("astrub_center", coordinates=(0, 0))
graph.add_map("plains_001", coordinates=(1, 0))
graph.add_connection("astrub_center", "plains_001", "east", cost=10)

path = graph.find_path("astrub_center", "plains_001")
# ['astrub_center', 'plains_001']
```

#### Ganymede Navigator (`core/navigation_system/ganymede_navigator.py`)

Utilise base de données Ganymede pour navigation enrichie.

**Features:**
- Données monde complètes
- Zaaps et téléportations
- Routes optimisées
- Détection éléments carte (NPCs, ressources, monstres)

---

### 5. Profession System

**Localisation:** `core/professions/`

**Métiers supportés:**
- `farmer.py` - Farmer (céréales)
- `lumberjack.py` - Bûcheron (bois)
- `miner.py` - Mineur (minerais)
- `alchemist.py` - Alchimiste (plantes)

**Système avancé:** `core/professions_advanced/`
- `profession_synergies.py` - Synergies entre métiers
- `resource_optimizer.py` - Optimisation multi-métiers

**Exemple:**
```python
from core.professions import Lumberjack, ProfessionManager

lumberjack = Lumberjack(level=50, current_xp=125000)
manager = ProfessionManager()
manager.add_profession(lumberjack)

# Trouver meilleurs spots
spots = lumberjack.find_best_spots(player_level=30)
# [{'map_id': 'forest_west', 'resource': 'Frêne', 'efficiency': 8.5}]
```

---

### 6. Quest System

**Localisation:** `core/quest_system/`

**Components:**
- `quest_manager.py` - Gestion quêtes
- `dialogue_system.py` - Dialogues NPCs
- `inventory_manager.py` - Gestion inventaire

**Format quête (JSON):**
```json
{
  "quest_id": "tutorial_001",
  "objectives": [
    {
      "type": "dialogue",
      "target_npc": "Otomaï",
      "map_id": "incarnam_center"
    },
    {
      "type": "kill_monsters",
      "monster_type": "Larve Bleue",
      "quantity_required": 5
    }
  ]
}
```

---

### 7. Intelligence Systems

#### HRM Reasoning (`core/hrm_reasoning/hrm_amd_core.py`)

**Architecture:**
- 108M paramètres PyTorch
- System 1 (rapide) + System 2 (réflexion)
- Optimisé AMD GPU (DirectML)

**Exemple:**
```python
from core.hrm_reasoning import DofusHRMAgent

agent = DofusHRMAgent(device='amd')  # ou 'cuda', 'cpu'
decision = agent.reason(game_state)
```

#### Passive Intelligence (`core/intelligence/`)

**Composants:**
- `opportunity_manager.py` - Détection opportunités
- `passive_intelligence.py` - Apprentissage patterns
- `fatigue_simulator.py` - Simulation comportement humain

**Exemple simulation fatigue:**
```python
from core.intelligence import FatigueSimulator

sim = FatigueSimulator()
sim.simulate_session(duration_hours=2)

# Ajoute:
# - Erreurs aléatoires (1-5%)
# - Pauses réalistes
# - Ralentissement progressif
# - Décisions sous-optimales
```

---

## 🔄 FLUX DE DONNÉES

### Boucle de Jeu Principale

```python
# game_engine.py:202
def _main_loop(self):
    while self.running:
        # 1. UPDATE GAME STATE
        self._update_game_state()

        # 2. VISION → Analyse
        frame = self.vision_system.capture_screen()
        vision_data = self.vision_system.analyze_frame(frame)
        self._apply_vision_to_state(vision_data)

        # 3. BRAIN → Décision
        decision = self.brain.decide(self.game_state, vision_data)

        # 4. ACTION → Exécution
        if decision:
            self._execute_decision(decision)

        # 5. STATS & CALLBACKS
        if self.on_state_update:
            self.on_state_update(self.game_state)
```

### Flux Vision → Brain

```
Screen Capture
     ↓
SAM 2 Segmentation
     ↓
TrOCR Text Recognition
     ↓
Template Matching
     ↓
Game State Extraction
     ↓
Brain Decision (17 systems)
     ↓
Action Execution
```

---

## 📡 APIs ET INTERFACES

### Game Engine API

#### Initialisation
```python
from core.game_loop import create_game_engine
from core.combat import CharacterClass

engine = create_game_engine(
    target_fps=10,
    observation_mode=True
)

# Initialiser systèmes
engine.character_class = CharacterClass.IOP
engine.initialize_systems()
```

#### Contrôle
```python
# Démarrer
engine.start()

# Callbacks
def on_state_update(game_state):
    print(f"HP: {game_state.character.hp}")

engine.set_state_callback(on_state_update)

# Arrêter
engine.stop()
```

### Vision API

```python
# Vision V2
from core.vision_engine_v2 import create_vision_engine

vision = create_vision_engine()

# Capture
frame = vision.capture_screen()

# Analyse complète
analysis = vision.analyze_frame(frame)

# Détection spécifique
monsters = vision.detect_monsters(frame)
hp_value = vision.read_hp_bar(frame)
```

### Combat API

```python
from core.combat import create_combat_engine, CharacterClass

# Créer engine
combat = create_combat_engine(CharacterClass.CRA)

# Configurer
combat.config['target_priority'] = TargetPriority.LOWEST_HP
combat.config['min_hp_threshold'] = 25.0

# Utiliser
decision = combat.decide_action(combat_state)
combat.log_action(decision, success=True)
```

---

## 💾 BASE DE DONNÉES

### Market Data (SQLite)

**Tables:**

```sql
-- Prix historiques
CREATE TABLE price_history (
    id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL,
    server_id TEXT NOT NULL,
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    timestamp DATETIME NOT NULL
);

-- Prédictions
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL,
    predicted_price REAL NOT NULL,
    actual_price REAL,
    prediction_date DATETIME NOT NULL,
    target_date DATETIME NOT NULL,
    accuracy REAL
);

-- Arbitrage
CREATE TABLE arbitrage_opportunities (
    id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL,
    buy_server TEXT NOT NULL,
    sell_server TEXT NOT NULL,
    profit_margin REAL NOT NULL,
    detected_at DATETIME NOT NULL
);
```

### Crafting Data (SQLite)

```sql
-- Recettes
CREATE TABLE recipes (
    recipe_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    profession TEXT NOT NULL,
    level_required INTEGER NOT NULL,
    result_item_id INTEGER NOT NULL,
    xp_gained INTEGER
);

-- Ingrédients
CREATE TABLE recipe_ingredients (
    recipe_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL
);

-- Historique crafts
CREATE TABLE craft_history (
    id INTEGER PRIMARY KEY,
    recipe_id INTEGER NOT NULL,
    quantity_crafted INTEGER NOT NULL,
    actual_profit REAL,
    timestamp DATETIME NOT NULL
);
```

### Feedback Database (JSON)

**Localisation:** `data/feedback/decisions_feedback.json`

**Format:**
```json
[
  {
    "decision_id": "decision_20250101_123045_001",
    "timestamp": "2025-01-01T12:30:45",
    "action_type": "engage_monster",
    "reason": "Farm optimal target",
    "user_feedback": "correct",
    "user_comment": "Bonne décision, Tofu facile",
    "suggested_action": null
  }
]
```

---

## 📝 SYSTÈME DE LOGGING

### Niveaux de Log

```python
INFO     # Informations générales
WARNING  # Avertissements
ERROR    # Erreurs
SUCCESS  # Succès
DECISION # Décisions du bot
```

### Configuration Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler()
    ]
)
```

### Logs Temps Réel (UI)

Le panneau `LogsLearningPanel` affiche logs en temps réel:

```python
from ui.modern_app.logs_learning_panel import create_logs_learning_panel

panel = create_logs_learning_panel(parent_frame)

# Ajouter logs
panel.add_log("INFO", "Bot démarré")
panel.add_log("DECISION", "Engage Tofu à position (350, 250)")

# Ajouter décision
from ui.modern_app.logs_learning_panel import BotDecision

decision = BotDecision(
    timestamp=datetime.now(),
    decision_id="dec_001",
    action_type="engage_monster",
    reason="Farm optimal",
    details={'target': 'Tofu'},
    context={'hp': 450, 'pa': 6}
)
panel.add_decision(decision)
```

---

## ✅ TESTS ET QUALITÉ

### Tests Unitaires

**Localisation:** `tests/`

**Exécution:**
```bash
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_combat.py -v
pytest tests/test_map_system.py -v
```

**Couverture actuelle:** 95% (60/63 tests passing)

### Tests Principaux

```python
# tests/test_combat.py
def test_target_selection():
    """Test sélection de cible"""
    combat = create_combat_engine(CharacterClass.IOP)
    # ...

# tests/test_map_system.py
def test_pathfinding():
    """Test A* pathfinding"""
    graph = MapGraph()
    # ...

# tests/test_professions.py
def test_lumberjack_xp():
    """Test calcul XP bûcheron"""
    lumber = Lumberjack(level=50)
    # ...
```

---

## 🚀 DÉPLOIEMENT

### Prérequis Système

```
Python 3.9+
Windows 10/11 (pour win32 API)
GPU AMD RX 7800 XT (optionnel pour HRM)
16GB RAM minimum
```

### Installation

```bash
# Créer environnement
python -m venv venv
venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt

# Installer PyTorch AMD (optionnel)
pip install torch-directml
```

### Configuration

**Fichier:** `config/bot_config.yaml`

```yaml
game_engine:
  target_fps: 10
  observation_mode: true

vision:
  engine_version: 2  # 1 ou 2
  capture_method: "win32"

combat:
  character_class: "IOP"
  min_hp_threshold: 30.0

safety:
  observation_enabled: true
  max_session_duration: 3600  # 1 heure
```

### Lancement

```bash
# Mode observation (sécurisé)
python launch_autonomous_full.py --duration 30

# Avec calibration
python launch_autonomous_full.py --calibrate --duration 30

# Interface graphique
python launch_ui.py
```

---

## 📞 SUPPORT

Pour questions techniques:
- Consulter `GUIDE_DEMARRAGE.md`
- Vérifier `CHECK_UP_COMPLET.md` pour statut système
- Logs disponibles dans `logs/`

---

**Dernière mise à jour:** Janvier 2025
