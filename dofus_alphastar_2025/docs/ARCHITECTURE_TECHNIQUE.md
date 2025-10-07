# 📐 ARCHITECTURE TECHNIQUE - DOFUS AlphaStar 2025

**Date**: 2025-09-30
**Version**: 1.0.0
**Auteur**: Bot Development Team

---

## 📋 Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Architecture Globale](#architecture-globale)
3. [Modules Core](#modules-core)
4. [Systèmes Principaux](#systèmes-principaux)
5. [Flux de Données](#flux-de-données)
6. [Sécurité](#sécurité)
7. [Performance](#performance)
8. [API Reference](#api-reference)

---

## 🎯 Vue d'Ensemble

DOFUS AlphaStar 2025 est un bot intelligent inspiré de l'architecture AlphaStar de DeepMind, adapté pour le MMORPG Dofus Unity.

### Principes Architecturaux

1. **Modularité**: Chaque système est indépendant et remplaçable
2. **Sécurité First**: Mode observation obligatoire par défaut
3. **Pas d'Injection**: Utilise UNIQUEMENT screen/keyboard/mouse (NO packets, NO RAM)
4. **AI-Driven**: Utilise Deep RL (PPO, IMPALA) + HRM reasoning
5. **Vision-Based**: SAM 2 pour segmentation + TrOCR pour OCR

### Technologies Clés

- **Python 3.10+**
- **PyTorch 2.x** (Deep Learning)
- **OpenCV** (Vision)
- **NetworkX** (Pathfinding)
- **PyAutoGUI** (Input/Output)
- **AMD ROCm / DirectML** (GPU Acceleration)

---

## 🏗️ Architecture Globale

```
┌─────────────────────────────────────────────────────────────┐
│                     UI LAYER (Tkinter)                      │
│                      launch_ui_integrated.py                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    BRIDGE LAYER                             │
│                    core/ui_bridge.py                        │
│  - Communication UI <-> Core                                │
│  - État global (UIState)                                    │
│  - Callbacks & Events                                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    CORE SYSTEMS                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Calibration │  │ Map System  │  │  Safety     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ External    │  │   Vision    │  │     HRM     │        │
│  │    Data     │  │   Engine    │  │  Reasoning  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐                         │
│  │     RL      │  │ AlphaStar   │                         │
│  │  Training   │  │   Engine    │                         │
│  └─────────────┘  └─────────────┘                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              INTERACTION LAYER                              │
│  - PyAutoGUI (Screen Capture, Mouse, Keyboard)              │
│  - NO Packet Reading                                        │
│  - NO RAM Access                                            │
│  - NO Code Injection                                        │
└─────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  DOFUS UNITY CLIENT                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 Modules Core

### 1. Calibration System (`core/calibration/`)

**Objectif**: Auto-découverte des éléments UI de Dofus

#### Structure
```
calibration/
├── __init__.py              # Exports
├── dofus_calibrator.py      # Calibrateur principal
├── config/
│   └── dofus_knowledge.json # Base de connaissances générée
```

#### Processus de Calibration (6 phases)

1. **Détection fenêtre** (`detect_window()`)
   - Trouve fenêtre Dofus via pygetwindow
   - Récupère dimensions et position

2. **Mapping UI** (`map_ui_elements()`)
   - Détecte HP bar, PA bar, minimap
   - OCR pour identifier éléments

3. **Découverte raccourcis** (`discover_shortcuts()`)
   - Teste combinaisons clavier courantes
   - Valide avec screen capture

4. **Scan interactifs** (`scan_interactive_elements()`)
   - Identifie NPCs, ressources, zaaps
   - Segmentation avec SAM 2

5. **Analyse options** (`scan_game_options()`)
   - Parse menu options du jeu
   - Extrait paramètres pertinents

6. **Build knowledge base** (`build_knowledge_base()`)
   - Compile toutes les données
   - Sauvegarde `dofus_knowledge.json`

#### API

```python
from core.calibration import create_calibrator

calibrator = create_calibrator()
result = calibrator.run_full_calibration()

if result.success:
    print(f"UI Elements: {len(result.ui_elements)}")
    print(f"Shortcuts: {len(result.shortcuts)}")
    print(f"Duration: {result.duration_seconds}s")
```

---

### 2. Map System (`core/map_system/`)

**Objectif**: Navigation globale dans le monde Dofus (600+ maps)

#### Structure
```
map_system/
├── __init__.py           # Exports
├── map_graph.py          # Graph NetworkX des maps
├── map_discovery.py      # Auto-découverte progressive
```

#### Architecture du Graph

```python
@dataclass(frozen=True)
class MapCoords:
    x: int  # Coordonnée X
    y: int  # Coordonnée Y

class MapGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # NetworkX directed graph
        self.maps: Dict[MapCoords, Dict] = {}
        self.discovered_maps: Set[MapCoords] = set()
```

#### Pathfinding

```python
# Utilise A* de NetworkX
path = graph.find_path(
    from_coords=MapCoords(5, -18),  # Astrub
    to_coords=MapCoords(-3, 4),     # Bonta
    avoid_pvp=True,
    use_zaaps=True
)
```

#### Découverte Progressive

Le système apprend les maps au fur et à mesure :

```python
discovery = create_map_discovery()

# Pendant exploration
discovered_map = discovery.discover_current_map()

# Ajoute automatiquement au graph
graph.add_discovered_map(discovered_map)
```

---

### 3. Safety System (`core/safety/`)

**Objectif**: SYSTÈME CRITIQUE - Empêche actions non désirées

#### ObservationMode

**Le composant LE PLUS IMPORTANT du bot.**

```python
class ObservationMode:
    def intercept_action(
        self,
        action_type: str,
        action_details: Dict,
        game_state: Dict,
        reason: str
    ) -> Optional[Any]:
        """
        Intercepte TOUTES les actions

        Returns:
            None si observation activé (bloque)
            action_details sinon (laisse passer)
        """
        if self.enabled:
            # MODE OBSERVATION: Bloque l'action
            self.observations.append(log)
            return None  # Action NOT executed

        return action_details  # Normal mode
```

#### Utilisation

```python
from core.safety import create_observation_mode

obs = create_observation_mode(auto_enabled=True)  # SÉCURITÉ!

# Dans la boucle du bot
action = {
    'type': 'mouse_click',
    'position': (100, 200)
}

# Intercepte l'action
result = obs.intercept_action(
    action_type='mouse_click',
    action_details=action,
    game_state={'hp': 100},
    reason='Collecte ressource'
)

if result is None:
    # Action bloquée, ne rien faire
    pass
else:
    # Mode normal, exécuter l'action
    pyautogui.click(result['position'])
```

#### Analyse & Recommandations

```python
# Après session d'observation
obs.save_observations()

analysis = obs.analyze_observations()

print(f"Safety Score: {analysis['safety_score']}/100")
print(f"Recommendations: {analysis['recommendations']}")
```

---

### 4. External Data (`core/external_data/`)

**Objectif**: Accès aux données du jeu via API externe

#### DofusDB Client

```python
from core.external_data import create_dofusdb_client

client = create_dofusdb_client()

# Recherche item
items = client.search_items("Dofus Émeraude")
for item in items:
    print(f"{item.name} - Level {item.level}")

# Get spell
spell = client.get_spell(spell_id=123)

# Calcul dégâts
damage = client.get_spell_damage(
    spell_id=123,
    level=5,
    target_resistances={'fire': 20}
)
```

#### Cache Intelligent

- **Mémoire**: Cache RAM (rapide)
- **Disque**: Cache `cache/dofusdb/` (persistant)
- **Fallback**: Si API offline, utilise cache

```python
# Stats du client
stats = client.get_stats()
print(f"Requests: {stats['requests']}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache ratio: {stats['cache_ratio']}")
```

---

### 5. Vision Engine (`core/vision_engine_v2/`)

**Objectif**: Vision par ordinateur de pointe

#### Technologies

- **SAM 2** (Segment Anything Model v2)
  - Segmentation d'objets
  - Détection NPCs, monstres, ressources

- **TrOCR** (Transformer-based OCR)
  - OCR haute précision
  - Lecture textes UI, dialogues, noms

#### Pipeline Vision

```
Screen Capture (PyAutoGUI)
         │
         ▼
Preprocessing (OpenCV)
         │
         ▼
SAM 2 Segmentation ────┐
         │             │
         ▼             ▼
Object Detection   Text Regions
         │             │
         │             ▼
         │        TrOCR OCR
         │             │
         └─────┬───────┘
               ▼
         Game State
```

---

### 6. HRM Reasoning (`core/hrm_reasoning/`)

**Objectif**: Raisonnement hiérarchique (System 1/System 2)

#### Architecture HRM

```
┌────────────────────────────────────┐
│         SYSTEM 2 (Slow)            │
│    Strategic Reasoning             │
│  - Long-term planning              │
│  - Quest strategy                  │
│  - Economic decisions              │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│         SYSTEM 1 (Fast)            │
│    Reactive Decisions              │
│  - Combat actions                  │
│  - Movement                        │
│  - Resource gathering              │
└────────────────────────────────────┘
```

---

### 7. RL Training (`core/rl_training/`)

**Objectif**: Entraînement Deep Reinforcement Learning

#### Algorithmes

- **PPO** (Proximal Policy Optimization)
- **IMPALA** (Importance Weighted Actor-Learner)
- **DQN** (Deep Q-Network) - backup

#### Environment

```python
class DofusEnv(gym.Env):
    """
    OpenAI Gym environment pour Dofus
    """

    def __init__(self):
        # Observation space: Screenshot + Game State
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(0, 255, (1080, 1920, 3)),
            'hp': spaces.Box(0, 100, (1,)),
            'pa': spaces.Box(0, 12, (1,)),
            # ...
        })

        # Action space: All possible actions
        self.action_space = spaces.Discrete(100)

    def step(self, action):
        # Execute action via ObservationMode
        # ...
        return obs, reward, done, info
```

---

## 🔄 Flux de Données

### 1. Démarrage du Bot

```
launch_ui_integrated.py
    │
    ├─> UIBridge.initialize_systems()
    │      │
    │      ├─> create_calibrator()
    │      ├─> create_map_graph()
    │      ├─> create_dofusdb_client()
    │      └─> create_observation_mode() [CRITIQUE!]
    │
    └─> UI Ready
```

### 2. Session d'Observation

```
User clicks "Start Bot"
    │
    ├─> UIBridge.start_bot(observation_only=True)
    │      │
    │      ├─> ObservationMode.enable()
    │      └─> Start monitoring_thread
    │
    ├─> Main loop:
    │      │
    │      ├─> Capture screen (PyAutoGUI)
    │      ├─> Vision analysis (SAM 2 + TrOCR)
    │      ├─> HRM decides action
    │      ├─> ObservationMode.intercept_action()
    │      │      └─> [BLOCKED] Action logged, not executed
    │      └─> Update UI stats
    │
    └─> User clicks "Stop Bot"
           │
           ├─> UIBridge.stop_bot()
           └─> ObservationMode.save_observations()
```

### 3. Analyse Post-Session

```
observations.json generated
    │
    ├─> ObservationMode.analyze_observations()
    │      │
    │      ├─> Compute safety_score
    │      ├─> Detect patterns
    │      └─> Generate recommendations
    │
    └─> User reviews:
           │
           ├─> Safety score > 70? → Safe to continue
           └─> Safety score < 70? → Adjust behavior
```

---

## 🔒 Sécurité

### Niveaux de Sécurité

#### Niveau 1: ObservationMode (OBLIGATOIRE)
- ✅ Activé par défaut
- ✅ Bloque 100% des actions
- ✅ Logs toutes les décisions
- ❌ Aucune interaction avec Dofus

#### Niveau 2: DryRunMode (À implémenter)
- ⚠️ Simule actions sans exécution réelle
- ⚠️ Valide séquences complexes
- ⚠️ Test algorithmes RL

#### Niveau 3: RealMode (DANGER!)
- ⛔ Désactive ObservationMode
- ⛔ Exécute actions réellement
- ⛔ UNIQUEMENT sur compte jetable
- ⛔ Ban quasi-garanti

### Règles de Sécurité

1. **JAMAIS sur compte principal**
2. **TOUJOURS observer avant d'agir**
3. **Limiter sessions à 5-10 minutes au début**
4. **Analyser safety_score systématiquement**
5. **Randomiser timings entre actions**

---

## ⚡ Performance

### Optimisations GPU

#### AMD 7800XT + ROCm

```python
import torch

# Vérifie ROCm
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

#### DirectML (Fallback)

```bash
pip install torch-directml
```

```python
import torch_directml

device = torch_directml.device()
model = model.to(device)
```

### Benchmarks Cibles

| Operation | Target | Actuel |
|-----------|--------|--------|
| Screen Capture | < 50ms | ✅ ~30ms |
| SAM 2 Segmentation | < 100ms | ⚠️ ~200ms |
| TrOCR OCR | < 80ms | ⚠️ ~150ms |
| HRM Decision | < 20ms | ✅ ~15ms |
| **Total Loop** | **< 250ms** | **⚠️ ~400ms** |

**FPS cible**: 4 FPS (acceptable pour MMORPG tour par tour)

---

## 📚 API Reference

### Core Classes

#### UIBridge

```python
class UIBridge:
    def initialize_systems() -> bool
    def start_bot(observation_only: bool = True) -> bool
    def stop_bot() -> bool
    def run_calibration() -> bool
    def toggle_observation_mode() -> bool
    def get_state() -> Dict[str, Any]
    def get_logs(limit: int = 100) -> List[Dict]
```

#### ObservationMode

```python
class ObservationMode:
    def __init__(log_file: str, auto_enabled: bool = True)
    def intercept_action(...) -> Optional[Any]
    def enable() -> None
    def disable() -> None  # DANGER!
    def is_enabled() -> bool
    def get_observations(limit: int = None) -> List[ObservationLog]
    def analyze_observations() -> Dict[str, Any]
    def save_observations(output_file: str = None) -> None
```

#### MapGraph

```python
class MapGraph:
    def __init__()
    def add_map(coords: MapCoords, ...) -> None
    def find_path(from_coords: MapCoords, to_coords: MapCoords, ...) -> List[MapCoords]
    def mark_discovered(coords: MapCoords) -> None
    def get_map_info(coords: MapCoords) -> Dict
```

#### DofusDBClient

```python
class DofusDBClient:
    def __init__(cache_dir: str, rate_limit_delay: float = 0.1)
    def get_item(item_id: int) -> Optional[ItemData]
    def get_spell(spell_id: int) -> Optional[SpellData]
    def get_monster(monster_id: int) -> Optional[MonsterData]
    def search_items(query: str, item_type: str = None, limit: int = 20) -> List[ItemData]
    def get_stats() -> Dict[str, int]
```

---

## 🎓 Best Practices

### Code Style

```python
# ✅ BON: Factory functions
from core.calibration import create_calibrator
calibrator = create_calibrator()

# ❌ MAUVAIS: Direct instantiation
from core.calibration.dofus_calibrator import DofusCalibrator
calibrator = DofusCalibrator()  # Éviter

# ✅ BON: Type hints
def process_action(action: Dict[str, Any]) -> bool:
    pass

# ✅ BON: Dataclasses pour structures
@dataclass
class GameState:
    hp: int
    pa: int
    position: Tuple[int, int]
```

### Error Handling

```python
# ✅ BON: Try-except avec logging
try:
    result = calibrator.run_full_calibration()
except Exception as e:
    logger.error(f"Calibration failed: {e}")
    # Fallback gracieux
    return False

# ✅ BON: Validation inputs
def find_path(from_coords: MapCoords, to_coords: MapCoords):
    if from_coords not in self.maps:
        raise ValueError(f"Unknown map: {from_coords}")
```

### Testing

```python
# ✅ BON: Unit tests avec mocks
@patch('core.calibration.dofus_calibrator.pyautogui')
def test_window_detection(mock_pyautogui):
    mock_window = Mock()
    mock_pyautogui.getWindowsWithTitle.return_value = [mock_window]

    calibrator = create_calibrator()
    result = calibrator.detect_window()

    assert result is True
```

---

## 📞 Support & Contribution

### Signaler un Bug

1. Vérifier que le bug n'est pas déjà connu
2. Créer une issue GitHub avec:
   - Description détaillée
   - Steps to reproduce
   - Logs pertinents
   - Configuration système

### Contribuer

1. Fork le repo
2. Créer branche feature
3. Commit avec messages descriptifs
4. Tests unitaires
5. Pull request avec description

---

**Documentation générée pour DOFUS AlphaStar 2025**
**Dernière mise à jour**: 2025-09-30

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>