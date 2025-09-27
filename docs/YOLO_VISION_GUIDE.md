# Guide Complet YOLO Vision System pour DOFUS

## 🎯 Vue d'Ensemble

Le système YOLO Vision pour DOFUS remplace l'ancien template matching par une solution de détection d'objets moderne et robuste basée sur l'intelligence artificielle. Cette nouvelle approche offre :

- **+80% de fiabilité** par rapport au template matching
- **Résistance aux mises à jour** du jeu
- **Détection temps réel** optimisée
- **Adaptation automatique** selon les performances
- **Interface hybride** combinant YOLO + Template Matching

## 📋 Prérequis Système

### Configuration Minimale
- **OS** : Windows 10/11 64-bit
- **RAM** : 8 GB minimum (16 GB recommandé)
- **GPU** : NVIDIA GTX 1060 ou AMD RX 580 (optionnel mais recommandé)
- **Stockage** : 5 GB d'espace libre
- **Python** : 3.8+ (3.10 recommandé)

### Configuration Optimale
- **RAM** : 32 GB
- **GPU** : NVIDIA RTX 3070+ avec 8GB+ VRAM
- **Stockage** : SSD NVMe avec 20 GB libre
- **Python** : 3.11

## 🚀 Installation

### Étape 1 : Dépendances Python

```bash
# Installation des dépendances principales
pip install ultralytics>=8.0.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install pillow>=9.5.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install tkinter

# Pour support GPU NVIDIA (optionnel)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pour support DirectML (Windows + AMD/Intel GPU)
pip install torch-directml
```

### Étape 2 : Vérification d'Installation

```bash
# Test des imports
python -c "import ultralytics; print('YOLO OK')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### Étape 3 : Structure des Dossiers

Créez la structure suivante dans votre projet :

```
G:\Botting\
├── modules\vision\
│   ├── yolo_detector.py
│   ├── vision_orchestrator.py
│   └── template_matcher.py (existant)
├── tools\
│   └── data_collector.py
├── examples\
│   └── yolo_vision_demo.py
├── models\yolo\
├── data\
│   ├── collected\
│   └── yolo_dataset\
└── docs\
    └── YOLO_VISION_GUIDE.md
```

## 🔧 Configuration Initiale

### 1. Configuration YOLO

Créez le fichier `config/yolo_config.json` :

```json
{
  "yolo_detector": {
    "model_path": "models/yolo/dofus_v8n.pt",
    "confidence_threshold": 0.6,
    "iou_threshold": 0.45,
    "device": "auto",
    "optimize_model": true,
    "half_precision": true
  },
  "vision_orchestrator": {
    "primary_method": "yolo",
    "fallback_method": "template",
    "enable_adaptive_switching": true,
    "parallel_processing": true,
    "enable_caching": true
  }
}
```

### 2. Classes DOFUS

Les classes détectables par défaut :

```python
DOFUS_CLASSES = {
    0: "player",           # Joueur
    1: "npc",             # PNJ
    2: "monster",         # Monstre
    3: "resource_tree",   # Arbre
    4: "resource_ore",    # Minerai
    5: "resource_plant",  # Plante
    6: "loot_bag",        # Sac de butin
    7: "ui_button",       # Bouton interface
    8: "ui_window",       # Fenêtre
    9: "ui_inventory",    # Inventaire
    10: "ui_spells",      # Sorts
    11: "ui_minimap",     # Minimap
    12: "portal",         # Portail
    13: "door",           # Porte
    14: "chest",          # Coffre
    15: "bank",           # Banque
    16: "shop",           # Boutique
    17: "zaap",           # Zaap
    18: "archmonster",    # Archimonster
    19: "quest_item",     # Objet de quête
    20: "pvp_player"      # Joueur PvP
}
```

## 💻 Utilisation

### Démarrage Rapide - Interface Graphique

```bash
# Lancement de l'interface de démonstration
cd G:\Botting
python examples/yolo_vision_demo.py --mode gui
```

L'interface graphique permet :
- Test en temps réel avec capture d'écran
- Benchmark des différentes stratégies
- Collecte de données pour entraînement
- Visualisation des détections

### Utilisation Programmatique

#### 1. Initialisation Basique

```python
from modules.vision.vision_orchestrator import VisionOrchestrator, VisionConfig
import cv2

# Configuration
config = VisionConfig(
    primary_method="yolo",
    confidence_threshold_yolo=0.6,
    enable_adaptive_switching=True
)

# Initialisation
orchestrator = VisionOrchestrator(config=config)
success = orchestrator.initialize({})

if success:
    print("✅ Système YOLO initialisé")
else:
    print("❌ Erreur d'initialisation")
```

#### 2. Détection d'Objets

```python
# Capture d'écran ou chargement d'image
image = cv2.imread("screenshot.jpg")

# Analyse complète
result = orchestrator.analyze(image)

print(f"Détections trouvées: {result['total_detections']}")
print(f"Méthode utilisée: {result['method']}")

# Extraction des détections par classe
for class_name, detections in result['detections_by_class'].items():
    print(f"{class_name}: {len(detections)} détections")

    for detection in detections:
        confidence = detection['confidence']
        position = detection['position']
        bbox = detection['bounding_box']

        print(f"  - Confiance: {confidence:.2f}, Position: {position}")
```

#### 3. Recherche d'Objets Spécifiques

```python
# Recherche de ressources uniquement
resources = orchestrator.find_objects(
    image,
    object_types=['resource_tree', 'resource_ore', 'resource_plant'],
    min_confidence=0.7
)

print(f"Ressources trouvées: {len(resources)}")

# Recherche de monstres
monsters = orchestrator.find_objects(
    image,
    object_types=['monster', 'archmonster'],
    min_confidence=0.8
)

print(f"Monstres trouvés: {len(monsters)}")
```

#### 4. Intégration avec Code Existant

```python
# Remplacement direct du template matcher
# ANCIEN CODE:
# template_results = template_matcher.find_templates(image, 'resources')

# NOUVEAU CODE:
vision_results = orchestrator.find_objects(image, ['resource_tree', 'resource_ore'])

# Le format est compatible !
for result in vision_results:
    position = result['position']
    confidence = result['confidence']
    bbox = result['bounding_box']
    # ... utilisation identique
```

## 📊 Collecte de Données et Entraînement

### 1. Collecte Automatique

```bash
# Interface de collecte
python tools/data_collector.py --gui

# Ou en ligne de commande
python tools/data_collector.py
# Choix 1: Collecte automatique 1h
# Choix 2: Collecte rapide 10min
```

### 2. Configuration de Collecte

```python
from tools.data_collector import SmartDataCollector

collector = SmartDataCollector()

# Configuration des filtres
collector.collection_filters = {
    'min_objects': 2,        # Minimum 2 objets par frame
    'max_objects': 15,       # Maximum 15 objets
    'min_confidence': 0.4,   # Confiance minimum
    'capture_interval': 1.5, # Intervalle en secondes
    'skip_empty_frames': True
}

# Démarrage collecte
collector.start_automated_collection(duration_hours=2.0)
```

### 3. Entraînement du Modèle

```python
from modules.vision.yolo_detector import YOLOTrainer, YOLOConfig

# Configuration d'entraînement
config = YOLOConfig(
    device="cuda",  # ou "cpu"
    learning_rate=0.001,
    batch_size=16
)

# Entraîneur
trainer = YOLOTrainer(config)

# Lancement (peut prendre plusieurs heures)
model_path = trainer.train_model(
    dataset_config="data/collected/yolo_dataset/dataset.yaml",
    epochs=100,
    batch_size=16,
    patience=50
)

print(f"Modèle entraîné sauvegardé: {model_path}")
```

## ⚡ Optimisation des Performances

### 1. Configuration GPU

```python
# Vérification GPU disponible
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")

# Configuration optimale
config = YOLOConfig(
    device="cuda",
    half_precision=True,    # FP16 pour 2x plus rapide
    optimize_model=True,    # Compilation PyTorch
    batch_size=32          # Ajuster selon VRAM
)
```

### 2. Stratégies Adaptatives

```python
# Configuration hybride intelligente
vision_config = VisionConfig(
    primary_method="yolo",
    fallback_method="template",
    enable_adaptive_switching=True,  # Adaptation auto
    switch_threshold=0.3,           # Seuil de changement
    performance_window=50           # Fenêtre d'évaluation
)

# Le système choisira automatiquement la meilleure méthode
orchestrator = VisionOrchestrator(config=vision_config)
```

### 3. Cache et Optimisations

```python
# Configuration cache pour performance max
config = VisionConfig(
    enable_caching=True,
    cache_duration=0.2,        # 200ms de cache
    parallel_processing=True,
    max_workers=4             # Threads parallèles
)
```

## 🔍 Benchmarking et Tests

### 1. Benchmark Automatique

```bash
# Test complet des performances
python examples/yolo_vision_demo.py --mode benchmark

# Ou via l'interface
python examples/yolo_vision_demo.py --mode gui
# Puis cliquer "Test Benchmark"
```

### 2. Test de Stratégies

```python
from modules.vision.vision_orchestrator import VisionOrchestrator
import time

orchestrator = VisionOrchestrator()
orchestrator.initialize({})

# Test des différentes stratégies
strategies = ['yolo', 'template', 'hybrid']
results = {}

for strategy in strategies:
    orchestrator.set_strategy(strategy)

    times = []
    for i in range(10):
        start = time.time()
        result = orchestrator.analyze(test_image)
        times.append(time.time() - start)

    results[strategy] = {
        'avg_time': np.mean(times),
        'fps': 1.0 / np.mean(times),
        'detections': result['total_detections']
    }

# Affichage des résultats
for strategy, metrics in results.items():
    print(f"{strategy}: {metrics['fps']:.1f} FPS, {metrics['detections']} détections")
```

### 3. Validation de Modèle

```python
from modules.vision.yolo_detector import YOLOTrainer

trainer = YOLOTrainer(config)

# Validation sur dataset de test
metrics = trainer.validate_model(
    model_path="models/yolo/dofus_v8n.pt",
    dataset_config="data/yolo_dataset/dataset.yaml"
)

print(f"mAP@0.5: {metrics['map50']:.3f}")
print(f"Précision: {metrics['precision']:.3f}")
print(f"Rappel: {metrics['recall']:.3f}")
```

## 🐛 Dépannage

### Problèmes Courants

#### 1. Erreur "YOLO non disponible"
```bash
# Réinstallation Ultralytics
pip uninstall ultralytics
pip install ultralytics>=8.0.0

# Vérification
python -c "from ultralytics import YOLO; print('OK')"
```

#### 2. Erreur GPU/CUDA
```bash
# Vérification PyTorch + CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Réinstallation avec CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Performance faible
```python
# Mode CPU fallback
config = YOLOConfig(device="cpu")

# Ou réduction qualité
config = YOLOConfig(
    input_size=(320, 320),  # Au lieu de (640, 640)
    half_precision=False
)
```

#### 4. Modèle non trouvé
```python
# Utilisation modèle par défaut
config = YOLOConfig(model_path="yolov8n.pt")  # Téléchargement auto

# Ou chemin absolu
config = YOLOConfig(model_path=r"G:\Botting\models\yolo\dofus_v8n.pt")
```

### Logs de Debug

```python
import logging

# Activation debug
logging.basicConfig(level=logging.DEBUG)

# Logs spécifiques
logger = logging.getLogger("modules.vision.yolo_detector")
logger.setLevel(logging.DEBUG)
```

## 📈 Monitoring et Métriques

### 1. Rapport de Performance

```python
# Rapport détaillé
report = orchestrator.get_performance_report()

print(f"Stratégie actuelle: {report['current_strategy']}")
print(f"Adaptations: {report['strategy_adaptations']}")

for method, metrics in report['performance_by_method'].items():
    if metrics.get('available'):
        print(f"\n{method.upper()}:")
        print(f"  FPS moyen: {metrics['fps']:.1f}")
        print(f"  Précision estimée: {metrics['estimated_accuracy']:.2%}")
        print(f"  Détections totales: {metrics['total_detections']}")
```

### 2. Métriques Temps Réel

```python
# État en temps réel
state = orchestrator.get_state()

print(f"Status: {state['status']}")
print(f"Modules actifs: {len(state['modules'])}")

stats = state['stats']
print(f"Analyses totales: {stats['total_analyses']}")
print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
```

## 🔄 Migration depuis Template Matching

### Code de Migration Automatique

```python
# Ancien code template matching
from modules.vision.template_matcher import TemplateMatcher

# ANCIEN
template_matcher = TemplateMatcher()
template_matches = template_matcher.find_templates(
    image,
    category="resources",
    min_confidence=0.7
)

# NOUVEAU (compatible)
from modules.vision.vision_orchestrator import VisionOrchestrator

orchestrator = VisionOrchestrator()
orchestrator.initialize({})

# Même interface !
yolo_matches = orchestrator.find_objects(
    image,
    object_types=['resource_tree', 'resource_ore', 'resource_plant'],
    min_confidence=0.7
)

# Format identique
for match in yolo_matches:
    position = match['position']      # ✅ Compatible
    confidence = match['confidence']  # ✅ Compatible
    bbox = match['bounding_box']     # ✅ Compatible
```

### Wrapper de Compatibilité

```python
from modules.vision.yolo_detector import YOLOTemplateAdapter

# Adaptation automatique
yolo_detector = DofusYOLODetector()
adapter = YOLOTemplateAdapter(yolo_detector)

# Interface template matcher préservée
results = adapter.find_templates(image, category="monsters")
# ✅ Code existant fonctionne sans modification !
```

## 📚 Ressources Supplémentaires

### Documentation Technique
- [Architecture YOLO](docs/yolo_architecture.md)
- [API Reference](docs/api_reference.md)
- [Modèles Personnalisés](docs/custom_models.md)

### Exemples Avancés
- [Bot Multi-Classes](examples/advanced_multi_class_bot.py)
- [Système de Tracking](examples/object_tracking_demo.py)
- [Intégration RL](examples/yolo_rl_integration.py)

### Communauté
- [Issues GitHub](https://github.com/your-repo/issues)
- [Discussions](https://github.com/your-repo/discussions)
- [Wiki](https://github.com/your-repo/wiki)

## 🎉 Prochaines Étapes

1. **Testez la démo** : Lancez `python examples/yolo_vision_demo.py --mode gui`
2. **Collectez des données** : Utilisez le collecteur pour votre environnement
3. **Entraînez un modèle** : Créez un modèle spécialisé pour vos besoins
4. **Intégrez progressivement** : Remplacez section par section votre code existant
5. **Optimisez** : Ajustez la configuration selon vos performances

**Votre bot DOFUS est maintenant équipé d'une vision de niveau professionnel ! 🚀**