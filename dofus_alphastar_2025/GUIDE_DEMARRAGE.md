# 🚀 GUIDE DE DÉMARRAGE - DOFUS AlphaStar 2025

## ⚠️ AVANT DE COMMENCER - LECTURE OBLIGATOIRE

```
🚨 AVERTISSEMENTS CRITIQUES:

1. ❌ NE JAMAIS utiliser sur compte principal
2. ✅ TOUJOURS utiliser compte jetable uniquement
3. 🔒 Mode observation OBLIGATOIRE pour premiers tests
4. ⏱️ Limiter sessions à 5-10 minutes au début
5. 📊 Analyser logs AVANT d'activer mode réel

VIOLATION DES ToS DE DOFUS = BAN PERMANENT
VOUS ÊTES SEUL RESPONSABLE DE L'UTILISATION
```

---

## 📋 Table des Matières

1. [Installation](#installation)
2. [Calibration Automatique](#calibration)
3. [Mode Observation](#mode-observation)
4. [Test des Systèmes](#tests)
5. [Interface Moderne](#interface)
6. [FAQ et Troubleshooting](#faq)

---

## 1️⃣ Installation

### Prérequis

- **OS**: Windows 10/11, Linux (Ubuntu 20.04+)
- **Python**: 3.10 ou 3.11
- **RAM**: 16GB minimum (32GB recommandé)
- **GPU**: AMD 7800XT (ou NVIDIA RTX 3060+)
- **Espace disque**: 20GB

### Étape 1: Clone du Projet

```bash
git clone https://github.com/Asurelia/botting.git
cd botting/dofus_alphastar_2025
```

### Étape 2: Environnement Virtuel

```bash
# Créer venv
python -m venv venv

# Activer
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### Étape 3: Dépendances

```bash
# Dépendances principales
pip install -r requirements.txt

# PyTorch pour AMD GPU
pip install torch torch-directml

# OU PyTorch pour NVIDIA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Étape 4: Dépendances Optionnelles

```bash
# Vision avancée
pip install opencv-python pytesseract easyocr

# Map system
pip install networkx matplotlib

# UI moderne
pip install pillow
```

### Vérification Installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Devrait afficher la version de PyTorch et True si GPU détecté.

---

## 2️⃣ Calibration Automatique

### Qu'est-ce que la Calibration?

La calibration est une phase **UNIQUE** de 5-10 minutes où le bot :
- Détecte automatiquement la fenêtre Dofus
- Mappe tous les éléments UI (HP, PA, sorts, etc.)
- Découvre les raccourcis clavier
- Scanne les éléments interactifs

### Préparation

1. **Lance Dofus Unity**
2. **Connecte-toi** à un personnage
3. **Va sur une map vide** (village de départ recommandé)
4. **Fullscreen ou fenêtré** (le bot détectera automatiquement)

### Lancement

```bash
python launch_safe.py --calibrate
```

### Déroulement

```
[Phase 1/6] Détection de la fenêtre Dofus...
  ✓ Fenêtre trouvée: Dofus (1920x1080)

[Phase 2/6] Mapping de l'interface...
  ✓ HP bar détecté
  ✓ PA bar détecté
  ✓ Minimap détectée
  ... (12 éléments au total)

[Phase 3/6] Détection des raccourcis...
  ✓ inventory = i
  ✓ character = c
  ... (8 raccourcis au total)

[Phase 4/6] Scan des éléments interactifs...
  ✓ 15 éléments trouvés

[Phase 5/6] Analyse des options du jeu...
  ✓ Options analysées: 3 catégories

[Phase 6/6] Construction de la base de connaissances...
  ✓ Base sauvegardée: config/dofus_knowledge.json

✅ CALIBRATION TERMINÉE!
   Durée: 8.5 secondes
```

### Fichier Généré

`config/dofus_knowledge.json`:
```json
{
  "version": "1.0",
  "calibration_date": "2025-09-30T10:30:00",
  "window": {
    "x": 0,
    "y": 0,
    "width": 1920,
    "height": 1080,
    "is_fullscreen": true
  },
  "ui_elements": [...],
  "shortcuts": [...],
  "game_options": {...}
}
```

---

## 3️⃣ Mode Observation

### Qu'est-ce que le Mode Observation?

**MODE LE PLUS IMPORTANT** pour débuter:

- ✅ Le bot **observe** sans agir
- ✅ Prend des décisions (loggées)
- ✅ Analyse l'état du jeu
- ❌ **N'exécute AUCUNE action** (clavier/souris)

### Premier Test (10 minutes)

```bash
python launch_safe.py --observe 10
```

### Déroulement

```
╔════════════════════════════════════════════════════════════════╗
║    🤖 DOFUS ALPHASTAR 2025 - MODE SÉCURISÉ                    ║
║                                                                 ║
║  ⚠️  MODE OBSERVATION ACTIVÉ - AUCUNE ACTION                   ║
╚════════════════════════════════════════════════════════════════╝

✓ Mode observation initialisé

📝 Le bot va maintenant observer...
   Vous pouvez jouer manuellement pendant ce temps
   Le bot va logger toutes les décisions qu'il AURAIT prises

[00:01] [OBSERVATION] navigation: {'target': (200, 250)}
  Raison: Exploration de la map
  ❌ ACTION BLOQUÉE

[00:05] [OBSERVATION] mouse_click: {'position': (150, 180)}
  Raison: Collecte ressource
  ❌ ACTION BLOQUÉE

...

⌛ Session terminée (10 minutes)

📊 Analyse des observations...
```

### Rapport d'Analyse

```
📊 RAPPORT D'OBSERVATION
════════════════════════════════════════════════════════════════

📈 Statistiques:
  • Durée: 600.0s
  • Décisions: 120
  • Actions bloquées: 120
  • Actions/min: 12.0

🎯 Score de sécurité: 75.0/100

🔝 Top 5 actions:
  • navigation: 45
  • mouse_click: 30
  • key_press: 25
  • spell_cast: 15
  • item_use: 5

💡 Recommandations:
  • ✓ Comportement semble naturel
  • Randomiser les délais entre actions
```

### Logs Sauvegardés

`logs/observation.json`:
```json
{
  "mode": "observation",
  "enabled": true,
  "statistics": {
    "total_decisions": 120,
    "actions_blocked": 120,
    "duration_seconds": 600.0
  },
  "observations": [
    {
      "timestamp": 1696069800.5,
      "action_type": "navigation",
      "action_details": {"target": [200, 250]},
      "game_state": {"hp": 100, "position": [150, 200]},
      "decision_reason": "Exploration de la map",
      "would_execute": false
    },
    ...
  ]
}
```

---

## 4️⃣ Test des Systèmes

### Test DofusDB API

```bash
python launch_safe.py --test-dofusdb
```

Vérifie la connexion à l'API DofusDB:

```
📡 Test de connexion DofusDB...
  Recherche: 'Dofus'

✓ 5 résultats trouvés:
    • Dofus Émeraude (lvl 200)
    • Dofus Pourpre (lvl 200)
    • Dofus Ocre (lvl 200)
    • Dofus Turquoise (lvl 200)
    • Dofus Ivoire (lvl 200)

📊 Statistiques:
    • Requêtes: 1
    • Cache hits: 0
    • Cache ratio: 0.0%
```

### Test Map System

```python
# test_map_system.py
from core.map_system import create_map_graph, MapCoords

map_graph = create_map_graph()

# Pathfinding
path = map_graph.find_path(
    from_coords=MapCoords(5, -18),
    to_coords=MapCoords(-3, 4)
)

if path:
    print(f"Chemin trouvé: {len(path)} maps")
    for coords in path:
        print(f"  → {coords}")
```

### Test Vision System

```python
# test_vision.py
from core.vision_system import create_screen_analyzer

analyzer = create_screen_analyzer()

# Analyse écran
result = analyzer.analyze_current_screen()

print(f"Coordonnées map: {result.map_coords}")
print(f"HP: {result.hp}/{result.max_hp}")
print(f"PA: {result.pa}/{result.max_pa}")
```

---

## 5️⃣ Interface Moderne

### Lancement Standalone

```bash
python test_themes_direct.py
```

### Features

**Dashboard** (`Onglet Dashboard`)
- Monitoring temps réel
- Statuts du bot et personnage
- Métriques de performance
- Graphiques temps réel

**Contrôle** (`Onglet Contrôle`)
- Configuration du bot
- Paramètres de quêtes
- Options combat/navigation
- Configuration IA

**Analytics** (`Onglet Analytics`)
- Graphiques avancés
- Statistiques détaillées
- Rapports d'efficacité
- Export de données

**Monitoring** (`Onglet Monitoring`)
- Console de logs
- Filtrage avancé
- Monitoring système
- Debug tools

**Configuration** (`Onglet Configuration`)
- Paramètres globaux
- Personnalisation thèmes
- Notifications
- Import/export config

### Changement de Thème

Boutons en bas:
- **Theme Sombre** (défaut)
- **Theme Clair**

---

## 6️⃣ FAQ et Troubleshooting

### Q: Le bot ne détecte pas la fenêtre Dofus

**R:** Vérifier que:
1. Dofus est bien lancé et visible
2. Titre de la fenêtre contient "Dofus"
3. Fenêtre pas minimisée

### Q: Calibration échoue phase 3 (raccourcis)

**R:** Les raccourcis par défaut peuvent varier. Solution:
1. Vérifier config des touches dans Dofus
2. Relancer calibration
3. Si persiste, éditer manuellement `config/dofus_knowledge.json`

### Q: Mode observation: rien ne se passe

**R:** C'est NORMAL ! Le bot observe sans agir:
- Logs dans `logs/observation.json`
- Aucune action visible à l'écran
- Analyser les logs après session

### Q: DofusDB API ne répond pas

**R:**
1. Vérifier connexion Internet
2. API peut être temporairement indisponible
3. Le cache local prendra le relais si disponible

### Q: GPU AMD non détecté

**R:**
```bash
# Réinstaller torch-directml
pip uninstall torch torch-directml
pip install torch torch-directml

# Vérifier
python -c "import torch_directml; print(torch_directml.device())"
```

### Q: Erreur "pygetwindow not found"

**R:**
```bash
pip install pygetwindow

# Linux: Peut nécessiter xdotool
sudo apt-get install xdotool python3-xlib
```

### Q: Interface UI freeze ou ne répond pas

**R:**
1. Vérifier que tous les threads de monitoring sont bien lancés
2. Fermer et relancer
3. Mode standalone: `python test_themes_direct.py`

---

## 🎯 Prochaines Étapes

### Après Calibration + Observation Réussie

1. **Analyser logs d'observation** (crucial!)
2. **Valider score de sécurité** (>70 recommandé)
3. **Tests courts** (5 min max) en mode observation
4. **Uniquement si satisfait**: Tests compte jetable

### Activation Mode Réel (DANGER!)

```bash
python launch_safe.py --unsafe --observe 5

# Confirmation requise:
# "JE COMPRENDS LES RISQUES"
```

**⚠️ NE JAMAIS sur compte principal!**

---

## 📚 Ressources Supplémentaires

- [README.md](README.md) - Vue d'ensemble complète
- [Architecture](docs/alphastar_architecture.md)
- [HRM Reasoning](docs/hrm_reasoning.md)
- [Map System](docs/map_system.md)
- [Safety Guide](docs/safety_testing.md)

---

## 🆘 Support

**Problèmes**: [GitHub Issues](https://github.com/Asurelia/botting/issues)

**Questions**: [GitHub Discussions](https://github.com/Asurelia/botting/discussions)

---

**Bon courage et restez prudent ! 🚀**