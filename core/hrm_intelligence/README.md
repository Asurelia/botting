# 🤖 HRM Intelligence - Bot Intelligent pour TacticalBot

## Vue d'ensemble

HRM Intelligence est un système d'intelligence artificielle avancé qui transforme votre TacticalBot en un joueur intelligent capable d'apprendre, de s'adapter et de prendre des décisions stratégiques complexes.

## ✨ Fonctionnalités

### 🧠 Intelligence HRM
- **Modèle de raisonnement hiérarchique** 27M paramètres
- **Compatibilité DirectML** pour GPU AMD (RX 7800 XT)
- **Encodage sophistiqué** de l'état du jeu
- **Prédictions d'actions** avec niveau de confiance

### 🎯 Apprentissage Adaptatif
- **Replay buffer** pour stockage d'expériences
- **Comportement humain** simulé (délais, erreurs naturelles)
- **Évolution de stratégies** basée sur les performances
- **Auto-adaptation** continue en temps réel

### 👁️ Analyse Intelligente
- **Vision par ordinateur** pour analyser l'écran de jeu
- **Détection automatique** des éléments d'interface
- **Planification stratégique** à long terme
- **Évaluation des risques** contextuels

### 📋 Suivi de Quêtes
- **OCR automatique** pour détecter les quêtes
- **Base de données SQLite** intégrée
- **Recommandations intelligentes** de quêtes
- **Suivi de progression** en temps réel

### 🎮 Interface de Contrôle
- **Interface graphique** complète avec monitoring
- **Contrôles temps réel** (start/stop/pause)
- **Graphiques de performance** en direct
- **Configuration avancée** personnalisable

## 🚀 Installation

### Prérequis
```bash
# Python 3.12 recommandé (pour DirectML)
# Conda environment avec PyTorch DirectML
conda create -n hrm_bot python=3.12
conda activate hrm_bot

# Installation des dépendances
pip install torch torch-directml torchvision torchaudio
pip install numpy opencv-python pillow matplotlib
pip install pytesseract sqlite3
```

### Vérification du système
```bash
cd "G:\Botting\core\hrm_intelligence"
python launcher.py --status
```

## 📖 Guide d'utilisation

### 1. Lancement Interface Graphique (Recommandé)
```bash
cd "G:\Botting\core\hrm_intelligence"
python launcher.py
```

**Interface complète avec :**
- 🎮 **Contrôle** : Démarrer/arrêter le bot, actions manuelles
- 📊 **Monitoring** : Statistiques et graphiques temps réel
- 🧠 **Apprentissage** : Configuration et historique d'expériences
- 📋 **Quêtes** : Gestion et suivi automatique des quêtes
- 📝 **Logs** : Monitoring des événements système
- ⚙️ **Configuration** : Paramètres avancés

### 2. Mode Console (Pour serveurs)
```bash
# Mode production
python launcher.py --console --player-id "mon_bot"

# Mode debug
python launcher.py --console --debug

# Mode test (30 secondes)
python launcher.py --console --test
```

### 3. Tests d'intégration
```bash
python launcher.py --test
```

## 🎛️ Configuration

### Configuration de base
```python
config = HRMSystemConfig()
config.player_id = "mon_bot"
config.learning_enabled = True
config.human_like_delays = True
config.decision_timeout = 1.0
config.auto_save_interval = 300  # 5 minutes
```

### Comportement humain
- **Délais de réaction** : 100-500ms variables
- **Erreurs occasionnelles** : 2% d'actions aléatoires
- **Pauses naturelles** : Micro-pauses réalistes
- **Patterns d'apprentissage** : Amélioration progressive

## 📊 Monitoring et Métriques

### Métriques de performance
- **Taux de succès** : Pourcentage d'actions réussies
- **Confiance moyenne** : Niveau de certitude des décisions
- **Décisions par minute** : Vitesse de prise de décision
- **Récompenses totales** : Progression et gains
- **Sessions d'apprentissage** : Expériences accumulées

### Graphiques temps réel
- Évolution du taux de succès
- Confiance dans les décisions
- Vitesse de décision
- Taux de récompenses

## 🎯 Utilisation Pratique

### Démarrage rapide
1. **Ouvrir l'interface** : `python launcher.py`
2. **Configurer le bot** : Onglet "Configuration"
3. **Définir l'objectif** : Onglet "Contrôle" → Instructions
4. **Démarrer** : Bouton "▶️ Démarrer"
5. **Surveiller** : Onglets "Monitoring" et "Quêtes"

### Actions disponibles
```python
actions = [
    'move_up', 'move_down', 'move_left', 'move_right',
    'attack', 'defend', 'use_skill_1', 'use_skill_2',
    'use_potion', 'open_inventory', 'interact', 'cast_spell',
    'rest', 'explore', 'gather_resource', 'craft_item',
    'accept_quest', 'complete_quest', 'trade', 'teleport'
]
```

### Commandes manuelles
- **Actions ponctuelles** : Interface "Commandes Manuelles"
- **Objectifs long terme** : Zone "Instructions/Objectif"
- **Contrôles temps réel** : Pause/reprise à tout moment

## 🔧 Architecture Technique

### Modules principaux
```
hrm_intelligence/
├── hrm_core.py              # Cœur HRM (reasoning engine)
├── adaptive_learner.py      # Apprentissage adaptatif
├── intelligent_decision_maker.py  # Décisions multi-couches
├── quest_tracker.py         # Suivi automatique des quêtes
├── main_hrm_system.py       # Orchestrateur principal
├── hrm_gui.py              # Interface graphique
├── launcher.py             # Point d'entrée
└── integration_test.py     # Tests complets
```

### Flux de données
```
Écran de jeu → Vision AI → État du jeu → HRM Core → Décision enrichie → Action → Résultat → Apprentissage
```

## 🛠️ Personnalisation

### Adaptation au jeu spécifique
1. **Modifier `get_current_game_state()`** dans `main_hrm_system.py`
2. **Adapter les actions** dans `action_mapping` de `hrm_core.py`
3. **Configurer l'OCR** pour vos quêtes dans `quest_tracker.py`
4. **Ajuster la vision** dans `VisionAnalyzer` de `intelligent_decision_maker.py`

### Stratégies d'apprentissage
- **Conservateur** : Évite les risques, apprentissage lent
- **Adaptatif** : Équilibre risque/récompense (défaut)
- **Agressif** : Prend plus de risques, apprentissage rapide

## 📁 Structure des Données

### Sauvegarde automatique
```
G:/Botting/
├── models/                  # Modèles HRM entraînés
├── data/hrm/               # Données d'apprentissage
│   ├── learning/           # Expériences et stratégies
│   ├── quests/            # Base de données des quêtes
│   └── screenshots/       # Captures d'écran (si activé)
└── logs/                  # Logs système
```

### Formats de données
- **Modèles** : `.pth` (PyTorch)
- **Apprentissage** : `.json` (expériences)
- **Quêtes** : `.db` (SQLite)
- **Configuration** : `.json`

## 🚨 Dépannage

### Problèmes courants

**1. DirectML non disponible**
```bash
# Réinstaller avec Python 3.12
conda create -n hrm_bot python=3.12
pip install torch-directml
```

**2. Interface ne se lance pas**
```bash
# Vérifier tkinter
python -c "import tkinter; print('OK')"
# Installer matplotlib
pip install matplotlib
```

**3. OCR ne fonctionne pas**
```bash
# Installer Tesseract
# Windows: https://github.com/tesseract-ocr/tesseract
pip install pytesseract
```

**4. Performances lentes**
- Vérifier GPU DirectML activé
- Réduire `decision_timeout` dans config
- Désactiver `save_screenshots` si activé

### Logs de débogage
```bash
python launcher.py --console --debug
# Logs détaillés dans G:/Botting/logs/hrm_system.log
```

## 🎮 Intégration TacticalBot

Le système HRM Intelligence est conçu pour s'intégrer parfaitement avec votre TacticalBot existant :

1. **Point d'entrée unique** via `main_hrm_system.py`
2. **Interface standard** pour capture d'état de jeu
3. **Actions compatibles** avec systèmes existants
4. **Sauvegarde séparée** pour éviter les conflits

## 📈 Optimisation Performance

### Recommandations GPU AMD
- **Driver récent** avec DirectML support
- **Python 3.12** pour meilleure compatibilité
- **Mémoire VRAM** : 4GB+ recommandé
- **Monitoring température** pendant utilisation intensive

### Paramètres optimaux
```python
# Pour performances maximales
config.decision_timeout = 0.5        # Décisions rapides
config.human_like_delays = False     # Pas de délais
config.screenshot_interval = 1.0     # Moins de captures

# Pour réalisme maximal
config.decision_timeout = 1.5        # Réflexion plus longue
config.human_like_delays = True      # Délais naturels
config.random_actions_probability = 0.02  # Erreurs humaines
```

## 🔮 Développements Futurs

### Améliorations prévues
- 🎯 **Auto-questing** : Sélection automatique optimale des quêtes
- 🤝 **Multi-bot coordination** : Plusieurs bots coordonnés
- 📱 **Interface mobile** : Contrôle à distance
- 🎨 **Thèmes interface** : Personnalisation visuelle
- 🌐 **Cloud learning** : Partage d'expériences entre bots

### API Extensions
- **Webhook notifications** : Alertes Discord/Slack
- **REST API** : Contrôle programmatique
- **Plugin system** : Extensions tierces
- **ML pipelines** : Entraînement personnalisé

---

## 🎉 Le système HRM Intelligence est maintenant opérationnel !

Votre TacticalBot est désormais équipé d'une intelligence artificielle avancée capable d'apprendre, de s'adapter et de jouer de manière autonome tout en conservant un comportement naturel et humain.

**Bon gaming ! 🎮**