# 🤖 TacticalBot - Bot Intelligent Multi-Modules

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/tacticalbot/tacticalbot)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Active-brightgreen.svg)](https://github.com/tacticalbot/tacticalbot)

**TacticalBot** est un système de bot intelligent et modulaire conçu pour l'automatisation avancée de jeux en ligne. Il propose une architecture robuste avec gestion d'événements, IA décisionnelle, modules spécialisés et monitoring complet.

## 🌟 Caractéristiques Principales

- **Architecture Modulaire** : Système de modules indépendants et interchangeables
- **IA Décisionnelle** : Moteur intelligent de prise de décision multi-critères
- **Système de Métiers** : Gestion complète de 4 métiers avec optimisation automatique
- **Sécurité Avancée** : Évitement de détection et comportement humain-like
- **Vision Intelligente** : Reconnaissance et analyse d'écran en temps réel
- **Gestion d'État** : Suivi précis de l'état du jeu et du personnage
- **Performance** : 30 FPS avec monitoring et optimisation automatique
- **Extensibilité** : API simple pour ajouter de nouveaux modules

## 📋 Table des Matières

- [Installation Rapide](#-installation-rapide)
- [Utilisation](#-utilisation)
- [Modules Disponibles](#-modules-disponibles)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Développement](#-développement)
- [Support](#-support)
- [Licence](#-licence)

## 🚀 Installation Rapide

### Prérequis

- **Python 3.8+** (recommandé : Python 3.9+)
- **Windows 10/11** (support principal)
- **8 GB RAM** minimum (16 GB recommandé)
- **Résolution d'écran** : 1920x1080 ou supérieure
- **Connexion Internet** stable

### Installation Automatique

```bash
# Cloner le dépôt
git clone https://github.com/tacticalbot/tacticalbot.git
cd tacticalbot

# Installer les dépendances
pip install -r requirements.txt

# Configuration initiale
python setup.py configure

# Test de l'installation
python tacticalbot.py --test
```

### Installation Manuelle

1. **Télécharger les dépendances Python** :
   ```bash
   pip install numpy opencv-python pillow pyautogui keyboard mouse
   pip install python-decouple psutil requests
   pip install dataclasses-json typing-extensions
   ```

2. **Configurer les variables d'environnement** :
   ```bash
   # Créer le fichier .env
   echo "GAME_WINDOW_TITLE=Votre Jeu" > .env
   echo "LOG_LEVEL=INFO" >> .env
   echo "ENABLE_SAFETY=true" >> .env
   ```

3. **Initialiser la configuration** :
   ```bash
   python -c "from engine.core import BotEngine; print('Installation réussie!')"
   ```

## 🎮 Utilisation

### Démarrage Rapide

```python
# Exemple de base - Démarrage du bot avec configuration par défaut
from engine.core import BotEngine, EngineConfig
from modules.professions import ProfessionManager

# Configuration du moteur
config = EngineConfig(
    target_fps=30,
    decision_fps=10,
    enable_logging=True,
    log_level="INFO"
)

# Initialisation du bot
bot = BotEngine(config)

# Ajout du module de métiers
profession_manager = ProfessionManager()
bot.register_module(profession_manager)

# Démarrage
if bot.initialize():
    bot.start()
    print("✅ TacticalBot démarré avec succès!")
```

### Interface en Ligne de Commande

```bash
# Démarrage standard
python tacticalbot.py

# Démarrage avec profil spécifique
python tacticalbot.py --profile farmer_efficient

# Mode debug avec logging détaillé
python tacticalbot.py --debug --verbose

# Session limitée dans le temps (4 heures)
python tacticalbot.py --duration 4h

# Test des modules sans exécution
python tacticalbot.py --test-modules

# Interface web pour monitoring
python tacticalbot.py --web-ui --port 8080
```

### Configuration Rapide par Profils

```python
# Profils prédéfinis disponibles
PROFILS_DISPONIBLES = {
    'farmer_safe': 'Farming sécurisé avec priorité survie',
    'farmer_efficient': 'Farming efficace avec optimisation temps',
    'combat_aggressive': 'Combat agressif pour maximiser gains',
    'combat_defensive': 'Combat défensif avec priorité sécurité',
    'explorer_balanced': 'Exploration équilibrée risque/efficacité',
    'social_cooperative': 'Jeu coopératif avec interactions sociales'
}

# Application d'un profil
from modules.decision.config import DecisionConfigManager

config_manager = DecisionConfigManager()
config_manager.apply_profile('farmer_efficient')
```

## 🧩 Modules Disponibles

### 🔧 Modules Core

| Module | Description | État |
|--------|-------------|------|
| **Engine** | Moteur central et orchestrateur | ✅ Stable |
| **Event Bus** | Système d'événements inter-modules | ✅ Stable |
| **State Management** | Gestion d'état temps réel | ✅ Stable |

### 🎯 Modules Intelligence

| Module | Description | Fonctionnalités |
|--------|-------------|-----------------|
| **Decision Engine** | IA décisionnelle multi-critères | 6 stratégies, 10 situations |
| **Combat AI** | Intelligence de combat avancée | Support 3 classes |
| **Safety Manager** | Évitement détection | Comportement humain-like |

### 🎮 Modules Gameplay

| Module | Description | Ressources |
|--------|-------------|------------|
| **Professions** | Système de métiers complet | 4 métiers, 100+ ressources |
| **Navigation** | Pathfinding intelligent | A*, évitement obstacles |
| **Economy** | Gestion économique | Marché, crafting, inventaire |
| **Social** | Interactions sociales | Chat, groupe, guilde |

### 👁️ Modules Vision

| Module | Description | Capacités |
|--------|-------------|-----------|
| **Screen Analyzer** | Analyse écran temps réel | OCR, reconnaissance formes |
| **Template Matcher** | Reconnaissance de patterns | Templates adaptatifs |

### 🤖 Modules Automation

| Module | Description | Fonctions |
|--------|-------------|----------|
| **Daily Routine** | Routines quotidiennes | Quêtes, récompenses |
| **Leveling Automation** | Automatisation level | XP optimisé |
| **Quest Automation** | Gestion automatique quêtes | 50+ types supportés |

## ⚙️ Configuration

### Fichiers de Configuration

```
config/
├── engine.json          # Configuration moteur principal
├── modules.json         # Configuration des modules
├── professions.json     # Données métiers et progression
├── safety.json          # Paramètres de sécurité
├── ui.json             # Interface utilisateur
└── profiles/           # Profils personnalisés
    ├── farmer.json
    ├── combat.json
    └── explorer.json
```

### Configuration du Moteur

```json
{
  "engine": {
    "target_fps": 30,
    "decision_fps": 10,
    "max_modules": 50,
    "auto_recovery": true,
    "safety_checks": true
  },
  "logging": {
    "level": "INFO",
    "file": "logs/tacticalbot.log",
    "max_size": "100MB",
    "backup_count": 5
  },
  "performance": {
    "memory_limit": "512MB",
    "cpu_limit": 80,
    "monitoring": true
  }
}
```

### Variables d'Environnement

```bash
# .env
GAME_WINDOW_TITLE="Dofus 2.0"
GAME_EXECUTABLE_PATH="C:/Program Files/Dofus/Dofus.exe"

# Sécurité
ENABLE_SAFETY_CHECKS=true
HUMAN_BEHAVIOR_ENABLED=true
RANDOMIZATION_LEVEL=0.7

# Performance
MAX_CPU_USAGE=80
MAX_MEMORY_MB=512
ENABLE_PERFORMANCE_MONITORING=true

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_MAX_SIZE=100MB

# Réseau
ENABLE_WEB_INTERFACE=false
WEB_PORT=8080
API_ENABLED=false
```

## 📚 Documentation

### Documentation Technique

- **[Architecture](docs/ARCHITECTURE.md)** - Architecture détaillée du système
- **[Modules](docs/MODULES.md)** - Guide complet des modules
- **[Configuration](docs/CONFIGURATION.md)** - Configuration avancée
- **[API Reference](docs/API.md)** - Référence complète de l'API

### Guides d'Utilisation

- **[Guide de Démarrage](docs/GETTING_STARTED.md)** - Premier pas avec TacticalBot
- **[Exemples](docs/EXAMPLES.md)** - Cas d'usage et exemples pratiques
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Résolution de problèmes
- **[FAQ](docs/FAQ.md)** - Questions fréquemment posées

### Documentation des Modules

- **[Professions Guide](modules/professions/README.md)** - Système de métiers
- **[Decision Engine](modules/decision/README.md)** - IA décisionnelle
- **[Combat AI](modules/combat/README.md)** - Intelligence de combat
- **[Safety Manager](modules/safety/README.md)** - Système de sécurité

## 🛠️ Développement

### Créer un Nouveau Module

```python
from engine.module_interface import IModule, ModuleStatus

class MonModule(IModule):
    def __init__(self):
        super().__init__("mon_module")
    
    def initialize(self, config):
        # Initialisation du module
        self.status = ModuleStatus.ACTIVE
        return True
    
    def update(self, game_state):
        # Logique principale du module
        return {"action": "mon_action"}
    
    def handle_event(self, event):
        # Gestion des événements
        return True
    
    def get_state(self):
        # Retourne l'état du module
        return {"status": "running"}
    
    def cleanup(self):
        # Nettoyage des ressources
        pass
```

### Enregistrer le Module

```python
# Dans le moteur principal
mon_module = MonModule()
bot.register_module(mon_module, dependencies=["state_manager"])
```

### Tests et Développement

```bash
# Tests unitaires
python -m pytest tests/ -v

# Tests d'intégration
python -m pytest tests/integration/ -v

# Couverture de code
python -m pytest --cov=modules tests/

# Linting et formatage
python -m flake8 modules/
python -m black modules/

# Documentation automatique
python -m sphinx-build docs/ docs/_build/
```

## 🔍 Monitoring et Statistiques

### Interface Web (Optionnelle)

```bash
# Démarrer l'interface web
python tacticalbot.py --web-ui --port 8080
# Accéder à http://localhost:8080
```

### Métriques Disponibles

- **Performance** : FPS, temps de cycle, usage CPU/RAM
- **Modules** : État, erreurs, statistiques par module
- **Gameplay** : XP/h, Kamas/h, actions exécutées
- **Sécurité** : Détections évitées, temps de pause
- **Économie** : Profits, dépenses, ROI par activité

### Logs et Débogage

```bash
# Logs en temps réel
tail -f logs/tacticalbot.log

# Analyse des erreurs
grep "ERROR" logs/tacticalbot.log | tail -20

# Statistiques de performance
python scripts/analyze_performance.py logs/tacticalbot.log
```

## 🚨 Sécurité et Bonnes Pratiques

### Fonctionnalités de Sécurité

- **Randomisation** : Timing et patterns aléatoires
- **Comportement Humain** : Pauses réalistes, mouvements naturels
- **Détection d'Anomalies** : Arrêt automatique si détection
- **Respect des TOS** : Limitations automatiques

### Recommandations

1. **Utilisation Responsable** : Respecter les termes d'utilisation du jeu
2. **Sessions Modérées** : Limiter à 4-6h par session
3. **Surveillance** : Toujours surveiller le bot en fonctionnement
4. **Mise à Jour** : Maintenir le bot à jour régulièrement

## 🤝 Support et Communauté

### Obtenir de l'Aide

- **Documentation** : Consultez d'abord la documentation
- **Issues GitHub** : Rapportez les bugs et problèmes
- **Discussions** : Échangez avec la communauté
- **Wiki** : Base de connaissances collaborative

### Contribuer

1. Fork le projet
2. Créez une branche pour votre fonctionnalité
3. Committez vos changements
4. Poussez vers la branche
5. Ouvrez une Pull Request

### Roadmap

- [ ] Support macOS et Linux
- [ ] Interface graphique complète
- [ ] Machine Learning avancé
- [ ] Support multi-comptes
- [ ] Plugin système
- [ ] Cloud synchronization

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🔗 Liens Utiles

- **[Documentation Complète](https://tacticalbot.github.io/docs/)**
- **[Guide de Démarrage Rapide](docs/GETTING_STARTED.md)**
- **[Exemples de Code](docs/EXAMPLES.md)**
- **[API Reference](docs/API.md)**
- **[Changelog](CHANGELOG.md)**

---

**⚠️ Avertissement** : Ce bot est fourni à des fins éducatives. L'utilisation de bots peut violer les conditions d'utilisation de certains jeux. Utilisez-le de manière responsable et à vos propres risques.

---

*Développé avec ❤️ par la communauté TacticalBot*