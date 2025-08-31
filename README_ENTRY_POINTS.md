# Points d'Entrée du Bot DOFUS

Ce document décrit l'utilisation des quatre points d'entrée principaux du bot DOFUS.

## 📋 Vue d'Ensemble

Le bot DOFUS dispose de 4 points d'entrée principaux :

1. **`main.py`** - Point d'entrée principal avec CLI/GUI/Service
2. **`bot_launcher.py`** - Launcher intelligent avec gestion des profils
3. **`calibrate.py`** - Outil de calibration de l'interface DOFUS
4. **`monitor.py`** - Dashboard de monitoring en temps réel

## 🚀 Installation et Configuration

### Prérequis

```bash
# Installer les dépendances
pip install -r requirements.txt

# Créer les dossiers nécessaires
mkdir -p config/profiles logs
```

### Configuration Initiale

```bash
# Créer un profil par défaut
python bot_launcher.py --create-profile "MonBot" --type farmer --character-name "MonPerso"

# Calibrer l'interface DOFUS
python calibrate.py --gui

# Tester le monitoring
python monitor.py --cli
```

## 🎯 main.py - Point d'Entrée Principal

### Description
Point d'entrée unifié supportant 3 modes d'opération : CLI interactif, GUI graphique, et service en arrière-plan.

### Utilisation

#### Mode CLI (par défaut)
```bash
# Démarrage simple
python main.py

# Avec profil spécifique
python main.py --profile "MonBot"

# Avec démarrage automatique
python main.py --auto-start --profile "Farmer"

# Mode debug
python main.py --debug
```

#### Mode GUI
```bash
# Interface graphique complète
python main.py --mode gui

# Avec profil prédéfini
python main.py --mode gui --profile "MonBot"
```

#### Mode Service/Daemon
```bash
# Service interactif
python main.py --mode service --profile "MonBot"

# Daemon en arrière-plan
python main.py --mode service --daemon --profile "MonBot"
```

### Commandes CLI Interactives

Une fois en mode CLI, utilisez ces commandes :

- `start` - Démarrer le bot
- `stop` - Arrêter le bot
- `status` - Afficher le statut actuel
- `modules` - Lister les modules disponibles
- `profile` - Changer de profil
- `help` - Afficher l'aide
- `exit` - Quitter l'application

### Exemple de Session

```
$ python main.py --profile "Farmer"

=== BOT DOFUS - MODE CLI ===
Profil: Farmer
Mode debug: Non
Commandes disponibles:
  start    - Démarrer le bot
  stop     - Arrêter le bot
  status   - Afficher le statut
  modules  - Lister les modules
  profile  - Changer de profil
  exit     - Quitter l'application
----------------------------------------

bot> start
✓ Bot démarré avec succès

bot> status
Statut du bot: En cours
Profil actuel: Farmer
Temps d'exécution: 00:02:15
Actions effectuées: 45
Modules actifs: 3

bot> stop
✓ Bot arrêté

bot> exit
```

## 🎛️ bot_launcher.py - Launcher Intelligent

### Description
Launcher avancé avec gestion de profils, détection automatique des conditions optimales, et planification des sessions.

### Gestion des Profils

#### Créer un Profil
```bash
# Profil farmer basique
python bot_launcher.py --create-profile "Farmer" --type farmer

# Profil complet avec personnalisation
python bot_launcher.py --create-profile "MinerPro" --type miner \
  --character-name "MonMineur" --character-class "enutrof" --server "Ily"
```

#### Lister les Profils
```bash
python bot_launcher.py --list-profiles
```

#### Supprimer un Profil
```bash
python bot_launcher.py --delete-profile "AncienProfil"
```

### Lancement de Profils

#### Lancement Simple
```bash
# Lancement immédiat
python bot_launcher.py --launch-profile "Farmer"
```

#### Lancement Conditionnel
```bash
# Attendre que le système soit inactif
python bot_launcher.py --launch-profile "Miner" --conditions system_idle

# Attendre CPU faible + jeu détecté
python bot_launcher.py --launch-profile "Combat" \
  --conditions low_cpu game_detected
```

#### Lancement Programmé
```bash
# Programmer à 14h00
python bot_launcher.py --launch-profile "Farmer" --schedule "14:00"
```

### Lancement en Série
```bash
# Lancer plusieurs profils avec intervalle
python bot_launcher.py --batch-profiles "Farmer,Miner,Alchemist" --interval 60
```

### Détection Automatique
```bash
# Sélectionner automatiquement le meilleur profil
python bot_launcher.py --auto-detect
```

### Types de Profils Disponibles

- **farmer** - Collecte de ressources agricoles
- **miner** - Minage de minerais
- **lumberjack** - Coupe de bois
- **alchemist** - Alchimie et crafting
- **combat** - Combat et leveling
- **dungeon** - Exploration de donjons
- **leveling** - Montée de niveau optimisée
- **economy** - Trading et économie
- **custom** - Configuration personnalisée

## 🎨 calibrate.py - Outil de Calibration

### Description
Outil de calibration pour configurer la détection des éléments de l'interface DOFUS et optimiser les performances du bot.

### Interface Graphique (Recommandée)
```bash
python calibrate.py --gui
```

#### Fonctionnalités GUI
- **Détection automatique de la fenêtre DOFUS**
- **Capture d'écran en temps réel**
- **Sélection visuelle des zones d'interface**
- **Test de précision de détection**
- **Sauvegarde/chargement de configurations**
- **Export/import de profils de calibration**

#### Utilisation GUI
1. Cliquer "Détecter Fenêtre" pour localiser DOFUS
2. Cliquer "Capturer Écran" pour prendre une image
3. Sélectionner des zones en cliquant-glissant sur l'image
4. Nommer et ajouter les zones importantes
5. Tester la détection avec "Test Zone"
6. Sauvegarder la configuration

### Mode Ligne de Commande

#### Détection Automatique
```bash
# Calibration automatique complète
python calibrate.py --auto-detect

# Sauvegarder une capture d'écran
python calibrate.py --save-screenshot capture.png
```

#### Test de Détection
```bash
python calibrate.py --test-detection
```

### Zones de Calibration Standards

Le système détecte automatiquement ces zones :

- **chat_zone** - Zone de chat du jeu
- **inventory** - Fenêtre d'inventaire
- **character_stats** - Statistiques du personnage
- **minimap** - Mini-carte
- **action_bar** - Barre d'actions/sorts
- **game_area** - Zone de jeu principale
- **resource_nodes** - Noeuds de ressources
- **monsters** - Monstres et ennemis
- **npc** - Personnages non-joueurs

### Configuration Avancée

Le fichier `config/calibration.json` contient :
- Coordonnées des zones d'interface
- Seuils de confiance pour la détection
- Paramètres de reconnaissance de couleurs
- Configuration des templates de reconnaissance

## 📊 monitor.py - Dashboard de Monitoring

### Description
Interface de surveillance complète avec métriques en temps réel, alertes configurables, et dashboard web optionnel.

### Interface Graphique
```bash
python monitor.py --dashboard
```

#### Onglets Disponibles
- **Vue d'ensemble** - Indicateurs principaux et graphiques temps réel
- **Système** - Métriques système détaillées (CPU, mémoire, réseau)
- **Bot** - Statistiques du bot et performances des modules
- **Alertes** - Alertes actives et configuration
- **Logs** - Journaux en temps réel avec filtrage

### Dashboard Web
```bash
# Serveur web sur port 8080
python monitor.py --web-server --port 8080

# Accessible sur http://localhost:8080
```

### Mode CLI
```bash
# Monitoring en ligne de commande
python monitor.py --cli
```

### Export de Rapports
```bash
# Rapport JSON des dernières 24h
python monitor.py --export-report --format json

# Rapport HTML (si supporté)
python monitor.py --export-report --format html
```

### Métriques Surveillées

#### Métriques Système
- Utilisation CPU (système et bot)
- Utilisation mémoire (système et bot)
- Utilisation disque
- Trafic réseau
- Nombre de processus

#### Métriques Bot
- Statut d'exécution
- Temps de fonctionnement
- Actions effectuées par minute
- Taux de succès des actions
- Modules actifs et leurs performances
- Ressources du personnage (niveau, kamas, objets)

#### Alertes Configurables
- CPU élevé (> 80%)
- Mémoire élevée (> 90%)
- Bot arrêté inopinément
- Taux de succès faible (< 50%)
- Aucune action depuis 5 minutes

### Configuration des Alertes

Créer/modifier `config/alerts.json` :

```json
{
  "custom_alert": {
    "name": "custom_alert",
    "condition": "bot_metrics.character_level >= 50",
    "message": "Niveau 50 atteint !",
    "severity": "info",
    "enabled": true,
    "cooldown": 3600
  }
}
```

## 🔧 Outils Rapides

### Accès Direct aux Outils

```bash
# Calibration rapide
python main.py --calibrate

# Monitoring rapide
python main.py --monitor
```

### Scripts de Démarrage Recommandés

#### Windows (start_bot.bat)
```batch
@echo off
echo Démarrage du bot DOFUS...
python main.py --mode gui --profile "MonBot"
pause
```

#### Linux/Mac (start_bot.sh)
```bash
#!/bin/bash
echo "Démarrage du bot DOFUS..."
python3 main.py --mode gui --profile "MonBot"
```

## 📁 Structure des Fichiers de Configuration

```
config/
├── profiles/           # Profils utilisateur
│   ├── MonBot.json
│   └── Farmer.json
├── calibration.json    # Configuration calibration
├── alerts.json         # Configuration alertes
└── settings.json       # Paramètres généraux

logs/
├── main_20240831.log   # Logs principaux
├── metrics.db          # Base de données métriques
└── calibration_screenshot.png
```

## ⚠️ Dépannage Courant

### Interface Graphique Non Disponible
```bash
pip install tkinter matplotlib
# Sur Ubuntu/Debian:
sudo apt-get install python3-tk
```

### Détection de Fenêtre Échoue
```bash
pip install pygetwindow pyautogui
# Vérifier que DOFUS est ouvert et visible
```

### Monitoring Web Inaccessible
```bash
pip install flask
# Vérifier le port avec: netstat -an | grep 8080
```

### Base de Données Corrompue
```bash
# Supprimer et recréer
rm logs/metrics.db
python monitor.py --cli  # Recrée automatiquement
```

## 🎯 Workflow Recommandé

1. **Installation** : `pip install -r requirements.txt`
2. **Profil** : `python bot_launcher.py --create-profile "MonBot" --type farmer`
3. **Calibration** : `python calibrate.py --gui`
4. **Test** : `python main.py --mode cli --profile "MonBot"`
5. **Monitoring** : `python monitor.py --dashboard`
6. **Production** : `python main.py --mode service --profile "MonBot"`

## 📚 Ressources Supplémentaires

- Configuration avancée des profils : voir `config/profiles/`
- Templates de calibration : voir `modules/vision/`
- Logs détaillés : voir `logs/`
- Documentation API : voir `docs/`

---

*Dernière mise à jour : 31/08/2025*