# DOFUS Multi-Account Manager

Un système complet de gestion multi-comptes pour DOFUS inspiré de Dindo Bot, offrant une solution sécurisée, performante et scalable pour gérer plusieurs comptes simultanément.

## 🚀 Fonctionnalités principales

### Gestion des comptes
- **Stockage sécurisé** des credentials avec chiffrement AES
- **Gestion centralisée** de tous les comptes
- **Configuration personnalisée** par compte
- **Système de rôles** (leader/follower/independent)
- **Statistiques détaillées** par compte

### Gestion des fenêtres
- **Lancement automatique** des instances DOFUS Unity
- **Positionnement intelligent** des fenêtres multi-écrans
- **Gestion mémoire optimisée** pour les multi-instances
- **Organisation automatique** en grille
- **Surveillance temps réel** des processus

### Synchronisation des actions
- **Système leader/follower** pour coordination
- **Communication inter-processus** (IPC) efficace
- **Synchronisation temps réel** des mouvements et actions
- **Gestion des groupes** de comptes
- **Actions de combat coordonnées**

### Planification des sessions
- **Planificateur automatique** avec rotation des comptes
- **Système de sécurité avancé** pour éviter la détection
- **Pauses intelligentes** basées sur l'analyse des risques
- **Sessions programmées** par compte ou groupe
- **Gestion des horaires** optimisée

### Interface graphique complète
- **Dashboard unifié** pour tous les comptes
- **Surveillance temps réel** des statuts
- **Contrôles intuitifs** pour chaque fonctionnalité
- **Gestion visuelle** des groupes et planifications
- **Logs centralisés** avec filtrage

## 🏗️ Architecture

```
core/multi_account/
├── account_manager.py      # Gestionnaire centralisé des comptes
├── window_manager.py       # Gestion fenêtres multiples DOFUS Unity  
├── account_synchronizer.py # Synchronisation actions entre comptes
├── session_scheduler.py    # Planification sessions par compte
├── multi_account_gui.py    # Interface graphique complète
├── main.py                 # Point d'entrée principal
├── __init__.py             # Module Python avec API simplifiée
└── README.md               # Documentation
```

### Composants principaux

#### AccountManager
Gestionnaire centralisé pour tous les comptes DOFUS :
- Stockage sécurisé des credentials avec chiffrement AES-256
- Base de données SQLite pour persistance
- Gestion des configurations par compte
- Surveillance des statuts en temps réel
- Système de groupes intégré

#### WindowManager
Gestion avancée des fenêtres multiples :
- Détection automatique des instances DOFUS
- Positionnement intelligent multi-écrans
- Surveillance mémoire et performance
- Contrôles complets (focus, minimiser, fermer, organiser)
- Gestion des processus avec psutil

#### AccountSynchronizer
Système de synchronisation inter-comptes :
- Communication IPC via sockets TCP
- Système de files d'actions prioritaires
- Coordination leader/follower
- Synchronisation temps réel des mouvements
- Gestion des groupes avec formations

#### SessionScheduler
Planificateur intelligent des sessions :
- Algorithme de sécurité anti-détection
- Rotation automatique des comptes
- Pauses intelligentes basées sur les risques
- Planification flexible par compte/groupe
- Historique et statistiques détaillées

## 📦 Installation

### Prérequis
```bash
# Python 3.9+
pip install cryptography psutil schedule PySide6 sqlite3
```

### Installation du système
```bash
# Cloner ou copier les fichiers dans G:/Botting/core/multi_account/
# Assurer que les répertoires existent
mkdir -p G:/Botting/{data,logs,config,temp}
```

## 🎯 Usage

### Lancement rapide
```python
# Interface graphique (recommandé)
python main.py --gui

# Mode console interactif
python main.py --console

# Démo avec comptes factices
python main.py --demo

# Tests automatiques
python main.py --test
```

### Usage programmatique
```python
from core.multi_account import MultiAccountSystem

# Créer et démarrer le système
system = MultiAccountSystem()
system.start()

# Ajouter des comptes
account1_id = system.add_account(
    username="mon_compte",
    password="mon_mot_de_passe", 
    character_name="Mon-Personnage",
    server="Ily"
)

# Créer un groupe
group_id = system.create_group(
    name="Mon Groupe",
    leader_id=account1_id,
    member_ids=[account2_id, account3_id]
)

# Planifier une session
from datetime import datetime, timedelta
schedule_id = system.schedule_session(
    account_id=account1_id,
    session_type="farming",
    start_time=datetime.now() + timedelta(minutes=5),
    duration=timedelta(hours=2)
)

# Lancer l'interface graphique
system.launch_gui()
```

### Configuration avancée
```python
from core.multi_account import AccountConfig

# Configuration personnalisée par compte
config = AccountConfig(
    max_session_duration=4*3600,  # 4 heures max
    auto_reconnect=True,
    farming_priority=["Blé", "Orge", "Avoine"],
    combat_role="dps",
    rest_intervals={"min": 300, "max": 900}  # 5-15 min
)

account_id = system.add_account(
    username="compte_avance",
    password="password",
    character_name="Personnage",
    server="Meriana",
    config=config
)
```

## 🔐 Sécurité

### Chiffrement des données
- **AES-256** pour les mots de passe
- **PBKDF2** avec 100,000 itérations
- **Salt unique** par installation
- **Clés dérivées** du mot de passe maître

### Anti-détection
- **Analyse des risques** basée sur les patterns
- **Pauses intelligentes** avec randomisation
- **Rotation automatique** des comptes
- **Limites de sécurité** configurables
- **Surveillance continue** des métriques

### Bonnes pratiques
```python
# Limites recommandées
MAX_CONCURRENT_ACCOUNTS = 8
MAX_DAILY_HOURS = 12
MAX_CONSECUTIVE_HOURS = 4

# Pauses recommandées entre sessions
MIN_BREAK = 5 * 60     # 5 minutes
MAX_BREAK = 15 * 60    # 15 minutes
LONG_BREAK = 2 * 3600  # 2 heures
```

## 📊 Surveillance et statistiques

### Métriques disponibles
- Temps de jeu par compte et total
- Utilisation mémoire par instance
- Taux de succès des actions synchronisées
- Score de risque par compte
- Performance du système

### Dashboard temps réel
- Statut de tous les comptes
- Fenêtres actives avec positions
- Sessions planifiées et en cours
- Groupes et synchronisation
- Logs centralisés avec filtres

## 🛠️ Maintenance

### Outils intégrés
```bash
# Mode maintenance
python main.py --maintenance

# Options disponibles:
# 1. Nettoyer les logs
# 2. Réinitialiser la base de données  
# 3. Vérifier l'intégrité du système
# 4. Exporter/Importer configuration
```

### Base de données
Le système utilise SQLite avec les tables :
- `accounts` : Comptes et configurations
- `sessions` : Historique des sessions  
- `schedules` : Planifications actives
- `groups` : Groupes de comptes
- `session_history` : Statistiques détaillées

### Fichiers de configuration
```
G:/Botting/
├── data/           # Bases de données SQLite
├── logs/           # Fichiers de logs rotatifs
├── config/         # Configurations sauvegardées  
└── temp/           # Fichiers temporaires
```

## 🚨 Limitations et avertissements

### Limites techniques
- **Maximum 8 comptes simultanés** (configurable)
- **Windows uniquement** (API win32)
- **DOFUS Unity requis**
- **Mémoire RAM significative** (2-4GB par instance)

### Considérations légales
- ⚠️ **Respecter les ToS de DOFUS**
- ⚠️ **Utilisation à vos risques et périls**
- ⚠️ **Pas de garantie contre les sanctions**
- ⚠️ **Usage éducatif recommandé**

## 🔧 Dépannage

### Problèmes courants

#### Erreur de lancement DOFUS
```python
# Vérifier le chemin vers DOFUS
window_manager = WindowManager(
    dofus_path=r"C:\Program Files (x86)\DOFUS\DOFUS.exe"
)
```

#### Problèmes de mémoire
```python
# Réduire le nombre d'instances
system = MultiAccountSystem(max_accounts=4)

# Ou surveiller l'usage mémoire
stats = system.get_statistics()
memory_mb = stats['windows']['total_memory_mb']
```

#### Erreurs de synchronisation
```python
# Redémarrer le synchroniseur
system.synchronizer.stop()
system.synchronizer.start()

# Vérifier les ports IPC
import socket
sock = socket.socket()
sock.connect(('localhost', 12000))  # Port par défaut
```

### Logs et debugging
```python
# Activer les logs détaillés
import logging
logging.getLogger('core.multi_account').setLevel(logging.DEBUG)

# Consulter les logs
tail -f G:/Botting/logs/multi_account.log
```

## 🤝 Contribution

### Structure du code
- **Type hints** obligatoires
- **Docstrings** pour toutes les fonctions publiques
- **Logging** approprié pour debug/info/error
- **Tests unitaires** pour nouvelles fonctionnalités

### Nouvelles fonctionnalités
1. Fork le projet
2. Créer une branche feature
3. Implémenter avec tests
4. Documenter les changements
5. Soumettre une pull request

## 📜 Licence

Ce projet est fourni à des fins éducatives et de recherche. L'utilisation en production est à vos risques et périls.

## 📞 Support

Pour les questions techniques :
1. Consulter cette documentation
2. Vérifier les logs d'erreur
3. Tester avec `--demo` ou `--test`
4. Utiliser le mode `--maintenance`

---

**Version**: 1.0.0  
**Dernière mise à jour**: 2024  
**Développé pour**: DOFUS Unity  
**Plateforme**: Windows 10/11