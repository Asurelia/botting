# 🎮 GUIDE UTILISATEUR COMPLET - DOFUS AlphaStar 2025

**Version:** 1.0.0
**Pour:** Utilisateurs débutants à avancés
**Date:** Janvier 2025

---

## 📚 TABLE DES MATIÈRES

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Premier Démarrage](#premier-démarrage)
4. [Interface Utilisateur](#interface-utilisateur)
5. [Fonctionnalités](#fonctionnalités)
6. [Système d'Apprentissage](#système-dapprentissage)
7. [FAQ](#faq)
8. [Dépannage](#dépannage)

---

## 🌟 INTRODUCTION

### Qu'est-ce que DOFUS AlphaStar 2025?

DOFUS AlphaStar 2025 est un bot **d'observation et d'apprentissage** pour DOFUS, conçu avec une intelligence artificielle avancée inspirée d'AlphaStar (DeepMind).

**Ce qu'il FAIT:**
- 👁️ Observe le jeu en temps réel
- 🧠 Prend des décisions intelligentes
- 📊 Analyse les opportunités économiques
- 📚 Apprend de vos retours
- 📈 Optimise les stratégies de farming/craft

**Ce qu'il NE FAIT PAS:**
- ❌ Jouer automatiquement (mode observation par défaut)
- ❌ Garantir l'absence de bannissement
- ❌ Remplacer le jeu manuel

---

## 💻 INSTALLATION

### Étape 1: Prérequis

**Système:**
- Windows 10 ou 11
- Python 3.9 ou supérieur
- 8GB RAM minimum (16GB recommandé)
- GPU optionnel (AMD RX 7800 XT pour performances maximales)

**Logiciels:**
- DOFUS installé
- Git (optionnel)

### Étape 2: Téléchargement

```bash
# Option 1: Clone avec Git
git clone https://github.com/votre-repo/dofus-alphastar-2025.git
cd dofus-alphastar-2025

# Option 2: Téléchargement ZIP
# Télécharger depuis GitHub et extraire
```

### Étape 3: Installation Python

```bash
# Créer environnement virtuel
python -m venv venv

# Activer environnement
venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
```

**⏱️ Temps d'installation:** ~10-15 minutes

### Étape 4: Vérification

```bash
# Test d'installation
python -c "from core.game_loop import create_game_engine; print('✅ Installation OK')"
```

---

## 🚀 PREMIER DÉMARRAGE

### Mode 1: Interface Graphique (Recommandé)

```bash
python launch_ui.py
```

**Avantages:**
- Interface visuelle intuitive
- Contrôles faciles
- Logs en temps réel
- Statistiques graphiques

### Mode 2: Ligne de Commande

```bash
# Observation 30 minutes
python launch_autonomous_full.py --duration 30

# Avec calibration
python launch_autonomous_full.py --calibrate --duration 30
```

### Premier Test (5 minutes)

1. **Lancer DOFUS** et se connecter
2. **Lancer le bot:**
   ```bash
   python launch_safe.py --observe 5
   ```
3. **Observer:** Le bot va logger ses observations pendant 5 minutes
4. **Vérifier logs:** `logs/observation.json`

---

## 🎨 INTERFACE UTILISATEUR

### Vue d'ensemble

L'interface moderne comprend 6 onglets:

```
┌─────────────────────────────────────────────────┐
│ 📊 Dashboard | ⚙️ Config | 📈 Analytics        │
├─────────────────────────────────────────────────┤
│ 🎮 Contrôles | 📡 Monitoring | 📝 Logs          │
└─────────────────────────────────────────────────┘
```

### 1. 📊 Onglet Dashboard

**Informations affichées:**
- Statut du bot (démarré/arrêté/pause)
- HP/PA/PM actuels
- Position sur carte
- État combat
- Statistiques session

**Indicateurs:**
- 🟢 Vert = Tout va bien
- 🟡 Orange = Attention (HP bas, etc.)
- 🔴 Rouge = Danger/Erreur

### 2. ⚙️ Onglet Configuration

**Paramètres disponibles:**

#### Section Général
- **Classe personnage:** IOP, CRA, ENIRIPSA, IOPS, ECA, SRAM
- **Mode observation:** ON/OFF (⚠️ Laisser ON!)
- **FPS cible:** 5-30 (10 recommandé)

#### Section Combat
- **Stratégie cible:**
  - HP le plus bas (débutant)
  - Menace la plus élevée (avancé)
  - Le plus proche (rapide)
  - Défense la plus faible (optimal)

- **Seuil HP critique:** 20-40% (30% recommandé)
- **PA à réserver fuite:** 0-6 (2 recommandé)

#### Section Métiers
- ✅ Farmer (Niveau 1+)
- ✅ Bûcheron (Niveau 1+)
- ✅ Mineur (Niveau 1+)
- ✅ Alchimiste (Niveau 1+)

### 3. 📈 Onglet Analytics

**Graphiques temps réel:**
- XP gagnée (dernière heure)
- Kamas gagnés
- Combats effectués
- Ressources récoltées

**Tableaux:**
- Top actions (par fréquence)
- Monstres combattus
- Profit/heure

### 4. 🎮 Onglet Contrôles

**Boutons principaux:**

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  ▶️ START    │  │  ⏸️ PAUSE    │  │  ⏹️ STOP    │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Contrôles avancés:**
- 🔄 Reset état
- 📸 Calibration
- 💾 Sauvegarder config
- 📤 Exporter logs

### 5. 📡 Onglet Monitoring

**Temps réel:**
- Vision: FPS capture, latence détection
- Brain: Décisions/minute, temps raisonnement
- Actions: Succès/échecs, queue d'actions
- Safety: Blocages, warnings

**Santé système:**
- CPU: %
- RAM: MB utilisés
- GPU: % (si disponible)
- Threads: actifs

### 6. 📝 Onglet Logs & Apprentissage

#### Section Logs
**Filtres:**
- Tous les niveaux
- INFO uniquement
- Warnings uniquement
- Erreurs uniquement
- Décisions uniquement

**Actions:**
- 🗑️ Clear logs
- 💾 Exporter logs (.txt ou .json)
- Auto-scroll ON/OFF

#### Section Apprentissage

**Tableau décisions:**
```
Heure    | Action          | Raison                 | Statut
---------|-----------------|------------------------|--------
12:30:45 | engage_monster  | Farm optimal target    | ✅
12:31:10 | move            | Tactical positioning   | ⏳
12:31:25 | spell_cast      | Burst combo IOP        | ✅
```

**Feedback:**
1. Sélectionner une décision
2. Voir détails complets
3. Donner feedback:
   - ✅ Correct
   - ❌ Incorrect
   - 🔄 À améliorer
4. Ajouter commentaire (optionnel)
5. Suggérer action correcte (optionnel)
6. 💾 Soumettre

**Statistiques apprentissage:**
- Total feedbacks soumis
- Taux de réussite (% correct)
- Corrections apportées
- Amélioration progressive

---

## 🎯 FONCTIONNALITÉS

### 1. Mode Observation

**🛡️ Mode le plus sûr - Recommandé**

**Ce qu'il fait:**
- Observe le jeu
- Prend des décisions théoriques
- Log tout sans agir
- 0% risque d'actions indésirables

**Utilisation:**
```bash
python launch_autonomous_full.py --duration 30
```

**Sortie:**
- `logs/observation.json` - Toutes les décisions
- `logs/autonomous_full.log` - Logs détaillés

**Analyse:**
```python
import json

with open('logs/observation.json') as f:
    decisions = json.load(f)

print(f"Décisions: {len(decisions)}")
print(f"Types: {set(d['action_type'] for d in decisions)}")
```

### 2. Farming Automatique

**⚠️ Mode actif - Compte jetable uniquement**

**Stratégies disponibles:**
1. **Farm Tofus (Niveau 1-10)**
   - Zone: Plaines d'Astrub
   - XP/h: ~500
   - Kamas/h: ~1000

2. **Farm Forêt (Niveau 15-30)**
   - Zone: Forêt d'Astrub
   - XP/h: ~3000
   - Kamas/h: ~6000
   - Métiers: Bûcheron + Alchimiste

3. **Farm Cimetière (Niveau 20-40)**
   - Zone: Cimetière d'Astrub
   - XP/h: ~6000
   - Kamas/h: ~10000

**Configuration:**
```python
# Dans l'interface ou config file
target_zone = "astrub_forest"
target_monsters = ["Moskito", "Arakne"]
farm_duration = 1800  # 30 minutes
```

### 3. Économie & Craft

#### Analyse de Marché

**Accès:** Onglet Analytics → Market Analysis

**Fonctions:**
1. **Scan HDV**
   - Détecte prix automatiquement
   - Historise en base de données
   - Détecte anomalies

2. **Prédictions ML**
   - Prédit prix futurs (7 jours)
   - Intervalle de confiance
   - Tendance (hausse/baisse/stable)

3. **Arbitrage**
   - Compare serveurs
   - Calcule profits nets
   - Évalue risques

**Exemple:**
```
Item: Blé
Prix actuel: 15k
Prédiction 7j: 18k (+20%)
Confiance: 87%
→ Recommandation: ACHETER
```

#### Optimisation Craft

**Accès:** Onglet Analytics → Crafting Optimizer

**Fonctions:**
1. **Analyse rentabilité**
   - Profit par craft
   - ROI %
   - XP/heure vs Profit/heure

2. **Queue optimisée**
   - Ordonne crafts par objectif
   - Vérifie ressources
   - Suggère acquisitions

3. **Plans multi-craft**
   - Optimise séquences
   - Minimise déplacements
   - Maximise synergies

**Exemple plan:**
```
Objectif: 100 Pains Complets
└─ Besoin: 1000 Blé + 500 Eau
   ├─ Blé: 700 en stock, 300 à acheter (4500k)
   └─ Eau: 500 à récolter (30 min)
Profit estimé: 12000k
Temps total: 2h15
XP Boulanger: +12500
```

### 4. Quêtes Automatiques

**⚠️ Fonctionnalité expérimentale**

**Quêtes supportées:**
- Tutorial Incarnam
- Quêtes Astrub (niveau 1-30)
- Farming loops

**Workflow:**
1. Charger quête depuis `data/quests/`
2. Parser objectifs
3. Naviguer zones
4. Dialogues NPCs
5. Combats/récoltes
6. Validation

**Exemple:**
```bash
python -c "
from core.quest_system import QuestManager

qm = QuestManager()
qm.load_quest('tutorial_incarnam.json')
qm.start_quest()
"
```

### 5. Navigation Intelligente

**Système:** Pathfinding A* + Ganymede DB

**Capacités:**
- Calcul chemins optimaux
- Évitement obstacles
- Utilisation Zaaps
- Détection maps intérieures

**API:**
```python
from core.navigation_system import GanymedeNavigator

nav = GanymedeNavigator()

# Trouver chemin
path = nav.find_path(
    start="astrub_center",
    goal="forest_west"
)
# ['astrub_center', 'astrub_west', 'forest_center', 'forest_west']

# Utiliser zaap
zaap_path = nav.use_zaap(
    current="plains_001",
    destination="bonta_center"
)
```

---

## 🧠 SYSTÈME D'APPRENTISSAGE

### Fonctionnement

Le bot **apprend de vos retours** pour améliorer ses décisions.

### Processus

```
1. Bot prend décision
      ↓
2. Affichée dans Logs
      ↓
3. Vous donnez feedback
      ↓
4. Sauvegardé en BDD
      ↓
5. Bot ajuste comportement
```

### Types de Feedback

#### ✅ Décision Correcte
**Quand l'utiliser:**
- Le bot a fait le bon choix
- Action adaptée au contexte
- Résultat positif

**Effet:**
- Renforce cette stratégie
- Augmente confiance décision similaire

**Exemple:**
```
Décision: "Engage Tofu (HP bas)"
Contexte: Player HP=450, PA=6, Tofu visible
→ ✅ CORRECT: Cible facile, bonne décision
```

#### ❌ Décision Incorrecte
**Quand l'utiliser:**
- Mauvais choix évident
- Action dangereuse
- Erreur tactique

**Effet:**
- Pénalise cette stratégie
- Évite répétition

**Exemple:**
```
Décision: "Engage groupe 5 Chafers"
Contexte: Player HP=200/500, Level=20
→ ❌ INCORRECT: Trop dangereux, groupe trop gros
Suggestion: "Fuir ou chercher cible plus facile"
```

#### 🔄 À Améliorer
**Quand l'utiliser:**
- Décision acceptable mais sous-optimale
- Meilleure option disponible
- Timing/détails à ajuster

**Effet:**
- Note pour amélioration
- Recherche alternative

**Exemple:**
```
Décision: "Utiliser sort 1 PA sur Tofu"
Contexte: PA=6, Tofu HP=50
→ 🔄 À AMÉLIORER: Aurait pu utiliser combo 4 PA pour finir plus vite
Suggestion: "Utiliser combo IOP_BURST (4 PA, 200 dmg)"
```

### Commentaires Détaillés

**Bonnes pratiques:**

✅ **BON:**
```
"Bonne cible mais aurait pu se rapprocher avant (économie PM)"
"Parfait, Moskito HP bas = finish rapide"
"Combo optimal, mais attention à sauvegarder PA pour heal si besoin"
```

❌ **À ÉVITER:**
```
"Nul"
"Pas bon"
"Ok"
```

### Statistiques d'Apprentissage

**Métriques:**
- **Taux de réussite:** % décisions correctes
- **Amélioration:** Évolution sur 7/30 jours
- **Par type d'action:** Précision selon contexte

**Graphiques:**
- Courbe apprentissage
- Heatmap erreurs
- Top décisions

**Exemple progression:**
```
Semaine 1: 65% correct
Semaine 2: 72% correct (+7%)
Semaine 3: 78% correct (+6%)
Semaine 4: 84% correct (+6%)
```

### Export Données Apprentissage

**Format JSON:**
```bash
python -c "
from ui.modern_app.logs_learning_panel import LogsLearningPanel
panel.export_logs('learning_data.json')
"
```

**Contenu:**
```json
{
  "total_decisions": 1247,
  "feedbacks": 389,
  "accuracy": 0.84,
  "by_action_type": {
    "engage_monster": {"correct": 145, "incorrect": 12},
    "spell_cast": {"correct": 98, "incorrect": 8}
  }
}
```

---

## ❓ FAQ

### Général

**Q: Le bot peut-il me faire bannir?**
R: En mode observation (par défaut), **risque = 0%**. En mode actif, **risque existe** → utiliser compte jetable uniquement.

**Q: Fonctionne sur Mac/Linux?**
R: Non, Windows uniquement (dépendance win32 API).

**Q: Consomme beaucoup de ressources?**
R:
- CPU: 10-20%
- RAM: 1-2 GB
- GPU: Optionnel (améliore Vision V2)

**Q: Fonctionne en multi-compte?**
R: Une instance = un compte. Lancer plusieurs instances pour multi-compte.

### Technique

**Q: Vision ne détecte rien?**
R:
1. Vérifier fenêtre DOFUS active
2. Relancer calibration: `--calibrate`
3. Vérifier logs pour erreurs

**Q: Combat Engine ne décide rien?**
R:
1. Vérifier classe configurée
2. Vérifier PA/PM détectés
3. Vérifier combat_state.my_turn = True

**Q: Erreur "ImportError: No module named..."?**
R:
```bash
pip install -r requirements.txt --upgrade
```

### Apprentissage

**Q: Combien de feedbacks nécessaires?**
R:
- Minimum: 50-100 pour démarrage
- Optimal: 500+ pour précision
- Idéal: 1000+ feedbacks variés

**Q: Le bot utilise-t-il mes feedbacks immédiatement?**
R:
- Feedbacks sauvegardés en temps réel
- Application au prochain redémarrage
- Ré-entraînement HRM nécessaire pour changements majeurs

**Q: Puis-je importer feedbacks d'autres utilisateurs?**
R: Oui, copier `data/feedback/decisions_feedback.json`

---

## 🔧 DÉPANNAGE

### Problème: Bot ne démarre pas

**Symptômes:**
```
ERROR: Failed to initialize systems
```

**Solutions:**
1. Vérifier Python 3.9+:
   ```bash
   python --version
   ```

2. Réinstaller dépendances:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. Vérifier logs:
   ```bash
   tail -f logs/autonomous_full.log
   ```

### Problème: Vision ne capture rien

**Symptômes:**
- Frame vide
- Erreur "No window found"

**Solutions:**
1. Vérifier DOFUS ouvert et visible
2. Tester capture manuelle:
   ```python
   from core.vision_engine_v2 import create_vision_engine
   v = create_vision_engine()
   frame = v.capture_screen()
   print(f"Frame shape: {frame.shape}")
   ```

3. Changer méthode capture (config):
   ```yaml
   vision:
     capture_method: "mss"  # au lieu de "win32"
   ```

### Problème: Décisions incohérentes

**Symptômes:**
- Actions aléatoires
- Décisions non adaptées

**Solutions:**
1. Vérifier calibration HP/PA/PM:
   ```bash
   python launch_autonomous_full.py --calibrate
   ```

2. Vérifier détection combat:
   ```python
   # Dans logs, chercher:
   vision_data['combat']['in_combat'] = True/False
   ```

3. Donner feedbacks pour améliorer

### Problème: Crash/Freeze

**Symptômes:**
- Application se fige
- Erreur Python

**Solutions:**
1. Réduire FPS:
   ```yaml
   game_engine:
     target_fps: 5  # au lieu de 10
   ```

2. Désactiver systèmes lourds:
   ```yaml
   hrm_reasoning:
     enabled: false
   ```

3. Vérifier RAM disponible:
   ```bash
   # Au moins 2GB libres
   ```

### Logs Utiles

**Localisation:**
- `logs/autonomous_full.log` - Logs principaux
- `logs/observation.json` - Décisions (mode observation)
- `logs/error.log` - Erreurs uniquement

**Analyse:**
```bash
# 10 dernières erreurs
grep "ERROR" logs/autonomous_full.log | tail -10

# Décisions du jour
grep "DECISION" logs/autonomous_full.log | grep "2025-01-15"
```

---

## 📞 SUPPORT

### Ressources

- 📖 **Documentation technique:** `docs/DOCUMENTATION_TECHNIQUE.md`
- ✅ **Check-up système:** `CHECK_UP_COMPLET.md`
- 🚀 **Quick start:** `QUICK_START_FINAL.md`

### Contact

- 🐛 **Bugs:** Créer issue sur GitHub
- 💬 **Questions:** Discord server (lien dans README)
- 📧 **Email:** support@alphastar-dofus.com

---

## 🎓 TUTORIELS VIDÉO

### Débutant

1. **Installation complète** (10 min)
   - Téléchargement
   - Installation Python
   - Premier lancement

2. **Interface utilisateur** (15 min)
   - Tour des onglets
   - Configuration basique
   - Premier test observation

3. **Système apprentissage** (12 min)
   - Donner feedbacks
   - Interpréter statistiques
   - Améliorer décisions

### Intermédiaire

4. **Farming optimisé** (20 min)
   - Configurer zones
   - Sélection stratégies
   - Analyse résultats

5. **Économie & Craft** (25 min)
   - Scan HDV
   - Prédictions ML
   - Optimisation craft

6. **Navigation avancée** (18 min)
   - Pathfinding
   - Zaaps
   - Quêtes

### Avancé

7. **Customisation HRM** (30 min)
   - Entraînement modèle
   - Tuning hyperparamètres
   - Optimisation GPU

8. **Développement plugins** (45 min)
   - Architecture système
   - Créer module custom
   - Intégration Brain

---

**Dernière mise à jour:** Janvier 2025
**Version guide:** 1.0.0

---

**Bon farming et amusez-vous bien! 🎮✨**
