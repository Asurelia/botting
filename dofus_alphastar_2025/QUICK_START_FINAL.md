# 🚀 QUICK START - DOFUS AlphaStar 2025

**Bot autonome type humain avec IA avancée**

---

## ⚡ DÉMARRAGE RAPIDE (2 minutes)

### 1. Prérequis
```bash
# Python 3.10 ou 3.11
python --version

# Vérifier dépendances
pip install torch numpy opencv-python pytesseract pyautogui networkx
```

### 2. Tests du système
```bash
cd dofus_alphastar_2025

# Vérifier que tout fonctionne
pytest tests/ -v
# Résultat attendu: 60/63 passing (95%)
```

### 3. Premier lancement (MODE SÉCURISÉ)
```bash
# Session 1 minute en mode observation
python launch_autonomous_full.py --duration 1

# Résultat: 30 décisions prises, aucune action exécutée
# Logs: logs/observation.json
```

---

## 🎮 UTILISATION

### Mode Observation (RECOMMANDÉ pour débuter)
```bash
# Session 30 minutes
python launch_autonomous_full.py --duration 30

# Avec calibration automatique
python launch_autonomous_full.py --calibrate --duration 30
```

### Mode Observation Simple
```bash
# Version simplifiée (sans systèmes avancés)
python launch_safe.py --observe 10
```

### Interface Graphique
```bash
# Dashboard moderne
python launch_ui.py
```

---

## 🧠 SYSTÈMES INTÉGRÉS

Le launcher `launch_autonomous_full.py` intègre automatiquement :

✅ **HRM Reasoning** (108M paramètres)
- System 1 : Décisions intuitives rapides
- System 2 : Raisonnement complexe

✅ **Vision Engine V2**
- SAM 2 : Segmentation avancée
- TrOCR : OCR de nouvelle génération

✅ **Quest System**
- Gestion quêtes intelligente
- Intégration Ganymede

✅ **Professions**
- 4 métiers : Farmer, Lumberjack, Miner, Alchemist
- Synergies automatiques

✅ **Navigation**
- Ganymede maps
- Pathfinding A* optimisé

✅ **Intelligence**
- Détection opportunités
- Apprentissage passif
- Simulation fatigue

✅ **Safety**
- Mode observation par défaut
- Blocage 100% actions
- Logs complets

---

## 📊 COMPRENDRE LES LOGS

### Pendant l'exécution
```
[10] Session: 18s / 60s (reste: 42s)
  Stats: 9 decisions, 0 moves
```
- Itération 10
- 18 secondes écoulées
- 9 décisions prises
- Aucun mouvement (mode observation)

### Fin de session
```
Duree totale: 60.0s (1.0 minutes)
Decisions prises: 30
Actions interceptees: 30
Logs sauvegardes: logs/observation.json
```

### Analyser les observations
```bash
# Voir les décisions prises
cat logs/observation.json

# Statistiques
python -c "
import json
with open('logs/observation.json') as f:
    data = json.load(f)
    print(f'Total decisions: {len(data)}')
    print(f'Types: {set(d[\"action_type\"] for d in data)}')
"
```

---

## ⚠️ SÉCURITÉ

### Mode Observation (Défaut)
- ✅ AUCUNE action exécutée
- ✅ Seulement observation et logs
- ✅ Analyse comportement
- ✅ 100% sécurisé

### Mode Actif (DANGER)
```bash
# Nécessite confirmation explicite
python launch_autonomous_full.py --active --duration 5

# Prompt de sécurité apparaît:
# "Taper 'OUI JE COMPRENDS LES RISQUES' pour continuer"
```

**⚠️ ATTENTION MODE ACTIF:**
- ❌ Compte jetable OBLIGATOIRE
- ❌ Risque de ban PERMANENT
- ❌ Sessions courtes (<10 min)
- ❌ Surveillance constante

---

## 🔧 RÉSOLUTION PROBLÈMES

### "ModuleNotFoundError"
```bash
# Installer dépendances manquantes
pip install -r requirements.txt
```

### "Tests échouent"
```bash
# Vérifier environnement
python -c "import torch, numpy, cv2; print('OK')"

# Relancer tests
pytest tests/ -v --tb=short
```

### "Fenêtre Dofus non détectée"
```bash
# Vérifier fenêtre ouverte
python -c "
import pyautogui
windows = pyautogui.getWindowsWithTitle('Dofus')
print(f'Fenêtres trouvées: {len(windows)}')
"

# Lancer calibration
python launch_autonomous_full.py --calibrate
```

### "UnicodeEncodeError"
- Normal sur Windows avec emojis
- Déjà corrigé dans le code
- Si persiste : vérifier encodage console

---

## 📈 PROGRESSION RECOMMANDÉE

### Semaine 1 : Observation
```bash
# Jour 1-2 : Tests courts (1-5 min)
python launch_autonomous_full.py --duration 1
python launch_autonomous_full.py --duration 5

# Jour 3-4 : Sessions moyennes (10-30 min)
python launch_autonomous_full.py --duration 10
python launch_autonomous_full.py --duration 30

# Jour 5-7 : Analyses
# - Étudier logs/observation.json
# - Comprendre décisions prises
# - Ajuster si nécessaire
```

### Semaine 2 : Données
```bash
# Créer données de quêtes
mkdir -p data/quests
# Ajouter quêtes Ganymède

# Créer données de maps
mkdir -p data/maps
# Ajouter topologie Ganymède

# Créer guides
mkdir -p data/guides
# Ajouter guides farming/leveling
```

### Semaine 3+ : Tests Réels
```bash
# Avec compte jetable uniquement !
# Sessions 5-10 minutes max
# Mode observation toujours actif au début
```

---

## 🎯 COMMANDES UTILES

### Tests
```bash
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_safety.py -v
pytest tests/test_map_system.py -v

# Avec couverture
pytest tests/ --cov=core --cov-report=html
```

### Imports
```bash
# Vérifier systèmes avancés
python -c "
from core.hrm_reasoning import DofusHRMAgent
from core.vision_engine_v2 import create_vision_engine
from core.quest_system import QuestManager
from core.professions import ProfessionManager
from core.navigation_system import GanymedeNavigator
from core.intelligence import OpportunityManager
print('✅ Tous les systèmes OK')
"
```

### Logs
```bash
# Logs bot
tail -f logs/autonomous_full.log

# Logs observations
tail -f logs/observation.json

# Nettoyer logs
rm logs/*.log logs/*.json
```

---

## 📚 DOCUMENTATION

- **ARCHITECTURE_REELLE.md** : Architecture complète
- **GUIDE_DEMARRAGE.md** : Guide détaillé
- **README.md** : Vue d'ensemble
- **tests/** : Exemples d'utilisation

---

## 🆘 SUPPORT

### Problème technique
1. Vérifier tests : `pytest tests/ -v`
2. Vérifier imports (commande ci-dessus)
3. Consulter ARCHITECTURE_REELLE.md

### Amélioration
1. Créer issue avec détails
2. Proposer amélioration
3. Tester avant commit

### Contribution
```bash
# Créer branche
git checkout -b feature/ma-feature

# Développer + tests
# ...

# Commit
git commit -m "feat: ma nouvelle fonctionnalité"

# Push
git push origin feature/ma-feature
```

---

## ✅ CHECKLIST PREMIÈRE UTILISATION

- [ ] Python 3.10+ installé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Tests passent (`pytest tests/ -v`)
- [ ] Imports OK (commande vérification ci-dessus)
- [ ] Launcher testé (`python launch_autonomous_full.py --duration 1`)
- [ ] Logs générés (`logs/observation.json`)
- [ ] Documentation lue (au moins ARCHITECTURE_REELLE.md)

---

**Prêt à utiliser !** 🚀

Le bot est maintenant opérationnel en mode observation. Profite des systèmes avancés tout en restant en sécurité.

**Rappel important :** Toujours utiliser sur compte jetable uniquement !

---

**Créé par Claude Code - Septembre 2025**
