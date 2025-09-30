# 👤 USER GUIDE - DOFUS Unity World Model AI

**Version 2025.1.0** | **Guide Utilisateur Complet** | **Septembre 2025**

---

## 📋 Table des Matières

1. [Introduction](#-introduction)
2. [Première Utilisation](#-première-utilisation)
3. [Interface Assistant](#-interface-assistant)
4. [Modules et Fonctionnalités](#-modules-et-fonctionnalités)
5. [Scenarios d'Usage](#-scenarios-dusage)
6. [Configuration Avancée](#-configuration-avancée)
7. [Troubleshooting](#-troubleshooting)
8. [Bonnes Pratiques](#-bonnes-pratiques)

---

## 🎯 Introduction

### Qu'est-ce que DOFUS Unity World Model AI ?

DOFUS Unity World Model AI est un système d'intelligence artificielle avancé conçu pour analyser, comprendre et assister dans le jeu DOFUS Unity. Le système combine:

- **Vision par ordinateur** pour analyser l'interface du jeu
- **Base de connaissances** complète sur DOFUS (sorts, monstres, cartes, économie)
- **Apprentissage adaptatif** pour optimiser les stratégies
- **Simulation humaine** pour des interactions naturelles
- **Interface assistant** intuitive pour le contrôle

### À qui s'adresse ce guide ?

Ce guide est destiné aux **joueurs DOFUS** qui souhaitent :
- **Améliorer leur gameplay** avec des analyses intelligentes
- **Optimiser leurs stratégies** de combat et d'économie
- **Apprendre** les mécaniques avancées du jeu
- **Automatiser** certaines tâches répétitives
- **Analyser** les données de marché en temps réel

### Conformité et Responsabilité

⚠️ **IMPORTANT** : Ce système est développé à des fins **éducatives et de recherche**.
- Respectez les **Conditions d'Utilisation** de DOFUS
- Utilisez de manière **responsable** et **éthique**
- Aucune garantie de conformité avec les règles anti-bot
- L'utilisateur est **seul responsable** de l'usage

---

## 🚀 Première Utilisation

### Étape 1 : Installation et Vérification

Assurez-vous d'avoir suivi le [Guide d'Installation](INSTALLATION.md) :

```bash
# Vérification que tout fonctionne
cd dofus_vision_2025
python tests/test_complete_system.py

# Sortie attendue :
# [OK] Vision Engine operationnel
# [OK] Knowledge Base operationnel
# [OK] Learning Engine operationnel
# [OK] Human Simulation operationnel
# [OK] Assistant Interface pret
# [OK] Data Extraction operationnel
```

### Étape 2 : Premier Lancement

#### **Lancement de l'Interface Assistant**
```bash
# Activer l'environnement virtuel
venv_dofus_ai\Scripts\activate  # Windows
source venv_dofus_ai/bin/activate  # Linux/Mac

# Lancer l'interface
python assistant_interface/intelligent_assistant.py
```

#### **Première Configuration**
1. **Sélection du serveur** DOFUS (Julith, Draconiros, etc.)
2. **Profil de personnage** (classe, niveau)
3. **Préférences de sécurité** (profil comportemental)
4. **Modules à activer** (selon vos besoins)

### Étape 3 : Test avec DOFUS

1. **Lancez DOFUS** et connectez-vous
2. **Positionnez la fenêtre** DOFUS visible à l'écran
3. **Cliquez "Détecter DOFUS"** dans l'interface assistant
4. **Vérifiez la reconnaissance** de l'interface

---

## 🎮 Interface Assistant

### Vue d'Ensemble de l'Interface

L'interface assistant est organisée en plusieurs sections :

```
┌─────────────────────────────────────────────────┐
│                  MENU PRINCIPAL                 │
├─────────────────┬───────────────┬───────────────┤
│   VISION        │   KNOWLEDGE   │   LEARNING    │
│   ENGINE        │     BASE      │    ENGINE     │
├─────────────────┼───────────────┼───────────────┤
│ Status: ✅ ON   │ Sorts: 1,247  │ Session: #42  │
│ FPS: 30         │ Monstres: 623 │ Actions: 156  │
│ OCR: 97.3%      │ Cartes: 842   │ Score: 87.4%  │
├─────────────────┴───────────────┴───────────────┤
│                  CONSOLE LOGS                   │
│ [INFO] Vision: Interface DOFUS détectée         │
│ [INFO] Knowledge: Base de données à jour        │
│ [INFO] Learning: Session démarrée              │
├─────────────────────────────────────────────────┤
│           CONTRÔLES UTILISATEUR                 │
│ [Démarrer] [Pause] [Stop] [Config] [Aide]      │
└─────────────────────────────────────────────────┘
```

### Sections Principales

#### **1. Vision Engine Dashboard**
- **Status** : État du moteur de vision (ON/OFF/ERROR)
- **FPS** : Fréquence d'analyse des captures d'écran
- **OCR Precision** : Précision de reconnaissance de texte
- **Window Detection** : Détection de la fenêtre DOFUS
- **Screenshot Preview** : Aperçu de la dernière capture

#### **2. Knowledge Base Panel**
- **Database Stats** : Statistiques des bases de données
- **Query History** : Historique des requêtes récentes
- **Market Data** : Données de marché en temps réel
- **Recommendations** : Suggestions intelligentes
- **Update Status** : État des mises à jour automatiques

#### **3. Learning Engine Monitor**
- **Current Session** : Informations session en cours
- **Performance Metrics** : Métriques d'apprentissage
- **Pattern Recognition** : Patterns détectés
- **Action History** : Historique des actions
- **Efficiency Score** : Score d'efficacité global

#### **4. Human Simulation Settings**
- **Behavior Profile** : Profil comportemental actuel
- **Safety Level** : Niveau de sécurité anti-détection
- **Randomization** : Paramètres de randomisation
- **Timing Controls** : Contrôles de timing
- **Error Simulation** : Simulation d'erreurs humaines

### Raccourcis Clavier

| Raccourci | Action |
|-----------|--------|
| `F1` | Aide et documentation |
| `F2` | Capture d'écran manuelle |
| `F3` | Pause/Reprendre système |
| `F4` | Configuration rapide |
| `F5` | Actualiser données |
| `Ctrl+S` | Sauvegarder configuration |
| `Ctrl+L` | Afficher logs détaillés |
| `Ctrl+Q` | Quitter proprement |
| `Espace` | Pause/Reprendre apprentissage |

---

## 🧠 Modules et Fonctionnalités

### 1. Vision Engine - Analyse Visuelle

#### **Fonctionnalités Principales**
- **Détection Interface** : Reconnaissance automatique des éléments UI
- **Lecture OCR** : Extraction de texte (PV, PA, PM, sorts, etc.)
- **Analyse Combat** : Reconnaissance de la grille de combat
- **État du Jeu** : Extraction complète de l'état actuel

#### **Utilisation Pratique**
```python
# Exemple d'usage manuel
from core.vision_engine import DofusWindowCapture, DofusUnityInterfaceReader

# Capture d'écran
capture = DofusWindowCapture()
screenshot = capture.capture_screenshot()

# Analyse de l'interface
reader = DofusUnityInterfaceReader()
game_state = reader.extract_game_state(screenshot)

print(f"PV: {game_state.player_hp}/{game_state.player_max_hp}")
print(f"PA: {game_state.player_ap}, PM: {game_state.player_mp}")
```

#### **Interface Assistant - Vision**
1. **Mode Manuel** : Analyse à la demande via bouton
2. **Mode Continu** : Analyse en temps réel (configurable FPS)
3. **Mode Combat** : Analyse spécialisée pendant les combats
4. **Calibration** : Ajustement des paramètres de reconnaissance

### 2. Knowledge Base - Base de Connaissances

#### **Types de Requêtes Disponibles**

##### **Requêtes de Sorts**
```python
# Via interface ou code
result = kb.query_optimal_spells(target_type="enemy", distance=2)

# Résultat exemple :
{
    "spells": [
        {
            "name": "Pression",
            "damage": "120-150",
            "ap_cost": 4,
            "range": "1-2",
            "effectiveness": 0.92
        }
    ]
}
```

##### **Requêtes de Stratégie Monstre**
```python
result = kb.query_monster_strategy("Bouftou Royal")

# Résultat exemple :
{
    "strategy": {
        "approach": "distance",
        "priority_spells": ["Pression", "Compulsion"],
        "resistances": {"terre": 20, "feu": -10},
        "ai_pattern": "aggressive_melee"
    }
}
```

##### **Analyse de Marché**
```python
result = kb.query_market_opportunities(server="Julith")

# Résultat exemple :
{
    "opportunities": [
        {
            "item": "Blé",
            "buy_price": 10,
            "sell_price": 15,
            "profit_percent": 50,
            "confidence": 0.87
        }
    ]
}
```

#### **Interface Assistant - Knowledge**
1. **Recherche Interactive** : Barre de recherche pour requêtes
2. **Suggestions Contextuelles** : Recommandations selon situation
3. **Historique** : Historique des requêtes et résultats
4. **Favoris** : Sauvegarde des requêtes fréquentes
5. **Export** : Export des données vers Excel/CSV

### 3. Learning Engine - Apprentissage Adaptatif

#### **Types d'Apprentissage**

##### **Apprentissage de Combat**
- **Séquences de sorts** optimales par situation
- **Positionnement tactique** sur la grille
- **Adaptation aux patterns** ennemis
- **Optimisation PA/PM** par tour

##### **Apprentissage Économique**
- **Patterns de prix** sur les marchés
- **Opportunités d'arbitrage** entre serveurs
- **Tendances saisonnières** des items
- **Stratégies d'investissement** rentables

##### **Apprentissage Comportemental**
- **Rythmes de jeu** personnels
- **Préférences de gameplay** individuelles
- **Adaptation au style** de chaque joueur
- **Prédiction des actions** probables

#### **Interface Assistant - Learning**
1. **Session Management** : Création/gestion des sessions
2. **Real-time Metrics** : Métriques en temps réel
3. **Pattern Viewer** : Visualisation des patterns appris
4. **Performance Graph** : Graphiques de performance
5. **Export Learning** : Export des données d'apprentissage

### 4. Human Simulation - Anti-Détection

#### **Profils Comportementaux**

##### **Profil "Natural" (Recommandé)**
- Mouvements de souris **fluides** avec courbes Bézier
- Délais de réaction **humains** (200-500ms)
- Erreurs occasionnelles **réalistes**
- Rythme de frappe **variable**

##### **Profil "Nervous"**
- Mouvements plus **erratiques**
- Délais de réaction **courts** (100-300ms)
- Taux d'erreur **plus élevé**
- Accélérations **imprévisibles**

##### **Profil "Smooth"**
- Mouvements très **réguliers**
- Délais **constants** optimisés
- Erreurs **minimales**
- Performance **maximale**

#### **Configuration dans l'Interface**
1. **Sélection Profil** : Choix du profil comportemental
2. **Customization** : Ajustement des paramètres individuels
3. **Test Mode** : Mode test pour validation
4. **Safety Level** : Niveau de sécurité anti-détection
5. **Real-time Preview** : Aperçu en temps réel

---

## 📖 Scenarios d'Usage

### Scenario 1 : Assistance Combat PvM

#### **Objectif** : Optimiser les combats contre monstres

#### **Configuration**
1. **Activer Vision Engine** en mode combat
2. **Configurer Knowledge Base** pour votre classe
3. **Démarrer Learning Session** pour combat PvM
4. **Profil Human Simulation** : "Natural"

#### **Utilisation**
1. **Avant Combat** :
   - Système analyse les monstres présents
   - Suggère la stratégie optimale
   - Propose l'ordre des sorts

2. **Pendant Combat** :
   - Analyse en temps réel de la grille
   - Recommandations d'actions par tour
   - Adaptation selon évolution du combat

3. **Après Combat** :
   - Enregistrement des résultats
   - Apprentissage des patterns efficaces
   - Mise à jour des stratégies

#### **Exemple Interface**
```
┌─────────────────────────────────────────────────┐
│                COMBAT ASSISTANT                 │
├─────────────────────────────────────────────────┤
│ Monstres: Bouftou Royal (Niv 50) + 2 Bouftous  │
│ Stratégie: Distance + Focus Royal               │
│ Tours estimés: 4-6                             │
├─────────────────────────────────────────────────┤
│ TOUR 1 - RECOMMANDATIONS:                      │
│ 1. Déplacement: (7,8) pour ligne de vue        │
│ 2. Sort: Pression sur Bouftou Royal            │
│ 3. Fin de tour (2 PA restants)                 │
│                                                 │
│ Confiance: 89% | Dégâts estimés: 120-140       │
└─────────────────────────────────────────────────┘
```

### Scenario 2 : Analyse de Marché

#### **Objectif** : Identifier les opportunités économiques

#### **Configuration**
1. **Activer Knowledge Base** avec focus économie
2. **Configurer serveurs** à analyser
3. **Définir items** d'intérêt
4. **Paramétrer alertes** de prix

#### **Utilisation**
1. **Analyse Temps Réel** :
   - Monitoring continu des prix
   - Détection d'opportunités d'arbitrage
   - Alertes sur fluctuations importantes

2. **Recherche Ciblée** :
   - Requêtes sur items spécifiques
   - Comparaison inter-serveurs
   - Historique et tendances

3. **Stratégie d'Investissement** :
   - Recommandations d'achat/vente
   - Prédictions de tendances
   - Optimisation de portefeuille

#### **Exemple Interface**
```
┌─────────────────────────────────────────────────┐
│              MARKET ANALYZER                    │
├─────────────────────────────────────────────────┤
│ 🔥 OPPORTUNITÉS CHAUDES:                       │
│                                                 │
│ Blé (Julith → Ombre)                           │
│ Achat: 8k | Vente: 12k | Profit: +50%         │
│ Confiance: 92% | Volume: 847 unités           │
│                                                 │
│ Fer (Draconiros → Rushu)                       │
│ Achat: 15k | Vente: 19k | Profit: +27%        │
│ Confiance: 78% | Volume: 234 unités           │
└─────────────────────────────────────────────────┘
```

### Scenario 3 : Formation et Apprentissage

#### **Objectif** : Apprendre les mécaniques avancées

#### **Configuration**
1. **Mode Tutorial** activé
2. **Learning Engine** en mode formation
3. **Explications détaillées** activées
4. **Sauvegarde des leçons** activée

#### **Utilisation**
1. **Analyse Explicative** :
   - Explication des choix recommandés
   - Théorie derrière les stratégies
   - Comparaison d'alternatives

2. **Mode Questions/Réponses** :
   - Possibilité de questionner le système
   - Explications sur les mécaniques
   - Conseils personnalisés

3. **Progression Trackée** :
   - Suivi des améliorations
   - Identification des points faibles
   - Objectifs d'apprentissage

### Scenario 4 : Multi-Comptes (Future)

#### **Objectif** : Gérer plusieurs personnages simultanément

#### **Configuration** (Version future)
1. **Multi-Window Detection**
2. **Coordination Cross-Characters**
3. **Synchronized Learning**
4. **Advanced Human Simulation**

---

## ⚙️ Configuration Avancée

### Fichier de Configuration Principal

Le fichier `.env` contient tous les paramètres configurables :

```bash
# PERFORMANCE
MAX_MEMORY_USAGE=512MB          # Limite mémoire
CACHE_TTL=3600                  # TTL cache (secondes)
WORKER_THREADS=4                # Threads de traitement
SCREENSHOT_QUALITY=95           # Qualité captures (0-100)

# VISION ENGINE
OCR_LANGUAGES=fr,en             # Langues OCR
ANALYSIS_TIMEOUT=5000           # Timeout analyse (ms)
CONFIDENCE_THRESHOLD=0.85       # Seuil confiance OCR

# LEARNING ENGINE
LEARNING_RATE=0.01              # Taux apprentissage
BATCH_SIZE=32                   # Taille batch ML
MODEL_SAVE_INTERVAL=300         # Sauvegarde modèle (sec)

# HUMAN SIMULATION
DEFAULT_BEHAVIOR_PROFILE=natural # Profil par défaut
MOUSE_SPEED_FACTOR=1.0          # Multiplicateur vitesse souris
KEYBOARD_DELAY_FACTOR=1.0       # Multiplicateur délai clavier

# SECURITY
ENABLE_TELEMETRY=false          # Télémétrie (désactivée)
LOG_SENSITIVE_DATA=false        # Logs données sensibles
ENABLE_CRASH_REPORTING=true     # Rapports de crash
```

### Configurations Spécialisées

#### **Pour Gaming Compétitif**
```bash
# Configuration haute performance
SCREENSHOT_QUALITY=100
ANALYSIS_TIMEOUT=1000
CONFIDENCE_THRESHOLD=0.95
DEFAULT_BEHAVIOR_PROFILE=smooth
LEARNING_RATE=0.02
```

#### **Pour Sécurité Maximum**
```bash
# Configuration anti-détection maximale
DEFAULT_BEHAVIOR_PROFILE=nervous
MOUSE_SPEED_FACTOR=0.8
KEYBOARD_DELAY_FACTOR=1.3
ENABLE_TELEMETRY=false
LOG_SENSITIVE_DATA=false
```

#### **Pour Apprentissage Intensif**
```bash
# Configuration apprentissage optimisé
LEARNING_RATE=0.05
BATCH_SIZE=64
MODEL_SAVE_INTERVAL=60
CACHE_TTL=7200
```

### Profils Utilisateur

#### **Création de Profils**
```python
# profiles/competitive_player.json
{
    "name": "Joueur Compétitif",
    "description": "Configuration pour gameplay compétitif",
    "settings": {
        "vision_engine": {
            "fps": 60,
            "quality": "ultra",
            "precision": 0.98
        },
        "learning_engine": {
            "aggressiveness": "high",
            "adaptation_speed": "fast",
            "risk_tolerance": "medium"
        },
        "human_simulation": {
            "profile": "smooth",
            "error_rate": 0.01,
            "speed": 1.2
        }
    }
}
```

#### **Chargement de Profils**
```python
# Interface ou code
assistant.load_user_profile("competitive_player")
```

---

## 🔧 Troubleshooting

### Problèmes Fréquents

#### **1. "DOFUS Window Not Found"**

**Symptômes** :
- Message d'erreur dans l'interface
- Vision Engine en status ERROR
- Captures d'écran vides

**Solutions** :
1. **Vérifier DOFUS ouvert** et visible à l'écran
2. **Titre de fenêtre** : Vérifier que le titre contient "Dofus"
3. **Permissions** : Lancer en administrateur si nécessaire
4. **Configuration** : Ajuster `window_title` dans la config

```python
# Test manuel de détection
from core.vision_engine import DofusWindowCapture
capture = DofusWindowCapture()
info = capture.get_window_info()
print(f"Fenêtre détectée: {info}")
```

#### **2. "OCR Recognition Poor"**

**Symptômes** :
- Précision OCR < 80%
- Texte mal reconnu
- État du jeu incorrect

**Solutions** :
1. **Résolution** : Augmenter la résolution de DOFUS
2. **Zoom Interface** : Ajuster le zoom d'interface
3. **Qualité** : Augmenter `SCREENSHOT_QUALITY`
4. **Langues** : Vérifier `OCR_LANGUAGES`

```python
# Test OCR manuel
from core.vision_engine import DofusUnityInterfaceReader
reader = DofusUnityInterfaceReader()
# Tester avec une capture d'écran
text = reader.read_interface_text(screenshot)
print(f"Texte détecté: {text}")
```

#### **3. "Learning Engine Not Improving"**

**Symptômes** :
- Score d'efficacité stagnant
- Pas d'amélioration des recommandations
- Patterns non détectés

**Solutions** :
1. **Données** : Vérifier suffisamment de données d'entraînement
2. **Variété** : Diversifier les situations d'apprentissage
3. **Paramètres** : Ajuster `LEARNING_RATE`
4. **Reset** : Redémarrer la session d'apprentissage

```python
# Vérification métrics
engine = get_learning_engine()
metrics = engine.get_learning_metrics()
print(f"Données d'entraînement: {metrics['total_samples']}")
print(f"Taux d'amélioration: {metrics['improvement_rate']}")
```

#### **4. "High Memory Usage"**

**Symptômes** :
- Utilisation mémoire > 500MB
- Ralentissements système
- Erreurs out of memory

**Solutions** :
1. **Limite** : Réduire `MAX_MEMORY_USAGE`
2. **Cache** : Réduire `CACHE_TTL`
3. **Qualité** : Réduire `SCREENSHOT_QUALITY`
4. **Threads** : Réduire `WORKER_THREADS`

```python
# Monitoring mémoire
import psutil
import os
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Mémoire utilisée: {memory_mb:.1f} MB")
```

### Diagnostic Automatique

#### **Script de Diagnostic Utilisateur**
```python
# diagnostic_user.py - Script de diagnostic simple
import sys
from pathlib import Path

def run_user_diagnostic():
    """Diagnostic rapide pour utilisateurs"""

    print("=== DIAGNOSTIC UTILISATEUR DOFUS AI ===\n")

    # 1. Vérification installation
    try:
        import core
        print("✅ Installation: OK")
    except ImportError as e:
        print(f"❌ Installation: ERREUR - {e}")
        return

    # 2. Test modules principaux
    modules = [
        ("Vision Engine", "core.vision_engine"),
        ("Knowledge Base", "core.knowledge_base"),
        ("Learning Engine", "core.learning_engine"),
        ("Human Simulation", "core.human_simulation")
    ]

    for name, module in modules:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
        except Exception as e:
            print(f"❌ {name}: ERREUR - {e}")

    # 3. Test base de données
    try:
        from core.knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        print("✅ Base de données: OK")
    except Exception as e:
        print(f"❌ Base de données: ERREUR - {e}")

    # 4. Test configuration
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Configuration: Fichier .env trouvé")
    else:
        print("⚠️ Configuration: Fichier .env manquant")

    print("\n=== FIN DIAGNOSTIC ===")

if __name__ == "__main__":
    run_user_diagnostic()
```

---

## 🎯 Bonnes Pratiques

### Utilisation Responsable

#### **Règles d'Or**
1. **Respecter les ToS** de DOFUS en permanence
2. **Usage éducatif** et d'assistance uniquement
3. **Pas d'automation complète** sans supervision
4. **Pauses régulières** dans l'utilisation
5. **Surveillance des patterns** d'usage

#### **Recommandations Sécurité**
1. **Profil Natural** obligatoire en usage normal
2. **Délais réalistes** entre actions
3. **Variation comportementale** régulière
4. **Logs minimaux** en production
5. **Pas de partage** de configurations sensibles

### Optimisation Performance

#### **Pour Machines Limitées**
1. **Réduire FPS** d'analyse (15-20 FPS)
2. **Qualité screenshot** à 75-85%
3. **Cache TTL** réduit (1800s)
4. **Threads limités** (2-3)
5. **Désactiver profiling** en production

#### **Pour Machines Puissantes**
1. **FPS élevé** (30-60 FPS)
2. **Qualité maximale** (95-100%)
3. **Cache étendu** (7200s)
4. **Multi-threading** optimisé (6-8 threads)
5. **GPU acceleration** si disponible

### Maintenance Régulière

#### **Quotidienne**
1. **Vérifier logs** pour erreurs
2. **Monitoring mémoire** et CPU
3. **Backup automatique** des données apprentissage
4. **Nettoyage cache** si nécessaire

#### **Hebdomadaire**
1. **Update base de données** DOFUS
2. **Validation modèles** apprentissage
3. **Nettoyage logs** anciens
4. **Vérification sauvegardes**

#### **Mensuelle**
1. **Mise à jour dépendances** Python
2. **Optimisation base de données** (VACUUM)
3. **Review configurations** utilisateur
4. **Backup complet** système

### Communauté et Support

#### **Ressources Communautaires**
1. **Discord Server** : Support temps réel
2. **GitHub Issues** : Rapports de bugs
3. **Wiki Community** : Guides partagés
4. **Forums** : Discussions et tips

#### **Contribution**
1. **Partage configurations** efficaces
2. **Reporting bugs** détaillés
3. **Suggestions améliorations**
4. **Documentation** utilisateur

---

## 📊 Métriques Utilisateur

### Tableau de Bord Personnel

Suivez vos performances avec les métriques intégrées :

| Métrique | Description | Valeur Cible |
|----------|-------------|--------------|
| **Efficacité Combat** | % victoires optimales | > 85% |
| **Précision Sorts** | % sorts optimaux utilisés | > 90% |
| **Économie ROI** | Retour investissement marché | > 15% |
| **Temps Analyse** | Temps moyen prise décision | < 2s |
| **Apprentissage** | Vitesse d'amélioration | +5%/semaine |

### Export et Analyse

```python
# Export des données personnelles
from core.learning_engine import get_learning_engine

engine = get_learning_engine()
data = engine.export_user_data()

# Sauvegarde CSV pour analyse externe
import pandas as pd
df = pd.DataFrame(data)
df.to_csv("my_dofus_ai_stats.csv", index=False)
```

---

## 🎓 Formation Continue

### Niveaux d'Expertise

#### **Débutant** (0-2 semaines)
- ✅ Installation et configuration
- ✅ Interface assistant maîtrisée
- ✅ Usage basique modules core
- ✅ Compréhension sécurité

#### **Intermédiaire** (2-8 semaines)
- ✅ Configuration avancée
- ✅ Optimisation performance
- ✅ Customisation profils
- ✅ Analyse données apprentissage

#### **Avancé** (2+ mois)
- ✅ Développement modules custom
- ✅ Intégration APIs externes
- ✅ Contribution code
- ✅ Mentorship communauté

### Objectifs d'Apprentissage

Définissez vos objectifs personnels :

```
□ Maîtriser l'interface assistant (Semaine 1)
□ Optimiser combat PvM avec IA (Semaine 2-3)
□ Analyser marché efficacement (Semaine 4-5)
□ Configurer anti-détection avancée (Semaine 6-7)
□ Contribuer à la communauté (Semaine 8+)
```

---

*Guide Utilisateur maintenu par Claude Code - AI Development Specialist*
*Version 2025.1.0 - Septembre 2025*
*Mis à jour pour refléter les fonctionnalités réelles du système*