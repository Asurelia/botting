# 🧪 Guide de Test et Intégration - DOFUS Unity World Model AI

**Version 2025.1.0** - Documentation complète des nouvelles fonctionnalités

## 📋 Vue d'Ensemble

Ce guide documente les nouvelles fonctionnalités ajoutées au système DOFUS Unity World Model AI, incluant les stratégies de test offline, l'apprentissage passif, et l'intégration d'applications externes.

## 🚀 Démarrage Rapide

### Lanceur Unifié

```bash
# Lancement avec menu interactif
python launch_unified_system.py

# Modes directs
python launch_unified_system.py --mode tests
python launch_unified_system.py --mode passive --duration 600
python launch_unified_system.py --mode ganymede
python launch_unified_system.py --mode all
```

## 🧪 Environnement de Test Synthétique

### Fonctionnalités

- **Génération de screenshots DOFUS simulés** avec interface complète
- **États de jeu mockés** pour tests sans connexion serveur
- **Séquences cohérentes** d'exploration et combat
- **Base de données de test** persistante

### Utilisation

```python
from tests.test_synthetic_environment import SyntheticTestEnvironment

# Créer environnement
test_env = SyntheticTestEnvironment("test_data")

# Générer dataset complet
config = test_env.setup_complete_test_environment(
    num_screenshots=100,
    num_state_sequences=20
)

# Données disponibles dans :
# - test_data/screenshots/
# - test_data/state_sequences.json
# - test_data/test_environment_config.json
```

### Structure des Screenshots Synthétiques

- **Interface de combat** : Grille tactique 15x17 avec positions joueur/ennemi
- **Barres de statistiques** : HP/MP/AP avec valeurs réalistes
- **Barre de sorts** : 8 sorts Iop avec raccourcis clavier
- **Fenêtre de chat** : Messages de combat simulés
- **Métadonnées** : Classe, niveau, état combat

## 🔍 Mode Apprentissage Passif

### Fonctionnalités

- **Capture temps réel** de DOFUS Unity sans interaction
- **Détection de patterns** : Séquences de sorts, mouvements tactiques
- **Anti-détection avancée** : Délais variables, comportement humain
- **Métriques temps réel** : FPS, qualité session, patterns appris

### Utilisation

```python
from core.vision_engine.passive_learning_mode import get_passive_learning_engine

# Créer moteur
engine = get_passive_learning_engine()

# Démarrer session
session_id = engine.start_passive_learning_session("IOPS", 150, "test")

# L'apprentissage se fait automatiquement
# Surveillance fenêtre DOFUS + analyse patterns

# Arrêter session
session = engine.stop_passive_learning_session()

print(f"Captures: {session.total_captures}")
print(f"Patterns: {session.patterns_learned}")
print(f"Qualité: {session.data_quality_score}")
```

### Patterns Détectés

1. **Séquences de sorts** : Optimisation AP/dégâts
2. **Mouvements tactiques** : Positionnement optimal
3. **Gestion ressources** : HP/MP critique

### Configuration

```python
config = {
    "capture_interval": 0.5,        # Capture toutes les 500ms
    "max_session_duration": 3600,   # 1 heure max
    "min_confidence_threshold": 0.7,
    "save_screenshots": True,
    "pattern_detection_enabled": True,
    "anti_detection_delays": True
}
```

## 🔗 Connecteurs API Externes

### APIs Supportées

#### Dofapi.fr (API non-officielle)
- **Sorts par classe** : Tous les sorts avec statistiques complètes
- **Items par catégorie** : Equipment, weapons, consumables
- **Recherche d'items** : Par nom avec correspondance partielle

#### Doduapi (api.dofusdu.de)
- **Encyclopedia complète** : Items, sets, équipements
- **Multi-langue** : FR/EN support
- **Images** : URLs des icônes items

#### Ganymede Guides (ganymede-app.com)
- **Guides communautaires** : Scraping intelligent des guides DOFUS
- **Catégories multiples** : Quêtes, donjons, métiers, PvP
- **Enrichissement contextuel** : Adaptation selon classe/niveau joueur
- **Cache local SQLite** : Synchronisation et recherche rapide

### Utilisation

```python
from core.external_integration.dofus_api_connector import get_unified_api_manager

# Gestionnaire unifié
manager = get_unified_api_manager()

# Recherche comprehensive
response = manager.get_comprehensive_item_data("Dofus")
if response.success:
    print(f"Items trouvés: {len(response.data)}")

# Sorts de classe
response = manager.get_class_spells_comprehensive("iop")
if response.success:
    for spell in response.data:
        print(f"{spell['name']}: {spell['ap_cost']} AP")

# Statistiques
stats = manager.get_statistics()
print(f"Cache: {stats['cache_size']} entrées")

# Guides Ganymede
from core.external_integration.ganymede_guides_connector import get_ganymede_connector

ganymede = get_ganymede_connector()

# Synchroniser guides localement
stats = ganymede.sync_guides_database(categories=["quetes", "donjons"])
print(f"Guides synchronisés: {stats['guides_updated']}")

# Rechercher guide par sujet
guide = ganymede.get_guide_by_topic("temple obscur")
if guide:
    print(f"Guide trouvé: {guide['title']}")
    print(f"Étapes: {len(guide['steps'])}")

# Recherche générale
results = ganymede.search_guides("iop")
print(f"Guides Iop trouvés: {len(results)}")
```

### Cache Intelligent

- **Cache mémoire** : 1 heure par défaut
- **Cache SQLite** : Persistant entre sessions
- **Mise à jour automatique** : Détection changements API
- **Rate limiting** : Protection contre spam

## 🕵️ Framework Reverse Engineering

### Fonctionnalités

- **Détection processus** : Ganymede, DOFUS Guide, applications cibles
- **Capture réseau** : Interception trafic HTTP/HTTPS
- **Analyse comportementale** : CPU, mémoire, fichiers ouverts
- **Détection APIs** : Endpoints automatiques, formats de réponse

### Utilisation

```python
from core.external_integration.reverse_engineering_framework import get_reverse_engineering_framework

# Créer orchestrateur
orchestrator = get_reverse_engineering_framework()

# Découvrir applications
processes = orchestrator.discover_target_applications()

if processes:
    # Analyser application
    analysis = orchestrator.analyze_application(
        processes[0],
        capture_duration=120
    )

    # Générer guide d'intégration
    guide = orchestrator.generate_integration_guide(analysis['session_id'])
```

### Analyse Réseau

- **Capture Scapy** : Trafic TCP/UDP sur ports communs
- **Détection HTTP** : Requêtes/réponses automatiques
- **Pattern API** : JSON, REST endpoints
- **Export données** : JSON avec métadonnées

### Recommandations Automatiques

Le système génère automatiquement :
- **Endpoints détectés** : URLs et méthodes HTTP
- **Formats de données** : JSON/XML identification
- **Connexions externes** : Services tiers utilisés
- **Fichiers config** : Configurations découvertes

## 📊 Intégration avec Système Existant

### Knowledge Base

```python
# Enrichissement avec APIs externes et guides Ganymede
from core.knowledge_base.knowledge_integration import get_knowledge_base

kb = get_knowledge_base()

# Les connecteurs API alimentent automatiquement la KB
# Mise à jour sorts, items, données économiques

# Interroger guides Ganymede depuis la KB
guide_result = kb.query_ganymede_guide("donjon royal")
if guide_result.success:
    guide = guide_result.data
    print(f"Guide: {guide['title']}")
    print(f"Adaptations contextuelles: {guide.get('level_adaptation', 'Aucune')}")

# Recommandations contextuelles automatiques
recommendations = kb.get_contextual_guide_recommendations()
if recommendations.success:
    for rec in recommendations.data['recommendations']:
        print(f"- {rec['topic']}: {rec['guide']['title']}")

# Synchronisation guides depuis KB
sync_stats = kb.sync_ganymede_guides(categories=["donjons", "pvp"])
print(f"Synchronisation: {sync_stats}")
```

### Learning Engine

```python
# Apprentissage passif intégré
from core.learning_engine.adaptive_learning_engine import get_learning_engine

engine = get_learning_engine()

# Les patterns détectés en mode passif enrichissent l'apprentissage
# Optimisation automatique des stratégies
```

### HRM Integration

```python
# Données externes dans les décisions
from core.world_model.hrm_dofus_integration import DofusIntelligentDecisionMaker

decision_maker = DofusIntelligentDecisionMaker()

# Utilise APIs externes pour contexte enrichi
# Recommandations basées sur données communautaires
```

## 🔧 Configuration Avancée

### Variables d'Environnement

```bash
# APIs externes
export DOFAPI_CACHE_DURATION=3600
export DODUAPI_LANGUAGE=fr

# Apprentissage passif
export PASSIVE_CAPTURE_INTERVAL=0.5
export PASSIVE_MAX_DURATION=7200

# Reverse engineering
export RE_CAPTURE_PORTS="80,443,8080,8443"
export RE_OUTPUT_DIR="reverse_data"
```

### Fichiers de Configuration

#### `config/api_settings.json`
```json
{
  "dofapi": {
    "base_url": "https://api.dofapi.fr",
    "timeout": 10,
    "cache_duration": 3600
  },
  "doduapi": {
    "base_url": "https://api.dofusdu.de",
    "timeout": 15,
    "cache_duration": 3600
  }
}
```

#### `config/passive_learning.json`
```json
{
  "capture_settings": {
    "interval": 0.5,
    "max_screenshots": 1000,
    "save_screenshots": true
  },
  "pattern_detection": {
    "enabled": true,
    "min_confidence": 0.7,
    "spell_sequences": true,
    "movement_patterns": true
  }
}
```

## 🚨 Sécurité et Anti-Détection

### Apprentissage Passif

- **Vision uniquement** : Aucune interaction avec le jeu
- **Délais humanisés** : Randomisation des timings
- **Capture respectueuse** : Limitation FPS, ressources
- **Logs sécurisés** : Aucune donnée sensible

### Reverse Engineering

- **Analyse locale** : Aucune donnée transmise
- **Permissions utilisateur** : Respect des droits d'accès
- **Capture éthique** : Applications utilisateur uniquement
- **Anonymisation** : Données personnelles exclues

## 📈 Métriques et Monitoring

### Dashboard Temps Réel

```python
# Métriques apprentissage passif
metrics = engine.get_current_metrics()
print(f"Captures/min: {metrics['real_time_metrics']['captures_per_minute']}")
print(f"Patterns: {metrics['patterns_learned']}")
print(f"Qualité: {metrics['real_time_metrics']['session_quality']}")

# Statistiques APIs
stats = manager.get_statistics()
print(f"Requêtes Dofapi: {stats['current_session']['dofapi_requests']}")
print(f"Cache hits: {stats['current_session']['cache_hits']}")
```

### Logging Avancé

- **Niveaux configurables** : DEBUG, INFO, WARNING, ERROR
- **Rotation automatique** : Fichiers journaliers
- **Métriques performance** : Temps réponse, erreurs
- **Alertes système** : Seuils configurable

## 🔄 Workflow Recommandé

### 1. Test Initial
```bash
python launch_unified_system.py --mode tests
```

### 2. Environnement Synthétique
```bash
python launch_unified_system.py --mode synthetic
```

### 3. Apprentissage Passif
```bash
# Démarrer DOFUS Unity puis :
python launch_unified_system.py --mode passive --duration 900
```

### 4. Tests Guides Ganymede
```bash
python launch_unified_system.py --mode ganymede
```

### 5. Analyse Applications (Optionnel)
```bash
# Démarrer Ganymede/DOFUS Guide puis :
python launch_unified_system.py --mode reverse
```

### 6. Interface Complète
```bash
python launch_unified_system.py --mode gui
```

## 🐛 Dépannage

### Problèmes Courants

#### Apprentissage Passif
- **Fenêtre DOFUS non trouvée** : Vérifier que DOFUS Unity est ouvert
- **Captures vides** : Ajuster seuils de confiance OCR
- **Performance lente** : Réduire fréquence capture

#### APIs Externes
- **Timeouts** : Vérifier connexion internet
- **Rate limiting** : Respecter délais entre requêtes
- **Cache corrompu** : Supprimer `external_apis_cache.db`

#### Guides Ganymede
- **Guides non trouvés** : Vérifier connexion internet et URL ganymede-app.com
- **Scraping échoue** : Structure HTML peut avoir changé, vérifier logs
- **Cache corrompu** : Supprimer `ganymede_guides_cache.json` et `ganymede_knowledge.db`

#### Reverse Engineering
- **Privilèges insuffisants** : Lancer en administrateur
- **Processus non détectés** : Vérifier noms dans `target_processes`
- **Capture réseau échoue** : Installer Npcap/WinPcap

### Commandes Debug

```bash
# Tests avec logs détaillés
python -u tests/test_complete_system.py 2>&1 | tee debug.log

# Mode passif debug
python -c "
from core.vision_engine.passive_learning_mode import get_passive_learning_engine
engine = get_passive_learning_engine()
# Activer debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
"
```

## 🚀 Performances

### Optimisations AMD 7800XT

- **DirectML** : Accélération IA native Windows
- **ROCm fallback** : Support Linux/Docker
- **Cache GPU** : Réutilisation modèles chargés
- **Batch processing** : Traitement par lots optimisé

### Métriques Typiques

- **Apprentissage passif** : 2-3 FPS capture, <100MB RAM
- **APIs externes** : <200ms latence, 95% cache hit
- **Reverse engineering** : 1000+ paquets/min analysés
- **Tests synthétiques** : 50 screenshots/5s génération

## 📚 Références

### Documentation Technique
- [API Reference](API_REFERENCE.md) - Référence complète des APIs
- [Architecture](ARCHITECTURE.md) - Architecture système détaillée
- [Performance](PERFORMANCE.md) - Optimisations et benchmarks

### APIs Externes
- [Dofapi Documentation](https://dofapi.fr) - API non-officielle DOFUS
- [Doduapi GitHub](https://github.com/dofusdude/doduapi) - API encyclopedia ouverte

### Outils Reverse Engineering
- [Scapy Documentation](https://scapy.net) - Capture et analyse réseau
- [Burp Suite](https://portswigger.net/burp) - Proxy web pour APIs

---

## 🎯 Conclusion

Les nouvelles fonctionnalités transforment le système en une plateforme complète de test, apprentissage et intégration pour DOFUS Unity. L'approche modulaire permet une utilisation flexible selon les besoins spécifiques.

**Points Clés :**
- ✅ **Tests offline complets** sans connexion serveur
- ✅ **Apprentissage passif sécurisé** avec anti-détection
- ✅ **Intégration APIs communautaires** automatique
- ✅ **Reverse engineering éthique** applications tierces
- ✅ **Lanceur unifié** pour tous les modes

**Prochaines Étapes :**
1. Tester chaque mode individuellement
2. Configurer selon vos besoins spécifiques
3. Intégrer dans votre workflow de développement
4. Monitorer les performances et ajuster

> *"L'intelligence artificielle au service de l'excellence tactique DOFUS"* 🎮🤖