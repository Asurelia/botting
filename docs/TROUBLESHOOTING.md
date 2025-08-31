# 🔧 Guide de Résolution de Problèmes - TacticalBot

## 📋 Vue d'Ensemble

Ce guide vous aidera à diagnostiquer et résoudre les problèmes les plus courants avec TacticalBot. Les solutions sont organisées par catégorie avec des étapes de diagnostic progressives.

## 🚨 Diagnostic Rapide

### Vérification Système Rapide

```bash
# Test de fonctionnement global
python tacticalbot.py --test

# Vérification des dépendances
python -c "import numpy, cv2, PIL; print('✅ Dépendances OK')"

# Test de configuration
python -c "from engine.core import BotEngine; print('✅ Configuration OK')"

# Vérification des logs
tail -20 logs/tacticalbot.log
```

### Indicateurs de Santé Système

```python
# Script de diagnostic automatique
python scripts/health_check.py

# Sortie attendue :
# ✅ Moteur : OK (30.2 FPS)
# ✅ Modules : 12/12 actifs
# ✅ Mémoire : 156MB/512MB (30%)
# ✅ Vision : OK (confiance: 94%)
# ⚠️ Réseau : Latence élevée (250ms)
```

---

## 🔧 Problèmes d'Installation

### Erreur : Dépendances Manquantes

**Symptômes**
```
ImportError: No module named 'cv2'
ModuleNotFoundError: No module named 'numpy'
```

**Solutions**
```bash
# Solution 1 : Installation complète
pip install -r requirements.txt

# Solution 2 : Installation manuelle
pip install numpy opencv-python pillow pyautogui keyboard mouse psutil requests

# Solution 3 : Environnement virtuel
python -m venv tacticalbot_env
tacticalbot_env\Scripts\activate
pip install -r requirements.txt

# Solution 4 : Conda (si disponible)
conda create -n tacticalbot python=3.9
conda activate tacticalbot
conda install numpy opencv pillow
pip install pyautogui keyboard mouse psutil requests
```

### Erreur : Python Version

**Symptômes**
```
SyntaxError: invalid syntax (typing features)
AttributeError: module 'typing' has no attribute 'Optional'
```

**Solutions**
```bash
# Vérifier version Python
python --version
# Doit être >= 3.8

# Mise à jour Python
# Windows : Télécharger depuis python.org
# Ubuntu : sudo apt update && sudo apt install python3.9
# macOS : brew install python@3.9

# Utilisation de py launcher (Windows)
py -3.9 tacticalbot.py
```

### Erreur : Permissions

**Symptômes**
```
PermissionError: [Errno 13] Permission denied
FileNotFoundError: [Errno 2] No such file or directory: 'logs/'
```

**Solutions**
```bash
# Windows : Exécuter en administrateur
# Ou créer les dossiers manuellement
mkdir logs
mkdir config
mkdir temp

# Linux/macOS
sudo chmod +x tacticalbot.py
chmod 755 logs/ config/ temp/

# Vérifier les droits d'écriture
python -c "import os; print('Writable:' if os.access('.', os.W_OK) else 'No write access')"
```

---

## 🎮 Problèmes de Démarrage

### Le Bot Ne Démarre Pas

**Symptômes**
```
Engine failed to initialize
No response from bot after start command
```

**Diagnostic Étape par Étape**

1. **Vérification des Logs**
```bash
# Vérifier les derniers logs
tail -50 logs/tacticalbot.log

# Rechercher les erreurs critiques
grep "ERROR\|CRITICAL" logs/tacticalbot.log | tail -10
```

2. **Test des Modules Critiques**
```python
# Test du moteur principal
python -c "
from engine.core import BotEngine
engine = BotEngine()
print('✅ Moteur OK' if engine else '❌ Erreur moteur')
"

# Test du système d'événements
python -c "
from engine.event_bus import EventBus
bus = EventBus()
print('✅ EventBus OK' if bus.start() else '❌ Erreur EventBus')
"
```

3. **Vérification Configuration**
```python
# Test de chargement config
python -c "
import json
with open('config/engine.json') as f:
    config = json.load(f)
print('✅ Config chargée:', len(config), 'sections')
"
```

**Solutions Communes**
```python
# Solution 1 : Reset configuration
from config.config_manager import ConfigManager
config = ConfigManager()
config.reset_to_defaults()

# Solution 2 : Mode minimal
python tacticalbot.py --minimal --no-modules

# Solution 3 : Debug verbose
python tacticalbot.py --debug --verbose --log-level DEBUG
```

### Erreur : Jeu Non Détecté

**Symptômes**
```
GameWindow not found
Unable to attach to game process
Screen capture failed
```

**Diagnostic**
```python
# Test de détection du jeu
from modules.vision.screen_analyzer import ScreenAnalyzer
analyzer = ScreenAnalyzer()

# Vérifier les fenêtres disponibles
windows = analyzer.list_windows()
for window in windows:
    print(f"Fenêtre: {window.title} (PID: {window.pid})")

# Test de capture d'écran
screenshot = analyzer.capture_screen()
print(f"Capture: {screenshot.size if screenshot else 'Échec'}")
```

**Solutions**
1. **Vérifier le titre de fenêtre**
```bash
# Modifier .env avec le bon titre
echo 'GAME_WINDOW_TITLE="Dofus - Pseudo"' > .env
```

2. **Permissions écran (Windows)**
```bash
# Activer l'accès à l'écran dans Paramètres > Confidentialité
# Redémarrer en administrateur si nécessaire
```

3. **Multi-écrans**
```python
# Configuration multi-écrans
from config.config_manager import ConfigManager
config = ConfigManager()
config.set_config("vision.screen_index", 0)  # Écran principal
config.set_config("vision.capture_region", "auto_detect")
```

---

## 📊 Problèmes de Performance

### FPS Faible / Lag

**Symptômes**
```
FPS: 15/30 (target)
Temps de cycle élevé : 120ms
Avertissement performance CPU
```

**Diagnostic Performance**
```python
# Profiling complet
python scripts/performance_profiler.py

# Métriques en temps réel
from engine.core import BotEngine
engine = BotEngine()
stats = engine.get_statistics()

print(f"FPS actuel: {stats['performance']['fps_actual']}")
print(f"Temps cycle: {stats['performance']['loop_time_avg']*1000:.1f}ms")
print(f"Mémoire: {stats['performance']['memory_usage_mb']}MB")
print(f"CPU: {stats['performance']['cpu_usage_percent']:.1f}%")
```

**Solutions d'Optimisation**

1. **Réduction FPS**
```json
// config/engine.json
{
  "engine": {
    "target_fps": 20,          // Au lieu de 30
    "decision_fps": 5,         // Au lieu de 10
    "performance_monitoring": false
  }
}
```

2. **Optimisation Modules**
```python
# Désactiver modules non essentiels
from config.config_manager import ConfigManager
config = ConfigManager()

config.set_config("modules.social.enabled", False)
config.set_config("modules.automation.daily_routine.enabled", False)
config.set_config("vision.template_matching.multi_scale_matching", False)
```

3. **Optimisation Vision**
```json
// config/screen_analysis.json
{
  "screen_capture": {
    "capture_frequency": 20,    // Au lieu de 30
    "optimize_for_speed": true,
    "enable_caching": true
  },
  "image_processing": {
    "preprocessing_enabled": false,
    "noise_reduction": false
  }
}
```

### Utilisation Mémoire Élevée

**Symptômes**
```
Mémoire: 800MB/512MB (limite dépassée)
Memory warning: High usage detected
Garbage collection frequent
```

**Diagnostic Mémoire**
```python
# Analyse mémoire détaillée
import psutil
import gc

process = psutil.Process()
memory_info = process.memory_info()

print(f"Mémoire RSS: {memory_info.rss / 1024 / 1024:.1f}MB")
print(f"Mémoire VMS: {memory_info.vms / 1024 / 1024:.1f}MB")

# Objets en mémoire
gc.collect()
print(f"Objets non collectés: {len(gc.garbage)}")
```

**Solutions Mémoire**
```python
# Configuration mémoire optimisée
{
  "memory": {
    "max_memory_mb": 256,           // Limite plus stricte
    "garbage_collection_interval": 60,  // Plus fréquent
    "cache_cleanup_frequency": 120,     // Plus agressif
    "enable_memory_profiling": true
  },
  "vision": {
    "cache_processed_images": false,
    "template_cache_size": 50      // Au lieu de 200
  }
}
```

---

## 🎯 Problèmes de Modules

### Module en Erreur / Inactif

**Symptômes**
```
Module 'profession_farmer' status: ERROR
Module restarted 3 times
Last error: Template not found
```

**Diagnostic Module**
```python
# État détaillé d'un module
from engine.core import BotEngine
engine = BotEngine()

module = engine.get_module("profession_farmer")
if module:
    print(f"Status: {module.status}")
    print(f"Erreurs: {module._error_count}")
    print(f"Dernière erreur: {module.get_last_error()}")
    print(f"Métriques: {module.get_metrics()}")
```

**Solutions par Type d'Erreur**

1. **Template Not Found**
```python
# Régénérer les templates
from modules.vision.template_matcher import TemplateMatcher
matcher = TemplateMatcher()
matcher.regenerate_templates()
matcher.validate_all_templates()
```

2. **Module Dependency Error**
```python
# Vérifier les dépendances
from engine.core import BotEngine
engine = BotEngine()

dependencies = engine.module_dependencies
for module, deps in dependencies.items():
    print(f"{module} dépend de: {deps}")
    
    # Vérifier que toutes les dépendances sont actives
    missing_deps = [dep for dep in deps if not engine.get_module(dep)]
    if missing_deps:
        print(f"❌ Dépendances manquantes: {missing_deps}")
```

3. **Configuration Error**
```python
# Réinitialiser la configuration d'un module
from config.config_manager import ConfigManager
config = ConfigManager()

# Reset module spécifique
config.reset_module_config("profession_farmer")

# Ou appliquer profil par défaut
config.apply_profile("farmer_safe")
```

### Module Lent / Non Réactif

**Diagnostic Timing**
```python
# Profiling d'un module spécifique
import time
from modules.professions import ProfessionManager

prof_manager = ProfessionManager()

# Test de performance
start_time = time.perf_counter()
result = prof_manager.update(game_state)
execution_time = (time.perf_counter() - start_time) * 1000

print(f"Temps d'exécution: {execution_time:.2f}ms")
if execution_time > 33:  # Plus de 33ms pour 30FPS
    print("⚠️ Module trop lent")
```

**Optimisations Module**
```python
# Configuration optimisée pour modules lents
{
  "modules": {
    "profession_manager": {
      "config": {
        "update_frequency": 10,      // Au lieu de 30
        "route_calculation": "cached", // Au lieu de "realtime"
        "enable_deep_analysis": false
      }
    }
  }
}
```

---

## 👁️ Problèmes de Vision

### Reconnaissance Échouée

**Symptômes**
```
Template matching failed: confidence 0.3 < 0.8
OCR extraction returned empty string
UI element not found: health_bar
```

**Diagnostic Vision**
```python
# Test complet du système vision
from modules.vision.screen_analyzer import ScreenAnalyzer
from modules.vision.template_matcher import TemplateMatcher

# Test capture écran
analyzer = ScreenAnalyzer()
screenshot = analyzer.capture_screen()
if screenshot:
    print(f"✅ Capture OK: {screenshot.size}")
    screenshot.save("test_capture.png")
else:
    print("❌ Capture échouée")

# Test template matching
matcher = TemplateMatcher()
templates = matcher.list_available_templates()
print(f"Templates disponibles: {len(templates)}")

# Test sur template simple
if "ui_health_bar" in templates:
    matches = matcher.find_template("ui_health_bar", confidence=0.6)
    print(f"Matches santé: {len(matches)}")
```

**Solutions Vision**

1. **Recalibrage Templates**
```python
# Régénération templates avec échantillons actuels
from modules.vision.template_matcher import TemplateMatcher

matcher = TemplateMatcher()

# Nouveau training
matcher.start_training_mode()
# Interface : Cliquez sur les éléments à reconnaître
matcher.train_template("health_bar", region=(50, 50, 200, 20))
matcher.train_template("mana_bar", region=(50, 80, 200, 20))
matcher.save_training_session()
```

2. **Ajustement Seuils**
```json
// config/screen_analysis.json
{
  "template_matching": {
    "confidence_threshold": 0.6,     // Au lieu de 0.8
    "multi_scale_matching": true,
    "rotation_tolerance": 10,        // Plus tolérant
    "adaptive_templates": true
  },
  "ocr_settings": {
    "confidence_threshold": 0.5,     // Moins strict
    "preprocessing": "enhanced"       // Meilleur préprocessing
  }
}
```

3. **Résolution/Scaling**
```python
# Configuration pour différentes résolutions
from config.config_manager import ConfigManager
config = ConfigManager()

# Auto-détection résolution
screen_width = 1920  # Votre résolution
scaling_factor = screen_width / 1920  # Facteur de mise à l'échelle

config.set_config("vision.ui_scaling_factor", scaling_factor)
config.set_config("vision.template_scaling", True)
```

### OCR Défaillant

**Diagnostic OCR**
```python
# Test OCR sur région spécifique
from modules.vision.screen_analyzer import ScreenAnalyzer
import PIL.Image

analyzer = ScreenAnalyzer()

# Capture zone de texte
text_region = (100, 100, 300, 50)  # x, y, width, height
region_image = analyzer.capture_region(text_region)

if region_image:
    region_image.save("ocr_test.png")
    
    # Test OCR
    extracted_text = analyzer.ocr_region(text_region)
    print(f"Texte extrait: '{extracted_text}'")
    
    # Test avec préprocessing
    enhanced_text = analyzer.ocr_region(text_region, preprocess=True)
    print(f"Texte amélioré: '{enhanced_text}'")
```

**Amélioration OCR**
```python
# Configuration OCR optimisée
{
  "ocr_settings": {
    "ocr_engine": "tesseract",
    "language": "fra+eng",              // Multi-langues
    "psm": 6,                           // Mode de segmentation
    "oem": 3,                           // Mode OCR
    "confidence_threshold": 0.6,
    "preprocessing": "adaptive_threshold",
    "character_whitelist": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz%/- +",
    "enable_spell_check": true,
    "dictionary": ["vie", "mana", "kamas", "level", "experience"]
  }
}
```

---

## 🤖 Problèmes d'IA et Décision

### Décisions Incohérentes

**Symptômes**
```
Bot switches between actions rapidly
Decision confidence very low (< 0.4)
Conflicting actions from different modules
```

**Diagnostic IA**
```python
# Analyse des décisions récentes
from modules.decision.decision_engine import DecisionEngine

engine = DecisionEngine()
stats = engine.get_decision_stats()

print(f"Décisions totales: {stats['total_decisions']}")
print(f"Taux de succès: {stats['success_rate']:.1%}")
print(f"Confiance moyenne: {stats['average_confidence']:.2f}")

# Dernières décisions
recent_decisions = engine.get_recent_decisions(10)
for decision in recent_decisions:
    print(f"{decision.timestamp}: {decision.action_id} (conf: {decision.confidence:.2f})")
```

**Solutions IA**

1. **Ajustement Poids Critères**
```python
# Réglage des poids de décision
from config.config_manager import ConfigManager
config = ConfigManager()

# Augmenter le poids de la confiance
config.update_setting("decision_engine.criteria_weights.confidence", 2.0)

# Pénaliser plus le risque
config.update_setting("decision_engine.criteria_weights.risk", -2.0)

# Privilégier la stabilité
config.update_setting("decision_engine.conflict_resolution.timeout_resolution", 5.0)
```

2. **Filtrage Décisions**
```python
# Configuration anti-oscillation
{
  "decision_engine": {
    "min_confidence_threshold": 0.6,
    "decision_cooldown": 2.0,           // Délai minimum entre décisions
    "action_change_penalty": 0.2,       // Pénalité pour changer d'action
    "consistency_bonus": 0.1            // Bonus pour actions cohérentes
  }
}
```

### Apprentissage Inefficace

**Diagnostic Apprentissage**
```python
# Analyse de l'apprentissage
from modules.decision.decision_engine import DecisionEngine

engine = DecisionEngine()
learning_stats = engine.get_learning_statistics()

print(f"Échantillons d'apprentissage: {learning_stats['sample_count']}")
print(f"Taux d'amélioration: {learning_stats['improvement_rate']:.1%}")
print(f"Actions les mieux apprises: {learning_stats['best_learned_actions']}")
```

**Optimisation Apprentissage**
```python
# Configuration apprentissage optimisée
{
  "learning_parameters": {
    "learning_rate": 0.05,              // Plus conservateur
    "memory_size": 2000,                // Plus de mémoire
    "adaptation_threshold": 0.1,        // Adaptation plus sensible
    "exploration_rate": 0.15,           // Moins d'exploration
    "batch_learning": true              // Apprentissage par batch
  }
}
```

---

## 🔒 Problèmes de Sécurité

### Détection Suspectée

**Symptômes**
```
Safety alert: Pattern detection warning
Unusual activity detected
Account action suspected
```

**Actions Immédiates**
```python
# Arrêt d'urgence sécurisé
python tacticalbot.py --emergency-stop

# Activation mode furtif
python tacticalbot.py --stealth-mode

# Vérification des patterns
python scripts/behavior_analysis.py --last-session
```

**Analyse Post-Détection**
```python
# Analyse des patterns récents
from modules.safety.detection_avoidance import SafetyManager

safety = SafetyManager()
behavior_report = safety.analyze_recent_behavior(hours=4)

print("Analyse comportementale:")
print(f"Actions répétitives: {behavior_report['repetitive_actions']}")
print(f"Timing suspect: {behavior_report['timing_anomalies']}")
print(f"Patterns détectés: {behavior_report['detected_patterns']}")
```

**Amélioration Sécurité**
```python
# Configuration sécurité renforcée
{
  "detection_avoidance": {
    "randomization_level": 0.9,         // Maximum
    "human_behavior_simulation": true,
    "anti_pattern_detection": true,
    "mistake_injection": 0.05           // 5% d'erreurs simulées
  },
  "session_limits": {
    "max_session_duration": 2.0,       // Sessions plus courtes
    "mandatory_break_interval": 0.75,   // Pauses plus fréquentes
    "min_break_duration": 0.5          // Pauses plus longues
  }
}
```

---

## 📋 Checklist de Diagnostic

### Diagnostic Systématique

1. **✅ Vérifications de Base**
   - [ ] Python >= 3.8 installé
   - [ ] Toutes les dépendances présentes
   - [ ] Fichiers de configuration valides
   - [ ] Permissions d'écriture OK
   - [ ] Logs accessibles et récents

2. **✅ Vérifications Moteur**
   - [ ] Moteur s'initialise sans erreur
   - [ ] Bus d'événements opérationnel
   - [ ] Modules core chargés
   - [ ] FPS stable (>= 25)
   - [ ] Utilisation mémoire < limite

3. **✅ Vérifications Jeu**
   - [ ] Jeu détecté et accessible
   - [ ] Capture d'écran fonctionnelle
   - [ ] Templates reconnus
   - [ ] OCR extrait du texte
   - [ ] État du jeu lu correctement

4. **✅ Vérifications Modules**
   - [ ] Tous modules critiques actifs
   - [ ] Pas d'erreurs de dépendances
   - [ ] Configuration modules valide
   - [ ] Performance acceptable

5. **✅ Vérifications Sécurité**
   - [ ] Randomisation active
   - [ ] Limites de session respectées
   - [ ] Pas d'alertes de patterns
   - [ ] Comportement humain-like

### Script de Diagnostic Automatique

```python
#!/usr/bin/env python
"""
Script de diagnostic automatique TacticalBot
Usage: python diagnostic.py [--verbose] [--fix-common]
"""

import sys
import json
import traceback
from pathlib import Path

def run_diagnostic(verbose=False, fix_common=False):
    """Execute diagnostic complet"""
    
    results = {
        "system": check_system(),
        "dependencies": check_dependencies(), 
        "config": check_configuration(),
        "engine": check_engine(),
        "modules": check_modules(),
        "vision": check_vision(),
        "security": check_security()
    }
    
    # Rapport final
    print("\n📊 RAPPORT DE DIAGNOSTIC")
    print("=" * 50)
    
    total_checks = sum(len(category) for category in results.values())
    passed_checks = sum(
        sum(1 for check in category.values() if check["status"] == "OK")
        for category in results.values()
    )
    
    print(f"Tests réussis: {passed_checks}/{total_checks}")
    print(f"Taux de succès: {passed_checks/total_checks:.1%}")
    
    # Problèmes critiques
    critical_issues = []
    for category, checks in results.items():
        for check_name, check_result in checks.items():
            if check_result["status"] == "CRITICAL":
                critical_issues.append(f"{category}.{check_name}")
    
    if critical_issues:
        print(f"\n🚨 PROBLÈMES CRITIQUES ({len(critical_issues)}):")
        for issue in critical_issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Aucun problème critique détecté")
    
    # Auto-réparation
    if fix_common:
        print("\n🔧 TENTATIVE DE RÉPARATION AUTOMATIQUE...")
        fix_results = attempt_common_fixes(results)
        for fix, success in fix_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {fix}")
    
    return results

if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    fix_common = "--fix-common" in sys.argv
    
    try:
        results = run_diagnostic(verbose, fix_common)
        sys.exit(0 if all(
            check["status"] in ["OK", "WARNING"] 
            for category in results.values()
            for check in category.values()
        ) else 1)
    except Exception as e:
        print(f"❌ Erreur lors du diagnostic: {e}")
        if verbose:
            traceback.print_exc()
        sys.exit(2)
```

---

## 📞 Support et Aide

### Collecte d'Informations de Debug

Avant de demander de l'aide, collectez ces informations :

```bash
# Informations système
python --version
pip list | grep -E "(numpy|opencv|pillow|pyautogui)"

# Logs récents
tail -100 logs/tacticalbot.log > debug_logs.txt

# Configuration actuelle
python -c "
import json
from config.config_manager import ConfigManager
config = ConfigManager()
print(json.dumps(config.get_full_config(), indent=2))
" > debug_config.json

# État des modules
python -c "
from engine.core import BotEngine
engine = BotEngine()
stats = engine.get_statistics()
print(json.dumps(stats, indent=2))
" > debug_stats.json
```

### Rapports de Bug

Format recommandé pour rapporter un problème :

```markdown
## 🐛 Rapport de Bug

**Description**
Description claire du problème

**Étapes pour Reproduire**
1. Action 1
2. Action 2  
3. Résultat observé

**Comportement Attendu**
Ce qui devrait se passer

**Environnement**
- OS : Windows 10 / Ubuntu 20.04 / macOS 12
- Python : 3.9.7
- TacticalBot : 2.0.0
- Configuration : Profil utilisé

**Logs**
```
[Coller les logs pertinents]
```

**Fichiers Joints**
- debug_logs.txt
- debug_config.json
- screenshot.png (si problème visuel)
```

---

Cette documentation de dépannage couvre la majorité des problèmes rencontrés avec TacticalBot. En cas de problème persistant, utilisez le script de diagnostic automatique et collectez les informations de debug avant de demander de l'aide.