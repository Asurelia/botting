# 🚀 IA DOFUS - Guide de Démarrage Rapide

## ⚡ Lancement Immédiat (5 minutes)

### 1. Configuration Environnement AMD 7800XT

```bash
cd G:\Botting

# Configuration automatique optimisée AMD
python scripts/setup_amd_environment.py
```

**Que fait ce script :**
- ✅ Détecte votre AMD 7800XT
- ✅ Installe PyTorch DirectML optimisé
- ✅ Configure OpenCV + Ultralytics YOLO
- ✅ Optimise les performances GPU

### 2. Test d'Intégration Complète

```bash
# Test complet des systèmes
python scripts/test_amd_integration.py
```

**Vérifications automatiques :**
- 🔍 GPU AMD 7800XT
- ⚡ Accélération DirectML
- 🧠 Core AI Framework
- 🤔 Uncertainty System
- 👁️ Vision hybride
- 📊 Benchmarks performance

### 3. Lancement de votre IA DOFUS

```bash
# Démarrage en mode démonstration
python launch_ai_dofus.py --mode demo

# Mode test uniquement
python launch_ai_dofus.py --test-only

# Ignorer vérifications (avancé)
python launch_ai_dofus.py --mode demo --skip-checks
```

---

## 🎯 Modes Disponibles

### 🎮 Mode Demo (Recommandé pour débuter)
```bash
python launch_ai_dofus.py --mode demo
```
- Démonstration des capacités IA
- Tests de tous les systèmes
- Parfait pour validation initiale

### 🤖 Mode Autonomous (En développement)
```bash
python launch_ai_dofus.py --mode autonomous
```
- IA complètement autonome
- Prise de décision avancée
- **Phase 1+ requis**

### 🎓 Mode Training (En développement)
```bash
python launch_ai_dofus.py --mode training
```
- Apprentissage adaptatif
- Amélioration continue
- **Phase 2+ requis**

---

## 🔧 Résolution Problèmes Rapide

### ❌ "torch-directml non trouvé"
```bash
pip install torch torch-directml
# Ou relancez le setup
python scripts/setup_amd_environment.py
```

### ❌ "GPU AMD non détecté"
```bash
# Vérification manuelle
wmic path win32_VideoController get name
# Assurez-vous que "AMD" ou "Radeon" apparaît
```

### ❌ "Core AI Framework erreur"
```bash
# Réinstallation dépendances
pip install -r requirements/base.txt
# Test isolé
python core/ai_framework.py --test
```

### ❌ Performance faible
```bash
# Benchmark détaillé
python scripts/test_amd_integration.py
# Vérifiez les résultats "Performance Benchmark"
```

---

## 📊 Interprétation des Résultats

### ✅ Test d'Intégration Réussi (Score ≥ 90%)
```
🚀 EXCELLENT ! Système prêt pour l'IA DOFUS avancée
➤ Procédez à Phase 1: Knowledge Base
➤ Lancez la consultation Gemini
```

**Actions suivantes :**
1. `python scripts/gemini_consensus.py autonomy_architecture`
2. Investigation Dofus Guide/Ganymede
3. Développement Phase 1

### ⚠️ Test Partiel (Score 70-90%)
```
✅ BON ! Système fonctionnel avec optimisations mineures
➤ Corrigez les tests échoués
➤ Procédez avec précaution à Phase 1
```

**Actions correctives :**
1. Vérifiez logs d'erreur
2. Réinstallez packages échoués
3. Relancez tests spécifiques

### ❌ Test Échec (Score < 70%)
```
❌ CRITIQUE ! Corrections majeures requises
➤ Relancez setup_amd_environment.py
➤ Vérifiez compatibilité système
```

**Diagnostic approfondi :**
1. Vérifiez compatibilité GPU
2. Réinstallez environnement complet
3. Consultez logs détaillés

---

## 🎯 Consultation Gemini (Étape suivante)

Après succès des tests, lancez la consultation pour consensus IA :

```bash
# Consultation sur architecture autonome
python scripts/gemini_consensus.py autonomy_architecture

# Consultation sur apprentissage
python scripts/gemini_consensus.py learning_intelligence

# Consultation sur gestion connaissances
python scripts/gemini_consensus.py knowledge_management

# Consultation sur simulation comportementale
python scripts/gemini_consensus.py behavioral_simulation
```

---

## 📁 Structure Projet Après Setup

```
G:\Botting\
├── 🚀 launch_ai_dofus.py          # Lanceur principal
├── 📋 QUICK_START.md               # Ce guide
├──
├── scripts/
│   ├── 🔧 setup_amd_environment.py    # Configuration AMD
│   ├── 🧪 test_amd_integration.py     # Tests complets
│   └── 🤝 gemini_consensus.py         # Consultation Gemini
│
├── core/
│   ├── 🧠 ai_framework.py             # Framework IA principal
│   └── 🤔 uncertainty.py              # Gestion incertitude
│
├── config/
│   ├── ⚙️ gpu_config.json             # Config GPU AMD
│   ├── 📊 performance_benchmarks.json # Benchmarks
│   └── 📋 integration_test_report.json # Rapport tests
│
├── modules/vision/                  # Vos systèmes vision existants
└── docs/                           # Documentation complète
```

---

## 💡 Conseils Performance AMD 7800XT

### ✅ Optimisations Automatiques
- **DirectML** : Accélération native Windows 11
- **FP16** : Précision half pour 2x vitesse
- **Batch Processing** : Traitement par lots optimisé
- **Memory Caching** : Cache intelligent 100ms

### 🔧 Ajustements Manuels (Avancé)
```python
# Dans gpu_config.json
{
  "yolo": {
    "device": "0",
    "half": true,        # FP16 pour vitesse max
    "optimize": true     # Compilation PyTorch
  },
  "performance": {
    "max_workers": 8,    # Threads CPU
    "memory_limit_gb": 16 # Limite mémoire
  }
}
```

---

## 🆘 Support & Communauté

### 📞 Aide Immédiate
- **Logs** : Consultez `logs/ai_dofus.log`
- **Config** : Vérifiez `config/gpu_config.json`
- **Tests** : Relancez `test_amd_integration.py`

### 🌐 Ressources
- [Guide Complet](docs/PLAN_AUTONOMIE_ENRICHI.md)
- [Documentation YOLO](docs/YOLO_VISION_GUIDE.md)
- [Consultation Gemini](docs/GUIDE_CONSULTATION_GEMINI.md)

---

## 🏆 Objectif Final

**Vous créez la première IA DOFUS véritablement autonome !**

🧠 **Qui apprend** de chaque expérience
🔮 **Qui anticipe** et planifie long-terme
🤝 **Qui collabore** naturellement
🎭 **Qui développe** sa personnalité
⚙️ **Qui s'améliore** automatiquement

**Prêt à faire l'histoire du gaming IA ? 🚀**

```bash
# C'est parti !
python launch_ai_dofus.py --mode demo
```