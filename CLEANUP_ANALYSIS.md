# Analyse de Nettoyage - Racine du Projet

**Date**: 2025-10-07
**Post-Migration**: Analyse de tous les fichiers/dossiers restants après consolidation

---

## Vue d'Ensemble

**Après migration**: Il reste 20 dossiers + 50+ fichiers à la racine
**Question**: Qu'est-ce qui est vraiment utile pour `dofus_alphastar_2025/` ?

---

## Analyse par Catégorie

### 🎯 ESSENTIEL - À GARDER (5 items)

| Item | Taille | Raison |
|------|--------|--------|
| `dofus_alphastar_2025/` | 13M | **Système principal - TOUT LE CODE ACTIF** |
| `venv_rocm/` | 13G | Environnement virtuel ROCm pour AMD GPU |
| `archive/` | 14M | Archive des modules legacy (créée aujourd'hui) |
| `requirements/` | <1M | Fichiers de dépendances (base.txt, amd_gpu.txt, dev.txt) |
| `models/` | 63M | Modèles ML entraînés (YOLO, etc.) |

**Total essentiel**: ~26GB (principalement venv_rocm)

---

### ⚠️ LEGACY - À ARCHIVER (11 dossiers)

#### Anciens Modules (Redondants)
| Dossier | Taille | Statut | Équivalent dans alphastar |
|---------|--------|--------|---------------------------|
| `dofus_assistant/` | ? | LEGACY | `dofus_alphastar_2025/core/` |
| `dofus_knowledge/` | ? | LEGACY | `dofus_alphastar_2025/data/` |
| `human_simulation/` | ? | LEGACY | `dofus_alphastar_2025/core/safety/` |
| `learning_dofus/` | ? | LEGACY | `dofus_alphastar_2025/core/rl_training/` |
| `world_model_dofus/` | ? | LEGACY | `dofus_alphastar_2025/core/worldmodel/` |

#### Anciens Répertoires
| Dossier | Taille | Statut | Équivalent dans alphastar |
|---------|--------|--------|---------------------------|
| `docs/` | 176K | DUPLICATED | `dofus_alphastar_2025/docs/` |
| `scripts/` | 124K | DUPLICATED | `dofus_alphastar_2025/` (scripts intégrés) |
| `src/` | 128K | DUPLICATED | `dofus_alphastar_2025/core/hrm_reasoning/` |
| `data/` | 536K | LEGACY | Pas utilisé par alphastar |
| `backup_20250928_085637/` | ? | OLD BACKUP | Backup du 28 sept (obsolète) |

**Action**: Déplacer vers `archive/legacy_root_extras/`

---

### 🗑️ TEMPORAIRE - À SUPPRIMER (3 items)

| Item | Taille | Raison |
|------|--------|--------|
| `htmlcov/` | 7.6M | Rapports de couverture de tests (régénérables) |
| `__pycache__/` | 48K | Cache Python (régénérable) |
| `.coverage` | <1K | Fichier coverage (régénérable) |

**Action**: Supprimer directement

---

### ❓ À VÉRIFIER (4 items)

| Item | Taille | À Vérifier |
|------|--------|------------|
| `datasets/` | ? | Si vide → créer dans alphastar. Si plein → garder |
| `logs/` | ? | Si logs récents → garder. Si vieux → archiver |
| `venv_dofus_ai/` | 1.9G | Env virtuel ancien - utilisé ? Si non → supprimer |
| `backup_20250928_085637/` | ? | Contenu du backup - encore nécessaire ? |

---

### 📄 FICHIERS RACINE

#### Scripts Windows (INUTILES sur Linux)
```
activate_env.bat
activate_ia_dofus.bat
activate_ia_dofus.ps1
```
**Action**: Archiver dans `archive/legacy_windows_scripts/`

#### Documentation (90% REDONDANTE)

**À Garder** (5 fichiers):
- `README.md` - Readme racine principal
- `MIGRATION_REPORT.md` - Notre rapport de migration
- `QUICKSTART_LINUX.md` - Guide démarrage Linux
- `ROADMAP_AGA_VISION_2025.md` - Roadmap stratégique
- `.gitignore` - Configuration Git

**À Archiver** (40+ fichiers):
Tous les autres *.md sont redondants:
- `AMELIORATIONS_COMPLETEES.md`
- `ANALYSE_COMPLETE_EXISTANT.md`
- `ANALYSE_MODULES.json`
- `ARCHITECTURE_2025.md`
- `ARCHITECTURE_WORLD_MODEL_2025.md`
- `AUDIT_ARCHITECTURE_COMPLETE.md`
- `CHECKPOINT_*.md` (8 fichiers)
- `CLEAN_ARCHITECTURE_2025.md`
- `COMMENT_CA_MARCHE.md`
- `FICHIERS_CREES_20251006.md`
- `HRM_AMD_Migration_Analysis_Complete.md`
- `MIGRATION_*.md` (3 fichiers - sauf MIGRATION_REPORT.md)
- `PHASE*_SUMMARY.md` (4 fichiers)
- `PROJET_*.md` (2 fichiers)
- `RAPPORT_*.md` (1 fichier)
- `README_*.md` (5 fichiers - sauf README.md principal)
- `REINFORCEMENT_LEARNING_SUMMARY.md`
- `RESSOURCES_EXTERNES.md`
- `ROADMAP_DEVELOPPEMENT.md`
- `SECURITY_*.md` (2 fichiers)
- `STATUS_*.md` (2 fichiers)
- `SYSTEME_*.md` (3 fichiers)
- `sylvain.md` (60K - rapport pour utilisateur)
- `🎯 Système complet créé.txt`

**Action**: Déplacer vers `archive/legacy_docs/`

#### Scripts Python Racine (LEGACY)

**À Garder** (3 fichiers):
- `setup_linux_dependencies.sh` - Setup Linux
- `setup_pytorch_rocm.sh` - Setup PyTorch ROCm
- `run_all_tests.sh` - Lancer tests

**À Archiver** (20+ fichiers):
Tous les autres scripts Python sont obsolètes:
- `bot_launcher.py` (32K)
- `calibrate.py` (44K)
- `check_amd_compatibility.py`
- `demo_complete.py`
- `enhanced_ai_launcher.py` (32K)
- `example_usage.py`
- `launch_ai_dofus.py`
- `launch_linux.sh`
- `launch_linux_autonomous.py`
- `main.py`
- `monitor.py` (52K)
- `quick_extract_test.py`
- `setup.py` (32K)
- `setup_environment*.py` (3 fichiers)
- `setup_rocm_env.sh`
- `simple_bot_main.py`
- `test_*.py` (12 fichiers de tests obsolètes)

**Action**: Déplacer vers `archive/legacy_scripts/`

#### Fichiers Requirements

**À Garder**:
- `requirements/` (dossier complet)
- `requirements.txt` (lien principal)

**À Archiver**:
- `requirements_advanced.txt`
- `requirements_python313.txt`

**Action**: Déplacer vers `archive/legacy_requirements/`

#### Autres Fichiers

**À Garder**:
- `pytest.ini` (racine - pour tests globaux)
- `.env.example` (si présent)
- `.gitignore`

**À Archiver**:
- Tous les autres fichiers de config obsolètes

---

## Résumé des Actions

### ✅ À Garder (Total: ~26GB)

**Dossiers** (5):
1. `dofus_alphastar_2025/` (13M)
2. `venv_rocm/` (13G)
3. `archive/` (14M)
4. `requirements/` (<1M)
5. `models/` (63M)

**Fichiers** (8):
1. `README.md`
2. `MIGRATION_REPORT.md`
3. `QUICKSTART_LINUX.md`
4. `ROADMAP_AGA_VISION_2025.md`
5. `setup_linux_dependencies.sh`
6. `setup_pytorch_rocm.sh`
7. `run_all_tests.sh`
8. `pytest.ini`

---

### 🗃️ À Archiver (Total: ~800K + 11 dossiers)

**Créer**: `archive/legacy_root_extras/`

**Dossiers** (11):
1. `dofus_assistant/`
2. `dofus_knowledge/`
3. `human_simulation/`
4. `learning_dofus/`
5. `world_model_dofus/`
6. `docs/`
7. `scripts/`
8. `src/`
9. `data/`
10. `backup_20250928_085637/`
11. `venv_dofus_ai/` (1.9GB - si non utilisé)

**Fichiers** (~60 fichiers):
- 40+ fichiers *.md redondants
- 20+ scripts Python obsolètes
- 3 scripts Windows (*.bat, *.ps1)
- 2 fichiers requirements obsolètes

---

### 🗑️ À Supprimer (Total: 7.6M)

**Dossiers** (2):
1. `htmlcov/` (7.6M)
2. `__pycache__/` (48K)

**Fichiers** (1):
1. `.coverage`

**Commande**:
```bash
rm -rf htmlcov/ __pycache__/ .coverage
```

---

### ❓ À Vérifier Manuellement (4 items)

1. `datasets/` - Contenu ? Si vide → supprimer ou déplacer dans alphastar
2. `logs/` - Logs récents ? Si vieux → archiver
3. `venv_dofus_ai/` (1.9GB) - Encore utilisé ? Si non → supprimer
4. `backup_20250928_085637/` - Nécessaire ? Si non → supprimer

---

## Plan de Nettoyage Détaillé

### Phase 1: Supprimer temporaires (immédiat)
```bash
rm -rf htmlcov/ __pycache__/ .coverage
```
**Gain**: 7.6M libérés

### Phase 2: Archiver legacy modules
```bash
mkdir -p archive/legacy_root_extras/{docs,scripts,windows,requirements}
mv docs/ archive/legacy_root_extras/
mv scripts/ archive/legacy_root_extras/
mv src/ archive/legacy_root_extras/
mv data/ archive/legacy_root_extras/
mv dofus_assistant/ archive/legacy_root_extras/
mv dofus_knowledge/ archive/legacy_root_extras/
mv human_simulation/ archive/legacy_root_extras/
mv learning_dofus/ archive/legacy_root_extras/
mv world_model_dofus/ archive/legacy_root_extras/
mv backup_20250928_085637/ archive/legacy_root_extras/
```

### Phase 3: Archiver fichiers legacy
```bash
# Scripts Windows
mv *.bat *.ps1 archive/legacy_root_extras/windows/

# Documentation redondante
mv AMELIORATIONS_COMPLETEES.md archive/legacy_root_extras/docs/
mv ANALYSE_*.md archive/legacy_root_extras/docs/
mv ARCHITECTURE_*.md archive/legacy_root_extras/docs/
mv AUDIT_*.md archive/legacy_root_extras/docs/
mv CHECKPOINT_*.md archive/legacy_root_extras/docs/
mv CLEAN_*.md archive/legacy_root_extras/docs/
mv COMMENT_CA_MARCHE.md archive/legacy_root_extras/docs/
mv FICHIERS_*.md archive/legacy_root_extras/docs/
mv HRM_*.md archive/legacy_root_extras/docs/
mv MIGRATION_LINUX_COMPLETE.md archive/legacy_root_extras/docs/
mv MIGRATION_STATUS.md archive/legacy_root_extras/docs/
mv PHASE*_SUMMARY.md archive/legacy_root_extras/docs/
mv PROJET_*.md archive/legacy_root_extras/docs/
mv RAPPORT_*.md archive/legacy_root_extras/docs/
mv README_ENHANCED_AI.md archive/legacy_root_extras/docs/
mv README_ENTRY_POINTS.md archive/legacy_root_extras/docs/
mv README_LINUX.md archive/legacy_root_extras/docs/
mv README_SIMPLE.md archive/legacy_root_extras/docs/
mv REINFORCEMENT_*.md archive/legacy_root_extras/docs/
mv RESSOURCES_*.md archive/legacy_root_extras/docs/
mv ROADMAP_DEVELOPPEMENT.md archive/legacy_root_extras/docs/
mv SECURITY_*.md archive/legacy_root_extras/docs/
mv STATUS_*.md archive/legacy_root_extras/docs/
mv SYSTEME_*.md archive/legacy_root_extras/docs/
mv sylvain.md archive/legacy_root_extras/docs/
mv "🎯 Système complet créé.txt" archive/legacy_root_extras/docs/

# Scripts Python obsolètes
mv bot_launcher.py archive/legacy_root_extras/scripts/
mv calibrate.py archive/legacy_root_extras/scripts/
mv check_amd_compatibility.py archive/legacy_root_extras/scripts/
mv demo_complete.py archive/legacy_root_extras/scripts/
mv enhanced_ai_launcher.py archive/legacy_root_extras/scripts/
mv example_usage.py archive/legacy_root_extras/scripts/
mv launch_ai_dofus.py archive/legacy_root_extras/scripts/
mv launch_linux.sh archive/legacy_root_extras/scripts/
mv launch_linux_autonomous.py archive/legacy_root_extras/scripts/
mv main.py archive/legacy_root_extras/scripts/
mv monitor.py archive/legacy_root_extras/scripts/
mv quick_extract_test.py archive/legacy_root_extras/scripts/
mv setup.py archive/legacy_root_extras/scripts/
mv setup_environment*.py archive/legacy_root_extras/scripts/
mv setup_rocm_env.sh archive/legacy_root_extras/scripts/
mv simple_bot_main.py archive/legacy_root_extras/scripts/
mv test_dofus_detection.py archive/legacy_root_extras/scripts/
mv test_enhanced_ai.py archive/legacy_root_extras/scripts/
mv test_final.py archive/legacy_root_extras/scripts/
mv test_integrated_ai.py archive/legacy_root_extras/scripts/
mv test_phase*.py archive/legacy_root_extras/scripts/
mv test_simple.py archive/legacy_root_extras/scripts/
mv test_world_model.py archive/legacy_root_extras/scripts/

# Requirements obsolètes
mv requirements_advanced.txt archive/legacy_root_extras/requirements/
mv requirements_python313.txt archive/legacy_root_extras/requirements/
```

### Phase 4: Vérifications manuelles
```bash
# Vérifier contenu
ls -lah datasets/
ls -lah logs/
du -sh venv_dofus_ai/
du -sh backup_20250928_085637/

# Si inutiles, supprimer ou archiver
```

---

## Structure Finale Attendue

```
/home/spoukie/Documents/Botting/
├── dofus_alphastar_2025/          # 13M - SYSTÈME PRINCIPAL
├── venv_rocm/                      # 13G - Env virtuel ROCm
├── archive/                        # 15M
│   ├── legacy_root_modules/       # Modules archivés (migration précédente)
│   └── legacy_root_extras/        # Fichiers/dossiers archivés (ce cleanup)
├── requirements/                   # <1M - Dépendances
├── models/                         # 63M - Modèles ML
├── logs/                           # À vérifier
├── datasets/                       # À vérifier
├── README.md                       # Readme principal
├── MIGRATION_REPORT.md            # Rapport migration
├── QUICKSTART_LINUX.md            # Guide Linux
├── ROADMAP_AGA_VISION_2025.md     # Roadmap
├── setup_linux_dependencies.sh    # Setup Linux
├── setup_pytorch_rocm.sh          # Setup ROCm
├── run_all_tests.sh               # Tests
├── pytest.ini                     # Config pytest
└── requirements.txt               # Lien vers requirements/
```

**Total**: ~26GB (principalement venv_rocm)
**Dossiers racine**: 7-9 au lieu de 20
**Fichiers racine**: 8-10 au lieu de 50+

---

## Bénéfices du Nettoyage

1. **Clarté**: Racine propre, structure évidente
2. **Performance**: Moins de fichiers à scanner
3. **Maintenance**: Facile de trouver ce qui est actif vs legacy
4. **Espace**: ~10M libérés (temporaires) + clarification de 2-3GB (venv inutile?)

---

## Sécurité

**Tout est archivé, rien n'est perdu**:
- `archive/legacy_root_modules/` → Modules Phase 1-4
- `archive/legacy_root_extras/` → Fichiers/dossiers restants

**Rollback possible**:
```bash
# Restaurer tout
cp -r archive/legacy_root_extras/* .
```

---

## Recommandation

**Action immédiate**: ✅ OUI

Le nettoyage est **sûr et bénéfique**:
- Tout est archivé (récupérable)
- Gain de clarté énorme
- Pas de perte de fonctionnalités (tout est dans alphastar)

**Prochaine étape**: Exécuter Phase 1-4 du plan de nettoyage
