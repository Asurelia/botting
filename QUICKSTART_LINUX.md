# ⚡ QUICK START - Bot Dofus Linux Mint

**Version:** 1.0.0 | **Date:** 2025-10-06 | **Agent:** Claude Code

---

## 🚀 DÉMARRAGE RAPIDE (10 minutes)

### Étape 1: Installer Dépendances Système (2 min)

```bash
cd /home/spoukie/Documents/Botting
bash setup_linux_dependencies.sh
```

### Étape 2: Setup PyTorch + ROCm (5 min)

```bash
source venv_rocm/bin/activate
bash setup_pytorch_rocm.sh
```

### Étape 3: Générer Données de Test (1 min)

```bash
python tests/setup_test_data.py
firefox tests/test_data/mock_dofus_ui.html
```

### Étape 4: Lancer Tests (2 min)

```bash
pytest tests/test_gpu.py -v -s
# Attendu: 15/15 tests passing
```

---

## 📚 DOCUMENTATION COMPLÈTE

Voir: **CHECKPOINT_LINUX_MIGRATION_20251006.md** (50+ pages)

---

**🤖 Généré par Claude Code | ✅ Ready!**
