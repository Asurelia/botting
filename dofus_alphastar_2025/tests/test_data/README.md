# 🧪 Test Data Directory

Ce répertoire contient les données de test pour le bot Dofus autonome.

## 📁 Structure

```
test_data/
├── screenshots/         # Screenshots Dofus pour tests
│   ├── combat_*.png    # Scènes de combat
│   ├── farming_*.png   # Farming de ressources
│   ├── ui_*.png        # Interface UI (HP/MP/PA)
│   └── map_*.png       # Navigation sur cartes
├── videos/             # Vidéos gameplay YouTube
│   └── dofus_gameplay.mp4
├── templates/          # Templates UI pour matching
│   ├── hp_bar.png
│   ├── mp_icon.png
│   ├── pa_icon.png
│   └── monster_template.png
├── annotations/        # Ground truth annotations
│   └── screenshots_annotations.json
└── README.md          # Ce fichier
```

## 🔧 Utilisation

### 1. Setup Initial

```bash
# Télécharger images de test depuis web
python tests/setup_test_data.py

# Ou manuellement placer des screenshots Dofus ici
```

### 2. Annotations Format

```json
{
  "combat_01.png": {
    "hp_bar": {
      "bbox": [10, 10, 200, 30],
      "value": 0.75
    },
    "monsters": [
      {"bbox": [500, 300, 100, 100], "class": "goblin"}
    ]
  }
}
```

### 3. Tests

```bash
# Lancer tests sur images
pytest tests/test_vision.py -v

# Lancer tests sur vidéos
pytest tests/test_video_processing.py -v
```

## 📥 Sources d'Images

### Screenshots Officiels
- Forum Dofus: https://www.dofus.com/fr/forum/
- JeuxOnline: https://dofus.jeuxonline.info/

### Vidéos YouTube
- Recherche: "dofus unity gameplay 2024"
- Chaînes: Ankama, Dofus Retro, GameplayFR

### Mock UI
- Généré automatiquement: `python tests/create_mock_ui.py`
- Fichier: `mock_dofus_ui.html`

## ⚠️ Notes

- **Ne pas commit** les images/vidéos (trop volumineuses)
- **.gitignore** configuré pour ignorer *.png, *.jpg, *.mp4
- Garder annotations JSON pour reproduire tests

---

**Généré par:** Claude Code
**Date:** 2025-10-06
