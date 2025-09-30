# 🔍 Guide d'Extraction des Données Dofus

## 📖 Vue d'Ensemble

L'outil `dofus_data_extractor.py` permet d'extraire automatiquement les données du jeu Dofus Unity **SANS MODIFIER les fichiers originaux**.

---

## 🎯 Objectif

Récupérer toutes les données nécessaires au bot :
- 🐉 **Monstres** : Stats, résistances, sorts, drops, locations
- ⚔️ **Sorts** : Dégâts, coût PA, portée, effets
- 🎒 **Items** : Stats, niveau requis, valeur
- 🗺️ **Maps** : Coordonnées, zones, sous-zones
- 💬 **NPCs** : Dialogues, quêtes, boutiques
- 📜 **Quêtes** : Objectifs, récompenses, étapes
- 🌾 **Ressources** : Niveau récolte, emplacements

---

## 🚀 Utilisation

### **1. Installation des dépendances**

```bash
pip install beautifulsoup4 requests
```

### **2. Lancement de l'extraction**

```bash
python tools/dofus_data_extractor.py
```

### **3. Résultat**

```
============================================================
🎮 DOFUS UNITY DATA EXTRACTOR
============================================================

🚀 Configuration Dofus Data Manager...
✅ Installation trouvée: C:\Program Files (x86)\Steam\steamapps\common\Dofus Unity
✅ 1 installation(s) trouvée(s)

📊 Extraction des données locales...
🔍 Début extraction données Dofus Unity...
📄 Extraction fichiers JSON...
📄 Extraction fichiers XML...
🎮 Analyse Unity Assets...
💾 Extraction bases de données...
💾 Sauvegarde données extraites...
  ✅ monsters: 156 entrées → monsters_20250929_232500.json
  ✅ spells: 423 entrées → spells_20250929_232500.json
  ✅ items: 2847 entrées → items_20250929_232500.json
  ✅ maps: 1024 entrées → maps_20250929_232500.json
✅ Extraction terminée !

✅ RÉSULTATS:
  • Monsters: 156 entrées
  • Spells: 423 entrées
  • Items: 2847 entrées
  • Maps: 1024 entrées

============================================================
✅ Extraction terminée !
============================================================
```

---

## 📁 Structure des Données Extraites

### **Fichiers générés** :

```
data/
├── extracted/
│   ├── monsters_20250929_232500.json
│   ├── spells_20250929_232500.json
│   ├── items_20250929_232500.json
│   ├── maps_20250929_232500.json
│   ├── npcs_20250929_232500.json
│   ├── quests_20250929_232500.json
│   └── resources_20250929_232500.json
└── fansite/
    └── (données récupérées en ligne)
```

### **Format des données** :

#### **Exemple : Monstre**
```json
{
  "bouftou": {
    "id": "bouftou",
    "name": "Bouftou",
    "level": 10,
    "health": 85,
    "ap": 6,
    "mp": 3,
    "resistances": {
      "neutral": 5,
      "earth": 15,
      "fire": 0,
      "water": 0,
      "air": -10
    },
    "damages": {
      "min": 12,
      "max": 22,
      "element": "earth"
    },
    "locations": [
      {
        "map": "Champs d'Astrub",
        "coordinates": [5, -20],
        "spawn_rate": 0.5
      }
    ],
    "drops": [
      {
        "item": "Laine de Bouftou",
        "rate": 0.8,
        "value": 15
      }
    ],
    "xp_reward": 35,
    "kamas_reward": 18,
    "difficulty": "easy",
    "ai_behavior": "aggressive",
    "spell_list": ["Coup de Tête", "Charge", "Laine Magique"],
    "special_abilities": ["heal_allies"]
  }
}
```

---

## 🔍 Comment ça Fonctionne

### **1. Recherche de l'Installation**

L'outil cherche Dofus Unity dans :
- ✅ Steam : `C:\Program Files (x86)\Steam\steamapps\common\Dofus Unity`
- ✅ Ankama Launcher : `C:\Users\{username}\AppData\Local\Ankama\Dofus`
- ✅ Installation standalone : `C:\Dofus`, `D:\Dofus`, etc.
- ✅ Tous les disques (C:, D:, E:, F:, G:, H:)

### **2. Extraction des Données**

#### **A. Fichiers JSON**
```python
# Recherche récursive de tous les .json
for json_file in data_folder.rglob("*.json"):
    data = json.load(json_file)
    # Classification automatique par type
    classify_data(data)
```

#### **B. Fichiers XML**
```python
# Parsing XML → Dict
tree = ET.parse(xml_file)
data = xml_to_dict(tree.getroot())
```

#### **C. Unity Assets**
```python
# Détection fichiers .assets
# (Nécessite UnityPy pour extraction complète)
asset_files = data_folder.rglob("*.assets")
```

#### **D. Bases de Données**
```python
# Recherche .db, .sqlite, .dat
db_files = data_folder.rglob("*.db")
# (Nécessite sqlite3 pour lecture)
```

### **3. Classification Intelligente**

L'outil détecte automatiquement le type de données :

```python
if "monster" in filename or "level" in data:
    → Type: Monster
    
if "spell" in filename or "ap_cost" in data:
    → Type: Spell
    
if "item" in filename or "stats" in data:
    → Type: Item
```

---

## 🌐 Mode Fansite (Fallback)

Si **aucune installation locale** n'est trouvée, l'outil utilise les fansites :

### **Sources disponibles** :
- 🔹 **DofusDB** : https://dofusdb.fr
- 🔹 **Dofus Pour Les Noobs** : https://www.dofuspourlesnoobs.com
- 🔹 **DofusBook** : https://dofusbook.net
- 🔹 **Krosmoz** : https://www.krosmoz.com/fr/dofus

### **Utilisation** :
```python
from tools.dofus_data_extractor import DofusDataManager

manager = DofusDataManager()
manager.setup()

# Récupération d'un monstre
monster_data = manager.get_monster_info("Bouftou")
```

---

## 🔧 Intégration avec le Bot

### **1. Chargement des Données**

```python
import json
from pathlib import Path

def load_monster_database():
    """Charge la base de données des monstres"""
    data_file = Path("data/extracted/monsters_latest.json")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

monsters = load_monster_database()
```

### **2. Utilisation dans le Bot**

```python
# Exemple : Stratégie de combat
def get_combat_strategy(enemy_name):
    """Récupère la meilleure stratégie contre un ennemi"""
    enemy_data = monsters.get(enemy_name.lower())
    
    if not enemy_data:
        return None
    
    # Analyse résistances
    resistances = enemy_data["resistances"]
    weak_element = min(resistances, key=resistances.get)
    
    # Recommandation
    return {
        "weak_to": weak_element,
        "recommended_distance": 4,  # Distance optimale
        "difficulty": enemy_data["difficulty"],
        "expected_xp": enemy_data["xp_reward"]
    }

# Utilisation
strategy = get_combat_strategy("Bouftou")
# → {"weak_to": "air", "recommended_distance": 4, ...}
```

### **3. Mise à Jour Automatique**

```python
# Exécution périodique (1x par semaine)
import schedule

def update_game_data():
    """Met à jour les données du jeu"""
    manager = DofusDataManager()
    manager.setup()
    manager.extract_all_data()

# Planification
schedule.every().week.do(update_game_data)
```

---

## ⚠️ Important

### **Sécurité** :
- ✅ **Lecture seule** : Aucune modification des fichiers Dofus
- ✅ **Non-intrusif** : Pas d'injection mémoire
- ✅ **Légal** : Analyse de fichiers locaux uniquement

### **Performance** :
- ⏱️ Extraction complète : ~2-5 minutes
- 💾 Espace disque : ~50-100 MB de données
- 🔄 Mise à jour recommandée : 1x par semaine

### **Limitations** :
- 📦 Unity Assets nécessite `UnityPy` (optionnel)
- 💾 Bases de données nécessitent `sqlite3` (inclus Python)
- 🌐 Fansites peuvent avoir rate-limiting

---

## 🚀 Prochaines Étapes

1. **Exécuter l'extraction** :
   ```bash
   python tools/dofus_data_extractor.py
   ```

2. **Vérifier les données** :
   ```bash
   ls data/extracted/
   ```

3. **Intégrer au bot** :
   - Charger les données dans `knowledge_graph.py`
   - Utiliser dans `decision_engine.py`
   - Référencer dans `combat_advisor.py`

4. **Automatiser** :
   - Ajouter au script de démarrage
   - Planifier mises à jour hebdomadaires

---

## 📊 Statistiques Attendues

Avec une installation Dofus Unity complète :

| Catégorie | Entrées Attendues |
|-----------|-------------------|
| Monstres | 150-300 |
| Sorts | 400-800 |
| Items | 2000-5000 |
| Maps | 1000-2000 |
| NPCs | 200-500 |
| Quêtes | 500-1000 |
| Ressources | 100-300 |

**Total : ~4,000-10,000 entrées** 📈

---

## 🎯 Conclusion

L'outil d'extraction vous permet d'avoir une **base de données complète et à jour** du jeu Dofus Unity, essentielle pour que votre bot prenne des décisions intelligentes basées sur des données réelles ! 🎮🧠
