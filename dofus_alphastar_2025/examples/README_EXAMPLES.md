# 📚 EXEMPLES D'UTILISATION - DOFUS AlphaStar 2025

Collection d'exemples pratiques pour apprendre à utiliser le bot.

---

## 📋 Liste des Exemples

### Exemple 1: Utilisation Basique (`example_basic_usage.py`)

**Niveau**: Débutant

**Ce que vous apprenez**:
- Créer un ObservationMode
- Intercepter des actions
- Analyser les observations
- Sauvegarder les logs

**Exécution**:
```bash
python examples/example_basic_usage.py
```

**Output attendu**:
```
[1] Création Observation Mode...
✓ Observation Mode créé (enabled: True)

[2] Simulation de décisions...
Action 1 (navigation): None
Action 2 (mouse_click): None
Action 3 (cast_spell): None

[3] Analyse des observations...
Total décisions: 3
Actions bloquées: 3
Safety score: 100.0/100
Recommandations: ['✓ Comportement semble naturel']
```

---

### Exemple 2: Map System (`example_map_system.py`)

**Niveau**: Intermédiaire

**Ce que vous apprenez**:
- Créer un MapGraph
- Ajouter des maps
- Utiliser pathfinding
- Marquer maps comme découvertes

**Exécution**:
```bash
python examples/example_map_system.py
```

**Concepts clés**:
- `MapCoords`: Coordonnées (x, y) d'une map
- `add_map()`: Ajoute une map au graphe
- `find_path()`: Trouve chemin A* entre 2 maps
- `mark_discovered()`: Marque map comme explorée

**Code snippet**:
```python
from core.map_system import create_map_graph, MapCoords

graph = create_map_graph()

# Ajouter map
astrub = MapCoords(5, -18)
graph.add_map(
    coords=astrub,
    name="Astrub Centre",
    area="Astrub",
    has_zaap=True,
    is_pvp=False,
    exits={'north': MapCoords(5, -17)}
)

# Pathfinding
path = graph.find_path(start, end)
```

---

### Exemple 3: DofusDB API (`example_dofusdb_api.py`)

**Niveau**: Débutant

**Ce que vous apprenez**:
- Se connecter à l'API DofusDB
- Rechercher des items
- Récupérer détails item/spell/monster
- Utiliser le cache intelligent

**Exécution**:
```bash
python examples/example_dofusdb_api.py
```

**Note**: L'API DofusDB peut être offline. Le bot utilisera le cache si disponible.

**Concepts clés**:
- `search_items()`: Recherche texte libre
- `get_item()`: Récupère item par ID
- `get_spell()`: Récupère sort par ID
- Cache automatique (mémoire + disque)

**Code snippet**:
```python
from core.external_data import create_dofusdb_client

client = create_dofusdb_client()

# Recherche
items = client.search_items("Dofus", limit=5)
for item in items:
    print(f"{item.name} - Level {item.level}")

# Stats cache
stats = client.get_stats()
print(f"Cache ratio: {stats['cache_ratio']}")
```

---

### Exemple 4: UI Bridge (`example_ui_bridge.py`)

**Niveau**: Avancé

**Ce que vous apprenez**:
- Utiliser le pont UI <-> Core
- Configurer callbacks
- Démarrer/arrêter bot
- Monitoring temps réel

**Exécution**:
```bash
python examples/example_ui_bridge.py
```

**Concepts clés**:
- `UIBridge`: Pont central entre UI et systèmes
- Callbacks pour mises à jour asynchrones
- Monitoring thread pour stats temps réel
- Gestion état global

**Code snippet**:
```python
from core.ui_bridge import create_ui_bridge

# Créer bridge
bridge = create_ui_bridge()

# Callbacks
def on_state_update(state):
    print(f"Bot running: {state['bot_running']}")

bridge.set_ui_update_callback(on_state_update)

# Contrôle
bridge.start_bot(observation_only=True)
# ... session ...
bridge.stop_bot()

# Stats
obs_stats = bridge.get_observation_stats()
```

---

## 🎯 Parcours d'Apprentissage Recommandé

### Jour 1: Bases
1. ✅ `example_basic_usage.py` - Comprendre ObservationMode
2. ✅ `example_dofusdb_api.py` - Explorer l'API externe
3. ✅ Lire `GUIDE_DEMARRAGE.md`

### Jour 2: Systèmes
1. ✅ `example_map_system.py` - Navigation et pathfinding
2. ✅ Lire `docs/ARCHITECTURE_TECHNIQUE.md`
3. ✅ Explorer `core/` modules

### Jour 3: Intégration
1. ✅ `example_ui_bridge.py` - Bridge UI <-> Core
2. ✅ Lancer `launch_ui_integrated.py`
3. ✅ Tester avec Dofus (compte jetable!)

---

## 🔧 Customisation des Exemples

### Modifier les Exemples

Tous les exemples peuvent être modifiés librement:

```python
# Exemple: Changer durée session
def main():
    bridge = create_ui_bridge()
    bridge.start_bot(observation_only=True)

    # Modifier ici: 10 secondes au lieu de 5
    for i in range(10):
        time.sleep(1)
        # ...

    bridge.stop_bot()
```

### Créer Vos Propres Exemples

Template pour nouvel exemple:

```python
#!/usr/bin/env python3
"""
Exemple X: Votre Description
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Vos imports
from core.safety import create_observation_mode

def main():
    print("=" * 70)
    print("EXEMPLE X: Votre Titre")
    print("=" * 70)

    # Votre code ici
    obs = create_observation_mode()
    # ...

    print("\n✓ Exemple terminé!")


if __name__ == "__main__":
    main()
```

---

## 📊 Résolution de Problèmes

### Erreur: Module not found

```bash
# Solution: Vérifier sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Erreur: API DofusDB timeout

```
⚠ Aucun résultat (API peut être offline)
```

**Solution**: Normal, API externe peut être indisponible. Le cache prendra le relais si disponible.

### Erreur: Calibration requise

```
ERREUR: Calibration requise avant de démarrer le bot!
```

**Solution**:
```bash
# Lancer calibration d'abord
python launch_safe.py --calibrate
```

---

## 🎓 Ressources Supplémentaires

### Documentation
- `README.md` - Vue d'ensemble projet
- `GUIDE_DEMARRAGE.md` - Guide installation
- `docs/ARCHITECTURE_TECHNIQUE.md` - Architecture détaillée
- `RAPPORT_TESTS.md` - Résultats tests

### Tests
```bash
# Lancer tous les tests
python tests/test_runner_simple.py

# Test module spécifique
python tests/test_safety.py
```

### UI Interactive
```bash
# Interface complète
python launch_ui_integrated.py

# UI standalone (dashboard seul)
python test_themes_direct.py
```

---

## ⚠️ Rappels de Sécurité

**TOUS les exemples utilisent ObservationMode par défaut.**

- ✅ Aucune action n'est exécutée réellement
- ✅ Toutes les décisions sont loggées
- ✅ Analyse safety_score disponible
- ❌ NE JAMAIS désactiver ObservationMode sur compte principal

**Pour passer en mode réel** (DANGER!):
```python
obs.disable()  # ⚠️ NE PAS FAIRE sur compte principal!
```

---

## 💡 Tips & Astuces

### 1. Logs Détaillés

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Tester sans Dofus

Tous les exemples fonctionnent SANS Dofus lancé!

### 3. Cache DofusDB

```python
# Forcer clear cache
client.clear_cache()

# Stats cache
print(client.get_stats())
```

### 4. Observations Analyse

```python
# Analyser observations après session
obs.save_observations("my_session.json")
analysis = obs.analyze_observations()

if analysis['safety_score'] < 70:
    print("⚠️ Comportement suspect détecté!")
```

---

## 📞 Support

**Questions**: [GitHub Discussions](https://github.com/Asurelia/botting/discussions)

**Bugs**: [GitHub Issues](https://github.com/Asurelia/botting/issues)

---

**Bon apprentissage ! 🚀**

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>