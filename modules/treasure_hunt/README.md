# 🏴‍☠️ DOFUS Treasure Hunt Automation

Système d'automatisation complet pour les chasses aux trésors de DOFUS, similaire à DofuBot, avec intelligence artificielle et interface graphique avancée.

## 🌟 Fonctionnalités Principales

### 🧠 Intelligence Artificielle
- **Solveur d'indices intelligent** avec reconnaissance de patterns
- **Base de données évolutive** d'indices avec apprentissage automatique
- **Reconnaissance visuelle** pour identifier les éléments sur la carte
- **Algorithmes de correspondance** textuelle et visuelle avancés

### 🗺️ Navigation Automatique
- **Calcul de chemins optimaux** avec algorithme de Dijkstra
- **Navigation intelligente** entre étapes avec gestion des obstacles
- **Support multi-transports** (marche, course, monture, zaaps)
- **Détection automatique** de la position et des blocages

### ⚡ Automatisation Complète
- **Gestion multi-chasses** en parallèle
- **Combat automatique** pendant les chasses
- **Collecte automatique** des récompenses
- **Gestion d'erreurs** et reprise automatique
- **Mode apprentissage** pour nouveaux indices

### 📊 Interface Graphique Avancée
- **Monitoring temps réel** de l'état des chasses
- **Statistiques détaillées** avec graphiques
- **Gestion de base de données** d'indices
- **Configuration avancée** et personnalisation
- **Historique complet** des sessions

## 🛠️ Installation

### Prérequis
```bash
pip install opencv-python numpy pillow matplotlib sqlite3
```

### Structure des Fichiers
```
treasure_hunt/
├── __init__.py              # Module principal
├── hint_database.py         # Base de données d'indices
├── treasure_solver.py       # Solveur intelligent
├── map_navigator.py         # Navigateur de carte
├── treasure_automation.py   # Automatisation complète
├── treasure_gui.py          # Interface graphique
├── example_usage.py         # Exemples d'utilisation
└── README.md               # Documentation
```

## 🚀 Utilisation Rapide

### Exemple Basique
```python
from treasure_hunt import create_treasure_hunt_system, TreasureHuntType

# Fonctions d'interface avec DOFUS (à adapter)
def click_handler(x, y):
    # Votre code pour cliquer à la position (x, y)
    pass

def screen_capture():
    # Votre code pour capturer l'écran
    # Retourner une image numpy array
    pass

# Créer le système
system = create_treasure_hunt_system(click_handler, screen_capture)

# Démarrer une chasse
system.start_treasure_hunt(TreasureHuntType.CLASSIC, "MonPersonnage")

# Surveiller l'état
status = system.get_current_status()
print(f"État: {status['state']}")

# Arrêter le système
system.stop_automation()
system.close()
```

### Interface Graphique
```python
from treasure_hunt import create_treasure_hunt_system, create_treasure_hunt_gui

# Créer le système d'automatisation
system = create_treasure_hunt_system(click_handler, screen_capture)

# Créer et lancer l'interface graphique
gui = create_treasure_hunt_gui(system)
gui.run()  # Ouvre la fenêtre GUI
```

## 🔧 Configuration

### Configuration par Défaut
```python
DEFAULT_CONFIG = {
    'max_attempts_per_step': 3,
    'step_timeout': 300,  # 5 minutes
    'combat_timeout': 180,  # 3 minutes
    'auto_fight': True,
    'auto_collect_rewards': True,
    'save_screenshots': True,
    'learning_mode': True
}
```

### Personnalisation
```python
# Modifier la configuration
system.config.update({
    'max_attempts_per_step': 5,
    'step_timeout': 600,
    'auto_fight': False  # Combat manuel
})
```

## 📊 Base de Données d'Indices

### Ajout d'Indices
```python
from treasure_hunt import HintDatabase, HintData, HintType, HintDifficulty
from datetime import datetime

db = HintDatabase()

# Créer un nouvel indice
hint = HintData(
    id="unique_id",
    text="Cherchez près de la taverne de Bonta",
    hint_type=HintType.BUILDING,
    difficulty=HintDifficulty.MEDIUM,
    area_name="Bonta",
    description="Taverne principale de la ville",
    keywords=["taverne", "bonta", "auberge"],
    created_at=datetime.now(),
    updated_at=datetime.now()
)

db.add_hint(hint)
```

### Import/Export
```python
# Exporter la base
db.export_hints("ma_base_indices.json")

# Importer une base communautaire
db.import_hints("base_communautaire.json")
```

## 🧪 Solveur d'Indices

### Utilisation du Solveur
```python
from treasure_hunt import TreasureSolver, HintDatabase

db = HintDatabase()
solver = TreasureSolver(db)

# Résoudre un indice
solutions = solver.solve_hint("Allez vers le nord de la taverne")

for solution in solutions:
    print(f"Solution: {solution.reasoning}")
    print(f"Confiance: {solution.confidence:.2f}")
    print(f"Coordonnées estimées: {solution.estimated_coordinates}")
```

### Types de Solutions
- **EXACT_MATCH**: Correspondance exacte dans la base
- **FUZZY_MATCH**: Correspondance approximative textuelle
- **VISUAL_MATCH**: Reconnaissance visuelle d'éléments
- **PATTERN_MATCH**: Analyse de patterns linguistiques
- **AI_INFERENCE**: Inférence par règles heuristiques

## 🗺️ Navigation

### Navigation Manuelle
```python
from treasure_hunt import MapNavigator, MapPosition

navigator = MapNavigator(click_handler)

# Définir une destination
target = MapPosition(x=10, y=20, area_id=1, sub_area_id=1)

# Naviguer
success = navigator.navigate_to(target, screen_image)

# Suivre la progression
progress = navigator.get_navigation_progress()
print(f"Progression: {progress['progress']:.1%}")
```

### Types de Mouvements
- **WALK**: Marche normale
- **RUN**: Course rapide
- **MOUNT**: Déplacement avec monture
- **ZAAP**: Téléportation par zaap
- **SUBWAY**: Transport souterrain

## 📈 Statistiques et Monitoring

### Récupération des Statistiques
```python
# Statistiques globales
stats = system.get_current_status()['global_statistics']

print(f"Chasses complétées: {stats['total_hunts_completed']}")
print(f"Taux de réussite: {stats['success_rate']:.1%}")
print(f"Temps moyen: {stats['average_completion_time']:.1f}s")

# Historique des sessions
history = system.get_session_history(limit=10)
for session in history:
    print(f"Session: {session['session_id']} - Succès: {session['success']}")
```

## 🎮 Interface Graphique

### Onglets Disponibles
1. **🎮 Contrôle**: Démarrage/arrêt des chasses, état temps réel
2. **📈 Monitoring**: Aperçu écran, progression, métriques
3. **📊 Statistiques**: Graphiques, historique des sessions
4. **🗄️ Base de Données**: Gestion des indices, import/export
5. **⚙️ Paramètres**: Configuration, chemins, informations système

### Fonctionnalités GUI
- ✅ **Contrôle temps réel** des chasses en cours
- 📊 **Graphiques interactifs** des performances
- 🔍 **Recherche et édition** d'indices
- 📤 **Import/Export** de bases communautaires
- ⚙️ **Configuration avancée** avec sauvegarde
- 📋 **Logs détaillés** avec niveaux de gravité

## 🔄 Callbacks et Événements

### Enregistrement de Callbacks
```python
def on_hunt_started(session):
    print(f"Chasse démarrée: {session.session_id}")

def on_step_completed(step):
    print(f"Étape {step.step_number} terminée")

def on_error(error_msg):
    print(f"Erreur: {error_msg}")

# Enregistrer les callbacks
system.register_callback('on_hunt_started', on_hunt_started)
system.register_callback('on_step_completed', on_step_completed) 
system.register_callback('on_error', on_error)
```

### Événements Disponibles
- `on_hunt_started`: Démarrage d'une chasse
- `on_step_completed`: Fin d'une étape
- `on_hint_solved`: Résolution d'un indice
- `on_hunt_completed`: Fin de chasse réussie
- `on_error`: Erreur détectée
- `on_state_changed`: Changement d'état

## 🛡️ Gestion d'Erreurs

### Récupération Automatique
- **Détection de blocages** avec retry automatique
- **Gestion des timeouts** par étape
- **Reprise après combat** automatique
- **Sauvegarde d'état** pour reprendre après interruption

### Mode Debug
```python
system.config['debug_mode'] = True  # Active les logs détaillés
system.config['save_screenshots'] = True  # Capture d'écran des erreurs
```

## 📝 Types de Chasses Supportés

- **CLASSIC**: Chasses aux trésors classiques
- **LEGENDARY**: Chasses légendaires haute difficulté  
- **WEEKLY**: Chasses hebdomadaires
- **DAILY**: Chasses quotidiennes
- **EVENT**: Chasses d'événements spéciaux

## 🔬 Mode Apprentissage

Le système peut apprendre de nouveaux indices automatiquement:

```python
# Activer l'apprentissage
solver.learning_mode = True

# Le système apprend automatiquement des succès/échecs
solver.learn_from_solution(hint_text, solution, success=True, time_taken=30.0)
```

## ⚠️ Limitations et Avertissements

1. **Respect des CGU**: Ce système est conçu à des fins éducatives
2. **Performance**: Les temps de résolution dépendent de la base d'indices
3. **Reconnaissance visuelle**: Nécessite une résolution d'écran stable
4. **Adaptation**: Peut nécessiter des ajustements selon les mises à jour DOFUS

## 🤝 Contribution

### Structure du Code
- **Code modulaire** avec séparation claire des responsabilités
- **Documentation complète** en français dans le code
- **Tests unitaires** pour chaque composant
- **Gestion d'erreurs** robuste partout

### Ajout de Nouvelles Fonctionnalités
1. Étendre la classe appropriée (Solver, Navigator, etc.)
2. Ajouter les tests correspondants
3. Mettre à jour la documentation
4. Intégrer dans l'interface GUI si nécessaire

## 📄 Licence

Ce projet est fourni tel quel à des fins éducatives et de recherche. 
L'utilisation doit respecter les conditions générales d'utilisation de DOFUS.

## 🆘 Support

Pour le support et les questions:
1. Consulter cette documentation
2. Examiner les exemples dans `example_usage.py`
3. Activer le mode debug pour plus d'informations
4. Consulter les logs détaillés

---

**Version**: 1.0.0  
**Auteur**: DofuBot System  
**Date**: 2025  

*Bon jeu et bonnes chasses aux trésors ! 🏴‍☠️*