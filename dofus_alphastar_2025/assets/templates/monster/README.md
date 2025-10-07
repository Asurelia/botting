# Templates de Monstres DOFUS

Ce dossier contient les templates de détection de monstres pour le bot DOFUS AlphaStar 2025.

## Structure

- `monster_templates.json` - Configuration JSON avec tous les templates de monstres
- Images de templates (à venir) - Screenshots de monstres pour template matching

## Format des Templates

Chaque monstre a les propriétés suivantes :

```json
{
  "id": "monster_id",
  "name": "Nom du Monstre",
  "level_range": [min, max],
  "visual_features": {
    "primary_color": "couleur principale",
    "secondary_color": "couleur secondaire",
    "size": "small|medium|large",
    "shape": "forme générale",
    "distinctive_features": ["traits distinctifs"]
  },
  "detection_keywords": ["mots-clés pour OCR"],
  "hp_bar_color": "couleur de la barre HP",
  "aggro_range": 3,
  "difficulty": "very_easy|easy|medium|hard|very_hard",
  "drops": ["items droppés"],
  "xp_value": 15
}
```

## Ajout de Nouveaux Monstres

### 1. Capturer un Screenshot

Prenez un screenshot clair du monstre dans le jeu :
- Résolution native du jeu
- Monstre au centre de l'image
- Pas d'obstruction (UI, autres entités)
- Fond représentatif de l'environnement

### 2. Ajouter au JSON

Ajoutez une nouvelle entrée dans `monster_templates.json` :

```json
"nouveau_monstre": {
  "id": "nouveau_monstre",
  "name": "Nouveau Monstre",
  "level_range": [10, 20],
  "visual_features": {
    "primary_color": "red",
    "secondary_color": "black",
    "size": "medium",
    "shape": "humanoid",
    "distinctive_features": ["armor", "weapon"]
  },
  "detection_keywords": ["Nouveau Monstre", "Monstre Nouveau"],
  "hp_bar_color": "red",
  "aggro_range": 4,
  "difficulty": "medium",
  "drops": ["Drop 1", "Drop 2"],
  "xp_value": 30
}
```

### 3. Méthodes de Détection

Le système utilise 3 méthodes de détection :

1. **YOLO Detection** - Détection d'objets en temps réel
2. **Template Matching** - Correspondance de templates OpenCV
3. **OCR** - Reconnaissance de texte (TrOCR)

Les monstres seront détectés si **au moins 2 des 3 méthodes** les identifient.

## Monstres Actuellement Supportés

- ✅ Bouftou (niveau 1-15)
- ✅ Pissenlit Diabolique (niveau 1-12)
- ✅ Sanglier (niveau 10-25)
- ✅ Moskito (niveau 12-20)
- ✅ Arakne (niveau 15-30)
- ✅ Larve Bleue (niveau 5-12)
- ✅ Tofu (niveau 1-8)
- ✅ Abeille (niveau 8-18)
- ✅ Grenouille (niveau 10-20)
- ✅ Loup (niveau 18-35)

## Configuration de Détection

Paramètres globaux dans `detection_config` :

- `confidence_threshold`: 0.6 - Seuil de confiance minimum
- `iou_threshold`: 0.45 - Seuil IoU pour suppression non-maximale
- `max_detections_per_frame`: 10 - Maximum de détections par frame
- `template_matching_threshold`: 0.7 - Seuil pour template matching
- `color_tolerance`: 30 - Tolérance de couleur (0-255)
- `size_tolerance`: 0.2 - Tolérance de taille (20%)

## Priorités de Combat

Configuration dans `combat_priorities` :

- `level_difference_threshold`: 5 - Différence de niveau max acceptable
- `prefer_solo_targets`: true - Préférer les cibles isolées
- `avoid_groups_larger_than`: 3 - Éviter les groupes de 3+ monstres
- `max_aggro_range`: 5 - Portée d'aggro maximum
- `min_hp_percentage_to_engage`: 60 - HP minimum pour engager (60%)

## Utilisation dans le Code

```python
from core.vision_engine_v2 import create_vision_engine
import json

# Charger les templates
with open("assets/templates/monster/monster_templates.json") as f:
    templates = json.load(f)

# Créer le moteur de vision
vision_engine = create_vision_engine()

# Détecter les monstres
monsters_detected = vision_engine.detect_monsters(frame, templates)

for monster in monsters_detected:
    print(f"Monstre détecté: {monster['name']} (niveau {monster['level_range']})")
    print(f"Position: {monster['bbox']}")
    print(f"Confiance: {monster['confidence']}")
```

## Améliorations Futures

- [ ] Ajouter images de référence pour chaque monstre
- [ ] Support de variantes de monstres (saisonnier, événements)
- [ ] Détection de boss et archimonstres
- [ ] Classification par famille de monstres
- [ ] Prédiction des patterns de comportement
