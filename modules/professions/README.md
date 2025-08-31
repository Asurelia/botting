# 🎮 Système de Métiers Complet pour Botting

## 📋 Vue d'ensemble

Ce module fournit un système complet de gestion des métiers pour un bot de jeu, incluant:

- **4 métiers spécialisés** : Fermier, Bûcheron, Mineur, Alchimiste  
- **Optimisation automatique** de routes et stratégies de farming
- **Calculs de rentabilité** avancés (XP/h, Kamas/h, ROI)
- **Synergies inter-métiers** pour maximiser l'efficacité
- **Patterns de reconnaissance** et farming intelligent
- **Gestion multi-métiers** avec allocation optimale du temps

## 🏗️ Architecture

```
modules/professions/
├── base.py              # Classe abstraite commune
├── farmer.py            # Métier Fermier (29 ressources agricoles)
├── lumberjack.py        # Métier Bûcheron (25 types d'arbres)
├── miner.py            # Métier Mineur (30 minerais + gemmes)
├── alchemist.py        # Métier Alchimiste (16 recettes de potions)
├── profession_manager.py # Gestionnaire global et optimisation
├── example_usage.py    # Exemples d'utilisation complète
└── __init__.py         # Exports du module
```

## 🚀 Utilisation rapide

### Métier individuel
```python
from modules.professions import Farmer

# Créer un fermier
farmer = Farmer()

# Obtenir la route optimale pour les niveaux 1-30
route = farmer.get_optimal_route((1, 30))

# Calculer la rentabilité du blé
profit = farmer.calculate_profitability('ble')
print(f"Blé: {profit['kamas_per_hour']:.0f} kamas/h, {profit['xp_per_hour']:.0f} XP/h")

# Pattern de farming optimisé
pattern = farmer.get_farming_pattern(route[:5], 'cluster')
```

### Gestionnaire global
```python
from modules.professions import ProfessionManager, OptimizationStrategy

# Créer le gestionnaire
manager = ProfessionManager()

# Optimiser une session de 4h pour les profits
session = manager.optimize_global_session(4.0, OptimizationStrategy.PROFIT_FOCUSED)

# Exécuter la session
results = manager.execute_session(session)
print(f"Gains: {results['totals']['total_kamas_gained']:,} kamas")
```

## 🎯 Métiers disponibles

### 🌾 Fermier (29 ressources)
- **Céréales** : Blé, Orge, Avoine, Seigle, Riz
- **Légumes** : Navet, Carotte, Radis, Poireau, Chou, Artichaut
- **Fruits** : Pomme, Cerise, Fraise, Orange, Kiwi, Banane, Noix de Coco
- **Plantes médicinales** : Menthe, Orchidée, Ginseng, Belladone, Mandragore
- **Ressources rares** : Lin, Chanvre, Houblon, Bambou, Bambou Sacré

**Patterns de farming** : Linéaire, Spiral, Zigzag, Cluster

### 🪓 Bûcheron (25 types d'arbres)
- **Arbres communs** : Chêne, Frêne, Noyer, Châtaignier, Hêtre
- **Arbres nobles** : Bouleau, Merisier, Orme, Érable, Charme
- **Conifères** : Pin, Sapin, Épicéa, If
- **Bois exotiques** : Bambou Géant, Teck, Acajou, Palissandre
- **Arbres légendaires** : Olivier Ancien, Séquoia, Baobab, Arbre-Monde, Yggdrasil

**Outils supportés** : 6 niveaux de haches (débutant → diamant)
**Gestion du respawn** : Calculs avancés tenant compte des temps de réapparition

### ⛏️ Mineur (30 minerais + gemmes)
- **Surface (1-30)** : Fer, Cuivre, Bronze, Étain, Argent, Bauxite, Or, Cobalt
- **Mines peu profondes (20-50)** : Manganèse, Silice, Platine, Palladium, Titane
- **Mines profondes (40-70)** : Mithril, Adamantium, Orichalque, Vibranium
- **Mines abyssales (60-100)** : Obsidienne, Stellarium, Voidstone, Chronite, Cosmicium
- **Gemmes rares** : Émeraude, Saphir, Rubis, Diamant, Diamant Noir

**Stratégies spécialisées** : Chasse aux gemmes, suivi de filons, détection d'épuisement

### ⚗️ Alchimiste (16 recettes + gestion d'ingrédients)
- **Potions de base** : Soin (mineure → moyenne), Mana (mineure → moyenne)
- **Potions d'amélioration** : Force, Agilité, Résistances élémentaires
- **Potions rares** : Élixir de Vie Éternelle, Potion de Transformation
- **Potions légendaires** : Nectar Divin, Panacée Universelle
- **Utilitaires** : Rappel, Huile d'Arme, Invisibilité

**Système d'ingrédients** : 20+ ingrédients avec coûts et disponibilités
**Gestion d'atelier** : 5 niveaux d'amélioration avec calculs ROI

## 🔧 Fonctionnalités avancées

### Optimisation multi-objectifs
- **Balanced** : Équilibre entre tous les métiers
- **XP Focused** : Maximise l'expérience globale  
- **Profit Focused** : Maximise les profits
- **Leveling** : Rattrapage du métier le plus faible
- **Synergy** : Exploite les synergies inter-métiers

### Synergies entre métiers
- Fermier → Alchimiste (ingrédients botaniques)
- Mineur → Alchimiste (ingrédients minéraux)
- Production intégrée avec bonus d'efficacité jusqu'à 2.0x

### Calculs économiques
- **Rentabilité** : XP/h, Kamas/h, ROI par ressource
- **Coûts** : Temps de déplacement, gestion inventaire, respawn
- **Prédictions** : Sessions avec variance réaliste
- **Analyse comparative** : Toutes stratégies en parallèle

## 📊 Métriques et statistiques

### Globales
```python
stats = manager.get_global_statistics()
# Retourne : niveaux, XP totale, kamas gagnés, efficacité par métier
```

### Par métier
```python
profit = profession.calculate_profitability('resource_id')
# Retourne : xp_per_hour, kamas_per_hour, efficiency, success_rate, etc.
```

### Comparaison stratégies
```python
comparison = manager.compare_strategies(4.0)
# Compare les 5 stratégies sur une durée donnée
```

## 🎮 Intégration avec le bot

Le système est conçu pour s'intégrer facilement avec :

1. **Module de vision** : Recognition des ressources sur écran
2. **Système d'événements** : Notifications de récolte, level up, etc.
3. **Gestion d'état** : Suivi en temps réel du joueur
4. **Interface utilisateur** : Affichage des métriques et contrôles

### Exemple d'intégration
```python
# Dans votre bot principal
from modules.professions import ProfessionManager

# Initialisation
profession_manager = ProfessionManager()

# Exécution d'une session optimisée
session = profession_manager.optimize_global_session(
    duration_hours=4.0,
    strategy=OptimizationStrategy.PROFIT_FOCUSED
)

# Le bot suit les instructions de la session
for profession_id, time_allocation in session.profession_allocation.items():
    profession = profession_manager.get_profession(profession_id)
    optimal_resources = profession.get_optimal_route()
    
    # Implémenter la logique de farming pour chaque ressource
    # selon les patterns et calculs fournis
```

## ⚙️ Configuration

Le système sauvegarde automatiquement :
- Niveaux et expérience de tous les métiers
- Statistiques détaillées (temps, récoltes, gains)
- Préférences utilisateur (ressources favorites/blacklistées)
- Paramètres d'outils et d'ateliers

Fichier : `G:/Botting/config/professions.json`

## 🧪 Tests et validation

Lancer la démonstration complète :
```bash
cd G:/Botting
python modules/professions/example_usage.py
```

Cette démonstration teste :
- Tous les métiers individuellement
- Optimisation et exécution de sessions
- Calculs de rentabilité et recommandations
- Synergies et patterns avancés

## 🔮 Extensions possibles

Le système est extensible pour :
- **Nouveaux métiers** : Pêcheur, Forgeron, Tailleur, etc.
- **Ressources dynamiques** : Prix de marché en temps réel
- **IA avancée** : Machine learning pour optimisation adaptive
- **Multi-joueur** : Coordination entre plusieurs bots
- **Événements temporels** : Bonus saisonniers, événements spéciaux

---

*Développé avec Claude Code - Système de métiers intelligent pour botting avancé*