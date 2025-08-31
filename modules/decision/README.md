# Module de Décision Intelligent

Le module de décision fournit un système centralisé et intelligent pour la prise de décision dans le bot Dofus. Il combine un moteur de décision multi-critères avec un sélecteur de stratégies adaptatif.

## 🎯 Fonctionnalités Principales

### Moteur de Décision (`DecisionEngine`)
- **Évaluation multi-critères** avec poids configurables
- **Gestion des priorités** (Critique > Élevée > Moyenne > Faible)
- **Évaluation contextuelle** selon la situation du personnage
- **Apprentissage automatique** des préférences utilisateur
- **Résolution des conflits** entre modules
- **Recommandations intelligentes** avec explications

### Sélecteur de Stratégies (`StrategySelector`)
- **6 stratégies prédéfinies** (Agressive, Défensive, Équilibrée, Efficace, Furtive, Sociale)
- **Détection automatique** de 10 situations différentes
- **Adaptation dynamique** selon les conditions changeantes
- **Apprentissage des performances** par stratégie
- **Métriques de performance** avec analyse historique

### Gestionnaire de Configuration (`DecisionConfigManager`)
- **6 profils prédéfinis** pour différents styles de jeu
- **Configuration personnalisable** des priorités et seuils
- **Interface simple** pour la modification des paramètres
- **Sauvegarde/chargement** automatique des préférences
- **Export/import** de configurations

## 📁 Structure du Module

```
modules/decision/
├── __init__.py              # Point d'entrée du module
├── decision_engine.py       # Moteur de décision centralisé
├── strategy_selector.py     # Sélecteur de stratégies adaptatif
├── config.py               # Gestionnaire de configuration
├── example_usage.py        # Exemples d'utilisation
└── README.md              # Documentation
```

## 🚀 Utilisation Rapide

### Initialisation Basique

```python
from modules.decision import DecisionEngine, StrategySelector
from modules.decision.config import DecisionConfigManager

# Initialisation avec configuration par défaut
config_manager = DecisionConfigManager()
engine = DecisionEngine()
selector = StrategySelector()

# Configuration automatique
config_manager.configure_decision_engine(engine)
config_manager.configure_strategy_selector(selector)
```

### Prise de Décision

```python
from modules.decision.decision_engine import DecisionContext, Decision, Priority, ActionType

# Créer le contexte actuel
context = DecisionContext(
    health_percent=75.0,
    mana_percent=80.0,
    in_combat=False,
    safe_zone=True,
    current_objective="farm_cereals"
)

# Définir les actions possibles
decisions = [
    Decision(
        action_id="harvest_wheat",
        action_type=ActionType.PROFESSION,
        priority=Priority.MEDIUM,
        confidence=0.9,
        estimated_duration=15.0,
        success_probability=0.95,
        risk_level=0.1,
        reward_estimate=0.7
    ),
    # ... autres décisions
]

# Prendre la meilleure décision
best_decision = engine.make_decision(decisions, context)
print(f"Action choisie: {best_decision.action_id}")
```

### Sélection de Stratégie

```python
# Sélectionner la stratégie optimale
strategy_type, strategy_config = selector.select_strategy(context)
print(f"Stratégie: {strategy_type.value}")

# Obtenir des recommandations
recommendations = selector.get_strategy_recommendations(context, top_n=3)
for strategy, score, explanation in recommendations:
    print(f"{strategy.value}: {score:.2f} - {explanation}")
```

## ⚙️ Configuration

### Profils Prédéfinis

- **farmer_safe**: Farming sécurisé avec priorité à la survie
- **farmer_efficient**: Farming efficace avec optimisation du temps
- **combat_aggressive**: Combat agressif pour maximiser les gains
- **combat_defensive**: Combat défensif avec priorité à la sécurité
- **explorer_balanced**: Exploration équilibrée entre risque et efficacité
- **social_cooperative**: Jeu coopératif avec interactions sociales

### Application d'un Profil

```python
# Appliquer un profil prédéfini
config_manager.apply_profile('farmer_efficient')

# Personnaliser les priorités
config_manager.update_priority_weights({
    'survival': 2.5,
    'efficiency': 1.8,
    'maintenance': 0.5
})

# Modifier les seuils d'activation
config_manager.update_activation_thresholds({
    'critical_health': 25.0,
    'low_health': 50.0,
    'full_inventory': 85.0
})
```

## 🧠 Système d'Apprentissage

Le système apprend automatiquement des résultats des décisions :

```python
# Mettre à jour le résultat d'une décision
engine.update_decision_outcome(
    decision_id="harvest_wheat_123456",
    success=True,
    actual_duration=18.0,
    actual_reward=0.8
)

# Mettre à jour les performances d'une stratégie
selector.update_strategy_outcome(
    strategy_type=StrategyType.EFFICIENT,
    success=True,
    reward=0.7,
    duration=120.0
)
```

## 📊 Statistiques et Analytics

```python
# Statistiques du moteur de décision
stats = engine.get_decision_stats()
print(f"Taux de succès: {stats['success_rate']:.1%}")
print(f"Décisions totales: {stats['total_decisions']}")

# Analytics des stratégies
analytics = selector.get_strategy_analytics()
print(f"Meilleure stratégie: {analytics['best_performing_strategy']}")
```

## 🎛️ Situations Détectées Automatiquement

1. **PEACEFUL_FARMING**: Farm tranquille sans danger
2. **DANGEROUS_AREA**: Zone avec risques élevés
3. **CROWDED_AREA**: Zone avec beaucoup de joueurs
4. **DUNGEON_EXPLORATION**: Exploration de donjon
5. **PVP_ZONE**: Zone de combat PvP
6. **RESOURCE_COMPETITION**: Compétition pour ressources
7. **BOSS_FIGHT**: Combat de boss
8. **LOW_RESOURCES**: Ressources (vie, mana) faibles
9. **INVENTORY_FULL**: Inventaire plein
10. **MISSION_CRITICAL**: Mission importante en cours

## 🔧 Types d'Actions Supportées

- **SURVIVAL**: Actions de survie (heal, potions)
- **COMBAT**: Actions de combat (attaque, sorts)
- **MOVEMENT**: Actions de déplacement
- **PROFESSION**: Actions de métiers (farm, craft)
- **INVENTORY**: Gestion d'inventaire
- **SOCIAL**: Interactions sociales
- **MAINTENANCE**: Maintenance du bot

## 📈 Priorités du Système

1. **CRITICAL (100)**: Survie immédiate (fuite, potion critique)
2. **HIGH (80)**: Sécurité (heal, bouclier)
3. **MEDIUM (60)**: Objectifs principaux (farm, combat)
4. **LOW (40)**: Efficacité (optimisation, confort)
5. **MINIMAL (20)**: Maintenance (tri inventaire, réparation)

## 💾 Sauvegarde et Persistance

```python
# Sauvegarder l'état complet
config_manager.save_engine_state(engine)
config_manager.save_strategy_state(selector)

# Exporter la configuration
config_manager.export_config("my_config.json")

# Importer une configuration
config_manager.import_config("my_config.json")
```

## 🚨 Gestion des Conflits

Le système résout automatiquement les conflits entre modules :

- **Combat vs Profession**: Priorité au combat si en danger
- **Mouvement vs Action**: Évite les mouvements pendant actions importantes
- **Inventaire vs Objectif**: Priorité à l'inventaire si pods pleins

## 🔮 Recommandations Intelligentes

Le système génère des explications pour ses recommandations :

```python
recommendations = engine.get_recommendations(decisions, context, top_n=3)
for decision, score, explanation in recommendations:
    print(f"{decision.action_id}: {explanation}")
```

Exemple de sortie :
```
harvest_wheat: Meilleure option - priorité élevée, adapté au farm tranquille, haute probabilité de succès
move_to_farming_area: Option #2 - action rapide, zone sûre disponible
bank_valuable_items: Option #3 - optimisation à long terme, récompenses élevées historiques
```

## 🎯 Intégration avec d'Autres Modules

```python
# Exemple d'intégration avec le module combat
from modules.combat.ai.combat_ai import CombatAI

class IntelligentBot:
    def __init__(self):
        self.config_manager = DecisionConfigManager()
        self.decision_engine = DecisionEngine()
        self.strategy_selector = StrategySelector()
        self.combat_ai = CombatAI()
        
        # Configuration
        self.config_manager.configure_decision_engine(self.decision_engine)
        self.config_manager.configure_strategy_selector(self.strategy_selector)
    
    def run_decision_loop(self):
        while True:
            # Obtenir le contexte actuel
            context = self.get_current_context()
            
            # Sélectionner la stratégie
            strategy, _ = self.strategy_selector.select_strategy(context)
            
            # Obtenir les actions possibles de tous les modules
            possible_actions = []
            possible_actions.extend(self.combat_ai.get_possible_actions(context))
            possible_actions.extend(self.profession_manager.get_possible_actions(context))
            # ... autres modules
            
            # Prendre la meilleure décision
            best_action = self.decision_engine.make_decision(possible_actions, context)
            
            # Exécuter l'action
            if best_action:
                self.execute_action(best_action)
```

---

*Ce module fournit une base solide pour un système de décision intelligent et adaptatif, capable d'apprendre et de s'améliorer au fil du temps.*