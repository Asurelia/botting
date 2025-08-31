"""
Exemple d'utilisation du module de décision.

Ce fichier démontre comment utiliser le moteur de décision et le sélecteur
de stratégies dans différentes situations de jeu.
"""

import logging
import time
from typing import List

# Imports du module de décision
from .decision_engine import (
    DecisionEngine, Decision, DecisionContext, 
    Priority, ActionType
)
from .strategy_selector import StrategySelector, StrategyType, Situation
from .config import DecisionConfigManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_sample_decisions() -> List[Decision]:
    """Crée des exemples de décisions possibles."""
    decisions = [
        # Décisions de survie
        Decision(
            action_id="heal_critical",
            action_type=ActionType.SURVIVAL,
            priority=Priority.CRITICAL,
            confidence=0.95,
            estimated_duration=3.0,
            success_probability=0.98,
            risk_level=0.1,
            reward_estimate=0.9,
            prerequisites=["health_below_30"],
            module_source="combat"
        ),
        Decision(
            action_id="drink_mana_potion",
            action_type=ActionType.SURVIVAL,
            priority=Priority.HIGH,
            confidence=0.9,
            estimated_duration=2.0,
            success_probability=0.99,
            risk_level=0.05,
            reward_estimate=0.7,
            prerequisites=["mana_below_20"],
            module_source="combat"
        ),
        
        # Décisions de combat
        Decision(
            action_id="attack_weakest_enemy",
            action_type=ActionType.COMBAT,
            priority=Priority.MEDIUM,
            confidence=0.8,
            estimated_duration=5.0,
            success_probability=0.85,
            risk_level=0.4,
            reward_estimate=0.8,
            prerequisites=["in_combat"],
            module_source="combat"
        ),
        Decision(
            action_id="cast_area_spell",
            action_type=ActionType.COMBAT,
            priority=Priority.MEDIUM,
            confidence=0.7,
            estimated_duration=8.0,
            success_probability=0.75,
            risk_level=0.6,
            reward_estimate=1.2,
            prerequisites=["mana_above_50", "multiple_enemies"],
            module_source="combat"
        ),
        
        # Décisions de profession
        Decision(
            action_id="harvest_wheat",
            action_type=ActionType.PROFESSION,
            priority=Priority.LOW,
            confidence=0.95,
            estimated_duration=15.0,
            success_probability=0.98,
            risk_level=0.1,
            reward_estimate=0.6,
            prerequisites=["not_in_combat", "wheat_available"],
            module_source="profession"
        ),
        Decision(
            action_id="mine_iron_ore",
            action_type=ActionType.PROFESSION,
            priority=Priority.LOW,
            confidence=0.9,
            estimated_duration=20.0,
            success_probability=0.9,
            risk_level=0.2,
            reward_estimate=0.8,
            prerequisites=["not_in_combat", "pickaxe_equipped"],
            module_source="profession"
        ),
        
        # Décisions de déplacement
        Decision(
            action_id="move_to_safe_zone",
            action_type=ActionType.MOVEMENT,
            priority=Priority.HIGH,
            confidence=0.85,
            estimated_duration=10.0,
            success_probability=0.9,
            risk_level=0.3,
            reward_estimate=0.5,
            prerequisites=["not_safe_zone"],
            module_source="navigation"
        ),
        Decision(
            action_id="move_to_farming_area",
            action_type=ActionType.MOVEMENT,
            priority=Priority.MEDIUM,
            confidence=0.9,
            estimated_duration=30.0,
            success_probability=0.95,
            risk_level=0.2,
            reward_estimate=0.4,
            prerequisites=["safe_zone"],
            module_source="navigation"
        ),
        
        # Décisions d'inventaire
        Decision(
            action_id="drop_low_value_items",
            action_type=ActionType.INVENTORY,
            priority=Priority.MEDIUM,
            confidence=0.8,
            estimated_duration=5.0,
            success_probability=0.95,
            risk_level=0.1,
            reward_estimate=0.3,
            prerequisites=["inventory_above_80"],
            module_source="inventory"
        ),
        Decision(
            action_id="bank_valuable_items",
            action_type=ActionType.INVENTORY,
            priority=Priority.LOW,
            confidence=0.9,
            estimated_duration=45.0,
            success_probability=0.98,
            risk_level=0.1,
            reward_estimate=0.7,
            prerequisites=["near_bank"],
            module_source="inventory"
        )
    ]
    
    return decisions


def create_sample_contexts() -> List[DecisionContext]:
    """Crée des exemples de contextes de décision."""
    contexts = [
        # Contexte 1: Farming paisible
        DecisionContext(
            health_percent=85.0,
            mana_percent=90.0,
            pod_percent=45.0,
            in_combat=False,
            enemies_count=0,
            allies_count=0,
            combat_difficulty=0.0,
            current_map="Champs de Cania",
            safe_zone=True,
            resources_available=["wheat", "barley", "flax"],
            current_objective="farm_cereals",
            objective_progress=0.3,
            session_time=1800.0,
            risk_tolerance=0.4,
            efficiency_focus=0.7
        ),
        
        # Contexte 2: Combat dangereux
        DecisionContext(
            health_percent=25.0,
            mana_percent=15.0,
            pod_percent=70.0,
            in_combat=True,
            enemies_count=3,
            allies_count=1,
            combat_difficulty=0.8,
            current_map="Donjon des Scarafeuilles",
            safe_zone=False,
            resources_available=[],
            current_objective="complete_dungeon",
            objective_progress=0.6,
            session_time=3600.0,
            risk_tolerance=0.3,
            efficiency_focus=0.4
        ),
        
        # Contexte 3: Exploration équilibrée
        DecisionContext(
            health_percent=65.0,
            mana_percent=70.0,
            pod_percent=85.0,
            in_combat=False,
            enemies_count=1,
            allies_count=2,
            combat_difficulty=0.4,
            current_map="Forêt Malléfique",
            safe_zone=False,
            resources_available=["ash_wood", "chestnut_wood"],
            current_objective="explore_area",
            objective_progress=0.8,
            session_time=5400.0,
            risk_tolerance=0.6,
            efficiency_focus=0.5
        )
    ]
    
    return contexts


def demonstrate_decision_engine():
    """Démontre l'utilisation du moteur de décision."""
    print("=== DÉMONSTRATION DU MOTEUR DE DÉCISION ===\n")
    
    # Créer le moteur de décision
    engine = DecisionEngine()
    
    # Préparer les données d'exemple
    decisions = create_sample_decisions()
    contexts = create_sample_contexts()
    
    for i, context in enumerate(contexts, 1):
        print(f"--- Scénario {i}: {context.current_objective} ---")
        print(f"Santé: {context.health_percent}%, Mana: {context.mana_percent}%")
        print(f"En combat: {context.in_combat}, Zone sûre: {context.safe_zone}")
        
        # Prendre une décision
        best_decision = engine.make_decision(decisions, context)
        
        if best_decision:
            print(f"✅ Décision choisie: {best_decision.action_id}")
            print(f"   Type: {best_decision.action_type.value}")
            print(f"   Priorité: {best_decision.priority.name}")
            print(f"   Confiance: {best_decision.confidence:.2f}")
            print(f"   Durée estimée: {best_decision.estimated_duration:.1f}s")
        else:
            print("❌ Aucune décision valide trouvée")
        
        # Obtenir des recommandations
        print("\n📋 Recommandations:")
        recommendations = engine.get_recommendations(decisions, context, top_n=3)
        for j, (decision, score, explanation) in enumerate(recommendations, 1):
            print(f"   {j}. {decision.action_id} (score: {score:.1f})")
            print(f"      {explanation}")
        
        print("\n" + "="*60 + "\n")


def demonstrate_strategy_selector():
    """Démontre l'utilisation du sélecteur de stratégies."""
    print("=== DÉMONSTRATION DU SÉLECTEUR DE STRATÉGIES ===\n")
    
    # Créer le sélecteur
    selector = StrategySelector()
    
    # Préparer les contextes
    contexts = create_sample_contexts()
    context_names = ["Farming paisible", "Combat dangereux", "Exploration équilibrée"]
    
    for i, (context, name) in enumerate(zip(contexts, context_names)):
        print(f"--- Scénario: {name} ---")
        
        # Sélectionner une stratégie
        strategy_type, strategy_config = selector.select_strategy(context)
        
        print(f"🎯 Stratégie sélectionnée: {strategy_type.value}")
        print(f"   Description: {strategy_config.description}")
        print(f"   Poids - Survie: {strategy_config.weights.survival:.1f}, "
              f"Efficacité: {strategy_config.weights.efficiency:.1f}")
        print(f"   Tolérance risque: {strategy_config.weights.risk_tolerance:.1f}")
        
        # Obtenir des recommandations de stratégies
        print("\n📊 Recommandations de stratégies:")
        recommendations = selector.get_strategy_recommendations(context, top_n=3)
        for j, (strat_type, score, explanation) in enumerate(recommendations, 1):
            print(f"   {j}. {strat_type.value} (score: {score:.2f})")
            print(f"      {explanation}")
        
        # Simuler un résultat et mettre à jour
        success = True if i != 1 else False  # Le combat dangereux échoue
        reward = 0.8 if success else 0.2
        duration = 120.0
        
        selector.update_strategy_outcome(strategy_type, success, reward, duration)
        print(f"\n📈 Résultat simulé: {'Succès' if success else 'Échec'} "
              f"(récompense: {reward}, durée: {duration}s)")
        
        print("\n" + "="*60 + "\n")


def demonstrate_config_manager():
    """Démontre l'utilisation du gestionnaire de configuration."""
    print("=== DÉMONSTRATION DU GESTIONNAIRE DE CONFIGURATION ===\n")
    
    # Créer le gestionnaire
    config_manager = DecisionConfigManager()
    
    # Afficher les profils disponibles
    print("📁 Profils disponibles:")
    for profile in config_manager.get_available_profiles():
        description = config_manager.get_profile_description(profile)
        print(f"   - {profile}: {description}")
    
    print("\n" + "-"*50 + "\n")
    
    # Appliquer différents profils et montrer les différences
    test_profiles = ['farmer_safe', 'combat_aggressive', 'explorer_balanced']
    
    for profile in test_profiles:
        print(f"🔧 Application du profil: {profile}")
        config_manager.apply_profile(profile)
        
        config = config_manager.get_config()
        print(f"   Poids survie: {config.priority_weights['survival']:.1f}")
        print(f"   Poids efficacité: {config.priority_weights['efficiency']:.1f}")
        print(f"   Limite combat: {config.time_limits['combat']:.0f}s")
        print(f"   Seuil santé critique: {config.activation_thresholds['critical_health']:.0f}%")
        
        print()
    
    # Démonstration des modifications personnalisées
    print("🎛️  Personnalisation des priorités:")
    config_manager.update_priority_weights({
        'survival': 3.0,
        'efficiency': 0.5
    })
    
    config_manager.update_time_limits({
        'combat': 30.0,  # Combat très court
        'profession': 7200.0  # Farming très long
    })
    
    print("   ✅ Priorités personnalisées appliquées")
    
    # Configuration d'un moteur avec le gestionnaire
    engine = DecisionEngine()
    selector = StrategySelector()
    
    config_manager.configure_decision_engine(engine)
    config_manager.configure_strategy_selector(selector)
    
    print("   ✅ Moteur et sélecteur configurés")
    
    print("\n" + "="*60 + "\n")


def demonstrate_complete_workflow():
    """Démontre un workflow complet avec tous les composants."""
    print("=== WORKFLOW COMPLET ===\n")
    
    # Initialisation
    config_manager = DecisionConfigManager()
    engine = DecisionEngine()
    selector = StrategySelector()
    
    # Configuration pour farming efficace
    config_manager.apply_profile('farmer_efficient')
    config_manager.configure_decision_engine(engine)
    config_manager.configure_strategy_selector(selector)
    
    print("🚀 Système initialisé avec profil 'farmer_efficient'\n")
    
    # Simulation d'une session de jeu
    decisions = create_sample_decisions()
    farming_context = create_sample_contexts()[0]  # Contexte de farming
    
    print("📍 Situation: Farming de céréales")
    print(f"   Santé: {farming_context.health_percent}%")
    print(f"   Inventaire: {farming_context.pod_percent}%")
    print(f"   Objectif: {farming_context.current_objective}")
    
    # 1. Sélectionner la stratégie
    strategy_type, strategy_config = selector.select_strategy(farming_context)
    print(f"\n🎯 Stratégie: {strategy_type.value}")
    
    # 2. Prendre une décision
    best_decision = engine.make_decision(decisions, farming_context)
    print(f"⚡ Action: {best_decision.action_id}")
    
    # 3. Simuler l'exécution
    print(f"⏱️  Exécution pendant {best_decision.estimated_duration:.0f}s...")
    time.sleep(0.1)  # Simulation
    
    # 4. Mettre à jour les résultats
    success = True
    actual_duration = best_decision.estimated_duration * 1.1  # Légèrement plus long
    actual_reward = 0.7
    
    engine.update_decision_outcome(
        f"{best_decision.action_id}_{int(time.time())}", 
        success, 
        actual_duration, 
        actual_reward
    )
    
    selector.update_strategy_outcome(
        strategy_type, 
        success, 
        actual_reward, 
        actual_duration
    )
    
    print(f"✅ Résultat: {'Succès' if success else 'Échec'}")
    print(f"   Récompense: {actual_reward}")
    print(f"   Durée: {actual_duration:.1f}s")
    
    # 5. Afficher les statistiques
    print("\n📊 Statistiques:")
    engine_stats = engine.get_decision_stats()
    strategy_stats = selector.get_strategy_analytics()
    
    print(f"   Décisions prises: {engine_stats.get('total_decisions', 0)}")
    print(f"   Taux de succès: {engine_stats.get('success_rate', 0):.1%}")
    print(f"   Usage stratégies: {strategy_stats.get('total_usage', 0)}")
    
    # 6. Sauvegarder l'état
    config_manager.save_engine_state(engine)
    config_manager.save_strategy_state(selector)
    print("\n💾 État sauvegardé")
    
    print("\n" + "="*60)


def main():
    """Fonction principale qui exécute tous les exemples."""
    print("🤖 SYSTÈME DE DÉCISION INTELLIGENT - DÉMONSTRATIONS")
    print("="*70 + "\n")
    
    try:
        # Exécuter toutes les démonstrations
        demonstrate_decision_engine()
        demonstrate_strategy_selector()
        demonstrate_config_manager()
        demonstrate_complete_workflow()
        
        print("\n✅ Toutes les démonstrations terminées avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur pendant la démonstration: {e}")
        logging.exception("Erreur complète:")


if __name__ == "__main__":
    main()