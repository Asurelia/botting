"""
Exemple d'utilisation du système de métiers complet.
Démontre toutes les fonctionnalités principales des modules de professions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.professions import (
    ProfessionManager, OptimizationStrategy,
    Farmer, Lumberjack, Miner, Alchemist
)

def demo_individual_professions():
    """Démonstration des métiers individuels"""
    print("=" * 60)
    print("DÉMONSTRATION DES MÉTIERS INDIVIDUELS")
    print("=" * 60)
    
    # === FERMIER ===
    print("\n🌾 FERMIER")
    print("-" * 40)
    farmer = Farmer()
    print(f"Métier: {farmer}")
    
    # Meilleures ressources niveau débutant
    best_farming = farmer.get_best_resources_by_level(15, 3)
    print(f"\nMeilleures ressources niveau 15:")
    for resource in best_farming:
        profit = farmer.calculate_profitability(resource.id)
        print(f"  • {resource.name} (Niv.{resource.level_required}): "
              f"{profit['kamas_per_hour']:.0f} kamas/h, {profit['xp_per_hour']:.0f} XP/h")
    
    # Route optimisée
    route = farmer.get_optimal_route((1, 20))
    print(f"\nRoute optimisée (niveaux 1-20): {len(route)} ressources")
    pattern = farmer.get_farming_pattern(route[:5], 'cluster')
    print(f"Pattern de farming: cluster sur {len(pattern)} coordonnées")
    
    # === BÛCHERON ===
    print("\n🪓 BÛCHERON")
    print("-" * 40)
    lumberjack = Lumberjack()
    print(f"Métier: {lumberjack}")
    
    # Amélioration d'outil
    tool_recs = lumberjack.get_tool_recommendations()
    print(f"\nRecommandations d'outils: {len(tool_recs)}")
    for tool, rec in list(tool_recs.items())[:2]:
        if rec['recommended']:
            print(f"  • {tool}: +{rec['improvement_percent']:.1f}% efficacité")
    
    # Pattern de coupe
    trees = lumberjack.get_available_resources((20, 40))
    cutting_pattern = lumberjack.get_cutting_pattern(trees[:8], 'sustainable')
    print(f"\nPattern durable: {cutting_pattern['strategy']}")
    print(f"Ressources sélectionnées: {len(cutting_pattern['resources'])}")
    
    # === MINEUR ===
    print("\n⛏️ MINEUR")
    print("-" * 40)
    miner = Miner()
    print(f"Métier: {miner}")
    
    # Stratégie gemmes
    gem_strategy = miner.get_gem_hunting_strategy()
    if 'recommended_gems' in gem_strategy:
        print(f"\nStratégie gemmes: {len(gem_strategy['recommended_gems'])} gemmes recommandées")
        print(f"Profit quotidien estimé: {gem_strategy['estimated_daily_profit']:.0f} kamas")
        print(f"⚠️  {gem_strategy['warning']}")
    
    # Analyse de dépletion
    common_ores = ['fer', 'cuivre', 'bronze']
    depletion = miner.estimate_forest_depletion(common_ores)
    print(f"\nAnalyse de dépletion des minerais communs:")
    for ore_id, data in depletion.items():
        if data['sustainable']:
            print(f"  • {ore_id}: Exploitation durable ✓")
        else:
            print(f"  • {ore_id}: Dépletion temporaire de {data['depletion_time_minutes']:.1f}min")
    
    # === ALCHIMISTE ===
    print("\n⚗️ ALCHIMISTE")
    print("-" * 40)
    alchemist = Alchemist()
    print(f"Métier: {alchemist}")
    
    # Session de crafting
    session_estimate = alchemist.estimate_crafting_session('potion_soin_moyenne', 20, 2.0)
    print(f"\nSession de crafting '{session_estimate['recipe_name']}':")
    print(f"  Objectif: {session_estimate['target_quantity']} potions")
    print(f"  Réalisable: {session_estimate['achievable_quantity']} potions")
    print(f"  Temps: {session_estimate['session_time_hours']}h")
    print(f"  Profit: {session_estimate['total_profit']:.0f} kamas")
    print(f"  Rentabilité: {session_estimate['profitability_rating']}")
    
    # Gestion des ingrédients
    recipes_to_craft = ['potion_soin_legere', 'potion_mana_mineure']
    ingredient_mgmt = alchemist.manage_ingredient_stock(recipes_to_craft, 3.0)
    print(f"\nGestion ingrédients pour {ingredient_mgmt['total_recipes']} recettes:")
    print(f"  Coût total: {ingredient_mgmt['total_shopping_cost']:.0f} kamas")
    print(f"  Articles à acheter: {len(ingredient_mgmt['shopping_list'])}")
    print(f"  Articles à récolter: {len(ingredient_mgmt['harvesting_list'])}")

def demo_profession_manager():
    """Démonstration du gestionnaire de métiers"""
    print("\n" + "=" * 60)
    print("GESTIONNAIRE GLOBAL DE MÉTIERS")
    print("=" * 60)
    
    # Initialisation
    manager = ProfessionManager()
    print(f"Gestionnaire initialisé: {manager}")
    
    # Statistiques globales
    global_stats = manager.get_global_statistics()
    print(f"\n📊 STATISTIQUES GLOBALES")
    print("-" * 40)
    summary = global_stats['profession_summary']
    print(f"Niveau total: {summary['total_levels']}")
    print(f"Niveau moyen: {summary['average_level']}")
    print(f"Ressources disponibles: {summary['available_resources']}/{summary['total_resources']}")
    
    print(f"\nDétails par métier:")
    for name, details in global_stats['profession_details'].items():
        print(f"  • {name.title()}: Niv.{details['level']} - "
              f"{details['kamas_earned']} kamas - "
              f"Efficacité: {details['efficiency_rating']}")
    
    print(f"\nRecommandations ({len(global_stats['recommendations'])}):")
    for i, rec in enumerate(global_stats['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Comparaison des stratégies
    print(f"\n🎯 COMPARAISON DES STRATÉGIES")
    print("-" * 40)
    strategy_comparison = manager.compare_strategies(4.0)
    
    print(f"Analyse sur {strategy_comparison['comparison_duration_hours']}h:")
    comparison = strategy_comparison['detailed_comparison']
    
    for strategy, data in comparison.items():
        print(f"\n{strategy.replace('_', ' ').title()}:")
        print(f"  XP total: {data['total_xp']:,.0f}")
        print(f"  Kamas total: {data['total_kamas']:,.0f}")
        print(f"  XP/h: {data['avg_xp_per_hour']:,.0f}")
        print(f"  Kamas/h: {data['avg_kamas_per_hour']:,.0f}")
    
    recs = strategy_comparison['recommendations']
    print(f"\n🏆 RECOMMANDATIONS:")
    print(f"  Meilleure pour XP: {recs['best_for_xp'].replace('_', ' ').title()}")
    print(f"  Meilleure pour Kamas: {recs['best_for_kamas'].replace('_', ' ').title()}")
    print(f"  Écart XP: {recs['xp_difference_percent']}%")
    print(f"  Écart Kamas: {recs['kamas_difference_percent']}%")
    
    # Optimisation et exécution d'une session
    print(f"\n🚀 SESSION OPTIMISÉE")
    print("-" * 40)
    
    # Session profit-focused
    optimal_session = manager.optimize_global_session(3.0, OptimizationStrategy.PROFIT_FOCUSED)
    print(f"Session optimisée: {optimal_session.strategy.value}")
    print(f"Durée: {optimal_session.duration_hours}h")
    
    print(f"\nAllocation du temps:")
    for prof, allocation in optimal_session.profession_allocation.items():
        time_hours = allocation * optimal_session.duration_hours
        print(f"  • {prof.title()}: {time_hours:.1f}h ({allocation*100:.1f}%)")
    
    print(f"\nRésultats attendus:")
    expected_totals = optimal_session.expected_results['totals']
    print(f"  XP total: {expected_totals['total_xp']:,.0f}")
    print(f"  Kamas total: {expected_totals['total_kamas']:,.0f}")
    if 'profit_optimization_bonus' in expected_totals:
        print(f"  Bonus optimisation: {expected_totals['profit_optimization_bonus']:.1f}x")
    
    # Simulation d'exécution
    print(f"\n⚡ EXÉCUTION SIMULÉE")
    print("-" * 40)
    execution_results = manager.execute_session(optimal_session)
    
    session_summary = execution_results['session_summary']
    totals = execution_results['totals']
    performance = execution_results['performance']
    
    print(f"Stratégie exécutée: {session_summary['strategy']}")
    print(f"Temps d'exécution: {session_summary['execution_time_seconds']}s")
    
    print(f"\nRésultats obtenus:")
    print(f"  XP gagnée: {totals['total_xp_gained']:,.0f} ({totals['average_xp_per_hour']:,.0f}/h)")
    print(f"  Kamas gagnés: {totals['total_kamas_gained']:,.0f} ({totals['average_kamas_per_hour']:,.0f}/h)")
    
    print(f"\nPerformance vs prédictions:")
    print(f"  XP: {performance['vs_expected_xp']}% de l'attendu")
    print(f"  Kamas: {performance['vs_expected_kamas']}% de l'attendu")
    
    if performance['new_records']['xp'] or performance['new_records']['kamas']:
        records = []
        if performance['new_records']['xp']:
            records.append("XP/h")
        if performance['new_records']['kamas']:
            records.append("Kamas/h")
        print(f"  🎉 Nouveau record: {' et '.join(records)}!")
    
    print(f"\nDétails par métier:")
    for prof_name, results in execution_results['results_by_profession'].items():
        print(f"  • {prof_name.title()}: "
              f"+{results['xp_gained']:,} XP, "
              f"+{results['kamas_gained']:,} kamas, "
              f"Niv.{results['new_level']}")
    
    # Sauvegarde
    manager.save_configuration()
    print(f"\n💾 Configuration sauvegardée")

def demo_advanced_features():
    """Démonstration des fonctionnalités avancées"""
    print("\n" + "=" * 60)
    print("FONCTIONNALITÉS AVANCÉES")
    print("=" * 60)
    
    # Synergies entre métiers
    manager = ProfessionManager()
    
    print(f"\n🔗 SYNERGIES ENTRE MÉTIERS")
    print("-" * 40)
    synergy_session = manager.optimize_global_session(2.0, OptimizationStrategy.SYNERGY)
    
    if synergy_session.synergies_used:
        print(f"Synergies exploitées: {len(synergy_session.synergies_used)}")
        for synergy in synergy_session.synergies_used[:3]:  # Top 3
            print(f"  • {synergy.profession1.title()} → {synergy.profession2.title()}: "
                  f"{synergy.efficiency_bonus:.1f}x bonus")
            print(f"    {synergy.description}")
        
        totals = synergy_session.expected_results['totals']
        bonus_avg = totals.get('synergy_bonus_average', 1.0)
        print(f"\nBonus moyen de synergie: {bonus_avg:.2f}x")
    else:
        print("Aucune synergie disponible au niveau actuel")
    
    # Patterns avancés
    print(f"\n🎨 PATTERNS DE FARMING AVANCÉS")
    print("-" * 40)
    
    # Pattern fermier
    farmer = Farmer()
    resources = farmer.get_available_resources((20, 40))[:6]
    
    patterns = ['linear', 'spiral', 'zigzag', 'cluster']
    print(f"Patterns de farming testés sur {len(resources)} ressources:")
    
    for pattern_name in patterns:
        coordinates = farmer.get_farming_pattern(resources, pattern_name)
        distance_total = 0
        for i in range(1, len(coordinates)):
            x1, y1 = coordinates[i-1]
            x2, y2 = coordinates[i]
            distance_total += ((x2-x1)**2 + (y2-y1)**2)**0.5
        
        print(f"  • {pattern_name.title()}: {len(coordinates)} points, "
              f"distance totale: {distance_total:.1f}")
    
    # Calculs économiques avancés
    print(f"\n💰 ANALYSES ÉCONOMIQUES")
    print("-" * 40)
    
    # ROI des améliorations
    alchemist = Alchemist()
    workshop_analysis = alchemist.get_workshop_upgrades()
    
    if 'roi_analysis' in workshop_analysis:
        roi = workshop_analysis['roi_analysis']
        print(f"Analyse ROI atelier alchimie:")
        print(f"  Niveau actuel: {workshop_analysis['current_level']}")
        print(f"  Coût amélioration: {workshop_analysis['upgrade_cost']:,} kamas")
        print(f"  Amélioration quotidienne: {roi['daily_improvement']:.0f} kamas/jour")
        print(f"  Retour sur investissement: {roi['payback_period_days']:.0f} jours")
        print(f"  Recommandation: {workshop_analysis['recommendation']}")

if __name__ == "__main__":
    print("🎮 SYSTÈME DE MÉTIERS COMPLET - DÉMONSTRATION")
    print("Développé avec Claude Code")
    
    try:
        # Démonstrations principales
        demo_individual_professions()
        demo_profession_manager() 
        demo_advanced_features()
        
        print("\n" + "=" * 60)
        print("✅ DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
        print("=" * 60)
        print("\nToutes les fonctionnalités ont été testées:")
        print("• Métiers individuels (Fermier, Bûcheron, Mineur, Alchimiste)")
        print("• Calculs de rentabilité et optimisation de routes")
        print("• Patterns de farming et stratégies de récolte")
        print("• Gestionnaire global et synergies entre métiers")
        print("• Optimisation multi-objectifs et sessions automatisées")
        print("• Analyses économiques et recommandations intelligentes")
        
        print(f"\nLe système est prêt pour l'intégration au bot principal!")
        
    except Exception as e:
        print(f"\n❌ ERREUR LORS DE LA DÉMONSTRATION: {e}")
        import traceback
        traceback.print_exc()