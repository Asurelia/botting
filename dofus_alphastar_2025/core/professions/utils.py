"""
Utilitaires pour le système de métiers.
Fonctions d'aide pour développeurs et débogage.
"""

import json
from typing import Dict, List, Any, Optional
from .profession_manager import ProfessionManager
from .base import BaseProfession

def print_profession_summary(profession: BaseProfession) -> None:
    """Affiche un résumé détaillé d'un métier"""
    print(f"\n{'='*50}")
    print(f"MÉTIER: {profession.name.upper()}")
    print(f"{'='*50}")
    
    # Stats générales
    stats = profession.stats
    print(f"Niveau: {stats.level}")
    print(f"Expérience: {stats.experience:,}")
    print(f"Total récolté: {stats.total_harvested:,}")
    if hasattr(stats, 'total_crafted'):
        print(f"Total crafté: {stats.total_crafted:,}")
    print(f"Kamas gagnés: {stats.kamas_earned:,}")
    if stats.time_spent > 0:
        print(f"Temps joué: {stats.time_spent/3600:.1f}h")
        print(f"XP/h moyen: {(stats.experience / (stats.time_spent/3600)):.0f}")
        print(f"Kamas/h moyen: {(stats.kamas_earned / (stats.time_spent/3600)):.0f}")
    
    # Ressources disponibles
    all_resources = profession.resources
    available_resources = profession.get_available_resources()
    print(f"\nRessources: {len(available_resources)}/{len(all_resources)} disponibles")
    
    # Top 5 des meilleures ressources
    if available_resources:
        print(f"\nTop 5 des ressources (par rentabilité):")
        top_resources = []
        for resource in available_resources:
            profit = profession.calculate_profitability(resource.id)
            kamas_per_hour = profit.get('kamas_per_hour', 0)
            top_resources.append((resource, kamas_per_hour))
        
        top_resources.sort(key=lambda x: x[1], reverse=True)
        for i, (resource, kamas_h) in enumerate(top_resources[:5], 1):
            xp_h = profession.calculate_xp_per_hour(resource.id)
            print(f"  {i}. {resource.name} (Niv.{resource.level_required}): "
                  f"{kamas_h:.0f} kamas/h, {xp_h:.0f} XP/h")

def benchmark_all_strategies(manager: ProfessionManager, duration_hours: float = 2.0) -> Dict[str, Any]:
    """Compare toutes les stratégies et retourne les résultats"""
    from .profession_manager import OptimizationStrategy
    
    results = {}
    strategies = [
        OptimizationStrategy.BALANCED,
        OptimizationStrategy.XP_FOCUSED,
        OptimizationStrategy.PROFIT_FOCUSED,
        OptimizationStrategy.LEVELING,
        OptimizationStrategy.SYNERGY
    ]
    
    print(f"\n🔍 BENCHMARK DES STRATÉGIES ({duration_hours}h)")
    print("="*60)
    
    for strategy in strategies:
        session = manager.optimize_global_session(duration_hours, strategy)
        totals = session.expected_results.get('totals', {})
        
        xp = totals.get('total_xp', totals.get('avg_xp_per_hour', 0) * duration_hours)
        kamas = totals.get('total_kamas', totals.get('avg_kamas_per_hour', 0) * duration_hours)
        
        results[strategy.value] = {
            'total_xp': xp,
            'total_kamas': kamas,
            'xp_per_hour': xp / duration_hours,
            'kamas_per_hour': kamas / duration_hours,
            'synergies': len(session.synergies_used)
        }
        
        print(f"{strategy.value.replace('_', ' ').title():<20}: "
              f"{xp:>8,.0f} XP | {kamas:>10,.0f} kamas | "
              f"{len(session.synergies_used)} synergies")
    
    # Trouver les meilleures
    best_xp = max(results.values(), key=lambda x: x['total_xp'])
    best_kamas = max(results.values(), key=lambda x: x['total_kamas'])
    
    print(f"\n🏆 Meilleure XP: {best_xp['xp_per_hour']:,.0f} XP/h")
    print(f"🏆 Meilleurs kamas: {best_kamas['kamas_per_hour']:,.0f} kamas/h")
    
    return results

def export_profession_data(manager: ProfessionManager, filepath: str = "profession_export.json") -> None:
    """Exporte toutes les données des métiers vers un fichier JSON"""
    export_data = {
        'export_timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'global_stats': manager.global_stats,
        'professions': {}
    }
    
    for name, profession in manager.professions.items():
        resources_data = {}
        for res_id, resource in profession.resources.items():
            profitability = profession.calculate_profitability(res_id)
            resources_data[res_id] = {
                'name': resource.name,
                'level_required': resource.level_required,
                'market_value': resource.market_value,
                'profitability': profitability
            }
        
        export_data['professions'][name] = {
            'stats': {
                'level': profession.stats.level,
                'experience': profession.stats.experience,
                'kamas_earned': profession.stats.kamas_earned,
                'time_spent_hours': round(profession.stats.time_spent / 3600, 2)
            },
            'resources_count': len(profession.resources),
            'available_count': len(profession.get_available_resources()),
            'top_resources': resources_data
        }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Données exportées vers {filepath}")

def simulate_progression(manager: ProfessionManager, sessions: int = 10) -> None:
    """Simule une progression sur plusieurs sessions"""
    from .profession_manager import OptimizationStrategy
    import random
    
    print(f"\n⚡ SIMULATION DE PROGRESSION ({sessions} sessions)")
    print("="*60)
    
    strategies = [
        OptimizationStrategy.BALANCED,
        OptimizationStrategy.XP_FOCUSED,
        OptimizationStrategy.PROFIT_FOCUSED
    ]
    
    initial_levels = {name: prof.stats.level for name, prof in manager.professions.items()}
    total_xp = 0
    total_kamas = 0
    
    for session_num in range(1, sessions + 1):
        # Stratégie aléatoire
        strategy = random.choice(strategies)
        duration = random.uniform(1.0, 4.0)  # 1-4h
        
        # Optimiser et exécuter
        session = manager.optimize_global_session(duration, strategy)
        results = manager.execute_session(session)
        
        session_xp = results['totals']['total_xp_gained']
        session_kamas = results['totals']['total_kamas_gained']
        
        total_xp += session_xp
        total_kamas += session_kamas
        
        if session_num % 2 == 0 or session_num == sessions:  # Afficher toutes les 2 sessions
            print(f"Session {session_num:2d}: {strategy.value:<15} "
                  f"({duration:.1f}h) -> +{session_xp:>6,.0f} XP, +{session_kamas:>8,.0f} kamas")
    
    # Résumé final
    print(f"\n📈 RÉSUMÉ DE LA PROGRESSION")
    print("-" * 40)
    final_levels = {name: prof.stats.level for name, prof in manager.professions.items()}
    
    for name in manager.professions.keys():
        level_gain = final_levels[name] - initial_levels[name]
        print(f"{name.title():<12}: Niveau {initial_levels[name]:2d} -> {final_levels[name]:2d} (+{level_gain})")
    
    print(f"\nTotaux: +{total_xp:,} XP, +{total_kamas:,} kamas")
    avg_session_xp = total_xp / sessions
    avg_session_kamas = total_kamas / sessions
    print(f"Moyenne: {avg_session_xp:,.0f} XP/session, {avg_session_kamas:,.0f} kamas/session")

def analyze_resource_efficiency(profession: BaseProfession) -> Dict[str, Any]:
    """Analyse l'efficacité de toutes les ressources d'un métier"""
    print(f"\n🔬 ANALYSE D'EFFICACITÉ - {profession.name.upper()}")
    print("="*60)
    
    analysis = {
        'by_level': {},
        'by_efficiency': [],
        'by_profitability': [],
        'recommendations': []
    }
    
    # Grouper par tranches de niveau
    level_ranges = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
    
    for min_level, max_level in level_ranges:
        range_resources = [r for r in profession.resources.values() 
                          if min_level <= r.level_required <= max_level]
        
        if range_resources:
            # Meilleure ressource de cette tranche
            best_resource = None
            best_score = 0
            
            for resource in range_resources:
                profit = profession.calculate_profitability(resource.id)
                score = profit.get('kamas_per_hour', 0) + profit.get('xp_per_hour', 0) * 0.1
                
                if score > best_score:
                    best_score = score
                    best_resource = resource
            
            if best_resource:
                analysis['by_level'][f"{min_level}-{max_level}"] = {
                    'resource': best_resource.name,
                    'level': best_resource.level_required,
                    'score': round(best_score, 1)
                }
                
                print(f"Niveau {min_level:2d}-{max_level:2d}: {best_resource.name:<20} "
                      f"(Niv.{best_resource.level_required:2d}) - Score: {best_score:>8.1f}")
    
    # Top 10 par efficacité pure (XP/temps)
    all_resources = list(profession.resources.values())
    efficiency_scores = []
    
    for resource in all_resources:
        if profession.can_harvest(resource.id):
            profit = profession.calculate_profitability(resource.id)
            efficiency = profit.get('efficiency', 0)  # XP par seconde
            efficiency_scores.append((resource, efficiency))
    
    efficiency_scores.sort(key=lambda x: x[1], reverse=True)
    analysis['by_efficiency'] = efficiency_scores[:10]
    
    print(f"\n🚀 TOP 10 EFFICACITÉ (XP/temps):")
    for i, (resource, efficiency) in enumerate(efficiency_scores[:10], 1):
        print(f"{i:2d}. {resource.name:<20} (Niv.{resource.level_required:2d}): {efficiency:.3f} XP/sec")
    
    return analysis

if __name__ == "__main__":
    # Démonstration des utilitaires
    print("🛠️  UTILITAIRES SYSTÈME DE MÉTIERS")
    
    # Créer le gestionnaire
    manager = ProfessionManager()
    
    # Résumé de chaque métier
    for name, profession in manager.professions.items():
        print_profession_summary(profession)
    
    # Benchmark des stratégies
    benchmark_all_strategies(manager, 2.0)
    
    # Simulation de progression
    simulate_progression(manager, 5)
    
    # Analyse d'efficacité du fermier
    farmer = manager.get_profession('farmer')
    analyze_resource_efficiency(farmer)
    
    print(f"\n✅ Démonstration des utilitaires terminée!")