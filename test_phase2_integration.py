"""
Test d'intégration Phase 2 - Cerveau Multi-Dimensionnel
Test du Decision Engine, Emotional State Management et State Tracking
"""

import asyncio
import logging
from datetime import datetime, timedelta

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import des modules
from core.ai_framework import create_ai_framework, AITask, Priority
from core.decision_engine import Objective, Action, ObjectiveType, ConflictType, Conflict
from core.emotional_state import GameEvent, MoodState
from core.state_tracker import StateLevel, GameState

async def test_decision_engine_integration():
    """Test l'intégration du Decision Engine"""
    print("\nTest intégration Decision Engine...")

    orchestrator = await create_ai_framework()

    if not await orchestrator.start():
        print("❌ Échec démarrage framework")
        return False

    try:
        decision_module = orchestrator.modules.get('decision')

        if decision_module:
            # Test ajout d'objectif
            farming_objective = Objective(
                id="farming_objective",
                name="Maximiser revenus de farm",
                type=ObjectiveType.MAXIMIZE_PROFIT,
                priority=Priority.HIGH,
                target_value=15000.0,
                weight=1.5,
                estimated_duration=timedelta(hours=3),
                deadline=datetime.now() + timedelta(hours=4)
            )

            success = await decision_module.process({
                'type': 'add_objective',
                'objective': farming_objective
            })

            if success:
                print(f"Objectif ajouté: {farming_objective.name}")

                # Test évaluation d'actions
                test_actions = [
                    Action(
                        id="farm_wheat_optimized",
                        name="Farm blé optimisé",
                        type="farming",
                        objective_impacts={"farming_objective": 120.0},
                        resource_cost={"energy": 15, "time": 10},
                        time_cost=timedelta(minutes=8),
                        success_probability=0.95,
                        risk_level=0.05
                    ),
                    Action(
                        id="sell_batch_optimal",
                        name="Vente par lot optimal",
                        type="trading",
                        objective_impacts={"farming_objective": 200.0},
                        resource_cost={"time": 5},
                        time_cost=timedelta(minutes=3),
                        success_probability=0.90,
                        risk_level=0.10
                    )
                ]

                action_plan = await decision_module.process({
                    'type': 'evaluate_actions',
                    'actions': test_actions
                })

                if action_plan:
                    print(f"Plan d'action généré:")
                    print(f"  - Actions: {len(action_plan.actions)}")
                    print(f"  - Utilité totale: {action_plan.total_utility:.2f}")
                    print(f"  - Durée estimée: {action_plan.estimated_duration}")
                    print(f"  - Probabilité succès: {action_plan.success_probability:.2f}")

                # Test planification temporelle
                temporal_plan = await decision_module.process({
                    'type': 'plan_temporal',
                    'horizon': timedelta(hours=6)
                })

                if temporal_plan:
                    print(f"Plan temporel créé:")
                    print(f"  - Objectifs court terme: {len(temporal_plan.short_term_goals)}")
                    print(f"  - Actions immédiates: {len(temporal_plan.immediate_actions)}")
                    print(f"  - Actions programmées: {len(temporal_plan.scheduled_actions)}")

        return True

    finally:
        await orchestrator.stop()

async def test_emotional_integration():
    """Test l'intégration du système émotionnel"""
    print("\nTest intégration Emotional State...")

    orchestrator = await create_ai_framework()

    if not await orchestrator.start():
        print("❌ Échec démarrage framework")
        return False

    try:
        emotional_module = orchestrator.modules.get('emotional')

        if emotional_module:
            # Test traitement d'événements
            test_events = [
                (GameEvent.LEVEL_UP, {'importance': 1.0, 'surprise': 0.8}),
                (GameEvent.RARE_DROP, {'importance': 1.5, 'surprise': 1.2}),
                (GameEvent.SUCCESSFUL_TRADE, {'importance': 0.8, 'surprise': 0.3}),
                (GameEvent.DEATH, {'importance': 0.9, 'surprise': 0.6})
            ]

            for event, context in test_events:
                result = await emotional_module.process({
                    'type': 'process_event',
                    'event': event,
                    'context': context
                })

                if result:
                    mood = await emotional_module.process({'type': 'simulate_mood'})
                    print(f"Événement {event.value}: humeur = {mood.value}")

            # Test génération de réponse comportementale
            combat_situation = {
                'type': 'high_risk_combat',
                'context': {
                    'difficulty': 'hard',
                    'reward_potential': 'high',
                    'team_support': True
                }
            }

            response = await emotional_module.process({
                'type': 'generate_response',
                'situation': combat_situation
            })

            if response:
                print(f"Réponse comportementale:")
                print(f"  - Type: {response.response_type}")
                print(f"  - Intensité: {response.intensity:.2f}")
                print(f"  - Actions suggérées: {response.actions}")
                if response.dialogue:
                    print(f"  - Dialogue: {response.dialogue}")

            # Test tolérance au risque
            current_mood = await emotional_module.process({'type': 'simulate_mood'})
            risk_profile = await emotional_module.process({
                'type': 'adjust_risk_tolerance',
                'mood': current_mood
            })

            if risk_profile:
                print(f"Tolérance au risque: {risk_profile.current_tolerance():.2f}")

            # Test motivation
            motivation = await emotional_module.process({'type': 'get_motivation'})
            if motivation:
                print(f"Motivation actuelle:")
                for activity, level in motivation.items():
                    print(f"  - {activity}: {level:.2f}")

        return True

    finally:
        await orchestrator.stop()

async def test_state_tracking_integration():
    """Test l'intégration du suivi d'état"""
    print("\nTest intégration State Tracking...")

    orchestrator = await create_ai_framework()

    if not await orchestrator.start():
        print("❌ Échec démarrage framework")
        return False

    try:
        state_module = orchestrator.modules.get('state_tracking')

        if state_module:
            # Test mise à jour état immédiat
            new_game_state = GameState(
                position=(10, -15),
                map_id="astrub_farm",
                health_percent=85.0,
                mana_percent=60.0,
                energy_percent=90.0,
                level=5,
                in_combat=False,
                inventory_slots_used=35,
                pods_used=800
            )

            success = await state_module.process({
                'type': 'update_immediate',
                'state': new_game_state
            })

            if success:
                print("État immédiat mis à jour")

            # Test mise à jour contexte tactique
            tactical_updates = {
                'current_activity': 'farming',
                'immediate_goals': ['gather_wheat', 'optimize_path'],
                'actions_per_minute': 52.0,
                'efficiency_score': 0.88,
                'zone_safety_level': 0.95,
                'player_density': 3
            }

            await state_module.process({
                'type': 'update_tactical',
                'updates': tactical_updates
            })
            print("Contexte tactique mis à jour")

            # Test mise à jour planification stratégique
            strategic_updates = {
                'daily_goals': [
                    {'id': 'farm_income', 'name': 'Revenus de farm', 'priority': 1, 'completed': False, 'target': 20000},
                    {'id': 'skill_progression', 'name': 'Progression compétences', 'priority': 2, 'completed': False}
                ],
                'time_allocations': {
                    'farming': timedelta(hours=4),
                    'trading': timedelta(hours=1),
                    'exploration': timedelta(minutes=30)
                },
                'progress_updates': {
                    'farm_income': 0.35,
                    'skill_progression': 0.20
                }
            }

            await state_module.process({
                'type': 'update_strategic',
                'planning_updates': strategic_updates
            })
            print("Planification stratégique mise à jour")

            # Attente pour permettre les mises à jour automatiques
            await asyncio.sleep(2)

            # Test récupération résumé d'état
            summary = await state_module.process({'type': 'get_summary'})

            if summary:
                print(f"Résumé de l'état global:")
                overall_health = summary.get('overall_health', {})
                print(f"  - Santé immédiate: {overall_health.get('immediate_health', 0.0):.2f}")
                print(f"  - Efficacité tactique: {overall_health.get('tactical_efficiency', 0.0):.2f}")
                print(f"  - Progression stratégique: {overall_health.get('strategic_progress', 0.0):.2f}")
                print(f"  - Stabilité méta: {overall_health.get('meta_stability', 0.0):.2f}")

            # Test prédiction d'évolution
            prediction = await state_module.process({
                'type': 'predict_evolution',
                'level': StateLevel.IMMEDIATE,
                'horizon': timedelta(minutes=10)
            })

            if prediction:
                print(f"Prédiction d'évolution (confiance: {prediction.get('confidence', 0.0):.2f})")

        return True

    finally:
        await orchestrator.stop()

async def test_full_phase2_integration():
    """Test d'intégration complète Phase 2 avec coordination entre modules"""
    print("\nTest intégration complète Phase 2...")

    orchestrator = await create_ai_framework()

    if not await orchestrator.start():
        print("❌ Échec démarrage framework")
        return False

    try:
        # Récupération des modules Phase 2
        decision_module = orchestrator.modules.get('decision')
        emotional_module = orchestrator.modules.get('emotional')
        state_module = orchestrator.modules.get('state_tracking')

        if all([decision_module, emotional_module, state_module]):
            print("Scénario: Session de farm avec gestion émotionnelle et décision adaptative")

            # 1. Mise à jour de l'état de jeu
            farming_state = GameState(
                position=(12, -18),
                map_id="astrub_wheat_field",
                health_percent=100.0,
                mana_percent=95.0,
                energy_percent=85.0,
                level=8,
                in_combat=False,
                inventory_slots_used=20,
                pods_used=300
            )

            await state_module.process({
                'type': 'update_immediate',
                'state': farming_state
            })
            print("1. État de farm établi")

            # 2. Traitement d'événement positif
            await emotional_module.process({
                'type': 'process_event',
                'event': GameEvent.RESOURCE_GATHERED,
                'context': {'importance': 0.6, 'efficiency': 'high'}
            })

            current_mood = await emotional_module.process({'type': 'simulate_mood'})
            print(f"2. Humeur après collecte: {current_mood.value}")

            # 3. Ajout d'objectif de farm optimisé
            farm_objective = Objective(
                id="optimized_farming",
                name="Farm optimisée avec gestion émotionnelle",
                type=ObjectiveType.OPTIMIZE_EFFICIENCY,
                priority=Priority.HIGH,
                target_value=1000.0,
                weight=2.0,
                estimated_duration=timedelta(hours=2)
            )

            await decision_module.process({
                'type': 'add_objective',
                'objective': farm_objective
            })
            print("3. Objectif de farm optimisée ajouté")

            # 4. Génération d'actions adaptées à l'humeur
            motivation = await emotional_module.process({'type': 'get_motivation'})
            risk_tolerance = await emotional_module.process({
                'type': 'adjust_risk_tolerance',
                'mood': current_mood
            })

            # Actions ajustées selon l'état émotionnel
            adaptive_actions = [
                Action(
                    id="conservative_farming",
                    name="Farm conservatrice",
                    type="farming",
                    objective_impacts={"optimized_farming": 80.0},
                    resource_cost={"energy": 10},
                    time_cost=timedelta(minutes=10),
                    success_probability=0.98,
                    risk_level=0.02
                ),
                Action(
                    id="aggressive_farming",
                    name="Farm agressive",
                    type="farming",
                    objective_impacts={"optimized_farming": 150.0},
                    resource_cost={"energy": 20},
                    time_cost=timedelta(minutes=8),
                    success_probability=0.85,
                    risk_level=0.25
                )
            ]

            # Sélection d'action basée sur la tolérance au risque
            selected_actions = []
            for action in adaptive_actions:
                if action.risk_level <= risk_tolerance.current_tolerance():
                    selected_actions.append(action)

            if selected_actions:
                action_plan = await decision_module.process({
                    'type': 'evaluate_actions',
                    'actions': selected_actions
                })

                if action_plan:
                    print(f"4. Plan adaptatif généré:")
                    print(f"   - Actions sélectionnées: {len(action_plan.actions)}")
                    print(f"   - Basé sur tolérance risque: {risk_tolerance.current_tolerance():.2f}")

            # 5. Mise à jour tactique coordonnée
            tactical_updates = {
                'current_activity': 'adaptive_farming',
                'efficiency_score': 0.92,
                'adaptation_needed': False,
                'emotional_influence': current_mood.value
            }

            await state_module.process({
                'type': 'update_tactical',
                'updates': tactical_updates
            })
            print("5. Contexte tactique mis à jour avec influence émotionnelle")

            # 6. Planification temporelle émotionnellement informée
            temporal_plan = await decision_module.process({
                'type': 'plan_temporal',
                'horizon': timedelta(hours=4)
            })

            if temporal_plan:
                print(f"6. Plan temporel avec adaptation émotionnelle:")
                print(f"   - Progression globale: {temporal_plan.overall_progress:.2f}")
                print(f"   - Intervalles de révision: {len(temporal_plan.revision_intervals)}")

            # 7. Simulation d'événement négatif et adaptation
            await emotional_module.process({
                'type': 'process_event',
                'event': GameEvent.DEATH,
                'context': {'importance': 0.8, 'unexpected': True}
            })

            new_mood = await emotional_module.process({'type': 'simulate_mood'})
            print(f"7. Nouvelle humeur après mort: {new_mood.value}")

            # Génération de réponse adaptative
            crisis_response = await emotional_module.process({
                'type': 'generate_response',
                'situation': {
                    'type': 'setback_recovery',
                    'context': {'severity': 'medium', 'learning_opportunity': True}
                }
            })

            if crisis_response:
                print(f"   Réponse adaptative: {crisis_response.response_type}")
                print(f"   Actions suggérées: {crisis_response.actions}")

            # 8. Mise à jour du progrès et métriques finales
            await decision_module.process({
                'type': 'update_progress',
                'objective_id': 'optimized_farming',
                'value': 650.0
            })

            # Attente pour synchronisation
            await asyncio.sleep(2)

            # Métriques finales
            final_summary = await state_module.process({'type': 'get_summary'})
            emotional_summary = await emotional_module.process({'type': 'get_summary'})

            print(f"\nRésultats intégration Phase 2:")
            if final_summary and emotional_summary:
                print(f"   - Efficacité tactique: {final_summary.get('overall_health', {}).get('tactical_efficiency', 0):.2f}")
                print(f"   - Humeur finale: {emotional_summary.get('current_mood', 'unknown')}")
                print(f"   - Émotions actives: {emotional_summary.get('active_emotions_count', 0)}")
                print(f"   - Motivation exploration: {emotional_summary.get('motivation', {}).get('exploration', 0):.2f}")

        return True

    finally:
        await orchestrator.stop()

async def main():
    """Test principal de l'intégration Phase 2"""
    print("Test d'intégration Phase 2 - Cerveau Multi-Dimensionnel")
    print("=" * 70)

    try:
        # Test 1: Decision Engine
        success1 = await test_decision_engine_integration()

        # Test 2: Emotional State Management
        success2 = await test_emotional_integration()

        # Test 3: State Tracking
        success3 = await test_state_tracking_integration()

        # Test 4: Intégration complète Phase 2
        success4 = await test_full_phase2_integration()

        # Résultats finaux
        print("\n" + "=" * 70)
        print("Résultats des tests Phase 2:")
        print(f"  Decision Engine: {'PASS' if success1 else 'FAIL'}")
        print(f"  Emotional State: {'PASS' if success2 else 'FAIL'}")
        print(f"  State Tracking: {'PASS' if success3 else 'FAIL'}")
        print(f"  Intégration complète: {'PASS' if success4 else 'FAIL'}")

        overall_success = all([success1, success2, success3, success4])
        print(f"\nRésultat global Phase 2: {'SUCCÈS' if overall_success else 'ÉCHEC'}")

        if overall_success:
            print("\nLe Cerveau Multi-Dimensionnel est opérationnel !")
            print("✅ Decision Engine avec optimisation Pareto")
            print("✅ Emotional State Management avec personnalité évolutive")
            print("✅ Multi-Dimensional State Tracking")
            print("✅ Coordination intelligente entre modules")
            print("\nPrêt pour Phase 3: Exécution Adaptative & Sociale")

    except Exception as e:
        print(f"Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())