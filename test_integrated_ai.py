"""
Test d'intégration complète de l'IA DOFUS autonome
Test de l'AI Framework avec Knowledge Graph et Predictive Engine intégrés
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import des modules
from core.ai_framework import create_ai_framework, AITask, Priority
from core.knowledge_graph import EntityType, RelationType
from core.predictive_engine import PredictionRequest, PredictionType, TimeWindow

async def test_knowledge_integration():
    """Test l'intégration du Knowledge Graph"""
    print("Test intégration Knowledge Graph...")

    # Création et démarrage du framework
    orchestrator = await create_ai_framework()

    if not await orchestrator.start():
        print("❌ Échec démarrage framework")
        return False

    try:
        # Test recherche d'entités
        knowledge_task = AITask(
            name="find_wheat_entities",
            priority=Priority.MEDIUM,
            function=lambda: None,  # Placeholder - sera remplacé par le module
            context={
                'module': 'knowledge',
                'query': {
                    'type': 'find_entity',
                    'name': 'blé'
                }
            }
        )

        # Récupération directe du module knowledge
        knowledge_module = orchestrator.modules.get('knowledge')
        if knowledge_module:
            # Test recherche d'entités
            wheat_entities = await knowledge_module.process({
                'type': 'find_entity',
                'name': 'blé'
            })

            if wheat_entities:
                print(f"Entités 'blé' trouvées: {len(wheat_entities)}")

                wheat = wheat_entities[0]
                print(f"  - {wheat.name} ({wheat.entity_type.value})")

                # Test recherche de relations
                relations = await knowledge_module.process({
                    'type': 'get_relations',
                    'entity_id': wheat.id
                })

                if relations:
                    print(f"  - Relations trouvées: {len(relations)}")
                    for relation in relations[:3]:  # Première 3
                        target_entity = knowledge_module.knowledge_graph.get_entity(relation.target_id)
                        if target_entity:
                            print(f"    → {relation.relation_type.value} → {target_entity.name}")

                # Test recherche de stratégie optimale
                strategy = await knowledge_module.process({
                    'type': 'find_strategy',
                    'goal': wheat.id,
                    'resources': []
                })

                if strategy:
                    print(f"  - Stratégie optimale: {strategy.get('steps', [])}")
            else:
                print("⚠️ Aucune entité 'blé' trouvée")

        # Statistiques du Knowledge Graph
        shared_data = await knowledge_module.get_shared_data()
        print(f"Knowledge Graph stats: {shared_data.get('entities_count', 0)} entités")

        return True

    finally:
        await orchestrator.stop()

async def test_prediction_integration():
    """Test l'intégration du Predictive Engine"""
    print("\nTest intégration Predictive Engine...")

    orchestrator = await create_ai_framework()

    if not await orchestrator.start():
        print("❌ Échec démarrage framework")
        return False

    try:
        prediction_module = orchestrator.modules.get('prediction')
        knowledge_module = orchestrator.modules.get('knowledge')

        if prediction_module and knowledge_module:
            # Récupération d'une entité pour test
            wheat_entities = await knowledge_module.process({
                'type': 'find_entity',
                'name': 'blé'
            })

            if wheat_entities:
                wheat = wheat_entities[0]

                # Test prédiction de marché
                market_prediction = await prediction_module.process({
                    'prediction_type': 'market_trend',
                    'target_entity_id': wheat.id,
                    'time_window': TimeWindow.MEDIUM_TERM,
                    'context': {}
                })

                if market_prediction:
                    print(f"Prédiction marché pour {wheat.name}:")
                    print(f"  - Prix prédit: {market_prediction.predicted_value:.2f}")
                    print(f"  - Confiance: {market_prediction.confidence:.2f}")
                    print(f"  - Valide jusqu'à: {market_prediction.valid_until.strftime('%H:%M:%S')}")

                # Test prédiction d'événement
                event_prediction = await prediction_module.process({
                    'prediction_type': 'server_event',
                    'target_entity_id': 'archmonster_spawn',
                    'time_window': TimeWindow.LONG_TERM,
                    'context': {}
                })

                if event_prediction:
                    print(f"Prédiction événement:")
                    print(f"  - Type: {event_prediction.target_name}")
                    print(f"  - Confiance: {event_prediction.confidence:.2f}")

                # Test timing optimal
                timing_prediction = await prediction_module.process({
                    'prediction_type': 'optimal_timing',
                    'target_entity_id': 'farming',
                    'time_window': TimeWindow.SHORT_TERM,
                    'context': {'activity_type': 'farming'}
                })

                if timing_prediction:
                    timing_data = timing_prediction.predicted_value
                    print(f"Timing optimal farming:")
                    print(f"  - Début optimal: {timing_data['optimal_start'].strftime('%H:%M')}")
                    print(f"  - Score efficacité: {timing_data['efficiency_score']:.2f}")

        # Statistiques du Predictive Engine
        shared_data = await prediction_module.get_shared_data()
        stats = shared_data.get('prediction_stats', {})
        print(f"Prediction stats: {stats.get('total_predictions', 0)} prédictions")

        return True

    finally:
        await orchestrator.stop()

async def test_uncertainty_integration():
    """Test l'intégration de l'Uncertainty Manager"""
    print("\nTest intégration Uncertainty Manager...")

    orchestrator = await create_ai_framework()

    if not await orchestrator.start():
        print("❌ Échec démarrage framework")
        return False

    try:
        uncertainty_module = orchestrator.modules.get('uncertainty')

        if uncertainty_module:
            # Test évaluation d'une décision
            decision_data = {
                'action_type': 'farming',
                'target_zone': 'astrub',
                'risk_factors': ['competition', 'spawn_rate'],
                'expected_reward': 100,
                'time_investment': 30
            }

            uncertainty_result = await uncertainty_module.process({
                'type': 'evaluate_decision',
                'decision_data': decision_data
            })

            if uncertainty_result:
                print(f"Évaluation incertitude:")
                print(f"  - Confiance: {uncertainty_result.confidence_score:.2f}")
                print(f"  - Niveau risque: {uncertainty_result.risk_level.name}")
                print(f"  - Niveau confiance: {uncertainty_result.confidence_level.name}")

            # Test création checkpoint
            checkpoint_result = await uncertainty_module.process({
                'type': 'create_checkpoint',
                'decision_id': 'test_farming_001',
                'state_snapshot': {
                    'position': [4, -13],
                    'inventory_slots': 45,
                    'kamas': 15000
                }
            })

            if checkpoint_result:
                print(f"Checkpoint créé: {checkpoint_result}")

        return True

    finally:
        await orchestrator.stop()

async def test_full_integration():
    """Test d'intégration complète avec coordination entre modules"""
    print("\nTest intégration complète...")

    orchestrator = await create_ai_framework()

    if not await orchestrator.start():
        print("❌ Échec démarrage framework")
        return False

    try:
        # Attente pour permettre la synchronisation des modules
        await asyncio.sleep(2)

        # Simulation d'un scénario complet : optimisation d'une session de farm
        print("Scénario: Optimisation session de farm...")

        knowledge_module = orchestrator.modules.get('knowledge')
        prediction_module = orchestrator.modules.get('prediction')
        uncertainty_module = orchestrator.modules.get('uncertainty')

        if all([knowledge_module, prediction_module, uncertainty_module]):
            # 1. Recherche de ressources disponibles
            wheat_entities = await knowledge_module.process({
                'type': 'find_entity',
                'name': 'blé'
            })

            if wheat_entities:
                wheat = wheat_entities[0]
                print(f"1. Ressource sélectionnée: {wheat.name}")

                # 2. Prédiction de marché pour la ressource
                market_forecast = await prediction_module.process({
                    'prediction_type': 'market_trend',
                    'target_entity_id': wheat.id,
                    'time_window': TimeWindow.MEDIUM_TERM
                })

                if market_forecast:
                    print(f"2. Prédiction marché: prix {market_forecast.predicted_value:.2f} (confiance: {market_forecast.confidence:.2f})")

                # 3. Recherche timing optimal
                optimal_timing = await prediction_module.process({
                    'prediction_type': 'optimal_timing',
                    'target_entity_id': 'farming',
                    'time_window': TimeWindow.SHORT_TERM,
                    'context': {'activity_type': 'farming'}
                })

                if optimal_timing:
                    timing_data = optimal_timing.predicted_value
                    print(f"3. Timing optimal: {timing_data['optimal_start'].strftime('%H:%M')} (efficacité: {timing_data['efficiency_score']:.2f})")

                # 4. Évaluation incertitude de la décision globale
                decision_data = {
                    'action_type': 'farming',
                    'target_resource': wheat.name,
                    'predicted_profit': market_forecast.predicted_value if market_forecast else 100,
                    'timing_efficiency': timing_data['efficiency_score'] if optimal_timing else 0.5,
                    'confidence_factors': {
                        'market_confidence': market_forecast.confidence if market_forecast else 0.5,
                        'timing_confidence': optimal_timing.confidence if optimal_timing else 0.5
                    }
                }

                uncertainty_eval = await uncertainty_module.process({
                    'type': 'evaluate_decision',
                    'decision_data': decision_data
                })

                if uncertainty_eval:
                    print(f"4. Évaluation finale: confiance {uncertainty_eval.confidence_score:.2f}, niveau {uncertainty_eval.confidence_level.name}")

                    # 5. Si décision positive, création d'un checkpoint
                    if uncertainty_eval.confidence_score > 0.6:
                        checkpoint = await uncertainty_module.process({
                            'type': 'create_checkpoint',
                            'decision_id': f'farming_{wheat.id}_{int(datetime.now().timestamp())}',
                            'state_snapshot': {
                                'strategy': 'farming',
                                'target': wheat.name,
                                'expected_profit': decision_data['predicted_profit'],
                                'start_time': datetime.now().isoformat()
                            }
                        })

                        if checkpoint:
                            print(f"5. Checkpoint stratégique créé")

        # Affichage du status global
        status = orchestrator.get_status()
        print(f"\nStatus final framework:")
        print(f"  - Modules actifs: {status['modules_healthy']}/{status['modules_count']}")
        print(f"  - Tâches en queue: {status['task_queue_size']}")

        # Statistiques des modules
        for module_name, module in orchestrator.modules.items():
            shared_data = await module.get_shared_data()
            print(f"  - {module_name}: {len(shared_data)} métriques partagées")

        return True

    finally:
        await orchestrator.stop()

async def main():
    """Test principal de l'intégration IA"""
    print("Test d'intégration complète IA DOFUS autonome")
    print("=" * 60)

    try:
        # Test 1: Knowledge Graph
        success1 = await test_knowledge_integration()

        # Test 2: Predictive Engine
        success2 = await test_prediction_integration()

        # Test 3: Uncertainty Manager
        success3 = await test_uncertainty_integration()

        # Test 4: Intégration complète
        success4 = await test_full_integration()

        # Résultats finaux
        print("\n" + "=" * 60)
        print("Résultats des tests:")
        print(f"  Knowledge Graph: {'PASS' if success1 else 'FAIL'}")
        print(f"  Predictive Engine: {'PASS' if success2 else 'FAIL'}")
        print(f"  Uncertainty Manager: {'PASS' if success3 else 'FAIL'}")
        print(f"  Intégration complète: {'PASS' if success4 else 'FAIL'}")

        overall_success = all([success1, success2, success3, success4])
        print(f"\nRésultat global: {'SUCCÈS' if overall_success else 'ÉCHEC'}")

        if overall_success:
            print("\nL'IA DOFUS autonome est prête pour la Phase 1+ !")
            print("   Prochaines étapes: Développement Phase 2 (Decision Engine avancé)")

    except Exception as e:
        print(f"Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())