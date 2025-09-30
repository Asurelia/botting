"""
DOFUS Knowledge Base Integration - Systeme Unifie de Connaissance
Interface centrale pour toutes les bases de donnees DOFUS Unity
Approche 100% vision - Coordination intelligente des modules
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .spells_database import get_spells_database, DofusClass
from .monsters_database import get_monsters_database
from .maps_database import get_maps_database
from .economy_tracker import get_economy_tracker
from .dofus_data_extractor import get_dofus_extractor

# Import du connecteur Ganymede
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from external_integration.ganymede_guides_connector import get_ganymede_connector
    GANYMEDE_AVAILABLE = True
except ImportError:
    print("⚠️ Connecteur Ganymede non disponible")
    GANYMEDE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeQueryResult:
    """Resultat d'une requete sur la base de connaissance"""
    query_type: str
    success: bool
    data: Any
    source_modules: List[str]
    confidence_score: float
    execution_time_ms: float
    suggestions: List[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []

@dataclass
class GameContext:
    """Contexte de jeu actuel pour requetes contextuelles"""
    player_class: Optional[DofusClass] = None
    player_level: int = 200
    current_server: str = "Julith"
    current_map_id: Optional[int] = None
    available_ap: int = 6
    available_mp: int = 3
    distance_to_target: int = 2
    in_combat: bool = False

class DofusKnowledgeBase:
    """
    Systeme unifie de connaissance DOFUS Unity
    Interface centrale pour toutes les bases de donnees
    """

    def __init__(self):
        # Modules de base de donnees
        self.spells_db = get_spells_database()
        self.monsters_db = get_monsters_database()
        self.maps_db = get_maps_database()
        self.economy_tracker = get_economy_tracker()
        self.data_extractor = get_dofus_extractor()

        # Connecteur Ganymede
        if GANYMEDE_AVAILABLE:
            try:
                self.ganymede_connector = get_ganymede_connector()
                logger.info("Connecteur Ganymede initialisé")
            except Exception as e:
                logger.warning(f"Erreur initialisation Ganymede: {e}")
                self.ganymede_connector = None
        else:
            self.ganymede_connector = None

        # Cache et contexte
        self.game_context = GameContext()
        self.query_cache: Dict[str, KnowledgeQueryResult] = {}
        self.cache_ttl = 300  # 5 minutes

        # Statistiques
        self.query_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "modules_usage": {
                "spells": 0, "monsters": 0, "maps": 0, "economy": 0, "ganymede": 0
            }
        }

        # Thread de mise a jour automatique
        self.auto_update_thread = None
        self.auto_update_running = False

        logger.info("DofusKnowledgeBase initialise")

    def update_game_context(self, context: GameContext):
        """Met a jour le contexte de jeu"""
        self.game_context = context
        # Invalide le cache contextuel
        self._clear_contextual_cache()
        logger.info(f"Contexte mis a jour: Classe {context.player_class}, Niveau {context.player_level}")

    def _clear_contextual_cache(self):
        """Efface le cache contextuel"""
        contextual_keys = [k for k in self.query_cache.keys() if "contextual" in k]
        for key in contextual_keys:
            del self.query_cache[key]

    def _get_cache_key(self, query_type: str, **params) -> str:
        """Genere une cle de cache pour une requete"""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{query_type}_{param_str}"

    def _is_cache_valid(self, result: KnowledgeQueryResult) -> bool:
        """Verifie si le cache est encore valide"""
        # Pour l'instant, cache simple sans expiration
        return True

    def query_optimal_spells(self, **kwargs) -> KnowledgeQueryResult:
        """Requete sorts optimaux selon contexte"""
        start_time = time.time()
        cache_key = self._get_cache_key("optimal_spells", **kwargs)

        # Verification cache
        if cache_key in self.query_cache and self._is_cache_valid(self.query_cache[cache_key]):
            self.query_stats["cache_hits"] += 1
            return self.query_cache[cache_key]

        # Requete avec contexte
        try:
            optimal_spells = self.spells_db.find_optimal_spells(
                ap_available=kwargs.get('ap_available', self.game_context.available_ap),
                distance=kwargs.get('distance', self.game_context.distance_to_target),
                class_type=kwargs.get('class_type', self.game_context.player_class)
            )

            execution_time = (time.time() - start_time) * 1000

            result = KnowledgeQueryResult(
                query_type="optimal_spells",
                success=True,
                data=optimal_spells,
                source_modules=["spells"],
                confidence_score=0.9,
                execution_time_ms=execution_time,
                suggestions=[f"Sort recommande: {spell.name}" for spell in optimal_spells[:3]]
            )

            # Sauvegarde cache
            self.query_cache[cache_key] = result
            self.query_stats["modules_usage"]["spells"] += 1

            return result

        except Exception as e:
            logger.error(f"Erreur requete sorts optimaux: {e}")
            return KnowledgeQueryResult(
                query_type="optimal_spells",
                success=False,
                data=None,
                source_modules=["spells"],
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def query_monster_strategy(self, monster_name: str) -> KnowledgeQueryResult:
        """Requete strategie contre un monstre"""
        start_time = time.time()
        cache_key = self._get_cache_key("monster_strategy", monster=monster_name)

        if cache_key in self.query_cache and self._is_cache_valid(self.query_cache[cache_key]):
            self.query_stats["cache_hits"] += 1
            return self.query_cache[cache_key]

        try:
            monster = self.monsters_db.get_monster_by_name(monster_name)
            if not monster:
                return KnowledgeQueryResult(
                    query_type="monster_strategy",
                    success=False,
                    data=None,
                    source_modules=["monsters"],
                    confidence_score=0.0,
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            strategy = self.monsters_db.get_counter_strategy(monster)

            # Enrichissement avec sorts efficaces
            if self.game_context.player_class:
                effective_spells = []
                for element in strategy["preferred_elements"]:
                    element_spells = self.spells_db.get_spells_by_element(element)
                    class_spells = [s for s in element_spells if s.class_type == self.game_context.player_class]
                    effective_spells.extend(class_spells[:2])

                strategy["recommended_spells"] = [s.name for s in effective_spells]

            execution_time = (time.time() - start_time) * 1000

            result = KnowledgeQueryResult(
                query_type="monster_strategy",
                success=True,
                data=strategy,
                source_modules=["monsters", "spells"],
                confidence_score=0.85,
                execution_time_ms=execution_time,
                suggestions=[
                    f"Distance optimale: {strategy['optimal_distance']}",
                    f"Elements efficaces: {', '.join(strategy['preferred_elements'])}"
                ]
            )

            self.query_cache[cache_key] = result
            self.query_stats["modules_usage"]["monsters"] += 1

            return result

        except Exception as e:
            logger.error(f"Erreur requete strategie monstre: {e}")
            return KnowledgeQueryResult(
                query_type="monster_strategy",
                success=False,
                data=None,
                source_modules=["monsters"],
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def query_farming_route(self, resource_name: str) -> KnowledgeQueryResult:
        """Requete route de farming optimale"""
        start_time = time.time()
        cache_key = self._get_cache_key("farming_route", resource=resource_name)

        if cache_key in self.query_cache and self._is_cache_valid(self.query_cache[cache_key]):
            self.query_stats["cache_hits"] += 1
            return self.query_cache[cache_key]

        try:
            farming_route = self.maps_db.get_optimal_farming_route(
                resource_name=resource_name,
                player_level=self.game_context.player_level
            )

            execution_time = (time.time() - start_time) * 1000

            result = KnowledgeQueryResult(
                query_type="farming_route",
                success=len(farming_route) > 0,
                data=farming_route,
                source_modules=["maps"],
                confidence_score=0.8,
                execution_time_ms=execution_time,
                suggestions=[f"Carte recommandee: {map_obj.name}" for map_obj in farming_route[:3]]
            )

            self.query_cache[cache_key] = result
            self.query_stats["modules_usage"]["maps"] += 1

            return result

        except Exception as e:
            logger.error(f"Erreur requete route farming: {e}")
            return KnowledgeQueryResult(
                query_type="farming_route",
                success=False,
                data=None,
                source_modules=["maps"],
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def query_market_opportunities(self) -> KnowledgeQueryResult:
        """Requete opportunites de marche"""
        start_time = time.time()
        cache_key = self._get_cache_key("market_opportunities", server=self.game_context.current_server)

        if cache_key in self.query_cache and self._is_cache_valid(self.query_cache[cache_key]):
            self.query_stats["cache_hits"] += 1
            return self.query_cache[cache_key]

        try:
            opportunities = self.economy_tracker.detect_market_opportunities(
                self.game_context.current_server
            )

            execution_time = (time.time() - start_time) * 1000

            result = KnowledgeQueryResult(
                query_type="market_opportunities",
                success=True,
                data=opportunities,
                source_modules=["economy"],
                confidence_score=0.75,
                execution_time_ms=execution_time,
                suggestions=[f"Opportunite: {opp.action} {opp.item_name}" for opp in opportunities[:3]]
            )

            self.query_cache[cache_key] = result
            self.query_stats["modules_usage"]["economy"] += 1

            return result

        except Exception as e:
            logger.error(f"Erreur requete opportunites marche: {e}")
            return KnowledgeQueryResult(
                query_type="market_opportunities",
                success=False,
                data=None,
                source_modules=["economy"],
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def query_comprehensive_advice(self, situation: str) -> KnowledgeQueryResult:
        """Requete conseil complet selon situation"""
        start_time = time.time()

        advice = {
            "situation": situation,
            "recommendations": [],
            "tactical_advice": [],
            "economic_advice": [],
            "strategic_advice": []
        }

        modules_used = []

        try:
            if "combat" in situation.lower():
                # Conseils de combat
                if self.game_context.player_class:
                    optimal_spells = self.query_optimal_spells()
                    if optimal_spells.success:
                        advice["tactical_advice"].extend(optimal_spells.suggestions)
                        modules_used.append("spells")

            if "farming" in situation.lower():
                # Conseils de farming
                overview = self.economy_tracker.get_server_economy_overview(
                    self.game_context.current_server
                )
                advice["economic_advice"].append(f"Economie serveur: {overview['economy_health']}")
                modules_used.append("economy")

            if "exploration" in situation.lower():
                # Conseils d'exploration
                maps = self.maps_db.get_maps_by_level_range(
                    self.game_context.player_level - 10,
                    self.game_context.player_level + 10
                )[:5]
                advice["strategic_advice"].extend([f"Zone recommandee: {m.name}" for m in maps])
                modules_used.append("maps")

            # Consolidation recommendations
            all_advice = (advice["tactical_advice"] + advice["economic_advice"] + advice["strategic_advice"])
            advice["recommendations"] = all_advice[:5]

            execution_time = (time.time() - start_time) * 1000

            result = KnowledgeQueryResult(
                query_type="comprehensive_advice",
                success=len(advice["recommendations"]) > 0,
                data=advice,
                source_modules=modules_used,
                confidence_score=0.7,
                execution_time_ms=execution_time,
                suggestions=advice["recommendations"]
            )

            return result

        except Exception as e:
            logger.error(f"Erreur requete conseil complet: {e}")
            return KnowledgeQueryResult(
                query_type="comprehensive_advice",
                success=False,
                data=None,
                source_modules=modules_used,
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def query_ganymede_guide(self, topic: str) -> KnowledgeQueryResult:
        """Requête guide Ganymede pour un sujet donné"""
        start_time = time.time()
        cache_key = self._get_cache_key("ganymede_guide", topic=topic.lower())

        if cache_key in self.query_cache and self._is_cache_valid(self.query_cache[cache_key]):
            self.query_stats["cache_hits"] += 1
            return self.query_cache[cache_key]

        try:
            if not self.ganymede_connector:
                return KnowledgeQueryResult(
                    query_type="ganymede_guide",
                    success=False,
                    data=None,
                    source_modules=["ganymede"],
                    confidence_score=0.0,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    suggestions=["Connecteur Ganymede non disponible"]
                )

            # Rechercher guide par sujet
            guide = self.ganymede_connector.get_guide_by_topic(topic)

            if guide:
                # Enrichir le guide avec le contexte actuel
                enriched_guide = self._enrich_guide_with_context(guide)

                result = KnowledgeQueryResult(
                    query_type="ganymede_guide",
                    success=True,
                    data=enriched_guide,
                    source_modules=["ganymede"],
                    confidence_score=0.9,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    suggestions=[
                        f"Guide: {guide['title']}",
                        f"Difficulté: {guide.get('difficulty_level', 'moyen')}",
                        f"Étapes: {len(guide.get('steps', []))}"
                    ]
                )
            else:
                # Rechercher guides similaires
                similar_guides = self.ganymede_connector.search_guides(topic)

                result = KnowledgeQueryResult(
                    query_type="ganymede_guide",
                    success=len(similar_guides) > 0,
                    data={"similar_guides": similar_guides[:5]},
                    source_modules=["ganymede"],
                    confidence_score=0.6,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    suggestions=[f"Guide similaire: {g['title']}" for g in similar_guides[:3]]
                )

            self.query_cache[cache_key] = result
            self.query_stats["modules_usage"]["ganymede"] += 1

            return result

        except Exception as e:
            logger.error(f"Erreur requête guide Ganymede: {e}")
            return KnowledgeQueryResult(
                query_type="ganymede_guide",
                success=False,
                data=None,
                source_modules=["ganymede"],
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                suggestions=[f"Erreur: {str(e)}"]
            )

    def _enrich_guide_with_context(self, guide: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit un guide Ganymede avec le contexte de jeu actuel"""
        enriched = guide.copy()

        try:
            # Adapter selon la classe du joueur
            if self.game_context.player_class:
                class_name = self.game_context.player_class.value.lower()

                # Ajouter sorts recommandés si c'est un guide de combat
                if any(keyword in guide['title'].lower() for keyword in ['combat', 'pvp', 'donjon']):
                    optimal_spells = self.query_optimal_spells()
                    if optimal_spells.success:
                        enriched['recommended_spells_for_class'] = [
                            spell['name'] for spell in optimal_spells.data.get('spells', [])[:3]
                        ]

                # Adapter niveau de difficulté
                player_level = self.game_context.player_level
                if player_level < 100:
                    enriched['level_adaptation'] = "Guide adapté pour niveau débutant"
                elif player_level > 180:
                    enriched['level_adaptation'] = "Guide adapté pour niveau expert"

            # Ajouter informations économiques si pertinentes
            if any(keyword in guide['title'].lower() for keyword in ['farm', 'kamas', 'métier']):
                market_data = self.query_market_opportunities()
                if market_data.success and market_data.data:
                    relevant_opportunities = market_data.data[:3]
                    enriched['current_market_opportunities'] = [
                        f"{op['item']}: {op['profit_margin']:.1f}%"
                        for op in relevant_opportunities
                    ]

            # Ajouter timestamp d'enrichissement
            enriched['context_enriched_at'] = datetime.now().isoformat()
            enriched['player_context'] = {
                'class': self.game_context.player_class.value if self.game_context.player_class else None,
                'level': self.game_context.player_level,
                'server': self.game_context.current_server
            }

        except Exception as e:
            logger.warning(f"Erreur enrichissement guide: {e}")

        return enriched

    def sync_ganymede_guides(self, categories: List[str] = None) -> Dict[str, Any]:
        """Synchronise les guides Ganymede localement"""
        if not self.ganymede_connector:
            return {"error": "Connecteur Ganymede non disponible"}

        try:
            logger.info("Synchronisation guides Ganymede...")

            # Catégories par défaut selon le contexte de jeu
            if not categories:
                if self.game_context.player_level < 100:
                    categories = ["quetes", "guide-debutant", "metiers"]
                elif self.game_context.player_level > 150:
                    categories = ["donjons", "pvp", "optimisation"]
                else:
                    categories = ["quetes", "donjons", "metiers"]

            stats = self.ganymede_connector.sync_guides_database(categories)

            logger.info(f"Synchronisation terminée: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Erreur sync Ganymede: {e}")
            return {"error": str(e)}

    def get_contextual_guide_recommendations(self) -> KnowledgeQueryResult:
        """Recommandations de guides basées sur le contexte actuel"""
        start_time = time.time()

        try:
            if not self.ganymede_connector:
                return KnowledgeQueryResult(
                    query_type="contextual_guides",
                    success=False,
                    data=None,
                    source_modules=["ganymede"],
                    confidence_score=0.0,
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            recommendations = []

            # Recommandations basées sur le niveau
            level = self.game_context.player_level
            if level < 50:
                topics = ["guide débutant", "première quête", "tutoriel"]
            elif level < 100:
                topics = ["quêtes niveau moyen", "premiers donjons", "métiers"]
            elif level < 150:
                topics = ["donjons intermédiaires", "optimisation stuff"]
            else:
                topics = ["donjons haut niveau", "pvp", "end game"]

            # Rechercher guides pour chaque topic
            for topic in topics:
                guides = self.ganymede_connector.search_guides(topic)
                if guides:
                    best_guide = max(guides, key=lambda g: g.get('votes', 0))
                    recommendations.append({
                        'topic': topic,
                        'guide': best_guide,
                        'relevance_score': min(1.0, best_guide.get('votes', 0) / 10)
                    })

            # Trier par pertinence
            recommendations.sort(key=lambda r: r['relevance_score'], reverse=True)

            result = KnowledgeQueryResult(
                query_type="contextual_guides",
                success=len(recommendations) > 0,
                data={"recommendations": recommendations[:5]},
                source_modules=["ganymede"],
                confidence_score=0.8,
                execution_time_ms=(time.time() - start_time) * 1000,
                suggestions=[
                    f"{r['topic']}: {r['guide']['title']}"
                    for r in recommendations[:3]
                ]
            )

            self.query_stats["modules_usage"]["ganymede"] += 1
            return result

        except Exception as e:
            logger.error(f"Erreur recommandations contextuelles: {e}")
            return KnowledgeQueryResult(
                query_type="contextual_guides",
                success=False,
                data=None,
                source_modules=["ganymede"],
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def start_auto_update(self, interval_minutes: int = 30):
        """Demarre la mise a jour automatique des donnees"""
        if self.auto_update_running:
            return

        self.auto_update_running = True
        self.auto_update_thread = threading.Thread(target=self._auto_update_loop, args=(interval_minutes,))
        self.auto_update_thread.daemon = True
        self.auto_update_thread.start()

        logger.info(f"Mise a jour automatique demarree (interval: {interval_minutes}min)")

    def stop_auto_update(self):
        """Arrete la mise a jour automatique"""
        self.auto_update_running = False
        if self.auto_update_thread:
            self.auto_update_thread.join(timeout=5.0)
        logger.info("Mise a jour automatique arretee")

    def _auto_update_loop(self, interval_minutes: int):
        """Boucle de mise a jour automatique"""
        interval_seconds = interval_minutes * 60

        while self.auto_update_running:
            try:
                # Mise a jour economie
                self.economy_tracker.detect_market_opportunities(self.game_context.current_server)

                # Nettoyage cache
                self._cleanup_cache()

                # Statistiques
                self._update_statistics()

                logger.info("Mise a jour automatique effectuee")

            except Exception as e:
                logger.error(f"Erreur mise a jour automatique: {e}")

            time.sleep(interval_seconds)

    def _cleanup_cache(self):
        """Nettoie le cache expire"""
        # Pour l'instant, nettoyage simple
        if len(self.query_cache) > 1000:
            # Garde les 500 plus recents
            sorted_items = sorted(self.query_cache.items(),
                                key=lambda x: x[1].execution_time_ms, reverse=True)
            self.query_cache = dict(sorted_items[:500])

    def _update_statistics(self):
        """Met a jour les statistiques d'usage"""
        self.query_stats["total_queries"] += 1

        if self.query_stats["total_queries"] > 0:
            cache_hit_rate = self.query_stats["cache_hits"] / self.query_stats["total_queries"]
            logger.debug(f"Cache hit rate: {cache_hit_rate:.2%}")

    def get_system_status(self) -> Dict[str, Any]:
        """Retourne le statut du systeme"""
        return {
            "knowledge_base": {
                "spells_count": len(self.spells_db.spells),
                "monsters_count": len(self.monsters_db.monsters),
                "maps_count": len(self.maps_db.maps),
                "tracked_items": len(self.economy_tracker.tracked_items)
            },
            "performance": {
                "cache_size": len(self.query_cache),
                "cache_hit_rate": (self.query_stats["cache_hits"] / max(1, self.query_stats["total_queries"])) * 100,
                "total_queries": self.query_stats["total_queries"],
                "modules_usage": self.query_stats["modules_usage"]
            },
            "context": {
                "player_class": self.game_context.player_class.value if self.game_context.player_class else None,
                "player_level": self.game_context.player_level,
                "current_server": self.game_context.current_server,
                "in_combat": self.game_context.in_combat
            },
            "auto_update_running": self.auto_update_running
        }

    def export_knowledge_summary(self, output_path: str):
        """Exporte un resume de la base de connaissance"""
        status = self.get_system_status()

        summary = {
            "export_date": datetime.now().isoformat(),
            "system_status": status,
            "sample_data": {
                "spells": [spell.name for spell in list(self.spells_db.spells.values())[:10]],
                "monsters": [monster.name for monster in list(self.monsters_db.monsters.values())[:10]],
                "maps": [map_obj.name for map_obj in list(self.maps_db.maps.values())[:10]]
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Resume knowledge base exporte: {output_path}")

# Instance globale
_knowledge_base_instance = None

def get_knowledge_base() -> DofusKnowledgeBase:
    """Retourne l'instance singleton de la base de connaissance"""
    global _knowledge_base_instance
    if _knowledge_base_instance is None:
        _knowledge_base_instance = DofusKnowledgeBase()
    return _knowledge_base_instance

# Test du module
if __name__ == "__main__":
    kb = DofusKnowledgeBase()

    # Test contexte
    context = GameContext(
        player_class=DofusClass.IOPS,
        player_level=150,
        current_server="Julith",
        available_ap=6,
        distance_to_target=2
    )
    kb.update_game_context(context)

    # Test requetes
    spells_result = kb.query_optimal_spells()
    print(f"Sorts optimaux: {spells_result.success}, {len(spells_result.suggestions)} suggestions")

    monster_result = kb.query_monster_strategy("Bouftou")
    print(f"Strategie monstre: {monster_result.success}")

    # Statut systeme
    status = kb.get_system_status()
    print(f"Statut systeme: {status['knowledge_base']['spells_count']} sorts charges")

    # Export
    kb.export_knowledge_summary("test_knowledge_summary.json")