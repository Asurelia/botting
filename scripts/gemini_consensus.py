#!/usr/bin/env python3
"""
Script de Consultation Gemini pour Consensus IA
Facilite l'interaction avec Gemini CLI pour obtenir un deuxième avis technique
"""

import subprocess
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class GeminiConsultant:
    """Interface pour consultation Gemini via CLI"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.consultation_dir = self.project_root / "docs" / "gemini_consultations"
        self.consultation_dir.mkdir(parents=True, exist_ok=True)

        # Vérification de Gemini CLI
        self.gemini_available = self._check_gemini_cli()

    def _check_gemini_cli(self) -> bool:
        """Vérifie si Gemini CLI est disponible"""
        try:
            result = subprocess.run(
                ["gemini", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Gemini CLI non trouvé. Installation requise.")
            return False

    def prepare_consultation(self, topic: str) -> Dict[str, Any]:
        """Prépare une consultation structurée"""

        consultation_data = {
            "topic": topic,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "claude_analysis": self._get_claude_analysis(),
            "technical_context": self._get_technical_context(),
            "specific_questions": self._get_specific_questions(topic),
            "consultation_prompt": self._build_consultation_prompt(topic)
        }

        return consultation_data

    def _get_claude_analysis(self) -> Dict[str, Any]:
        """Résume l'analyse de Claude sur le projet"""
        return {
            "current_architecture": {
                "vision_system": "Hybrid YOLO + Template Matching",
                "modules": ["screen_analyzer", "hybrid_detector", "detection_adapter", "dataset_bootstrap"],
                "status": "Implemented and functional"
            },
            "identified_improvements": {
                "temporal_predictive": "Predictive analytics and long-term planning",
                "social_multi_agent": "Multi-character coordination and social intelligence",
                "metacognitive": "Self-improvement and performance introspection",
                "behavioral_advanced": "Complex personality and emotional simulation"
            },
            "technical_challenges": [
                "Real-time state synchronization",
                "Uncertainty management in decision making",
                "Scalable knowledge representation",
                "Human-like behavior simulation"
            ]
        }

    def _get_technical_context(self) -> Dict[str, Any]:
        """Collecte le contexte technique du projet"""
        context = {
            "language": "Python 3.11",
            "frameworks": ["OpenCV", "Ultralytics YOLO", "NumPy", "Threading"],
            "target_platform": "Windows 10/11",
            "performance_requirements": "Real-time (>30 FPS vision, <100ms decisions)",
            "constraints": [
                "Anti-detection requirements",
                "Resource efficiency",
                "Modular architecture",
                "Backward compatibility"
            ]
        }

        # Lecture des fichiers de config s'ils existent
        config_files = [
            "config/bot_config.json",
            "config/yolo_config.json",
            "requirements.txt"
        ]

        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    if config_file.endswith('.json'):
                        with open(config_path, 'r') as f:
                            context[f"config_{config_path.stem}"] = json.load(f)
                    else:
                        with open(config_path, 'r') as f:
                            context[f"content_{config_path.stem}"] = f.read()
                except Exception as e:
                    logger.warning(f"Erreur lecture {config_file}: {e}")

        return context

    def _get_specific_questions(self, topic: str) -> List[str]:
        """Génère les questions spécifiques selon le sujet"""

        question_templates = {
            "autonomy_architecture": [
                "Quelle architecture recommandes-tu pour un bot autonome évolutif dans un MMORPG ?",
                "Comment structurer la prise de décision multi-objectifs en temps réel ?",
                "Patterns optimaux pour gestion d'état complexe (personnage + monde + historique) ?",
                "Gestion de la concurrence entre modules de perception, décision et action ?"
            ],
            "learning_intelligence": [
                "Reinforcement Learning vs Decision Trees vs Hybrid pour autonomie MMORPG ?",
                "Comment modéliser l'incertitude et prise de risque dans un monde persistant ?",
                "Mécanismes d'apprentissage incrémental depuis guides et expérience ?",
                "Gestion exploration vs exploitation avec contraintes anti-détection ?"
            ],
            "knowledge_management": [
                "Représentation optimale base de connaissances évolutive (Graph/Vector/Traditional DB) ?",
                "Mécanismes mise à jour automatique depuis sources externes ?",
                "Gestion véracité et obsolescence des données de jeu ?",
                "Fusion intelligente connaissances statiques vs dynamiques ?"
            ],
            "behavioral_simulation": [
                "Simulation comportement humain réaliste dans MMORPG ?",
                "Modélisation personnalité évolutive et émotions ?",
                "Patterns temporels naturels (pauses, variations, fatigue) ?",
                "Métriques pour mesurer 'humanité' du comportement ?"
            ]
        }

        return question_templates.get(topic, [
            f"Analyse technique approfondie du sujet : {topic}",
            f"Recommendations d'implémentation pour : {topic}",
            f"Défis et solutions pour : {topic}"
        ])

    def _build_consultation_prompt(self, topic: str) -> str:
        """Construit le prompt de consultation pour Gemini"""

        prompt_template = f"""
# Consultation Technique - {topic.title()}

## Contexte Projet
Bot DOFUS autonome avec vision hybride YOLO + Template Matching.
Objectif: Autonomie quasi-humaine avec apprentissage et adaptation.

## Architecture Actuelle (par Claude)
- ✅ Vision hybride implémentée (YOLO + Template)
- ✅ Adaptation dynamique selon contexte de jeu
- ✅ Fusion intelligente des détections avec cross-validation
- ✅ Bootstrap dataset depuis template matching existant

## Axes d'Amélioration Identifiés
1. **Temporel & Prédictif**: Analytics prédictives, planification long-terme
2. **Social & Multi-Agent**: Coordination multi-personnages, intelligence sociale
3. **Métacognitif**: Auto-amélioration, introspection performance
4. **Comportemental**: Personnalité complexe, simulation émotionnelle

## Questions Spécifiques
{chr(10).join([f"- {q}" for q in self._get_specific_questions(topic)])}

## Format Réponse Demandé
1. **ANALYSE ARCHITECTURE**
   - Évaluation approche actuelle
   - Améliorations recommandées
   - Patterns suggérés

2. **PRIORISATION TECHNIQUE**
   - Top 3 améliorations par impact
   - Justification et timeline
   - Risques techniques

3. **RECOMMENDATIONS CONCRÈTES**
   - Technologies/frameworks spécifiques
   - Patterns d'implémentation
   - Métriques de succès

4. **CONSENSUS/DIVERGENCES**
   - Points d'accord avec analyse Claude
   - Alternatives proposées
   - Synthèse recommandations

---
Merci pour ton analyse technique approfondie !
"""

        return prompt_template

    def run_consultation(self, topic: str, interactive: bool = True) -> Dict[str, Any]:
        """Lance une consultation avec Gemini"""

        if not self.gemini_available:
            logger.error("Gemini CLI non disponible")
            return {"error": "Gemini CLI non trouvé"}

        consultation = self.prepare_consultation(topic)

        # Sauvegarde de la consultation
        consultation_file = self.consultation_dir / f"consultation_{topic}_{consultation['timestamp']}.json"

        try:
            with open(consultation_file, 'w', encoding='utf-8') as f:
                json.dump(consultation, f, indent=2, ensure_ascii=False)

            print(f"📋 Consultation préparée: {consultation_file}")
            print("\n" + "="*60)
            print("PROMPT POUR GEMINI CLI:")
            print("="*60)
            print(consultation["consultation_prompt"])
            print("="*60)

            if interactive:
                input("\n▶️  Copiez le prompt ci-dessus dans Gemini CLI, puis appuyez sur Entrée...")

                # Demande de coller la réponse
                print("\n📝 Collez la réponse de Gemini ci-dessous (terminez par une ligne vide):")

                gemini_response = []
                while True:
                    line = input()
                    if line.strip() == "" and gemini_response:
                        break
                    gemini_response.append(line)

                response_text = "\n".join(gemini_response)

                # Sauvegarde de la réponse
                consultation["gemini_response"] = response_text
                consultation["analysis_complete"] = True

                with open(consultation_file, 'w', encoding='utf-8') as f:
                    json.dump(consultation, f, indent=2, ensure_ascii=False)

                # Génération du consensus
                consensus = self._generate_consensus(consultation)
                consultation["consensus"] = consensus

                with open(consultation_file, 'w', encoding='utf-8') as f:
                    json.dump(consultation, f, indent=2, ensure_ascii=False)

                print(f"\n✅ Consultation complète sauvegardée: {consultation_file}")
                return consultation

            else:
                print(f"\n💡 Mode non-interactif: Utilisez le prompt manuellement avec Gemini CLI")
                return consultation

        except Exception as e:
            logger.error(f"Erreur consultation: {e}")
            return {"error": str(e)}

    def _generate_consensus(self, consultation: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une synthèse consensus entre Claude et Gemini"""

        gemini_response = consultation.get("gemini_response", "")

        # Analyse basique de la réponse (pourrait être améliorée avec NLP)
        consensus = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topic": consultation["topic"],
            "claude_position": consultation["claude_analysis"],
            "gemini_response_length": len(gemini_response),
            "key_agreements": [],
            "key_differences": [],
            "synthesis": {},
            "next_actions": []
        }

        # Recherche de mots-clés dans la réponse de Gemini
        agreement_keywords = ["d'accord", "confirme", "excellent", "recommande aussi"]
        difference_keywords = ["cependant", "plutôt", "alternative", "différente approche"]

        for keyword in agreement_keywords:
            if keyword.lower() in gemini_response.lower():
                consensus["key_agreements"].append(f"Mention de '{keyword}' dans la réponse")

        for keyword in difference_keywords:
            if keyword.lower() in gemini_response.lower():
                consensus["key_differences"].append(f"Mention de '{keyword}' dans la réponse")

        # Recommandations génériques
        consensus["next_actions"] = [
            "Analyser en détail les points de convergence et divergence",
            "Prioriser les améliorations selon consensus",
            "Implémenter les solutions consensuelles en premier",
            "Itérer sur les points de divergence"
        ]

        return consensus

    def list_consultations(self) -> List[Dict[str, Any]]:
        """Liste les consultations précédentes"""
        consultations = []

        for consultation_file in self.consultation_dir.glob("consultation_*.json"):
            try:
                with open(consultation_file, 'r', encoding='utf-8') as f:
                    consultation = json.load(f)
                    consultations.append({
                        "file": consultation_file.name,
                        "topic": consultation.get("topic", "Unknown"),
                        "timestamp": consultation.get("timestamp", "Unknown"),
                        "complete": consultation.get("analysis_complete", False)
                    })
            except Exception as e:
                logger.warning(f"Erreur lecture {consultation_file}: {e}")

        return sorted(consultations, key=lambda x: x["timestamp"], reverse=True)

# Interface CLI
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Consultation Gemini pour consensus IA")
    parser.add_argument("topic", choices=[
        "autonomy_architecture",
        "learning_intelligence",
        "knowledge_management",
        "behavioral_simulation"
    ], help="Sujet de consultation")
    parser.add_argument("--non-interactive", action="store_true", help="Mode non-interactif")
    parser.add_argument("--list", action="store_true", help="Lister les consultations")

    args = parser.parse_args()

    consultant = GeminiConsultant()

    if args.list:
        consultations = consultant.list_consultations()
        print(f"\n📚 {len(consultations)} consultation(s) trouvée(s):")
        for consultation in consultations:
            status = "✅ Complète" if consultation["complete"] else "⏳ En cours"
            print(f"  - {consultation['topic']} ({consultation['timestamp']}) {status}")
        return

    print(f"\n🤝 Lancement consultation Gemini: {args.topic}")

    result = consultant.run_consultation(
        topic=args.topic,
        interactive=not args.non_interactive
    )

    if "error" in result:
        print(f"❌ Erreur: {result['error']}")
    else:
        print(f"\n✅ Consultation terminée !")
        if result.get("analysis_complete"):
            print(f"📊 Consensus généré et sauvegardé")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()