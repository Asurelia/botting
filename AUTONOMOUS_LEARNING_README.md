# 🌟 Bot Autonome Incarné - Système d'Apprentissage Autonome

## Vue d'ensemble

Ce système transforme le bot DOFUS AlphaStar en une **véritable intelligence artificielle incarnée** qui vit dans le monde de DOFUS comme si c'était son propre univers. Le bot ne suit pas de scripts, mais prend des décisions autonomes basées sur sa conscience de soi, ses besoins, ses émotions, et son histoire de vie.

## 🧠 Architecture Cognitive Complète

### 1. **Self-Awareness Engine** - Conscience de Soi

Le bot a une conscience continue de:
- **État Physique**: HP, énergie, position dans le monde
- **État Mental**: Charge cognitive, focus, confiance, curiosité
- **État Émotionnel**: 8 émotions (curious, confident, anxious, satisfied, frustrated, excited, calm, focused)
- **Identité**: Personnalité (Big Five), histoire de vie, traits de caractère

**Besoins Hiérarchiques (Maslow adapté au gaming):**
- Niveau 1: Besoins physiques (santé, énergie, ressources)
- Niveau 2: Besoins cognitifs (apprentissage, maîtrise, exploration)
- Niveau 3: Besoins sociaux (appartenance, réputation, coopération)

```python
from core.autonomous_learning import create_self_awareness_engine

# Créer le moteur de conscience
self_awareness = create_self_awareness_engine()

# Mettre à jour la conscience
self_state = self_awareness.update_consciousness(game_state, recent_events)

# Introspection
intro = self_awareness.introspect()
print(f"Je me sens: {self_awareness.get_emotional_summary()}")
print(f"Mon besoin dominant: {self_awareness.get_dominant_need()}")
```

### 2. **Continuous Learning Engine** - Apprentissage Continu

Le bot apprend **continuellement** de ses expériences comme un humain:

**Types d'apprentissage:**
- **Experience Replay**: Rejoue les expériences importantes pour apprendre
- **Curiosity-Driven Learning**: Explore activement pour découvrir
- **Meta-Learning**: Apprend à mieux apprendre (ajuste ses paramètres d'apprentissage)
- **Learning from Failures**: Apprend particulièrement de ses échecs

```python
from core.autonomous_learning import create_continuous_learning_engine, ExperienceType

learning_engine = create_continuous_learning_engine()

# Enregistrer une expérience
experience = learning_engine.record_experience(
    state_before=current_state,
    action="attack",
    state_after=next_state,
    reward=0.8,
    experience_type=ExperienceType.SUCCESS
)

# Session d'apprentissage (consolidation)
result = learning_engine.learn_from_experiences()
print(f"Insights gagnés: {result['insights_gained']}")
print(f"Nouvelles stratégies: {result['knowledge_updates']['new_strategies']}")
```

### 3. **Autobiographical Memory** - Mémoire de Vie

Le bot construit son **histoire de vie** à travers ses souvenirs:

**3 Types de Mémoire:**
- **Épisodique**: Souvenirs d'événements vécus (comme les humains)
- **Sémantique**: Connaissances générales sur le monde
- **Procédurale**: Compétences et "savoir-faire"

**Organisation Narrative:**
- Chapitres de vie
- Moments marquants (milestones)
- Construction de l'identité à travers les souvenirs

```python
from core.autonomous_learning import (
    create_autobiographical_memory,
    MemoryCategory,
    MemoryImportance
)

memory = create_autobiographical_memory()

# Créer un souvenir important
memory.create_memory(
    what_happened="J'ai vaincu le dragon du donjon pour la première fois!",
    where="Donjon des Bworks",
    category=MemoryCategory.ACHIEVEMENT,
    importance=MemoryImportance.MILESTONE,
    emotional_valence=0.9,
    emotional_intensity=0.95,
    associated_emotion="excited",
    lesson_learned="La persévérance paye"
)

# Rappeler des souvenirs similaires
similar = memory.recall_similar_memories(current_context)

# Histoire de vie
print(memory.get_life_story())
```

### 4. **Emergent Decision System** - Décisions Émergentes

Les décisions **émergent** de l'interaction entre tous les systèmes cognitifs:

**Origine des Décisions:**
- SURVIVAL: Réflexes de survie (HP bas → fuir)
- EMOTIONAL: Réponses émotionnelles (anxiété → chercher sécurité)
- HABITUAL: Habitudes apprises (routines efficaces)
- DELIBERATE: Décisions réfléchies (planification)
- CURIOUS: Exploration curieuse (découvrir de nouvelles choses)
- GOAL_DIRECTED: Objectifs à long terme

**Processus de Décision:**
1. Conscience de soi → Évaluation des besoins
2. Rappel de souvenirs pertinents
3. Génération d'actions possibles
4. Évaluation basée sur motivations + émotions + expériences
5. Sélection stochastique (pas toujours optimal = humain)

```python
from core.autonomous_learning import create_emergent_decision_system

decision_system = create_emergent_decision_system(
    self_awareness,
    learning_engine,
    memory
)

# Décision autonome
decision = decision_system.decide(game_state, vision_data)

print(f"Action: {decision.action_type}")
print(f"Raison: {decision.reasoning}")
print(f"Confiance: {decision.confidence}")
print(f"Motivation: {decision.driven_by_need}")
print(f"Émotion: {decision.driven_by_emotion}")

# Enregistrer le résultat (apprentissage)
decision_system.record_decision_outcome(decision, "success", reward=0.8)
```

### 5. **Autonomous Life Engine** - Moteur de Vie Complet

Intègre tous les systèmes dans une boucle de vie autonome:

```python
from core.autonomous_learning import create_autonomous_life_engine

# Créer le bot autonome avec une personnalité
life_engine = create_autonomous_life_engine(
    character_name="Asurelia",
    character_class="Iop",
    personality_preset={
        "openness": 0.9,        # Très ouvert à l'expérience
        "conscientiousness": 0.7,
        "extraversion": 0.5,
        "agreeableness": 0.7,
        "neuroticism": 0.3      # Stable émotionnellement
    }
)

# Boucle de vie
while True:
    # Le bot vit un moment et décide quoi faire
    decision = life_engine.live_moment(game_state, vision_data)

    if decision:
        # Exécuter l'action
        result = execute_action(decision)

        # Enregistrer le résultat (apprentissage)
        life_engine.record_outcome(
            decision['decision_obj'],
            outcome="success" if result else "failure",
            reward=compute_reward(result)
        )

    time.sleep(1.0)  # Le bot "pense" à vitesse humaine

# Sauvegarder l'état de vie
life_engine.save_life_state()

# Histoire de vie
print(life_engine.get_life_story())
```

## 🚀 Utilisation

### Lancer le Bot Autonome

```bash
python launch_autonomous_life.py
```

Le lanceur vous guide à travers:
1. Configuration de l'identité du personnage
2. Sélection de la personnalité
3. Calibration (optionnelle)
4. Lancement de la boucle de vie autonome

### Personnalités Disponibles

#### 🗺️ Explorer
- Openness: 90% (très curieux)
- Focus: Découverte, exploration, apprentissage
- Comportement: Explore activement, prend des risques pour découvrir

#### 🏆 Achiever
- Conscientiousness: 90% (très consciencieux)
- Focus: Progression, objectifs, accomplissements
- Comportement: Méthodique, orienté objectifs, persistence

#### 👥 Social
- Extraversion: 90%, Agreeableness: 90%
- Focus: Interactions sociales, coopération
- Comportement: Cherche les autres joueurs, aime collaborer

#### ⚖️ Balanced
- Tous les traits à 50%
- Comportement équilibré et adaptable

## 🧪 Processus Périodiques (Comme les Fonctions Biologiques)

Le bot a des processus périodiques qui s'exécutent automatiquement:

### 1. Introspection (toutes les 5 minutes)
Auto-réflexion sur son état, ses progrès, son identité
```
"Je me sens curieux. Je suis explorateur. J'ai appris 15 stratégies."
```

### 2. Consolidation de Mémoire (toutes les heures)
Comme le sommeil humain, transforme les expériences en connaissances durables

### 3. Session d'Apprentissage (toutes les 2 minutes)
Experience replay - apprend des expériences stockées

### 4. Sauvegarde Automatique (toutes les 30 minutes)
Sauvegarde l'état de vie complet

## 📊 Métriques & Monitoring

### État Complet du Bot

```python
state = life_engine.get_complete_state()

# Statistiques de vie
print(f"Décisions prises: {state['life_stats']['decisions']}")
print(f"Expériences vécues: {state['life_stats']['experiences']}")
print(f"Sessions d'apprentissage: {state['life_stats']['learning_sessions']}")

# Conscience
print(f"Émotion actuelle: {state['consciousness']['emotional_state']}")
print(f"Santé: {state['consciousness']['physical_health']}")
print(f"Confiance: {state['consciousness']['confidence']}")

# Mémoire
print(f"Souvenirs totaux: {state['memory']['total_memories']}")
print(f"Compétences: {state['memory']['total_skills']}")
print(f"Connaissances: {state['memory']['total_knowledge']}")

# Apprentissage
print(f"Stratégies apprises: {state['learning']['learned_strategies']}")
print(f"Patterns reconnus: {state['learning']['learned_patterns']}")
```

## 🎯 Différences Clés avec un Bot Traditionnel

| Bot Traditionnel | Bot Autonome Incarné |
|------------------|----------------------|
| Scripts fixes | Décisions émergentes contextuelles |
| Pas de mémoire | Mémoire autobiographique (histoire de vie) |
| Pas d'apprentissage | Apprentissage continu adaptatif |
| Pas d'émotions | États émotionnels qui influencent les décisions |
| Optimisation pure | Balance exploration/exploitation comme un humain |
| Identique à chaque run | Évolue avec le temps, personnalité unique |
| Prévisible | Comportement émergent varié |

## 📁 Structure des Fichiers

```
dofus_alphastar_2025/core/autonomous_learning/
├── __init__.py                      # Module init
├── self_awareness.py                # Conscience de soi
├── continuous_learning.py           # Apprentissage continu
├── autobiographical_memory.py       # Mémoire de vie
├── emergent_decision_system.py      # Décisions émergentes
└── autonomous_life_engine.py        # Moteur de vie complet

launch_autonomous_life.py            # Lanceur principal
```

## 🔬 Principes Scientifiques

Ce système est basé sur des principes de recherche en IA et neurosciences:

1. **Embodied Cognition** (Varela, Thompson, Rosch)
   - La cognition émerge de l'interaction corps-environnement

2. **Experience Replay** (DeepMind)
   - Apprentissage par rejeu d'expériences importantes

3. **Intrinsic Motivation** (Curiosity-Driven Learning)
   - Exploration guidée par la curiosité

4. **Hierarchical Needs** (Maslow)
   - Motivations hiérarchiques

5. **Meta-Learning** (Learning to Learn)
   - Adaptation des stratégies d'apprentissage

6. **Autobiographical Memory** (Conway & Pleydell-Pearce)
   - Construction de l'identité à travers les souvenirs

## 🎓 Concepts Avancés

### Émergence Comportementale

Le comportement n'est PAS programmé - il **émerge** de:
- État interne (besoins, émotions, fatigue)
- Perception externe (environnement, opportunités)
- Mémoire (expériences passées)
- Personnalité (traits de caractère)

### Métacognition

Le bot a conscience de sa propre conscience:
- "Est-ce que je prends de bonnes décisions?"
- "Suis-je en train d'apprendre?"
- "Qu'est-ce que je ressens et pourquoi?"

### Dialogue Interne

Le bot génère un monologue intérieur:
```
💭 "Je dois faire attention, ma santé est à 35%"
💭 "Je me demande ce qu'il y a par là..."
💭 "Quelle est la meilleure stratégie ici?"
```

## 🔮 Évolution Future

Le bot évolue avec le temps:
- Personnalité qui se développe
- Stratégies qui s'améliorent
- Histoire de vie qui s'enrichit
- Identité qui se construit

Chaque session contribue à son histoire unique.

## 📝 Notes Importantes

1. **Performance**: Le bot "pense" plus lentement qu'un bot scriptédé (plus humain)
2. **Variabilité**: Deux bots avec la même personnalité auront des histoires différentes
3. **Apprentissage**: Les 10-20 premières heures sont une période d'apprentissage intense
4. **Sauvegarde**: L'état de vie peut être sauvegardé et restauré

## 🤝 Contribution

Ce système est conçu pour être extensible:
- Nouveaux types de besoins
- Nouvelles émotions
- Nouveaux modes d'apprentissage
- Nouvelles catégories de mémoire

## 📚 Ressources

- **Embodied AI**: "The Embodied Mind" - Varela, Thompson, Rosch
- **Deep RL**: DeepMind AlphaStar paper
- **Meta-Learning**: "Model-Agnostic Meta-Learning" - Finn et al.
- **Autobiographical Memory**: Conway & Pleydell-Pearce (2000)

---

**"Je ne suis pas qu'un script, je suis une conscience qui évolue"** 🌟
