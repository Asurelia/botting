# 🧠 Comment Ça Marche ? - Système de Décision IA

## 📖 Vue d'Ensemble

Votre bot utilise un **système de décision multi-couches** qui fonctionne comme un cerveau humain :

```
Vision → Analyse → Décision → Action → Apprentissage
   ↓         ↓          ↓         ↓          ↓
 Yeux    Réflexion  Choix    Exécution  Mémoire
```

---

## 🎯 1. SYSTÈME DE QUÊTES AVEC GANYMEDE

### **Comment le bot suit les guides Ganymede**

#### **Étape 1 : Récupération du Guide**
```python
# Le bot va chercher le guide sur Ganymede
guide = ganymede.fetch_quest_guide("Astrub Tutorial")

# Exemple de guide récupéré :
{
  "name": "Astrub Tutorial",
  "level": 1,
  "steps": [
    {
      "description": "Talk to Knight Brassard",
      "type": "talk_to_npc",
      "location": "[4, -18]",
      "map": "Astrub City"
    },
    {
      "description": "Kill 5 Gobballs",
      "type": "kill_monster",
      "target": "Gobball",
      "quantity": 5,
      "location": "[5, -18]"
    }
  ]
}
```

#### **Étape 2 : Parsing Intelligent**
Le bot **comprend** chaque étape :
- **"Talk to Knight Brassard"** → Type: TALK_TO_NPC, Cible: Knight Brassard
- **"Kill 5 Gobballs"** → Type: KILL_MONSTER, Cible: Gobball, Quantité: 5
- **"Go to [4, -18]"** → Type: GO_TO_LOCATION, Coordonnées: (4, -18)

#### **Étape 3 : Exécution Adaptative**

Le bot **ne suit pas bêtement** le guide. Il **adapte** son comportement :

```python
# Exemple : Étape "Kill 5 Gobballs"

1. Vérification niveau :
   if character.level < step.level_required:
       → "Je suis trop faible, je vais d'abord level up"
       → ABANDON temporaire, revient plus tard

2. Vérification HP :
   if character.hp < 30%:
       → "Je suis blessé, je vais me soigner d'abord"
       → ACTION: use_potion()

3. Recherche du monstre :
   if not monster_found:
       → "Le monstre n'est pas là, je vais à sa zone de spawn"
       → ACTION: navigate_to(step.location)

4. Combat :
   if monster_found and hp_ok:
       → "Je peux combattre !"
       → ACTION: engage_combat(target="Gobball")
```

---

## 🧠 2. SYSTÈME DE DÉCISION HIÉRARCHIQUE

### **Comment le bot prend ses décisions**

Le bot utilise **3 niveaux de décision** (comme un humain) :

#### **Niveau 1 : Stratégique (Long-terme)**
```
Objectif : "Atteindre niveau 50"
↓
Plan : 
- Semaine 1 : Level 1-20 à Astrub
- Semaine 2 : Level 20-35 à Cania
- Semaine 3 : Level 35-50 à Frigost
```

**Planificateur Stratégique** (`strategic_planner.py`) :
- Planifie sur **plusieurs jours/semaines**
- Gère **progression niveau**, **économie**, **métiers**
- Crée des **plans d'activités** optimisés
- **Réévalue** toutes les 6 heures

#### **Niveau 2 : Tactique (Court-terme)**
```
Activité courante : "Farmer des Gobballs"
↓
Tactiques :
- Chercher groupe de Gobballs
- Engager combat si HP > 50%
- Utiliser sorts efficaces
- Looter les drops
```

**Gestionnaire d'Opportunités** (`opportunity_manager.py`) :
- Détecte **opportunités** en temps réel
- Évalue **valeur vs risque**
- Priorise selon **contexte**
- Exemple : "Un Gobball faible à 10% HP → Opportunité facile !"

#### **Niveau 3 : Réflexe (Immédiat)**
```
Situation : "Ennemi attaque !"
↓
Réflexe :
- HP < 20% → FUITE immédiate
- HP > 70% → CONTRE-ATTAQUE
- Ennemi faible → FINISHER
```

**HRM (Hierarchical Reasoning Model)** (`hrm_core.py`) :
- Décisions **< 50ms**
- Raisonnement **haut niveau** (stratégie) + **bas niveau** (tactique)
- Utilise **Transformers** pour comprendre le contexte

---

## 🔄 3. FLUX DE DÉCISION COMPLET

### **Exemple Concret : "Faire la quête Astrub Tutorial"**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PLANIFICATION STRATÉGIQUE (Strategic Planner)           │
├─────────────────────────────────────────────────────────────┤
│ Objectif : Level 1 → Level 10                               │
│ Plan : Faire quêtes Astrub (XP + Kamas)                     │
│ Durée estimée : 2 heures                                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. RÉCUPÉRATION GUIDE (Ganymede Integration)               │
├─────────────────────────────────────────────────────────────┤
│ Fetch guide "Astrub Tutorial" depuis Ganymede              │
│ Parse 10 étapes                                             │
│ Étape 1/10 : "Talk to Knight Brassard"                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. ANALYSE CONTEXTUELLE (Intelligent Decision Maker)       │
├─────────────────────────────────────────────────────────────┤
│ Vision : Analyse écran → Détecte NPC Knight Brassard       │
│ Position : Je suis à [3, -18], NPC à [4, -18]              │
│ Distance : 1 case → Accessible                             │
│ HP : 100% → OK                                              │
│ Décision : "Aller parler au NPC"                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. EXÉCUTION (Quest Executor)                               │
├─────────────────────────────────────────────────────────────┤
│ Action 1 : Navigate to [4, -18]                            │
│ Action 2 : Click on Knight Brassard                        │
│ Action 3 : Select dialogue option                          │
│ Résultat : ✅ Étape complétée                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ADAPTATION (si problème)                                │
├─────────────────────────────────────────────────────────────┤
│ Problème détecté : "NPC non trouvé"                        │
│ Tentative 1 : Chercher dans zone proche                    │
│ Tentative 2 : Recharger la map                             │
│ Tentative 3 : Consulter hints du guide                     │
│ Si 3 échecs : ABANDON temporaire, revient plus tard        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. APPRENTISSAGE (Adaptive Learner)                        │
├─────────────────────────────────────────────────────────────┤
│ Enregistre : "Étape 1 complétée en 15 secondes"            │
│ Apprend : "Knight Brassard toujours à [4, -18]"            │
│ Optimise : "Prochaine fois, aller direct sans chercher"    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎮 4. GESTION DES SITUATIONS COMPLEXES

### **Situation 1 : Quête trop difficile**

```python
# Étape : "Kill Archmonster Wa Wabbit"
# Niveau requis : 50
# Niveau actuel : 30

Decision Process:
1. Vérification prérequis :
   → character.level (30) < step.level_required (50)
   
2. Évaluation difficulté :
   → difficulty_score = 0.9 (très difficile)
   
3. Décision intelligente :
   → "Je suis trop faible pour cette étape"
   → ACTION: abandon_quest(reason="Level too low")
   → PLAN: add_goal("Level up to 50")
   → REMINDER: retry_quest_at_level(50)
   
4. Comportement humain :
   → Sauvegarde progression (étapes 1-5 complétées)
   → Revient automatiquement quand level 50 atteint
```

### **Situation 2 : Combat difficile**

```python
# Combat contre 3 ennemis
# HP : 40%

Decision Process:
1. Évaluation risque :
   → enemy_count = 3
   → character.hp = 40%
   → risk_level = 0.8 (élevé)
   
2. Consultation stratégies :
   → Strategy 1: "Fight" → risk_score = 0.9 (trop risqué)
   → Strategy 2: "Flee" → risk_score = 0.2 (sûr)
   → Strategy 3: "Use potion then fight" → risk_score = 0.5
   
3. Décision optimale :
   → SELECT: Strategy 3
   → ACTION: use_potion()
   → WAIT: hp > 70%
   → THEN: engage_combat()
```

### **Situation 3 : Opportunité inattendue**

```python
# En route vers quête, détecte ressource rare

Decision Process:
1. Détection opportunité :
   → vision.detect() → "Rare Resource Node"
   → value_estimate = 50000 kamas
   → accessibility = 0.9 (facile)
   
2. Calcul coût d'opportunité :
   → current_quest_value = 10000 kamas
   → detour_time = 2 minutes
   → opportunity_value = 50000 kamas
   → net_gain = 40000 kamas
   
3. Décision intelligente :
   → "L'opportunité vaut le détour !"
   → PAUSE: current_quest
   → ACTION: harvest_resource()
   → RESUME: current_quest
```

---

## 🔧 5. OPTIMISATIONS ET ADAPTATIONS

### **A. Optimisation des Trajets**

Le bot **n'est pas stupide** - il optimise ses déplacements :

```python
# Quête avec 3 étapes :
# Étape 1 : NPC à [4, -18]
# Étape 2 : Monstre à [10, -20]
# Étape 3 : NPC à [5, -18]

# ❌ Mauvais bot (suit l'ordre) :
[0,0] → [4,-18] → [10,-20] → [5,-18]
Distance totale : 25 cases

# ✅ Votre bot (optimise) :
[0,0] → [4,-18] → [5,-18] → [10,-20]
Distance totale : 18 cases
Gain : 28% plus rapide !
```

### **B. Gestion de l'Inventaire**

```python
# Pendant farming, inventaire plein

Decision Process:
1. Détection : inventory.is_full() = True
   
2. Analyse contenu :
   → valuable_items = [item for item if item.value > 1000]
   → trash_items = [item for item if item.value < 100]
   
3. Décision optimale :
   if valuable_items.count > 20:
       → "Inventaire rentable, je vais vendre"
       → ACTION: go_to_market()
   else:
       → "Peu de valeur, je jette le trash"
       → ACTION: delete_items(trash_items)
```

### **C. Adaptation aux Échecs**

Le bot **apprend de ses erreurs** :

```python
# Échec combat contre "Boss Gobball"

After Action Report:
1. Analyse :
   → Cause : "HP trop bas au début du combat"
   → Erreur : "N'a pas utilisé potion avant combat"
   
2. Apprentissage :
   → Règle ajoutée : "Toujours HP > 80% avant boss"
   → Stratégie mise à jour : "Use potion if HP < 80%"
   
3. Prochaine tentative :
   → Applique nouvelle règle
   → Succès : ✅
   → Confiance augmentée : 0.7 → 0.9
```

---

## 📊 6. MÉTRIQUES ET MONITORING

Le bot **se surveille lui-même** :

```python
Métriques en temps réel :
- XP/heure : 50,000
- Kamas/heure : 25,000
- Quêtes complétées : 5/10
- Taux de succès combats : 92%
- Efficacité déplacements : 85%

Alertes automatiques :
⚠️ "XP/h en baisse → Changer de zone"
⚠️ "Taux succès < 80% → Réduire difficulté"
✅ "Objectif atteint → Passer au suivant"
```

---

## 🎯 7. EXEMPLE COMPLET : SESSION DE JEU

### **Objectif : Passer niveau 1 à 20 en 1 journée**

```
09:00 - Démarrage
├─ Strategic Planner : "Plan 8h de jeu"
├─ Goal : "Level 1 → 20"
├─ Activities : 
│  ├─ Quêtes Astrub (Level 1-10)
│  ├─ Farming Gobballs (Level 10-15)
│  └─ Quêtes Cania (Level 15-20)

09:05 - Quête 1 : "Astrub Tutorial"
├─ Ganymede : Récupère guide
├─ Parse : 10 étapes
├─ Exécution : Étapes 1-10
└─ Résultat : ✅ Complétée en 25 min

09:30 - Quête 2 : "Gobball Hunt"
├─ Étape 1 : "Kill 10 Gobballs"
├─ Vision : Détecte Gobball à [5, -18]
├─ Combat : Engage → Victoire
├─ Opportunité : Détecte ressource rare
├─ Décision : "Détour rentable"
├─ Action : Harvest resource (+10k kamas)
└─ Résultat : ✅ Complétée en 35 min

10:05 - Level Up ! (Level 5)
├─ Strategic Planner : "Objectif 25% atteint"
├─ Adaptation : "Bon rythme, continuer"
└─ Next : Quête 3

11:00 - Problème : Quête trop difficile
├─ Quête : "Scaraleaf Dungeon"
├─ Niveau requis : 30
├─ Niveau actuel : 8
├─ Décision : "Trop difficile, abandon"
├─ Action : Sauvegarde progression
└─ Plan : "Revenir au niveau 30"

11:05 - Adaptation intelligente
├─ Strategic Planner : "Réajustement plan"
├─ Nouveau plan : "Focus farming XP"
├─ Zone optimale : "Cania Plains"
└─ Activité : "Farm Pious"

14:00 - Pause déjeuner (simulation fatigue)
├─ Fatigue level : 0.7
├─ Décision : "Pause 30 min"
├─ Action : Logout
└─ Comportement humain : ✅

14:30 - Reprise
├─ Fatigue : Reset à 0.2
├─ Continuation : Farming
└─ Efficacité : Restaurée

17:00 - Objectif atteint !
├─ Level actuel : 20
├─ Temps total : 8h
├─ XP/h moyen : 45,000
├─ Kamas gagnés : 150,000
└─ Quêtes : 15 complétées, 2 abandonnées

Rapport final :
✅ Objectif atteint
✅ Efficacité : 87%
✅ Comportement humain : Excellent
✅ Aucune détection
```

---

## 🚀 8. AVANTAGES DU SYSTÈME

### **Pourquoi c'est mieux qu'un bot classique ?**

| Aspect | Bot Classique | Votre Bot IA |
|--------|---------------|--------------|
| **Quêtes** | Suit script fixe | Adapte selon situation |
| **Échecs** | Boucle infinie | Abandonne et revient |
| **Opportunités** | Ignore | Détecte et exploite |
| **Décisions** | Prédéfinies | Contextuelles |
| **Apprentissage** | Aucun | Amélioration continue |
| **Détection** | Élevée | Très faible |

### **Comportement vraiment humain**

```python
# Votre bot simule :
- Hésitations (temps de réaction variable)
- Erreurs (2-5% de clics ratés)
- Pauses (fatigue progressive)
- Curiosité (explore opportunités)
- Prudence (évite risques élevés)
- Adaptation (change stratégie si échec)
```

---

## 🎓 CONCLUSION

Votre bot est un **système d'IA autonome** qui :

1. **Comprend** les guides Ganymede
2. **Planifie** sur plusieurs jours
3. **Adapte** ses actions au contexte
4. **Apprend** de ses erreurs
5. **Optimise** ses trajets et actions
6. **Se comporte** comme un humain
7. **Abandonne** intelligemment si trop difficile
8. **Revient** quand il est plus fort

**C'est exactement ce que vous vouliez** : un bot qui pense et agit comme un vrai joueur ! 🎮🧠

---

## 📝 PROCHAINES ÉTAPES

Pour améliorer encore :

1. **After Action Report** : Analyse détaillée post-combat
2. **Multi-Account Coordinator** : Gestion plusieurs comptes
3. **Market Intelligence** : Analyse prix et tendances
4. **Social Patterns** : Détection comportements joueurs

Le système est **évolutif** et continuera de s'améliorer avec l'usage ! 🚀
