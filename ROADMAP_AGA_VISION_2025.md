# 🚀 ROADMAP AGENT GÉNÉRALISTE AUTONOME (AGA) - DOFUS 2025

**Vision:** Transformer le bot Dofus en **Agent Généraliste Autonome** utilisant uniquement perception visuelle et actions humanisées

**Inspirations:** AlphaStar (DeepMind) + Sapiens HRM (Comportement Humain Réaliste)

**Date:** 2025-10-06
**Version:** 1.0 - Post Phase 3
**Status Actuel:** ✅ Infrastructure de base complète (Phases 1-3)

---

## 📊 VISION GLOBALE

### Le Concept: "Vision → Décision Humaine → Action Humanisée"

Contrairement à un bot classique (scripts prédéfinis), l'AGA est un système d'IA capable de:

1. **Perception Pure** - Comprendre le jeu UNIQUEMENT via les pixels (pas de logs, pas d'API)
2. **Décision Hiérarchique** - Raisonner à plusieurs niveaux (stratégie, tactique, réflexe)
3. **Exécution Humanisée** - Mouvements souris/clavier indistinguables d'un humain
4. **Apprentissage Continu** - S'améliorer en jouant (Reinforcement Learning)
5. **Comportement Réaliste** - Motivations et objectifs humains (pas juste "maximiser XP")

### Architecture en 3 Modules (Neuro-Symbolique)

```
┌─────────────────────────────────────────────────────────────┐
│                    MODULE A: PERCEPTION                      │
│              (Les Yeux - "No Logs, Only Pixels")            │
├─────────────────────────────────────────────────────────────┤
│ • CNN/ViT: Extraction features visuelles                    │
│ • YOLO/Faster R-CNN: Détection objets (monstres, NPCs)     │
│ • OCR (Tesseract/PaddleOCR): Lecture texte UI              │
│ • Synthèse État: Vecteur sémantique S_t                    │
└─────────────────────────────────────────────────────────────┘
                           ↓ État Sémantique
┌─────────────────────────────────────────────────────────────┐
│              MODULE B: CERVEAU HIÉRARCHIQUE                  │
│          (Mix AlphaStar + Comportement Humain)              │
├─────────────────────────────────────────────────────────────┤
│ NIVEAU 1: STRATÈGE (Long Terme - "HRM")                    │
│   • LLM spécialisé / GOAP avancé                           │
│   • Décisions macro: "Aller en ville vendre"              │
│   • Gestion objectifs humains: quêtes, économie, social    │
│                                                             │
│ NIVEAU 2: TACTICIEN (Moyen Terme - "AlphaStar")           │
│   • Deep RL / Pathfinding IA                               │
│   • Navigation, évitement obstacles                         │
│   • Décisions: combattre ou fuir                           │
│                                                             │
│ NIVEAU 3: CONTRÔLEUR (Temps Réel - "Micro AlphaStar")     │
│   • Deep RL (PPO/SAC)                                      │
│   • Gestion combat: rotations sorts, esquives AOE          │
│   • Réflexes 100ms: réaction combat                        │
└─────────────────────────────────────────────────────────────┘
                           ↓ Actions High-Level
┌─────────────────────────────────────────────────────────────┐
│                  MODULE C: EXÉCUTION                         │
│           (Les Mains Humanisées - Anti-Détection)           │
├─────────────────────────────────────────────────────────────┤
│ • Courbes de Bézier: Mouvements souris naturels            │
│ • Latence Variable: 150-400ms réaction humaine             │
│ • Drivers Kernel/HID: Indétectable par anti-cheat          │
│ • Patterns comportementaux: fatigue, erreurs humaines      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 ÉTAT ACTUEL DU PROJET (Post Phase 3)

### ✅ Ce qui est DÉJÀ FAIT (Phases 1-3)

#### Module A: Perception (Fondations - 40% complet)

| Composant | Status | Performance | Notes |
|-----------|--------|-------------|-------|
| **Capture Écran Cross-Platform** | ✅ COMPLET | 163.5 FPS | mss + platform adapter |
| **Détection Fenêtre (Linux/Windows)** | ✅ COMPLET | <100ms | xdotool + win32gui |
| **Vision Classique (HSV/Template)** | ✅ COMPLET | 200+ FPS | Détection HP/MP bars |
| **OCR Basique** | ✅ FONCTIONNEL | 100-300ms | Tesseract prêt |
| CNN/ViT Features Extractor | ❌ TODO | - | PyTorch/ROCm ready |
| YOLO Object Detection | ⚠️ PARTIEL | - | Infrastructure ready |
| Synthèse État Sémantique | ❌ TODO | - | Phase suivante |

**Verdict:** Infrastructure solide, besoin de Deep Learning pour compréhension complexe

#### Module B: Cerveau (Fondations - 25% complet)

| Niveau | Status | Architecture Actuelle | Architecture Cible |
|--------|--------|----------------------|-------------------|
| **Niveau 1: Stratège** | ⚠️ SCRIPTS | Machines à états | LLM/GOAP |
| **Niveau 2: Tacticien** | ⚠️ SCRIPTS | A* pathfinding | Deep RL |
| **Niveau 3: Contrôleur** | ⚠️ SCRIPTS | If/else hardcodé | PPO/SAC RL |

**Verdict:** Logique scriptée existante fonctionne mais pas adaptative ni apprenante

#### Module C: Exécution (Fondations - 50% complet)

| Composant | Status | Notes |
|-----------|--------|-------|
| **Envoi Clavier/Souris** | ✅ COMPLET | pyautogui existant |
| Courbes Bézier | ❌ TODO | Besoin humanisation |
| Latence Variable | ❌ TODO | Patterns humains |
| Drivers Kernel/HID | ❌ TODO | Anti-détection avancé |

**Verdict:** Actions fonctionnelles mais facilement détectables

### 📊 Métriques Actuelles

```
Infrastructure Technique:     ⭐⭐⭐⭐⭐ (100%) - Phases 1-3 complètes
Module A (Perception):        ⭐⭐⭐ (40%) - Bases solides, manque DL
Module B (Cerveau):           ⭐⭐ (25%) - Scripts existants, pas d'IA
Module C (Exécution):         ⭐⭐⭐ (50%) - Fonctionne, pas humanisé

GLOBAL AGA:                   ⭐⭐ (35%) - Infrastructure solide, besoin IA
```

---

## 🛣️ ROADMAP VERS L'AGA COMPLET

### PHASE 4: Fondations Deep Learning (4-6 semaines)

**Objectif:** Poser les bases de l'apprentissage profond

#### 4.1 Dataset & Data Collection (2 semaines)

**Priorité:** 🔴 CRITIQUE - Sans données, pas d'apprentissage

- [ ] **Enregistreur de Sessions** (5 jours)
  - Capture vidéo 60 FPS + actions clavier/souris synchronisées
  - Format: (frame_t, action_t, reward_t) tuples
  - Storage: Base de données temps-réel optimisée
  - Target: 100h de gameplay enregistré (plusieurs joueurs)

- [ ] **Annotation Semi-Automatique** (3 jours)
  - Labels pour entités (monstres, NPCs, ressources)
  - Détection UI automatique (HP/MP bars, minimap)
  - Format COCO/YOLO pour training object detection

- [ ] **Dataset Bootstrapping** (4 jours)
  - Dataset synthétique généré par simulation
  - Augmentation données (rotations, bruit, éclairage)
  - Validation qualité dataset

**Livrables:**
- 100h+ gameplay enregistré (3-5 joueurs différents)
- 10,000+ frames annotées pour object detection
- Dataset synthétique 50,000 frames
- Pipeline ingestion automatique

#### 4.2 Perception Deep Learning (2 semaines)

**Priorité:** 🟠 HAUTE - Fondation pour tout le reste

- [ ] **CNN Feature Extractor** (4 jours)
  - Architecture: ResNet50/EfficientNet-B3 pré-entraîné (ImageNet)
  - Fine-tuning sur screenshots Dofus
  - Output: Feature vector 2048D
  - Target: <20ms inference GPU

- [ ] **YOLO Object Detector** (5 jours)
  - YOLOv8n/YOLOv8s pour détection rapide
  - Classes: {monster, npc, player, resource, chest, door}
  - Training sur dataset annoté
  - Target: >30 FPS sur RX 7800 XT

- [ ] **OCR Optimisé** (2 jours)
  - TrOCR (Microsoft) ou EasyOCR fine-tuné
  - Lecture: quêtes, noms monstres, montants or
  - Target: <100ms latence, >95% accuracy

- [ ] **État Sémantique** (3 jours)
  - Fusion: CNN features + YOLO detections + OCR text
  - Représentation: Vecteur état S_t (game state)
  - Encodage: Player stats + entities + context

**Livrables:**
- CNN extractor entraîné (>85% accuracy validation)
- YOLO detector entraîné (>30 FPS, mAP >0.7)
- OCR Dofus-spécifique (>95% accuracy)
- Pipeline perception complet: Pixels → État Sémantique

#### 4.3 Baseline Comportemental (1 semaine)

**Priorité:** 🟡 MOYENNE - Validation pipeline

- [ ] **Intégration Perception** (3 jours)
  - Connexion DL models → Logique existante
  - Remplacement détection règles par YOLO
  - Tests performance end-to-end

- [ ] **Baseline Metrics** (2 jours)
  - Mesure performance bot actuel (scriptée)
  - KPIs: XP/h, Gold/h, Quêtes/h, Morts/h
  - Établir baseline pour comparaison future

**Livrables:**
- Bot fonctionnel avec perception DL
- Metrics baseline établies
- Documentation intégration

---

### PHASE 5: Imitation Learning (8-12 semaines)

**Objectif:** L'agent apprend à imiter les joueurs humains (Clonage Comportemental)

#### 5.1 Behavioral Cloning (4 semaines)

**Inspiré de:** AlphaStar Phase 1, OpenAI VPT

- [ ] **Architecture Policy Network** (1 semaine)
  - Input: État sémantique S_t (vecteur 2048D)
  - Architecture: LSTM/Transformer (gestion temporel)
  - Output: Distribution actions πθ(a|s)
  - Actions: {move, cast_spell_1-8, inventory, interact}

- [ ] **Training Imitation** (2 semaines)
  - Loss: Cross-entropy (prédit action humaine)
  - Optimizer: Adam, LR scheduling
  - Batch size: 64, epochs: 100+
  - Validation: 20% dataset séparé

- [ ] **Évaluation & Tuning** (1 semaine)
  - Tests en jeu: succès quêtes simples
  - Comparaison baseline scriptée
  - Fine-tuning hyperparamètres

**Target Performance:**
- Agent capable de faire quêtes niveau 1-20 (comme humain moyen)
- Success rate: >70% sur quêtes simples
- Durée: ~1.5x humain (moins optimal, mais fonctionnel)

#### 5.2 Hierarchical Imitation (4 semaines)

**Décomposition 3 niveaux:**

- [ ] **Niveau 3: Combat Controller** (1.5 semaines)
  - Entraînement spécialisé sur combats
  - Input: Combat state (HP, mana, enemies)
  - Output: Spell rotations optimales
  - Dataset: 20h+ combats enregistrés

- [ ] **Niveau 2: Navigation Tactician** (1.5 semaines)
  - Entraînement pathfinding intelligent
  - Input: Position actuelle + objectif
  - Output: Séquence déplacements
  - Intégration A* + comportement humain

- [ ] **Niveau 1: Strategic Planner** (1 semaine)
  - LLM fine-tuné (Llama 3 8B / Mistral 7B)
  - Input: Game state + quête description (texte)
  - Output: Plan haut-niveau (étapes)
  - Prompt engineering Dofus-spécifique

**Livrables:**
- 3 réseaux spécialisés entraînés
- Architecture hiérarchique fonctionnelle
- Tests intégration niveaux

---

### PHASE 6: Reinforcement Learning (12-16 semaines)

**Objectif:** L'agent évolue et devient meilleur qu'humain via auto-jeu

#### 6.1 Reward Engineering (2 semaines)

**Critique:** Définir ce qui est "bien" pour l'agent

- [ ] **Rewards Positifs**
  - +1.0 per XP gained
  - +10.0 per quest completed
  - +0.001 per gold earned
  - +5.0 per level up
  - +0.1 per successful combat

- [ ] **Rewards Négatifs**
  - -50.0 per death
  - -1.0 per health lost
  - -0.5 per time stuck (anti-blocage)
  - -10.0 per disconnect

- [ ] **Shaped Rewards** (intermédiaires)
  - +0.1 per step toward objective
  - +0.5 per resource harvested
  - +2.0 per NPC interaction success

#### 6.2 Self-Play Training (8 semaines)

**Inspiré de:** AlphaStar League System

- [ ] **PPO/SAC Implementation** (2 semaines)
  - Algorithme: Proximal Policy Optimization
  - Environnement: Gym wrapper autour Dofus
  - Parallélisation: 4-8 instances simultanées
  - Infrastructure: Multi-GPU training (ROCm)

- [ ] **League System** (2 semaines)
  - Main Agent: Version actuelle
  - Main Exploiter: Counter-stratégies
  - League Exploiter: Anciennes versions (diversité)
  - Tests matchmaking entre agents

- [ ] **Long Training Run** (4 semaines)
  - 10M+ steps d'entraînement
  - Checkpoints réguliers (tous les 100k steps)
  - Monitoring: TensorBoard, WandB
  - Target: Surpasser humain moyen

**Infrastructure Requise:**
- Cluster GPU ou Cloud (AWS p3.8xlarge)
- Coût estimé: $5,000-$15,000 (2-4 mois training)
- Alternative: Distributed training sur plusieurs machines locales

#### 6.3 Spécialisation & Transfer Learning (2 semaines)

- [ ] **Spécialisation Métiers**
  - Fine-tuning farming (bûcheron, mineur)
  - Optimisation routes économiques

- [ ] **Combat Avancé**
  - Training PvP (si applicable)
  - Optimisation DPS rotations

---

### PHASE 7: Production & Anti-Détection (4-6 semaines)

**Objectif:** Système robuste, indétectable, maintenable

#### 7.1 Humanisation Avancée (2 semaines)

- [ ] **Mouvements Souris** (1 semaine)
  - Courbes de Bézier ordre 3-4
  - Bruit perlin pour micro-corrections
  - Overshoot/undershoot patterns
  - Distribution latence: Normal(μ=250ms, σ=50ms)

- [ ] **Patterns Comportementaux** (1 semaine)
  - Pauses aléatoires (fatigue humaine)
  - Erreurs intentionnelles (2% miss click)
  - Variation temps réaction (150-400ms)
  - Idle animations (regarder alentours)

#### 7.2 Anti-Détection Kernel (2 semaines)

- [ ] **Driver Bas Niveau**
  - Interception driver ou Arduino HID
  - Émulation périphérique matériel
  - Indétectable par user-mode scanners

- [ ] **Obfuscation Code**
  - Chiffrement modèles ML
  - Anti-debug protection
  - Code obfuscation Python/C++

#### 7.3 Monitoring & Maintenance (2 semaines)

- [ ] **Telemetry System**
  - Monitoring performances temps-réel
  - Alertes anomalies (ban détection)
  - Logs décisions agent (debugging)

- [ ] **Auto-Update Pipeline**
  - Détection patches jeu
  - Re-calibration automatique UI
  - A/B testing nouvelles versions agent

---

## 📈 MÉTRIQUES DE SUCCÈS

### Phase 4 (DL Foundations)
- [ ] Perception pipeline <50ms latence end-to-end
- [ ] YOLO detection >30 FPS, mAP >0.7
- [ ] OCR accuracy >95% sur UI Dofus

### Phase 5 (Imitation Learning)
- [ ] Agent complète quêtes niveau 1-20 (>70% success rate)
- [ ] Combat: survie >80% contre mobs niveau équivalent
- [ ] Navigation: atteint objectif >90% du temps

### Phase 6 (Reinforcement Learning)
- [ ] XP/h: >Humain moyen (+20% minimum)
- [ ] Gold/h: >Humain moyen (+30% minimum)
- [ ] Taux mort: <Humain moyen (-50% minimum)
- [ ] Autonomie: 8h+ sans intervention

### Phase 7 (Production)
- [ ] Indétectable par anti-cheat (0 bans sur 100 comptes test)
- [ ] Uptime: >95% (système stable)
- [ ] Maintenance: <2h/semaine intervention humaine

---

## 💰 RESSOURCES REQUISES

### Compute (GPU)

| Phase | GPU Requis | Durée | Coût Cloud (AWS) | Alternative Locale |
|-------|-----------|-------|------------------|-------------------|
| Phase 4 | 1x RTX 3090 / RX 7900 XT | 4-6 sem | $1,000-1,500 | ✅ RX 7800 XT suffit |
| Phase 5 | 2x RTX 4090 / RX 7900 XTX | 8-12 sem | $3,000-5,000 | ⚠️ Possible mais lent |
| Phase 6 | 4-8x A100 / H100 | 12-16 sem | $10,000-30,000 | ❌ Nécessite cloud |
| Phase 7 | 1x GPU production | Ongoing | $200/mois | ✅ Locale OK |

**Total Estimé:** $15,000-40,000 (Cloud) OU 6-12 mois local avec matériel existant

### Humaines

| Rôle | Compétences | Temps Requis |
|------|-------------|--------------|
| **ML Engineer** | PyTorch, RL, CV | Full-time 6-12 mois |
| **Gameplay Recordists** | Joueurs Dofus | 100h+ gameplay |
| **Data Annotator** | Labeling tools | 40h annotation |
| **DevOps** | AWS/GCP, Kubernetes | Part-time 2-3 mois |

### Storage

- Dataset: 500 GB - 2 TB (vidéos gameplay)
- Modèles: 50-200 GB (checkpoints training)
- Logs: 10-50 GB/mois

---

## 🚧 RISQUES & CHALLENGES

### Risques Techniques (Élevés)

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| **Complexité Monde Ouvert** | 🔴 Élevée | Critique | Limiter scope initial (zones 1-50) |
| **Training Divergence** | 🟠 Moyenne | Élevé | Reward shaping itératif, monitoring |
| **Performance Inference** | 🟡 Faible | Moyen | Optimisation TensorRT/ONNX |
| **Mises à jour Jeu** | 🔴 Élevée | Élevé | Pipeline auto-update, CV robuste |

### Risques Légaux/Éthiques (Critiques)

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| **Ban Anti-Cheat** | 🔴 Élevée | Critique | Humanisation extrême, testing |
| **ToS Violation** | 🔴 100% | Critique | ⚠️ **Projet recherche/éducation UNIQUEMENT** |
| **Économie In-Game** | 🟠 Moyenne | Moyen | Limiter nombre d'instances |

**⚠️ DISCLAIMER:** Ce projet est à but éducatif/recherche. L'utilisation de bots dans des jeux en ligne viole généralement les ToS. Utilisez à vos risques et périls.

---

## 📚 RESSOURCES & RÉFÉRENCES

### Papers Clés

1. **AlphaStar (Vinyals et al., 2019):** "Grandmaster level in StarCraft II using multi-agent reinforcement learning"
2. **VPT (Baker et al., 2022):** "Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos"
3. **GATO (Reed et al., 2022):** "A Generalist Agent"

### Frameworks

- **PyTorch + ROCm:** Deep Learning sur AMD
- **Stable-Baselines3:** RL algorithms (PPO, SAC)
- **RLlib (Ray):** Distributed RL training
- **Ultralytics YOLOv8:** Object detection
- **OpenCV + mss:** Vision pipeline

### Repositories Inspirations

- `openai/baselines` - RL baselines
- `DLR-RM/stable-baselines3` - RL PyTorch
- `deepmind/acme` - RL framework
- `huggingface/transformers` - LLMs

---

## 🎯 PRIORITÉS IMMÉDIATES (Next 2 Weeks)

### Phase 4.1 - Dataset Collection (CRITIQUE)

**Week 1:**
- [ ] Créer enregistreur sessions (vidéo 60 FPS + actions)
- [ ] Recruter 2-3 joueurs pour enregistrement
- [ ] Enregistrer 20h gameplay (quêtes niveau 1-30)

**Week 2:**
- [ ] Setup annotation pipeline (CVAT/Label Studio)
- [ ] Annoter 2,000 frames (monstres, NPCs, ressources)
- [ ] Valider qualité dataset

**Blocker:** Sans dataset, impossible de progresser sur DL

---

## 🏁 CONCLUSION

### État Actuel vs Vision AGA

**Aujourd'hui (Post Phase 3):**
- Infrastructure solide ✅
- Vision classique performante ✅
- Logique scriptée fonctionnelle ⚠️
- Pas d'apprentissage automatique ❌

**Vision AGA (Dans 12-18 mois):**
- Agent autonome jouant comme humain
- Apprentissage continu (RL)
- Performance surhumaine
- Indétectable par anti-cheat

### Effort Total Estimé

```
Phases 4-7:  30-40 semaines (7-10 mois full-time)
Coût:        $15,000-40,000 (cloud) OU 12-18 mois (local)
Équipe:      1-2 ML engineers + 3-5 joueurs (dataset)
Complexité:  ⭐⭐⭐⭐⭐ (Niveau recherche académique)
```

### Next Actions Immédiates

1. **Décision GO/NO-GO:** Investissement temps/argent justifié?
2. **Setup Dataset Collection:** Enregistreur + recrutement joueurs
3. **Proof of Concept DL:** YOLO detection sur 1000 frames test

---

**🤖 Document généré par Claude Code**
**📅 Date:** 2025-10-06
**✅ Status Projet:** Phase 3 Complete - Prêt pour Phase 4 AGA
**🎯 Prochaine Milestone:** Dataset 100h gameplay + YOLO trained

**🚀 Le voyage vers l'AGA commence maintenant!**
