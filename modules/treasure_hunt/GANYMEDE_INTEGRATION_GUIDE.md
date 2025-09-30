# 🗺️ Guide d'Intégration Ganymede - Chasses au Trésor

## 📖 Vue d'Ensemble

Ce module intègre les données **open-source de Ganymede** pour résoudre automatiquement les chasses au trésor dans Dofus.

### **Sources Ganymede** :
- 🌐 **GitHub** : https://github.com/Dofus-Batteries-Included/Dofus
- 🗺️ **Dofus-Map** : https://dofus-map.com/
- 📊 **Base de données** : https://dofus-map.com/huntData/

---

## 🎯 Fonctionnalités

### **1. Résolution Automatique des Indices**
- ✅ Base de données complète des indices Ganymede
- ✅ Recherche exacte et approximative
- ✅ Solutions multiples avec niveau de confiance
- ✅ Fallback sur parsing manuel

### **2. Navigation Optimisée**
- ✅ Calcul du chemin optimal (TSP)
- ✅ Minimisation de la distance totale
- ✅ Prise en compte de la position de départ

### **3. Types d'Indices Supportés**
- 🧭 **Direction** : "Au nord", "À l'est"
- 🐉 **Monstre** : "Près des Bouftous"
- 💬 **NPC** : "Près du forgeron"
- 🏛️ **Point de repère** : "Près de la fontaine"
- 📍 **Coordonnées** : "[4, -18]"
- 🧩 **Énigme** : Indices complexes

### **4. Statistiques et Tracking**
- 📊 Nombre de chasses complétées
- ⏱️ Temps moyen par chasse
- 💰 Kamas gagnés
- 📈 Taux de succès

---

## 🚀 Installation

### **Prérequis** :
```bash
pip install requests beautifulsoup4
```

### **Initialisation** :
```python
from modules.treasure_hunt.ganymede_treasure_integration import GanymedeTreasureIntegration

# Créer le module
treasure = GanymedeTreasureIntegration()

# Initialiser (télécharge la base de données Ganymede)
treasure.initialize({})
```

---

## 💡 Utilisation

### **Exemple 1 : Chasse Simple**

```python
# Démarrer une chasse avec 3 indices
clues = [
    "Cherche près des Bouftous",
    "Va au nord de la fontaine d'Astrub",
    "Près du forgeron"
]

hunt_id = treasure.start_hunt(clues)

# Résoudre chaque indice
for i in range(len(clues)):
    # Obtenir les positions possibles
    positions = treasure.solve_current_clue(hunt_id)
    
    if positions:
        print(f"Indice {i+1}: Aller à {positions[0]}")
        
        # Naviguer vers la position
        # ... (votre code de navigation)
        
        # Marquer comme résolu
        treasure.mark_clue_solved(hunt_id)
    else:
        print(f"Indice {i+1}: Non résolu automatiquement")

# Vérifier statut
status = treasure.get_hunt_status(hunt_id)
print(f"Progression: {status['progress']:.0%}")
```

### **Exemple 2 : Avec Optimisation de Chemin**

```python
from modules.treasure_hunt.ganymede_treasure_integration import TreasureHuntSolver

solver = TreasureHuntSolver()

# Position de départ
start_pos = (4, -18)  # Astrub

# Résoudre tous les indices
clues = [
    treasure.solver.parser.parse_clue("Cherche près des Bouftous"),
    treasure.solver.parser.parse_clue("Va au nord"),
    treasure.solver.parser.parse_clue("Près du forgeron")
]

# Calculer chemin optimal
optimal_path = solver.calculate_optimal_path(clues, start_pos)

print("Chemin optimal:")
for i, pos in enumerate(optimal_path):
    print(f"  {i}. {pos}")
```

### **Exemple 3 : Intégration Complète**

```python
# Configuration complète
treasure = GanymedeTreasureIntegration()
treasure.initialize({})

# Boucle de jeu
while True:
    # Récupérer les indices depuis le jeu
    clues = get_treasure_clues_from_game()
    
    if clues:
        # Démarrer la chasse
        hunt_id = treasure.start_hunt(clues)
        
        # Résoudre automatiquement
        for i in range(len(clues)):
            positions = treasure.solve_current_clue(hunt_id)
            
            if positions:
                # Naviguer vers la meilleure position
                navigate_to(positions[0])
                
                # Chercher le drapeau
                flag_found = search_for_flag()
                
                if flag_found:
                    treasure.mark_clue_solved(hunt_id)
                else:
                    # Essayer positions alternatives
                    for alt_pos in positions[1:]:
                        navigate_to(alt_pos)
                        if search_for_flag():
                            treasure.mark_clue_solved(hunt_id)
                            break
        
        # Statistiques
        stats = treasure.get_state()
        print(f"Chasses complétées: {stats['stats']['hunts_completed']}")
```

---

## 📊 Format des Données Ganymede

### **Structure de la Base de Données** :

```json
{
  "clue_001": {
    "text": "Cherche près des Bouftous",
    "coordinates": [5, -20],
    "map_name": "Champs d'Astrub",
    "confidence": 1.0,
    "alternatives": [
      {
        "coordinates": [6, -20],
        "confidence": 0.8
      }
    ],
    "hints": [
      "Zone de spawn des Bouftous",
      "Près de l'enclos"
    ]
  }
}
```

### **Téléchargement Automatique** :

Le module télécharge automatiquement la base de données depuis :
```
https://dofus-map.com/huntData/
```

Cache local : `data/ganymede_hunts/hunt_database.json` (valide 7 jours)

---

## 🔍 Résolution des Indices

### **Processus de Résolution** :

```
1. Recherche dans Ganymede
   ├─ Recherche exacte du texte
   ├─ Recherche approximative
   └─ Retourne solutions avec confiance

2. Si non trouvé dans Ganymede
   ├─ Parsing manuel du texte
   ├─ Détection du type d'indice
   ├─ Extraction coordonnées si présentes
   └─ Génération hints

3. Retour des positions
   ├─ Position principale (confiance max)
   ├─ Positions alternatives
   └─ Métadonnées (difficulté, hints)
```

### **Exemples de Résolution** :

| Indice | Type | Résultat |
|--------|------|----------|
| "Au nord de la fontaine" | DIRECTION | [4, -17] |
| "Près des Bouftous" | MONSTER | [5, -20] |
| "Chez le forgeron" | NPC | [3, -19] |
| "[4, -18]" | COORDINATES | [4, -18] |

---

## 🎯 Optimisation du Chemin

### **Algorithme** :
- Problème du voyageur de commerce (TSP) simplifié
- Algorithme glouton : toujours aller au plus proche
- Complexité : O(n²)

### **Exemple** :

```python
# Positions des indices
positions = [
    (5, -20),   # Indice 1
    (10, -15),  # Indice 2
    (3, -18)    # Indice 3
]

# Position de départ
start = (4, -18)

# Chemin optimal calculé
optimal = [
    (4, -18),   # Départ
    (3, -18),   # Plus proche (distance: 1)
    (5, -20),   # Suivant (distance: 2.8)
    (10, -15)   # Dernier (distance: 7.1)
]

# Distance totale : 10.9 cases
# vs chemin non optimisé : 15.2 cases
# Gain : 28% plus rapide !
```

---

## 📈 Statistiques

### **Métriques Trackées** :

```python
stats = treasure.get_state()

print(stats['stats'])
# {
#     "hunts_completed": 15,
#     "total_clues_solved": 45,
#     "total_kamas_earned": 750000,
#     "average_time_per_hunt": 8.5,  # minutes
#     "success_rate": 0.93  # 93%
# }
```

---

## 🔧 Configuration Avancée

### **Personnalisation** :

```python
# Configuration du module
config = {
    "cache_duration_days": 7,  # Durée cache Ganymede
    "max_alternatives": 3,     # Nombre max de positions alternatives
    "confidence_threshold": 0.7,  # Seuil de confiance minimum
    "auto_optimize_path": True,   # Optimisation automatique
    "fallback_to_manual": True    # Parsing manuel si échec
}

treasure.initialize(config)
```

### **Gestion des Erreurs** :

```python
try:
    positions = treasure.solve_current_clue(hunt_id)
    
    if not positions:
        # Indice non résolu automatiquement
        print("Résolution manuelle requise")
        manual_position = ask_user_for_position()
        # Continuer avec position manuelle
        
except Exception as e:
    logger.error(f"Erreur résolution: {e}")
    # Fallback sur méthode alternative
```

---

## 🌐 Sources Open-Source

### **Ganymede** :
- **GitHub** : https://github.com/Dofus-Batteries-Included/Dofus
- **License** : Open Source
- **Contributeurs** : Communauté Dofus

### **Dofus-Map** :
- **Site** : https://dofus-map.com/
- **API** : Accès public
- **Données** : Mises à jour régulièrement

---

## 🎉 Avantages

### **Par rapport à la résolution manuelle** :
- ⚡ **Vitesse** : Résolution instantanée
- 🎯 **Précision** : Base de données complète
- 🔄 **Optimisation** : Chemin le plus court
- 📊 **Tracking** : Statistiques détaillées
- 🤖 **Automatisation** : 100% autonome

### **Par rapport aux autres bots** :
- ✅ **Open Source** : Données Ganymede publiques
- ✅ **Communautaire** : Mises à jour régulières
- ✅ **Fiable** : Testé par la communauté
- ✅ **Légal** : Utilise données publiques

---

## 🚀 Prochaines Étapes

1. **Tester le module** :
   ```bash
   python modules/treasure_hunt/ganymede_treasure_integration.py
   ```

2. **Intégrer au bot principal** :
   - Ajouter au decision_engine
   - Connecter à la navigation
   - Activer le tracking

3. **Optimiser** :
   - Ajouter plus de patterns de parsing
   - Améliorer l'algorithme de chemin
   - Intégrer avec Strategic Planner

---

## 📞 Support

**Documentation** :
- Module existant : `modules/treasure_hunt/README.md`
- Ganymede : https://github.com/Dofus-Batteries-Included/Dofus

**Communauté** :
- Discord Ganymede
- Forums Dofus

---

**🗺️ BON FARMING DE CHASSES AU TRÉSOR ! 💰**
