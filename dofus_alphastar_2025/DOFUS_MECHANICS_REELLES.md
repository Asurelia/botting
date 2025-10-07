# 🎮 DOFUS - MÉCANIQUES RÉELLES DU JEU

**IMPORTANT:** DOFUS ne fonctionne PAS comme les autres MMORPGs !

---

## ❌ ERREURS À CORRIGER

### 1. **Combat N'est PAS Automatique**

**❌ FAUX (comme WoW, FF14, etc.):**
```
Si monstre détecté → On est en combat
```

**✅ VRAI (DOFUS):**
```
1. Voir des monstres à l'écran (toujours visibles)
2. CLIQUER sur le groupe de monstres
3. Écran de placement apparaît
4. ALORS combat commence
```

**Conséquence:**
- Le bot ne doit PAS détecter "in_combat" juste en voyant des monstres
- Il faut **activement cliquer** sur le groupe pour initier le combat

### 2. **Noms/Niveaux des Monstres Cachés**

**❌ FAUX:**
```python
# Vision détecte directement les noms avec OCR
monster_name = ocr.detect("Bouftou Niveau 12")
```

**✅ VRAI:**
```python
# Les noms sont CACHÉS par défaut
# Il faut appuyer sur une TOUCHE (TAB ou ALT par défaut)
# PUIS faire l'OCR
```

**Actions requises:**
1. Appuyer sur TAB/ALT pour afficher les infos
2. Attendre 100-200ms (temps d'affichage)
3. Faire l'OCR
4. Relâcher la touche

### 3. **Maps Imbriquées (Intérieurs)**

**❌ FAUX:**
```
Ganymède = 1 grande carte avec tout visible
```

**✅ VRAI:**
```
Ganymède Centre (carte extérieure)
  ├─ Maison 1 (carte intérieure séparée)
  ├─ Maison 2 (carte intérieure séparée)
  ├─ Banque (carte intérieure)
  └─ Taverne (carte intérieure)

Forêt de Ganymède
  ├─ Grotte 1 (carte intérieure)
  ├─ Grotte 2 (carte intérieure)
  └─ Cachette secrète (carte intérieure)
```

**Conséquence:**
- Navigation doit gérer les **transitions** entre cartes
- Reconnaître quand on est dans un intérieur vs extérieur
- Savoir sortir des intérieurs (porte, flèche de sortie)

### 4. **Calibration N'a Rien Montré**

**Problèmes identifiés:**
- Pas de screenshots sauvegardés
- Pas de logs détaillés
- Pas d'analyse des raccourcis dans les options
- Mode silencieux (pas de feedback visuel)

**Ce qui devrait se passer:**
```python
# Calibration DOIT:
1. Prendre screenshot de la fenêtre Dofus
2. Sauvegarder: calibration_step_1_window.png
3. Détecter les éléments UI (HP bar, PA/PM, inventaire)
4. Sauvegarder: calibration_step_2_ui_detection.png
5. Analyser les raccourcis (Échap → Options → Raccourcis)
6. Sauvegarder: calibration_step_3_shortcuts.png
7. Tester chaque détection
8. Sauvegarder: calibration_step_4_validation.png
9. Créer rapport JSON avec toutes les positions
```

---

## ✅ MÉCANIQUES CORRECTES

### Système de Combat

```
ÉTAPE 1: DÉTECTION
├─ Voir des groupes de monstres (silhouettes colorées)
├─ Appuyer sur TAB pour voir noms/niveaux
└─ Relâcher TAB

ÉTAPE 2: ENGAGEMENT
├─ Cliquer sur le groupe de monstres
├─ Écran de placement apparaît (grille de combat)
├─ Choisir position de départ (ou auto si timeout)
└─ Combat COMMENCE

ÉTAPE 3: COMBAT AU TOUR PAR TOUR
├─ Mon tour → Barre PA/PM visible
│   ├─ Utiliser sorts (touche 1-8)
│   ├─ Se déplacer (clic sur case)
│   └─ Passer tour (Espace)
├─ Tour des ennemis → Attendre
└─ Répéter jusqu'à victoire/défaite

ÉTAPE 4: FIN DE COMBAT
├─ Écran de loot apparaît
├─ Cliquer sur items ou "Tout prendre"
└─ Fermer écran de loot (Échap ou bouton)
```

### Système de Navigation

```
NAVIGATION ENTRE CARTES
├─ Carte actuelle: Ganymède Centre
├─ Cliquer sur bord de carte (flèches directionnelles)
│   ├─ Gauche → Carte voisine Ouest
│   ├─ Droite → Carte voisine Est
│   ├─ Haut → Carte voisine Nord
│   └─ Bas → Carte voisine Sud
└─ Transition (chargement 1-3 secondes)

NAVIGATION INTÉRIEUR/EXTÉRIEUR
├─ Cliquer sur porte/entrée → Entre dans bâtiment
├─ Nouvelle carte (intérieur) charge
├─ Cliquer sur sortie → Retourne à l'extérieur
└─ Carte extérieure recharge

ZAAP (Téléportation)
├─ Cliquer sur Zaap (grand cristal bleu)
├─ Interface de téléportation s'ouvre
├─ Sélectionner destination
├─ Confirmer (coûte kamas)
└─ Téléportation instantanée
```

### Système de Détection Visuelle

**Ce qu'on voit SANS interaction:**
- Terrain de la carte
- Personnage (cercle bleu sous les pieds)
- Groupes de monstres (silhouettes + cercle rouge)
- NPCs (silhouettes + cercle jaune)
- Ressources (objets au sol)
- UI (HP, PA/PM, inventaire, etc.)

**Ce qu'on voit UNIQUEMENT avec TAB/ALT:**
- Noms des monstres
- Niveaux des monstres
- Noms des NPCs
- Noms des ressources

**Ce qu'on voit UNIQUEMENT en combat:**
- Grille de combat (cases blanches/rouges/bleues)
- Timeline des tours
- Barres HP des ennemis
- PA/PM disponibles (gros chiffres)

---

## 🔧 CORRECTIONS À IMPLÉMENTER

### 1. Système de Détection de Combat RÉALISTE

**Nouveau fichier:** `core/vision/dofus_combat_detector.py`

```python
class DofusCombatDetector:
    """Détection spécifique à DOFUS"""

    def is_in_combat(self, frame):
        """
        Détecte si on est EN COMBAT (pas juste si des monstres sont visibles)

        Critères DOFUS:
        1. Grille de combat visible (cases blanches)
        2. Timeline visible en haut
        3. PA/PM gros chiffres visibles
        4. OU couleur de fond différente (teinte bleue en combat)
        """
        # Chercher la grille de combat (cases blanches/bleues)
        combat_grid = self.detect_combat_grid(frame)

        # Chercher la timeline en haut
        timeline = self.detect_timeline(frame)

        # Chercher les gros PA/PM
        large_papm = self.detect_large_papm(frame)

        # Au moins 2 des 3 critères = en combat
        combat_indicators = sum([
            combat_grid is not None,
            timeline is not None,
            large_papm is not None
        ])

        return combat_indicators >= 2

    def detect_monster_groups(self, frame, show_names=False):
        """
        Détecte les groupes de monstres

        Args:
            show_names: Si True, appuie sur TAB avant détection
        """
        if show_names:
            # Appuyer sur TAB
            pyautogui.keyDown('tab')
            time.sleep(0.15)  # Attendre affichage

        # Détecter cercles rouges (groupes de monstres)
        monster_groups = self.detect_red_circles(frame)

        if show_names:
            # Faire OCR des noms maintenant qu'ils sont visibles
            for group in monster_groups:
                name_region = self.get_name_region_above(group)
                name = self.ocr.detect(name_region)
                group['name'] = name

            # Relâcher TAB
            pyautogui.keyUp('tab')

        return monster_groups
```

### 2. Système de Navigation Réaliste

**Nouveau fichier:** `core/navigation_system/dofus_map_navigator.py`

```python
class DofusMapNavigator:
    """Navigation spécifique à DOFUS"""

    def __init__(self):
        self.current_map_id = None
        self.is_interior = False  # Intérieur ou extérieur?
        self.map_history = []  # Historique des cartes visitées

    def detect_current_map(self, frame):
        """
        Détecte la carte actuelle

        Méthodes:
        1. Lire les coordonnées en haut à droite [X, Y]
        2. Comparer avec screenshots de référence
        3. Analyser mini-map
        """
        # Lire coordonnées
        coords = self.read_map_coordinates(frame)

        # Détecter si intérieur (pas de coordonnées visibles)
        if coords is None:
            self.is_interior = True
        else:
            self.is_interior = False

        return coords

    def navigate_to_adjacent_map(self, direction):
        """
        Navigue vers une carte adjacente

        Args:
            direction: 'left', 'right', 'top', 'bottom'
        """
        # Positions des flèches sur les bords
        arrow_positions = {
            'left': (50, 400),    # Milieu bord gauche
            'right': (1550, 400), # Milieu bord droit
            'top': (800, 50),     # Milieu bord haut
            'bottom': (800, 750)  # Milieu bord bas
        }

        click_pos = arrow_positions.get(direction)
        if click_pos:
            pyautogui.click(click_pos)
            time.sleep(2.0)  # Attendre transition
            self.map_history.append(self.current_map_id)

    def enter_building(self, door_position):
        """Entre dans un bâtiment (carte intérieure)"""
        pyautogui.click(door_position)
        time.sleep(2.0)  # Attendre chargement
        self.is_interior = True

    def exit_building(self):
        """Sort d'un bâtiment"""
        # Chercher la sortie (flèche spéciale ou porte)
        exit_pos = self.detect_exit_arrow()
        if exit_pos:
            pyautogui.click(exit_pos)
            time.sleep(2.0)
            self.is_interior = False
```

### 3. Calibration Complète et Visible

**Nouveau fichier:** `calibrate_dofus_complete.py`

```python
import cv2
import pyautogui
from pathlib import Path
import time

class DofusCalibrator:
    """Calibration complète avec feedback visuel"""

    def __init__(self):
        self.output_dir = Path("calibration_output")
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}

    def calibrate(self):
        """Calibration complète"""
        print("=== CALIBRATION DOFUS ===")
        print("Assurez-vous que Dofus est ouvert en mode fenêtré\n")
        input("Appuyez sur Entrée pour continuer...")

        # 1. Détecter fenêtre
        print("\n[1/8] Détection de la fenêtre Dofus...")
        window = self.detect_dofus_window()
        self.save_screenshot(window, "01_window_detected.png")
        print(f"✓ Fenêtre trouvée: {window}")

        # 2. Détecter UI
        print("\n[2/8] Détection des éléments UI...")
        ui_elements = self.detect_ui_elements(window)
        self.save_annotated_screenshot(window, ui_elements, "02_ui_detected.png")
        print(f"✓ {len(ui_elements)} éléments UI détectés")

        # 3. Tester TAB
        print("\n[3/8] Test de la touche TAB (affichage noms)...")
        self.test_tab_key(window)
        print("✓ TAB fonctionne")

        # 4. Analyser raccourcis
        print("\n[4/8] Analyse des raccourcis...")
        shortcuts = self.analyze_shortcuts(window)
        print(f"✓ {len(shortcuts)} raccourcis trouvés")

        # 5. Détecter combat
        print("\n[5/8] Détection du système de combat...")
        combat_system = self.detect_combat_system(window)
        print("✓ Système de combat analysé")

        # 6. Détecter navigation
        print("\n[6/8] Détection du système de navigation...")
        nav_system = self.detect_navigation_system(window)
        print("✓ Système de navigation analysé")

        # 7. Créer rapport
        print("\n[7/8] Création du rapport...")
        report = self.create_report()
        self.save_report(report, "calibration_report.json")
        print("✓ Rapport créé")

        # 8. Validation
        print("\n[8/8] Validation finale...")
        validation = self.validate_calibration()
        print(f"✓ Validation: {validation['score']}/100")

        print(f"\n✅ CALIBRATION TERMINÉE!")
        print(f"Screenshots sauvegardés dans: {self.output_dir}")
        print(f"Rapport disponible dans: calibration_report.json")

        return report

    def save_screenshot(self, window, filename):
        """Sauvegarde un screenshot annoté"""
        screenshot = pyautogui.screenshot(region=window)
        filepath = self.output_dir / filename
        screenshot.save(filepath)
        print(f"   → Screenshot: {filepath}")

    def test_tab_key(self, window):
        """Teste si TAB affiche les noms"""
        # Screenshot avant
        before = pyautogui.screenshot(region=window)
        self.save_screenshot(window, "03a_before_tab.png")

        # Appuyer TAB
        pyautogui.keyDown('tab')
        time.sleep(0.2)

        # Screenshot pendant
        during = pyautogui.screenshot(region=window)
        self.save_screenshot(window, "03b_during_tab.png")

        # Relâcher TAB
        pyautogui.keyUp('tab')
        time.sleep(0.1)

        # Screenshot après
        after = pyautogui.screenshot(region=window)
        self.save_screenshot(window, "03c_after_tab.png")

        # Comparer pour voir si quelque chose a changé
        # (Plus de texte visible pendant TAB)
```

---

## 🎯 ACTIONS IMMÉDIATES

### Priorité 1: Corriger Détection Combat
- [ ] Créer `dofus_combat_detector.py`
- [ ] Détecter grille de combat (pas juste monstres)
- [ ] Tester avec Dofus réel

### Priorité 2: Corriger Détection Monstres
- [ ] Implémenter système TAB pour noms
- [ ] Détecter cercles rouges (groupes)
- [ ] Tester OCR avec TAB pressé

### Priorité 3: Calibration Visuelle
- [ ] Créer `calibrate_dofus_complete.py`
- [ ] Sauvegarder TOUS les screenshots
- [ ] Créer rapport JSON détaillé
- [ ] Afficher progression en temps réel

### Priorité 4: Navigation Intérieurs
- [ ] Détecter si carte intérieure
- [ ] Trouver sorties automatiquement
- [ ] Gérer transitions

---

## 📝 NOTES IMPORTANTES

**Le bot ACTUEL a ces problèmes:**
1. ❌ Pense être en combat quand il voit juste des monstres
2. ❌ Ne sait pas appuyer sur TAB pour voir les noms
3. ❌ Ne gère pas les cartes intérieures
4. ❌ Calibration silencieuse (aucun feedback)

**Le bot CORRIGÉ devra:**
1. ✅ Détecter combat = grille visible + timeline + PA/PM gros
2. ✅ Appuyer sur TAB → OCR → Relâcher TAB
3. ✅ Gérer intérieurs (détecter + sortir)
4. ✅ Calibration avec screenshots + rapport détaillé

---

**Créé le 30 Janvier 2025**
*Basé sur les vraies mécaniques de DOFUS 2.0+*
