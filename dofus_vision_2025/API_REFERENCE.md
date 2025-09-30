# 🔌 API REFERENCE - DOFUS Unity World Model AI

**Version 2025.1.0** | **Documentation API Complète** | **Septembre 2025**

---

## 📋 Table des Matières

1. [Vue d'Ensemble](#-vue-densemble)
2. [Core Module API](#-core-module-api)
3. [Vision Engine API](#-vision-engine-api)
4. [Knowledge Base API](#-knowledge-base-api)
5. [Learning Engine API](#-learning-engine-api)
6. [Human Simulation API](#-human-simulation-api)
7. [Assistant Interface API](#-assistant-interface-api)
8. [Data Types](#-data-types)
9. [Error Handling](#-error-handling)
10. [Code Examples](#-code-examples)

---

## 🎯 Vue d'Ensemble

### Import Principal

```python
# Import des modules principaux
from core import (
    DofusCombatGridAnalyzer,     # Analyse grille de combat
    DofusWindowCapture,          # Capture d'écran
    DofusUnityInterfaceReader,   # Lecture interface OCR
    DofusKnowledgeBase,          # Base de connaissances
    AdaptiveLearningEngine,      # Moteur d'apprentissage
    AdvancedHumanSimulator       # Simulation humaine
)

# Import des types de données
from core.knowledge_base.knowledge_integration import (
    GameContext, DofusClass, KnowledgeQueryResult
)
from core.vision_engine.unity_interface_reader import (
    GameState, InterfaceElement
)
from core.learning_engine.adaptive_learning_engine import (
    LearningSession, ActionRecommendation
)
```

### Conventions API

- **Return Types** : Tous les méthodes retournent des objets typés
- **Error Handling** : Exceptions spécifiques avec messages détaillés
- **Async Support** : Méthodes lourdes supportent l'exécution asynchrone
- **Type Hints** : Annotations complètes Python 3.8+
- **Documentation** : Docstrings Google format

---

## 🧠 Core Module API

### Factory Functions

#### `get_knowledge_base() -> DofusKnowledgeBase`
```python
def get_knowledge_base() -> DofusKnowledgeBase:
    """
    Retourne l'instance singleton de la base de connaissances DOFUS.

    Returns:
        DofusKnowledgeBase: Instance de la base de connaissances

    Raises:
        DatabaseConnectionError: Si connexion échoue
        InitializationError: Si initialisation impossible
    """
```

#### `get_learning_engine(config: Optional[Dict] = None) -> AdaptiveLearningEngine`
```python
def get_learning_engine(config: Optional[Dict] = None) -> AdaptiveLearningEngine:
    """
    Crée une nouvelle instance du moteur d'apprentissage.

    Args:
        config: Configuration optionnelle du moteur

    Returns:
        AdaptiveLearningEngine: Instance du moteur d'apprentissage

    Example:
        engine = get_learning_engine({"learning_rate": 0.01})
    """
```

#### `get_human_simulator(profile: Optional[str] = None) -> AdvancedHumanSimulator`
```python
def get_human_simulator(profile: Optional[str] = None) -> AdvancedHumanSimulator:
    """
    Crée un simulateur de comportement humain.

    Args:
        profile: Nom du profil comportemental ("natural", "smooth", "jittery")

    Returns:
        AdvancedHumanSimulator: Instance du simulateur

    Example:
        simulator = get_human_simulator("natural")
    """
```

---

## 🔍 Vision Engine API

### DofusWindowCapture

#### `__init__(self, window_title: str = "Dofus")`
```python
def __init__(self, window_title: str = "Dofus"):
    """
    Initialise le système de capture pour DOFUS.

    Args:
        window_title: Titre de la fenêtre DOFUS à capturer
    """
```

#### `capture_screenshot(self) -> Optional[np.ndarray]`
```python
def capture_screenshot(self) -> Optional[np.ndarray]:
    """
    Capture une image de la fenêtre DOFUS active.

    Returns:
        np.ndarray: Image capturée en format BGR, None si échec

    Raises:
        WindowNotFoundError: Si fenêtre DOFUS introuvable
        CaptureError: Si capture échoue

    Example:
        capture = DofusWindowCapture()
        screenshot = capture.capture_screenshot()
        if screenshot is not None:
            cv2.imshow("DOFUS", screenshot)
    """
```

#### `get_window_info(self) -> Dict[str, int]`
```python
def get_window_info(self) -> Dict[str, int]:
    """
    Retourne les informations de la fenêtre DOFUS.

    Returns:
        Dict contenant: {"x", "y", "width", "height", "pid"}

    Example:
        info = capture.get_window_info()
        print(f"Fenêtre: {info['width']}x{info['height']}")
    """
```

### DofusUnityInterfaceReader

#### `__init__(self, ocr_languages: List[str] = ["fr", "en"])`
```python
def __init__(self, ocr_languages: List[str] = ["fr", "en"]):
    """
    Initialise le lecteur d'interface Unity.

    Args:
        ocr_languages: Langues pour la reconnaissance OCR
    """
```

#### `read_interface_text(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> str`
```python
def read_interface_text(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> str:
    """
    Lit le texte dans une image d'interface.

    Args:
        image: Image à analyser (format BGR)
        region: Zone à analyser (x, y, width, height), None pour toute l'image

    Returns:
        str: Texte détecté dans l'interface

    Example:
        reader = DofusUnityInterfaceReader()
        text = reader.read_interface_text(screenshot, (100, 100, 200, 50))
    """
```

#### `extract_game_state(self, image: np.ndarray) -> GameState`
```python
def extract_game_state(self, image: np.ndarray) -> GameState:
    """
    Extrait l'état complet du jeu depuis une capture d'écran.

    Args:
        image: Capture d'écran complète du jeu

    Returns:
        GameState: État du jeu analysé

    Raises:
        AnalysisError: Si analyse impossible

    Example:
        state = reader.extract_game_state(screenshot)
        print(f"PV: {state.player_hp}/{state.player_max_hp}")
    """
```

### DofusCombatGridAnalyzer

#### `analyze_combat_grid(self, image: np.ndarray) -> CombatGridAnalysis`
```python
def analyze_combat_grid(self, image: np.ndarray) -> CombatGridAnalysis:
    """
    Analyse la grille de combat tactique.

    Args:
        image: Image de la grille de combat

    Returns:
        CombatGridAnalysis: Analyse complète de la grille

    Example:
        analyzer = DofusCombatGridAnalyzer()
        analysis = analyzer.analyze_combat_grid(combat_screenshot)
        for entity in analysis.entities:
            print(f"Entité {entity.type} en {entity.position}")
    """
```

#### `calculate_movement_range(self, start_pos: Tuple[int, int], mp_available: int) -> List[Tuple[int, int]]`
```python
def calculate_movement_range(self, start_pos: Tuple[int, int], mp_available: int) -> List[Tuple[int, int]]:
    """
    Calcule les positions accessibles avec les PM disponibles.

    Args:
        start_pos: Position de départ (x, y)
        mp_available: Points de mouvement disponibles

    Returns:
        List[Tuple[int, int]]: Liste des positions accessibles

    Example:
        positions = analyzer.calculate_movement_range((5, 5), 3)
        print(f"{len(positions)} positions accessibles")
    """
```

---

## 🧠 Knowledge Base API

### DofusKnowledgeBase

#### `update_game_context(self, context: GameContext) -> None`
```python
def update_game_context(self, context: GameContext) -> None:
    """
    Met à jour le contexte de jeu pour les requêtes.

    Args:
        context: Nouveau contexte de jeu

    Example:
        kb = get_knowledge_base()
        context = GameContext(
            player_class=DofusClass.IOPS,
            player_level=150,
            current_server="Julith"
        )
        kb.update_game_context(context)
    """
```

#### `query_optimal_spells(self, target_type: str = "enemy", distance: Optional[int] = None) -> KnowledgeQueryResult`
```python
def query_optimal_spells(self, target_type: str = "enemy", distance: Optional[int] = None) -> KnowledgeQueryResult:
    """
    Requête des sorts optimaux selon le contexte.

    Args:
        target_type: Type de cible ("enemy", "ally", "self")
        distance: Distance à la cible (utilise contexte si None)

    Returns:
        KnowledgeQueryResult: Résultat avec sorts recommandés

    Example:
        result = kb.query_optimal_spells("enemy", 2)
        if result.success:
            for spell in result.data:
                print(f"Sort: {spell['name']}, Dégâts: {spell['damage']}")
    """
```

#### `query_monster_strategy(self, monster_name: str) -> KnowledgeQueryResult`
```python
def query_monster_strategy(self, monster_name: str) -> KnowledgeQueryResult:
    """
    Requête de stratégie contre un monstre spécifique.

    Args:
        monster_name: Nom du monstre

    Returns:
        KnowledgeQueryResult: Stratégie recommandée

    Example:
        result = kb.query_monster_strategy("Bouftou")
        if result.success:
            strategy = result.data
            print(f"Stratégie: {strategy['approach']}")
            print(f"Résistances: {strategy['resistances']}")
    """
```

#### `query_market_opportunities(self, server: Optional[str] = None) -> KnowledgeQueryResult`
```python
def query_market_opportunities(self, server: Optional[str] = None) -> KnowledgeQueryResult:
    """
    Identifie les opportunités de marché rentables.

    Args:
        server: Nom du serveur (utilise contexte si None)

    Returns:
        KnowledgeQueryResult: Opportunités de marché

    Example:
        result = kb.query_market_opportunities("Julith")
        for opportunity in result.data:
            print(f"Item: {opportunity['name']}, Profit: {opportunity['profit']}%")
    """
```

#### `update_knowledge_from_experience(self, experience_data: Dict[str, Any]) -> bool`
```python
def update_knowledge_from_experience(self, experience_data: Dict[str, Any]) -> bool:
    """
    Met à jour la base de connaissances avec de nouvelles expériences.

    Args:
        experience_data: Données d'expérience structurées

    Returns:
        bool: True si mise à jour réussie

    Example:
        experience = {
            "action": "spell_cast",
            "spell": "Pression",
            "target": "Bouftou",
            "outcome": "critical_hit",
            "damage": 150
        }
        updated = kb.update_knowledge_from_experience(experience)
    """
```

---

## 🎯 Learning Engine API

### AdaptiveLearningEngine

#### `start_learning_session(self, character_class: str, level: int, server: str) -> str`
```python
def start_learning_session(self, character_class: str, level: int, server: str) -> str:
    """
    Démarre une nouvelle session d'apprentissage.

    Args:
        character_class: Classe du personnage
        level: Niveau du personnage
        server: Serveur de jeu

    Returns:
        str: ID unique de la session

    Example:
        engine = get_learning_engine()
        session_id = engine.start_learning_session("IOPS", 150, "Julith")
        print(f"Session démarrée: {session_id}")
    """
```

#### `record_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any], context: Dict[str, Any]) -> None`
```python
def record_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any], context: Dict[str, Any]) -> None:
    """
    Enregistre le résultat d'une action pour apprentissage.

    Args:
        action: Description de l'action effectuée
        outcome: Résultat de l'action
        context: Contexte dans lequel l'action a été effectuée

    Example:
        action = {"type": "spell_cast", "spell": "Pression", "target": "enemy"}
        outcome = {"success": True, "damage": 120, "critical": False}
        context = {"player_hp": 90, "enemy_hp": 80, "distance": 2}
        engine.record_action_outcome(action, outcome, context)
    """
```

#### `get_recommended_action(self, current_context: Dict[str, Any]) -> Optional[ActionRecommendation]`
```python
def get_recommended_action(self, current_context: Dict[str, Any]) -> Optional[ActionRecommendation]:
    """
    Recommande une action basée sur l'apprentissage et le contexte.

    Args:
        current_context: Contexte actuel du jeu

    Returns:
        ActionRecommendation: Action recommandée avec confiance

    Example:
        context = {"in_combat": True, "available_ap": 6, "enemy_distance": 2}
        recommendation = engine.get_recommended_action(context)
        if recommendation:
            print(f"Action: {recommendation.action_type}")
            print(f"Confiance: {recommendation.confidence:.2f}")
    """
```

#### `get_learning_metrics(self) -> Dict[str, float]`
```python
def get_learning_metrics(self) -> Dict[str, float]:
    """
    Retourne les métriques d'apprentissage actuelles.

    Returns:
        Dict[str, float]: Métriques de performance

    Example:
        metrics = engine.get_learning_metrics()
        print(f"Taux de succès: {metrics['success_rate']:.2f}")
        print(f"Efficacité: {metrics['efficiency_score']:.2f}")
    """
```

#### `end_learning_session(self) -> LearningSession`
```python
def end_learning_session(self) -> LearningSession:
    """
    Termine la session d'apprentissage en cours.

    Returns:
        LearningSession: Résumé de la session terminée

    Example:
        session = engine.end_learning_session()
        print(f"Session terminée - Score: {session.efficiency_score}")
    """
```

---

## 🎭 Human Simulation API

### AdvancedHumanSimulator

#### `generate_mouse_movement(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]`
```python
def generate_mouse_movement(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Génère un mouvement de souris naturel entre deux points.

    Args:
        start: Position de départ (x, y)
        end: Position d'arrivée (x, y)

    Returns:
        List[Tuple[int, int]]: Points du mouvement

    Example:
        simulator = get_human_simulator("natural")
        movement = simulator.generate_mouse_movement((100, 100), (300, 200))
        for point in movement:
            # Déplacer la souris vers point
            move_mouse_to(point)
            time.sleep(0.001)
    """
```

#### `simulate_spell_casting_sequence(self, spell_name: str, target_pos: Tuple[int, int]) -> Dict[str, Any]`
```python
def simulate_spell_casting_sequence(self, spell_name: str, target_pos: Tuple[int, int]) -> Dict[str, Any]:
    """
    Simule une séquence de lancement de sort humaine.

    Args:
        spell_name: Nom du sort à lancer
        target_pos: Position de la cible

    Returns:
        Dict: Séquence d'actions avec timings

    Example:
        sequence = simulator.simulate_spell_casting_sequence("Pression", (250, 150))
        for step in sequence['steps']:
            print(f"Action: {step['action']}, Délai: {step['delay']}ms")
    """
```

#### `generate_keyboard_rhythm(self, keys: List[str]) -> List[Dict[str, float]]`
```python
def generate_keyboard_rhythm(self, keys: List[str]) -> List[Dict[str, float]]:
    """
    Génère un rythme de frappe naturel pour une séquence de touches.

    Args:
        keys: Liste des touches à presser

    Returns:
        List[Dict]: Timing pour chaque touche

    Example:
        rhythm = simulator.generate_keyboard_rhythm(["1", "2", "3"])
        for i, timing in enumerate(rhythm):
            key = keys[i]
            press_key(key)
            time.sleep(timing['delay'])
    """
```

#### `set_behavior_profile(self, profile: str) -> None`
```python
def set_behavior_profile(self, profile: str) -> None:
    """
    Change le profil comportemental du simulateur.

    Args:
        profile: Nom du profil ("natural", "smooth", "jittery", "nervous")

    Raises:
        ValueError: Si profil inconnu

    Example:
        simulator.set_behavior_profile("nervous")  # Mouvement plus erratique
    """
```

---

## 🎮 Assistant Interface API

### IntelligentAssistantUI

#### `__init__(self, config: Optional[AssistantConfig] = None)`
```python
def __init__(self, config: Optional[AssistantConfig] = None):
    """
    Initialise l'interface assistant intelligente.

    Args:
        config: Configuration de l'interface
    """
```

#### `start_gui(self) -> None`
```python
def start_gui(self) -> None:
    """
    Lance l'interface graphique principale.

    Example:
        assistant = IntelligentAssistantUI()
        assistant.start_gui()  # Bloque jusqu'à fermeture
    """
```

#### `register_module_callback(self, module_name: str, callback: Callable) -> None`
```python
def register_module_callback(self, module_name: str, callback: Callable) -> None:
    """
    Enregistre un callback pour un module spécifique.

    Args:
        module_name: Nom du module
        callback: Fonction de callback

    Example:
        def on_vision_update(data):
            print(f"Vision mise à jour: {data}")

        assistant.register_module_callback("vision", on_vision_update)
    """
```

---

## 📊 Data Types

### Core Data Types

#### `GameContext`
```python
@dataclass
class GameContext:
    """Contexte de jeu actuel pour requêtes contextuelles"""
    player_class: Optional[DofusClass] = None
    player_level: int = 200
    current_server: str = "Julith"
    current_map_id: Optional[int] = None
    available_ap: int = 6
    available_mp: int = 3
    distance_to_target: int = 2
    in_combat: bool = False
```

#### `KnowledgeQueryResult`
```python
@dataclass
class KnowledgeQueryResult:
    """Résultat d'une requête sur la base de connaissance"""
    query_type: str
    success: bool
    data: Any
    source_modules: List[str]
    confidence_score: float
    execution_time_ms: float
    suggestions: List[str] = None
```

#### `DofusClass` (Enum)
```python
class DofusClass(Enum):
    """Classes de personnages DOFUS"""
    IOPS = "Iops"
    CRAS = "Cras"
    ENIRIPSA = "Eniripsa"
    ENUTROFS = "Enutrofs"
    SRAMS = "Srams"
    XELORS = "Xelors"
    ECAFLIPS = "Ecaflips"
    SADIDAS = "Sadidas"
    SACRIEURS = "Sacrieurs"
    PANDAS = "Pandas"
    ROUBLARDS = "Roublards"
    ZOUBALS = "Zoubals"
```

### Vision Engine Types

#### `GameState`
```python
@dataclass
class GameState:
    """État complet du jeu extrait de l'interface"""
    player_hp: int
    player_max_hp: int
    player_ap: int
    player_mp: int
    player_position: Optional[Tuple[int, int]]
    in_combat: bool
    interface_elements: List[InterfaceElement]
    timestamp: float
```

#### `InterfaceElement`
```python
@dataclass
class InterfaceElement:
    """Élément d'interface détecté"""
    element_type: str
    position: Tuple[int, int, int, int]  # x, y, width, height
    text: str
    confidence: float
    clickable: bool = False
```

### Learning Engine Types

#### `ActionRecommendation`
```python
@dataclass
class ActionRecommendation:
    """Recommandation d'action du moteur d'apprentissage"""
    action_type: str
    action_data: Dict[str, Any]
    confidence: float
    expected_outcome: Dict[str, Any]
    alternative_actions: List[Dict[str, Any]]
```

#### `LearningSession`
```python
@dataclass
class LearningSession:
    """Session d'apprentissage terminée"""
    session_id: str
    character_class: str
    character_level: int
    server: str
    start_time: datetime
    end_time: datetime
    total_actions: int
    success_rate: float
    efficiency_score: float
```

---

## ⚠️ Error Handling

### Exceptions Personnalisées

#### `DofusVisionError`
```python
class DofusVisionError(Exception):
    """Exception de base pour tous les erreurs du système"""
    pass
```

#### `WindowNotFoundError`
```python
class WindowNotFoundError(DofusVisionError):
    """Exception levée quand la fenêtre DOFUS n'est pas trouvée"""
    pass
```

#### `DatabaseConnectionError`
```python
class DatabaseConnectionError(DofusVisionError):
    """Exception levée lors de problèmes de connexion à la base"""
    pass
```

#### `AnalysisError`
```python
class AnalysisError(DofusVisionError):
    """Exception levée lors d'échec d'analyse d'image"""
    pass
```

### Error Handling Patterns

```python
try:
    # Utilisation de l'API
    kb = get_knowledge_base()
    result = kb.query_optimal_spells()

except DatabaseConnectionError as e:
    logger.error(f"Problème base de données: {e}")
    # Fallback vers cache local

except AnalysisError as e:
    logger.warning(f"Analyse échouée: {e}")
    # Retry avec paramètres différents

except DofusVisionError as e:
    logger.error(f"Erreur système: {e}")
    # Arrêt propre du système
```

---

## 💡 Code Examples

### Exemple Complet : Bot de Combat Basique

```python
from core import (
    DofusWindowCapture, DofusUnityInterfaceReader,
    DofusCombatGridAnalyzer, DofusKnowledgeBase,
    AdaptiveLearningEngine, AdvancedHumanSimulator
)
from core.knowledge_base.knowledge_integration import GameContext, DofusClass
import time

def main():
    # Initialisation des modules
    capture = DofusWindowCapture("Dofus")
    reader = DofusUnityInterfaceReader()
    analyzer = DofusCombatGridAnalyzer()
    kb = get_knowledge_base()
    engine = get_learning_engine()
    simulator = get_human_simulator("natural")

    # Configuration du contexte
    context = GameContext(
        player_class=DofusClass.IOPS,
        player_level=150,
        current_server="Julith",
        available_ap=6,
        available_mp=3
    )
    kb.update_game_context(context)

    # Démarrage session d'apprentissage
    session_id = engine.start_learning_session("IOPS", 150, "Julith")

    try:
        while True:
            # Capture et analyse
            screenshot = capture.capture_screenshot()
            if screenshot is None:
                time.sleep(0.1)
                continue

            game_state = reader.extract_game_state(screenshot)

            if game_state.in_combat:
                # Mode combat
                combat_analysis = analyzer.analyze_combat_grid(screenshot)

                # Requête stratégie optimale
                spells_result = kb.query_optimal_spells("enemy")

                if spells_result.success and spells_result.data:
                    # Sélection du meilleur sort
                    best_spell = spells_result.data[0]

                    # Simulation du lancement de sort
                    spell_sequence = simulator.simulate_spell_casting_sequence(
                        best_spell['name'],
                        combat_analysis.nearest_enemy_position
                    )

                    # Exécution de la séquence
                    for step in spell_sequence['steps']:
                        # Simulation de l'action
                        print(f"Action: {step['action']}")
                        time.sleep(step['delay'] / 1000.0)

                    # Enregistrement pour apprentissage
                    action = {
                        "type": "spell_cast",
                        "spell": best_spell['name'],
                        "target": "enemy"
                    }
                    outcome = {
                        "success": True,  # À déterminer par analyse post-action
                        "execution_time": spell_sequence['total_duration']
                    }
                    current_context = {
                        "in_combat": True,
                        "available_ap": game_state.player_ap,
                        "enemy_distance": combat_analysis.distance_to_nearest_enemy
                    }

                    engine.record_action_outcome(action, outcome, current_context)

            time.sleep(0.5)  # Délai entre analyses

    except KeyboardInterrupt:
        print("Arrêt du bot...")
    finally:
        # Fin de session d'apprentissage
        session = engine.end_learning_session()
        print(f"Session terminée - Score: {session.efficiency_score:.2f}")

if __name__ == "__main__":
    main()
```

### Exemple : Interface Simple

```python
from assistant_interface.intelligent_assistant import IntelligentAssistantUI, AssistantConfig

def create_simple_interface():
    """Crée une interface assistant simple"""

    # Configuration personnalisée
    config = AssistantConfig(
        window_title="DOFUS AI Assistant",
        theme="dark",
        auto_start_modules=True,
        log_level="INFO"
    )

    # Création et lancement
    assistant = IntelligentAssistantUI(config)

    # Callbacks personnalisés
    def on_spell_cast(spell_data):
        print(f"Sort lancé: {spell_data['name']}")

    def on_combat_analysis(analysis):
        print(f"Combat analysé: {len(analysis.entities)} entités")

    assistant.register_module_callback("spells", on_spell_cast)
    assistant.register_module_callback("combat", on_combat_analysis)

    # Lancement de l'interface
    assistant.start_gui()

if __name__ == "__main__":
    create_simple_interface()
```

### Exemple : Analyse de Marché

```python
from core import get_knowledge_base
from core.knowledge_base.knowledge_integration import GameContext

def analyze_market_opportunities():
    """Analyse les opportunités de marché sur différents serveurs"""

    kb = get_knowledge_base()
    servers = ["Julith", "Draconiros", "Ombre", "Rushu"]

    all_opportunities = {}

    for server in servers:
        print(f"\nAnalyse du serveur {server}...")

        # Configuration du contexte pour le serveur
        context = GameContext(current_server=server)
        kb.update_game_context(context)

        # Requête des opportunités
        result = kb.query_market_opportunities()

        if result.success:
            opportunities = result.data
            all_opportunities[server] = opportunities

            print(f"  Trouvé {len(opportunities)} opportunités")

            # Affichage des 3 meilleures
            for i, opp in enumerate(opportunities[:3]):
                print(f"  {i+1}. {opp['item_name']}: {opp['profit_percent']:.1f}% profit")
        else:
            print(f"  Erreur: {result.data}")

    # Analyse comparative
    print("\n=== RÉSUMÉ COMPARATIF ===")
    for server, opportunities in all_opportunities.items():
        if opportunities:
            best_profit = max(opp['profit_percent'] for opp in opportunities)
            print(f"{server}: Meilleur profit {best_profit:.1f}%")

if __name__ == "__main__":
    analyze_market_opportunities()
```

---

## 🔗 Liens Utiles

- **[Guide d'Installation](INSTALLATION.md)** - Configuration complète
- **[Guide Utilisateur](USER_GUIDE.md)** - Utilisation pratique
- **[Guide Développeur](DEVELOPER_GUIDE.md)** - Contribution et développement
- **[Architecture](ARCHITECTURE.md)** - Documentation technique
- **[Exemples](examples/)** - Code d'exemple complet

---

*API Reference maintenue par Claude Code - AI Development Specialist*
*Version 2025.1.0 - Septembre 2025*
*Dernière mise à jour : API complète et validée*