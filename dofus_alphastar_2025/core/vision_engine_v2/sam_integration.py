"""
SAM 2 Integration - Segment Anything Model pour analyse DOFUS
Détection et segmentation des éléments de jeu avec optimisations AMD
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

# SAM 2 imports
try:
    from ultralytics import SAM
    HAS_SAM2 = True
except ImportError:
    try:
        # Alternative import method
        import segment_anything_2 as sam2
        HAS_SAM2 = True
    except ImportError:
        HAS_SAM2 = False

from config import config, get_device

logger = logging.getLogger(__name__)

@dataclass
class SAMSegment:
    """Segment détecté par SAM"""
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: float
    confidence: float
    category: str = "unknown"
    center: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        """Calcule le centre du segment"""
        x, y, w, h = self.bbox
        self.center = (x + w // 2, y + h // 2)

@dataclass
class DofusUIElement:
    """Élément d'interface DOFUS détecté"""
    element_type: str  # "health_bar", "mana_bar", "spell_button", etc.
    segment: SAMSegment
    value: Optional[float] = None  # Valeur si applicable (HP%, MP%)
    text_content: Optional[str] = None  # Texte si applicable
    clickable: bool = False
    hotkey: Optional[str] = None

@dataclass
class DofusEntity:
    """Entité de jeu détectée (joueur, monstre, PNJ)"""
    entity_type: str  # "player", "monster", "npc", "item"
    segment: SAMSegment
    name: Optional[str] = None
    level: Optional[int] = None
    health_percentage: Optional[float] = None
    is_enemy: bool = False
    is_ally: bool = False
    position_cell: Optional[int] = None

class SAMProcessor:
    """Processeur SAM 2 optimisé pour AMD et DOFUS"""

    def __init__(self, model_size: str = "sam2_hiera_large"):
        self.model_size = model_size
        self.device = get_device()

        # SAM model
        self.model = None
        self.is_loaded = False

        # Cache pour optimisations
        self._image_cache = {}
        self._segmentation_cache = {}
        self._cache_max_size = 10

        # Performance metrics
        self.processing_times = []
        self.total_processed = 0

        # Load model
        self._load_model()

        logger.info(f"SAM Processor initialisé avec {model_size}")

    def _load_model(self):
        """Charge le modèle SAM 2"""
        if not HAS_SAM2:
            logger.error("SAM 2 non disponible. Installez: pip install ultralytics")
            return

        try:
            # Méthode Ultralytics (recommandée)
            model_path = config.vision.sam_checkpoint_path

            if Path(model_path).exists():
                self.model = SAM(model_path)
                logger.info(f"SAM 2 chargé depuis: {model_path}")
            else:
                # Auto-download
                self.model = SAM("sam2_hiera_l.pt")
                logger.info("SAM 2 téléchargé automatiquement")

            # Configuration device
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)

            self.is_loaded = True

        except Exception as e:
            logger.error(f"Erreur chargement SAM 2: {e}")
            self.is_loaded = False

    def segment_image(self,
                     image: Union[np.ndarray, Image.Image, str],
                     prompts: Optional[List[Dict]] = None,
                     everything: bool = False) -> List[SAMSegment]:
        """
        Segmente une image avec SAM 2

        Args:
            image: Image à segmenter
            prompts: Points/boîtes de prompt [{"type": "point", "data": [x, y]}]
            everything: Si True, segmente tout automatiquement
        """
        if not self.is_loaded:
            logger.warning("SAM 2 non chargé")
            return []

        start_time = time.time()

        # Préparation image
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Cache check
        image_hash = hash(image.tobytes())
        cache_key = (image_hash, str(prompts), everything)

        if cache_key in self._segmentation_cache:
            logger.debug("Utilisation cache segmentation")
            return self._segmentation_cache[cache_key]

        segments = []

        try:
            with torch.no_grad():
                if everything:
                    # Segment everything
                    results = self.model(image, verbose=False)

                    if results and len(results) > 0:
                        result = results[0]

                        if hasattr(result, 'masks') and result.masks is not None:
                            masks = result.masks.data.cpu().numpy()
                            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else None
                            confidences = result.boxes.conf.cpu().numpy() if result.boxes else None

                            for i, mask in enumerate(masks):
                                # Convertir mask en format correct
                                mask_uint8 = (mask * 255).astype(np.uint8)

                                # Calculer bbox si pas disponible
                                if boxes is not None and i < len(boxes):
                                    x1, y1, x2, y2 = boxes[i].astype(int)
                                    bbox = (x1, y1, x2 - x1, y2 - y1)
                                else:
                                    # Calculer bbox depuis le mask
                                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if contours:
                                        x, y, w, h = cv2.boundingRect(contours[0])
                                        bbox = (x, y, w, h)
                                    else:
                                        continue

                                confidence = confidences[i] if confidences is not None and i < len(confidences) else 0.5
                                area = np.sum(mask > 0)

                                segment = SAMSegment(
                                    mask=mask_uint8,
                                    bbox=bbox,
                                    area=area,
                                    confidence=confidence
                                )
                                segments.append(segment)

                else:
                    # Prompts-based segmentation
                    if prompts:
                        for prompt in prompts:
                            if prompt["type"] == "point":
                                point = prompt["data"]
                                # Implementation point prompt
                                results = self.model(image, points=point, verbose=False)

                                if results and len(results) > 0:
                                    # Process results similar to above
                                    pass

        except Exception as e:
            logger.error(f"Erreur segmentation SAM: {e}")

        # Cache result
        if len(self._segmentation_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._segmentation_cache))
            del self._segmentation_cache[oldest_key]

        self._segmentation_cache[cache_key] = segments

        # Performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.total_processed += 1

        logger.debug(f"SAM segmentation: {len(segments)} segments en {processing_time:.3f}s")

        return segments

    def filter_segments_by_area(self,
                               segments: List[SAMSegment],
                               min_area: int = 100,
                               max_area: Optional[int] = None) -> List[SAMSegment]:
        """Filtre les segments par aire"""
        filtered = []

        for segment in segments:
            if segment.area < min_area:
                continue
            if max_area and segment.area > max_area:
                continue
            filtered.append(segment)

        return filtered

    def filter_segments_by_confidence(self,
                                    segments: List[SAMSegment],
                                    min_confidence: float = 0.5) -> List[SAMSegment]:
        """Filtre les segments par confiance"""
        return [s for s in segments if s.confidence >= min_confidence]

    def get_performance_metrics(self) -> Dict[str, float]:
        """Retourne les métriques de performance"""
        if not self.processing_times:
            return {"avg_time": 0.0, "total_processed": 0}

        return {
            "avg_processing_time": np.mean(self.processing_times),
            "min_processing_time": np.min(self.processing_times),
            "max_processing_time": np.max(self.processing_times),
            "total_processed": self.total_processed,
            "cache_hits": len(self._segmentation_cache)
        }

class DofusSAMAnalyzer:
    """Analyseur SAM spécialisé pour DOFUS Unity"""

    def __init__(self):
        self.sam_processor = SAMProcessor()

        # Templates pour reconnaissance d'éléments DOFUS
        self.ui_templates = self._load_ui_templates()
        self.entity_classifiers = self._setup_entity_classifiers()

        # Zones d'intérêt DOFUS (coordonnées relatives)
        self.ui_regions = {
            "health_bar": (0.02, 0.02, 0.25, 0.05),     # Top-left
            "mana_bar": (0.02, 0.08, 0.25, 0.05),       # Below health
            "action_points": (0.02, 0.14, 0.15, 0.05),  # Below mana
            "spell_bar": (0.3, 0.85, 0.4, 0.12),        # Bottom center
            "chat": (0.02, 0.6, 0.35, 0.3),             # Left side
            "minimap": (0.75, 0.02, 0.23, 0.2),         # Top-right
            "combat_grid": (0.25, 0.2, 0.5, 0.6),       # Center
            "inventory": (0.6, 0.3, 0.35, 0.5),         # Right side (when open)
        }

        logger.info("DOFUS SAM Analyzer initialisé")

    def _load_ui_templates(self) -> Dict[str, Any]:
        """Charge les templates d'éléments UI DOFUS"""
        # Dans une vraie implémentation, on chargerait des templates pré-entraînés
        return {
            "health_bar_pattern": None,
            "spell_button_pattern": None,
            "monster_nameplate_pattern": None,
            "damage_number_pattern": None
        }

    def _setup_entity_classifiers(self) -> Dict[str, Any]:
        """Configure les classifieurs d'entités"""
        # Classifieurs basés sur forme/couleur/taille
        return {
            "player_classifier": {
                "min_area": 1000,
                "color_range": {"blue": (100, 150, 200)},
                "shape_aspect_ratio": (0.5, 2.0)
            },
            "monster_classifier": {
                "min_area": 800,
                "color_range": {"red": (200, 100, 100)},
                "has_nameplate": True
            },
            "npc_classifier": {
                "min_area": 600,
                "color_range": {"green": (100, 200, 100)},
                "stationary": True
            }
        }

    def analyze_screenshot(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyse complète d'un screenshot DOFUS"""

        analysis_start = time.time()

        # Segmentation SAM complète
        all_segments = self.sam_processor.segment_image(image, everything=True)

        # Filtrer par qualité
        quality_segments = self.sam_processor.filter_segments_by_confidence(
            all_segments, min_confidence=0.6
        )
        quality_segments = self.sam_processor.filter_segments_by_area(
            quality_segments, min_area=50, max_area=50000
        )

        # Analyser par type d'élément
        ui_elements = self._analyze_ui_elements(image, quality_segments)
        entities = self._analyze_entities(image, quality_segments)
        combat_info = self._analyze_combat_state(image, quality_segments)

        analysis_time = time.time() - analysis_start

        return {
            "ui_elements": ui_elements,
            "entities": entities,
            "combat_info": combat_info,
            "total_segments": len(all_segments),
            "quality_segments": len(quality_segments),
            "analysis_time": analysis_time,
            "timestamp": time.time()
        }

    def _analyze_ui_elements(self,
                           image: np.ndarray,
                           segments: List[SAMSegment]) -> List[DofusUIElement]:
        """Analyse les éléments d'interface utilisateur"""

        ui_elements = []
        h, w = image.shape[:2]

        for region_name, (rx, ry, rw, rh) in self.ui_regions.items():
            # Convertir coordonnées relatives en absolues
            x1, y1 = int(rx * w), int(ry * h)
            x2, y2 = int((rx + rw) * w), int((ry + rh) * h)

            # Trouver segments dans cette région
            region_segments = []
            for segment in segments:
                seg_x, seg_y, seg_w, seg_h = segment.bbox
                seg_center_x = seg_x + seg_w // 2
                seg_center_y = seg_y + seg_h // 2

                if x1 <= seg_center_x <= x2 and y1 <= seg_center_y <= y2:
                    region_segments.append(segment)

            # Analyser segments de la région
            for segment in region_segments:
                ui_element = self._classify_ui_element(
                    image, segment, region_name
                )
                if ui_element:
                    ui_elements.append(ui_element)

        return ui_elements

    def _classify_ui_element(self,
                           image: np.ndarray,
                           segment: SAMSegment,
                           region_name: str) -> Optional[DofusUIElement]:
        """Classifie un segment comme élément UI spécifique"""

        x, y, w, h = segment.bbox
        roi = image[y:y+h, x:x+w]

        if region_name == "health_bar":
            # Détecter barre de vie (rouge/verte)
            avg_color = np.mean(roi, axis=(0, 1))
            if avg_color[0] > 150 or avg_color[1] > 150:  # Rouge ou vert
                health_pct = self._estimate_bar_percentage(roi, "health")
                return DofusUIElement(
                    element_type="health_bar",
                    segment=segment,
                    value=health_pct
                )

        elif region_name == "mana_bar":
            # Détecter barre de mana (bleue)
            avg_color = np.mean(roi, axis=(0, 1))
            if avg_color[2] > 150:  # Bleu
                mana_pct = self._estimate_bar_percentage(roi, "mana")
                return DofusUIElement(
                    element_type="mana_bar",
                    segment=segment,
                    value=mana_pct
                )

        elif region_name == "spell_bar":
            # Détecter boutons de sorts
            if w > 30 and h > 30 and abs(w - h) < 10:  # Approximativement carré
                return DofusUIElement(
                    element_type="spell_button",
                    segment=segment,
                    clickable=True
                )

        return None

    def _estimate_bar_percentage(self, roi: np.ndarray, bar_type: str) -> float:
        """Estime le pourcentage d'une barre (vie, mana, etc.)"""

        # Conversion en HSV pour meilleure détection couleur
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        if bar_type == "health":
            # Détecter rouge/vert pour santé
            red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            health_mask = cv2.bitwise_or(red_mask, green_mask)

            # Calculer pourcentage rempli
            total_pixels = roi.shape[0] * roi.shape[1]
            filled_pixels = np.sum(health_mask > 0)
            return min(1.0, filled_pixels / total_pixels)

        elif bar_type == "mana":
            # Détecter bleu pour mana
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            total_pixels = roi.shape[0] * roi.shape[1]
            filled_pixels = np.sum(blue_mask > 0)
            return min(1.0, filled_pixels / total_pixels)

        return 0.0

    def _analyze_entities(self,
                         image: np.ndarray,
                         segments: List[SAMSegment]) -> List[DofusEntity]:
        """Analyse les entités de jeu (joueurs, monstres, PNJs)"""

        entities = []

        for segment in segments:
            # Filtrer par taille (entités ont une taille minimum)
            if segment.area < 500:
                continue

            entity = self._classify_entity(image, segment)
            if entity:
                entities.append(entity)

        return entities

    def _classify_entity(self,
                        image: np.ndarray,
                        segment: SAMSegment) -> Optional[DofusEntity]:
        """Classifie un segment comme entité de jeu"""

        x, y, w, h = segment.bbox
        roi = image[y:y+h, x:x+w]

        # Analyse basique par couleur dominante et forme
        avg_color = np.mean(roi, axis=(0, 1))
        aspect_ratio = w / h if h > 0 else 1.0

        # Détection joueur (généralement plus grand, couleurs variées)
        if (segment.area > 1500 and
            0.4 <= aspect_ratio <= 2.5 and
            y > image.shape[0] * 0.2):  # Pas dans l'UI

            # Vérifier si c'est le joueur principal (centre de l'écran généralement)
            screen_center_x = image.shape[1] // 2
            entity_center_x = x + w // 2

            if abs(entity_center_x - screen_center_x) < 100:
                return DofusEntity(
                    entity_type="player_self",
                    segment=segment
                )
            else:
                return DofusEntity(
                    entity_type="player_other",
                    segment=segment
                )

        # Détection monstre (couleurs rougeâtres souvent)
        if (segment.area > 800 and
            avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]):

            return DofusEntity(
                entity_type="monster",
                segment=segment,
                is_enemy=True
            )

        # Détection PNJ (souvent statiques, couleurs neutres)
        if (500 <= segment.area <= 1200 and
            abs(avg_color[0] - avg_color[1]) < 20):

            return DofusEntity(
                entity_type="npc",
                segment=segment
            )

        return None

    def _analyze_combat_state(self,
                            image: np.ndarray,
                            segments: List[SAMSegment]) -> Dict[str, Any]:
        """Analyse l'état du combat"""

        combat_info = {
            "in_combat": False,
            "combat_grid_detected": False,
            "turn_indicator": None,
            "action_points": 0,
            "movement_points": 0,
            "enemies_count": 0,
            "allies_count": 0
        }

        # Détecter grille de combat (motif géométrique régulier)
        grid_region = self._get_region_roi(image, "combat_grid")
        if grid_region is not None:
            combat_info["combat_grid_detected"] = self._detect_combat_grid(grid_region)
            combat_info["in_combat"] = combat_info["combat_grid_detected"]

        # Compter entités ennemies/alliées dans les segments
        for segment in segments:
            if segment.area > 800:  # Taille minimale entité
                entity_type = self._quick_entity_classification(image, segment)
                if "enemy" in entity_type:
                    combat_info["enemies_count"] += 1
                elif "ally" in entity_type:
                    combat_info["allies_count"] += 1

        return combat_info

    def _get_region_roi(self, image: np.ndarray, region_name: str) -> Optional[np.ndarray]:
        """Extrait la ROI d'une région nommée"""
        if region_name not in self.ui_regions:
            return None

        h, w = image.shape[:2]
        rx, ry, rw, rh = self.ui_regions[region_name]

        x1, y1 = int(rx * w), int(ry * h)
        x2, y2 = int((rx + rw) * w), int((ry + rh) * h)

        return image[y1:y2, x1:x2]

    def _detect_combat_grid(self, roi: np.ndarray) -> bool:
        """Détecte la présence de la grille de combat"""
        if roi is None or roi.size == 0:
            return False

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Détecter lignes (grille)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

        # Si assez de lignes détectées = grille probable
        return lines is not None and len(lines) > 10

    def _quick_entity_classification(self, image: np.ndarray, segment: SAMSegment) -> str:
        """Classification rapide d'entité pour comptage"""
        x, y, w, h = segment.bbox
        roi = image[y:y+h, x:x+w]

        avg_color = np.mean(roi, axis=(0, 1))

        # Rouge dominant = ennemi probable
        if avg_color[0] > avg_color[1] + 20 and avg_color[0] > avg_color[2] + 20:
            return "enemy"

        # Bleu/vert dominant = allié probable
        if avg_color[1] > avg_color[0] + 10 or avg_color[2] > avg_color[0] + 10:
            return "ally"

        return "neutral"

# Factory function
def create_sam_processor(model_size: str = "sam2_hiera_large") -> SAMProcessor:
    """Crée un processeur SAM configuré"""
    return SAMProcessor(model_size)

def create_dofus_sam_analyzer() -> DofusSAMAnalyzer:
    """Crée un analyseur SAM pour DOFUS"""
    return DofusSAMAnalyzer()