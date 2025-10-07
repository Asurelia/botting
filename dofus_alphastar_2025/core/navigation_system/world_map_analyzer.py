#!/usr/bin/env python3
"""
WorldMapAnalyzer - Analyseur de carte monde DOFUS avec SAM 2
Détecte zones, chemins, téléporteurs et points d'intérêt automatiquement
"""

import cv2
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from pathlib import Path
import json

import torch
from PIL import Image

from config import config
from core.vision_engine_v2 import create_vision_engine, SAMSegment, TextDetection

logger = logging.getLogger(__name__)

class MapElementType(Enum):
    """Types d'éléments de carte"""
    ZONE_BOUNDARY = "zone_boundary"
    PATH = "path"
    ZAAP = "zaap"
    TELEPORTER = "teleporter"
    NPC = "npc"
    MONSTER_GROUP = "monster_group"
    RESOURCE = "resource"
    BUILDING = "building"
    WATER = "water"
    OBSTACLE = "obstacle"
    ENTRANCE = "entrance"
    EXIT = "exit"
    UNKNOWN = "unknown"

class TerrainType(Enum):
    """Types de terrain"""
    PLAINS = "plains"
    FOREST = "forest"
    MOUNTAIN = "mountain"
    DESERT = "desert"
    SWAMP = "swamp"
    BEACH = "beach"
    CITY = "city"
    DUNGEON = "dungeon"
    UNDERWATER = "underwater"

@dataclass
class MapElement:
    """Élément détecté sur la carte"""
    element_id: str
    element_type: MapElementType
    position: Tuple[int, int]
    size: Tuple[int, int]
    confidence: float

    # Propriétés visuelles
    color_signature: Optional[np.ndarray] = None
    shape_features: Dict[str, float] = field(default_factory=dict)

    # Métadonnées
    name: Optional[str] = None
    level_requirement: Optional[int] = None
    is_accessible: bool = True
    cost_modifier: float = 1.0

    # Relations
    connected_elements: List[str] = field(default_factory=list)
    zone_id: Optional[str] = None

@dataclass
class MapRegion:
    """Région de carte analysée"""
    region_id: str
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    terrain_type: TerrainType
    elements: List[MapElement] = field(default_factory=list)

    # Caractéristiques de navigation
    walkable_areas: List[Tuple[int, int, int, int]] = field(default_factory=list)
    blocked_areas: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Métadonnées
    zone_name: Optional[str] = None
    level_range: Tuple[int, int] = (1, 200)
    monsters: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)

class ColorAnalyzer:
    """Analyseur de couleurs spécialisé DOFUS"""

    def __init__(self):
        # Palettes de couleurs typiques DOFUS
        self.dofus_colors = {
            'water': {
                'hue_range': (100, 130),
                'sat_range': (50, 255),
                'val_range': (50, 200)
            },
            'grass': {
                'hue_range': (40, 80),
                'sat_range': (30, 255),
                'val_range': (50, 200)
            },
            'path': {
                'hue_range': (15, 35),
                'sat_range': (20, 100),
                'val_range': (80, 180)
            },
            'building': {
                'hue_range': (0, 20),
                'sat_range': (10, 100),
                'val_range': (60, 160)
            },
            'zaap': {
                'hue_range': (120, 140),
                'sat_range': (100, 255),
                'val_range': (100, 255)
            }
        }

    def analyze_color_signature(self, image: np.ndarray) -> Dict[str, float]:
        """Analyse la signature colorimétrique d'une région"""
        if image.size == 0:
            return {}

        # Convertir en HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Calculer histogrammes
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        # Normaliser
        hist_h = hist_h.flatten() / hist_h.sum()
        hist_s = hist_s.flatten() / hist_s.sum()
        hist_v = hist_v.flatten() / hist_v.sum()

        # Calculer couleur dominante
        dominant_hue = np.argmax(hist_h)
        dominant_sat = np.argmax(hist_s)
        dominant_val = np.argmax(hist_v)

        # Identifier type de terrain
        terrain_scores = {}
        for terrain, color_range in self.dofus_colors.items():
            score = 0.0

            # Score de teinte
            hue_min, hue_max = color_range['hue_range']
            if hue_min <= dominant_hue <= hue_max:
                score += 0.4

            # Score de saturation
            sat_min, sat_max = color_range['sat_range']
            if sat_min <= dominant_sat <= sat_max:
                score += 0.3

            # Score de valeur
            val_min, val_max = color_range['val_range']
            if val_min <= dominant_val <= val_max:
                score += 0.3

            terrain_scores[terrain] = score

        return {
            'dominant_hue': int(dominant_hue),
            'dominant_sat': int(dominant_sat),
            'dominant_val': int(dominant_val),
            'terrain_scores': terrain_scores,
            'best_terrain': max(terrain_scores.items(), key=lambda x: x[1])[0] if terrain_scores else 'unknown'
        }

class ShapeAnalyzer:
    """Analyseur de formes pour éléments de carte"""

    def analyze_shape_features(self, contour: np.ndarray) -> Dict[str, float]:
        """Analyse les caractéristiques de forme d'un contour"""
        features = {}

        try:
            # Aires et périmètres
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if area > 0 and perimeter > 0:
                # Compacité (4π*aire / périmètre²)
                features['compactness'] = (4 * np.pi * area) / (perimeter ** 2)

                # Rectangularité
                rect = cv2.boundingRect(contour)
                rect_area = rect[2] * rect[3]
                features['rectangularity'] = area / rect_area if rect_area > 0 else 0

                # Convexité
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                features['convexity'] = area / hull_area if hull_area > 0 else 0

                # Élongation
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    features['elongation'] = minor_axis / major_axis if major_axis > 0 else 0

                # Solidité
                features['solidity'] = area / hull_area if hull_area > 0 else 0

            # Taille relative
            features['area'] = float(area)
            features['perimeter'] = float(perimeter)

        except Exception as e:
            logger.warning(f"Erreur analyse forme: {e}")

        return features

    def classify_shape_type(self, features: Dict[str, float]) -> MapElementType:
        """Classifie le type d'élément basé sur les caractéristiques de forme"""

        if not features:
            return MapElementType.UNKNOWN

        compactness = features.get('compactness', 0)
        rectangularity = features.get('rectangularity', 0)
        elongation = features.get('elongation', 0)
        area = features.get('area', 0)

        # Règles de classification
        if compactness > 0.8 and area > 100:
            return MapElementType.BUILDING
        elif rectangularity > 0.7 and elongation < 0.3:
            return MapElementType.PATH
        elif compactness > 0.6 and 50 < area < 500:
            return MapElementType.NPC
        elif area > 1000 and compactness < 0.4:
            return MapElementType.WATER
        elif rectangularity > 0.8 and area > 200:
            return MapElementType.BUILDING
        else:
            return MapElementType.UNKNOWN

class WorldMapAnalyzer:
    """Analyseur principal de carte monde"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Moteurs d'analyse
        self.vision_engine = create_vision_engine()
        self.color_analyzer = ColorAnalyzer()
        self.shape_analyzer = ShapeAnalyzer()

        # Cache d'analyse
        self.map_cache: Dict[str, MapRegion] = {}
        self.element_cache: Dict[str, MapElement] = {}

        # Configuration d'analyse
        self.min_element_size = 20
        self.max_element_size = 500
        self.confidence_threshold = 0.7

        # Base de données des zones
        self.zone_database = self._load_zone_database()

        logger.info("WorldMapAnalyzer initialisé avec succès")

    def _load_zone_database(self) -> Dict[str, Any]:
        """Charge la base de données des zones DOFUS"""
        db_path = Path("data/zones/zone_database.json")

        if db_path.exists():
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erreur chargement base zones: {e}")

        # Base par défaut
        return {
            "ganymede": {
                "name": "Ganymède",
                "level_range": [1, 20],
                "terrain": "plains",
                "monsters": ["Bouftou", "Larve Bleue", "Pissenlit"],
                "zaaps": ["Ganymède Centre"],
                "npcs": ["Gardien", "Marchand", "Maître d'Armes"]
            }
        }

    def analyze_map_screenshot(self, screenshot: np.ndarray, region_name: str = None) -> Dict[str, Any]:
        """Analyse complète d'une capture d'écran de carte"""

        analysis_start = time.time()

        try:
            # Phase 1: Segmentation SAM 2
            sam_results = self._segment_map_elements(screenshot)

            # Phase 2: Classification des éléments
            classified_elements = self._classify_map_elements(screenshot, sam_results)

            # Phase 3: Analyse de terrain
            terrain_analysis = self._analyze_terrain(screenshot)

            # Phase 4: Détection de chemins et navigation
            navigation_data = self._analyze_navigation_paths(screenshot, classified_elements)

            # Phase 5: Création de la région
            map_region = self._create_map_region(
                screenshot, classified_elements, terrain_analysis,
                navigation_data, region_name
            )

            analysis_time = time.time() - analysis_start

            return {
                "success": True,
                "region": map_region,
                "elements_count": len(classified_elements),
                "terrain_type": terrain_analysis.get("dominant_terrain"),
                "navigation_paths": len(navigation_data.get("paths", [])),
                "analysis_time": analysis_time,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Erreur analyse carte: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def _segment_map_elements(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Segmente les éléments de carte avec SAM 2"""

        try:
            # Analyser avec le moteur de vision
            vision_results = self.vision_engine.analyze_screenshot(screenshot)

            # Récupérer segments SAM
            sam_segments = vision_results.get("sam_segments", [])
            ui_elements = vision_results.get("ui_elements", [])
            text_detections = vision_results.get("text_detections", [])

            # Filtrer segments par taille
            valid_segments = []
            for segment in sam_segments:
                if (self.min_element_size <= segment.width <= self.max_element_size and
                    self.min_element_size <= segment.height <= self.max_element_size):
                    valid_segments.append(segment)

            return {
                "segments": valid_segments,
                "ui_elements": ui_elements,
                "text_detections": text_detections,
                "total_segments": len(sam_segments),
                "valid_segments": len(valid_segments)
            }

        except Exception as e:
            logger.error(f"Erreur segmentation SAM: {e}")
            return {"segments": [], "ui_elements": [], "text_detections": []}

    def _classify_map_elements(self, screenshot: np.ndarray, sam_results: Dict[str, Any]) -> List[MapElement]:
        """Classifie les éléments segmentés"""

        elements = []
        segments = sam_results.get("segments", [])
        text_detections = sam_results.get("text_detections", [])

        for i, segment in enumerate(segments):
            try:
                # Extraire la région de l'image
                x, y, w, h = segment.x, segment.y, segment.width, segment.height
                region = screenshot[y:y+h, x:x+w]

                if region.size == 0:
                    continue

                # Analyse couleur
                color_analysis = self.color_analyzer.analyze_color_signature(region)

                # Analyse forme (convertir segment en contour)
                mask = segment.mask if hasattr(segment, 'mask') else np.ones((h, w), dtype=np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                shape_features = {}
                element_type = MapElementType.UNKNOWN

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    shape_features = self.shape_analyzer.analyze_shape_features(largest_contour)
                    element_type = self.shape_analyzer.classify_shape_type(shape_features)

                # Affiner classification avec couleur
                element_type = self._refine_classification(element_type, color_analysis, shape_features)

                # Chercher texte associé
                element_name = self._find_associated_text(segment, text_detections)

                # Créer élément
                element = MapElement(
                    element_id=f"elem_{int(time.time())}_{i}",
                    element_type=element_type,
                    position=(x + w//2, y + h//2),
                    size=(w, h),
                    confidence=segment.confidence if hasattr(segment, 'confidence') else 0.8,
                    color_signature=color_analysis.get('dominant_hue'),
                    shape_features=shape_features,
                    name=element_name
                )

                elements.append(element)

            except Exception as e:
                logger.warning(f"Erreur classification élément {i}: {e}")
                continue

        return elements

    def _refine_classification(self,
                             initial_type: MapElementType,
                             color_analysis: Dict[str, Any],
                             shape_features: Dict[str, float]) -> MapElementType:
        """Affine la classification avec analyse couleur"""

        best_terrain = color_analysis.get('best_terrain', '')
        terrain_scores = color_analysis.get('terrain_scores', {})

        # Priorité aux scores couleur élevés
        if terrain_scores.get('zaap', 0) > 0.6:
            return MapElementType.ZAAP
        elif terrain_scores.get('water', 0) > 0.5:
            return MapElementType.WATER
        elif terrain_scores.get('path', 0) > 0.4:
            return MapElementType.PATH
        elif terrain_scores.get('building', 0) > 0.5:
            return MapElementType.BUILDING

        # Sinon garder classification initiale
        return initial_type

    def _find_associated_text(self, segment: SAMSegment, text_detections: List[TextDetection]) -> Optional[str]:
        """Trouve le texte associé à un segment"""

        segment_center = (segment.x + segment.width//2, segment.y + segment.height//2)
        max_distance = max(segment.width, segment.height) * 1.5

        closest_text = None
        min_distance = float('inf')

        for detection in text_detections:
            text_center = (detection.bbox[0] + detection.bbox[2]//2,
                          detection.bbox[1] + detection.bbox[3]//2)

            distance = np.sqrt((segment_center[0] - text_center[0])**2 +
                             (segment_center[1] - text_center[1])**2)

            if distance < max_distance and distance < min_distance:
                min_distance = distance
                closest_text = detection.text

        return closest_text

    def _analyze_terrain(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyse le type de terrain dominant"""

        # Diviser l'image en grille pour analyse
        h, w = screenshot.shape[:2]
        grid_size = 64

        terrain_votes = {}

        for y in range(0, h - grid_size, grid_size):
            for x in range(0, w - grid_size, grid_size):
                patch = screenshot[y:y+grid_size, x:x+grid_size]

                color_analysis = self.color_analyzer.analyze_color_signature(patch)
                best_terrain = color_analysis.get('best_terrain')

                if best_terrain:
                    terrain_votes[best_terrain] = terrain_votes.get(best_terrain, 0) + 1

        # Terrain dominant
        dominant_terrain = max(terrain_votes.items(), key=lambda x: x[1])[0] if terrain_votes else 'unknown'

        return {
            "dominant_terrain": dominant_terrain,
            "terrain_distribution": terrain_votes,
            "confidence": terrain_votes.get(dominant_terrain, 0) / max(sum(terrain_votes.values()), 1)
        }

    def _analyze_navigation_paths(self, screenshot: np.ndarray, elements: List[MapElement]) -> Dict[str, Any]:
        """Analyse les chemins de navigation"""

        # Éléments de chemin
        path_elements = [elem for elem in elements if elem.element_type == MapElementType.PATH]

        # Zones walkables (simplifiée)
        h, w = screenshot.shape[:2]
        walkable_mask = np.ones((h, w), dtype=np.uint8)

        # Marquer obstacles
        for elem in elements:
            if elem.element_type in [MapElementType.WATER, MapElementType.OBSTACLE, MapElementType.BUILDING]:
                x, y = elem.position
                size_x, size_y = elem.size

                x1 = max(0, x - size_x//2)
                x2 = min(w, x + size_x//2)
                y1 = max(0, y - size_y//2)
                y2 = min(h, y + size_y//2)

                walkable_mask[y1:y2, x1:x2] = 0

        # Détecter chemins principaux
        paths = self._extract_navigation_paths(walkable_mask)

        return {
            "paths": paths,
            "walkable_percentage": np.sum(walkable_mask) / walkable_mask.size * 100,
            "path_elements": len(path_elements)
        }

    def _extract_navigation_paths(self, walkable_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Extrait les chemins de navigation principaux"""

        paths = []

        # Squelettisation pour trouver chemins principaux
        from skimage.morphology import skeletonize
        from skimage.measure import label, regionprops

        try:
            skeleton = skeletonize(walkable_mask > 0)
            labeled_skeleton = label(skeleton)

            for region in regionprops(labeled_skeleton):
                if region.area > 50:  # Chemins suffisamment longs
                    coords = region.coords

                    # Points de début et fin
                    start_point = tuple(coords[0][::-1])  # (x, y)
                    end_point = tuple(coords[-1][::-1])

                    # Points intermédiaires (simplifiés)
                    step = max(1, len(coords) // 10)
                    waypoints = [tuple(coord[::-1]) for coord in coords[::step]]

                    paths.append({
                        "start": start_point,
                        "end": end_point,
                        "waypoints": waypoints,
                        "length": region.area,
                        "confidence": 0.8
                    })

        except Exception as e:
            logger.warning(f"Erreur extraction chemins: {e}")

        return paths

    def _create_map_region(self,
                          screenshot: np.ndarray,
                          elements: List[MapElement],
                          terrain_analysis: Dict[str, Any],
                          navigation_data: Dict[str, Any],
                          region_name: str = None) -> MapRegion:
        """Crée une région de carte complète"""

        h, w = screenshot.shape[:2]

        # Identifier zone dans base de données
        zone_info = None
        if region_name:
            zone_info = self.zone_database.get(region_name.lower())

        # Terrain type
        terrain_str = terrain_analysis.get("dominant_terrain", "plains")
        try:
            terrain_type = TerrainType(terrain_str)
        except ValueError:
            terrain_type = TerrainType.PLAINS

        # Zones walkables
        walkable_areas = [(0, 0, w, h)]  # Simplifiée pour l'instant

        # Zones bloquées (obstacles)
        blocked_areas = []
        for elem in elements:
            if elem.element_type in [MapElementType.WATER, MapElementType.OBSTACLE]:
                x, y = elem.position
                size_x, size_y = elem.size
                blocked_areas.append((x - size_x//2, y - size_y//2, size_x, size_y))

        region = MapRegion(
            region_id=region_name or f"region_{int(time.time())}",
            bounds=(0, 0, w, h),
            terrain_type=terrain_type,
            elements=elements,
            walkable_areas=walkable_areas,
            blocked_areas=blocked_areas,
            zone_name=zone_info.get("name") if zone_info else region_name,
            level_range=tuple(zone_info.get("level_range", [1, 200])) if zone_info else (1, 200),
            monsters=zone_info.get("monsters", []) if zone_info else [],
            resources=zone_info.get("resources", []) if zone_info else []
        )

        # Mettre en cache
        if region_name:
            self.map_cache[region_name] = region

        return region

    def get_cached_region(self, region_name: str) -> Optional[MapRegion]:
        """Récupère une région en cache"""
        return self.map_cache.get(region_name)

    def find_elements_by_type(self, region: MapRegion, element_type: MapElementType) -> List[MapElement]:
        """Trouve les éléments d'un type spécifique"""
        return [elem for elem in region.elements if elem.element_type == element_type]

    def find_nearest_element(self, region: MapRegion, position: Tuple[int, int], element_type: MapElementType = None) -> Optional[MapElement]:
        """Trouve l'élément le plus proche d'une position"""

        candidates = region.elements
        if element_type:
            candidates = [elem for elem in candidates if elem.element_type == element_type]

        if not candidates:
            return None

        min_distance = float('inf')
        nearest = None

        for elem in candidates:
            distance = np.sqrt((position[0] - elem.position[0])**2 +
                             (position[1] - elem.position[1])**2)

            if distance < min_distance:
                min_distance = distance
                nearest = elem

        return nearest

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Statistiques d'analyse"""
        return {
            "cached_regions": len(self.map_cache),
            "cached_elements": len(self.element_cache),
            "zone_database_size": len(self.zone_database),
            "analysis_config": {
                "min_element_size": self.min_element_size,
                "max_element_size": self.max_element_size,
                "confidence_threshold": self.confidence_threshold
            }
        }

def create_world_map_analyzer() -> WorldMapAnalyzer:
    """Factory function pour créer un WorldMapAnalyzer"""
    return WorldMapAnalyzer()