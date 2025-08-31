"""
Module de template matching avancé pour DOFUS
Système de reconnaissance visuelle robuste utilisant diverses techniques de matching
"""

import cv2
import numpy as np
import time
import threading
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import des modules internes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from engine.module_interface import IAnalysisModule, ModuleStatus
from engine.event_bus import EventType, EventPriority


@dataclass
class TemplateMatch:
    """Résultat d'un match de template"""
    template_name: str
    template_type: str  # resource, monster, npc, ui, etc.
    confidence: float
    position: Tuple[int, int]  # Centre du match
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    scale: float  # Échelle utilisée
    rotation: float  # Rotation utilisée (en degrés)
    method: str  # Méthode de matching utilisée
    additional_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_center(self) -> Tuple[int, int]:
        """Retourne le centre du match"""
        return self.position
    
    def get_area(self) -> int:
        """Retourne l'aire du match"""
        x1, y1, x2, y2 = self.bounding_box
        return (x2 - x1) * (y2 - y1)


@dataclass
class Template:
    """Template de matching avec métadonnées"""
    name: str
    category: str  # resource, monster, npc, ui, button, etc.
    image: np.ndarray  # Template principal
    mask: Optional[np.ndarray] = None  # Masque optionnel
    scales: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    rotations: List[float] = field(default_factory=lambda: [0])  # En degrés
    threshold: float = 0.75  # Seuil de confiance
    max_matches: int = 5  # Nombre maximum de matches
    roi_only: bool = False  # Si True, cherche seulement dans certaines régions
    valid_regions: List[str] = field(default_factory=list)  # Régions valides
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_hash(self) -> str:
        """Génère un hash unique pour ce template"""
        content = f"{self.name}_{self.category}_{self.threshold}_{len(self.scales)}_{len(self.rotations)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


class FeatureMatcher:
    """
    Matcher basé sur les caractéristiques (SIFT, ORB)
    Plus robuste aux transformations mais plus lent
    """
    
    def __init__(self):
        # Détecteurs de caractéristiques
        self.sift = cv2.SIFT_create(nfeatures=500)
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # Matcher FLANN pour SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Matcher BruteForce pour ORB
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Cache des descripteurs
        self.descriptor_cache = {}
        
    def extract_features(self, image: np.ndarray, method: str = 'sift') -> Tuple[List, np.ndarray]:
        """
        Extrait les caractéristiques d'une image
        
        Args:
            image: Image à analyser
            method: 'sift' ou 'orb'
            
        Returns:
            Tuple (keypoints, descriptors)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            if method.lower() == 'sift':
                keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            else:  # ORB
                keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            return keypoints, descriptors
            
        except Exception as e:
            logging.error(f"Erreur extraction caractéristiques: {e}")
            return [], None
    
    def match_features(self, template_desc: np.ndarray, image_desc: np.ndarray, 
                      method: str = 'sift') -> List[cv2.DMatch]:
        """
        Effectue le matching entre descripteurs
        
        Args:
            template_desc: Descripteurs du template
            image_desc: Descripteurs de l'image
            method: 'sift' ou 'orb'
            
        Returns:
            Liste des matches
        """
        try:
            if template_desc is None or image_desc is None:
                return []
            
            if method.lower() == 'sift':
                # FLANN matcher pour SIFT
                if len(template_desc) >= 2 and len(image_desc) >= 2:
                    matches = self.flann_matcher.knnMatch(template_desc, image_desc, k=2)
                    
                    # Test de ratio de Lowe
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)
                    return good_matches
            else:
                # BruteForce matcher pour ORB
                matches = self.bf_matcher.match(template_desc, image_desc)
                # Tri par distance
                matches = sorted(matches, key=lambda x: x.distance)
                return matches[:50]  # Top 50 matches
                
        except Exception as e:
            logging.error(f"Erreur matching caractéristiques: {e}")
            
        return []
    
    def find_template_with_features(self, template: Template, image: np.ndarray, 
                                   method: str = 'sift') -> List[TemplateMatch]:
        """
        Trouve un template en utilisant les caractéristiques
        
        Args:
            template: Template à chercher
            image: Image où chercher
            method: 'sift' ou 'orb'
            
        Returns:
            Liste des matches trouvés
        """
        matches = []
        
        try:
            # Extraction des caractéristiques du template (avec cache)
            template_hash = template.get_hash()
            cache_key = f"{template_hash}_{method}"
            
            if cache_key in self.descriptor_cache:
                template_kp, template_desc = self.descriptor_cache[cache_key]
            else:
                template_kp, template_desc = self.extract_features(template.image, method)
                self.descriptor_cache[cache_key] = (template_kp, template_desc)
            
            if template_desc is None or len(template_kp) < 10:
                return []
            
            # Extraction des caractéristiques de l'image
            image_kp, image_desc = self.extract_features(image, method)
            
            if image_desc is None or len(image_kp) < 10:
                return []
            
            # Matching des caractéristiques
            good_matches = self.match_features(template_desc, image_desc, method)
            
            if len(good_matches) >= 10:  # Minimum de matches requis
                # Extraction des points correspondants
                src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([image_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Calcul de l'homographie
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if homography is not None and mask.sum() >= 8:
                    # Calcul des coins du template transformé
                    h, w = template.image.shape[:2]
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, homography)
                    
                    # Vérification que la transformation est valide
                    if self._is_valid_transformation(transformed_corners, image.shape):
                        # Calcul de la confiance basée sur le nombre de points inliers
                        confidence = min(1.0, mask.sum() / len(good_matches))
                        
                        # Calcul du centre et de la bounding box
                        x_coords = transformed_corners[:, 0, 0]
                        y_coords = transformed_corners[:, 0, 1]
                        
                        x1, x2 = int(x_coords.min()), int(x_coords.max())
                        y1, y2 = int(y_coords.min()), int(y_coords.max())
                        
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Calcul de l'échelle et rotation estimées
                        scale = self._estimate_scale(corners, transformed_corners)
                        rotation = self._estimate_rotation(corners, transformed_corners)
                        
                        match = TemplateMatch(
                            template_name=template.name,
                            template_type=template.category,
                            confidence=confidence,
                            position=(center_x, center_y),
                            bounding_box=(x1, y1, x2, y2),
                            scale=scale,
                            rotation=rotation,
                            method=f"features_{method}",
                            additional_data={
                                "inliers": int(mask.sum()),
                                "total_matches": len(good_matches),
                                "homography": homography.tolist()
                            }
                        )
                        
                        matches.append(match)
                        
        except Exception as e:
            logging.error(f"Erreur matching par caractéristiques: {e}")
        
        return matches
    
    def _is_valid_transformation(self, corners: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """Vérifie si une transformation est valide"""
        try:
            h, w = image_shape[:2]
            
            # Vérification que tous les points sont dans l'image
            x_coords = corners[:, 0, 0]
            y_coords = corners[:, 0, 1]
            
            if (x_coords < 0).any() or (x_coords > w).any():
                return False
            if (y_coords < 0).any() or (y_coords > h).any():
                return False
            
            # Vérification que le quadrilatère n'est pas trop déformé
            area = cv2.contourArea(corners)
            if area < 100:  # Aire minimum
                return False
            
            # Vérification de la convexité approximative
            hull = cv2.convexHull(corners)
            hull_area = cv2.contourArea(hull)
            
            if area / hull_area < 0.8:  # Doit être assez convexe
                return False
            
            return True
            
        except:
            return False
    
    def _estimate_scale(self, original_corners: np.ndarray, transformed_corners: np.ndarray) -> float:
        """Estime l'échelle de transformation"""
        try:
            # Calcul des distances entre coins adjacents
            orig_dist = np.linalg.norm(original_corners[1] - original_corners[0])
            trans_dist = np.linalg.norm(transformed_corners[1] - transformed_corners[0])
            
            return float(trans_dist / orig_dist) if orig_dist > 0 else 1.0
        except:
            return 1.0
    
    def _estimate_rotation(self, original_corners: np.ndarray, transformed_corners: np.ndarray) -> float:
        """Estime la rotation de transformation"""
        try:
            # Vecteur horizontal original
            orig_vec = original_corners[1] - original_corners[0]
            trans_vec = transformed_corners[1] - transformed_corners[0]
            
            # Calcul des angles
            orig_angle = np.arctan2(orig_vec[0, 1], orig_vec[0, 0])
            trans_angle = np.arctan2(trans_vec[0, 1], trans_vec[0, 0])
            
            # Différence d'angle en degrés
            rotation = np.degrees(trans_angle - orig_angle)
            
            # Normalisation [-180, 180]
            while rotation > 180:
                rotation -= 360
            while rotation < -180:
                rotation += 360
                
            return float(rotation)
        except:
            return 0.0


class TemplateMatchingEngine:
    """
    Moteur de template matching utilisant différentes techniques
    """
    
    def __init__(self):
        # Méthodes de matching OpenCV
        self.cv_methods = {
            'ccoeff_normed': cv2.TM_CCOEFF_NORMED,
            'ccorr_normed': cv2.TM_CCORR_NORMED,
            'sqdiff_normed': cv2.TM_SQDIFF_NORMED
        }
        
        # Matcher par caractéristiques
        self.feature_matcher = FeatureMatcher()
        
        # Cache des templates redimensionnés/tournés
        self.template_cache = {}
        self.cache_size_limit = 1000
        
    def clear_cache(self) -> None:
        """Vide le cache des templates"""
        self.template_cache.clear()
        self.feature_matcher.descriptor_cache.clear()
    
    def _get_template_variants(self, template: Template) -> List[Tuple[np.ndarray, float, float]]:
        """
        Génère toutes les variantes d'un template (échelles/rotations)
        
        Returns:
            Liste de (image_variant, scale, rotation)
        """
        variants = []
        base_image = template.image
        
        for scale in template.scales:
            for rotation in template.rotations:
                cache_key = f"{template.name}_{scale}_{rotation}"
                
                if cache_key in self.template_cache:
                    variant = self.template_cache[cache_key]
                else:
                    variant = self._transform_template(base_image, scale, rotation)
                    
                    # Limitation du cache
                    if len(self.template_cache) < self.cache_size_limit:
                        self.template_cache[cache_key] = variant
                
                if variant is not None:
                    variants.append((variant, scale, rotation))
        
        return variants
    
    def _transform_template(self, template: np.ndarray, scale: float, rotation: float) -> Optional[np.ndarray]:
        """Applique transformation (échelle + rotation) à un template"""
        try:
            h, w = template.shape[:2]
            
            # Application de l'échelle
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Application de la rotation si nécessaire
            if abs(rotation) > 0.1:
                center = (new_w // 2, new_h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                
                # Calcul de la nouvelle taille après rotation
                cos_val = abs(rotation_matrix[0, 0])
                sin_val = abs(rotation_matrix[0, 1])
                new_width = int((new_h * sin_val) + (new_w * cos_val))
                new_height = int((new_h * cos_val) + (new_w * sin_val))
                
                # Ajustement de la translation
                rotation_matrix[0, 2] += (new_width / 2) - center[0]
                rotation_matrix[1, 2] += (new_height / 2) - center[1]
                
                rotated = cv2.warpAffine(scaled, rotation_matrix, (new_width, new_height))
                return rotated
            
            return scaled
            
        except Exception as e:
            logging.error(f"Erreur transformation template: {e}")
            return None
    
    def match_template_cv(self, template: Template, image: np.ndarray, 
                         method: str = 'ccoeff_normed') -> List[TemplateMatch]:
        """
        Template matching standard OpenCV
        
        Args:
            template: Template à chercher
            image: Image où chercher
            method: Méthode OpenCV à utiliser
            
        Returns:
            Liste des matches
        """
        matches = []
        
        try:
            if method not in self.cv_methods:
                method = 'ccoeff_normed'
            
            cv_method = self.cv_methods[method]
            
            # Conversion en niveaux de gris si nécessaire
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Test de toutes les variantes du template
            variants = self._get_template_variants(template)
            
            for variant, scale, rotation in variants:
                if len(variant.shape) == 3:
                    variant = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY)
                
                # Vérification des dimensions
                if variant.shape[0] > gray_image.shape[0] or variant.shape[1] > gray_image.shape[1]:
                    continue
                
                # Template matching
                if template.mask is not None:
                    mask_variant = self._transform_template(template.mask, scale, rotation)
                    result = cv2.matchTemplate(gray_image, variant, cv_method, mask=mask_variant)
                else:
                    result = cv2.matchTemplate(gray_image, variant, cv_method)
                
                # Recherche des pics
                locations = self._find_template_peaks(result, template.threshold, template.max_matches, method)
                
                # Création des objets TemplateMatch
                template_h, template_w = variant.shape[:2]
                
                for loc, confidence in locations:
                    x, y = loc
                    center_x = x + template_w // 2
                    center_y = y + template_h // 2
                    
                    match = TemplateMatch(
                        template_name=template.name,
                        template_type=template.category,
                        confidence=confidence,
                        position=(center_x, center_y),
                        bounding_box=(x, y, x + template_w, y + template_h),
                        scale=scale,
                        rotation=rotation,
                        method=f"cv_{method}",
                        additional_data={
                            "template_size": (template_w, template_h)
                        }
                    )
                    
                    matches.append(match)
                    
        except Exception as e:
            logging.error(f"Erreur template matching CV: {e}")
        
        return matches
    
    def _find_template_peaks(self, result: np.ndarray, threshold: float, 
                            max_matches: int, method: str) -> List[Tuple[Tuple[int, int], float]]:
        """Trouve les pics dans le résultat de template matching"""
        try:
            locations = []
            
            if method == 'sqdiff_normed':
                # Pour SQDIFF, les valeurs faibles sont meilleures
                mask = result <= (1.0 - threshold)
                threshold_val = 1.0 - threshold
                inverted = True
            else:
                # Pour les autres, les valeurs élevées sont meilleures
                mask = result >= threshold
                threshold_val = threshold
                inverted = False
            
            if not mask.any():
                return []
            
            # Suppression des non-maxima locaux
            result_filtered = result.copy()
            result_filtered[~mask] = 0 if not inverted else 1
            
            # Recherche des maxima locaux
            kernel_size = 15  # Taille de la zone d'exclusion autour de chaque pic
            
            for _ in range(max_matches):
                if inverted:
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_filtered)
                    if min_val <= threshold_val:
                        confidence = 1.0 - min_val
                        loc = min_loc
                    else:
                        break
                else:
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_filtered)
                    if max_val >= threshold_val:
                        confidence = max_val
                        loc = max_loc
                    else:
                        break
                
                locations.append((loc, float(confidence)))
                
                # Suppression de la zone autour du pic trouvé
                y, x = loc[1], loc[0]
                y1 = max(0, y - kernel_size // 2)
                y2 = min(result_filtered.shape[0], y + kernel_size // 2 + 1)
                x1 = max(0, x - kernel_size // 2)
                x2 = min(result_filtered.shape[1], x + kernel_size // 2 + 1)
                
                result_filtered[y1:y2, x1:x2] = 0 if not inverted else 1
            
            return locations
            
        except Exception as e:
            logging.error(f"Erreur détection pics: {e}")
            return []


class TemplateManager:
    """
    Gestionnaire des templates avec chargement et cache intelligent
    """
    
    def __init__(self, templates_dir: str = "assets/templates"):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, Template] = {}
        self.categories: Dict[str, List[str]] = defaultdict(list)
        self.auto_calibrated: Set[str] = set()
        
        # Métadonnées des templates
        self.template_metadata_file = self.templates_dir / "template_metadata.json"
        self.load_metadata()
        
        self.logger = logging.getLogger(f"{__name__}.TemplateManager")
        
    def load_metadata(self) -> None:
        """Charge les métadonnées des templates"""
        try:
            if self.template_metadata_file.exists():
                with open(self.template_metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            self.logger.error(f"Erreur chargement métadonnées: {e}")
            self.metadata = {}
    
    def save_metadata(self) -> None:
        """Sauvegarde les métadonnées des templates"""
        try:
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Construction des métadonnées à sauvegarder
            metadata_to_save = {}
            
            for name, template in self.templates.items():
                metadata_to_save[name] = {
                    "category": template.category,
                    "threshold": template.threshold,
                    "scales": template.scales,
                    "rotations": template.rotations,
                    "max_matches": template.max_matches,
                    "roi_only": template.roi_only,
                    "valid_regions": template.valid_regions,
                    "metadata": template.metadata
                }
            
            with open(self.template_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde métadonnées: {e}")
    
    def load_templates_from_directory(self) -> int:
        """
        Charge tous les templates depuis le répertoire
        
        Returns:
            Nombre de templates chargés
        """
        loaded_count = 0
        
        if not self.templates_dir.exists():
            self.logger.warning(f"Répertoire templates {self.templates_dir} non trouvé")
            return 0
        
        # Structure attendue: templates/category/template_name.png
        for category_dir in self.templates_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category = category_dir.name
            
            for template_file in category_dir.glob("*.png"):
                try:
                    template_name = template_file.stem
                    
                    # Chargement de l'image
                    image = cv2.imread(str(template_file))
                    if image is None:
                        continue
                    
                    # Chargement du masque optionnel
                    mask_file = template_file.parent / f"{template_name}_mask.png"
                    mask = None
                    if mask_file.exists():
                        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    
                    # Récupération des métadonnées sauvegardées
                    template_key = f"{category}_{template_name}"
                    saved_meta = self.metadata.get(template_key, {})
                    
                    # Création du template
                    template = Template(
                        name=template_name,
                        category=category,
                        image=image,
                        mask=mask,
                        scales=saved_meta.get('scales', [0.8, 0.9, 1.0, 1.1, 1.2]),
                        rotations=saved_meta.get('rotations', [0]),
                        threshold=saved_meta.get('threshold', 0.75),
                        max_matches=saved_meta.get('max_matches', 5),
                        roi_only=saved_meta.get('roi_only', False),
                        valid_regions=saved_meta.get('valid_regions', []),
                        metadata=saved_meta.get('metadata', {})
                    )
                    
                    self.add_template(template)
                    loaded_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Erreur chargement template {template_file}: {e}")
        
        self.logger.info(f"Chargé {loaded_count} templates depuis {self.templates_dir}")
        return loaded_count
    
    def add_template(self, template: Template) -> None:
        """Ajoute un template au gestionnaire"""
        template_key = f"{template.category}_{template.name}"
        self.templates[template_key] = template
        self.categories[template.category].append(template.name)
    
    def get_template(self, category: str, name: str) -> Optional[Template]:
        """Récupère un template spécifique"""
        template_key = f"{category}_{name}"
        return self.templates.get(template_key)
    
    def get_templates_by_category(self, category: str) -> List[Template]:
        """Récupère tous les templates d'une catégorie"""
        return [self.templates[f"{category}_{name}"] for name in self.categories[category]
                if f"{category}_{name}" in self.templates]
    
    def get_all_templates(self) -> List[Template]:
        """Récupère tous les templates"""
        return list(self.templates.values())
    
    def calibrate_template_threshold(self, template_name: str, category: str,
                                   positive_samples: List[np.ndarray],
                                   negative_samples: List[np.ndarray],
                                   matching_engine: TemplateMatchingEngine) -> bool:
        """
        Calibre automatiquement le seuil d'un template
        
        Args:
            template_name: Nom du template
            category: Catégorie du template
            positive_samples: Images contenant le template
            negative_samples: Images ne contenant pas le template
            matching_engine: Moteur de matching
            
        Returns:
            True si la calibration réussit
        """
        try:
            template = self.get_template(category, template_name)
            if not template:
                return False
            
            # Test de différents seuils
            thresholds = np.arange(0.5, 0.95, 0.05)
            best_threshold = 0.75
            best_score = 0
            
            for threshold in thresholds:
                template.threshold = threshold
                
                # Test sur échantillons positifs
                true_positives = 0
                for sample in positive_samples:
                    matches = matching_engine.match_template_cv(template, sample)
                    if matches:
                        true_positives += 1
                
                # Test sur échantillons négatifs
                false_positives = 0
                for sample in negative_samples:
                    matches = matching_engine.match_template_cv(template, sample)
                    if matches:
                        false_positives += 1
                
                # Calcul du score F1
                precision = true_positives / len(positive_samples) if positive_samples else 0
                recall = 1 - (false_positives / len(negative_samples)) if negative_samples else 1
                
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    if f1_score > best_score:
                        best_score = f1_score
                        best_threshold = threshold
            
            # Application du meilleur seuil
            template.threshold = best_threshold
            self.auto_calibrated.add(f"{category}_{template_name}")
            
            self.logger.info(f"Template {template_name} calibré avec seuil {best_threshold:.2f} (F1={best_score:.2f})")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur calibration template: {e}")
            return False


class TemplateMatcher(IAnalysisModule):
    """
    Module principal de template matching avancé pour DOFUS
    Coordonne tous les sous-systèmes de matching
    """
    
    def __init__(self, name: str = "template_matcher"):
        super().__init__(name)
        
        self.logger = logging.getLogger(f"{__name__}.TemplateMatcher")
        
        # Composants principaux
        self.template_manager = TemplateManager()
        self.matching_engine = TemplateMatchingEngine()
        
        # Configuration
        self.config = {
            "use_feature_matching": True,
            "use_cv_matching": True,
            "parallel_processing": True,
            "max_workers": 4,
            "cache_enabled": True,
            "cache_duration": 2.0,  # secondes
            "auto_calibration": True,
            "confidence_boost_factor": 1.1  # Boost pour matches avec plusieurs méthodes
        }
        
        # Cache des résultats
        self.results_cache: Dict[str, Tuple[List[TemplateMatch], datetime]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistiques
        self.stats = {
            "templates_loaded": 0,
            "total_matches": 0,
            "avg_match_time": 0,
            "false_positive_rate": 0.05,
            "cache_hit_rate": 0,
            "feature_matches": 0,
            "cv_matches": 0
        }
        
        # Définition des ROI (Régions d'Intérêt) pour DOFUS
        self.roi_definitions = {
            "ui_area": (0, 0, 1920, 200),  # Zone interface haute
            "game_area": (200, 200, 1520, 680),  # Zone de jeu principale
            "minimap": (1700, 50, 200, 200),  # Minimap
            "inventory": (1400, 300, 500, 600),  # Zone inventaire
            "chat": (20, 600, 400, 300),  # Zone de chat
            "spells": (400, 950, 600, 80),  # Barre de sorts
            "full_screen": (0, 0, 1920, 1080)  # Écran complet
        }
        
        # Détection des faux positifs
        self.false_positive_detector = self._init_false_positive_detector()
        
    def _init_false_positive_detector(self) -> Dict[str, Any]:
        """Initialise le détecteur de faux positifs"""
        return {
            "min_match_area": 50,  # Aire minimum d'un match valide
            "max_overlap_ratio": 0.7,  # Ratio maximum de recouvrement entre matches
            "consistency_threshold": 3,  # Nombre de frames pour confirmer un match
            "known_false_positives": set(),  # Positions connues de faux positifs
            "temporal_buffer": defaultdict(list)  # Buffer temporel pour validation
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialise le module de template matching"""
        try:
            self.logger.info("Initialisation du module template matching")
            
            # Mise à jour de la configuration
            self.config.update(config.get('template_matcher', {}))
            
            # Chargement des templates
            templates_loaded = self.template_manager.load_templates_from_directory()
            self.stats["templates_loaded"] = templates_loaded
            
            if templates_loaded == 0:
                self.logger.warning("Aucun template chargé - création de templates par défaut")
                self._create_default_templates()
            
            # Initialisation du pool de threads si activé
            if self.config["parallel_processing"]:
                self.thread_pool = ThreadPoolExecutor(max_workers=self.config["max_workers"])
            
            self.status = ModuleStatus.ACTIVE
            self.logger.info(f"Module template matching initialisé - {templates_loaded} templates chargés")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur initialisation template matcher: {e}")
            self.set_error(str(e))
            return False
    
    def _create_default_templates(self) -> None:
        """Crée des templates par défaut pour les tests"""
        # Cette méthode peut être utilisée pour créer des templates basiques
        # si aucun n'est trouvé dans le répertoire
        pass
    
    def update(self, game_state: Any) -> Optional[Dict[str, Any]]:
        """Met à jour le template matching"""
        try:
            if not self.is_active():
                return None
            
            return {
                "shared_data": {
                    "template_stats": self.stats,
                    "cache_stats": {
                        "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                        "cache_size": len(self.results_cache)
                    }
                },
                "module_status": "active"
            }
            
        except Exception as e:
            self.logger.error(f"Erreur update template matcher: {e}")
            return None
    
    def analyze(self, image: np.ndarray, target_categories: List[str] = None,
                roi: str = "full_screen") -> Dict[str, Any]:
        """
        Analyse complète d'une image avec template matching
        
        Args:
            image: Image à analyser
            target_categories: Catégories de templates à chercher (None = toutes)
            roi: Région d'intérêt à analyser
            
        Returns:
            Dict contenant tous les matches trouvés
        """
        start_time = time.perf_counter()
        
        try:
            # Génération d'une clé de cache
            cache_key = self._generate_cache_key(image, target_categories, roi)
            
            # Vérification du cache
            if self.config["cache_enabled"]:
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.cache_hits += 1
                    return cached_result
                else:
                    self.cache_misses += 1
            
            # Application de la ROI
            roi_image, roi_offset = self._apply_roi(image, roi)
            
            # Sélection des templates à tester
            templates_to_test = self._select_templates(target_categories, roi)
            
            # Matching parallèle ou séquentiel
            all_matches = []
            
            if self.config["parallel_processing"] and len(templates_to_test) > 2:
                all_matches = self._parallel_template_matching(roi_image, templates_to_test)
            else:
                all_matches = self._sequential_template_matching(roi_image, templates_to_test)
            
            # Ajustement des coordonnées pour la ROI
            for match in all_matches:
                match.position = (
                    match.position[0] + roi_offset[0],
                    match.position[1] + roi_offset[1]
                )
                x1, y1, x2, y2 = match.bounding_box
                match.bounding_box = (
                    x1 + roi_offset[0],
                    y1 + roi_offset[1],
                    x2 + roi_offset[0],
                    y2 + roi_offset[1]
                )
            
            # Post-traitement: suppression des doublons et faux positifs
            filtered_matches = self._post_process_matches(all_matches)
            
            # Groupement par catégorie
            matches_by_category = defaultdict(list)
            for match in filtered_matches:
                matches_by_category[match.template_type].append(match)
            
            # Création du résultat
            result = {
                "timestamp": datetime.now(),
                "matches_by_category": dict(matches_by_category),
                "total_matches": len(filtered_matches),
                "roi_used": roi,
                "processing_time": time.perf_counter() - start_time,
                "templates_tested": len(templates_to_test),
                "cache_used": False
            }
            
            # Mise en cache
            if self.config["cache_enabled"]:
                self._cache_result(cache_key, result)
            
            # Mise à jour des statistiques
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur analyse template matching: {e}")
            return {"error": str(e), "timestamp": datetime.now()}
    
    def _generate_cache_key(self, image: np.ndarray, target_categories: List[str], roi: str) -> str:
        """Génère une clé de cache pour un ensemble de paramètres"""
        # Hash basé sur une région d'image réduite pour la performance
        small_image = cv2.resize(image, (100, 100))
        image_hash = hashlib.md5(small_image.tobytes()).hexdigest()[:8]
        
        categories_str = "_".join(sorted(target_categories or []))
        return f"{image_hash}_{categories_str}_{roi}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Récupère un résultat du cache s'il est valide"""
        try:
            if cache_key in self.results_cache:
                result, timestamp = self.results_cache[cache_key]
                
                # Vérification de la validité temporelle
                if datetime.now() - timestamp <= timedelta(seconds=self.config["cache_duration"]):
                    result = result.copy()
                    result["cache_used"] = True
                    return result
                else:
                    # Suppression du cache expiré
                    del self.results_cache[cache_key]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur récupération cache: {e}")
            return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Met en cache un résultat"""
        try:
            # Limitation de la taille du cache
            max_cache_size = 100
            if len(self.results_cache) >= max_cache_size:
                # Suppression de la plus ancienne entrée
                oldest_key = min(self.results_cache.keys(), 
                               key=lambda k: self.results_cache[k][1])
                del self.results_cache[oldest_key]
            
            self.results_cache[cache_key] = (result.copy(), datetime.now())
            
        except Exception as e:
            self.logger.error(f"Erreur mise en cache: {e}")
    
    def _apply_roi(self, image: np.ndarray, roi: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Applique une région d'intérêt à l'image
        
        Returns:
            Tuple (image_roi, offset)
        """
        try:
            if roi not in self.roi_definitions:
                roi = "full_screen"
            
            x, y, w, h = self.roi_definitions[roi]
            
            # Ajustement aux dimensions réelles de l'image
            img_h, img_w = image.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            roi_image = image[y:y+h, x:x+w]
            return roi_image, (x, y)
            
        except Exception as e:
            self.logger.error(f"Erreur application ROI: {e}")
            return image, (0, 0)
    
    def _select_templates(self, target_categories: List[str], roi: str) -> List[Template]:
        """Sélectionne les templates appropriés"""
        try:
            all_templates = self.template_manager.get_all_templates()
            
            # Filtrage par catégorie
            if target_categories:
                templates = [t for t in all_templates if t.category in target_categories]
            else:
                templates = all_templates
            
            # Filtrage par ROI si nécessaire
            roi_filtered = []
            for template in templates:
                if template.roi_only and template.valid_regions:
                    if roi in template.valid_regions:
                        roi_filtered.append(template)
                else:
                    roi_filtered.append(template)
            
            return roi_filtered
            
        except Exception as e:
            self.logger.error(f"Erreur sélection templates: {e}")
            return []
    
    def _sequential_template_matching(self, image: np.ndarray, 
                                    templates: List[Template]) -> List[TemplateMatch]:
        """Effectue le matching séquentiel"""
        all_matches = []
        
        for template in templates:
            try:
                # Template matching OpenCV
                if self.config["use_cv_matching"]:
                    cv_matches = self.matching_engine.match_template_cv(template, image)
                    all_matches.extend(cv_matches)
                    self.stats["cv_matches"] += len(cv_matches)
                
                # Feature matching pour templates complexes
                if (self.config["use_feature_matching"] and 
                    template.category in ['monsters', 'npcs', 'complex_ui']):
                    
                    feature_matches = self.matching_engine.feature_matcher.find_template_with_features(
                        template, image, method='sift'
                    )
                    all_matches.extend(feature_matches)
                    self.stats["feature_matches"] += len(feature_matches)
                    
            except Exception as e:
                self.logger.error(f"Erreur matching template {template.name}: {e}")
        
        return all_matches
    
    def _parallel_template_matching(self, image: np.ndarray, 
                                  templates: List[Template]) -> List[TemplateMatch]:
        """Effectue le matching en parallèle"""
        all_matches = []
        
        try:
            # Division des templates en batches
            future_to_template = {}
            
            for template in templates:
                future = self.thread_pool.submit(self._match_single_template, image, template)
                future_to_template[future] = template
            
            # Collecte des résultats
            for future in as_completed(future_to_template):
                template = future_to_template[future]
                try:
                    matches = future.result(timeout=5.0)  # Timeout de 5s par template
                    all_matches.extend(matches)
                except Exception as e:
                    self.logger.error(f"Erreur matching parallèle {template.name}: {e}")
        
        except Exception as e:
            self.logger.error(f"Erreur matching parallèle: {e}")
        
        return all_matches
    
    def _match_single_template(self, image: np.ndarray, template: Template) -> List[TemplateMatch]:
        """Effectue le matching d'un seul template (pour parallélisation)"""
        matches = []
        
        try:
            # Template matching OpenCV
            if self.config["use_cv_matching"]:
                cv_matches = self.matching_engine.match_template_cv(template, image)
                matches.extend(cv_matches)
                self.stats["cv_matches"] += len(cv_matches)
            
            # Feature matching
            if (self.config["use_feature_matching"] and 
                template.category in ['monsters', 'npcs', 'complex_ui']):
                
                feature_matches = self.matching_engine.feature_matcher.find_template_with_features(
                    template, image, method='sift'
                )
                matches.extend(feature_matches)
                self.stats["feature_matches"] += len(feature_matches)
        
        except Exception as e:
            self.logger.error(f"Erreur matching template {template.name}: {e}")
        
        return matches
    
    def _post_process_matches(self, matches: List[TemplateMatch]) -> List[TemplateMatch]:
        """Post-traitement des matches: suppression doublons et faux positifs"""
        if not matches:
            return []
        
        try:
            # Tri par confiance décroissante
            matches.sort(key=lambda m: m.confidence, reverse=True)
            
            # Suppression des matches qui se chevauchent trop
            filtered_matches = []
            
            for match in matches:
                # Vérification de l'aire minimum
                if match.get_area() < self.false_positive_detector["min_match_area"]:
                    continue
                
                # Vérification du chevauchement avec matches déjà acceptés
                overlap_too_much = False
                
                for accepted_match in filtered_matches:
                    overlap_ratio = self._calculate_overlap_ratio(match, accepted_match)
                    
                    if overlap_ratio > self.false_positive_detector["max_overlap_ratio"]:
                        # Si même template, conserver le meilleur
                        if match.template_name == accepted_match.template_name:
                            if match.confidence > accepted_match.confidence:
                                # Remplacement
                                filtered_matches.remove(accepted_match)
                                break
                            else:
                                overlap_too_much = True
                                break
                        else:
                            # Templates différents qui se chevauchent -> conserver le plus confiant
                            if match.confidence <= accepted_match.confidence:
                                overlap_too_much = True
                                break
                
                if not overlap_too_much:
                    filtered_matches.append(match)
            
            # Boost de confiance pour les matches confirmés par plusieurs méthodes
            self._apply_confidence_boost(filtered_matches)
            
            # Filtrage temporel (validation sur plusieurs frames)
            validated_matches = self._temporal_validation(filtered_matches)
            
            return validated_matches
            
        except Exception as e:
            self.logger.error(f"Erreur post-traitement: {e}")
            return matches
    
    def _calculate_overlap_ratio(self, match1: TemplateMatch, match2: TemplateMatch) -> float:
        """Calcule le ratio de chevauchement entre deux matches"""
        try:
            x1_1, y1_1, x2_1, y2_1 = match1.bounding_box
            x1_2, y1_2, x2_2, y2_2 = match2.bounding_box
            
            # Intersection
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x1_inter >= x2_inter or y1_inter >= y2_inter:
                return 0.0  # Pas de chevauchement
            
            area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            area1 = match1.get_area()
            area2 = match2.get_area()
            
            # Ratio par rapport à la plus petite aire
            min_area = min(area1, area2)
            
            return area_inter / min_area if min_area > 0 else 0.0
            
        except:
            return 0.0
    
    def _apply_confidence_boost(self, matches: List[TemplateMatch]) -> None:
        """Applique un boost de confiance aux matches multi-méthodes"""
        try:
            # Groupement des matches par position approximative
            position_groups = defaultdict(list)
            
            for match in matches:
                # Groupement par zone de 50x50 pixels
                group_key = (match.position[0] // 50, match.position[1] // 50, match.template_name)
                position_groups[group_key].append(match)
            
            # Boost pour les matches avec plusieurs méthodes
            for group in position_groups.values():
                if len(group) > 1:
                    # Vérification que ce sont bien des méthodes différentes
                    methods = set(m.method for m in group)
                    if len(methods) > 1:
                        # Boost de confiance
                        for match in group:
                            match.confidence = min(1.0, match.confidence * self.config["confidence_boost_factor"])
                            match.additional_data["multi_method_boost"] = True
                            
        except Exception as e:
            self.logger.error(f"Erreur boost de confiance: {e}")
    
    def _temporal_validation(self, matches: List[TemplateMatch]) -> List[TemplateMatch]:
        """Validation temporelle des matches"""
        try:
            current_time = datetime.now()
            validated_matches = []
            
            # Nettoyage du buffer temporel (garde 5 secondes d'historique)
            cutoff_time = current_time - timedelta(seconds=5)
            
            for position_key in list(self.false_positive_detector["temporal_buffer"].keys()):
                buffer = self.false_positive_detector["temporal_buffer"][position_key]
                buffer[:] = [entry for entry in buffer if entry["timestamp"] > cutoff_time]
                
                if not buffer:
                    del self.false_positive_detector["temporal_buffer"][position_key]
            
            # Traitement des matches actuels
            for match in matches:
                # Clé basée sur la position et le template
                position_key = f"{match.template_name}_{match.position[0]//30}_{match.position[1]//30}"
                
                # Ajout au buffer temporel
                self.false_positive_detector["temporal_buffer"][position_key].append({
                    "timestamp": current_time,
                    "confidence": match.confidence,
                    "match": match
                })
                
                # Validation basée sur la consistance
                buffer = self.false_positive_detector["temporal_buffer"][position_key]
                recent_count = sum(1 for entry in buffer if current_time - entry["timestamp"] <= timedelta(seconds=1))
                
                # Si c'est un template UI ou critique, validation immédiate
                if (match.template_type in ['ui', 'button', 'interface'] or 
                    recent_count >= self.false_positive_detector["consistency_threshold"] or
                    match.confidence >= 0.9):
                    
                    validated_matches.append(match)
                    match.additional_data["temporal_validation"] = "passed"
                else:
                    match.additional_data["temporal_validation"] = "pending"
            
            return validated_matches
            
        except Exception as e:
            self.logger.error(f"Erreur validation temporelle: {e}")
            return matches
    
    def _update_stats(self, result: Dict[str, Any]) -> None:
        """Met à jour les statistiques du module"""
        try:
            # Moyenne mobile du temps de traitement
            processing_time = result.get("processing_time", 0)
            alpha = 0.1
            self.stats["avg_match_time"] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats["avg_match_time"]
            )
            
            # Compteurs
            self.stats["total_matches"] += result.get("total_matches", 0)
            
            # Taux de cache hit
            total_requests = self.cache_hits + self.cache_misses
            if total_requests > 0:
                self.stats["cache_hit_rate"] = self.cache_hits / total_requests
                
        except Exception as e:
            self.logger.error(f"Erreur mise à jour statistiques: {e}")
    
    def find_templates(self, image: np.ndarray, category: str = None, 
                      template_name: str = None, roi: str = "full_screen",
                      min_confidence: float = None) -> List[TemplateMatch]:
        """
        Interface simplifiée pour trouver des templates
        
        Args:
            image: Image à analyser
            category: Catégorie de templates (optionnel)
            template_name: Nom spécifique de template (optionnel)
            roi: Région d'intérêt
            min_confidence: Confiance minimum requise
            
        Returns:
            Liste des matches trouvés
        """
        try:
            target_categories = [category] if category else None
            
            result = self.analyze(image, target_categories, roi)
            
            if "error" in result:
                return []
            
            # Extraction des matches
            all_matches = []
            for matches in result["matches_by_category"].values():
                all_matches.extend(matches)
            
            # Filtrage par nom de template si spécifié
            if template_name:
                all_matches = [m for m in all_matches if m.template_name == template_name]
            
            # Filtrage par confiance si spécifié
            if min_confidence is not None:
                all_matches = [m for m in all_matches if m.confidence >= min_confidence]
            
            return all_matches
            
        except Exception as e:
            self.logger.error(f"Erreur recherche templates: {e}")
            return []
    
    def calibrate_templates(self, calibration_data: Dict[str, Dict[str, List[np.ndarray]]]) -> Dict[str, bool]:
        """
        Calibre automatiquement plusieurs templates
        
        Args:
            calibration_data: Dict {template_name: {"positive": [images], "negative": [images]}}
            
        Returns:
            Dict {template_name: success}
        """
        results = {}
        
        try:
            for template_name, data in calibration_data.items():
                # Recherche du template dans toutes les catégories
                template = None
                for category in self.template_manager.categories.keys():
                    template = self.template_manager.get_template(category, template_name)
                    if template:
                        break
                
                if not template:
                    results[template_name] = False
                    continue
                
                # Calibration
                success = self.template_manager.calibrate_template_threshold(
                    template_name,
                    template.category,
                    data.get("positive", []),
                    data.get("negative", []),
                    self.matching_engine
                )
                
                results[template_name] = success
            
            # Sauvegarde des nouveaux seuils
            if any(results.values()):
                self.template_manager.save_metadata()
                
        except Exception as e:
            self.logger.error(f"Erreur calibration templates: {e}")
        
        return results
    
    def handle_event(self, event: Any) -> bool:
        """Gestion des événements"""
        # Ce module ne traite pas d'événements spécifiques pour le moment
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état du module"""
        return {
            "status": self.status.value,
            "templates_loaded": self.stats["templates_loaded"],
            "total_matches": self.stats["total_matches"],
            "cache_hit_rate": self.stats["cache_hit_rate"],
            "avg_match_time": self.stats["avg_match_time"],
            "feature_matches": self.stats["feature_matches"],
            "cv_matches": self.stats["cv_matches"],
            "cache_size": len(self.results_cache),
            "auto_calibrated_templates": len(self.template_manager.auto_calibrated)
        }
    
    def cleanup(self) -> None:
        """Nettoie les ressources du module"""
        try:
            self.logger.info("Arrêt du module template matching")
            
            # Arrêt du pool de threads
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # Sauvegarde des métadonnées
            self.template_manager.save_metadata()
            
            # Nettoyage des caches
            self.results_cache.clear()
            self.matching_engine.clear_cache()
            
            self.status = ModuleStatus.INACTIVE
            self.logger.info("Module template matching arrêté")
            
        except Exception as e:
            self.logger.error(f"Erreur arrêt template matcher: {e}")
    
    def get_template_info(self) -> Dict[str, Any]:
        """Retourne des informations sur les templates chargés"""
        try:
            info = {
                "categories": {},
                "total_templates": 0,
                "auto_calibrated": list(self.template_manager.auto_calibrated)
            }
            
            for category, template_names in self.template_manager.categories.items():
                templates_info = []
                
                for template_name in template_names:
                    template = self.template_manager.get_template(category, template_name)
                    if template:
                        templates_info.append({
                            "name": template_name,
                            "threshold": template.threshold,
                            "scales": template.scales,
                            "rotations": template.rotations,
                            "max_matches": template.max_matches,
                            "image_size": template.image.shape[:2] if template.image is not None else None,
                            "has_mask": template.mask is not None
                        })
                
                info["categories"][category] = {
                    "count": len(templates_info),
                    "templates": templates_info
                }
                info["total_templates"] += len(templates_info)
            
            return info
            
        except Exception as e:
            self.logger.error(f"Erreur récupération info templates: {e}")
            return {}