"""
OCR Engine - Reconnaissance de texte optimisée pour DOFUS Unity
Utilise Tesseract + EasyOCR + PaddleOCR pour maximum précision

Fonctionnalités:
- Multi-engine OCR avec fallback
- Détection multilingue (FR/EN)
- Cache intelligent résultats
- Optimisation GPU pour EasyOCR
- Preprocessing images spécialisé DOFUS
"""

import time
import re
import threading
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

# Computer Vision
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# OCR Engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract non disponible")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR non disponible")

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("PaddleOCR non disponible")

@dataclass
class OCRResult:
    """Résultat reconnaissance OCR"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    engine: str
    processing_time: float
    language: str = "unknown"

@dataclass
class TextRegion:
    """Région de texte dans l'image"""
    x: int
    y: int
    width: int
    height: int
    name: str
    expected_content: str = "any"  # any, numeric, text, mixed

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

class OCREngine:
    """Moteur OCR multi-engine pour DOFUS"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Configuration OCR
        self.languages = ["fra", "eng"]  # Français + Anglais
        self.primary_engine = "tesseract"  # tesseract, easyocr, paddleocr
        self.fallback_engines = ["easyocr", "paddleocr"]
        self.confidence_threshold = 0.6

        # Engines instances
        self.tesseract_config = None
        self.easyocr_reader = None
        self.paddleocr_reader = None

        # Cache résultats
        self.enable_cache = True
        self.cache_duration = 1.0  # 1 seconde
        self.results_cache: Dict[str, Tuple[List[OCRResult], float]] = {}

        # Preprocessing configuration
        self.preprocessing_config = {
            "resize_factor": 2.0,  # Agrandir pour meilleure précision
            "denoise": True,
            "sharpen": True,
            "contrast_enhancement": 1.2,
            "brightness_adjustment": 0.1
        }

        # Régions prédéfinies DOFUS
        self.dofus_regions = self._initialize_dofus_regions()

        # Threading
        self.lock = threading.Lock()

        self.logger.info("OCREngine initialisé")

    def initialize(self) -> bool:
        """Initialise les moteurs OCR"""
        try:
            success_count = 0

            # Tesseract
            if TESSERACT_AVAILABLE:
                if self._initialize_tesseract():
                    success_count += 1
                    self.logger.info("Tesseract initialisé")
                else:
                    self.logger.warning("Échec initialisation Tesseract")

            # EasyOCR
            if EASYOCR_AVAILABLE:
                if self._initialize_easyocr():
                    success_count += 1
                    self.logger.info("EasyOCR initialisé")
                else:
                    self.logger.warning("Échec initialisation EasyOCR")

            # PaddleOCR
            if PADDLEOCR_AVAILABLE:
                if self._initialize_paddleocr():
                    success_count += 1
                    self.logger.info("PaddleOCR initialisé")
                else:
                    self.logger.warning("Échec initialisation PaddleOCR")

            if success_count > 0:
                self.logger.info(f"OCREngine: {success_count} moteurs disponibles")
                return True
            else:
                self.logger.error("Aucun moteur OCR disponible")
                return False

        except Exception as e:
            self.logger.error(f"Erreur initialisation OCREngine: {e}")
            return False

    def _initialize_tesseract(self) -> bool:
        """Initialise Tesseract OCR"""
        try:
            # Configuration optimisée
            self.tesseract_config = (
                "--oem 3 --psm 6 "  # OCR Engine Mode 3, Page Segmentation Mode 6
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789àâäéèêëïîôùûüÿç.,!?:;-+%/ "
            )

            # Test basique
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            test_text = pytesseract.image_to_string(test_image, config=self.tesseract_config)

            return True
        except Exception as e:
            self.logger.debug(f"Tesseract init failed: {e}")
            return False

    def _initialize_easyocr(self) -> bool:
        """Initialise EasyOCR"""
        try:
            # GPU si disponible, sinon CPU
            gpu_available = self._check_gpu_availability()

            self.easyocr_reader = easyocr.Reader(
                ['fr', 'en'],
                gpu=gpu_available,
                model_storage_directory='data/models/easyocr',
                download_enabled=True
            )

            return True
        except Exception as e:
            self.logger.debug(f"EasyOCR init failed: {e}")
            return False

    def _initialize_paddleocr(self) -> bool:
        """Initialise PaddleOCR"""
        try:
            # GPU si disponible
            use_gpu = self._check_gpu_availability()

            self.paddleocr_reader = paddleocr.PaddleOCR(
                use_angle_cls=True,
                lang='fr',  # Français principal
                use_gpu=use_gpu,
                show_log=False
            )

            return True
        except Exception as e:
            self.logger.debug(f"PaddleOCR init failed: {e}")
            return False

    def _check_gpu_availability(self) -> bool:
        """Vérifie disponibilité GPU pour OCR"""
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            return False

    def _initialize_dofus_regions(self) -> Dict[str, TextRegion]:
        """Initialise régions de texte prédéfinies DOFUS"""
        return {
            # Interface principale
            "chat": TextRegion(10, 500, 600, 150, "chat", "text"),
            "hp_value": TextRegion(20, 20, 100, 25, "hp_value", "numeric"),
            "mp_value": TextRegion(140, 20, 100, 25, "mp_value", "numeric"),
            "level": TextRegion(50, 50, 50, 20, "level", "numeric"),

            # Combat
            "spell_names": TextRegion(300, 600, 400, 100, "spell_names", "text"),
            "damage_values": TextRegion(0, 0, 1920, 1080, "damage_values", "numeric"),

            # Inventaire
            "item_names": TextRegion(800, 200, 400, 600, "item_names", "text"),
            "item_quantities": TextRegion(1150, 200, 50, 600, "item_quantities", "numeric"),

            # Map et position
            "map_name": TextRegion(10, 80, 300, 30, "map_name", "text"),
            "coordinates": TextRegion(10, 110, 100, 20, "coordinates", "mixed"),

            # Quêtes
            "quest_objectives": TextRegion(1400, 100, 400, 300, "quest_objectives", "text"),
            "quest_progress": TextRegion(1650, 100, 150, 300, "quest_progress", "mixed"),
        }

    def preprocess_image(self, image: np.ndarray, region_type: str = "general") -> np.ndarray:
        """Préprocessing image optimisé pour OCR"""
        try:
            # Configuration selon type de région
            config = self.preprocessing_config.copy()

            if region_type == "numeric":
                # Optimisé pour chiffres
                config["resize_factor"] = 3.0
                config["contrast_enhancement"] = 1.5
                config["sharpen"] = True
            elif region_type == "damage_values":
                # Texte de combat (souvent coloré)
                config["resize_factor"] = 2.5
                config["denoise"] = True

            # 1. Redimensionner
            if config["resize_factor"] != 1.0:
                height, width = image.shape[:2]
                new_width = int(width * config["resize_factor"])
                new_height = int(height * config["resize_factor"])
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # 2. Convertir en niveaux de gris
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # 3. Débruitage
            if config["denoise"]:
                gray = cv2.fastNlMeansDenoising(gray)

            # 4. Amélioration contraste
            if config["contrast_enhancement"] != 1.0:
                gray = cv2.convertScaleAbs(gray,
                                         alpha=config["contrast_enhancement"],
                                         beta=config["brightness_adjustment"] * 255)

            # 5. Accentuation netteté
            if config["sharpen"]:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                gray = cv2.filter2D(gray, -1, kernel)

            # 6. Binarisation adaptative
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            return binary

        except Exception as e:
            self.logger.error(f"Erreur preprocessing: {e}")
            return image

    def extract_text_tesseract(self, image: np.ndarray, region_type: str = "general") -> List[OCRResult]:
        """Extraction Tesseract"""
        if not TESSERACT_AVAILABLE or self.tesseract_config is None:
            return []

        try:
            start_time = time.time()

            # Preprocessing
            processed_image = self.preprocess_image(image, region_type)

            # OCR avec données détaillées
            data = pytesseract.image_to_data(
                processed_image,
                config=self.tesseract_config,
                lang='+'.join(self.languages),
                output_type=pytesseract.Output.DICT
            )

            results = []
            n_boxes = len(data['level'])

            for i in range(n_boxes):
                confidence = float(data['conf'][i])
                text = data['text'][i].strip()

                if confidence > self.confidence_threshold and text:
                    # Bounding box
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                    result = OCRResult(
                        text=text,
                        confidence=confidence / 100.0,  # Normaliser 0-1
                        bbox=(x, y, x + w, y + h),
                        engine="tesseract",
                        processing_time=time.time() - start_time,
                        language="multi"
                    )
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Erreur Tesseract: {e}")
            return []

    def extract_text_easyocr(self, image: np.ndarray, region_type: str = "general") -> List[OCRResult]:
        """Extraction EasyOCR"""
        if not EASYOCR_AVAILABLE or self.easyocr_reader is None:
            return []

        try:
            start_time = time.time()

            # EasyOCR préfère images couleur
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # OCR
            detections = self.easyocr_reader.readtext(image)

            results = []
            for detection in detections:
                bbox_points, text, confidence = detection

                if confidence > self.confidence_threshold and text.strip():
                    # Convertir bbox points vers rectangle
                    x_coords = [point[0] for point in bbox_points]
                    y_coords = [point[1] for point in bbox_points]
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))

                    result = OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        engine="easyocr",
                        processing_time=time.time() - start_time,
                        language="multi"
                    )
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Erreur EasyOCR: {e}")
            return []

    def extract_text_paddleocr(self, image: np.ndarray, region_type: str = "general") -> List[OCRResult]:
        """Extraction PaddleOCR"""
        if not PADDLEOCR_AVAILABLE or self.paddleocr_reader is None:
            return []

        try:
            start_time = time.time()

            # PaddleOCR accepte images couleur
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # OCR
            detections = self.paddleocr_reader.ocr(image, cls=True)

            results = []
            if detections and detections[0]:
                for detection in detections[0]:
                    bbox_points, (text, confidence) = detection

                    if confidence > self.confidence_threshold and text.strip():
                        # Convertir bbox
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]
                        x1, y1 = int(min(x_coords)), int(min(y_coords))
                        x2, y2 = int(max(x_coords)), int(max(y_coords))

                        result = OCRResult(
                            text=text.strip(),
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            engine="paddleocr",
                            processing_time=time.time() - start_time,
                            language="fr"
                        )
                        results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Erreur PaddleOCR: {e}")
            return []

    def extract_text_multi_engine(self, image: np.ndarray, region_type: str = "general") -> List[OCRResult]:
        """Extraction multi-engine avec consensus"""
        try:
            # Vérifier cache
            cache_key = self._generate_cache_key(image, region_type)
            if self.enable_cache and cache_key in self.results_cache:
                cached_results, timestamp = self.results_cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return cached_results

            all_results = []

            # Engine principal
            if self.primary_engine == "tesseract":
                primary_results = self.extract_text_tesseract(image, region_type)
            elif self.primary_engine == "easyocr":
                primary_results = self.extract_text_easyocr(image, region_type)
            elif self.primary_engine == "paddleocr":
                primary_results = self.extract_text_paddleocr(image, region_type)
            else:
                primary_results = []

            all_results.extend(primary_results)

            # Engines fallback si résultats insuffisants
            if len(primary_results) == 0 or max([r.confidence for r in primary_results], default=0) < 0.8:
                for engine in self.fallback_engines:
                    if engine == "tesseract" and engine != self.primary_engine:
                        fallback_results = self.extract_text_tesseract(image, region_type)
                    elif engine == "easyocr" and engine != self.primary_engine:
                        fallback_results = self.extract_text_easyocr(image, region_type)
                    elif engine == "paddleocr" and engine != self.primary_engine:
                        fallback_results = self.extract_text_paddleocr(image, region_type)
                    else:
                        continue

                    all_results.extend(fallback_results)

            # Fusion et déduplication
            final_results = self._merge_ocr_results(all_results)

            # Cache résultats
            if self.enable_cache:
                with self.lock:
                    self.results_cache[cache_key] = (final_results, time.time())

            return final_results

        except Exception as e:
            self.logger.error(f"Erreur extraction multi-engine: {e}")
            return []

    def _merge_ocr_results(self, results: List[OCRResult]) -> List[OCRResult]:
        """Fusionne et déduplique résultats OCR"""
        if not results:
            return []

        # Grouper par position similaire
        groups = []
        for result in results:
            added_to_group = False
            for group in groups:
                # Vérifier overlap des bounding boxes
                if self._bbox_overlap(result.bbox, group[0].bbox) > 0.7:
                    group.append(result)
                    added_to_group = True
                    break

            if not added_to_group:
                groups.append([result])

        # Sélectionner meilleur résultat par groupe
        merged_results = []
        for group in groups:
            # Prendre résultat avec plus haute confiance
            best_result = max(group, key=lambda r: r.confidence)
            merged_results.append(best_result)

        return merged_results

    def _bbox_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calcule overlap entre deux bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _generate_cache_key(self, image: np.ndarray, region_type: str) -> str:
        """Génère clé cache pour image"""
        # Hash simple basé sur checksum image
        image_hash = hash(image.tobytes())
        return f"{image_hash}_{region_type}_{image.shape}"

    def extract_text_from_region(self, image: np.ndarray, region: Union[str, TextRegion]) -> List[OCRResult]:
        """Extrait texte d'une région spécifique"""
        try:
            # Résoudre région
            if isinstance(region, str):
                if region in self.dofus_regions:
                    text_region = self.dofus_regions[region]
                else:
                    self.logger.warning(f"Région inconnue: {region}")
                    return []
            else:
                text_region = region

            # Extraire région de l'image
            h, w = image.shape[:2]
            x1 = max(0, text_region.x)
            y1 = max(0, text_region.y)
            x2 = min(w, text_region.x + text_region.width)
            y2 = min(h, text_region.y + text_region.height)

            if x2 <= x1 or y2 <= y1:
                self.logger.warning("Région invalide")
                return []

            region_image = image[y1:y2, x1:x2]

            # OCR sur région
            results = self.extract_text_multi_engine(region_image, text_region.expected_content)

            # Ajuster coordonnées au système global
            for result in results:
                bbox = result.bbox
                result.bbox = (bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1)

            return results

        except Exception as e:
            self.logger.error(f"Erreur extraction région: {e}")
            return []

    def extract_numbers(self, image: np.ndarray, region: Optional[Union[str, TextRegion]] = None) -> List[Tuple[int, float]]:
        """Extrait nombres avec confiance"""
        if region:
            results = self.extract_text_from_region(image, region)
        else:
            results = self.extract_text_multi_engine(image, "numeric")

        numbers = []
        for result in results:
            # Extraire nombres du texte
            number_matches = re.findall(r'\d+', result.text)
            for match in number_matches:
                try:
                    number = int(match)
                    numbers.append((number, result.confidence))
                except ValueError:
                    continue

        return numbers

    def get_cache_stats(self) -> Dict[str, Any]:
        """Statistiques cache"""
        with self.lock:
            return {
                "cache_enabled": self.enable_cache,
                "cache_size": len(self.results_cache),
                "cache_duration": self.cache_duration
            }

    def clear_cache(self):
        """Vide le cache"""
        with self.lock:
            self.results_cache.clear()
        self.logger.info("Cache OCR vidé")

    def get_available_engines(self) -> List[str]:
        """Retourne engines disponibles"""
        engines = []
        if TESSERACT_AVAILABLE and self.tesseract_config:
            engines.append("tesseract")
        if EASYOCR_AVAILABLE and self.easyocr_reader:
            engines.append("easyocr")
        if PADDLEOCR_AVAILABLE and self.paddleocr_reader:
            engines.append("paddleocr")
        return engines

# Factory function
def create_ocr_engine() -> OCREngine:
    """Crée instance OCREngine configurée"""
    ocr = OCREngine()
    if ocr.initialize():
        return ocr
    else:
        raise RuntimeError("Impossible d'initialiser OCREngine")

# Test de base
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        ocr = create_ocr_engine()
        print(f"Engines disponibles: {ocr.get_available_engines()}")

        # Test avec image simple
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "Test 123", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        results = ocr.extract_text_multi_engine(test_image)
        print(f"Résultats OCR: {len(results)}")
        for result in results:
            print(f"  - '{result.text}' (confiance: {result.confidence:.2f})")

    except Exception as e:
        print(f"Erreur test: {e}")