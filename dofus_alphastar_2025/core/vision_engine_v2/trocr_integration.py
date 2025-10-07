"""
TrOCR Integration - Transformer-based OCR pour DOFUS
Reconnaissance de texte avancée optimisée pour interface DOFUS Unity
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import logging
import time
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

# TrOCR imports
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from transformers import AutoProcessor, AutoModelForVision2Seq
    HAS_TROCR = True
except ImportError:
    HAS_TROCR = False

# Alternative OCR fallback
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

from config import config, get_device

logger = logging.getLogger(__name__)

@dataclass
class TextDetection:
    """Texte détecté avec métadonnées"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    language: str = "fr"
    text_type: str = "unknown"  # "damage", "name", "chat", "ui", "number"
    color: Optional[Tuple[int, int, int]] = None
    font_size: Optional[int] = None

@dataclass
class DofusTextContext:
    """Contexte de texte DOFUS avec sémantique"""
    original_detection: TextDetection
    semantic_type: str  # "player_name", "monster_name", "damage_number", "chat_message", "ui_label"
    parsed_value: Optional[Union[str, int, float]] = None
    game_significance: str = "low"  # "low", "medium", "high", "critical"

class TrOCRProcessor:
    """Processeur TrOCR optimisé pour AMD et DOFUS"""

    def __init__(self, model_name: str = "microsoft/trocr-large-printed"):
        self.model_name = model_name
        self.device = get_device()

        # Models
        self.processor = None
        self.model = None
        self.is_loaded = False

        # Fallback OCR
        self.easyocr_reader = None

        # Performance tracking
        self.processing_times = []
        self.total_processed = 0

        # Text cache
        self._text_cache = {}
        self._cache_max_size = 50

        # Load models
        self._load_models()

        logger.info(f"TrOCR Processor initialisé avec {model_name}")

    def _load_models(self):
        """Charge les modèles TrOCR et fallbacks"""
        if HAS_TROCR:
            try:
                # TrOCR principal
                self.processor = TrOCRProcessor.from_pretrained(self.model_name)
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

                # Déplacer vers device AMD
                if config.amd.use_directml or torch.cuda.is_available():
                    self.model = self.model.to(self.device)

                # Optimisations AMD
                if config.amd.use_mixed_precision:
                    self.model = self.model.half()

                # Mode évaluation
                self.model.eval()

                self.is_loaded = True
                logger.info("TrOCR chargé avec succès")

            except Exception as e:
                logger.error(f"Erreur chargement TrOCR: {e}")
                self.is_loaded = False

        # Fallback EasyOCR
        if HAS_EASYOCR and not self.is_loaded:
            try:
                self.easyocr_reader = easyocr.Reader(['fr', 'en'], gpu=torch.cuda.is_available())
                logger.info("EasyOCR chargé comme fallback")
            except Exception as e:
                logger.warning(f"Erreur EasyOCR: {e}")

    def extract_text_from_image(self,
                               image: Union[np.ndarray, Image.Image],
                               confidence_threshold: float = 0.7) -> List[TextDetection]:
        """Extrait le texte d'une image complète"""

        start_time = time.time()

        # Préparation image
        if isinstance(image, np.ndarray):
            # OpenCV BGR -> RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Cache check
        image_hash = hash(pil_image.tobytes())
        if image_hash in self._text_cache:
            logger.debug("Utilisation cache OCR")
            return self._text_cache[image_hash]

        detections = []

        # TrOCR si disponible
        if self.is_loaded and self.model is not None:
            detections = self._extract_with_trocr(pil_image, confidence_threshold)

        # Fallback EasyOCR
        if not detections and self.easyocr_reader:
            detections = self._extract_with_easyocr(pil_image, confidence_threshold)

        # Filtrer par confiance
        detections = [d for d in detections if d.confidence >= confidence_threshold]

        # Post-processing DOFUS
        detections = self._postprocess_dofus_text(detections)

        # Cache result
        if len(self._text_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._text_cache))
            del self._text_cache[oldest_key]

        self._text_cache[image_hash] = detections

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.total_processed += 1

        logger.debug(f"OCR: {len(detections)} textes détectés en {processing_time:.3f}s")

        return detections

    def _extract_with_trocr(self,
                           image: Image.Image,
                           confidence_threshold: float) -> List[TextDetection]:
        """Extraction avec TrOCR"""
        detections = []

        try:
            with torch.no_grad():
                # Préprocessing
                pixel_values = self.processor(image, return_tensors="pt").pixel_values

                if config.amd.use_directml or torch.cuda.is_available():
                    pixel_values = pixel_values.to(self.device)

                if config.amd.use_mixed_precision:
                    pixel_values = pixel_values.half()

                # Génération
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True
                )

                # Décodage
                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                if generated_text.strip():
                    # TrOCR traite l'image entière, on crée une détection globale
                    detection = TextDetection(
                        text=generated_text.strip(),
                        confidence=0.8,  # TrOCR ne donne pas de score explicite
                        bbox=(0, 0, image.width, image.height),
                        language="fr"
                    )
                    detections.append(detection)

        except Exception as e:
            logger.error(f"Erreur TrOCR: {e}")

        return detections

    def _extract_with_easyocr(self,
                             image: Image.Image,
                             confidence_threshold: float) -> List[TextDetection]:
        """Extraction avec EasyOCR (fallback)"""
        detections = []

        try:
            # Conversion pour EasyOCR
            image_np = np.array(image)

            # OCR avec EasyOCR
            results = self.easyocr_reader.readtext(image_np)

            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold:
                    # Bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    points = np.array(bbox).astype(int)
                    x1, y1 = points.min(axis=0)
                    x2, y2 = points.max(axis=0)

                    detection = TextDetection(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        language="fr"
                    )
                    detections.append(detection)

        except Exception as e:
            logger.error(f"Erreur EasyOCR: {e}")

        return detections

    def _postprocess_dofus_text(self, detections: List[TextDetection]) -> List[TextDetection]:
        """Post-traitement spécifique DOFUS"""
        processed = []

        for detection in detections:
            # Nettoyage texte
            text = detection.text
            text = re.sub(r'[^\w\s\-\+\.,\!\?\(\)]', '', text)  # Caractères valides
            text = text.strip()

            if len(text) < 1:
                continue

            # Classification du type de texte
            text_type = self._classify_dofus_text(text, detection.bbox)

            # Mise à jour detection
            detection.text = text
            detection.text_type = text_type

            processed.append(detection)

        return processed

    def _classify_dofus_text(self, text: str, bbox: Tuple[int, int, int, int]) -> str:
        """Classifie le type de texte DOFUS"""

        # Nombres (dégâts, points de vie, etc.)
        if re.match(r'^\d+$', text):
            if len(text) <= 4:
                return "number"
            else:
                return "big_number"

        # Dégâts avec signe
        if re.match(r'^[\-\+]?\d+$', text):
            return "damage"

        # Noms de joueurs (commencent par majuscule)
        if re.match(r'^[A-Z][a-z\-]+$', text):
            return "name"

        # Messages de chat (phrases)
        if len(text) > 15 and ' ' in text:
            return "chat"

        # Labels UI courts
        if len(text) <= 10:
            return "ui_label"

        return "unknown"

    def extract_text_from_regions(self,
                                 image: Union[np.ndarray, Image.Image],
                                 regions: List[Tuple[int, int, int, int]],
                                 region_names: Optional[List[str]] = None) -> Dict[str, List[TextDetection]]:
        """Extrait le texte de régions spécifiques"""

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        results = {}

        for i, (x, y, w, h) in enumerate(regions):
            region_name = region_names[i] if region_names and i < len(region_names) else f"region_{i}"

            # Extraire ROI
            roi = image.crop((x, y, x + w, y + h))

            # OCR sur la région
            detections = self.extract_text_from_image(roi, confidence_threshold=0.6)

            # Ajuster coordonnées à l'image complète
            for detection in detections:
                bbox_x, bbox_y, bbox_w, bbox_h = detection.bbox
                detection.bbox = (x + bbox_x, y + bbox_y, bbox_w, bbox_h)

            results[region_name] = detections

        return results

    def get_performance_metrics(self) -> Dict[str, float]:
        """Retourne les métriques de performance"""
        if not self.processing_times:
            return {"avg_time": 0.0, "total_processed": 0}

        return {
            "avg_processing_time": np.mean(self.processing_times),
            "min_processing_time": np.min(self.processing_times),
            "max_processing_time": np.max(self.processing_times),
            "total_processed": self.total_processed,
            "cache_size": len(self._text_cache),
            "model_loaded": self.is_loaded
        }

class DofusTextRecognizer:
    """Reconnaisseur de texte spécialisé pour DOFUS Unity"""

    def __init__(self):
        self.trocr_processor = TrOCRProcessor()

        # Régions texte DOFUS (coordonnées relatives)
        self.text_regions = {
            "player_stats": (0.02, 0.02, 0.25, 0.15),    # Stats joueur
            "chat_area": (0.02, 0.6, 0.35, 0.35),        # Zone chat
            "combat_log": (0.4, 0.7, 0.3, 0.25),         # Log de combat
            "monster_names": (0.2, 0.15, 0.6, 0.5),      # Zone centrale pour noms
            "damage_numbers": (0.15, 0.1, 0.7, 0.7),     # Zone large pour dégâts
            "ui_buttons": (0.75, 0.8, 0.23, 0.18),       # Boutons interface
            "quest_tracker": (0.75, 0.25, 0.23, 0.4),    # Suivi de quêtes
        }

        # Patterns de texte DOFUS
        self.text_patterns = {
            "player_name": r"^[A-Z][a-z\-]+$",
            "damage_number": r"^[\-\+]?\d{1,4}$",
            "health_number": r"^\d{1,4}\/\d{1,4}$",
            "level": r"^Niveau \d{1,3}$",
            "spell_name": r"^[A-Z][a-z ]+ [IVX]+$",
            "quest_name": r"^\[[^\]]+\].*"
        }

        logger.info("DOFUS Text Recognizer initialisé")

    def analyze_screenshot_text(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Analyse complète du texte dans un screenshot DOFUS"""

        analysis_start = time.time()

        # Conversion image
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            pil_image = Image.fromarray(image)
        else:
            w, h = image.size
            pil_image = image

        # Extraction par région
        region_results = {}
        all_contexts = []

        for region_name, (rx, ry, rw, rh) in self.text_regions.items():
            # Coordonnées absolues
            x1, y1 = int(rx * w), int(ry * h)
            x2, y2 = int((rx + rw) * w), int((ry + rh) * h)

            # ROI
            roi = pil_image.crop((x1, y1, x2, y2))

            # OCR
            detections = self.trocr_processor.extract_text_from_image(roi, confidence_threshold=0.6)

            # Ajuster coordonnées et créer contextes
            region_contexts = []
            for detection in detections:
                # Ajuster bbox à image complète
                bbox_x, bbox_y, bbox_w, bbox_h = detection.bbox
                detection.bbox = (x1 + bbox_x, y1 + bbox_y, bbox_w, bbox_h)

                # Créer contexte sémantique
                context = self._create_text_context(detection, region_name)
                region_contexts.append(context)
                all_contexts.append(context)

            region_results[region_name] = region_contexts

        # Analyse globale OCR (pour textes manqués)
        global_detections = self.trocr_processor.extract_text_from_image(pil_image, confidence_threshold=0.7)
        global_contexts = [self._create_text_context(d, "global") for d in global_detections]

        analysis_time = time.time() - analysis_start

        # Synthèse des informations extraites
        game_info = self._synthesize_game_info(all_contexts + global_contexts)

        return {
            "region_results": region_results,
            "global_contexts": global_contexts,
            "game_info": game_info,
            "total_detections": len(all_contexts) + len(global_contexts),
            "analysis_time": analysis_time,
            "timestamp": time.time()
        }

    def _create_text_context(self, detection: TextDetection, region_name: str) -> DofusTextContext:
        """Crée un contexte sémantique pour un texte détecté"""

        text = detection.text
        semantic_type = "unknown"
        parsed_value = text
        significance = "low"

        # Classification sémantique basée sur région et pattern
        if region_name == "player_stats":
            if re.match(self.text_patterns["health_number"], text):
                semantic_type = "health_display"
                significance = "critical"
                # Parser HP actuel/max
                try:
                    current, maximum = map(int, text.split('/'))
                    parsed_value = {"current": current, "max": maximum, "percentage": current/maximum}
                except:
                    pass

            elif re.match(self.text_patterns["level"], text):
                semantic_type = "player_level"
                significance = "medium"
                try:
                    parsed_value = int(re.findall(r'\d+', text)[0])
                except:
                    pass

        elif region_name == "chat_area":
            semantic_type = "chat_message"
            significance = "medium" if len(text) > 20 else "low"

        elif region_name == "combat_log":
            semantic_type = "combat_log_entry"
            significance = "high"

        elif region_name in ["monster_names", "global"]:
            if re.match(self.text_patterns["player_name"], text):
                semantic_type = "entity_name"
                significance = "high"

        elif region_name == "damage_numbers":
            if re.match(self.text_patterns["damage_number"], text):
                semantic_type = "damage_number"
                significance = "critical"
                try:
                    parsed_value = int(text.replace('+', '').replace('-', ''))
                except:
                    pass

        elif region_name == "quest_tracker":
            if re.match(self.text_patterns["quest_name"], text):
                semantic_type = "quest_objective"
                significance = "high"

        return DofusTextContext(
            original_detection=detection,
            semantic_type=semantic_type,
            parsed_value=parsed_value,
            game_significance=significance
        )

    def _synthesize_game_info(self, contexts: List[DofusTextContext]) -> Dict[str, Any]:
        """Synthétise les informations de jeu à partir des contextes texte"""

        info = {
            "player_info": {},
            "combat_info": {},
            "chat_info": {},
            "quest_info": {},
            "entities": []
        }

        for context in contexts:
            if context.semantic_type == "health_display" and isinstance(context.parsed_value, dict):
                info["player_info"]["health"] = context.parsed_value

            elif context.semantic_type == "player_level" and isinstance(context.parsed_value, int):
                info["player_info"]["level"] = context.parsed_value

            elif context.semantic_type == "damage_number":
                if "damage_events" not in info["combat_info"]:
                    info["combat_info"]["damage_events"] = []
                info["combat_info"]["damage_events"].append({
                    "damage": context.parsed_value,
                    "position": context.original_detection.bbox
                })

            elif context.semantic_type == "entity_name":
                info["entities"].append({
                    "name": context.original_detection.text,
                    "position": context.original_detection.bbox
                })

            elif context.semantic_type == "chat_message":
                if "recent_messages" not in info["chat_info"]:
                    info["chat_info"]["recent_messages"] = []
                info["chat_info"]["recent_messages"].append(context.original_detection.text)

            elif context.semantic_type == "quest_objective":
                if "active_objectives" not in info["quest_info"]:
                    info["quest_info"]["active_objectives"] = []
                info["quest_info"]["active_objectives"].append(context.original_detection.text)

        return info

    def extract_specific_text_type(self,
                                  image: Union[np.ndarray, Image.Image],
                                  text_type: str) -> List[DofusTextContext]:
        """Extrait un type spécifique de texte"""

        # Analyse complète
        analysis = self.analyze_screenshot_text(image)

        # Filtrer par type
        specific_contexts = []
        for region_contexts in analysis["region_results"].values():
            specific_contexts.extend([c for c in region_contexts if c.semantic_type == text_type])

        for context in analysis["global_contexts"]:
            if context.semantic_type == text_type:
                specific_contexts.append(context)

        return specific_contexts

# Factory functions
def create_trocr_processor(model_name: str = "microsoft/trocr-large-printed") -> TrOCRProcessor:
    """Crée un processeur TrOCR configuré"""
    return TrOCRProcessor(model_name)

def create_dofus_text_recognizer() -> DofusTextRecognizer:
    """Crée un reconnaisseur de texte DOFUS"""
    return DofusTextRecognizer()