"""
Annotation Tool - Pipeline d'annotation pour Dataset AGA

Fonctionnalités:
1. Extraction de frames depuis sessions HDF5
2. Interface annotation (CLI ou export pour CVAT/Label Studio)
3. Génération annotations YOLO format
4. Validation qualité dataset

Classes d'objets pour Dofus:
- monster: Monstres (adversaires)
- npc: NPCs (vendeurs, quêtes)
- player: Joueur principal
- resource: Ressources récoltables (arbres, minerais)
- chest: Coffres
- door: Portes/téléporteurs
- item: Items au sol
- ui_element: Éléments d'interface
"""

import cv2
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

# Classes d'objets Dofus
DOFUS_CLASSES = {
    0: 'monster',
    1: 'npc',
    2: 'player',
    3: 'resource',
    4: 'chest',
    5: 'door',
    6: 'item',
    7: 'ui_element'
}

CLASS_COLORS = {
    'monster': (0, 0, 255),      # Rouge
    'npc': (255, 0, 0),          # Bleu
    'player': (0, 255, 0),       # Vert
    'resource': (255, 255, 0),   # Cyan
    'chest': (0, 255, 255),      # Jaune
    'door': (255, 0, 255),       # Magenta
    'item': (128, 128, 128),     # Gris
    'ui_element': (255, 255, 255) # Blanc
}

@dataclass
class Annotation:
    """Annotation d'un objet dans une frame"""
    frame_index: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # x_center, y_center, width, height (normalized 0-1)
    confidence: float = 1.0

class AnnotationTool:
    """
    Outil d'annotation pour dataset Dofus

    Usage:
        tool = AnnotationTool("./datasets/sessions/session_xxx.h5")
        tool.export_frames(output_dir="./datasets/frames", stride=10)
        tool.export_cvat_format("./datasets/annotations/cvat_project.xml")
    """

    def __init__(self, session_path: str):
        """
        Initialise l'outil d'annotation

        Args:
            session_path: Chemin vers fichier HDF5 de session
        """
        self.session_path = Path(session_path)

        if not self.session_path.exists():
            raise FileNotFoundError(f"Session non trouvée: {session_path}")

        # Load session metadata
        with h5py.File(self.session_path, 'r') as f:
            self.metadata = dict(f['metadata'].attrs)
            self.num_frames = len(f['frames']) if 'frames' in f else 0

        self.annotations: List[Annotation] = []

        logger.info(f"AnnotationTool initialisé - Session: {self.session_path.name}")
        logger.info(f"   Frames: {self.num_frames}")

    def export_frames(
        self,
        output_dir: str,
        stride: int = 10,
        max_frames: Optional[int] = None,
        quality: int = 95
    ) -> List[Path]:
        """
        Exporte frames depuis HDF5 vers images individuelles

        Args:
            output_dir: Répertoire de sortie
            stride: Prendre 1 frame tous les N frames
            max_frames: Nombre max de frames à exporter
            quality: Qualité JPEG (0-100)

        Returns:
            Liste des chemins des frames exportées
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Export frames vers {output_dir}")
        logger.info(f"   Stride: {stride}, Max: {max_frames}, Quality: {quality}")

        exported_paths = []

        with h5py.File(self.session_path, 'r') as f:
            frames = f['frames']
            timestamps = f['frame_timestamps']

            num_to_export = min(
                len(frames) // stride,
                max_frames if max_frames else len(frames)
            )

            for i in range(0, len(frames), stride):
                if len(exported_paths) >= num_to_export:
                    break

                frame = frames[i]
                timestamp = timestamps[i]

                # Generate filename
                frame_id = f"{self.session_path.stem}_frame_{i:06d}"
                output_file = output_path / f"{frame_id}.jpg"

                # Save frame
                cv2.imwrite(
                    str(output_file),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )

                exported_paths.append(output_file)

                if (i // stride) % 100 == 0:
                    logger.info(f"   Exported {len(exported_paths)}/{num_to_export} frames")

        logger.info(f"✅ {len(exported_paths)} frames exportées")
        return exported_paths

    def export_cvat_format(self, output_path: str):
        """
        Exporte annotations au format CVAT XML

        CVAT (Computer Vision Annotation Tool) est un outil open-source
        d'annotation développé par Intel.

        Installation: docker run -it -p 8080:8080 cvat/server
        """

        # TODO: Implémenter export CVAT XML
        logger.warning("Export CVAT format - TODO")
        pass

    def export_yolo_format(
        self,
        output_dir: str,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1
    ):
        """
        Exporte annotations au format YOLO

        Format YOLO:
        - 1 fichier .txt par image
        - Chaque ligne: <class_id> <x_center> <y_center> <width> <height>
        - Coordonnées normalisées (0-1)

        Args:
            output_dir: Répertoire de sortie
            train_split: % training set
            val_split: % validation set
            test_split: % test set
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create YOLO directory structure
        (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_path / "images" / "test").mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / "test").mkdir(parents=True, exist_ok=True)

        # Create data.yaml
        data_yaml = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(DOFUS_CLASSES),
            'names': list(DOFUS_CLASSES.values())
        }

        with open(output_path / "data.yaml", 'w') as f:
            import yaml
            yaml.dump(data_yaml, f)

        logger.info(f"✅ Structure YOLO créée: {output_dir}")
        logger.info(f"   Classes: {len(DOFUS_CLASSES)}")

    def visualize_annotations(
        self,
        frame_index: int,
        annotations: Optional[List[Annotation]] = None,
        show: bool = True,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualise annotations sur une frame

        Args:
            frame_index: Index de la frame à visualiser
            annotations: Annotations à afficher (ou self.annotations si None)
            show: Afficher avec cv2.imshow
            output_path: Sauvegarder vers fichier

        Returns:
            Frame avec annotations dessinées
        """

        # Load frame
        with h5py.File(self.session_path, 'r') as f:
            frame = f['frames'][frame_index].copy()

        height, width = frame.shape[:2]

        # Get annotations for this frame
        if annotations is None:
            annotations = [a for a in self.annotations if a.frame_index == frame_index]

        # Draw annotations
        for ann in annotations:
            # Convert normalized bbox to pixel coordinates
            x_center, y_center, w, h = ann.bbox
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)

            # Color based on class
            color = CLASS_COLORS.get(ann.class_name, (255, 255, 255))

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{ann.class_name} ({ann.confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Show
        if show:
            cv2.imshow("Annotations", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save
        if output_path:
            cv2.imwrite(output_path, frame)

        return frame

    def auto_annotate_synthetic(
        self,
        frame_index: int,
        detection_method: str = "color"
    ) -> List[Annotation]:
        """
        Annotation automatique basique (pour bootstrapping dataset)

        Args:
            frame_index: Index de la frame
            detection_method: Méthode de détection ('color', 'template')

        Returns:
            Liste d'annotations générées
        """

        # Load frame
        with h5py.File(self.session_path, 'r') as f:
            frame = f['frames'][frame_index]

        height, width = frame.shape[:2]
        annotations = []

        if detection_method == "color":
            # Détection basique par couleur (monsters = rouge, NPCs = jaune, etc.)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Monsters (rouge)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv, lower_red, upper_red)

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filtre petits contours
                    x, y, w, h = cv2.boundingRect(contour)

                    # Normalize
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    w_norm = w / width
                    h_norm = h / height

                    ann = Annotation(
                        frame_index=frame_index,
                        class_id=0,  # monster
                        class_name='monster',
                        bbox=(x_center, y_center, w_norm, h_norm),
                        confidence=0.5  # Low confidence (auto-generated)
                    )
                    annotations.append(ann)

        return annotations

    def load_annotations(self, annotations_file: str):
        """Charge annotations depuis fichier JSON"""

        with open(annotations_file, 'r') as f:
            data = json.load(f)

        self.annotations = [
            Annotation(**ann) for ann in data['annotations']
        ]

        logger.info(f"✅ {len(self.annotations)} annotations chargées")

    def save_annotations(self, output_file: str):
        """Sauvegarde annotations vers fichier JSON"""

        data = {
            'session_id': self.session_path.stem,
            'num_frames': self.num_frames,
            'num_annotations': len(self.annotations),
            'annotations': [asdict(ann) for ann in self.annotations]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"✅ Annotations sauvegardées: {output_file}")

    def get_stats(self) -> Dict:
        """Statistiques des annotations"""

        class_counts = {}
        for ann in self.annotations:
            class_counts[ann.class_name] = class_counts.get(ann.class_name, 0) + 1

        return {
            'num_annotations': len(self.annotations),
            'num_frames': self.num_frames,
            'num_annotated_frames': len(set(a.frame_index for a in self.annotations)),
            'class_counts': class_counts,
            'avg_annotations_per_frame': len(self.annotations) / max(1, len(set(a.frame_index for a in self.annotations)))
        }


def create_dataset_from_sessions(
    sessions_dir: str,
    output_dir: str,
    stride: int = 30,  # 1 frame toutes les 30 = ~2 FPS @ 60 FPS source
    max_frames_per_session: int = 200
):
    """
    Helper: Crée dataset YOLO depuis multiple sessions

    Args:
        sessions_dir: Répertoire contenant sessions HDF5
        output_dir: Répertoire de sortie dataset YOLO
        stride: Stride pour extraction frames
        max_frames_per_session: Max frames par session
    """

    sessions_path = Path(sessions_dir)
    session_files = list(sessions_path.glob("*.h5"))

    logger.info(f"📦 Création dataset depuis {len(session_files)} sessions")
    logger.info(f"   Output: {output_dir}")

    # Create output structure
    output_path = Path(output_dir)
    frames_dir = output_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_exported = []

    for session_file in session_files:
        logger.info(f"Processing: {session_file.name}")

        tool = AnnotationTool(str(session_file))

        # Export frames
        exported = tool.export_frames(
            output_dir=str(frames_dir),
            stride=stride,
            max_frames=max_frames_per_session
        )

        all_exported.extend(exported)

    logger.info(f"✅ Dataset créé: {len(all_exported)} frames totales")

    # Create YOLO structure
    tool = AnnotationTool(str(session_files[0]))  # Use first session for structure
    tool.export_yolo_format(output_dir=str(output_path))

    return output_path


# Test du module
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("🏷️  Test Annotation Tool")
    print()

    # Check if test session exists
    test_session = Path("./datasets/test_sessions")

    if not test_session.exists():
        print("⚠️  Aucune session de test trouvée")
        print("   Exécuter d'abord: python session_recorder.py")
        exit(1)

    session_files = list(test_session.glob("*.h5"))

    if len(session_files) == 0:
        print("⚠️  Aucun fichier HDF5 trouvé")
        exit(1)

    print(f"Trouvé {len(session_files)} session(s)")
    print()

    # Load first session
    tool = AnnotationTool(str(session_files[0]))

    print(f"Session: {session_files[0].name}")
    print(f"Frames: {tool.num_frames}")
    print()

    # Export quelques frames
    if tool.num_frames > 0:
        print("📤 Export frames...")
        exported = tool.export_frames(
            output_dir="./datasets/test_frames",
            stride=5,
            max_frames=10
        )

        print(f"✅ {len(exported)} frames exportées")
        print()

    # Export YOLO structure
    print("📦 Création structure YOLO...")
    tool.export_yolo_format(output_dir="./datasets/test_yolo")
    print()

    # Stats
    print("📊 Statistiques:")
    stats = tool.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print()
    print("✅ Test terminé!")
