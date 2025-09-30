"""
Optimiseur GPU AMD pour TacticalBot
Utilise ROCm et DirectML pour accélérer les calculs sur AMD 7800XT

Fonctionnalités:
- Détection et configuration automatique du GPU AMD
- Optimisation des modèles PyTorch pour ROCm
- Accélération des calculs de vision et d'IA
- Monitoring des performances GPU
"""

import os
import sys
import logging
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

# Imports GPU AMD
try:
    import torch
    import torch_directml
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


@dataclass
class GPUCapabilities:
    """Capacités du GPU détectées"""
    model: str = "Unknown"
    memory_gb: float = 0.0
    compute_units: int = 0
    rocm_supported: bool = False
    directml_supported: bool = False
    cuda_cores: int = 0
    memory_bandwidth: float = 0.0


@dataclass
class OptimizationSettings:
    """Paramètres d'optimisation GPU"""
    enable_rocm: bool = True
    enable_directml: bool = True
    memory_fraction: float = 0.8  # Fraction de mémoire GPU à utiliser
    mixed_precision: bool = True  # FP16 pour accélération
    parallel_processing: bool = True
    vision_acceleration: bool = True
    model_optimization: bool = True


class AMDGPUDetector:
    """Détecteur de GPU AMD"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.capabilities = GPUCapabilities()

    def detect_gpu(self) -> GPUCapabilities:
        """Détecte les capacités du GPU AMD"""
        try:
            # Détection via rocm-smi si disponible
            if self._check_rocm_smi():
                self._detect_with_rocm_smi()
            else:
                # Détection via WMI/PowerShell sur Windows
                self._detect_with_system_info()

            # Vérification du support ROCm
            self.capabilities.rocm_supported = self._check_rocm_support()

            # Vérification du support DirectML
            self.capabilities.directml_supported = self._check_directml_support()

            self.logger.info(f"GPU AMD détecté: {self.capabilities.model} - {self.capabilities.memory_gb}GB")
            return self.capabilities

        except Exception as e:
            self.logger.error(f"Erreur détection GPU: {e}")
            return GPUCapabilities()

    def _check_rocm_smi(self) -> bool:
        """Vérifie si rocm-smi est disponible"""
        try:
            result = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _detect_with_rocm_smi(self):
        """Détection via rocm-smi"""
        try:
            result = subprocess.run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                                  capture_output=True, text=True, timeout=10)

            output = result.stdout
            lines = output.split('\n')

            for line in lines:
                if "GPU" in line and "Radeon" in line:
                    self.capabilities.model = line.strip()
                elif "VRAM Total Memory" in line:
                    # Extraction de la mémoire (ex: "8192 MB")
                    mem_str = line.split(':')[-1].strip()
                    if "MB" in mem_str:
                        self.capabilities.memory_gb = int(mem_str.replace(" MB", "")) / 1024

        except Exception as e:
            self.logger.warning(f"Erreur rocm-smi: {e}")

    def _detect_with_system_info(self):
        """Détection via informations système Windows"""
        try:
            # Utilisation de wmic pour détecter le GPU
            result = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"],
                                  capture_output=True, text=True, timeout=10)

            output = result.stdout
            lines = output.split('\n')

            for line in lines:
                if "Radeon" in line or "AMD" in line:
                    self.capabilities.model = line.strip()
                    break

            # Mémoire par défaut pour 7800XT
            if "7800" in self.capabilities.model:
                self.capabilities.memory_gb = 16.0
                self.capabilities.compute_units = 60
                self.capabilities.cuda_cores = 3840

        except Exception as e:
            self.logger.warning(f"Erreur détection système: {e}")

    def _check_rocm_support(self) -> bool:
        """Vérifie le support ROCm"""
        try:
            if not TORCH_AVAILABLE:
                return False

            # Test de création de tensor sur GPU AMD
            if torch_directml.is_available():
                device = torch_directml.device()
                test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
                return test_tensor.device.type == "privateuseone"
        except Exception:
            pass

        return False

    def _check_directml_support(self) -> bool:
        """Vérifie le support DirectML"""
        try:
            if not TORCH_AVAILABLE:
                return False

            return torch_directml.is_available()
        except Exception:
            return False


class AMDGPUOptimizer:
    """Optimiseur pour GPU AMD"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detector = AMDGPUDetector()
        self.capabilities = GPUCapabilities()
        self.settings = OptimizationSettings()
        self.optimization_active = False

    def initialize(self) -> bool:
        """Initialise l'optimiseur GPU"""
        try:
            # Détection du GPU
            self.capabilities = self.detector.detect_gpu()

            if not self.capabilities.model or self.capabilities.memory_gb == 0:
                self.logger.warning("Aucun GPU AMD détecté - fonctionnement en mode CPU")
                return False

            # Configuration des paramètres d'optimisation
            self._configure_optimization_settings()

            # Test des fonctionnalités
            if not self._test_gpu_functionality():
                self.logger.error("Test GPU échoué - désactivation de l'optimisation")
                return False

            self.optimization_active = True
            self.logger.info(f"Optimiseur GPU AMD activé pour {self.capabilities.model}")
            return True

        except Exception as e:
            self.logger.error(f"Erreur initialisation optimiseur GPU: {e}")
            return False

    def _configure_optimization_settings(self):
        """Configure les paramètres d'optimisation selon le GPU"""
        # Ajustements basés sur la mémoire GPU
        if self.capabilities.memory_gb >= 16:  # 7800XT a 16GB
            self.settings.memory_fraction = 0.9  # Utiliser 90% de la mémoire
            self.settings.mixed_precision = True
            self.settings.parallel_processing = True
        elif self.capabilities.memory_gb >= 8:
            self.settings.memory_fraction = 0.8
            self.settings.mixed_precision = True
        else:
            self.settings.memory_fraction = 0.6
            self.settings.mixed_precision = False

        # Ajustements basés sur les unités de calcul
        if self.capabilities.compute_units >= 60:  # 7800XT a 60 CU
            self.settings.vision_acceleration = True
            self.settings.model_optimization = True

    def _test_gpu_functionality(self) -> bool:
        """Test les fonctionnalités GPU"""
        try:
            if not TORCH_AVAILABLE:
                return False

            # Test DirectML
            if self.settings.enable_directml and torch_directml.is_available():
                device = torch_directml.device()
                test_tensor = torch.randn(100, 100, device=device)
                result = torch.matmul(test_tensor, test_tensor.t())
                return result.shape == torch.Size([100, 100])

            return True

        except Exception as e:
            self.logger.error(f"Test GPU échoué: {e}")
            return False

    def optimize_model(self, model) -> Any:
        """Optimise un modèle PyTorch pour AMD GPU"""
        if not self.optimization_active or not TORCH_AVAILABLE:
            return model

        try:
            # Déplacement vers le device AMD
            if torch_directml.is_available():
                device = torch_directml.device()
                model = model.to(device)

            # Optimisation de la mémoire
            if self.settings.mixed_precision:
                try:
                    from torch.cuda.amp import autocast, GradScaler
                    # Configuration pour mixed precision
                    model = self._apply_mixed_precision_optimizations(model)
                except ImportError:
                    pass

            # Optimisation des batchs
            if self.settings.parallel_processing:
                model = self._optimize_batch_processing(model)

            self.logger.debug("Modèle optimisé pour GPU AMD")
            return model

        except Exception as e:
            self.logger.error(f"Erreur optimisation modèle: {e}")
            return model

    def _apply_mixed_precision_optimizations(self, model):
        """Applique les optimisations mixed precision"""
        # Configuration pour FP16
        model.half()
        return model

    def _optimize_batch_processing(self, model):
        """Optimise le traitement par batch"""
        # Ajustement de la taille des batchs selon la mémoire
        if hasattr(model, 'batch_size'):
            optimal_batch_size = int(32 * self.settings.memory_fraction)
            model.batch_size = max(1, optimal_batch_size)

        return model

    def optimize_vision_processing(self, image_processor) -> Any:
        """Optimise le traitement d'images pour AMD GPU"""
        if not self.optimization_active or not OPENCV_AVAILABLE:
            return image_processor

        try:
            # Configuration OpenCV pour GPU AMD
            if self.settings.vision_acceleration:
                # Utilisation d'OpenCL si disponible
                if cv2.ocl.haveOpenCL():
                    cv2.ocl.setUseOpenCL(True)
                    self.logger.debug("OpenCL activé pour traitement d'images")

            return image_processor

        except Exception as e:
            self.logger.error(f"Erreur optimisation vision: {e}")
            return image_processor

    def get_gpu_utilization(self) -> Dict[str, Any]:
        """Retourne l'utilisation actuelle du GPU"""
        try:
            utilization = {
                "memory_used_gb": 0.0,
                "memory_total_gb": self.capabilities.memory_gb,
                "memory_percentage": 0.0,
                "temperature": 0,
                "power_usage": 0.0,
                "gpu_clock": 0
            }

            # Utilisation de rocm-smi si disponible
            if self._check_rocm_smi():
                try:
                    result = subprocess.run(["rocm-smi", "--showuse", "--showtemp", "--showpower"],
                                          capture_output=True, text=True, timeout=5)

                    output = result.stdout
                    # Parsing basique (à adapter selon la sortie réelle)
                    for line in output.split('\n'):
                        if "GPU use" in line:
                            utilization["memory_percentage"] = float(line.split(':')[-1].strip().replace('%', ''))
                        elif "Temperature" in line:
                            utilization["temperature"] = int(line.split(':')[-1].strip().replace('C', ''))

                except Exception as e:
                    self.logger.warning(f"Erreur récupération utilisation GPU: {e}")

            return utilization

        except Exception as e:
            self.logger.error(f"Erreur get_gpu_utilization: {e}")
            return {"error": str(e)}

    def _check_rocm_smi(self) -> bool:
        """Vérifie la disponibilité de rocm-smi"""
        try:
            subprocess.run(["rocm-smi"], capture_output=True, timeout=2)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def create_optimized_device(self):
        """Crée un device PyTorch optimisé pour AMD"""
        if not TORCH_AVAILABLE:
            return torch.device("cpu")

        try:
            if torch_directml.is_available():
                return torch_directml.device()
            else:
                return torch.device("cpu")
        except Exception:
            return torch.device("cpu")

    def benchmark_gpu_performance(self) -> Dict[str, float]:
        """Benchmark les performances du GPU"""
        if not self.optimization_active:
            return {"error": "GPU optimization not active"}

        try:
            device = self.create_optimized_device()

            # Test de calcul matriciel
            sizes = [1000, 2000, 4000]
            times = {}

            for size in sizes:
                # Création de matrices de test
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)

                # Mesure du temps
                start_time = time.time()
                c = torch.matmul(a, b)
                torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.time()

                times[f"matmul_{size}x{size}"] = end_time - start_time

            return times

        except Exception as e:
            self.logger.error(f"Erreur benchmark GPU: {e}")
            return {"error": str(e)}

    def get_optimization_report(self) -> Dict[str, Any]:
        """Génère un rapport d'optimisation"""
        return {
            "gpu_capabilities": {
                "model": self.capabilities.model,
                "memory_gb": self.capabilities.memory_gb,
                "compute_units": self.capabilities.compute_units,
                "rocm_supported": self.capabilities.rocm_supported,
                "directml_supported": self.capabilities.directml_supported
            },
            "optimization_settings": {
                "enable_rocm": self.settings.enable_rocm,
                "enable_directml": self.settings.enable_directml,
                "memory_fraction": self.settings.memory_fraction,
                "mixed_precision": self.settings.mixed_precision,
                "parallel_processing": self.settings.parallel_processing,
                "vision_acceleration": self.settings.vision_acceleration,
                "model_optimization": self.settings.model_optimization
            },
            "status": {
                "optimization_active": self.optimization_active,
                "torch_available": TORCH_AVAILABLE,
                "opencv_available": OPENCV_AVAILABLE
            },
            "performance": self.get_gpu_utilization(),
            "benchmark": self.benchmark_gpu_performance()
        }