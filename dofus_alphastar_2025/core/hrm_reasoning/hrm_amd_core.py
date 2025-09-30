"""
HRM AMD Core - Version optimisée pour AlphaStar DOFUS
Architecture hiérarchique avec optimisations RDNA3 et intégration AlphaStar

Basé sur: sapientinc/HRM + optimisations AlphaStar
Version: 3.0.0 - AlphaStar Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time

# Configuration imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, get_device, get_dtype

# Imports spécifiques AMD
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

try:
    # ROCm WMMA pour RDNA3
    import rocwmma
    HAS_ROCWMMA = True
except ImportError:
    HAS_ROCWMMA = False

logger = logging.getLogger(__name__)

@dataclass
class HRMOutput:
    """Sortie du système HRM pour AlphaStar"""
    # Raisonnement principal
    reasoning_output: torch.Tensor
    reasoning_steps: int
    confidence: float

    # Métriques système
    system_one_latency: float
    system_two_latency: float
    total_computation_time: float

    # États intermédiaires (pour debugging/analysis)
    system_one_states: Optional[torch.Tensor] = None
    system_two_states: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None

class AMDDeviceManager:
    """Gestionnaire de device optimisé pour AMD GPU - Version AlphaStar"""

    def __init__(self):
        self.config = config.amd
        self.device = self._setup_device()
        self.device_props = self._get_device_properties()
        self._setup_optimizations()

    def _setup_device(self) -> torch.device:
        """Configure le device optimal pour AMD"""
        if self.config.use_directml and HAS_DIRECTML:
            try:
                device = torch_directml.device()
                logger.info(f"DirectML device configuré: {device}")
                return device
            except Exception as e:
                logger.warning(f"Échec DirectML: {e}")

        # Fallback vers CUDA si disponible
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA device utilisé: {device}")
            return device

        # Fallback vers CPU
        device = torch.device("cpu")
        logger.warning("Utilisation du CPU - performance dégradée")
        return device

    def _setup_optimizations(self):
        """Configure les optimisations spécifiques AMD/AlphaStar"""
        # Configuration mémoire
        if self.device.type != "cpu":
            try:
                # Réserver fraction de mémoire configurée
                if hasattr(torch.cuda, 'set_memory_fraction'):
                    torch.cuda.set_memory_fraction(self.config.memory_fraction)

                # Activer cuDNN benchmark pour optimisation
                if hasattr(torch.backends.cudnn, 'benchmark'):
                    torch.backends.cudnn.benchmark = True
            except Exception as e:
                logger.warning(f"Optimisations mémoire échouées: {e}")

        # Configuration pour mixed precision
        if self.config.use_mixed_precision:
            logger.info("Mixed precision activée pour AMD")

    def _get_device_properties(self) -> Dict[str, Any]:
        """Récupère les propriétés du device"""
        props = {
            "name": self.config.device_name,
            "memory_gb": self.config.vram_gb,
            "compute_units": self.config.compute_units,
            "has_directml": HAS_DIRECTML,
            "has_rocwmma": HAS_ROCWMMA,
            "device_type": str(self.device),
            "mixed_precision": self.config.use_mixed_precision
        }

        if self.device.type not in ["cpu"]:
            try:
                # Informations mémoire GPU
                if torch.cuda.is_available():
                    props["memory_allocated_mb"] = torch.cuda.memory_allocated(self.device) / 1024 / 1024
                    props["memory_cached_mb"] = torch.cuda.memory_reserved(self.device) / 1024 / 1024
            except Exception:
                pass

        return props

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimise un modèle pour AMD/AlphaStar"""
        model = model.to(self.device)

        # Conversion mixed precision si activée
        if self.config.use_mixed_precision:
            try:
                model = model.half()
                logger.info("Modèle converti en FP16")
            except Exception as e:
                logger.warning(f"Conversion FP16 échouée: {e}")

        # Optimisation compilation (expérimental)
        if hasattr(torch, 'compile') and config.system.debug_mode:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Modèle compilé avec torch.compile")
            except Exception as e:
                logger.warning(f"Compilation torch.compile échouée: {e}")

        return model

class OptimizedRotaryEmbedding(nn.Module):
    """Rotary Embedding optimisé pour RDNA3 et AlphaStar"""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pré-calcul optimisé pour AMD
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache pour éviter recalculs
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward optimisé pour RDNA3"""
        if seq_len is None:
            seq_len = x.shape[-2]

        # Utilisation du cache si possible
        if seq_len <= self._cached_seq_len and self._cached_cos is not None:
            return self._cached_cos[:seq_len], self._cached_sin[:seq_len]

        # Génération des embeddings
        device = x.device
        dtype = x.dtype

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)

        # Optimisation mémoire avec bfloat16
        if config.amd.use_mixed_precision and dtype in [torch.float32]:
            freqs = freqs.to(torch.bfloat16)

        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)

        # Mise à jour cache
        self._cached_cos = cos
        self._cached_sin = sin
        self._cached_seq_len = seq_len

        return cos, sin

class AMDOptimizedAttention(nn.Module):
    """Attention multi-head optimisée pour AMD et compatible AlphaStar"""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: Optional[int] = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size

        # Projections linéaires avec optimisation AMD
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = OptimizedRotaryEmbedding(self.head_dim)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _apply_rotary_pos_emb(self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Application optimisée des embeddings rotatoires"""
        # Split en parties réelle et imaginaire
        x1, x2 = tensor.chunk(2, dim=-1)

        # Rotation optimisée pour RDNA3
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return rotated

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:

        batch_size, seq_len, _ = hidden_states.shape

        # Projections avec mixed precision automatique
        with torch.cuda.amp.autocast(enabled=config.amd.use_mixed_precision):
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # Reshape pour multi-head
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Application des embeddings rotatoires
        cos, sin = self.rotary_emb(value_states, seq_len)
        query_states = self._apply_rotary_pos_emb(query_states, cos, sin)
        key_states = self._apply_rotary_pos_emb(key_states, cos, sin)

        # Gestion du cache KV (important pour AlphaStar)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Attention optimisée
        attn_output = self._compute_attention_optimized(
            query_states, key_states, value_states, attention_mask
        )

        # Reshape et projection finale
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _compute_attention_optimized(self,
                                   query: torch.Tensor,
                                   key: torch.Tensor,
                                   value: torch.Tensor,
                                   attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calcul d'attention optimisé pour AMD avec fallbacks"""

        # Essayer Scaled Dot Product Attention (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            try:
                return F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,  # AlphaStar n'est pas forcément causal
                    scale=self.scale
                )
            except Exception as e:
                logger.debug(f"SDP Attention failed, fallback: {e}")

        # Implémentation manuelle optimisée
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax avec dtype stable
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query.dtype)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

class AMDOptimizedMLP(nn.Module):
    """MLP optimisé pour RDNA3 avec SwiGLU activation"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU implementation
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward optimisé avec SwiGLU"""
        with torch.cuda.amp.autocast(enabled=config.amd.use_mixed_precision):
            gate = self.gate_proj(x)
            up = self.up_proj(x)

            # SwiGLU activation (SiLU * Linear)
            intermediate = F.silu(gate) * up

            # Projection finale
            output = self.down_proj(intermediate)

        return output

class RMSNorm(nn.Module):
    """RMS Normalization optimisée pour RDNA3"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class HRMReasoningBlock(nn.Module):
    """Bloc de raisonnement HRM adapté pour AlphaStar"""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()

        # Couches principales
        self.self_attn = AMDOptimizedAttention(hidden_size, num_heads)
        self.mlp = AMDOptimizedMLP(hidden_size, intermediate_size)

        # Normalisation
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

        # Gradient checkpointing support
        self.gradient_checkpointing = config.amd.enable_gradient_checkpointing

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:

        residual = hidden_states

        # Pre-norm attention avec gradient checkpointing optionnel
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, present_key_value = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.self_attn),
                hidden_states,
                attention_mask,
                past_key_value,
                use_cache
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )

        # Connexion résiduelle
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.gradient_checkpointing and self.training:
            hidden_states = torch.utils.checkpoint.checkpoint(self.mlp, hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        # Connexion résiduelle
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value

class HRMSystemOne(nn.Module):
    """System 1 - Raisonnement rapide et intuitif pour AlphaStar"""

    def __init__(self):
        super().__init__()

        # Configuration depuis config global
        self.hidden_size = config.hrm.hidden_size
        self.num_heads = config.hrm.num_attention_heads
        self.intermediate_size = self.hidden_size * 4
        self.num_layers = config.hrm.system_one_layers

        # Couches de raisonnement rapide
        self.layers = nn.ModuleList([
            HRMReasoningBlock(
                self.hidden_size,
                self.num_heads,
                self.intermediate_size
            ) for _ in range(self.num_layers)
        ])

        self.norm = RMSNorm(self.hidden_size)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:

        past_key_values = [] if use_cache else None

        # Passage à travers les couches
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                past_key_values.append(layer_outputs[1])

        # Normalisation finale
        hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, past_key_values
        return hidden_states

class HRMSystemTwo(nn.Module):
    """System 2 - Raisonnement délibéré et analytique pour AlphaStar"""

    def __init__(self):
        super().__init__()

        # Configuration depuis config global
        self.hidden_size = config.hrm.hidden_size
        self.num_heads = config.hrm.num_attention_heads
        self.intermediate_size = self.hidden_size * 4
        self.num_layers = config.hrm.system_two_layers

        # Couches de raisonnement délibéré
        self.layers = nn.ModuleList([
            HRMReasoningBlock(
                self.hidden_size,
                self.num_heads,
                self.intermediate_size
            ) for _ in range(self.num_layers)
        ])

        self.norm = RMSNorm(self.hidden_size)

        # Mécanisme de halting adaptatif (Q-learning compatible)
        self.halting_linear = nn.Linear(self.hidden_size, 1)
        self.threshold = config.hrm.halting_threshold

    def forward(self,
                hidden_states: torch.Tensor,
                system_one_output: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                max_steps: Optional[int] = None) -> Tuple[torch.Tensor, int]:

        if max_steps is None:
            max_steps = config.hrm.max_reasoning_steps

        # Intégration avec System 1 (fusion additive)
        hidden_states = hidden_states + system_one_output

        halting_probs = []
        steps_taken = 0

        for step in range(max_steps):
            # Calcul de la probabilité de halting
            halting_prob = torch.sigmoid(self.halting_linear(hidden_states.mean(dim=1)))
            halting_probs.append(halting_prob)

            # Décision de halting (seulement après au moins 1 étape)
            if halting_prob.mean() > self.threshold and step > 0:
                break

            # Passage à travers une couche (cycling)
            layer_idx = step % self.num_layers
            layer_outputs = self.layers[layer_idx](
                hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs[0]
            steps_taken += 1

        # Normalisation finale
        hidden_states = self.norm(hidden_states)

        return hidden_states, steps_taken

class HRMAMDModel(nn.Module):
    """Modèle HRM complet optimisé pour AMD 7800XT et intégration AlphaStar"""

    def __init__(self):
        super().__init__()

        # Gestionnaire de device AMD
        self.device_manager = AMDDeviceManager()

        # Paramètres du modèle depuis config
        self.vocab_size = 32000  # Compatible avec tokenizers standards
        self.hidden_size = config.hrm.hidden_size
        self.max_position_embeddings = config.hrm.context_window

        # Embeddings d'entrée
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Systèmes de raisonnement
        self.system_one = HRMSystemOne()
        self.system_two = HRMSystemTwo()

        # Têtes de sortie pour AlphaStar
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Tête de valeur pour RL
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Initialisation des poids
        self.apply(self._init_weights)

        # Optimisation pour AMD
        self = self.device_manager.optimize_model(self)

        param_count = self.count_parameters()
        logger.info(f"HRM AMD Model initialisé avec {param_count:,} paramètres")

    def _init_weights(self, module):
        """Initialisation des poids optimisée pour AMD/AlphaStar"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Compte le nombre de paramètres"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                max_reasoning_steps: Optional[int] = None,
                return_reasoning_details: bool = False) -> Union[Dict[str, torch.Tensor], HRMOutput]:

        start_time = time.time()

        # Embeddings d'entrée
        hidden_states = self.embed_tokens(input_ids)

        # System 1: Raisonnement rapide
        system_one_start = time.time()
        with torch.cuda.amp.autocast(enabled=config.amd.use_mixed_precision):
            system_one_output = self.system_one(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache
            )

            if use_cache:
                system_one_output, past_key_values = system_one_output

        system_one_time = time.time() - system_one_start

        # System 2: Raisonnement délibéré
        system_two_start = time.time()
        with torch.cuda.amp.autocast(enabled=config.amd.use_mixed_precision):
            final_output, reasoning_steps = self.system_two(
                hidden_states,
                system_one_output,
                attention_mask=attention_mask,
                max_steps=max_reasoning_steps
            )
        system_two_time = time.time() - system_two_start

        total_time = time.time() - start_time

        # Prédictions finales
        logits = self.lm_head(final_output)
        values = self.value_head(final_output)

        # Calcul de confiance (moyenne des probabilités normalisées)
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values.mean().item()

        # Format de sortie pour AlphaStar
        if return_reasoning_details:
            return HRMOutput(
                reasoning_output=final_output,
                reasoning_steps=reasoning_steps,
                confidence=confidence,
                system_one_latency=system_one_time * 1000,  # ms
                system_two_latency=system_two_time * 1000,  # ms
                total_computation_time=total_time * 1000,   # ms
                system_one_states=system_one_output,
                system_two_states=final_output
            )

        # Format de sortie standard
        outputs = {
            "logits": logits,
            "values": values,
            "hidden_states": final_output,
            "reasoning_steps": reasoning_steps,
            "confidence": confidence,
            "system_one_output": system_one_output
        }

        # Calcul de la loss si labels fournis
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            outputs["loss"] = loss

        if use_cache:
            outputs["past_key_values"] = past_key_values

        return outputs

    def get_device_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le device AMD"""
        return self.device_manager.device_props

    def get_memory_usage(self) -> Dict[str, float]:
        """Retourne l'utilisation mémoire du modèle"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())

        total_size = param_size + buffer_size

        return {
            "parameters_mb": param_size / 1024 / 1024,
            "buffers_mb": buffer_size / 1024 / 1024,
            "total_mb": total_size / 1024 / 1024,
            "estimated_inference_mb": total_size * 2.0 / 1024 / 1024,  # Avec activations
            "vram_usage_percent": (total_size / (config.amd.vram_gb * 1024**3)) * 100
        }

    def to_device(self) -> 'HRMAMDModel':
        """Déplace le modèle vers le device AMD optimal"""
        return self.to(self.device_manager.device)

# Factory function pour simplifier l'utilisation
def create_hrm_model() -> HRMAMDModel:
    """Crée un modèle HRM optimisé pour AMD avec la configuration globale"""
    model = HRMAMDModel()
    logger.info("HRM AMD Model créé avec succès")
    return model