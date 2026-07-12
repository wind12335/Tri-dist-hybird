################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from triton_dist.models.dense import DenseLLM, DenseLLMLayer, _set_cos_sin_cache
from triton_dist.test.utils import LAYER_CONFIGS


_PRESET_META = {
    "LLaMA-7B": {"num_layers": 32, "num_heads": 32, "num_key_value_heads": 32, "rope_theta": 10000.0},
    "LLaMA-3.1-8B": {"num_layers": 32, "num_heads": 32, "num_key_value_heads": 8, "rope_theta": 500000.0},
    "LLaMA-3.1-70B": {"num_layers": 80, "num_heads": 64, "num_key_value_heads": 8, "rope_theta": 500000.0},
    "LLaMA-3.1-405B": {"num_layers": 126, "num_heads": 128, "num_key_value_heads": 16, "rope_theta": 500000.0},
    "Qwen2-72B": {"num_layers": 80, "num_heads": 64, "num_key_value_heads": 8, "rope_theta": 1000000.0},
}


@dataclass
class SyntheticModelSpec:
    name: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_key_value_heads: int
    head_dim: int
    max_length: int
    vocab_size: int
    rope_theta: float
    norm_eps: float
    init_std: float
    tie_word_embeddings: bool
    share_layer_weights: bool
    rank: int
    world_size: int
    dtype: torch.dtype


class _SyntheticLinear:

    def __init__(self, out_features: int, in_features: int, dtype: torch.dtype, init_std: float):
        weight = torch.randn((out_features, in_features), dtype=torch.float32) * init_std
        self.weight = weight.to(dtype=dtype)
        self.bias = None


class _SyntheticRMSNorm:

    def __init__(self, hidden_size: int, dtype: torch.dtype, eps: float):
        self.weight = torch.ones(hidden_size, dtype=dtype)
        self.variance_epsilon = eps


class _SyntheticAttentionModule:

    def __init__(self, spec: SyntheticModelSpec):
        qkv_hidden = spec.num_key_value_heads * spec.head_dim
        self.q_proj = _SyntheticLinear(spec.hidden_size, spec.hidden_size, spec.dtype, spec.init_std)
        self.k_proj = _SyntheticLinear(qkv_hidden, spec.hidden_size, spec.dtype, spec.init_std)
        self.v_proj = _SyntheticLinear(qkv_hidden, spec.hidden_size, spec.dtype, spec.init_std)
        self.o_proj = _SyntheticLinear(spec.hidden_size, spec.hidden_size, spec.dtype, spec.init_std)
        self.head_dim = spec.head_dim
        self.config = SimpleNamespace(num_key_value_heads=spec.num_key_value_heads)


class _SyntheticMLPModule:

    def __init__(self, spec: SyntheticModelSpec):
        self.gate_proj = _SyntheticLinear(spec.intermediate_size, spec.hidden_size, spec.dtype, spec.init_std)
        self.up_proj = _SyntheticLinear(spec.intermediate_size, spec.hidden_size, spec.dtype, spec.init_std)
        self.down_proj = _SyntheticLinear(spec.hidden_size, spec.intermediate_size, spec.dtype, spec.init_std)
        self.act_fn = F.silu


class _SyntheticDenseLayerModule:

    def __init__(self, spec: SyntheticModelSpec):
        self.self_attn = _SyntheticAttentionModule(spec)
        self.mlp = _SyntheticMLPModule(spec)
        self.input_layernorm = _SyntheticRMSNorm(spec.hidden_size, spec.dtype, spec.norm_eps)
        self.post_attention_layernorm = _SyntheticRMSNorm(spec.hidden_size, spec.dtype, spec.norm_eps)


def get_synthetic_preset_names():
    return sorted(_PRESET_META.keys())


def resolve_synthetic_spec(name: str | None,
                           hidden_size: int | None,
                           intermediate_size: int | None,
                           num_layers: int | None,
                           num_heads: int | None,
                           num_key_value_heads: int | None,
                           head_dim: int | None,
                           max_length: int,
                           vocab_size: int,
                           rope_theta: float | None,
                           norm_eps: float,
                           init_std: float,
                           tie_word_embeddings: bool,
                           share_layer_weights: bool,
                           rank: int,
                           world_size: int,
                           dtype: torch.dtype) -> SyntheticModelSpec:
    preset_name = name or "custom"
    if name is not None:
        if name not in _PRESET_META:
            raise ValueError(f"Unsupported synthetic preset: {name}")
        layer_cfg = LAYER_CONFIGS[name]
        preset_meta = _PRESET_META[name]
        hidden_size = layer_cfg["K"] if hidden_size is None else hidden_size
        intermediate_size = layer_cfg["N"] if intermediate_size is None else intermediate_size
        num_layers = preset_meta["num_layers"] if num_layers is None else num_layers
        num_heads = preset_meta["num_heads"] if num_heads is None else num_heads
        num_key_value_heads = preset_meta["num_key_value_heads"] if num_key_value_heads is None else num_key_value_heads
        rope_theta = preset_meta["rope_theta"] if rope_theta is None else rope_theta

    hidden_size = 4096 if hidden_size is None else hidden_size
    intermediate_size = 4 * hidden_size if intermediate_size is None else intermediate_size
    num_layers = 32 if num_layers is None else num_layers

    if head_dim is None and hidden_size is not None and num_heads is not None:
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}.")
        head_dim = hidden_size // num_heads
    elif head_dim is None and hidden_size is not None and num_heads is None:
        head_dim = 128
        if hidden_size % head_dim != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by inferred head_dim {head_dim}.")
        num_heads = hidden_size // head_dim
    elif head_dim is not None and hidden_size is not None and num_heads is None:
        if hidden_size % head_dim != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by head_dim {head_dim}.")
        num_heads = hidden_size // head_dim
    elif head_dim is not None and hidden_size is None and num_heads is not None:
        hidden_size = num_heads * head_dim
    elif head_dim is not None and hidden_size is not None and num_heads is not None:
        if hidden_size != num_heads * head_dim:
            raise ValueError(
                f"hidden_size ({hidden_size}) must equal num_heads ({num_heads}) * head_dim ({head_dim}).")
    else:
        head_dim = 128
        num_heads = hidden_size // head_dim

    num_key_value_heads = num_heads if num_key_value_heads is None else num_key_value_heads
    rope_theta = 10000.0 if rope_theta is None else rope_theta

    spec = SyntheticModelSpec(
        name=preset_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_length=max_length,
        vocab_size=vocab_size,
        rope_theta=rope_theta,
        norm_eps=norm_eps,
        init_std=init_std,
        tie_word_embeddings=tie_word_embeddings,
        share_layer_weights=share_layer_weights,
        rank=rank,
        world_size=world_size,
        dtype=dtype,
    )
    _validate_synthetic_spec(spec)
    return spec


def format_synthetic_spec(spec: SyntheticModelSpec):
    return (
        f"name={spec.name}, hidden={spec.hidden_size}, intermediate={spec.intermediate_size}, "
        f"layers={spec.num_layers}, heads={spec.num_heads}, kv_heads={spec.num_key_value_heads}, "
        f"head_dim={spec.head_dim}, vocab={spec.vocab_size}, share_weights={spec.share_layer_weights}"
    )


def _validate_synthetic_spec(spec: SyntheticModelSpec):
    if spec.hidden_size <= 0 or spec.intermediate_size <= 0 or spec.num_layers <= 0:
        raise ValueError("Synthetic model dimensions must be positive.")
    if spec.num_heads <= 0 or spec.num_key_value_heads <= 0 or spec.head_dim <= 0:
        raise ValueError("Synthetic attention dimensions must be positive.")
    if spec.head_dim % 2 != 0:
        raise ValueError("Synthetic head_dim must be even for RoPE.")
    if spec.hidden_size != spec.num_heads * spec.head_dim:
        raise ValueError("Synthetic hidden_size must equal num_heads * head_dim.")
    if spec.num_heads % spec.num_key_value_heads != 0:
        raise ValueError("Synthetic num_heads must be divisible by num_key_value_heads.")
    if spec.hidden_size % spec.world_size != 0:
        raise ValueError("Synthetic hidden_size must be divisible by world_size for tensor parallel sharding.")
    if spec.intermediate_size % spec.world_size != 0:
        raise ValueError("Synthetic intermediate_size must be divisible by world_size for tensor parallel sharding.")
    if spec.num_key_value_heads % spec.world_size != 0:
        raise ValueError(
            "Synthetic num_key_value_heads must be divisible by world_size for tensor parallel sharding.")
    if spec.vocab_size <= 0:
        raise ValueError("Synthetic vocab_size must be positive.")


def _build_rope_inv_freq(spec: SyntheticModelSpec):
    inv_freq = 1.0 / (spec.rope_theta**(torch.arange(0, spec.head_dim, 2, device="cuda", dtype=torch.float32) /
                                        spec.head_dim))
    return inv_freq


def _share_layer(template: DenseLLMLayer, layer_idx: int, group):
    layer = DenseLLMLayer(layer_idx, group)
    layer.attn = template.attn
    layer.mlp = template.mlp
    layer.input_norm_eps = template.input_norm_eps
    layer.input_norm_w = template.input_norm_w
    layer.post_norm_eps = template.post_norm_eps
    layer.post_norm_w = template.post_norm_w
    return layer


class SyntheticDenseLLM(DenseLLM):

    def __init__(self, spec: SyntheticModelSpec, group) -> None:
        self.synthetic_spec = spec
        self.dtype = spec.dtype
        self.model_name = f"synthetic::{spec.name}"
        self.max_length = spec.max_length
        self.hidden_size = spec.hidden_size
        self.num_heads = spec.num_heads
        self.head_dim = spec.head_dim
        self.num_key_value_heads = spec.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = spec.max_length
        self.rope_theta = spec.rope_theta
        self.rank = spec.rank
        self.world_size = spec.world_size
        self.group = group
        self.vocab_size = spec.vocab_size
        self.config = SimpleNamespace(
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_attention_heads=spec.num_heads,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            max_position_embeddings=spec.max_length,
            rope_theta=spec.rope_theta,
            vocab_size=spec.vocab_size,
        )

        self._init_synthetic_parameters()
        self.set_fwd()
        self.use_ar = False
        self.model_type = 'dense'

    def _init_synthetic_parameters(self):
        spec = self.synthetic_spec
        self.embed_tokens = torch.empty((spec.vocab_size, spec.hidden_size), device="cuda", dtype=spec.dtype)
        self.embed_tokens.normal_(mean=0.0, std=spec.init_std)
        if spec.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = torch.empty((spec.vocab_size, spec.hidden_size), device="cuda", dtype=spec.dtype)
            self.lm_head.normal_(mean=0.0, std=spec.init_std)
        self.norm_weight = torch.ones(spec.hidden_size, device="cuda", dtype=spec.dtype)
        self.norm_variance_epsilon = spec.norm_eps
        self.cos_sin_cache = _set_cos_sin_cache(_build_rope_inv_freq(spec), max_length=spec.max_length)

        first_layer = DenseLLMLayer(0, self.group)
        first_layer.init_parameters(_SyntheticDenseLayerModule(spec), rank=self.rank, world_size=self.world_size)
        self.layers = [first_layer]

        if spec.share_layer_weights:
            for layer_idx in range(1, spec.num_layers):
                self.layers.append(_share_layer(first_layer, layer_idx, self.group))
        else:
            for layer_idx in range(1, spec.num_layers):
                layer = DenseLLMLayer(layer_idx, self.group)
                layer.init_parameters(_SyntheticDenseLayerModule(spec), rank=self.rank, world_size=self.world_size)
                self.layers.append(layer)

        self.num_layers = len(self.layers)
