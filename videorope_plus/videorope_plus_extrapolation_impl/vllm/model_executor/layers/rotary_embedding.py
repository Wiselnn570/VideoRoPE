# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Positional Embeddings."""
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.model_executor.custom_op import CustomOp


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


@CustomOp.register("rotary_embedding")
class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float], device='cuda') -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float).to(device) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from vllm import _custom_ops as ops

        self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                   dtype=query.dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from vllm._ipex_ops import ipex_ops as ops

        self.cos_sin_cache = self.cos_sin_cache.to(positions.device,
                                                   dtype=query.dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

    def forward_hpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from habana_frameworks.torch.hpex.kernels import (
            RotaryPosEmbeddingMode, apply_rotary_pos_emb)
        if offsets is not None:
            offsets = offsets.view(positions.shape[0], -1)
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions).view(
            num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        # HPU RoPE kernel requires hidden dimension for cos and sin to be equal
        # to query hidden dimension, so the original tensors need to be
        # expanded
        # GPT-NeoX kernel requires position_ids = None, offset, mode = BLOCKWISE
        # and expansion of cos/sin tensors via concatenation
        # GPT-J kernel requires position_ids = None, offset = 0, mode = PAIRWISE
        # and expansion of cos/sin tensors via repeat_interleave
        rope_mode: RotaryPosEmbeddingMode
        if self.is_neox_style:
            rope_mode = RotaryPosEmbeddingMode.BLOCKWISE
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            rope_mode = RotaryPosEmbeddingMode.PAIRWISE
            sin = torch.repeat_interleave(sin,
                                          2,
                                          dim=-1,
                                          output_size=cos_sin.shape[-1])
            cos = torch.repeat_interleave(cos,
                                          2,
                                          dim=-1,
                                          output_size=cos_sin.shape[-1])

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0,
                                         rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling.

    It supports multiple scaling factors. Since multiple LoRA adapters may have
    different scaling factors, we need multiple cos/sin caches. In this way,
    instead of running rotary embedding kernel per lora, we can run multiple
    lora in a batched way.

    In addition to that, we also keep the cos/sin cache for the scaling factor
    of 1 (default) at all times.

    Exemplary for two scaling factors x=1, y and z with embeddings
    [[x11, x12, ... x1m], ..., [xn1, xn2, ..., xnm]] and
    [[y11, y12, ... y1o], ..., [yn1, yn2, ..., yno]], and
    [[z11, z12, ... z1p], ..., [zn1, zn2, ..., znp]],

    we construct the cos/sin cache as follows:
    [[x11, x12, ... x1m, y11, y12, ... y1o, z11, z12, ... z1p],
        ...
     [xn1, xn2, ... xnm, yn1, yn2, ... yno, zn1, zn2, ... znp]]

    We then use offsets to index into the cos/sin cache for
    the respective scaling factors.

    The offset to cache can be accessed via `scaling_factor_to_offset` API.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: Union[List[float], float],
        dtype: torch.dtype,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors  # noqa
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)
        # Lazy initialized.
        self._scaling_factor_to_offset: Dict[float, int]

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: List[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        offsets: List[int] = []
        for scaling_factor in self.scaling_factors:
            # NOTE(woosuk): self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        return torch.cat(cache_list, dim=0)

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self._scaling_factor_to_offset


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # NOTE(woosuk): self.max_position_embeddings is the original
        # maximum length before applying the rope scaling.
        # Thus, the maximum length after applying the rope scaling is
        # self.max_position_embeddings * self.scaling_factor.
        max_len = self.max_position_embeddings * self.scaling_factor
        base = self.base * (
            (self.scaling_factor * max_len / self.max_position_embeddings) -
            (self.scaling_factor - 1))**(self.rotary_dim /
                                         (self.rotary_dim - 2))
        inv_freq = self._compute_inv_freq(base)
        t = torch.arange(max_len, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(num_rotations: int,
                              dim: int,
                              base: float = 10000,
                              max_position_embeddings: int = 2048) -> float:
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base,
                                  max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(low: float, high: float, dim: int,
                           dtype: torch.dtype) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base**(
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) /
            self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
    """Phi3 family of models scaled rotary embedding.

    Based on the original RotaryEmbedding implementation.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        original_max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        short_factor: List[float],
        long_factor: List[float],
        short_mscale: Optional[float] = None,
        long_mscale: Optional[float] = None,
    ):
        super().__init__()

        if is_neox_style is False:
            raise ValueError(
                "`Phi3LongRoPEScaledRotaryEmbedding` only supports neox_style."
            )

        self.rotary_dim = rotary_dim
        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.short_factor = short_factor
        self.long_factor = long_factor

        scale = self.max_position_embeddings / \
                self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(
                1 + math.log(scale) /
                math.log(self.original_max_position_embeddings))
        if short_mscale is None:
            short_mscale = scaling_factor
        if long_mscale is None:
            long_mscale = scaling_factor

        self.short_mscale = short_mscale
        self.long_mscale = long_mscale

        short_cache = self._compute_cos_sin_cache(
            original_max_position_embeddings, short_factor, short_mscale)
        short_cache = short_cache.to(dtype)

        long_cache = self._compute_cos_sin_cache(max_position_embeddings,
                                                 long_factor, long_mscale)
        long_cache = long_cache.to(dtype)

        long_short_cache = torch.cat([short_cache, long_cache], dim=0)
        self.register_buffer("long_short_cos_sin_cache",
                             long_short_cache,
                             persistent=False)

    def _compute_inv_freq(self, rescale_factors: List[float]) -> torch.Tensor:
        rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32)
        inv_freq = 1.0 / (rescale_factors * (self.base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim)))
        return inv_freq

    def _compute_cos_sin_cache(
        self,
        max_position_embeddings: int,
        rescale_factors: List[float],
        mscale: float,
    ) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(rescale_factors)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        k = self.original_max_position_embeddings
        long_prompt_offset = (torch.any(positions > k).float() *
                              torch.full_like(positions, k)).long()
        idx = (torch.add(positions, long_prompt_offset)
               if long_prompt_offset is not None else positions)
        idx = torch.add(idx, offsets) if offsets is not None else idx
        cos_sin = torch.index_select(self.long_short_cos_sin_cache, 0, idx)

        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2).unsqueeze(-2)
        sin = sin.repeat(1, 2).unsqueeze(-2)

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = query_rot * cos + _rotate_neox(query_rot) * sin
        query = torch.cat((query_rot, query_pass), dim=-1)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = key_rot * cos + _rotate_neox(key_rot) * sin
        key = torch.cat((key_rot, key_pass), dim=-1)

        return query.flatten(-2), key.flatten(-2)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale)) /
            yarn_get_mscale(self.scaling_factor, float(mscale_all_dim)) *
            attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float, device="cuda") /
                                self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                         device="cuda",
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
            positions.device)
        cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
                                     if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key


class Llama3RotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor
                      ) / (self.high_freq_factor - self.low_freq_factor)
        else:
            smooth = 0
        new_freqs = torch.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor +
                smooth * inv_freqs,
            ),
        )
        return new_freqs

## TODO
class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""
    scaling_method = None
    factor = 4.0
    def post_init(
        self
    ):  
        self.scaling_method = MRotaryEmbedding.scaling_method
        print(MRotaryEmbedding.factor)
        head_size = self.head_size
        rotary_dim = self.rotary_dim

        self.max_position_embeddings = 32768
        self.original_max_seq_len = 32768
        max_position_embeddings = self.max_position_embeddings
        self.max_seq_len_cached = self.max_position_embeddings

        base = self.base
        is_neox_style = self.is_neox_style
        dtype = self.dtype
        mrope_section = self.mrope_section
        # print(self.scaling_method)
        if self.scaling_method == 'ntk':
            ## ntk
            base = base * MRotaryEmbedding.factor ** (rotary_dim / (rotary_dim - 2))
            
            inv_freq = self._compute_inv_freq(base)
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        elif self.scaling_method == 'yarn':
            ## yarn init
            extrapolation_factor_yarn = 1.0
            max_position_embeddings_yarn = self.max_position_embeddings
            beta_fast = 32
            beta_slow = 1

            attn_factor_yarn = 1.0
            self.mscale = float(
            _yarn_get_mscale(extrapolation_factor_yarn) * attn_factor_yarn)

            pos_freqs_yarn = base**(
                torch.arange(0, rotary_dim, 2, dtype=torch.float) /
                rotary_dim)
            inv_freq_extrapolation = 1.0 / pos_freqs_yarn
            inv_freq_interpolation = 1.0 / (MRotaryEmbedding.factor * pos_freqs_yarn)
            low, high = _yarn_find_correction_range(beta_fast, beta_slow,
                                                rotary_dim, base,
                                                max_position_embeddings_yarn)
            inv_freq_mask = (1 - _yarn_linear_ramp_mask(
                low, high, rotary_dim // 2,
                dtype=torch.float)) * extrapolation_factor_yarn
            inv_freq = inv_freq_interpolation * (
                1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        elif self.scaling_method == 'dynamic' or self.scaling_method == 'videoropepp_dynamic':
            ## dynamic_ntk init
            partial_rotary_factor_dynamic = 1.0
            dim = int(head_size * partial_rotary_factor_dynamic)
            seq_len = max_position_embeddings
            base = base * ((MRotaryEmbedding.factor * seq_len / max_position_embeddings) - (MRotaryEmbedding.factor - 1)) ** (dim / (dim - 2))
            inv_freq = self._compute_inv_freq(base)
            self.register_buffer("inv_freq", inv_freq, persistent=False)


        elif self.scaling_method == 'mrope++':
            print('mrope++')
            # 1. base scaling
            inv_freq_base_mropepp = 1.0 / (base**(torch.arange(
                0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
            
            # 2. Linear scaling
            inv_freq_linear_mropepp = 1.0 / (
                MRotaryEmbedding.factor * (base**(torch.arange(
                0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
            )

            # 3. NTK scaling
            max_target_position_mropepp = self.max_position_embeddings * MRotaryEmbedding.factor
            max_position_embeddings_mropepp = self.max_position_embeddings
            ntk_base = base * (
                (max_target_position_mropepp - 1) / (max_position_embeddings_mropepp - 1)
            ) ** (rotary_dim / (rotary_dim - 2))
            inv_freq_ntk_mropepp = 1.0 / (
                ntk_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
            )
            total_dims = sum(mrope_section)
            assert (
                rotary_dim // 2 == total_dims
            ), f"Dimension mismatch: dim//2({rotary_dim//2}) != sum(mrope_section)({total_dims})"

            # Combine inv_freq for three sections
            inv_freq_list = []
            current_pos = 0

            # Temporal section: use base version (pure extrapolation)
            inv_freq_list.append(inv_freq_base_mropepp[: mrope_section[0]])
            current_pos += mrope_section[0]

            # Height section: mix of linear and ntk
            ntk_factor = 0.5
            height_section = (
                inv_freq_linear_mropepp[current_pos : current_pos + mrope_section[1]]
                * (1 - ntk_factor)
                + inv_freq_ntk_mropepp[current_pos : current_pos + mrope_section[1]] * ntk_factor
            )
            inv_freq_list.append(height_section)
            current_pos += mrope_section[1]

            # Width section: use ntk version
            inv_freq_list.append(inv_freq_ntk_mropepp[current_pos : current_pos + mrope_section[2]])

            # Merge all sections
            
            inv_freq = torch.cat(inv_freq_list)
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        elif self.scaling_method == 'videorope++_v1':
            print("videorope++_v1")
            # 1. base scaling
            inv_freq_base_mropepp = 1.0 / (base**(torch.arange(
                0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
            
            # 2. Linear scaling
            inv_freq_linear_mropepp = 1.0 / (
                MRotaryEmbedding.factor * (base**(torch.arange(
                0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
            )

            # 3. NTK scaling
            max_target_position_mropepp = self.max_position_embeddings * MRotaryEmbedding.factor
            max_position_embeddings_mropepp = self.max_position_embeddings
            ntk_base = base * (
                (max_target_position_mropepp - 1) / (max_position_embeddings_mropepp - 1)
            ) ** (rotary_dim / (rotary_dim - 2))
            inv_freq_ntk_mropepp = 1.0 / (
                ntk_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
            )
            total_dims = sum(mrope_section)
            assert (
                rotary_dim // 2 == total_dims
            ), f"Dimension mismatch: dim//2({rotary_dim//2}) != sum(mrope_section)({total_dims})"

            # Combine inv_freq for three sections
            inv_freq_list = []
            current_pos = 0

            # height, width section: use base version (pure extrapolation)
            inv_freq_list.append(inv_freq_base_mropepp[: mrope_section[1] + mrope_section[2]])
            current_pos += mrope_section[1] + mrope_section[2]

            # Height section: mix of linear and ntk
            temporal_section = inv_freq_ntk_mropepp[current_pos : current_pos + mrope_section[0]]
                # inv_freq_linear_mropepp[current_pos : current_pos + mrope_section[1]]
                # * (1 - ntk_factor)
                
            
            inv_freq_list.append(temporal_section)

            # Width section: use ntk version
            # inv_freq_list.append(inv_freq_ntk_mropepp[current_pos : current_pos + mrope_section[2]])

            # Merge all sections
            
            inv_freq = torch.cat(inv_freq_list)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            
        elif self.scaling_method == 'default':
            inv_freq = self._compute_inv_freq(base)
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        else:
            raise ValueError(f"not impl {self.scaling_method}")
        self.original_inv_freq = self.inv_freq

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
    ) -> None:
        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        super().__init__(head_size, rotary_dim, self.cache_max_position_num,
                         base, is_neox_style, dtype)

        self.mrope_section = mrope_section
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

        inv_freq = self._compute_inv_freq(base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.is_post_init = False

    def _dynamic_frequency_update(
        self,
        positions: torch.Tensor,
    ):
        seq_len = torch.max(positions) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            # inv_freq, self.attention_scaling = self.rope_init_fn(
            #     self.config, device, seq_len=seq_len, **self.rope_kwargs
            # )
            base = self.base * ((MRotaryEmbedding.factor * seq_len / self.max_position_embeddings) - (MRotaryEmbedding.factor - 1)) ** (self.head_size / (self.head_size - 2))
            inv_freq = self._compute_inv_freq(base, positions.device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    def _videoropepp_dynamic_frequency_update(
        self,
        positions: torch.Tensor,
    ):
        seq_len = torch.max(positions) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            dim_extra = 2 * math.ceil((self.head_size / 2) * math.log(self.max_position_embeddings / (2 * math.pi), self.base))
            dynamic_factor = (seq_len / (2 * math.pi)) ** (self.head_size / dim_extra) / self.base            
            
            # Compute the inverse frequencies
            base = self.base * ((dynamic_factor * seq_len / self.max_position_embeddings) - (dynamic_factor - 1)) ** (self.head_size / (self.head_size - 2))
            inv_freq = self._compute_inv_freq(base, positions.device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    def m_modify_forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2
        """
        embeds = (positions.float().unsqueeze(dim=-1) @ self.inv_freq.unsqueeze(dim=0).to(positions.device))
        cos = embeds.cos()
        sin = embeds.sin()
        """
        if self.is_post_init == False and MRotaryEmbedding.scaling_method is not None:
            self.post_init()
            self.is_post_init = True
        if MRotaryEmbedding.scaling_method is not None and MRotaryEmbedding.scaling_method == 'dynamic':
            self._dynamic_frequency_update(positions)
        if MRotaryEmbedding.scaling_method is not None and MRotaryEmbedding.scaling_method == 'videoropepp_dynamic':
            self._videoropepp_dynamic_frequency_update(positions)
        print(MRotaryEmbedding.scaling_method)
        # print('m_modify_rope')
        num_tokens = positions.shape[-1]
        embeds = (positions.float().unsqueeze(dim=-1) @ self.inv_freq.unsqueeze(dim=0).to(positions.device))
        cos = embeds.cos()
        sin = embeds.sin()
        # import pdb; pdb.set_trace()
        if positions.ndim == 2:
            assert self.mrope_section

        mrope_section = [self.mrope_section[0], self.mrope_section[1] + self.mrope_section[2]]
        mrope_section = mrope_section[::-1]
        index = 0
        result_cos = []
        result_sin = []
        for i, section in enumerate(mrope_section):
            if i % 2 == 0:
                for j in range(section):
                    # import pdb; pdb.set_trace()
                    row = 1 if j % 2 == 0 else 2
                    
                    result_cos.append(cos[row, ..., index: index + 1])
                    result_sin.append(sin[row, ..., index: index + 1])
                    index += 1
            else:
                result_cos.append(cos[0, ..., index:index + section])
                result_sin.append(sin[0, ..., index:index + section])
                index += section
        cos, sin = torch.cat(result_cos, dim=-1), torch.cat(result_sin, dim=-1)
        # import pdb; pdb.set_trace()

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def ttxy_modify_forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2
        """
        embeds = (positions.float().unsqueeze(dim=-1) @ self.inv_freq.unsqueeze(dim=0).to(positions.device))
        cos = embeds.cos()
        sin = embeds.sin()
        """
        # print('m_modify_rope')
        if self.is_post_init == False and MRotaryEmbedding.scaling_method is not None:
            self.post_init()
            self.is_post_init = True
        if MRotaryEmbedding.scaling_method is not None and MRotaryEmbedding.scaling_method == 'dynamic':
            self._dynamic_frequency_update(positions)
        if MRotaryEmbedding.scaling_method is not None and MRotaryEmbedding.scaling_method == 'videoropepp_dynamic':
            self._videoropepp_dynamic_frequency_update(positions)
        print(MRotaryEmbedding.scaling_method)
        num_tokens = positions.shape[-1]
        embeds = (positions.float().unsqueeze(dim=-1) @ self.inv_freq.unsqueeze(dim=0).to(positions.device))
        cos = embeds.cos()
        sin = embeds.sin()
        # import pdb; pdb.set_trace()
        if positions.ndim == 2:
            assert self.mrope_section

        mrope_section = [self.mrope_section[0], self.mrope_section[1] + self.mrope_section[2]]
        mrope_section = mrope_section[::-1]
        index = 0
        result_cos = []
        result_sin = []
        for _ in range(16):
            result_cos.append(cos[0, ..., index: index + 1])
            result_sin.append(sin[0, ..., index: index + 1])
            index += 1
            result_cos.append(cos[0, ..., index: index + 1])
            result_sin.append(sin[0, ..., index: index + 1])
            index += 1
            result_cos.append(cos[1, ..., index: index + 1])
            result_sin.append(sin[1, ..., index: index + 1])
            index += 1
            result_cos.append(cos[2, ..., index: index + 1])
            result_sin.append(sin[2, ..., index: index + 1])
            index += 1
            result_cos.append(cos[1, ..., index: index + 1])
            result_sin.append(sin[1, ..., index: index + 1])
            index += 1
            result_cos.append(cos[2, ..., index: index + 1])
            result_sin.append(sin[2, ..., index: index + 1])
            index += 1
            result_cos.append(cos[1, ..., index: index + 1])
            result_sin.append(sin[1, ..., index: index + 1])
            index += 1
            result_cos.append(cos[2, ..., index: index + 1])
            result_sin.append(sin[2, ..., index: index + 1])
            index += 1
        cos, sin = torch.cat(result_cos, dim=-1), torch.cat(result_sin, dim=-1)
        # import pdb; pdb.set_trace()

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key
    
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2

        # print('m_rope_origin')

        # num_tokens = positions.shape[-1]
        # cos_sin = self.cos_sin_cache[positions]
        # cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_post_init == False and MRotaryEmbedding.scaling_method is not None:
            self.post_init()
            self.is_post_init = True
        if MRotaryEmbedding.scaling_method is not None and MRotaryEmbedding.scaling_method == 'dynamic':
            self._dynamic_frequency_update(positions)
        if MRotaryEmbedding.scaling_method is not None and MRotaryEmbedding.scaling_method == 'videoropepp_dynamic':
            self._videoropepp_dynamic_frequency_update(positions)
        num_tokens = positions.shape[-1]
        embeds = (positions.float().unsqueeze(dim=-1) @ self.inv_freq.unsqueeze(dim=0).to(positions.device))
        cos = embeds.cos()
        sin = embeds.sin()

        if positions.ndim == 2:
            assert self.mrope_section

            cos = torch.cat([
                m[i]
                for i, m in enumerate(cos.split(self.mrope_section, dim=-1))
            ],
                            dim=-1)
            sin = torch.cat([
                m[i]
                for i, m in enumerate(sin.split(self.mrope_section, dim=-1))
            ],
                            dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    # def forward(
    #     self,
    #     positions: torch.Tensor,
    #     query: torch.Tensor,
    #     key: torch.Tensor,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """PyTorch-native implementation equivalent to forward().

    #     Args:
    #         positions:
    #             [num_tokens,] (text only) or
    #             [3, num_tokens] (T/H/W positions with multimodal inputs)
    #         query: [num_tokens, num_heads * head_size]
    #         key: [num_tokens, num_kv_heads * head_size]
    #     """
    #     assert positions.ndim == 1 or positions.ndim == 2

    #     num_tokens = positions.shape[-1]
    #     cos_sin = self.cos_sin_cache[positions]
    #     cos, sin = cos_sin.chunk(2, dim=-1)
    #     if positions.ndim == 2:
    #         assert self.mrope_section

    #         cos = torch.cat([
    #             m[i]
    #             for i, m in enumerate(cos.split(self.mrope_section, dim=-1))
    #         ],
    #                         dim=-1)
    #         sin = torch.cat([
    #             m[i]
    #             for i, m in enumerate(sin.split(self.mrope_section, dim=-1))
    #         ],
    #                         dim=-1)

    #     query_shape = query.shape
    #     query = query.view(num_tokens, -1, self.head_size)
    #     query_rot = query[..., :self.rotary_dim]
    #     query_pass = query[..., self.rotary_dim:]
    #     query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
    #     query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    #     key_shape = key.shape
    #     key = key.view(num_tokens, -1, self.head_size)
    #     key_rot = key[..., :self.rotary_dim]
    #     key_pass = key[..., self.rotary_dim:]
    #     key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
    #     key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    #     return query, key

    @staticmethod
    def get_input_positions(
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[List[List[int]], int]:
        """Get mrope input positions and delta value."""

        llm_positions, mrope_position_delta = \
            MRotaryEmbedding.get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=None,
                context_len=context_len,
                seq_len=seq_len,
            )

        return llm_positions.tolist(), mrope_position_delta
    
    @staticmethod
    def get_input_time_positions(
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[List[List[int]], int]:
        """Get mrope input positions and delta value."""
        llm_positions, mrope_position_delta = \
            MRotaryEmbedding.get_input_time_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=None,
                context_len=context_len,
                seq_len=seq_len,
            )
        return llm_positions.tolist(), mrope_position_delta
    
    @staticmethod
    def get_input_vanilla_positions(
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[List[List[int]], int]:
        """Get mrope input positions and delta value."""
        llm_positions, mrope_position_delta = \
            MRotaryEmbedding.get_input_vanilla_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=None,
                context_len=context_len,
                seq_len=seq_len,
            )
        return llm_positions.tolist(), mrope_position_delta
    
    @staticmethod
    def get_input_t_scale_modify_positions(
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        scale_factor=None,
    ) -> Tuple[List[List[int]], int]:
        """Get mrope input positions and delta value."""
        llm_positions, mrope_position_delta = \
            MRotaryEmbedding.get_input_t_scale_modify_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=None,
                context_len=context_len,
                seq_len=seq_len,
                scale_factor=scale_factor
            )
        return llm_positions.tolist(), mrope_position_delta

    @staticmethod
    def get_input_positions_tensor(
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value."""

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config,
                                    "tokens_per_second", 1.0)

        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.tolist()

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_second_per_grid_t = 1.0
                if second_per_grid_ts is not None:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = \
                t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = (torch.arange(llm_grid_t).view(-1, 1).expand(
                -1, llm_grid_h * llm_grid_w) * video_second_per_grid_t *
                       tokens_per_second).long().flatten()

            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta
    
    @staticmethod
    def get_input_time_positions_tensor(
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value."""

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config,
                                    "tokens_per_second", 1.0)

        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.tolist()

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_second_per_grid_t = 1.0
                if second_per_grid_ts is not None:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = \
                t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, t_index, t_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta
    
    @staticmethod
    def get_input_vanilla_positions_tensor(
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value."""

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config,
                                    "tokens_per_second", 1.0)

        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.tolist()

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_second_per_grid_t = 1.0
                if second_per_grid_ts is not None:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = \
                t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_pos_ids_list.append(torch.arange(llm_grid_t * llm_grid_h * llm_grid_w).view(1, -1).expand(3, -1) + st_idx + text_len)
            
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta
    
    @staticmethod
    def get_input_t_scale_modify_positions_tensor(
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        scale_factor = None,
    ) -> Tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value."""

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config,
                                    "tokens_per_second", 1.0)

        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.tolist()

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_second_per_grid_t = 1.0
                if second_per_grid_ts is not None:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = \
                t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                -1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                llm_grid_t, -1, llm_grid_w).flatten() - (llm_grid_h-1) // 2
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                llm_grid_t, llm_grid_h, -1).flatten() - (llm_grid_w-1) // 2
            t_index = t_index * scale_factor
            t_index = t_index + text_len + st_idx
            h_index = h_index + t_index
            w_index = w_index + t_index
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]))
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    # @staticmethod
    # def get_next_input_positions(
    #     mrope_position_delta: int,
    #     context_len: int,
    #     seq_len: int,
    # ) -> List[List[int]]:
    #     return [
    #         list(
    #             range(context_len + mrope_position_delta,
    #                   seq_len + mrope_position_delta)) for _ in range(3)
    #     ]

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: float,
        context_len: float,
        seq_len: float,
        which_rope: str = None,
    ) -> List[List[float]]:
        def float_range(start, stop, step=1.0):
            while start < stop:
                yield start
                start += step
        if which_rope == 'time_rope':
            # 针对 time_rope 的逻辑，保留浮点数
            return [
                [2.0 * context_len + mrope_position_delta] for _ in range(3)
            ]
        else:
            # 非 time_rope 部分使用浮点数范围
            return [
                list(
                    float_range(
                        context_len + mrope_position_delta,
                        seq_len + mrope_position_delta,
                        step=1.0  # 步长为 1
                    )
                ) for _ in range(3)
            ]

    @staticmethod
    def get_next_input_positions_tensor(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> torch.Tensor:
        return torch.arange(
            mrope_position_delta + context_len,
            mrope_position_delta + seq_len,
        ).expand(3, -1)


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (head_size, rotary_dim, max_position, base, is_neox_style,
           rope_scaling_args, dtype)
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                                     is_neox_style, dtype)
    else:
        scaling_type = rope_scaling["rope_type"]
        # print(scaling_type)

        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            rotary_emb = Llama3RotaryEmbedding(head_size, rotary_dim,
                                               max_position, base,
                                               is_neox_style, dtype,
                                               scaling_factor, low_freq_factor,
                                               high_freq_factor,
                                               original_max_position)
        elif scaling_type == "default":
            if "mrope_section" in rope_scaling:
                rotary_emb = MRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                )
            else:
                rotary_emb = RotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                )
        elif scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = LinearScalingRotaryEmbedding(head_size, rotary_dim,
                                                      max_position, base,
                                                      is_neox_style,
                                                      scaling_factor, dtype)
        elif scaling_type == "dynamic":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = DynamicNTKScalingRotaryEmbedding(
                head_size, rotary_dim, max_position, base, is_neox_style,
                scaling_factor, dtype)
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow")
            }
            rotary_emb = YaRNScalingRotaryEmbedding(head_size, rotary_dim,
                                                    original_max_position,
                                                    base, is_neox_style,
                                                    scaling_factor, dtype,
                                                    **extra_kwargs)
        elif scaling_type == "deepseek_yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            # assert max_position == original_max_position * scaling_factor
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow", "mscale", "mscale_all_dim")
            }
            rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size, rotary_dim, original_max_position, base,
                is_neox_style, scaling_factor, dtype, **extra_kwargs)
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("short_mscale", "long_mscale")
            }
            rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                head_size, rotary_dim, max_position, original_max_position,
                base, is_neox_style, dtype, short_factor, long_factor,
                **extra_kwargs)
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    # import pdb; pdb.set_trace()
    return rotary_emb
