"""HuggingFace transformers KV cache integration.

Provides TurboQuantCache, a drop-in replacement for DynamicCache that
compresses keys with TurboQuantProd and values with TurboQuantMSE.

Storage layout
--------------
All heads for a layer are stored together in [H, B, S, ...] tensors.
Indices are bit-packed so actual storage matches theoretical bit widths:

  Keys (TurboQuantProd at b bits):
    mse_packed : uint8 [H, B, S, ceil(D*(b-1)/8/vpb)]  — (b-1)-bit MSE indices
    qjl_packed : uint8 [H, B, S, ceil(D/8)]             — 1-bit QJL signs
    gamma      : f32   [H, B, S]                         — residual norms
    norms      : f32   [H, B, S]                         — input norms

  Values (TurboQuantMSE at b bits):
    idx_packed : uint8 [H, B, S, ceil(D*b/8/vpb)]       — b-bit MSE indices
    norms      : f32   [H, B, S]                         — input norms

where vpb = 8 // b (values per byte for b-bit packing).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.configuration_utils import PretrainedConfig

from ..bit_packing import pack_bits, pack_signs, unpack_bits, unpack_signs
from ..quantizer_mse_torch import BatchedTurboQuantMSETorch
from ..quantizer_prod_torch import BatchedTurboQuantProdTorch


# ---------------------------------------------------------------------------
# Internal storage helpers
# ---------------------------------------------------------------------------


def _cat_or_init(
    existing: torch.Tensor | None, new: torch.Tensor, dim: int
) -> torch.Tensor:
    if existing is None:
        return new
    return torch.cat([existing, new], dim=dim)


class _BatchedKeyStore:
    """Compressed key storage for all H attention heads in a layer.

    Stores bit-packed tensors of shape [H, B, S, packed_D] to minimise VRAM.
    Sequence dimension (dim=2) grows by concatenation at each decode step.
    """

    def __init__(self, key_bits: int, head_dim: int) -> None:
        self.key_bits = key_bits
        self.head_dim = head_dim
        self.qjl_packed: torch.Tensor | None = None   # uint8 [H, B, S, ceil(D/8)]
        self.gamma: torch.Tensor | None = None         # f32   [H, B, S]
        self.norms: torch.Tensor | None = None         # f32   [H, B, S]
        self.mse_packed: torch.Tensor | None = None    # uint8 [H, B, S, packed_mse]

    def seq_length(self) -> int:
        return 0 if self.qjl_packed is None else self.qjl_packed.shape[2]

    def append(
        self,
        q_dict: dict,
        norms_in: torch.Tensor,
        B: int,
        S_new: int,
    ) -> None:
        H = q_dict["gamma"].shape[0]
        D = self.head_dim

        # qjl signs: [H, B*S_new, D] → pack (1-bit) → [H, B, S_new, ceil(D/8)]
        qjl_4d = q_dict["qjl"].reshape(H, B, S_new, D)
        self.qjl_packed = _cat_or_init(self.qjl_packed, pack_signs(qjl_4d), dim=2)

        # gamma: [H, B*S_new] → [H, B, S_new]
        gamma_3d = q_dict["gamma"].reshape(H, B, S_new)
        self.gamma = _cat_or_init(self.gamma, gamma_3d, dim=2)

        # norms: [H, B*S_new] → [H, B, S_new]
        norms_3d = norms_in.reshape(H, B, S_new)
        self.norms = _cat_or_init(self.norms, norms_3d, dim=2)

        # mse indices: [H, B*S_new, D] → pack at (key_bits-1) bits
        if q_dict["mse"] is not None:
            mse_bits = self.key_bits - 1
            mse_4d = q_dict["mse"]["idx"].reshape(H, B, S_new, D)
            self.mse_packed = _cat_or_init(
                self.mse_packed, pack_bits(mse_4d, n_bits=mse_bits), dim=2
            )

    def to_quantized_dict(self) -> tuple[dict, torch.Tensor]:
        """Unpack all stored tokens and return (q_dict, norms).

        Returns tensors shaped [H, B*S, D] / [H, B*S] suitable for the
        batched quantizer's dequantize_with_norm().
        """
        H, B, S = self.qjl_packed.shape[:3]
        D = self.head_dim

        qjl = unpack_signs(self.qjl_packed, n_values=D).reshape(H, B * S, D)
        gamma = self.gamma.reshape(H, B * S)
        norms = self.norms.reshape(H, B * S)

        mse_q = None
        if self.mse_packed is not None:
            mse_bits = self.key_bits - 1
            mse_idx = unpack_bits(
                self.mse_packed, n_bits=mse_bits, n_values=D
            ).reshape(H, B * S, D)
            mse_q = {"idx": mse_idx}

        return {"mse": mse_q, "qjl": qjl, "gamma": gamma}, norms

    # ---- lifecycle helpers -----------------------------------------------

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        """Reorder batch dimension for beam search (beam_idx selects B dim)."""
        if self.qjl_packed is not None:
            self.qjl_packed = self.qjl_packed[:, beam_idx, ...]
            self.gamma = self.gamma[:, beam_idx, ...]
            self.norms = self.norms[:, beam_idx, ...]
        if self.mse_packed is not None:
            self.mse_packed = self.mse_packed[:, beam_idx, ...]

    def crop(self, max_length: int) -> None:
        if self.qjl_packed is not None:
            self.qjl_packed = self.qjl_packed[:, :, :max_length, ...]
            self.gamma = self.gamma[:, :, :max_length]
            self.norms = self.norms[:, :, :max_length]
        if self.mse_packed is not None:
            self.mse_packed = self.mse_packed[:, :, :max_length, ...]

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.qjl_packed is not None:
            self.qjl_packed = self.qjl_packed.repeat_interleave(repeats, dim=1)
            self.gamma = self.gamma.repeat_interleave(repeats, dim=1)
            self.norms = self.norms.repeat_interleave(repeats, dim=1)
        if self.mse_packed is not None:
            self.mse_packed = self.mse_packed.repeat_interleave(repeats, dim=1)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.qjl_packed is not None:
            self.qjl_packed = self.qjl_packed[:, indices, ...]
            self.gamma = self.gamma[:, indices, ...]
            self.norms = self.norms[:, indices, ...]
        if self.mse_packed is not None:
            self.mse_packed = self.mse_packed[:, indices, ...]


class _BatchedValueStore:
    """Compressed value storage for all H attention heads in a layer."""

    def __init__(self, value_bits: int, head_dim: int) -> None:
        self.value_bits = value_bits
        self.head_dim = head_dim
        self.idx_packed: torch.Tensor | None = None   # uint8 [H, B, S, packed_D]
        self.norms: torch.Tensor | None = None         # f32   [H, B, S]

    def seq_length(self) -> int:
        return 0 if self.idx_packed is None else self.idx_packed.shape[2]

    def append(
        self,
        q_dict: dict,
        norms_in: torch.Tensor,
        B: int,
        S_new: int,
    ) -> None:
        H = norms_in.shape[0]
        D = self.head_dim

        # idx: [H, B*S_new, D] → pack at value_bits bits
        idx_4d = q_dict["idx"].reshape(H, B, S_new, D)
        self.idx_packed = _cat_or_init(
            self.idx_packed, pack_bits(idx_4d, n_bits=self.value_bits), dim=2
        )

        # norms: [H, B*S_new] → [H, B, S_new]
        norms_3d = norms_in.reshape(H, B, S_new)
        self.norms = _cat_or_init(self.norms, norms_3d, dim=2)

    def to_quantized_dict(self) -> tuple[dict, torch.Tensor]:
        H, B, S = self.idx_packed.shape[:3]
        D = self.head_dim

        idx = unpack_bits(
            self.idx_packed, n_bits=self.value_bits, n_values=D
        ).reshape(H, B * S, D)
        norms = self.norms.reshape(H, B * S)

        return {"idx": idx}, norms

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        if self.idx_packed is not None:
            self.idx_packed = self.idx_packed[:, beam_idx, ...]
            self.norms = self.norms[:, beam_idx, ...]

    def crop(self, max_length: int) -> None:
        if self.idx_packed is not None:
            self.idx_packed = self.idx_packed[:, :, :max_length, ...]
            self.norms = self.norms[:, :, :max_length]

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.idx_packed is not None:
            self.idx_packed = self.idx_packed.repeat_interleave(repeats, dim=1)
            self.norms = self.norms.repeat_interleave(repeats, dim=1)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.idx_packed is not None:
            self.idx_packed = self.idx_packed[:, indices, ...]
            self.norms = self.norms[:, indices, ...]


# ---------------------------------------------------------------------------
# Per-layer cache
# ---------------------------------------------------------------------------


class TurboQuantCacheLayer(CacheLayerMixin):
    """Per-layer KV cache with TurboQuant compression.

    Keys are quantized with TurboQuantProd (unbiased inner product).
    Values are quantized with TurboQuantMSE (MSE-optimal reconstruction).

    All H KV heads are processed in a single batched matmul call per layer,
    and indices are bit-packed to match the theoretical storage of b bits per
    coordinate.

    Args:
        layer_idx: Transformer layer index.
        num_kv_heads: Number of key-value attention heads.
        head_dim: Dimension per attention head.
        key_bits: Bits per coordinate for key quantization.
        value_bits: Bits per coordinate for value quantization.
        seed: Base random seed.
        device: Target device for quantizer matrices.
    """

    is_sliding = False

    def __init__(
        self,
        layer_idx: int,
        num_kv_heads: int,
        head_dim: int,
        key_bits: int = 4,
        value_bits: int = 4,
        seed: int = 42,
        device: torch.device | str | None = None,
        use_fused_kernel: bool = False,
    ):
        super().__init__()
        self._layer_idx = layer_idx
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._key_bits = key_bits
        self._value_bits = value_bits
        self._seed = seed
        self._target_device = device
        self._seq_length = 0
        self._use_fused_kernel = use_fused_kernel

        seed_base = seed + layer_idx * 1000

        # Non-overlapping seed ranges so MSE rotation, QJL projection, and
        # value rotation matrices are generated from independent seeds.
        # Each range accommodates up to 100 heads before collision.
        mse_seeds = [seed_base + h for h in range(num_kv_heads)]
        qjl_seeds = [seed_base + 100 + h for h in range(num_kv_heads)]
        val_seeds = [seed_base + 500 + h for h in range(num_kv_heads)]

        self._key_quantizer = BatchedTurboQuantProdTorch(
            d=head_dim,
            b=key_bits,
            num_heads=num_kv_heads,
            mse_seeds=mse_seeds,
            qjl_seeds=qjl_seeds,
            device=device,
        )
        self._value_quantizer = BatchedTurboQuantMSETorch(
            d=head_dim,
            b=value_bits,
            num_heads=num_kv_heads,
            seeds=val_seeds,
            device=device,
        )

        self._key_store = _BatchedKeyStore(key_bits=key_bits, head_dim=head_dim)
        self._value_store = _BatchedValueStore(value_bits=value_bits, head_dim=head_dim)

        # Incremental dense fp32 cache — avoids O(N²) full-dequant each step.
        # Holds [B, H, S_total, D] in fp32; returned as model_dtype at update() exit.
        self._dense_keys: torch.Tensor | None = None
        self._dense_values: torch.Tensor | None = None

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize new K/V states, return all dequantized K/V for attention.

        Args:
            key_states: [batch, num_kv_heads, seq_new, head_dim]
            value_states: [batch, num_kv_heads, seq_new, head_dim]
            cache_kwargs: Unused (accepted for API compatibility).

        Returns:
            (all_keys, all_values) both [batch, num_kv_heads, total_seq, head_dim].
            Output dtype matches the input dtype for fp16/bf16/fp32. For fp8 inputs
            (itemsize=1), output is promoted to fp16 because vanilla PyTorch attention
            cannot consume fp8 tensors. Internal quantizer math always runs in fp32.

        Speed note: dequantization is O(S_new) per step (only new tokens), not
        O(S_total). Total work over N decode steps is O(N) instead of O(N²).
        The compressed store is still populated for memory_tracker reporting.
        """
        if self.keys is None:
            self.lazy_initialization(key_states)

        model_dtype = key_states.dtype
        B, H, S_new, D = key_states.shape

        # Reshape to [H, B*S_new, D] so all heads process in one batched call.
        k_batched = key_states.permute(1, 0, 2, 3).reshape(H, B * S_new, D)
        v_batched = value_states.permute(1, 0, 2, 3).reshape(H, B * S_new, D)

        # quantize_with_norm casts to fp32 internally; no pre-cast needed.
        q_k, norms_k = self._key_quantizer.quantize_with_norm(k_batched)
        q_v, norms_v = self._value_quantizer.quantize_with_norm(v_batched)

        # Pack into compressed store (for memory_tracker / future fused kernel).
        self._key_store.append(q_k, norms_k, B, S_new)
        self._value_store.append(q_v, norms_v, B, S_new)
        self._seq_length += S_new

        # Fused-kernel fast path: skip dense buffer growth on decode. The custom
        # attention hook reads directly from the compressed stores; what we
        # return here is ignored for decode steps. Prefill (S_new > 1) still
        # needs dense K/V because the fused kernel is decode-only.
        if self._use_fused_kernel and S_new == 1 and self._seq_length > 1:
            empty = key_states.new_empty(0)
            return empty, empty

        # Dequantize only the NEW tokens (O(S_new), not O(S_total)).
        # q_k / q_v already have the right shapes [H, B*S_new, D] — pass directly.
        new_keys_flat = self._key_quantizer.dequantize_with_norm(q_k, norms_k)
        new_vals_flat = self._value_quantizer.dequantize_with_norm(q_v, norms_v)

        # Reshape from [H, B*S_new, D] → [B, H, S_new, D]
        new_keys = new_keys_flat.reshape(H, B, S_new, D).permute(1, 0, 2, 3)
        new_vals = new_vals_flat.reshape(H, B, S_new, D).permute(1, 0, 2, 3)

        # Append new tokens to dense fp32 buffers (one cheap cat per step).
        if self._dense_keys is None:
            self._dense_keys = new_keys
            self._dense_values = new_vals
        else:
            self._dense_keys = torch.cat([self._dense_keys, new_keys], dim=2)
            self._dense_values = torch.cat([self._dense_values, new_vals], dim=2)

        # FP8 guard: float8_e4m3fn / float8_e5m2 have itemsize=1.
        # Vanilla PyTorch attention matmuls cannot consume FP8 tensors, so
        # promote to FP16. Full FP8 support (Transformer Engine) is Phase 11f.
        safe_dtype = model_dtype if model_dtype.itemsize >= 2 else torch.float16
        return self._dense_keys.to(safe_dtype), self._dense_values.to(safe_dtype)

    def _dequantize_all_keys(
        self, batch_size: int, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        """Dequantize all cached keys, return [B, H, S, D]."""
        q_dict, norms = self._key_store.to_quantized_dict()
        # flat: [H, B*S, D]
        flat = self._key_quantizer.dequantize_with_norm(q_dict, norms)
        # Reshape and permute to [B, H, S, D]
        return flat.reshape(num_heads, batch_size, self._seq_length, head_dim).permute(
            1, 0, 2, 3
        )

    def _dequantize_all_values(
        self, batch_size: int, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        """Dequantize all cached values, return [B, H, S, D]."""
        q_dict, norms = self._value_store.to_quantized_dict()
        flat = self._value_quantizer.dequantize_with_norm(q_dict, norms)
        return flat.reshape(num_heads, batch_size, self._seq_length, head_dim).permute(
            1, 0, 2, 3
        )

    def get_seq_length(self) -> int:
        return self._seq_length

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position) -> tuple[int, int]:
        # HF's Cache.get_mask_sizes passes a plain int query_length to the
        # per-layer method, while create_causal_mask may pass a cache_position
        # tensor. Accept both.
        if hasattr(cache_position, "shape"):
            query_length = cache_position.shape[0]
        else:
            query_length = int(cache_position)
        kv_length = query_length + self._seq_length
        return kv_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._key_store.reorder(beam_idx)
        self._value_store.reorder(beam_idx)
        if self._dense_keys is not None:
            self._dense_keys = self._dense_keys[beam_idx]
            self._dense_values = self._dense_values[beam_idx]

    def reset(self) -> None:
        self._seq_length = 0
        self._key_store = _BatchedKeyStore(
            key_bits=self._key_bits, head_dim=self._head_dim
        )
        self._value_store = _BatchedValueStore(
            value_bits=self._value_bits, head_dim=self._head_dim
        )
        self._dense_keys = None
        self._dense_values = None
        if self.keys is not None:
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self._seq_length - abs(max_length)
        if self._seq_length <= max_length:
            return
        self._seq_length = max_length
        self._key_store.crop(max_length)
        self._value_store.crop(max_length)
        if self._dense_keys is not None:
            self._dense_keys = self._dense_keys[:, :, :max_length, :]
            self._dense_values = self._dense_values[:, :, :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._key_store.batch_repeat_interleave(repeats)
        self._value_store.batch_repeat_interleave(repeats)
        if self._dense_keys is not None:
            self._dense_keys = self._dense_keys.repeat_interleave(repeats, dim=0)
            self._dense_values = self._dense_values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._key_store.batch_select_indices(indices)
        self._value_store.batch_select_indices(indices)
        if self._dense_keys is not None:
            self._dense_keys = self._dense_keys[indices]
            self._dense_values = self._dense_values[indices]


# ---------------------------------------------------------------------------
# Full multi-layer cache
# ---------------------------------------------------------------------------


class TurboQuantCache(Cache):
    """HuggingFace-compatible KV cache with TurboQuant compression.

    Drop-in replacement for DynamicCache. Keys are quantized with
    TurboQuantProd (unbiased inner product preservation) and values
    with TurboQuantMSE (MSE-optimal reconstruction).

    Indices are bit-packed to achieve theoretical compression ratios:
    b=4 → ~3.7×, b=3 → ~4.7×, b=2 → ~6.7× vs FP16 DynamicCache.

    Args:
        config: HuggingFace model config (PretrainedConfig).
        key_bits: Bits per coordinate for key quantization (default 4).
        value_bits: Bits per coordinate for value quantization (default 4).
        per_layer_bits: Optional per-layer overrides, e.g.
            {0: {"key_bits": 3, "value_bits": 4}}.
        seed: Base random seed for reproducibility.
        device: Target device for quantizer matrices.

    Example::

        from transformers import AutoModelForCausalLM
        from turboquant.integrations.hf_cache import TurboQuantCache

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
        cache = TurboQuantCache(model.config, key_bits=4, value_bits=4)
        outputs = model.generate(**inputs, past_key_values=cache, use_cache=True)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        key_bits: int = 4,
        value_bits: int = 4,
        per_layer_bits: dict[int, dict[str, int]] | None = None,
        seed: int = 42,
        device: torch.device | str | None = None,
        use_fused_kernel: bool = False,
    ):
        text_config = config.get_text_config(decoder=True)
        num_layers = text_config.num_hidden_layers
        num_kv_heads = getattr(
            text_config, "num_key_value_heads", text_config.num_attention_heads
        )
        head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )

        per_layer_bits = per_layer_bits or {}

        layers = []
        for i in range(num_layers):
            layer_cfg = per_layer_bits.get(i, {})
            kb = layer_cfg.get("key_bits", key_bits)
            vb = layer_cfg.get("value_bits", value_bits)
            layers.append(
                TurboQuantCacheLayer(
                    layer_idx=i,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    key_bits=kb,
                    value_bits=vb,
                    seed=seed,
                    device=device,
                    use_fused_kernel=use_fused_kernel,
                )
            )

        super().__init__(layers=layers)

        self._key_bits = key_bits
        self._value_bits = value_bits
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._use_fused_kernel = use_fused_kernel

    def bind(self):
        """Context manager: route the fused attention hook to this cache."""
        from .attention import bind_cache

        return bind_cache(self)
