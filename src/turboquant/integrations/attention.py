"""Custom HF attention function that calls the TurboQuant fused kernel on decode.

Registered under the name ``"turboquant_fused"`` via :func:`register`. When a
model is loaded with ``config._attn_implementation = "turboquant_fused"`` and an
active :class:`TurboQuantCache` is bound (via ``cache.bind()`` context manager),
decode steps (S_q == 1) route through
:func:`fused_decompress_attention_triton`. Prefill and any step without a bound
cache fall back to stock SDPA so the model stays correct end-to-end.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional

import torch

_ACTIVE_CACHE = threading.local()


def _get_active_cache():
    return getattr(_ACTIVE_CACHE, "cache", None)


@contextmanager
def bind_cache(cache):
    """Context manager: bind a TurboQuantCache so the attention hook can reach it."""
    prev = getattr(_ACTIVE_CACHE, "cache", None)
    _ACTIVE_CACHE.cache = cache
    try:
        yield cache
    finally:
        _ACTIVE_CACHE.cache = prev


def turboquant_fused_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """HF-compatible attention function.

    Signature matches transformers' ``sdpa_attention_forward``. Shapes:
      query: [B, H_q, S_q, D]
      key/value: [B, H_kv, S_kv, D]  (post-cache-update, so S_kv == total seq)
    """
    from .attention import _get_active_cache  # local to avoid stale ref issues
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    cache = _get_active_cache()
    layer_idx = getattr(module, "layer_idx", None)

    if (
        cache is not None
        and layer_idx is not None
        and query.shape[2] == 1
        and hasattr(cache, "layers")
        and layer_idx < len(cache.layers)
        and getattr(cache.layers[layer_idx], "_use_fused_kernel", False)
    ):
        from ..kernels import fused_decompress_attention_triton

        layer = cache.layers[layer_idx]
        if layer._key_store.seq_length() > 0:
            out = fused_decompress_attention_triton(
                query,
                layer._key_store,
                layer._value_store,
                layer._key_quantizer,
                layer._value_quantizer,
                scale=scaling,
            )
            # sdpa_attention_forward returns [B, S_q, H_q, D] (it transposes
            # [B, H_q, S_q, D] → [B, S_q, H_q, D] before returning).
            return out.transpose(1, 2).contiguous(), None

    return sdpa_attention_forward(
        module, query, key, value, attention_mask,
        dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
    )


def register() -> None:
    """Register ``turboquant_fused`` in ``ALL_ATTENTION_FUNCTIONS``."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["turboquant_fused"] = turboquant_fused_attention_forward
