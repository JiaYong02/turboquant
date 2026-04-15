"""Fused decompress+attention kernels for TurboQuant KV cache.

Phase 9a: PyTorch reference (decompress_attn_ref).
Phase 9b: Triton fused kernel (decompress_attn).
"""

from .decompress_attn_ref import fused_decompress_attention_ref

__all__ = ["fused_decompress_attention_ref"]

try:
    from .decompress_attn import fused_decompress_attention_triton

    __all__.append("fused_decompress_attention_triton")
except ImportError:  # triton not installed
    pass
