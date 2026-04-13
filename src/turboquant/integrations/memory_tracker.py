"""VRAM tracking utilities for TurboQuant KV cache.

Note: the drop-in HF cache holds BOTH a compressed store (bit-packed) and a
dense fp32 buffer for fast attention.  This module reports only the compressed
store bytes.  Actual peak VRAM during generation includes the dense buffer
(comparable to FP16 baseline).  True memory savings require a fused
decompress-attention kernel (Phase 9 in spec.md) that reads the compressed
store directly without materialising dense tensors.
"""

from __future__ import annotations

import torch


def compressed_bytes(cache: "TurboQuantCache") -> dict[str, int]:  # noqa: F821
    """Total bytes stored in compressed key/value stores across all layers.

    Returns:
        Dict with 'keys', 'values', and 'total' byte counts.
    """
    key_bytes = 0
    value_bytes = 0

    for layer in cache.layers:
        ks = layer._key_store
        if ks.qjl_packed is not None:
            key_bytes += ks.qjl_packed.nelement() * ks.qjl_packed.element_size()
        if ks.gamma is not None:
            key_bytes += ks.gamma.nelement() * ks.gamma.element_size()
        if ks.norms is not None:
            key_bytes += ks.norms.nelement() * ks.norms.element_size()
        if ks.mse_packed is not None:
            key_bytes += ks.mse_packed.nelement() * ks.mse_packed.element_size()

        vs = layer._value_store
        if vs.idx_packed is not None:
            value_bytes += vs.idx_packed.nelement() * vs.idx_packed.element_size()
        if vs.norms is not None:
            value_bytes += vs.norms.nelement() * vs.norms.element_size()

    return {"keys": key_bytes, "values": value_bytes, "total": key_bytes + value_bytes}


def fp16_equivalent_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
) -> int:
    """Bytes that FP16 DynamicCache would use for the same geometry."""
    # K + V, each [batch, heads, seq, dim] in float16 (2 bytes)
    return 2 * batch_size * num_kv_heads * seq_len * head_dim * 2 * num_layers


def report(cache: "TurboQuantCache") -> dict:  # noqa: F821
    """Full memory report with compression ratio.

    Returns:
        Dict with 'compressed_bytes', 'fp16_bytes', 'compression_ratio',
        and 'seq_length'.
    """
    seq_len = cache.get_seq_length()
    if seq_len == 0:
        return {
            "compressed_bytes": 0,
            "fp16_bytes": 0,
            "compression_ratio": 1.0,
            "seq_length": 0,
        }

    comp = compressed_bytes(cache)

    # Infer batch size from first layer's key store (dim 1 of packed tensor).
    batch_size = 1
    first_layer = cache.layers[0]
    if first_layer._key_store.qjl_packed is not None:
        batch_size = first_layer._key_store.qjl_packed.shape[1]

    fp16 = fp16_equivalent_bytes(
        num_layers=cache._num_layers,
        num_kv_heads=cache._num_kv_heads,
        head_dim=cache._head_dim,
        seq_len=seq_len,
        batch_size=batch_size,
    )

    ratio = fp16 / comp["total"] if comp["total"] > 0 else float("inf")

    return {
        "compressed_bytes": comp["total"],
        "compressed_keys_bytes": comp["keys"],
        "compressed_values_bytes": comp["values"],
        "fp16_bytes": fp16,
        "compression_ratio": round(ratio, 2),
        "seq_length": seq_len,
        "batch_size": batch_size,
    }
